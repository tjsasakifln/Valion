"""
Worker Celery para processamento assíncrono
Executa tarefas computacionalmente intensivas sem travar a interface.
"""

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import logging
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path

# Imports dos módulos core
from src.core.data_loader import DataLoader
from src.core.transformations import VariableTransformer
from src.core.model_builder import ElasticNetModelBuilder
from src.core.nbr14653_validation import NBR14653Validator
from src.core.results_generator import ResultsGenerator
from src.config.settings import Settings

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
settings = Settings()

# Configuração do Celery
app = Celery('valion_worker')

# Configuração do broker (Redis/RabbitMQ)
app.conf.update(
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hora
    task_soft_time_limit=3300,  # 55 minutos
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Configuração de roteamento
app.conf.task_routes = {
    'valion_worker.tasks.process_evaluation': {'queue': 'evaluation'},
    'valion_worker.tasks.validate_data': {'queue': 'validation'},
    'valion_worker.tasks.train_model': {'queue': 'modeling'},
    'valion_worker.tasks.generate_report': {'queue': 'reporting'},
}

# Callbacks para monitoramento
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handler executado antes da tarefa iniciar."""
    logger.info(f"Iniciando tarefa {task_id}: {task.name}")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handler executado após a tarefa terminar."""
    logger.info(f"Tarefa {task_id} finalizada com estado: {state}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handler executado quando a tarefa falha."""
    logger.error(f"Tarefa {task_id} falhou: {exception}")

# Tarefas Celery

@app.task(bind=True, name='valion_worker.tasks.process_evaluation')
def process_evaluation(self, evaluation_id: str, file_path: str, target_column: str = "valor", config: Optional[Dict[str, Any]] = None):
    """
    Processa avaliação imobiliária completa.
    
    Args:
        evaluation_id: ID único da avaliação
        file_path: Caminho para o arquivo de dados
        target_column: Nome da coluna target
        config: Configurações adicionais
        
    Returns:
        Resultado completo da avaliação
    """
    try:
        config = config or {}
        
        # Atualizar progresso
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'phase': 'Iniciando'})
        
        # Fase 1: Carregamento e validação
        self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'phase': 'Carregando dados'})
        
        data_loader = DataLoader(config)
        raw_data = data_loader.load_data(file_path)
        
        self.update_state(state='PROGRESS', meta={'current': 20, 'total': 100, 'phase': 'Validando dados'})
        
        validation_result = data_loader.validate_data(raw_data)
        if not validation_result.is_valid:
            raise ValueError(f"Dados inválidos: {validation_result.errors}")
        
        cleaned_data = data_loader.clean_data(raw_data)
        
        # Fase 2: Transformações
        self.update_state(state='PROGRESS', meta={'current': 40, 'total': 100, 'phase': 'Transformando variáveis'})
        
        transformer = VariableTransformer(config)
        transformation_result = transformer.transform_data(cleaned_data, target_column)
        
        # Fase 3: Modelagem
        self.update_state(state='PROGRESS', meta={'current': 60, 'total': 100, 'phase': 'Treinando modelo'})
        
        model_builder = ElasticNetModelBuilder(config)
        model_result = model_builder.build_model(transformation_result.transformed_data, target_column)
        
        # Fase 4: Validação NBR
        self.update_state(state='PROGRESS', meta={'current': 80, 'total': 100, 'phase': 'Validando NBR 14653'})
        
        validator = NBR14653Validator(config)
        X_train, X_test, y_train, y_test = model_builder.prepare_data(transformation_result.transformed_data)
        nbr_result = validator.validate_model(model_result.model, X_train, X_test, y_train, y_test)
        
        # Fase 5: Geração de relatório
        self.update_state(state='PROGRESS', meta={'current': 90, 'total': 100, 'phase': 'Gerando relatório'})
        
        results_generator = ResultsGenerator(config)
        report = results_generator.generate_full_report(
            validation_result, transformation_result, model_result, nbr_result, cleaned_data
        )
        
        # Salvar modelo e relatório
        model_path = f"models/{evaluation_id}_model.joblib"
        report_path = f"reports/{evaluation_id}_report.json"
        
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        model_builder.save_model(model_result, model_path)
        results_generator.save_report(report, report_path)
        
        self.update_state(state='PROGRESS', meta={'current': 100, 'total': 100, 'phase': 'Concluído'})
        
        return {
            'evaluation_id': evaluation_id,
            'status': 'concluido',
            'report': report.__dict__,
            'model_path': model_path,
            'report_path': report_path,
            'performance': model_result.performance.__dict__,
            'nbr_validation': nbr_result.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na avaliação {evaluation_id}: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@app.task(bind=True, name='valion_worker.tasks.validate_data')
def validate_data(self, file_path: str, config: Optional[Dict[str, Any]] = None):
    """
    Valida dados de entrada.
    
    Args:
        file_path: Caminho para o arquivo
        config: Configurações
        
    Returns:
        Resultado da validação
    """
    try:
        config = config or {}
        
        self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'phase': 'Validando dados'})
        
        data_loader = DataLoader(config)
        raw_data = data_loader.load_data(file_path)
        validation_result = data_loader.validate_data(raw_data)
        
        return {
            'is_valid': validation_result.is_valid,
            'errors': validation_result.errors,
            'warnings': validation_result.warnings,
            'summary': validation_result.summary,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na validação: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@app.task(bind=True, name='valion_worker.tasks.train_model')
def train_model(self, data_dict: Dict[str, Any], target_column: str = "valor", config: Optional[Dict[str, Any]] = None):
    """
    Treina modelo de avaliação.
    
    Args:
        data_dict: Dados em formato dict
        target_column: Nome da coluna target
        config: Configurações
        
    Returns:
        Resultado do treinamento
    """
    try:
        config = config or {}
        
        self.update_state(state='PROGRESS', meta={'current': 25, 'total': 100, 'phase': 'Preparando dados'})
        
        # Converter dict para DataFrame
        df = pd.DataFrame(data_dict)
        
        # Transformar dados
        transformer = VariableTransformer(config)
        transformation_result = transformer.transform_data(df, target_column)
        
        self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'phase': 'Treinando modelo'})
        
        # Treinar modelo
        model_builder = ElasticNetModelBuilder(config)
        model_result = model_builder.build_model(transformation_result.transformed_data, target_column)
        
        self.update_state(state='PROGRESS', meta={'current': 75, 'total': 100, 'phase': 'Validando modelo'})
        
        # Validar NBR
        validator = NBR14653Validator(config)
        X_train, X_test, y_train, y_test = model_builder.prepare_data(transformation_result.transformed_data)
        nbr_result = validator.validate_model(model_result.model, X_train, X_test, y_train, y_test)
        
        return {
            'model_performance': model_result.performance.__dict__,
            'nbr_validation': nbr_result.__dict__,
            'best_params': model_result.best_params,
            'training_summary': model_result.training_summary,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@app.task(bind=True, name='valion_worker.tasks.generate_report')
def generate_report(self, evaluation_data: Dict[str, Any], output_format: str = "json"):
    """
    Gera relatório de avaliação.
    
    Args:
        evaluation_data: Dados da avaliação
        output_format: Formato do relatório (json, excel)
        
    Returns:
        Caminho do relatório gerado
    """
    try:
        self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'phase': 'Gerando relatório'})
        
        results_generator = ResultsGenerator({})
        
        # Reconstruir objetos a partir dos dados
        # Em implementação real, seria necessário deserializar adequadamente
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "json":
            report_path = f"reports/report_{timestamp}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False, default=str)
        
        elif output_format == "excel":
            report_path = f"reports/report_{timestamp}.xlsx"
            # Implementar exportação Excel
            pass
        
        return {
            'report_path': report_path,
            'format': output_format,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na geração do relatório: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@app.task(bind=True, name='valion_worker.tasks.make_prediction')
def make_prediction(self, model_path: str, features: Dict[str, Any]):
    """
    Faz predição usando modelo treinado.
    
    Args:
        model_path: Caminho para o modelo salvo
        features: Features para predição
        
    Returns:
        Resultado da predição
    """
    try:
        self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'phase': 'Fazendo predição'})
        
        # Carregar modelo
        model_builder = ElasticNetModelBuilder({})
        model_result = model_builder.load_model(model_path)
        
        # Converter features para DataFrame
        features_df = pd.DataFrame([features])
        
        # Fazer predição
        prediction = model_builder.predict(features_df)
        
        return {
            'prediction': float(prediction[0]),
            'features': features,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

# Tarefas de monitoramento

@app.task(name='valion_worker.tasks.health_check')
def health_check():
    """Verifica saúde do worker."""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'worker_id': os.getenv('HOSTNAME', 'unknown')
    }

@app.task(name='valion_worker.tasks.cleanup_old_files')
def cleanup_old_files(days_old: int = 7):
    """
    Limpa arquivos antigos.
    
    Args:
        days_old: Idade em dias para considerar arquivo antigo
        
    Returns:
        Resultado da limpeza
    """
    try:
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        cleaned_files = []
        
        # Limpar modelos antigos
        models_dir = Path("models")
        if models_dir.exists():
            for file_path in models_dir.glob("*.joblib"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_files.append(str(file_path))
        
        # Limpar relatórios antigos
        reports_dir = Path("reports")
        if reports_dir.exists():
            for file_path in reports_dir.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_files.append(str(file_path))
        
        return {
            'cleaned_files': cleaned_files,
            'count': len(cleaned_files),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na limpeza: {str(e)}")
        raise

if __name__ == '__main__':
    app.start()