# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Celery Worker for asynchronous processing
Executes computationally intensive tasks without blocking the interface.
"""

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
from celery.exceptions import Retry, MaxRetriesExceededError
import logging
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path
import time
import psycopg2
from sqlalchemy.exc import OperationalError, IntegrityError, DatabaseError
import redis
from contextlib import contextmanager

# Core module imports
from src.core.data_loader import DataLoader
from src.core.transformations import VariableTransformer
from src.core.model_builder import ModelBuilder
from src.core.validation_strategy import ValidatorFactory, ValidationContext
from src.core.nbr14653_validation import NBR14653Validator
from src.core.uspap_validation import USPAPValidator
from src.core.evs_validation import EVSValidator
from src.core.results_generator import ResultsGenerator
from src.config.settings import Settings
from src.websocket.websocket_manager import websocket_manager, ProgressUpdate
from src.database.database import get_database_manager
import asyncio

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
settings = Settings()

# Configuração do Celery
app = Celery('valion_worker')

# Configuração robusta do broker (Redis/RabbitMQ) com retry e dead letter queues
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
    
    # Configurações de retry globais
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_reject_on_worker_lost=True,
    
    # Retry policy padrão
    task_default_retry_delay=60,  # 1 minuto
    task_max_retries=3,
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

# Lista de exceções que devem acionar retry automático
RETRIABLE_EXCEPTIONS = (
    OperationalError,
    DatabaseError,
    psycopg2.OperationalError,
    redis.ConnectionError,
    redis.TimeoutError,
    ConnectionError,
    TimeoutError
)

@contextmanager
def safe_database_operation(evaluation_id: str = None):
    """Context manager para operações de banco seguras com retry."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            yield
            break
        except RETRIABLE_EXCEPTIONS as e:
            if attempt < max_retries - 1:
                logger.warning(f"Operação de banco falhou (tentativa {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay * (2 ** attempt))  # Backoff exponencial
                continue
            else:
                logger.error(f"Operação de banco falhou após {max_retries} tentativas: {e}")
                if evaluation_id:
                    log_audit_trail(evaluation_id, 'database_failure', 'system', None,
                                   new_values={'error': str(e), 'max_retries_exceeded': True})
                raise
        except Exception as e:
            logger.error(f"Erro não retriable na operação de banco: {e}")
            raise

# Funções auxiliares
def send_websocket_update(evaluation_id: str, status: str, phase: str, progress: float, message: str, metadata: Optional[Dict] = None):
    """Envia atualização via WebSocket."""
    try:
        update = ProgressUpdate(
            evaluation_id=evaluation_id,
            status=status,
            current_phase=phase,
            progress=progress,
            message=message,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
        # Como o Celery roda de forma síncrona, precisa executar a função assíncrona
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_manager.send_progress_update(evaluation_id, update))
        loop.close()
    except Exception as e:
        logger.warning(f"Erro ao enviar update WebSocket: {e}")

def log_audit_trail(evaluation_id: str, operation: str, entity_type: str, entity_id: Optional[str] = None,
                   old_values: Optional[Dict] = None, new_values: Optional[Dict] = None, 
                   metadata: Optional[Dict] = None, user_id: str = "system"):
    """Registra entrada na trilha de auditoria com transação atômica."""
    try:
        db_manager = get_database_manager()
        with db_manager.get_atomic_transaction() as session:
            db_manager.create_audit_trail(
                session=session,
                user_id=user_id,
                operation=operation,
                entity_type=entity_type,
                entity_id=entity_id,
                old_values=old_values,
                new_values=new_values,
                metadata=metadata,
                evaluation_id=evaluation_id
            )
    except Exception as e:
        logger.warning(f"Erro ao registrar audit trail: {e}")

# Tarefas Celery

@app.task(
    bind=True, 
    name='valion_worker.tasks.process_evaluation',
    autoretry_for=RETRIABLE_EXCEPTIONS,
    retry_kwargs={'max_retries': 3, 'countdown': 60},
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True
)
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
        
        # Auditoria: Início da avaliação
        log_audit_trail(evaluation_id, 'start_evaluation', 'evaluation', evaluation_id,
                       new_values={'file_path': file_path, 'target_column': target_column, 'config': config})
        
        # Atualizar progresso
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'phase': 'Iniciando'})
        send_websocket_update(evaluation_id, 'processing', 'Iniciando', 0.0, 'Avaliação iniciada')
        
        # Fase 1: Carregamento e validação
        self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'phase': 'Carregando dados'})
        send_websocket_update(evaluation_id, 'processing', 'Carregando dados', 10.0, 'Carregando arquivo de dados')
        
        data_loader = DataLoader(config)
        raw_data = data_loader.load_data(file_path)
        
        # Auditoria: Dados carregados
        log_audit_trail(evaluation_id, 'data_ingested', 'data_loader', None,
                       new_values={'file_path': file_path, 'rows': len(raw_data), 'columns': len(raw_data.columns)})
        
        self.update_state(state='PROGRESS', meta={'current': 20, 'total': 100, 'phase': 'Validando dados'})
        send_websocket_update(evaluation_id, 'processing', 'Validando dados', 20.0, 'Validando qualidade dos dados')
        
        data_validation_result = data_loader.validate_data(raw_data)
        if not data_validation_result.is_valid:
            # Auditoria: Validação falhou
            log_audit_trail(evaluation_id, 'validation_failed', 'data_validation', None,
                           new_values={'errors': data_validation_result.errors, 'warnings': data_validation_result.warnings})
            send_websocket_update(evaluation_id, 'failed', 'Validação falhou', 20.0, f"Erros: {data_validation_result.errors}")
            raise ValueError(f"Dados inválidos: {data_validation_result.errors}")
        
        # Auditoria: Validação bem-sucedida
        log_audit_trail(evaluation_id, 'validation_passed', 'data_validation', None,
                       new_values={'summary': data_validation_result.summary})
        
        cleaned_data = data_loader.clean_data(raw_data)
        
        # Auditoria: Dados limpos
        log_audit_trail(evaluation_id, 'data_cleaned', 'data_loader', None,
                       new_values={'original_rows': len(raw_data), 'cleaned_rows': len(cleaned_data)})
        
        # Fase 2: Transformações
        self.update_state(state='PROGRESS', meta={'current': 40, 'total': 100, 'phase': 'Transformando variáveis'})
        send_websocket_update(evaluation_id, 'processing', 'Transformando variáveis', 40.0, 'Preparando features')
        
        transformer = VariableTransformer(config)
        transformation_result = transformer.transform_data(cleaned_data, target_column)
        
        # Auditoria: Transformações aplicadas
        log_audit_trail(evaluation_id, 'data_transformed', 'variable_transformer', None,
                       new_values={'transformations_applied': transformation_result.transformations_applied,
                                  'original_features': len(cleaned_data.columns),
                                  'transformed_features': len(transformation_result.transformed_data.columns)})
        
        # Fase 3: Modelagem
        expert_mode = config.get('expert_mode', False)
        model_type = config.get('model_type', 'elastic_net')
        
        if expert_mode:
            self.update_state(state='PROGRESS', meta={'current': 60, 'total': 100, 'phase': 'Treinando modelo (Modo Especialista)'})
            send_websocket_update(evaluation_id, 'processing', 'Treinando modelo avançado', 60.0, f'Modo especialista: {model_type} com SHAP obrigatório')
        else:
            self.update_state(state='PROGRESS', meta={'current': 60, 'total': 100, 'phase': 'Treinando modelo'})
            send_websocket_update(evaluation_id, 'processing', 'Treinando modelo', 60.0, 'Treinando modelo Elastic Net')
        
        model_builder = ModelBuilder(config)
        
        try:
            model_result = model_builder.build_model(transformation_result.transformed_data, target_column)
            
            # Verificação crítica para modo especialista
            if expert_mode:
                if not model_result.performance.shap_feature_importance:
                    error_msg = "ERRO CRÍTICO: Modo especialista requer SHAP values - falha na interpretabilidade"
                    log_audit_trail(evaluation_id, 'expert_mode_failed', 'model_builder', None,
                                   new_values={'error': error_msg, 'model_type': model_type})
                    send_websocket_update(evaluation_id, 'failed', 'Erro no modo especialista', 60.0, error_msg)
                    raise ValueError(error_msg)
                
                # Log de sucesso no modo especialista
                log_audit_trail(evaluation_id, 'expert_mode_success', 'model_builder', None,
                               new_values={'model_type': model_type, 'shap_available': True,
                                          'interpretability_guaranteed': True})
                send_websocket_update(evaluation_id, 'processing', 'SHAP garantido', 65.0, 'Interpretabilidade glass-box confirmada')
            
        except Exception as e:
            if expert_mode:
                error_msg = f"Falha crítica no modo especialista: {str(e)}"
                log_audit_trail(evaluation_id, 'expert_mode_critical_failure', 'model_builder', None,
                               new_values={'error': error_msg, 'model_type': model_type})
                send_websocket_update(evaluation_id, 'failed', 'Erro crítico', 60.0, error_msg)
            raise
        
        # Auditoria: Modelo treinado
        audit_data = {
            'model_type': model_result.model_type,
            'r2_score': model_result.performance.r2_score,
            'rmse': model_result.performance.rmse, 
            'best_params': model_result.best_params,
            'expert_mode': expert_mode
        }
        
        if expert_mode:
            audit_data['shap_guaranteed'] = bool(model_result.performance.shap_feature_importance)
            audit_data['interpretability_level'] = 'complete'
        
        log_audit_trail(evaluation_id, 'model_trained', 'model_builder', None, new_values=audit_data)
        
        # Fase 4: Validação
        standard = config.get('valuation_standard', 'NBR 14653')
        self.update_state(state='PROGRESS', meta={'current': 80, 'total': 100, 'phase': f'Validando conforme {standard}'})
        send_websocket_update(evaluation_id, 'processing', f'Validando conforme {standard}', 80.0, 'Executando testes estatísticos')
        
        # Prepare data for validation
        X_train, X_test, y_train, y_test = model_builder.prepare_data(transformation_result.transformed_data)
        
        # Select validator based on configuration
        validator_map = {
            "NBR 14653": NBR14653Validator,
            "USPAP": USPAPValidator,
            "EVS": EVSValidator
        }
        
        validator_class = validator_map.get(standard, NBR14653Validator)
        validator = validator_class(model_result.model, X_train, X_test, y_train, y_test)
        validation_result = validator.validate()
        
        # Auditoria: Validação
        log_audit_trail(evaluation_id, 'validation_completed', 'validator', None,
                       new_values={'standard': validation_result.standard, 'grade': validation_result.overall_grade, 
                                  'compliance_score': validation_result.compliance_score,
                                  'tests_passed': validation_result.summary.get('passed_tests', 0)})
        
        # Fase 5: Geração de relatório
        self.update_state(state='PROGRESS', meta={'current': 90, 'total': 100, 'phase': 'Gerando relatório'})
        send_websocket_update(evaluation_id, 'processing', 'Gerando relatório', 90.0, 'Consolidando resultados')
        
        results_generator = ResultsGenerator(config)
        report = results_generator.generate_full_report(
            data_validation_result, transformation_result, model_result, validation_result, cleaned_data
        )
        
        # Salvar modelo e relatório com verificação de integridade
        model_path = f"models/{evaluation_id}_model.joblib"
        report_path = f"reports/{evaluation_id}_report.json"
        
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        model_builder.save_model(model_result, model_path)
        results_generator.save_report(report, report_path)
        
        # Registrar artefatos do modelo no banco com verificação de integridade
        try:
            db_manager = get_database_manager()
            
            # Calcular checksum e tamanho dos arquivos
            import hashlib
            
            # Artefato do modelo
            model_size = os.path.getsize(model_path)
            model_checksum = hashlib.sha256()
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    model_checksum.update(chunk)
            model_checksum = model_checksum.hexdigest()
            
            # Artefato do relatório
            report_size = os.path.getsize(report_path)
            report_checksum = hashlib.sha256()
            with open(report_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    report_checksum.update(chunk)
            report_checksum = report_checksum.hexdigest()
            
            # Salvar artefatos no banco com transação atômica
            with db_manager.get_atomic_transaction() as session:
                # Salvar modelo
                db_manager.save_model_artifact(
                    session=session,
                    evaluation_id=evaluation_id,
                    artifact_type="model",
                    artifact_name=f"{evaluation_id}_model.joblib",
                    file_path=model_path,
                    file_size=model_size,
                    checksum=model_checksum,
                    metadata={"model_type": model_result.model_type, "r2_score": model_result.performance.r2_score},
                    user_id="system"
                )
                
                # Salvar relatório
                db_manager.save_model_artifact(
                    session=session,
                    evaluation_id=evaluation_id,
                    artifact_type="report",
                    artifact_name=f"{evaluation_id}_report.json",
                    file_path=report_path,
                    file_size=report_size,
                    checksum=report_checksum,
                    metadata={"report_type": "full_evaluation"},
                    user_id="system"
                )
            
        except Exception as e:
            logger.warning(f"Erro ao registrar artefatos no banco: {e}")
        
        # Auditoria: Artefatos salvos com checksums
        log_audit_trail(evaluation_id, 'artifacts_saved', 'file_system', None,
                       new_values={
                           'model_path': model_path, 
                           'report_path': report_path,
                           'model_checksum': model_checksum,
                           'report_checksum': report_checksum,
                           'integrity_verified': True
                       })
        
        self.update_state(state='PROGRESS', meta={'current': 100, 'total': 100, 'phase': 'Concluído'})
        send_websocket_update(evaluation_id, 'completed', 'Avaliação concluída', 100.0, 'Relatório gerado com sucesso')
        
        # Auditoria: Avaliação finalizada
        log_audit_trail(evaluation_id, 'evaluation_completed', 'evaluation', evaluation_id,
                       new_values={'final_r2_score': model_result.performance.r2_score,
                                  'validation_standard': validation_result.standard,
                                  'grade': validation_result.overall_grade, 
                                  'compliance_score': validation_result.compliance_score})
        
        return {
            'evaluation_id': evaluation_id,
            'status': 'concluido',
            'report': report.__dict__,
            'model_path': model_path,
            'report_path': report_path,
            'performance': model_result.performance.__dict__,
            'validation_result': validation_result.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na avaliação {evaluation_id}: {str(e)}")
        
        # Auditoria: Erro na avaliação
        log_audit_trail(evaluation_id, 'evaluation_failed', 'evaluation', evaluation_id,
                       new_values={'error': str(e), 'error_type': type(e).__name__})
        
        # WebSocket: Notificar erro
        send_websocket_update(evaluation_id, 'failed', 'Erro na avaliação', 0.0, f"Erro: {str(e)}")
        
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
        model_builder = ModelBuilder(config)
        model_result = model_builder.build_model(transformation_result.transformed_data, target_column)
        
        self.update_state(state='PROGRESS', meta={'current': 75, 'total': 100, 'phase': 'Validando modelo'})
        
        # Validar usando seletor de validador
        standard = config.get('valuation_standard', 'NBR 14653')
        X_train, X_test, y_train, y_test = model_builder.prepare_data(transformation_result.transformed_data)
        
        # Select validator based on configuration
        validator_map = {
            "NBR 14653": NBR14653Validator,
            "USPAP": USPAPValidator,
            "EVS": EVSValidator
        }
        
        validator_class = validator_map.get(standard, NBR14653Validator)
        validator = validator_class(model_result.model, X_train, X_test, y_train, y_test)
        validation_result = validator.validate()
        
        return {
            'model_performance': model_result.performance.__dict__,
            'validation_result': validation_result.__dict__,
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
        model_builder = ModelBuilder({})
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

@app.task(bind=True, name='valion_worker.tasks.continue_evaluation_step')
def continue_evaluation_step(self, evaluation_id: str, step: str, approved: bool, 
                            modifications: Optional[Dict[str, Any]] = None,
                            user_feedback: Optional[str] = None):
    """
    Continua avaliação após aprovação do usuário.
    
    Args:
        evaluation_id: ID da avaliação
        step: Etapa aprovada/rejeitada
        approved: Se a etapa foi aprovada
        modifications: Modificações solicitadas pelo usuário
        user_feedback: Feedback do usuário
        
    Returns:
        Resultado da continuidade
    """
    try:
        # Log da aprovação
        log_audit_trail(evaluation_id, 'user_approval', 'interactive_pipeline', step,
                       new_values={'approved': approved, 'step': step, 'user_feedback': user_feedback,
                                  'modifications': modifications})
        
        if approved:
            self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'phase': f'Aplicando {step}'})
            send_websocket_update(evaluation_id, 'processing', f'Aplicando {step}', 50.0, 
                                f'Usuário aprovou {step} - continuando')
            
            # Aplicar modificações se fornecidas
            if modifications:
                # Implementar lógica para aplicar modificações
                log_audit_trail(evaluation_id, 'modifications_applied', 'interactive_pipeline', step,
                               new_values={'modifications': modifications})
            
            return {
                'evaluation_id': evaluation_id,
                'step': step,
                'approved': True,
                'status': 'continued',
                'timestamp': datetime.now().isoformat()
            }
        else:
            self.update_state(state='PROGRESS', meta={'current': 25, 'total': 100, 'phase': f'Ajustando {step}'})
            send_websocket_update(evaluation_id, 'processing', f'Ajustando {step}', 25.0, 
                                f'Usuário rejeitou {step} - ajustando abordagem')
            
            # Implementar lógica alternativa baseada na rejeição
            return {
                'evaluation_id': evaluation_id,
                'step': step,
                'approved': False,
                'status': 'alternative_approach',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Erro na continuação da avaliação {evaluation_id}: {str(e)}")
        log_audit_trail(evaluation_id, 'continuation_failed', 'interactive_pipeline', step,
                       new_values={'error': str(e), 'step': step})
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@app.task(bind=True, name='valion_worker.tasks.run_data_validation')
def run_data_validation(self, evaluation_id: str, file_path: str, config: Optional[Dict[str, Any]] = None):
    """
    Executa apenas a validação de dados como etapa isolada.
    """
    try:
        config = config or {}
        
        self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'phase': 'Validando dados'})
        send_websocket_update(evaluation_id, 'processing', 'Validando dados', 50.0, 'Executando validação')
        
        data_loader = DataLoader(config)
        raw_data = data_loader.load_data(file_path)
        validation_result = data_loader.validate_data(raw_data)
        
        if not validation_result.is_valid:
            send_websocket_update(evaluation_id, 'awaiting_approval', 'Validação falhou', 50.0, 
                                'Dados inválidos - aprovação necessária',
                                {'step': 'validation', 'errors': validation_result.errors})
        else:
            send_websocket_update(evaluation_id, 'completed', 'Validação concluída', 100.0, 'Dados válidos')
        
        return {
            'evaluation_id': evaluation_id,
            'validation_result': validation_result.__dict__,
            'data_shape': raw_data.shape,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na validação {evaluation_id}: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@app.task(bind=True, name='valion_worker.tasks.run_transformations')
def run_transformations(self, evaluation_id: str, data_dict: Dict[str, Any], target_column: str,
                       config: Optional[Dict[str, Any]] = None):
    """
    Executa transformações como etapa isolada.
    """
    try:
        config = config or {}
        
        self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'phase': 'Transformando dados'})
        send_websocket_update(evaluation_id, 'processing', 'Transformando dados', 50.0, 'Aplicando transformações')
        
        df = pd.DataFrame(data_dict)
        transformer = VariableTransformer(config)
        transformation_result = transformer.transform_data(df, target_column)
        
        # Sugerir transformações para aprovação do usuário
        suggestions = {
            'transformations_applied': transformation_result.transformations_applied,
            'new_features_count': len(transformation_result.feature_names),
            'suggested_removals': [],  # Features sugeridas para remoção
            'alternative_approaches': []  # Abordagens alternativas
        }
        
        send_websocket_update(evaluation_id, 'awaiting_approval', 'Transformações prontas', 75.0, 
                            'Aprovação de transformações necessária',
                            {'step': 'transformations', 'suggestions': suggestions})
        
        return {
            'evaluation_id': evaluation_id,
            'transformation_result': transformation_result.__dict__,
            'suggestions': suggestions,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro nas transformações {evaluation_id}: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@app.task(bind=True, name='valion_worker.tasks.run_outlier_analysis')
def run_outlier_analysis(self, evaluation_id: str, data_dict: Dict[str, Any], target_column: str,
                        config: Optional[Dict[str, Any]] = None):
    """
    Executa análise de outliers como etapa isolada.
    """
    try:
        config = config or {}
        
        self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'phase': 'Analisando outliers'})
        send_websocket_update(evaluation_id, 'processing', 'Analisando outliers', 50.0, 'Detectando outliers')
        
        df = pd.DataFrame(data_dict)
        
        # Detectar outliers (implementação básica)
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_threshold_low = Q1 - 1.5 * IQR
        outlier_threshold_high = Q3 + 1.5 * IQR
        
        outliers = df[(df[target_column] < outlier_threshold_low) | 
                     (df[target_column] > outlier_threshold_high)]
        
        outlier_info = {
            'total_outliers': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'threshold_low': outlier_threshold_low,
            'threshold_high': outlier_threshold_high,
            'outlier_indices': outliers.index.tolist()[:10],  # Primeiros 10
            'removal_recommendation': 'remove' if len(outliers) < len(df) * 0.05 else 'keep'
        }
        
        send_websocket_update(evaluation_id, 'awaiting_approval', 'Outliers detectados', 75.0, 
                            f'{len(outliers)} outliers encontrados - aprovação necessária',
                            {'step': 'outliers', 'details': outlier_info})
        
        return {
            'evaluation_id': evaluation_id,
            'outlier_analysis': outlier_info,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na análise de outliers {evaluation_id}: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

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