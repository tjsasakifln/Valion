# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Sistema de Métricas com Prometheus
Coleta e expõe métricas de negócio e performance para monitoramento.
"""

import time
from typing import Dict, Any, Optional, List
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from prometheus_client.core import CollectorRegistry
from functools import wraps
import logging
from contextlib import contextmanager
import threading
from datetime import datetime, timedelta
import json


# Instância singleton
_metrics_collector = None
_lock = threading.Lock()


class MetricsCollector:
    """Coletor centralizado de métricas para Valion."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.logger = logging.getLogger(__name__)
        
        # Métricas de negócio
        self.evaluations_total = Counter(
            'valion_evaluations_total',
            'Total de avaliações processadas',
            ['status', 'valuation_standard', 'model_type'],
            registry=self.registry
        )
        
        self.evaluation_duration = Histogram(
            'valion_evaluation_duration_seconds',
            'Duração das avaliações em segundos',
            ['valuation_standard', 'model_type'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
            registry=self.registry
        )
        
        self.model_performance = Gauge(
            'valion_model_r2_score',
            'R² score dos modelos treinados',
            ['model_type', 'valuation_standard'],
            registry=self.registry
        )
        
        self.model_training_duration = Histogram(
            'valion_model_training_duration_seconds',
            'Tempo de treinamento dos modelos',
            ['model_type'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'valion_cache_operations_total',
            'Operações de cache',
            ['cache_type', 'operation', 'status'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'valion_cache_hit_ratio',
            'Taxa de acerto do cache',
            ['cache_type'],
            registry=self.registry
        )
        
        # Métricas de qualidade de dados
        self.data_quality_score = Gauge(
            'valion_data_quality_score',
            'Score de qualidade dos dados',
            ['evaluation_id'],
            registry=self.registry
        )
        
        self.data_validation_errors = Counter(
            'valion_data_validation_errors_total',
            'Erros de validação de dados',
            ['error_type', 'valuation_standard'],
            registry=self.registry
        )
        
        # Métricas de API
        self.api_requests_total = Counter(
            'valion_api_requests_total',
            'Total de requisições da API',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'valion_api_request_duration_seconds',
            'Duração das requisições da API',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.active_websocket_connections = Gauge(
            'valion_active_websocket_connections',
            'Conexões WebSocket ativas',
            registry=self.registry
        )
        
        # Métricas de sistema
        self.celery_tasks_total = Counter(
            'valion_celery_tasks_total',
            'Total de tarefas Celery',
            ['task_name', 'status'],
            registry=self.registry
        )
        
        self.celery_task_duration = Histogram(
            'valion_celery_task_duration_seconds',
            'Duração das tarefas Celery',
            ['task_name'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
            registry=self.registry
        )
        
        self.database_operations = Counter(
            'valion_database_operations_total',
            'Operações de banco de dados',
            ['operation', 'table', 'status'],
            registry=self.registry
        )
        
        self.database_query_duration = Histogram(
            'valion_database_query_duration_seconds',
            'Duração das consultas ao banco',
            ['operation', 'table'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Métricas de geolocalização
        self.geocoding_requests = Counter(
            'valion_geocoding_requests_total',
            'Requisições de geocodificação',
            ['status', 'provider'],
            registry=self.registry
        )
        
        self.geocoding_duration = Histogram(
            'valion_geocoding_duration_seconds',
            'Duração da geocodificação',
            ['provider'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Métricas de arquivos
        self.file_uploads_total = Counter(
            'valion_file_uploads_total',
            'Total de uploads de arquivos',
            ['file_type', 'status'],
            registry=self.registry
        )
        
        self.file_processing_duration = Histogram(
            'valion_file_processing_duration_seconds',
            'Duração do processamento de arquivos',
            ['file_type'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # Informações do sistema
        self.system_info = Info(
            'valion_system_info',
            'Informações do sistema',
            registry=self.registry
        )
        
        # Estatísticas em tempo real
        self.current_active_evaluations = Gauge(
            'valion_current_active_evaluations',
            'Avaliações ativas no momento',
            registry=self.registry
        )
        
        self.average_model_accuracy = Gauge(
            'valion_average_model_accuracy',
            'Acurácia média dos modelos nas últimas 24h',
            registry=self.registry
        )
        
        # Histórico de performance
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {
            'evaluations': [],
            'models': [],
            'cache': []
        }
        
        self.logger.info("Sistema de métricas inicializado")
    
    def record_evaluation(self, evaluation_id: str, status: str, 
                         valuation_standard: str, model_type: str,
                         duration: float, r2_score: Optional[float] = None,
                         data_quality_score: Optional[float] = None):
        """
        Registra métricas de uma avaliação.
        
        Args:
            evaluation_id: ID da avaliação
            status: Status da avaliação (success, error, timeout)
            valuation_standard: Norma de avaliação utilizada
            model_type: Tipo do modelo
            duration: Duração em segundos
            r2_score: Score R² do modelo
            data_quality_score: Score de qualidade dos dados
        """
        # Contadores
        self.evaluations_total.labels(
            status=status,
            valuation_standard=valuation_standard,
            model_type=model_type
        ).inc()
        
        # Histograma de duração
        self.evaluation_duration.labels(
            valuation_standard=valuation_standard,
            model_type=model_type
        ).observe(duration)
        
        # Performance do modelo
        if r2_score is not None:
            self.model_performance.labels(
                model_type=model_type,
                valuation_standard=valuation_standard
            ).set(r2_score)
        
        # Qualidade dos dados
        if data_quality_score is not None:
            self.data_quality_score.labels(
                evaluation_id=evaluation_id
            ).set(data_quality_score)
        
        # Histórico
        self.performance_history['evaluations'].append({
            'timestamp': datetime.now(),
            'evaluation_id': evaluation_id,
            'status': status,
            'valuation_standard': valuation_standard,
            'model_type': model_type,
            'duration': duration,
            'r2_score': r2_score,
            'data_quality_score': data_quality_score
        })
        
        # Limpar histórico antigo (manter apenas 24h)
        self._cleanup_history('evaluations')
        
        self.logger.info(f"Métricas registradas para avaliação {evaluation_id}")
    
    def record_model_training(self, model_type: str, duration: float,
                             performance_metrics: Dict[str, float]):
        """
        Registra métricas de treinamento de modelo.
        
        Args:
            model_type: Tipo do modelo
            duration: Duração do treinamento
            performance_metrics: Métricas de performance
        """
        self.model_training_duration.labels(
            model_type=model_type
        ).observe(duration)
        
        # Histórico
        self.performance_history['models'].append({
            'timestamp': datetime.now(),
            'model_type': model_type,
            'duration': duration,
            'performance_metrics': performance_metrics
        })
        
        self._cleanup_history('models')
        
        self.logger.debug(f"Métricas de treinamento registradas para {model_type}")
    
    def record_cache_operation(self, cache_type: str, operation: str,
                              status: str, hit_ratio: Optional[float] = None):
        """
        Registra operações de cache.
        
        Args:
            cache_type: Tipo do cache (geospatial, model, etc.)
            operation: Operação (get, set, delete)
            status: Status (hit, miss, error)
            hit_ratio: Taxa de acerto atual
        """
        self.cache_operations.labels(
            cache_type=cache_type,
            operation=operation,
            status=status
        ).inc()
        
        if hit_ratio is not None:
            self.cache_hit_ratio.labels(
                cache_type=cache_type
            ).set(hit_ratio)
        
        self.logger.debug(f"Cache operation: {cache_type}.{operation} = {status}")
    
    def record_api_request(self, method: str, endpoint: str, 
                          status_code: int, duration: float):
        """
        Registra requisições da API.
        
        Args:
            method: Método HTTP
            endpoint: Endpoint da API
            status_code: Código de status
            duration: Duração da requisição
        """
        self.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_celery_task(self, task_name: str, status: str, duration: float):
        """
        Registra tarefas Celery.
        
        Args:
            task_name: Nome da tarefa
            status: Status (success, failure, retry)
            duration: Duração da tarefa
        """
        self.celery_tasks_total.labels(
            task_name=task_name,
            status=status
        ).inc()
        
        self.celery_task_duration.labels(
            task_name=task_name
        ).observe(duration)
    
    def record_database_operation(self, operation: str, table: str,
                                 status: str, duration: float):
        """
        Registra operações de banco de dados.
        
        Args:
            operation: Tipo de operação (select, insert, update, delete)
            table: Nome da tabela
            status: Status (success, error)
            duration: Duração da operação
        """
        self.database_operations.labels(
            operation=operation,
            table=table,
            status=status
        ).inc()
        
        self.database_query_duration.labels(
            operation=operation,
            table=table
        ).observe(duration)
    
    def record_geocoding_request(self, provider: str, status: str, duration: float):
        """
        Registra requisições de geocodificação.
        
        Args:
            provider: Provedor de geocodificação
            status: Status (success, failure)
            duration: Duração da requisição
        """
        self.geocoding_requests.labels(
            status=status,
            provider=provider
        ).inc()
        
        self.geocoding_duration.labels(
            provider=provider
        ).observe(duration)
    
    def record_file_operation(self, file_type: str, operation: str,
                             status: str, duration: float):
        """
        Registra operações de arquivo.
        
        Args:
            file_type: Tipo do arquivo (csv, xlsx, etc.)
            operation: Operação (upload, processing)
            status: Status (success, error)
            duration: Duração da operação
        """
        if operation == 'upload':
            self.file_uploads_total.labels(
                file_type=file_type,
                status=status
            ).inc()
        else:
            self.file_processing_duration.labels(
                file_type=file_type
            ).observe(duration)
    
    def record_data_validation_error(self, error_type: str, valuation_standard: str):
        """
        Registra erros de validação de dados.
        
        Args:
            error_type: Tipo do erro
            valuation_standard: Norma de avaliação
        """
        self.data_validation_errors.labels(
            error_type=error_type,
            valuation_standard=valuation_standard
        ).inc()
    
    def set_active_evaluations(self, count: int):
        """Define número de avaliações ativas."""
        self.current_active_evaluations.set(count)
    
    def set_websocket_connections(self, count: int):
        """Define número de conexões WebSocket ativas."""
        self.active_websocket_connections.set(count)
    
    def update_system_info(self, info: Dict[str, str]):
        """Atualiza informações do sistema."""
        self.system_info.info(info)
    
    def calculate_average_accuracy(self) -> float:
        """Calcula acurácia média dos modelos nas últimas 24h."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        recent_evaluations = [
            eval_data for eval_data in self.performance_history['evaluations']
            if eval_data['timestamp'] > cutoff_time and eval_data['r2_score'] is not None
        ]
        
        if not recent_evaluations:
            return 0.0
        
        avg_accuracy = sum(e['r2_score'] for e in recent_evaluations) / len(recent_evaluations)
        self.average_model_accuracy.set(avg_accuracy)
        
        return avg_accuracy
    
    def _cleanup_history(self, history_type: str):
        """Remove entradas antigas do histórico."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.performance_history[history_type] = [
            item for item in self.performance_history[history_type]
            if item['timestamp'] > cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtém resumo de performance."""
        return {
            'evaluations_24h': len(self.performance_history['evaluations']),
            'models_trained_24h': len(self.performance_history['models']),
            'average_accuracy': self.calculate_average_accuracy(),
            'active_evaluations': self.current_active_evaluations._value._value,
            'websocket_connections': self.active_websocket_connections._value._value
        }
    
    def start_metrics_server(self, port: int = 8000):
        """Inicia servidor de métricas Prometheus."""
        try:
            start_http_server(port, registry=self.registry)
            self.logger.info(f"Servidor de métricas iniciado na porta {port}")
        except Exception as e:
            self.logger.error(f"Erro ao iniciar servidor de métricas: {e}")


# Decoradores para instrumentação automática
def time_function(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator para medir tempo de execução de funções."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Registrar métrica
                collector = get_metrics_collector()
                if hasattr(collector, metric_name):
                    metric = getattr(collector, metric_name)
                    if labels:
                        metric.labels(**labels).observe(duration)
                    else:
                        metric.observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Registrar erro se houver
                raise
        return wrapper
    return decorator


def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator para contar chamadas de funções."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Registrar métrica de sucesso
                collector = get_metrics_collector()
                if hasattr(collector, metric_name):
                    metric = getattr(collector, metric_name)
                    success_labels = {**(labels or {}), 'status': 'success'}
                    metric.labels(**success_labels).inc()
                
                return result
            except Exception as e:
                # Registrar métrica de erro
                collector = get_metrics_collector()
                if hasattr(collector, metric_name):
                    metric = getattr(collector, metric_name)
                    error_labels = {**(labels or {}), 'status': 'error'}
                    metric.labels(**error_labels).inc()
                raise
        return wrapper
    return decorator


@contextmanager
def measure_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Context manager para medir tempo de execução."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        collector = get_metrics_collector()
        if hasattr(collector, metric_name):
            metric = getattr(collector, metric_name)
            if labels:
                metric.labels(**labels).observe(duration)
            else:
                metric.observe(duration)


def get_metrics_collector() -> MetricsCollector:
    """Obtém instância singleton do coletor de métricas."""
    global _metrics_collector
    with _lock:
        if _metrics_collector is None:
            _metrics_collector = MetricsCollector()
        return _metrics_collector


def init_metrics_collector(registry: Optional[CollectorRegistry] = None) -> MetricsCollector:
    """Inicializa o coletor de métricas."""
    global _metrics_collector
    with _lock:
        _metrics_collector = MetricsCollector(registry)
        return _metrics_collector