"""
Testes unitários para o módulo de métricas e monitoramento
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List


class TestMetricsCollection:
    """Testes para coleta de métricas"""
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock para coletor de métricas"""
        collector = Mock()
        collector.counter = Mock()
        collector.histogram = Mock()
        collector.gauge = Mock()
        collector.timer = Mock()
        return collector
    
    def test_counter_metrics(self, mock_metrics_collector):
        """Testa métricas de contador"""
        def increment_counter(name: str, labels: Dict[str, str] = None, value: float = 1):
            mock_metrics_collector.counter.inc.return_value = None
            mock_metrics_collector.counter.inc(value)
            return True
        
        # Testar incremento simples
        result = increment_counter("api_requests_total")
        assert result is True
        mock_metrics_collector.counter.inc.assert_called_with(1)
        
        # Testar incremento com valor customizado
        increment_counter("api_requests_total", value=5)
        mock_metrics_collector.counter.inc.assert_called_with(5)
    
    def test_histogram_metrics(self, mock_metrics_collector):
        """Testa métricas de histograma"""
        def observe_histogram(name: str, value: float, labels: Dict[str, str] = None):
            mock_metrics_collector.histogram.observe.return_value = None
            mock_metrics_collector.histogram.observe(value)
            return True
        
        # Testar observação de valor
        result = observe_histogram("request_duration_seconds", 0.125)
        assert result is True
        mock_metrics_collector.histogram.observe.assert_called_with(0.125)
    
    def test_gauge_metrics(self, mock_metrics_collector):
        """Testa métricas de gauge"""
        def set_gauge(name: str, value: float, labels: Dict[str, str] = None):
            mock_metrics_collector.gauge.set.return_value = None
            mock_metrics_collector.gauge.set(value)
            return True
        
        def inc_gauge(name: str, value: float = 1, labels: Dict[str, str] = None):
            mock_metrics_collector.gauge.inc.return_value = None
            mock_metrics_collector.gauge.inc(value)
            return True
        
        def dec_gauge(name: str, value: float = 1, labels: Dict[str, str] = None):
            mock_metrics_collector.gauge.dec.return_value = None
            mock_metrics_collector.gauge.dec(value)
            return True
        
        # Testar definição de valor
        result = set_gauge("active_connections", 42)
        assert result is True
        mock_metrics_collector.gauge.set.assert_called_with(42)
        
        # Testar incremento
        result = inc_gauge("memory_usage_bytes", 1024)
        assert result is True
        mock_metrics_collector.gauge.inc.assert_called_with(1024)
        
        # Testar decremento
        result = dec_gauge("active_sessions", 1)
        assert result is True
        mock_metrics_collector.gauge.dec.assert_called_with(1)
    
    def test_timing_decorator(self):
        """Testa decorator de timing"""
        execution_times = []
        
        def measure_time(func_name: str = None):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        end_time = time.time()
                        duration = end_time - start_time
                        execution_times.append({
                            'function': func_name or func.__name__,
                            'duration': duration,
                            'timestamp': start_time
                        })
                return wrapper
            return decorator
        
        @measure_time("test_function")
        def slow_function():
            time.sleep(0.1)  # Simular operação lenta
            return "done"
        
        # Executar função
        result = slow_function()
        assert result == "done"
        assert len(execution_times) == 1
        assert execution_times[0]['function'] == "test_function"
        assert execution_times[0]['duration'] >= 0.1
    
    def test_business_metrics(self):
        """Testa métricas de negócio específicas"""
        business_metrics = {
            'evaluations_created': 0,
            'models_trained': 0,
            'predictions_made': 0,
            'users_active': 0
        }
        
        def track_evaluation_created(evaluation_id: str, tenant_id: str, model_type: str):
            business_metrics['evaluations_created'] += 1
            return {
                'metric': 'evaluation_created',
                'evaluation_id': evaluation_id,
                'tenant_id': tenant_id,
                'model_type': model_type,
                'timestamp': datetime.now()
            }
        
        def track_model_trained(model_id: str, training_time: float, performance: Dict[str, float]):
            business_metrics['models_trained'] += 1
            return {
                'metric': 'model_trained',
                'model_id': model_id,
                'training_time': training_time,
                'performance': performance,
                'timestamp': datetime.now()
            }
        
        def track_predictions_made(model_id: str, count: int, inference_time: float):
            business_metrics['predictions_made'] += count
            return {
                'metric': 'predictions_made',
                'model_id': model_id,
                'count': count,
                'inference_time': inference_time,
                'timestamp': datetime.now()
            }
        
        # Testar rastreamento de métricas
        eval_metric = track_evaluation_created("eval_123", "tenant_456", "elastic_net")
        assert business_metrics['evaluations_created'] == 1
        assert eval_metric['evaluation_id'] == "eval_123"
        
        model_metric = track_model_trained("model_789", 120.5, {"r2": 0.85, "rmse": 45000})
        assert business_metrics['models_trained'] == 1
        assert model_metric['training_time'] == 120.5
        
        pred_metric = track_predictions_made("model_789", 5, 0.25)
        assert business_metrics['predictions_made'] == 5
        assert pred_metric['count'] == 5
    
    def test_error_tracking(self):
        """Testa rastreamento de erros"""
        error_log = []
        
        def track_error(error_type: str, error_message: str, context: Dict[str, Any] = None):
            error_entry = {
                'error_type': error_type,
                'error_message': error_message,
                'context': context or {},
                'timestamp': datetime.now(),
                'severity': 'error'
            }
            error_log.append(error_entry)
            return error_entry
        
        def track_warning(warning_message: str, context: Dict[str, Any] = None):
            warning_entry = {
                'error_type': 'warning',
                'error_message': warning_message,
                'context': context or {},
                'timestamp': datetime.now(),
                'severity': 'warning'
            }
            error_log.append(warning_entry)
            return warning_entry
        
        # Testar rastreamento de erro
        error = track_error(
            "ValidationError", 
            "Dados insuficientes para treinamento",
            {"evaluation_id": "eval_123", "data_size": 5}
        )
        assert len(error_log) == 1
        assert error['error_type'] == "ValidationError"
        assert error['severity'] == 'error'
        
        # Testar rastreamento de warning
        warning = track_warning(
            "Amostra pequena detectada",
            {"data_size": 25, "recommended_min": 30}
        )
        assert len(error_log) == 2
        assert warning['severity'] == 'warning'
    
    def test_performance_monitoring(self):
        """Testa monitoramento de performance"""
        performance_data = []
        
        class PerformanceMonitor:
            def __init__(self):
                self.start_time = None
                self.metrics = {}
            
            def start_monitoring(self):
                self.start_time = time.time()
                self.metrics = {
                    'cpu_usage': 0.0,
                    'memory_usage': 0.0,
                    'disk_usage': 0.0
                }
            
            def record_resource_usage(self, cpu: float, memory: float, disk: float):
                self.metrics.update({
                    'cpu_usage': cpu,
                    'memory_usage': memory,
                    'disk_usage': disk
                })
            
            def stop_monitoring(self):
                if self.start_time:
                    duration = time.time() - self.start_time
                    performance_entry = {
                        'duration': duration,
                        'metrics': self.metrics.copy(),
                        'timestamp': datetime.now()
                    }
                    performance_data.append(performance_entry)
                    return performance_entry
                return None
        
        # Testar monitoramento
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        assert monitor.start_time is not None
        
        # Simular uso de recursos
        monitor.record_resource_usage(cpu=25.5, memory=512.0, disk=1024.0)
        assert monitor.metrics['cpu_usage'] == 25.5
        
        time.sleep(0.01)  # Pequena pausa
        result = monitor.stop_monitoring()
        
        assert result is not None
        assert result['duration'] > 0
        assert len(performance_data) == 1
        assert performance_data[0]['metrics']['cpu_usage'] == 25.5
    
    def test_custom_metrics_registry(self):
        """Testa registro de métricas customizadas"""
        metrics_registry = {}
        
        def register_metric(name: str, metric_type: str, description: str, labels: List[str] = None):
            metrics_registry[name] = {
                'type': metric_type,
                'description': description,
                'labels': labels or [],
                'values': [],
                'created_at': datetime.now()
            }
            return True
        
        def record_metric_value(name: str, value: float, labels: Dict[str, str] = None):
            if name not in metrics_registry:
                raise ValueError(f"Métrica '{name}' não registrada")
            
            metrics_registry[name]['values'].append({
                'value': value,
                'labels': labels or {},
                'timestamp': datetime.now()
            })
            return True
        
        def get_metric_summary(name: str) -> Dict[str, Any]:
            if name not in metrics_registry:
                return None
            
            metric = metrics_registry[name]
            values = [v['value'] for v in metric['values']]
            
            if not values:
                return {
                    'name': name,
                    'count': 0,
                    'description': metric['description']
                }
            
            return {
                'name': name,
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'description': metric['description'],
                'last_updated': metric['values'][-1]['timestamp']
            }
        
        # Registrar métricas customizadas
        assert register_metric(
            "model_accuracy", 
            "gauge", 
            "Precisão do modelo de ML",
            ["model_type", "tenant_id"]
        ) is True
        
        assert register_metric(
            "api_response_time",
            "histogram", 
            "Tempo de resposta da API",
            ["endpoint", "method"]
        ) is True
        
        # Registrar valores
        assert record_metric_value("model_accuracy", 0.85, {"model_type": "elastic_net", "tenant_id": "tenant_1"})
        assert record_metric_value("model_accuracy", 0.92, {"model_type": "xgboost", "tenant_id": "tenant_1"})
        assert record_metric_value("api_response_time", 0.125, {"endpoint": "/predict", "method": "POST"})
        
        # Obter sumários
        accuracy_summary = get_metric_summary("model_accuracy")
        assert accuracy_summary['count'] == 2
        assert accuracy_summary['min'] == 0.85
        assert accuracy_summary['max'] == 0.92
        assert accuracy_summary['avg'] == 0.885
        
        response_time_summary = get_metric_summary("api_response_time")
        assert response_time_summary['count'] == 1
        assert response_time_summary['avg'] == 0.125
        
        # Métrica inexistente
        assert get_metric_summary("nonexistent_metric") is None


class TestDataDriftMonitoring:
    """Testes para monitoramento de data drift"""
    
    def test_statistical_drift_detection(self):
        """Testa detecção de drift estatístico"""
        import numpy as np
        
        def calculate_drift_score(reference_data: np.ndarray, current_data: np.ndarray) -> float:
            """Calcula score de drift usando teste de Kolmogorov-Smirnov"""
            from scipy import stats
            
            # Simular teste KS
            ks_statistic = abs(np.mean(current_data) - np.mean(reference_data)) / np.std(reference_data)
            return min(ks_statistic, 1.0)  # Normalizar entre 0 e 1
        
        def detect_drift(reference_data: np.ndarray, current_data: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
            drift_score = calculate_drift_score(reference_data, current_data)
            
            return {
                'drift_detected': drift_score > threshold,
                'drift_score': drift_score,
                'threshold': threshold,
                'severity': 'high' if drift_score > 0.3 else 'medium' if drift_score > 0.1 else 'low',
                'timestamp': datetime.now()
            }
        
        # Dados de referência (baseline)
        reference_data = np.random.normal(100, 15, 1000)  # Média 100, desvio 15
        
        # Dados similares (sem drift)
        similar_data = np.random.normal(102, 16, 1000)    # Média 102, desvio 16
        
        # Dados com drift
        drift_data = np.random.normal(150, 20, 1000)      # Média 150, desvio 20
        
        # Testar dados similares
        similar_result = detect_drift(reference_data, similar_data)
        assert similar_result['drift_detected'] is False
        assert similar_result['severity'] == 'low'
        
        # Testar dados com drift
        drift_result = detect_drift(reference_data, drift_data)
        assert drift_result['drift_detected'] is True
        assert drift_result['drift_score'] > 0.1
    
    def test_feature_drift_monitoring(self):
        """Testa monitoramento de drift por feature"""
        drift_history = []
        
        def monitor_feature_drift(feature_name: str, reference_stats: Dict[str, float], 
                                current_stats: Dict[str, float]) -> Dict[str, Any]:
            
            drift_indicators = {}
            
            # Verificar mudança na média
            if 'mean' in reference_stats and 'mean' in current_stats:
                mean_change = abs(current_stats['mean'] - reference_stats['mean']) / reference_stats['mean']
                drift_indicators['mean_drift'] = mean_change
            
            # Verificar mudança no desvio padrão
            if 'std' in reference_stats and 'std' in current_stats:
                std_change = abs(current_stats['std'] - reference_stats['std']) / reference_stats['std']
                drift_indicators['std_drift'] = std_change
            
            # Verificar mudança na distribuição (simulado)
            distribution_drift = max(drift_indicators.values()) if drift_indicators else 0
            
            result = {
                'feature_name': feature_name,
                'drift_indicators': drift_indicators,
                'overall_drift_score': distribution_drift,
                'drift_detected': distribution_drift > 0.1,
                'timestamp': datetime.now()
            }
            
            drift_history.append(result)
            return result
        
        # Simular monitoramento de diferentes features
        
        # Feature sem drift
        area_result = monitor_feature_drift(
            'area',
            {'mean': 100.0, 'std': 15.0},
            {'mean': 102.0, 'std': 16.0}
        )
        assert area_result['drift_detected'] is False
        assert area_result['overall_drift_score'] < 0.1
        
        # Feature com drift significativo
        price_result = monitor_feature_drift(
            'valor',
            {'mean': 400000.0, 'std': 50000.0},
            {'mean': 500000.0, 'std': 80000.0}
        )
        assert price_result['drift_detected'] is True
        assert price_result['overall_drift_score'] > 0.1
        
        assert len(drift_history) == 2
    
    def test_model_performance_drift(self):
        """Testa detecção de drift na performance do modelo"""
        performance_history = []
        
        def track_model_performance(model_id: str, predictions: List[float], 
                                  actual_values: List[float]) -> Dict[str, Any]:
            
            # Calcular métricas de performance
            errors = [abs(p - a) for p, a in zip(predictions, actual_values)]
            mae = sum(errors) / len(errors)
            rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
            
            performance_entry = {
                'model_id': model_id,
                'mae': mae,
                'rmse': rmse,
                'sample_size': len(predictions),
                'timestamp': datetime.now()
            }
            
            performance_history.append(performance_entry)
            return performance_entry
        
        def detect_performance_drift(model_id: str, lookback_days: int = 7) -> Dict[str, Any]:
            # Filtrar histórico recente
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_performance = [
                p for p in performance_history 
                if p['model_id'] == model_id and p['timestamp'] >= cutoff_date
            ]
            
            if len(recent_performance) < 2:
                return {'drift_detected': False, 'reason': 'insufficient_data'}
            
            # Calcular tendência
            recent_mae = [p['mae'] for p in recent_performance[-3:]]  # Últimas 3 medições
            baseline_mae = [p['mae'] for p in recent_performance[:3]]  # Primeiras 3 medições
            
            if baseline_mae and recent_mae:
                baseline_avg = sum(baseline_mae) / len(baseline_mae)
                recent_avg = sum(recent_mae) / len(recent_mae)
                
                drift_ratio = (recent_avg - baseline_avg) / baseline_avg
                
                return {
                    'drift_detected': abs(drift_ratio) > 0.1,  # 10% de mudança
                    'drift_ratio': drift_ratio,
                    'baseline_mae': baseline_avg,
                    'recent_mae': recent_avg,
                    'trend': 'degrading' if drift_ratio > 0.1 else 'improving' if drift_ratio < -0.1 else 'stable'
                }
            
            return {'drift_detected': False, 'reason': 'calculation_error'}
        
        # Simular dados de performance ao longo do tempo
        model_id = "model_123"
        
        # Performance inicial boa
        for i in range(3):
            track_model_performance(
                model_id,
                [400000, 500000, 350000],
                [410000, 490000, 360000]  # Erros pequenos
            )
        
        # Performance degradada
        for i in range(3):
            track_model_performance(
                model_id,
                [400000, 500000, 350000],
                [450000, 450000, 400000]  # Erros maiores
            )
        
        # Detectar drift
        drift_result = detect_performance_drift(model_id)
        
        assert drift_result['drift_detected'] is True
        assert drift_result['trend'] == 'degrading'
        assert drift_result['recent_mae'] > drift_result['baseline_mae']


@pytest.mark.integration
class TestMetricsIntegration:
    """Testes de integração para sistema de métricas"""
    
    def test_end_to_end_monitoring_workflow(self):
        """Testa fluxo completo de monitoramento"""
        # Simular sistema completo de monitoramento
        monitoring_system = {
            'metrics': {},
            'alerts': [],
            'performance_data': []
        }
        
        def initialize_monitoring():
            monitoring_system['metrics'] = {
                'api_requests': 0,
                'model_predictions': 0,
                'errors': 0,
                'active_users': 0
            }
            return True
        
        def record_api_request(endpoint: str, method: str, duration: float, status_code: int):
            monitoring_system['metrics']['api_requests'] += 1
            
            if status_code >= 400:
                monitoring_system['metrics']['errors'] += 1
            
            # Alertar se tempo de resposta muito alto
            if duration > 5.0:
                monitoring_system['alerts'].append({
                    'type': 'slow_response',
                    'endpoint': endpoint,
                    'duration': duration,
                    'timestamp': datetime.now()
                })
            
            return True
        
        def record_prediction(model_id: str, inference_time: float, batch_size: int):
            monitoring_system['metrics']['model_predictions'] += batch_size
            
            monitoring_system['performance_data'].append({
                'model_id': model_id,
                'inference_time': inference_time,
                'batch_size': batch_size,
                'timestamp': datetime.now()
            })
            
            return True
        
        def check_system_health() -> Dict[str, Any]:
            total_requests = monitoring_system['metrics']['api_requests']
            total_errors = monitoring_system['metrics']['errors']
            
            error_rate = (total_errors / total_requests) if total_requests > 0 else 0
            
            return {
                'status': 'healthy' if error_rate < 0.05 else 'warning' if error_rate < 0.1 else 'critical',
                'error_rate': error_rate,
                'total_requests': total_requests,
                'total_errors': total_errors,
                'active_alerts': len(monitoring_system['alerts'])
            }
        
        # Executar workflow completo
        
        # 1. Inicializar monitoramento
        assert initialize_monitoring() is True
        
        # 2. Simular atividade normal
        for i in range(100):
            record_api_request('/predict', 'POST', 0.25, 200)
        
        for i in range(10):
            record_prediction('model_123', 0.15, 5)
        
        # 3. Simular alguns erros
        for i in range(3):
            record_api_request('/predict', 'POST', 0.30, 500)
        
        # 4. Simular resposta lenta
        record_api_request('/train', 'POST', 6.0, 200)
        
        # 5. Verificar saúde do sistema
        health = check_system_health()
        
        assert health['total_requests'] == 104  # 100 + 3 + 1
        assert health['total_errors'] == 3
        assert health['error_rate'] < 0.05  # 3/104 ≈ 0.029
        assert health['status'] == 'healthy'
        assert health['active_alerts'] == 1  # Alerta de resposta lenta
        
        # 6. Verificar dados de performance
        assert len(monitoring_system['performance_data']) == 10
        assert monitoring_system['metrics']['model_predictions'] == 50  # 10 batches * 5 each