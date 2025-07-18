# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Sistema de Detecção de Data Drift e Anomalias
Implementa testes estatísticos (KS, PSI) e detecção de anomalias com autoencoders.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error
import logging
import warnings
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class DriftTestResult:
    """Resultado de teste de drift."""
    feature_name: str
    test_type: str
    statistic: float
    p_value: Optional[float]
    threshold: float
    is_drift: bool
    severity: str  # 'low', 'medium', 'high'
    interpretation: str
    recommendation: str


@dataclass
class AnomalyResult:
    """Resultado de detecção de anomalias."""
    sample_id: Union[int, str]
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str
    confidence: float
    feature_contributions: Dict[str, float]


@dataclass
class DataQualityReport:
    """Relatório de qualidade de dados."""
    timestamp: datetime
    dataset_name: str
    total_samples: int
    drift_results: List[DriftTestResult]
    anomaly_results: List[AnomalyResult]
    overall_drift_score: float
    overall_anomaly_rate: float
    recommendations: List[str]
    data_quality_score: float


class DriftDetector(ABC):
    """Interface base para detectores de drift."""
    
    @abstractmethod
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame) -> List[DriftTestResult]:
        """Detecta drift entre datasets."""
        pass


class KSTestDriftDetector(DriftDetector):
    """Detector de drift usando teste Kolmogorov-Smirnov."""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
    
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame) -> List[DriftTestResult]:
        """
        Detecta drift usando teste KS.
        
        Args:
            reference_data: Dados de referência
            current_data: Dados atuais
            
        Returns:
            Lista de resultados de drift
        """
        results = []
        
        # Selecionar apenas colunas numéricas
        numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in current_data.columns:
                try:
                    # Dados válidos (não nulos)
                    ref_valid = reference_data[column].dropna()
                    curr_valid = current_data[column].dropna()
                    
                    if len(ref_valid) < 30 or len(curr_valid) < 30:
                        logger.warning(f"Dados insuficientes para KS test em {column}")
                        continue
                    
                    # Teste KS
                    statistic, p_value = ks_2samp(ref_valid, curr_valid)
                    
                    # Determinar drift
                    is_drift = p_value < self.threshold
                    
                    # Severidade baseada no p-value
                    if p_value < 0.001:
                        severity = 'high'
                    elif p_value < 0.01:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    # Interpretação
                    interpretation = self._interpret_ks_result(statistic, p_value)
                    
                    # Recomendação
                    recommendation = self._generate_ks_recommendation(
                        column, statistic, p_value, is_drift
                    )
                    
                    result = DriftTestResult(
                        feature_name=column,
                        test_type='KS_Test',
                        statistic=statistic,
                        p_value=p_value,
                        threshold=self.threshold,
                        is_drift=is_drift,
                        severity=severity,
                        interpretation=interpretation,
                        recommendation=recommendation
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Erro no KS test para {column}: {e}")
        
        return results
    
    def _interpret_ks_result(self, statistic: float, p_value: float) -> str:
        """Interpreta resultado do teste KS."""
        if p_value < 0.001:
            return f"Forte evidência de drift (KS={statistic:.4f}, p<0.001)"
        elif p_value < 0.01:
            return f"Evidência moderada de drift (KS={statistic:.4f}, p={p_value:.4f})"
        elif p_value < 0.05:
            return f"Evidência fraca de drift (KS={statistic:.4f}, p={p_value:.4f})"
        else:
            return f"Sem evidência de drift (KS={statistic:.4f}, p={p_value:.4f})"
    
    def _generate_ks_recommendation(self, column: str, statistic: float, 
                                   p_value: float, is_drift: bool) -> str:
        """Gera recomendação baseada no resultado KS."""
        if is_drift:
            if p_value < 0.001:
                return f"Investigar {column} urgentemente - distribuição mudou significativamente"
            elif p_value < 0.01:
                return f"Monitorar {column} de perto - possível mudança na distribuição"
            else:
                return f"Verificar {column} - pequena mudança detectada"
        else:
            return f"Distribuição de {column} estável"


class PSITestDriftDetector(DriftDetector):
    """Detector de drift usando Population Stability Index (PSI)."""
    
    def __init__(self, threshold: float = 0.1, bins: int = 10):
        self.threshold = threshold
        self.bins = bins
    
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame) -> List[DriftTestResult]:
        """
        Detecta drift usando PSI.
        
        Args:
            reference_data: Dados de referência
            current_data: Dados atuais
            
        Returns:
            Lista de resultados de drift
        """
        results = []
        
        # Selecionar apenas colunas numéricas
        numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in current_data.columns:
                try:
                    # Dados válidos
                    ref_valid = reference_data[column].dropna()
                    curr_valid = current_data[column].dropna()
                    
                    if len(ref_valid) < 100 or len(curr_valid) < 100:
                        logger.warning(f"Dados insuficientes para PSI em {column}")
                        continue
                    
                    # Calcular PSI
                    psi_value = self._calculate_psi(ref_valid, curr_valid)
                    
                    # Determinar drift
                    is_drift = psi_value > self.threshold
                    
                    # Severidade baseada no PSI
                    if psi_value > 0.25:
                        severity = 'high'
                    elif psi_value > 0.1:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    # Interpretação
                    interpretation = self._interpret_psi_result(psi_value)
                    
                    # Recomendação
                    recommendation = self._generate_psi_recommendation(column, psi_value)
                    
                    result = DriftTestResult(
                        feature_name=column,
                        test_type='PSI_Test',
                        statistic=psi_value,
                        p_value=None,
                        threshold=self.threshold,
                        is_drift=is_drift,
                        severity=severity,
                        interpretation=interpretation,
                        recommendation=recommendation
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Erro no PSI test para {column}: {e}")
        
        return results
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series) -> float:
        """Calcula Population Stability Index."""
        try:
            # Definir bins baseados nos dados de referência
            _, bin_edges = np.histogram(reference, bins=self.bins)
            
            # Garantir que os bins cubram todos os dados
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            
            if bin_edges[0] > min_val:
                bin_edges[0] = min_val
            if bin_edges[-1] < max_val:
                bin_edges[-1] = max_val
            
            # Calcular frequências
            ref_freq, _ = np.histogram(reference, bins=bin_edges)
            curr_freq, _ = np.histogram(current, bins=bin_edges)
            
            # Converter para proporções
            ref_prop = ref_freq / len(reference)
            curr_prop = curr_freq / len(current)
            
            # Evitar divisão por zero
            ref_prop = np.where(ref_prop == 0, 1e-10, ref_prop)
            curr_prop = np.where(curr_prop == 0, 1e-10, curr_prop)
            
            # Calcular PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            
            return psi
            
        except Exception as e:
            logger.error(f"Erro ao calcular PSI: {e}")
            return 0.0
    
    def _interpret_psi_result(self, psi_value: float) -> str:
        """Interpreta resultado do PSI."""
        if psi_value > 0.25:
            return f"Mudança populacional significativa (PSI={psi_value:.4f})"
        elif psi_value > 0.1:
            return f"Mudança populacional moderada (PSI={psi_value:.4f})"
        else:
            return f"População estável (PSI={psi_value:.4f})"
    
    def _generate_psi_recommendation(self, column: str, psi_value: float) -> str:
        """Gera recomendação baseada no PSI."""
        if psi_value > 0.25:
            return f"Retreinar modelo urgentemente - {column} mudou significativamente"
        elif psi_value > 0.1:
            return f"Considerar retreino do modelo - {column} apresenta mudanças"
        else:
            return f"Distribuição de {column} estável"


class AnomalyDetector:
    """Detector de anomalias usando múltiplos algoritmos."""
    
    def __init__(self, methods: List[str] = None):
        self.methods = methods or ['isolation_forest', 'autoencoder', 'one_class_svm']
        self.detectors = {}
        self.scalers = {}
        self.fitted = False
    
    def fit(self, training_data: pd.DataFrame):
        """
        Treina detectores de anomalias.
        
        Args:
            training_data: Dados de treinamento (normais)
        """
        # Selecionar apenas colunas numéricas
        numeric_data = training_data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("Nenhuma coluna numérica encontrada para treinar detectores")
        
        # Remover valores nulos
        clean_data = numeric_data.dropna()
        
        if len(clean_data) < 50:
            raise ValueError("Dados insuficientes para treinar detectores (mínimo 50 amostras)")
        
        # Padronizar dados
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_data)
        self.scalers['standard'] = scaler
        self.feature_names = clean_data.columns.tolist()
        
        # Treinar cada detector
        for method in self.methods:
            try:
                if method == 'isolation_forest':
                    detector = IsolationForest(
                        contamination=0.05,
                        random_state=42,
                        n_estimators=100
                    )
                    detector.fit(scaled_data)
                    self.detectors[method] = detector
                    
                elif method == 'autoencoder':
                    detector = self._create_autoencoder(scaled_data.shape[1])
                    detector.fit(scaled_data, scaled_data)
                    self.detectors[method] = detector
                    
                elif method == 'one_class_svm':
                    detector = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
                    detector.fit(scaled_data)
                    self.detectors[method] = detector
                    
            except Exception as e:
                logger.error(f"Erro ao treinar {method}: {e}")
        
        self.fitted = True
        logger.info(f"Detectores treinados: {list(self.detectors.keys())}")
    
    def _create_autoencoder(self, input_dim: int) -> MLPRegressor:
        """Cria autoencoder simples usando MLPRegressor."""
        hidden_dim = max(2, input_dim // 2)
        
        # Usar MLPRegressor como autoencoder
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, hidden_dim // 2, hidden_dim),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        return autoencoder
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        threshold: float = 0.5) -> List[AnomalyResult]:
        """
        Detecta anomalias nos dados.
        
        Args:
            data: Dados para análise
            threshold: Limiar para classificar como anomalia
            
        Returns:
            Lista de resultados de anomalias
        """
        if not self.fitted:
            raise ValueError("Detectores não foram treinados. Execute fit() primeiro.")
        
        # Preparar dados
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data[self.feature_names]
        clean_data = numeric_data.fillna(numeric_data.mean())
        
        if clean_data.empty:
            return []
        
        # Padronizar
        scaled_data = self.scalers['standard'].transform(clean_data)
        
        results = []
        
        for idx, row in enumerate(scaled_data):
            anomaly_scores = {}
            
            # Calcular scores para cada detector
            for method, detector in self.detectors.items():
                try:
                    if method == 'isolation_forest':
                        score = detector.decision_function([row])[0]
                        # Normalizar para [0, 1]
                        anomaly_scores[method] = max(0, -score)
                        
                    elif method == 'autoencoder':
                        reconstruction = detector.predict([row])
                        mse = mean_squared_error([row], [reconstruction])
                        # Normalizar MSE
                        anomaly_scores[method] = min(1.0, mse * 10)
                        
                    elif method == 'one_class_svm':
                        score = detector.decision_function([row])[0]
                        # Normalizar para [0, 1]
                        anomaly_scores[method] = max(0, -score)
                        
                except Exception as e:
                    logger.error(f"Erro ao calcular score com {method}: {e}")
            
            if anomaly_scores:
                # Score final (média dos métodos)
                final_score = np.mean(list(anomaly_scores.values()))
                
                # Determinar se é anomalia
                is_anomaly = final_score > threshold
                
                # Tipo de anomalia
                if final_score > 0.8:
                    anomaly_type = 'severe'
                elif final_score > 0.6:
                    anomaly_type = 'moderate'
                else:
                    anomaly_type = 'mild'
                
                # Confiança
                confidence = final_score
                
                # Contribuições das features (simplificado)
                feature_contributions = {}
                original_row = clean_data.iloc[idx]
                feature_means = clean_data.mean()
                
                for i, feature in enumerate(self.feature_names):
                    deviation = abs(original_row[feature] - feature_means[feature])
                    feature_contributions[feature] = deviation
                
                # Normalizar contribuições
                max_contrib = max(feature_contributions.values()) if feature_contributions else 1
                if max_contrib > 0:
                    feature_contributions = {
                        k: v / max_contrib for k, v in feature_contributions.items()
                    }
                
                result = AnomalyResult(
                    sample_id=idx,
                    anomaly_score=final_score,
                    is_anomaly=is_anomaly,
                    anomaly_type=anomaly_type,
                    confidence=confidence,
                    feature_contributions=feature_contributions
                )
                
                results.append(result)
        
        return results


class DataQualityMonitor:
    """Monitor completo de qualidade de dados."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.drift_detectors = {
            'ks_test': KSTestDriftDetector(
                threshold=self.config.get('ks_threshold', 0.05)
            ),
            'psi_test': PSITestDriftDetector(
                threshold=self.config.get('psi_threshold', 0.1),
                bins=self.config.get('psi_bins', 10)
            )
        }
        
        self.anomaly_detector = AnomalyDetector(
            methods=self.config.get('anomaly_methods', ['isolation_forest', 'autoencoder'])
        )
        
        self.reference_data = None
        self.reports_history = []
    
    def set_reference_data(self, reference_data: pd.DataFrame):
        """Define dados de referência."""
        self.reference_data = reference_data.copy()
        
        # Treinar detector de anomalias
        try:
            self.anomaly_detector.fit(reference_data)
            logger.info("Detector de anomalias treinado com dados de referência")
        except Exception as e:
            logger.error(f"Erro ao treinar detector de anomalias: {e}")
    
    def analyze_data_quality(self, current_data: pd.DataFrame,
                           dataset_name: str = "unnamed") -> DataQualityReport:
        """
        Analisa qualidade dos dados atuais.
        
        Args:
            current_data: Dados atuais
            dataset_name: Nome do dataset
            
        Returns:
            Relatório de qualidade
        """
        if self.reference_data is None:
            raise ValueError("Dados de referência não foram definidos")
        
        # Detectar drift
        drift_results = []
        for detector_name, detector in self.drift_detectors.items():
            try:
                results = detector.detect_drift(self.reference_data, current_data)
                drift_results.extend(results)
            except Exception as e:
                logger.error(f"Erro no detector {detector_name}: {e}")
        
        # Detectar anomalias
        anomaly_results = []
        try:
            anomaly_results = self.anomaly_detector.detect_anomalies(current_data)
        except Exception as e:
            logger.error(f"Erro na detecção de anomalias: {e}")
        
        # Calcular métricas gerais
        overall_drift_score = self._calculate_overall_drift_score(drift_results)
        overall_anomaly_rate = len([r for r in anomaly_results if r.is_anomaly]) / len(anomaly_results) if anomaly_results else 0
        
        # Gerar recomendações
        recommendations = self._generate_recommendations(drift_results, anomaly_results)
        
        # Calcular score geral de qualidade
        data_quality_score = self._calculate_data_quality_score(
            overall_drift_score, overall_anomaly_rate
        )
        
        # Criar relatório
        report = DataQualityReport(
            timestamp=datetime.now(),
            dataset_name=dataset_name,
            total_samples=len(current_data),
            drift_results=drift_results,
            anomaly_results=anomaly_results,
            overall_drift_score=overall_drift_score,
            overall_anomaly_rate=overall_anomaly_rate,
            recommendations=recommendations,
            data_quality_score=data_quality_score
        )
        
        # Salvar no histórico
        self.reports_history.append(report)
        
        # Manter apenas últimos 100 relatórios
        if len(self.reports_history) > 100:
            self.reports_history = self.reports_history[-100:]
        
        logger.info(f"Análise de qualidade concluída para {dataset_name}")
        return report
    
    def _calculate_overall_drift_score(self, drift_results: List[DriftTestResult]) -> float:
        """Calcula score geral de drift."""
        if not drift_results:
            return 0.0
        
        drift_features = [r for r in drift_results if r.is_drift]
        return len(drift_features) / len(drift_results)
    
    def _calculate_data_quality_score(self, drift_score: float, anomaly_rate: float) -> float:
        """Calcula score geral de qualidade (0-100)."""
        # Penalizar drift e anomalias
        drift_penalty = drift_score * 30  # Max 30 pontos
        anomaly_penalty = anomaly_rate * 40  # Max 40 pontos
        
        quality_score = 100 - drift_penalty - anomaly_penalty
        return max(0, quality_score)
    
    def _generate_recommendations(self, drift_results: List[DriftTestResult],
                                 anomaly_results: List[AnomalyResult]) -> List[str]:
        """Gera recomendações baseadas nos resultados."""
        recommendations = []
        
        # Recomendações para drift
        high_drift_features = [r for r in drift_results if r.is_drift and r.severity == 'high']
        if high_drift_features:
            recommendations.append(
                f"Retreinar modelo urgentemente - {len(high_drift_features)} features com drift alto"
            )
        
        medium_drift_features = [r for r in drift_results if r.is_drift and r.severity == 'medium']
        if medium_drift_features:
            recommendations.append(
                f"Monitorar de perto - {len(medium_drift_features)} features com drift moderado"
            )
        
        # Recomendações para anomalias
        severe_anomalies = [r for r in anomaly_results if r.is_anomaly and r.anomaly_type == 'severe']
        if severe_anomalies:
            recommendations.append(
                f"Investigar {len(severe_anomalies)} anomalias severas detectadas"
            )
        
        anomaly_rate = len([r for r in anomaly_results if r.is_anomaly]) / len(anomaly_results) if anomaly_results else 0
        if anomaly_rate > 0.1:
            recommendations.append(
                f"Taxa de anomalias alta ({anomaly_rate:.1%}) - verificar qualidade dos dados"
            )
        
        # Recomendações gerais
        if not recommendations:
            recommendations.append("Qualidade dos dados dentro dos parâmetros normais")
        
        return recommendations
    
    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Analisa tendências de qualidade dos dados."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_reports = [
            r for r in self.reports_history
            if r.timestamp > cutoff_date
        ]
        
        if not recent_reports:
            return {'message': 'Dados insuficientes para análise de tendência'}
        
        # Calcular tendências
        drift_scores = [r.overall_drift_score for r in recent_reports]
        anomaly_rates = [r.overall_anomaly_rate for r in recent_reports]
        quality_scores = [r.data_quality_score for r in recent_reports]
        
        return {
            'reports_count': len(recent_reports),
            'drift_trend': {
                'current': drift_scores[-1] if drift_scores else 0,
                'average': np.mean(drift_scores) if drift_scores else 0,
                'trend': 'increasing' if len(drift_scores) > 1 and drift_scores[-1] > drift_scores[0] else 'stable'
            },
            'anomaly_trend': {
                'current': anomaly_rates[-1] if anomaly_rates else 0,
                'average': np.mean(anomaly_rates) if anomaly_rates else 0,
                'trend': 'increasing' if len(anomaly_rates) > 1 and anomaly_rates[-1] > anomaly_rates[0] else 'stable'
            },
            'quality_trend': {
                'current': quality_scores[-1] if quality_scores else 100,
                'average': np.mean(quality_scores) if quality_scores else 100,
                'trend': 'improving' if len(quality_scores) > 1 and quality_scores[-1] > quality_scores[0] else 'stable'
            }
        }
    
    def export_report(self, report: DataQualityReport, filepath: str):
        """Exporta relatório para arquivo."""
        report_data = {
            'timestamp': report.timestamp.isoformat(),
            'dataset_name': report.dataset_name,
            'total_samples': report.total_samples,
            'overall_drift_score': report.overall_drift_score,
            'overall_anomaly_rate': report.overall_anomaly_rate,
            'data_quality_score': report.data_quality_score,
            'recommendations': report.recommendations,
            'drift_results': [
                {
                    'feature_name': r.feature_name,
                    'test_type': r.test_type,
                    'statistic': r.statistic,
                    'p_value': r.p_value,
                    'is_drift': r.is_drift,
                    'severity': r.severity,
                    'interpretation': r.interpretation,
                    'recommendation': r.recommendation
                }
                for r in report.drift_results
            ],
            'anomaly_summary': {
                'total_anomalies': len([r for r in report.anomaly_results if r.is_anomaly]),
                'severe_anomalies': len([r for r in report.anomaly_results if r.is_anomaly and r.anomaly_type == 'severe']),
                'moderate_anomalies': len([r for r in report.anomaly_results if r.is_anomaly and r.anomaly_type == 'moderate']),
                'mild_anomalies': len([r for r in report.anomaly_results if r.is_anomaly and r.anomaly_type == 'mild'])
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório exportado para {filepath}")


def create_data_quality_monitor(config: Dict[str, Any] = None) -> DataQualityMonitor:
    """
    Cria monitor de qualidade de dados.
    
    Args:
        config: Configuração do monitor
        
    Returns:
        Monitor configurado
    """
    return DataQualityMonitor(config)


def get_default_drift_config() -> Dict[str, Any]:
    """Retorna configuração padrão para detecção de drift."""
    return {
        'ks_threshold': 0.05,
        'psi_threshold': 0.1,
        'psi_bins': 10,
        'anomaly_methods': ['isolation_forest', 'autoencoder'],
        'anomaly_threshold': 0.5
    }