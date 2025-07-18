# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Model Validator para MLOps
Sistema de validação e teste de modelos antes do deployment.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging
from abc import ABC, abstractmethod
import joblib
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from ..monitoring.logging_config import get_logger
from ..monitoring.data_drift import DataDriftDetector
import structlog


class ValidationStatus(Enum):
    """Status da validação."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class ValidationSeverity(Enum):
    """Severidade da validação."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Resultado de uma validação."""
    validator_name: str
    status: ValidationStatus
    severity: ValidationSeverity
    score: float
    threshold: float
    message: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            **asdict(self),
            'status': self.status.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ValidationReport:
    """Relatório completo de validação."""
    model_id: str
    version: str
    validation_id: str
    overall_status: ValidationStatus
    results: List[ValidationResult]
    metrics: Dict[str, float]
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    created_at: datetime
    total_execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'validation_id': self.validation_id,
            'overall_status': self.overall_status.value,
            'results': [r.to_dict() for r in self.results],
            'metrics': self.metrics,
            'warnings': self.warnings,
            'errors': self.errors,
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat(),
            'total_execution_time': self.total_execution_time
        }


class BaseValidator(ABC):
    """Classe base para validadores."""
    
    def __init__(self, name: str, threshold: float, severity: ValidationSeverity):
        self.name = name
        self.threshold = threshold
        self.severity = severity
        self.logger = get_logger(f"validator_{name}")
    
    @abstractmethod
    async def validate(self, model: Any, data: pd.DataFrame, 
                      target: pd.Series = None, **kwargs) -> ValidationResult:
        """Executa validação."""
        pass
    
    def create_result(self, status: ValidationStatus, score: float, 
                     message: str, details: Dict[str, Any] = None,
                     execution_time: float = 0.0) -> ValidationResult:
        """Cria resultado de validação."""
        return ValidationResult(
            validator_name=self.name,
            status=status,
            severity=self.severity,
            score=score,
            threshold=self.threshold,
            message=message,
            details=details or {},
            execution_time=execution_time,
            timestamp=datetime.now()
        )


class PerformanceValidator(BaseValidator):
    """Validador de performance do modelo."""
    
    def __init__(self, r2_threshold: float = 0.7, mae_threshold: float = 50000):
        super().__init__("performance", r2_threshold, ValidationSeverity.HIGH)
        self.mae_threshold = mae_threshold
    
    async def validate(self, model: Any, data: pd.DataFrame, 
                      target: pd.Series = None, **kwargs) -> ValidationResult:
        """Valida performance do modelo."""
        start_time = datetime.now()
        
        try:
            if target is None:
                return self.create_result(
                    ValidationStatus.FAILED,
                    0.0,
                    "Target values required for performance validation",
                    execution_time=0.0
                )
            
            # Fazer predições
            predictions = model.predict(data)
            
            # Calcular métricas
            r2 = r2_score(target, predictions)
            mae = mean_absolute_error(target, predictions)
            rmse = np.sqrt(mean_squared_error(target, predictions))
            
            # Avaliar performance
            r2_passed = r2 >= self.threshold
            mae_passed = mae <= self.mae_threshold
            
            overall_passed = r2_passed and mae_passed
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            details = {
                "r2_score": r2,
                "mae": mae,
                "rmse": rmse,
                "r2_threshold": self.threshold,
                "mae_threshold": self.mae_threshold,
                "r2_passed": r2_passed,
                "mae_passed": mae_passed,
                "predictions_count": len(predictions)
            }
            
            if overall_passed:
                status = ValidationStatus.PASSED
                message = f"Model performance acceptable (R² = {r2:.3f}, MAE = {mae:.0f})"
            else:
                status = ValidationStatus.FAILED
                failed_metrics = []
                if not r2_passed:
                    failed_metrics.append(f"R² = {r2:.3f} < {self.threshold}")
                if not mae_passed:
                    failed_metrics.append(f"MAE = {mae:.0f} > {self.mae_threshold}")
                message = f"Model performance insufficient: {', '.join(failed_metrics)}"
            
            return self.create_result(
                status, r2, message, details, execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return self.create_result(
                ValidationStatus.FAILED,
                0.0,
                f"Performance validation failed: {str(e)}",
                {"error": str(e)},
                execution_time
            )


class DataDriftValidator(BaseValidator):
    """Validador de data drift."""
    
    def __init__(self, drift_threshold: float = 0.05):
        super().__init__("data_drift", drift_threshold, ValidationSeverity.MEDIUM)
        self.drift_detector = DataDriftDetector()
    
    async def validate(self, model: Any, data: pd.DataFrame, 
                      target: pd.Series = None, **kwargs) -> ValidationResult:
        """Valida data drift."""
        start_time = datetime.now()
        
        try:
            reference_data = kwargs.get('reference_data')
            if reference_data is None:
                return self.create_result(
                    ValidationStatus.WARNING,
                    0.0,
                    "No reference data provided for drift detection",
                    execution_time=0.0
                )
            
            # Detectar drift
            drift_report = self.drift_detector.detect_drift(reference_data, data)
            
            # Analisar resultados
            significant_drift_features = [
                feature for feature, p_value in drift_report.ks_test_results.items()
                if p_value < self.threshold
            ]
            
            drift_ratio = len(significant_drift_features) / len(drift_report.ks_test_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            details = {
                "drift_ratio": drift_ratio,
                "significant_drift_features": significant_drift_features,
                "total_features": len(drift_report.ks_test_results),
                "ks_test_results": drift_report.ks_test_results,
                "psi_scores": drift_report.psi_scores,
                "drift_threshold": self.threshold
            }
            
            if drift_ratio <= 0.1:  # Menos de 10% das features com drift
                status = ValidationStatus.PASSED
                message = f"No significant data drift detected ({drift_ratio:.1%} features affected)"
            elif drift_ratio <= 0.3:  # Entre 10% e 30%
                status = ValidationStatus.WARNING
                message = f"Moderate data drift detected ({drift_ratio:.1%} features affected)"
            else:  # Mais de 30%
                status = ValidationStatus.FAILED
                message = f"Significant data drift detected ({drift_ratio:.1%} features affected)"
            
            return self.create_result(
                status, 1 - drift_ratio, message, details, execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return self.create_result(
                ValidationStatus.FAILED,
                0.0,
                f"Data drift validation failed: {str(e)}",
                {"error": str(e)},
                execution_time
            )


class StabilityValidator(BaseValidator):
    """Validador de estabilidade do modelo."""
    
    def __init__(self, cv_threshold: float = 0.8):
        super().__init__("stability", cv_threshold, ValidationSeverity.MEDIUM)
    
    async def validate(self, model: Any, data: pd.DataFrame, 
                      target: pd.Series = None, **kwargs) -> ValidationResult:
        """Valida estabilidade através de cross-validation."""
        start_time = datetime.now()
        
        try:
            if target is None:
                return self.create_result(
                    ValidationStatus.FAILED,
                    0.0,
                    "Target values required for stability validation",
                    execution_time=0.0
                )
            
            # Cross-validation
            cv_scores = cross_val_score(model, data, target, cv=5, scoring='r2')
            
            # Calcular métricas de estabilidade
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            cv_min = np.min(cv_scores)
            cv_max = np.max(cv_scores)
            
            # Coefficient of variation
            cv_coefficient = cv_std / cv_mean if cv_mean != 0 else float('inf')
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            details = {
                "cv_scores": cv_scores.tolist(),
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "cv_min": cv_min,
                "cv_max": cv_max,
                "cv_coefficient": cv_coefficient,
                "cv_threshold": self.threshold
            }
            
            if cv_mean >= self.threshold and cv_coefficient <= 0.2:
                status = ValidationStatus.PASSED
                message = f"Model stability acceptable (CV mean = {cv_mean:.3f}, CV = {cv_coefficient:.3f})"
            elif cv_mean >= self.threshold:
                status = ValidationStatus.WARNING
                message = f"Model performance good but unstable (CV mean = {cv_mean:.3f}, CV = {cv_coefficient:.3f})"
            else:
                status = ValidationStatus.FAILED
                message = f"Model stability insufficient (CV mean = {cv_mean:.3f} < {self.threshold})"
            
            return self.create_result(
                status, cv_mean, message, details, execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return self.create_result(
                ValidationStatus.FAILED,
                0.0,
                f"Stability validation failed: {str(e)}",
                {"error": str(e)},
                execution_time
            )


class BiasValidator(BaseValidator):
    """Validador de bias do modelo."""
    
    def __init__(self, bias_threshold: float = 0.1):
        super().__init__("bias", bias_threshold, ValidationSeverity.HIGH)
    
    async def validate(self, model: Any, data: pd.DataFrame, 
                      target: pd.Series = None, **kwargs) -> ValidationResult:
        """Valida bias do modelo."""
        start_time = datetime.now()
        
        try:
            if target is None:
                return self.create_result(
                    ValidationStatus.FAILED,
                    0.0,
                    "Target values required for bias validation",
                    execution_time=0.0
                )
            
            # Fazer predições
            predictions = model.predict(data)
            residuals = target - predictions
            
            # Calcular métricas de bias
            mean_residual = np.mean(residuals)
            residual_std = np.std(residuals)
            
            # Teste de normalidade dos resíduos
            _, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])
            
            # Teste de homocedasticidade (simplificado)
            abs_residuals = np.abs(residuals)
            correlation_coef = np.corrcoef(predictions, abs_residuals)[0, 1]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            details = {
                "mean_residual": mean_residual,
                "residual_std": residual_std,
                "shapiro_p_value": shapiro_p,
                "heteroscedasticity_correlation": correlation_coef,
                "bias_threshold": self.threshold
            }
            
            # Avaliar bias
            bias_score = abs(mean_residual) / residual_std if residual_std != 0 else 0
            
            if bias_score <= self.threshold:
                status = ValidationStatus.PASSED
                message = f"No significant bias detected (bias score = {bias_score:.3f})"
            elif bias_score <= self.threshold * 2:
                status = ValidationStatus.WARNING
                message = f"Moderate bias detected (bias score = {bias_score:.3f})"
            else:
                status = ValidationStatus.FAILED
                message = f"Significant bias detected (bias score = {bias_score:.3f})"
            
            return self.create_result(
                status, 1 - bias_score, message, details, execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return self.create_result(
                ValidationStatus.FAILED,
                0.0,
                f"Bias validation failed: {str(e)}",
                {"error": str(e)},
                execution_time
            )


class DataQualityValidator(BaseValidator):
    """Validador de qualidade dos dados."""
    
    def __init__(self, missing_threshold: float = 0.05):
        super().__init__("data_quality", missing_threshold, ValidationSeverity.MEDIUM)
    
    async def validate(self, model: Any, data: pd.DataFrame, 
                      target: pd.Series = None, **kwargs) -> ValidationResult:
        """Valida qualidade dos dados."""
        start_time = datetime.now()
        
        try:
            # Analisar dados faltantes
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            
            # Analisar outliers (IQR method)
            outlier_counts = {}
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | 
                               (data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts[col] = outlier_count
            
            total_outliers = sum(outlier_counts.values())
            outlier_ratio = total_outliers / len(data)
            
            # Analisar duplicatas
            duplicate_count = data.duplicated().sum()
            duplicate_ratio = duplicate_count / len(data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            details = {
                "missing_ratio": missing_ratio,
                "outlier_ratio": outlier_ratio,
                "duplicate_ratio": duplicate_ratio,
                "outlier_counts": outlier_counts,
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "missing_threshold": self.threshold
            }
            
            # Avaliar qualidade
            issues = []
            if missing_ratio > self.threshold:
                issues.append(f"High missing data ratio: {missing_ratio:.1%}")
            if outlier_ratio > 0.1:
                issues.append(f"High outlier ratio: {outlier_ratio:.1%}")
            if duplicate_ratio > 0.01:
                issues.append(f"High duplicate ratio: {duplicate_ratio:.1%}")
            
            if not issues:
                status = ValidationStatus.PASSED
                message = "Data quality acceptable"
            elif len(issues) == 1:
                status = ValidationStatus.WARNING
                message = f"Data quality issues: {issues[0]}"
            else:
                status = ValidationStatus.FAILED
                message = f"Multiple data quality issues: {', '.join(issues)}"
            
            quality_score = 1 - (missing_ratio + outlier_ratio + duplicate_ratio) / 3
            
            return self.create_result(
                status, quality_score, message, details, execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return self.create_result(
                ValidationStatus.FAILED,
                0.0,
                f"Data quality validation failed: {str(e)}",
                {"error": str(e)},
                execution_time
            )


class ModelValidator:
    """Sistema de validação de modelos."""
    
    def __init__(self, registry):
        self.registry = registry
        self.logger = get_logger("model_validator")
        self.struct_logger = structlog.get_logger("model_validator")
        
        # Validadores padrão
        self.validators = {
            "performance": PerformanceValidator(),
            "data_drift": DataDriftValidator(),
            "stability": StabilityValidator(),
            "bias": BiasValidator(),
            "data_quality": DataQualityValidator()
        }
        
        # Configurações
        self.validation_configs = {
            "performance": {"required": True, "blocking": True},
            "data_drift": {"required": True, "blocking": False},
            "stability": {"required": True, "blocking": False},
            "bias": {"required": True, "blocking": False},
            "data_quality": {"required": True, "blocking": False}
        }
    
    def add_validator(self, name: str, validator: BaseValidator, 
                     required: bool = True, blocking: bool = False):
        """Adiciona validador customizado."""
        self.validators[name] = validator
        self.validation_configs[name] = {
            "required": required,
            "blocking": blocking
        }
    
    async def validate_model(self, model_id: str, version: str, 
                           validation_data: pd.DataFrame,
                           target: pd.Series = None,
                           reference_data: pd.DataFrame = None,
                           validators: List[str] = None) -> ValidationReport:
        """Executa validação completa do modelo."""
        start_time = datetime.now()
        validation_id = f"validation_{model_id}_{version}_{int(start_time.timestamp())}"
        
        try:
            # Carregar modelo
            model = self.registry.load_model(model_id, version)
            if not model:
                raise ValueError(f"Model {model_id}:{version} not found")
            
            # Selecionar validadores
            if validators is None:
                validators = list(self.validators.keys())
            
            # Executar validações
            results = []
            warnings = []
            errors = []
            
            for validator_name in validators:
                if validator_name not in self.validators:
                    errors.append(f"Unknown validator: {validator_name}")
                    continue
                
                try:
                    validator = self.validators[validator_name]
                    
                    # Executar validação
                    result = await validator.validate(
                        model, validation_data, target,
                        reference_data=reference_data
                    )
                    
                    results.append(result)
                    
                    # Analisar resultado
                    if result.status == ValidationStatus.FAILED:
                        errors.append(f"{validator_name}: {result.message}")
                    elif result.status == ValidationStatus.WARNING:
                        warnings.append(f"{validator_name}: {result.message}")
                    
                except Exception as e:
                    error_msg = f"Validator {validator_name} failed: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Determinar status geral
            overall_status = self._determine_overall_status(results)
            
            # Gerar recomendações
            recommendations = self._generate_recommendations(results)
            
            # Calcular métricas
            metrics = self._calculate_metrics(results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Criar relatório
            report = ValidationReport(
                model_id=model_id,
                version=version,
                validation_id=validation_id,
                overall_status=overall_status,
                results=results,
                metrics=metrics,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations,
                created_at=start_time,
                total_execution_time=execution_time
            )
            
            self.struct_logger.info(
                "Model validation completed",
                model_id=model_id,
                version=version,
                validation_id=validation_id,
                overall_status=overall_status.value,
                validators_count=len(results),
                execution_time=execution_time
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            raise
    
    def _determine_overall_status(self, results: List[ValidationResult]) -> ValidationStatus:
        """Determina status geral baseado nos resultados."""
        if not results:
            return ValidationStatus.FAILED
        
        # Verificar validadores críticos
        critical_failures = [
            r for r in results 
            if r.status == ValidationStatus.FAILED and 
            r.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]
        ]
        
        if critical_failures:
            return ValidationStatus.FAILED
        
        # Verificar falhas em validadores blocking
        blocking_failures = [
            r for r in results 
            if r.status == ValidationStatus.FAILED and 
            self.validation_configs.get(r.validator_name, {}).get("blocking", False)
        ]
        
        if blocking_failures:
            return ValidationStatus.FAILED
        
        # Verificar warnings
        warnings = [r for r in results if r.status == ValidationStatus.WARNING]
        failures = [r for r in results if r.status == ValidationStatus.FAILED]
        
        if failures:
            return ValidationStatus.WARNING
        elif warnings:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.PASSED
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Gera recomendações baseadas nos resultados."""
        recommendations = []
        
        for result in results:
            if result.status == ValidationStatus.FAILED:
                if result.validator_name == "performance":
                    recommendations.append("Consider retraining with more data or feature engineering")
                elif result.validator_name == "data_drift":
                    recommendations.append("Update model with recent data or implement adaptive learning")
                elif result.validator_name == "stability":
                    recommendations.append("Review model complexity and regularization parameters")
                elif result.validator_name == "bias":
                    recommendations.append("Investigate feature selection and data preprocessing")
                elif result.validator_name == "data_quality":
                    recommendations.append("Improve data cleaning and validation processes")
            
            elif result.status == ValidationStatus.WARNING:
                recommendations.append(f"Monitor {result.validator_name} metrics closely in production")
        
        return recommendations
    
    def _calculate_metrics(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Calcula métricas agregadas."""
        if not results:
            return {}
        
        passed_count = len([r for r in results if r.status == ValidationStatus.PASSED])
        warning_count = len([r for r in results if r.status == ValidationStatus.WARNING])
        failed_count = len([r for r in results if r.status == ValidationStatus.FAILED])
        
        return {
            "pass_rate": passed_count / len(results),
            "warning_rate": warning_count / len(results),
            "failure_rate": failed_count / len(results),
            "average_score": np.mean([r.score for r in results]),
            "total_validators": len(results)
        }
    
    def get_validation_summary(self, model_id: str, version: str = None) -> Dict[str, Any]:
        """Obtém resumo das validações do modelo."""
        try:
            # Implementar busca por validações anteriores
            # Por enquanto, retorna estrutura básica
            return {
                "model_id": model_id,
                "version": version,
                "validation_history": [],
                "latest_validation": None,
                "validation_trends": {}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting validation summary: {e}")
            return {}


def create_model_validator(registry) -> ModelValidator:
    """Cria instância do model validator."""
    return ModelValidator(registry)