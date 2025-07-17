"""
Fase 4: Validação NBR 14653
Implementa bateria completa de testes exigidos pela norma técnica brasileira.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass
import logging


@dataclass
class NBRTestResult:
    """Resultado de um teste NBR 14653."""
    test_name: str
    passed: bool
    value: float
    threshold: float
    description: str
    recommendation: str


@dataclass
class NBRValidationResult:
    """Resultado completo da validação NBR 14653."""
    overall_grade: str  # Superior, Normal, Inferior
    individual_tests: List[NBRTestResult]
    summary: Dict[str, Any]
    compliance_score: float


class NBR14653Validator:
    """Validador conforme NBR 14653 para avaliação imobiliária."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Thresholds NBR 14653
        self.thresholds = {
            'r2_superior': 0.90,
            'r2_normal': 0.80,
            'r2_inferior': 0.70,
            'f_test_p_value': 0.05,
            't_test_p_value': 0.05,
            'durbin_watson_lower': 1.5,
            'durbin_watson_upper': 2.5,
            'max_vif': 10.0,
            'min_sample_size': 30,
            'max_outliers_percent': 5.0
        }
    
    def test_coefficient_determination(self, model, X_test: pd.DataFrame, 
                                     y_test: pd.Series) -> NBRTestResult:
        """
        Teste do coeficiente de determinação (R²).
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Resultado do teste R²
        """
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Determinar grau de precisão
        if r2 >= self.thresholds['r2_superior']:
            grade = "Superior"
            passed = True
        elif r2 >= self.thresholds['r2_normal']:
            grade = "Normal"
            passed = True
        elif r2 >= self.thresholds['r2_inferior']:
            grade = "Inferior"
            passed = True
        else:
            grade = "Inadequado"
            passed = False
        
        return NBRTestResult(
            test_name="Coeficiente de Determinação (R²)",
            passed=passed,
            value=r2,
            threshold=self.thresholds['r2_inferior'],
            description=f"R² = {r2:.4f} - Grau de precisão: {grade}",
            recommendation="R² deve ser >= 0.70 para aprovação mínima"
        )
    
    def test_f_significance(self, model, X_train: pd.DataFrame, 
                          y_train: pd.Series) -> NBRTestResult:
        """
        Teste F de significância do modelo.
        
        Args:
            model: Modelo treinado
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Resultado do teste F
        """
        from sklearn.metrics import mean_squared_error
        
        y_pred = model.predict(X_train)
        
        # Calcular F-statistic
        n = len(y_train)
        k = len(model.coef_)
        
        mse_model = mean_squared_error(y_train, y_pred)
        mse_total = np.var(y_train)
        
        f_stat = ((mse_total - mse_model) / k) / (mse_model / (n - k - 1))
        
        # Calcular p-value
        p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
        
        passed = p_value < self.thresholds['f_test_p_value']
        
        return NBRTestResult(
            test_name="Teste F de Significância",
            passed=passed,
            value=p_value,
            threshold=self.thresholds['f_test_p_value'],
            description=f"F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}",
            recommendation="p-value deve ser < 0.05 para significância"
        )
    
    def test_coefficients_significance(self, model, X_train: pd.DataFrame, 
                                     y_train: pd.Series) -> NBRTestResult:
        """
        Teste t de significância dos coeficientes.
        
        Args:
            model: Modelo treinado
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Resultado do teste t
        """
        from sklearn.metrics import mean_squared_error
        
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        
        # Calcular erro padrão dos coeficientes
        X_array = X_train.values
        xtx_inv = np.linalg.inv(X_array.T @ X_array)
        se_coef = np.sqrt(mse * np.diag(xtx_inv))
        
        # Calcular t-statistics
        t_stats = model.coef_ / se_coef
        
        # Calcular p-values
        df = len(y_train) - len(model.coef_) - 1
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
        
        # Contar coeficientes significativos
        significant_coefs = np.sum(p_values < self.thresholds['t_test_p_value'])
        total_coefs = len(model.coef_)
        
        passed = significant_coefs / total_coefs >= 0.5  # Pelo menos 50% significativos
        
        return NBRTestResult(
            test_name="Teste t dos Coeficientes",
            passed=passed,
            value=significant_coefs / total_coefs,
            threshold=0.5,
            description=f"{significant_coefs}/{total_coefs} coeficientes significativos",
            recommendation="Pelo menos 50% dos coeficientes devem ser significativos"
        )
    
    def test_residuals_normality(self, model, X_test: pd.DataFrame, 
                                y_test: pd.Series) -> NBRTestResult:
        """
        Teste de normalidade dos resíduos (Shapiro-Wilk).
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Resultado do teste de normalidade
        """
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Teste Shapiro-Wilk
        if len(residuals) > 5000:
            # Para amostras grandes, usar Kolmogorov-Smirnov
            stat, p_value = stats.kstest(residuals, 'norm')
            test_name = "Normalidade dos Resíduos (Kolmogorov-Smirnov)"
        else:
            stat, p_value = stats.shapiro(residuals)
            test_name = "Normalidade dos Resíduos (Shapiro-Wilk)"
        
        passed = p_value > 0.05
        
        return NBRTestResult(
            test_name=test_name,
            passed=passed,
            value=p_value,
            threshold=0.05,
            description=f"Estatística = {stat:.4f}, p-value = {p_value:.4f}",
            recommendation="p-value > 0.05 indica normalidade dos resíduos"
        )
    
    def test_autocorrelation(self, model, X_test: pd.DataFrame, 
                           y_test: pd.Series) -> NBRTestResult:
        """
        Teste de autocorrelação dos resíduos (Durbin-Watson).
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Resultado do teste Durbin-Watson
        """
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Calcular estatística Durbin-Watson
        diff = np.diff(residuals)
        dw_stat = np.sum(diff**2) / np.sum(residuals**2)
        
        passed = (self.thresholds['durbin_watson_lower'] <= dw_stat <= 
                 self.thresholds['durbin_watson_upper'])
        
        return NBRTestResult(
            test_name="Teste de Autocorrelação (Durbin-Watson)",
            passed=passed,
            value=dw_stat,
            threshold=2.0,
            description=f"DW = {dw_stat:.4f}",
            recommendation="DW entre 1.5 e 2.5 indica ausência de autocorrelação"
        )
    
    def test_multicollinearity(self, X: pd.DataFrame) -> NBRTestResult:
        """
        Teste de multicolinearidade (VIF - Variance Inflation Factor).
        
        Args:
            X: Features
            
        Returns:
            Resultado do teste VIF
        """
        from sklearn.linear_model import LinearRegression
        
        vif_values = []
        
        for i in range(len(X.columns)):
            # Regressão de cada variável contra as demais
            X_temp = X.drop(columns=[X.columns[i]])
            y_temp = X.iloc[:, i]
            
            model = LinearRegression()
            model.fit(X_temp, y_temp)
            y_pred = model.predict(X_temp)
            
            # Calcular R²
            r2 = r2_score(y_temp, y_pred)
            
            # Calcular VIF
            vif = 1 / (1 - r2) if r2 < 0.999 else float('inf')
            vif_values.append(vif)
        
        max_vif = max(vif_values)
        passed = max_vif < self.thresholds['max_vif']
        
        return NBRTestResult(
            test_name="Teste de Multicolinearidade (VIF)",
            passed=passed,
            value=max_vif,
            threshold=self.thresholds['max_vif'],
            description=f"VIF máximo = {max_vif:.2f}",
            recommendation="VIF < 10 indica ausência de multicolinearidade severa"
        )
    
    def test_sample_size(self, df: pd.DataFrame) -> NBRTestResult:
        """
        Teste de tamanho da amostra.
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Resultado do teste de tamanho
        """
        sample_size = len(df)
        passed = sample_size >= self.thresholds['min_sample_size']
        
        return NBRTestResult(
            test_name="Tamanho da Amostra",
            passed=passed,
            value=sample_size,
            threshold=self.thresholds['min_sample_size'],
            description=f"Tamanho da amostra = {sample_size}",
            recommendation="Mínimo de 30 observações para análise estatística"
        )
    
    def validate_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series) -> NBRValidationResult:
        """
        Executa bateria completa de testes NBR 14653.
        
        Args:
            model: Modelo treinado
            X_train: Features de treino
            X_test: Features de teste
            y_train: Target de treino
            y_test: Target de teste
            
        Returns:
            Resultado completo da validação
        """
        tests = []
        
        # Executar todos os testes
        tests.append(self.test_coefficient_determination(model, X_test, y_test))
        tests.append(self.test_f_significance(model, X_train, y_train))
        tests.append(self.test_coefficients_significance(model, X_train, y_train))
        tests.append(self.test_residuals_normality(model, X_test, y_test))
        tests.append(self.test_autocorrelation(model, X_test, y_test))
        tests.append(self.test_multicollinearity(X_train))
        
        # Combinar datasets para teste de tamanho
        df_combined = pd.concat([X_train, X_test])
        tests.append(self.test_sample_size(df_combined))
        
        # Calcular score de conformidade
        passed_tests = sum(1 for test in tests if test.passed)
        compliance_score = passed_tests / len(tests)
        
        # Determinar grau geral
        r2_test = tests[0]  # Primeiro teste é sempre R²
        if r2_test.value >= self.thresholds['r2_superior'] and compliance_score >= 0.8:
            overall_grade = "Superior"
        elif r2_test.value >= self.thresholds['r2_normal'] and compliance_score >= 0.7:
            overall_grade = "Normal"
        elif r2_test.value >= self.thresholds['r2_inferior'] and compliance_score >= 0.6:
            overall_grade = "Inferior"
        else:
            overall_grade = "Inadequado"
        
        # Resumo
        summary = {
            'total_tests': len(tests),
            'passed_tests': passed_tests,
            'compliance_score': compliance_score,
            'r2_value': r2_test.value,
            'overall_grade': overall_grade
        }
        
        return NBRValidationResult(
            overall_grade=overall_grade,
            individual_tests=tests,
            summary=summary,
            compliance_score=compliance_score
        )