"""
Phase 4: NBR 14653 Validation
Implements complete battery of tests required by the Brazilian technical standard.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning


@dataclass
class NBRTestResult:
    """Resultado de um teste NBR 14653 com tratamento de casos especiais."""
    test_name: str
    passed: bool
    value: Union[float, str]  # Permitir 'N/A' para testes não aplicáveis
    threshold: float
    description: str
    recommendation: str
    applicable: bool = True  # Se o teste é aplicável aos dados
    warning: Optional[str] = None  # Avisos sobre limitações do teste


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
        Teste F de significância do modelo com tratamento de casos especiais.
        
        Args:
            model: Modelo treinado
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Resultado do teste F
        """
        from sklearn.metrics import mean_squared_error
        
        try:
            y_pred = model.predict(X_train)
            
            # Verificar se o modelo tem coeficientes (necessário para teste F)
            if not hasattr(model, 'coef_'):
                return NBRTestResult(
                    test_name="Teste F de Significância",
                    passed=False,
                    value="N/A",
                    threshold=self.thresholds['f_test_p_value'],
                    description="Modelo não suporta teste F (sem coeficientes lineares)",
                    recommendation="Use modelo linear para teste F",
                    applicable=False
                )
            
            # Calcular F-statistic
            n = len(y_train)
            k = len(model.coef_)
            
            # Verificar se há graus de liberdade suficientes
            if n <= k + 1:
                return NBRTestResult(
                    test_name="Teste F de Significância",
                    passed=False,
                    value="N/A",
                    threshold=self.thresholds['f_test_p_value'],
                    description=f"Graus de liberdade insuficientes: n={n}, k={k}",
                    recommendation="Aumente o tamanho da amostra ou reduza o número de features",
                    applicable=False
                )
            
            mse_model = mean_squared_error(y_train, y_pred)
            mse_total = np.var(y_train)
            
            # Verificar divisão por zero
            if mse_model == 0 or mse_total == 0:
                return NBRTestResult(
                    test_name="Teste F de Significância",
                    passed=False,
                    value="N/A",
                    threshold=self.thresholds['f_test_p_value'],
                    description="MSE zero detectado - modelo pode ter sobreajuste perfeito",
                    recommendation="Verificar overfitting ou variância zero no target",
                    applicable=False
                )
            
            f_stat = ((mse_total - mse_model) / k) / (mse_model / (n - k - 1))
            
            # Verificar se F-statistic é válida
            if np.isnan(f_stat) or np.isinf(f_stat) or f_stat < 0:
                return NBRTestResult(
                    test_name="Teste F de Significância",
                    passed=False,
                    value="N/A",
                    threshold=self.thresholds['f_test_p_value'],
                    description=f"F-statistic inválida: {f_stat}",
                    recommendation="Verificar estabilidade numérica dos dados",
                    applicable=False
                )
            
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
            
        except Exception as e:
            self.logger.error(f"Erro no teste F: {e}")
            return NBRTestResult(
                test_name="Teste F de Significância",
                passed=False,
                value="N/A",
                threshold=self.thresholds['f_test_p_value'],
                description=f"Erro na execução: {str(e)}",
                recommendation="Verificar integridade dos dados e modelo",
                applicable=False
            )
    
    def test_coefficients_significance(self, model, X_train: pd.DataFrame, 
                                     y_train: pd.Series) -> NBRTestResult:
        """
        Teste t de significância dos coeficientes com tratamento robusto.
        
        Args:
            model: Modelo treinado
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Resultado do teste t
        """
        from sklearn.metrics import mean_squared_error
        
        try:
            # Verificar se o modelo tem coeficientes
            if not hasattr(model, 'coef_'):
                return NBRTestResult(
                    test_name="Teste t dos Coeficientes",
                    passed=False,
                    value="N/A",
                    threshold=0.5,
                    description="Modelo não suporta teste t (sem coeficientes lineares)",
                    recommendation="Use modelo linear para teste de coeficientes",
                    applicable=False
                )
            
            y_pred = model.predict(X_train)
            mse = mean_squared_error(y_train, y_pred)
            
            # Verificar se MSE é válido
            if mse == 0 or np.isnan(mse):
                return NBRTestResult(
                    test_name="Teste t dos Coeficientes",
                    passed=False,
                    value="N/A",
                    threshold=0.5,
                    description="MSE inválido para cálculo do erro padrão",
                    recommendation="Verificar overfitting ou problemas nos dados",
                    applicable=False
                )
            
            # Calcular erro padrão dos coeficientes com tratamento de singularidade
            X_array = X_train.values
            
            try:
                xtx = X_array.T @ X_array
                
                # Verificar condição da matriz
                condition_number = np.linalg.cond(xtx)
                if condition_number > 1e12:
                    return NBRTestResult(
                        test_name="Teste t dos Coeficientes",
                        passed=False,
                        value="N/A",
                        threshold=0.5,
                        description=f"Matriz mal condicionada (cond={condition_number:.2e})",
                        recommendation="Verificar multicolinearidade entre features",
                        applicable=False,
                        warning="Matriz próxima da singularidade"
                    )
                
                xtx_inv = np.linalg.inv(xtx)
                se_coef = np.sqrt(mse * np.diag(xtx_inv))
                
            except np.linalg.LinAlgError:
                return NBRTestResult(
                    test_name="Teste t dos Coeficientes",
                    passed=False,
                    value="N/A",
                    threshold=0.5,
                    description="Matriz singular - não é possível calcular erro padrão",
                    recommendation="Remover features colineares ou usar regularização",
                    applicable=False
                )
            
            # Verificar se erro padrão é válido
            if np.any(se_coef == 0) or np.any(np.isnan(se_coef)):
                return NBRTestResult(
                    test_name="Teste t dos Coeficientes",
                    passed=False,
                    value="N/A",
                    threshold=0.5,
                    description="Erro padrão inválido detectado",
                    recommendation="Verificar estabilidade numérica do modelo",
                    applicable=False
                )
            
            # Calcular t-statistics
            t_stats = model.coef_ / se_coef
            
            # Verificar graus de liberdade
            df = len(y_train) - len(model.coef_) - 1
            if df <= 0:
                return NBRTestResult(
                    test_name="Teste t dos Coeficientes",
                    passed=False,
                    value="N/A",
                    threshold=0.5,
                    description=f"Graus de liberdade insuficientes: df={df}",
                    recommendation="Aumente o tamanho da amostra ou reduza features",
                    applicable=False
                )
            
            # Calcular p-values
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
            
            # Contar coeficientes significativos
            significant_coefs = np.sum(p_values < self.thresholds['t_test_p_value'])
            total_coefs = len(model.coef_)
            
            if total_coefs == 0:
                return NBRTestResult(
                    test_name="Teste t dos Coeficientes",
                    passed=False,
                    value="N/A",
                    threshold=0.5,
                    description="Nenhum coeficiente encontrado",
                    recommendation="Verificar configuração do modelo",
                    applicable=False
                )
            
            significance_ratio = significant_coefs / total_coefs
            passed = significance_ratio >= 0.5  # Pelo menos 50% significativos
            
            warning = None
            if significance_ratio < 0.3:
                warning = "Muito poucos coeficientes significativos - modelo pode estar mal especificado"
            
            return NBRTestResult(
                test_name="Teste t dos Coeficientes",
                passed=passed,
                value=significance_ratio,
                threshold=0.5,
                description=f"{significant_coefs}/{total_coefs} coeficientes significativos ({significance_ratio:.1%})",
                recommendation="Pelo menos 50% dos coeficientes devem ser significativos",
                warning=warning
            )
            
        except Exception as e:
            self.logger.error(f"Erro no teste t: {e}")
            return NBRTestResult(
                test_name="Teste t dos Coeficientes",
                passed=False,
                value="N/A",
                threshold=0.5,
                description=f"Erro na execução: {str(e)}",
                recommendation="Verificar integridade dos dados e modelo",
                applicable=False
            )
    
    def test_residuals_normality(self, model, X_test: pd.DataFrame, 
                                y_test: pd.Series) -> NBRTestResult:
        """
        Teste de normalidade dos resíduos com tratamento de casos especiais.
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Resultado do teste de normalidade
        """
        try:
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            
            # Verificar se há resíduos suficientes
            if len(residuals) < 3:
                return NBRTestResult(
                    test_name="Normalidade dos Resíduos",
                    passed=False,
                    value="N/A",
                    threshold=0.05,
                    description=f"Amostra muito pequena: {len(residuals)} resíduos",
                    recommendation="Mínimo de 3 observações necessário para teste de normalidade",
                    applicable=False
                )
            
            # Verificar variância zero
            if np.var(residuals) == 0:
                return NBRTestResult(
                    test_name="Normalidade dos Resíduos",
                    passed=False,
                    value="N/A",
                    threshold=0.05,
                    description="Resíduos com variância zero (overfitting perfeito)",
                    recommendation="Verificar overfitting do modelo",
                    applicable=False,
                    warning="Variância zero pode indicar overfitting"
                )
            
            # Remover valores infinitos ou NaN
            residuals_clean = residuals[np.isfinite(residuals)]
            
            if len(residuals_clean) < len(residuals):
                warning = f"Removidos {len(residuals) - len(residuals_clean)} valores não finitos"
            else:
                warning = None
            
            if len(residuals_clean) < 3:
                return NBRTestResult(
                    test_name="Normalidade dos Resíduos",
                    passed=False,
                    value="N/A",
                    threshold=0.05,
                    description="Insuficientes resíduos válidos após limpeza",
                    recommendation="Verificar predições do modelo",
                    applicable=False,
                    warning=warning
                )
            
            # Escolher teste baseado no tamanho da amostra
            if len(residuals_clean) > 5000:
                # Para amostras grandes, usar Kolmogorov-Smirnov
                # Normalizar resíduos
                residuals_normalized = (residuals_clean - np.mean(residuals_clean)) / np.std(residuals_clean)
                stat, p_value = stats.kstest(residuals_normalized, 'norm')
                test_name = "Normalidade dos Resíduos (Kolmogorov-Smirnov)"
                
            elif len(residuals_clean) <= 50:
                # Para amostras pequenas, usar Shapiro-Wilk
                try:
                    stat, p_value = stats.shapiro(residuals_clean)
                    test_name = "Normalidade dos Resíduos (Shapiro-Wilk)"
                except Exception as e:
                    return NBRTestResult(
                        test_name="Normalidade dos Resíduos",
                        passed=False,
                        value="N/A",
                        threshold=0.05,
                        description=f"Erro no teste Shapiro-Wilk: {str(e)}",
                        recommendation="Verificar distribuição dos resíduos",
                        applicable=False
                    )
            else:
                # Para amostras médias, usar Anderson-Darling
                try:
                    result = stats.anderson(residuals_clean, dist='norm')
                    stat = result.statistic
                    # Aproximar p-value baseado nos valores críticos
                    if stat < result.critical_values[2]:  # 5% significance level
                        p_value = 0.1  # Approximate
                    else:
                        p_value = 0.01  # Approximate
                    test_name = "Normalidade dos Resíduos (Anderson-Darling)"
                except Exception:
                    # Fallback para Shapiro-Wilk
                    stat, p_value = stats.shapiro(residuals_clean)
                    test_name = "Normalidade dos Resíduos (Shapiro-Wilk)"
            
            # Verificar se resultados são válidos
            if np.isnan(stat) or np.isnan(p_value):
                return NBRTestResult(
                    test_name="Normalidade dos Resíduos",
                    passed=False,
                    value="N/A",
                    threshold=0.05,
                    description="Estatística de teste inválida (NaN)",
                    recommendation="Verificar distribuição dos resíduos",
                    applicable=False
                )
            
            passed = p_value > 0.05
            
            # Adicionar warning se p-value for muito baixo
            if p_value < 0.001 and warning is None:
                warning = "Forte evidência contra normalidade - considere transformações"
            
            return NBRTestResult(
                test_name=test_name,
                passed=passed,
                value=p_value,
                threshold=0.05,
                description=f"Estatística = {stat:.4f}, p-value = {p_value:.4f}",
                recommendation="p-value > 0.05 indica normalidade dos resíduos",
                warning=warning
            )
            
        except Exception as e:
            self.logger.error(f"Erro no teste de normalidade: {e}")
            return NBRTestResult(
                test_name="Normalidade dos Resíduos",
                passed=False,
                value="N/A",
                threshold=0.05,
                description=f"Erro na execução: {str(e)}",
                recommendation="Verificar integridade dos dados",
                applicable=False
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
    
    def test_extrapolation_range(self, model, X_train: pd.DataFrame, 
                                X_predict: pd.DataFrame) -> NBRTestResult:
        """
        Teste de extrapolação - verifica se predições estão dentro do range de treino.
        
        Args:
            model: Modelo treinado
            X_train: Features de treino
            X_predict: Features para predição
            
        Returns:
            Resultado do teste de extrapolação
        """
        extrapolation_violations = 0
        total_features = len(X_train.columns)
        
        for col in X_train.columns:
            if col in X_predict.columns:
                train_min = X_train[col].min()
                train_max = X_train[col].max()
                
                # Verificar valores fora do range
                violations = ((X_predict[col] < train_min) | 
                             (X_predict[col] > train_max)).sum()
                extrapolation_violations += violations
        
        # Calcular porcentagem de extrapolação
        total_predictions = len(X_predict) * total_features
        extrapolation_rate = extrapolation_violations / total_predictions if total_predictions > 0 else 0
        
        # Aprovar se menos de 10% das predições são extrapolações
        passed = extrapolation_rate < 0.1
        
        return NBRTestResult(
            test_name="Teste de Extrapolação",
            passed=passed,
            value=extrapolation_rate,
            threshold=0.1,
            description=f"Taxa de extrapolação = {extrapolation_rate:.2%}",
            recommendation="Menos de 10% das predições devem ser extrapolações"
        )
    
    def test_prediction_intervals(self, model, X_test: pd.DataFrame, 
                                 y_test: pd.Series, confidence_level: float = 0.95) -> NBRTestResult:
        """
        Teste de intervalos de predição.
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            confidence_level: Nível de confiança
            
        Returns:
            Resultado do teste de intervalos
        """
        from sklearn.metrics import mean_squared_error
        
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Calcular erro padrão da predição
        mse = mean_squared_error(y_test, y_pred)
        se_pred = np.sqrt(mse)
        
        # Calcular intervalos de confiança
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bound = y_pred - z_score * se_pred
        upper_bound = y_pred + z_score * se_pred
        
        # Verificar quantos valores estão dentro dos intervalos
        within_intervals = ((y_test >= lower_bound) & (y_test <= upper_bound)).sum()
        coverage_rate = within_intervals / len(y_test)
        
        # Aprovar se taxa de cobertura está próxima do nível de confiança
        passed = abs(coverage_rate - confidence_level) < 0.05
        
        return NBRTestResult(
            test_name=f"Intervalos de Predição ({confidence_level:.0%})",
            passed=passed,
            value=coverage_rate,
            threshold=confidence_level,
            description=f"Taxa de cobertura = {coverage_rate:.2%}",
            recommendation=f"Taxa de cobertura deve estar próxima de {confidence_level:.0%}"
        )
    
    def test_homoscedasticity(self, model, X_test: pd.DataFrame, 
                             y_test: pd.Series) -> NBRTestResult:
        """
        Teste de homoscedasticidade dos resíduos (Breusch-Pagan).
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Resultado do teste de homoscedasticidade
        """
        from sklearn.linear_model import LinearRegression
        
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        squared_residuals = residuals ** 2
        
        # Regressão dos resíduos quadráticos contra valores preditos
        reg = LinearRegression()
        reg.fit(y_pred.reshape(-1, 1), squared_residuals)
        r2_residuals = reg.score(y_pred.reshape(-1, 1), squared_residuals)
        
        # Teste Breusch-Pagan
        n = len(residuals)
        lm_statistic = n * r2_residuals
        p_value = 1 - stats.chi2.cdf(lm_statistic, df=1)
        
        # Homoscedasticidade se p-value > 0.05
        passed = p_value > 0.05
        
        return NBRTestResult(
            test_name="Teste de Homoscedasticidade (Breusch-Pagan)",
            passed=passed,
            value=p_value,
            threshold=0.05,
            description=f"LM = {lm_statistic:.4f}, p-value = {p_value:.4f}",
            recommendation="p-value > 0.05 indica homoscedasticidade"
        )
    
    def validate_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series, 
                      X_predict: Optional[pd.DataFrame] = None) -> NBRValidationResult:
        """
        Executa bateria completa de testes NBR 14653 com tratamento robusto de casos especiais.
        
        Args:
            model: Modelo treinado
            X_train: Features de treino
            X_test: Features de teste
            y_train: Target de treino
            y_test: Target de teste
            X_predict: Features para predição (opcional, para teste de extrapolação)
            
        Returns:
            Resultado completo da validação
        """
        tests = []
        
        # Executar todos os testes com tratamento de erro
        test_functions = [
            ("R²", lambda: self.test_coefficient_determination(model, X_test, y_test)),
            ("F", lambda: self.test_f_significance(model, X_train, y_train)),
            ("Coeficientes", lambda: self.test_coefficients_significance(model, X_train, y_train)),
            ("Normalidade", lambda: self.test_residuals_normality(model, X_test, y_test)),
            ("Autocorrelação", lambda: self.test_autocorrelation(model, X_test, y_test)),
            ("Multicolinearidade", lambda: self.test_multicollinearity(X_train)),
            ("Homoscedasticidade", lambda: self.test_homoscedasticity(model, X_test, y_test)),
            ("Intervalos", lambda: self.test_prediction_intervals(model, X_test, y_test))
        ]
        
        for test_name, test_func in test_functions:
            try:
                result = test_func()
                tests.append(result)
            except Exception as e:
                self.logger.error(f"Erro no teste {test_name}: {e}")
                # Adicionar resultado de falha
                tests.append(NBRTestResult(
                    test_name=f"Teste {test_name}",
                    passed=False,
                    value="N/A",
                    threshold=0.0,
                    description=f"Erro na execução: {str(e)}",
                    recommendation="Verificar integridade dos dados e modelo",
                    applicable=False
                ))
        
        # Teste de tamanho da amostra
        try:
            df_combined = pd.concat([X_train, X_test])
            tests.append(self.test_sample_size(df_combined))
        except Exception as e:
            self.logger.error(f"Erro no teste de tamanho da amostra: {e}")
            tests.append(NBRTestResult(
                test_name="Tamanho da Amostra",
                passed=False,
                value="N/A",
                threshold=self.thresholds['min_sample_size'],
                description=f"Erro na verificação: {str(e)}",
                recommendation="Verificar integridade dos datasets",
                applicable=False
            ))
        
        # Teste de extrapolação se dados de predição fornecidos
        if X_predict is not None and len(X_predict) > 0:
            try:
                tests.append(self.test_extrapolation_range(model, X_train, X_predict))
            except Exception as e:
                self.logger.error(f"Erro no teste de extrapolação: {e}")
                tests.append(NBRTestResult(
                    test_name="Teste de Extrapolação",
                    passed=False,
                    value="N/A",
                    threshold=0.1,
                    description=f"Erro na verificação: {str(e)}",
                    recommendation="Verificar compatibilidade entre datasets",
                    applicable=False
                ))
        
        # Calcular score de conformidade considerando apenas testes aplicáveis
        applicable_tests = [t for t in tests if t.applicable]
        passed_applicable = sum(1 for test in applicable_tests if test.passed)
        
        if len(applicable_tests) == 0:
            compliance_score = 0.0
            overall_grade = "Inadequado"
            self.logger.error("Nenhum teste NBR foi aplicável aos dados")
        else:
            compliance_score = passed_applicable / len(applicable_tests)
            
            # Determinar grau geral baseado no R² e conformidade
            r2_test = next((t for t in tests if "R²" in t.test_name), None)
            
            if r2_test is None or not r2_test.applicable or r2_test.value == "N/A":
                overall_grade = "Inadequado"
                self.logger.warning("Teste R² não disponível - classificação inadequada")
            else:
                r2_value = r2_test.value
                if r2_value >= self.thresholds['r2_superior'] and compliance_score >= 0.8:
                    overall_grade = "Superior"
                elif r2_value >= self.thresholds['r2_normal'] and compliance_score >= 0.7:
                    overall_grade = "Normal"
                elif r2_value >= self.thresholds['r2_inferior'] and compliance_score >= 0.6:
                    overall_grade = "Inferior"
                else:
                    overall_grade = "Inadequado"
        
        # Resumo expandido com informações sobre aplicabilidade
        summary = {
            'total_tests': len(tests),
            'applicable_tests': len(applicable_tests),
            'passed_tests': sum(1 for test in tests if test.passed),
            'passed_applicable': passed_applicable,
            'compliance_score': compliance_score,
            'r2_value': r2_test.value if r2_test and r2_test.applicable else "N/A",
            'overall_grade': overall_grade,
            'extrapolation_tested': X_predict is not None,
            'critical_failures': [test.test_name for test in tests 
                                if not test.passed and test.applicable and 'R²' in test.test_name],
            'non_applicable_tests': [test.test_name for test in tests if not test.applicable],
            'tests_with_warnings': [test.test_name for test in tests if test.warning],
            'advanced_tests_passed': sum(1 for test in tests 
                                       if test.passed and test.applicable and test.test_name in [
                                           'Teste de Homoscedasticidade (Breusch-Pagan)',
                                           'Intervalos de Predição (95%)',
                                           'Teste de Extrapolação'
                                       ])
        }
        
        self.logger.info(f"Validação NBR 14653 concluída: {overall_grade} ({compliance_score:.2%} conformidade, {len(applicable_tests)}/{len(tests)} testes aplicáveis)")
        
        return NBRValidationResult(
            overall_grade=overall_grade,
            individual_tests=tests,
            summary=summary,
            compliance_score=compliance_score
        )
    
    def generate_validation_report(self, validation_result: NBRValidationResult) -> str:
        """
        Gera relatório textual da validação NBR 14653.
        
        Args:
            validation_result: Resultado da validação
            
        Returns:
            Relatório em formato texto
        """
        report = []
        report.append("=" * 60)
        report.append("RELATÓRIO DE VALIDAÇÃO NBR 14653")
        report.append("=" * 60)
        report.append("")
        
        # Resumo geral
        report.append(f"GRAU DE PRECISÃO: {validation_result.overall_grade}")
        report.append(f"SCORE DE CONFORMIDADE: {validation_result.compliance_score:.1%}")
        report.append(f"TESTES APROVADOS: {validation_result.summary['passed_tests']}/{validation_result.summary['total_tests']}")
        report.append("")
        
        # Resultados por teste
        report.append("RESULTADOS DETALHADOS:")
        report.append("-" * 40)
        
        for test in validation_result.individual_tests:
            status = "✓ PASSOU" if test.passed else "✗ FALHOU"
            report.append(f"{test.test_name}: {status}")
            report.append(f"  Valor: {test.value:.4f} (Limiar: {test.threshold:.4f})")
            report.append(f"  {test.description}")
            report.append(f"  Recomendação: {test.recommendation}")
            report.append("")
        
        # Conclusões
        report.append("CONCLUSÕES:")
        report.append("-" * 40)
        
        if validation_result.overall_grade == "Superior":
            report.append("O modelo atende aos critérios de grau SUPERIOR da NBR 14653.")
            report.append("Pode ser utilizado com alta confiança para avaliações.")
        elif validation_result.overall_grade == "Normal":
            report.append("O modelo atende aos critérios de grau NORMAL da NBR 14653.")
            report.append("Adequado para a maioria das avaliações imobiliárias.")
        elif validation_result.overall_grade == "Inferior":
            report.append("O modelo atende aos critérios de grau INFERIOR da NBR 14653.")
            report.append("Uso limitado, requer cuidados adicionais.")
        else:
            report.append("O modelo NÃO ATENDE aos critérios mínimos da NBR 14653.")
            report.append("Não recomendado para avaliações formais.")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)