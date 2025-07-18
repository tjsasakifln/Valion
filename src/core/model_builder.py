"""
Fase 3: Construção e Treinamento do Modelo Elastic Net
Responsável por treinar o modelo de regressão com regularização.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb
import shap
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path
import warnings
from scipy import stats
from scipy.linalg import LinAlgError
import time


@dataclass
class ModelPerformance:
    """Métricas de performance do modelo."""
    r2_score: float
    rmse: float
    mae: float
    mape: float
    cv_scores: List[float]
    feature_coefficients: Dict[str, float]
    shap_values: Optional[np.ndarray] = None
    shap_feature_importance: Optional[Dict[str, float]] = None
    permutation_importance: Optional[Dict[str, float]] = None


@dataclass
class ModelResult:
    """Resultado do treinamento do modelo."""
    model: Any  # Can be ElasticNet, XGBoost, or GradientBoosting
    performance: ModelPerformance
    best_params: Dict[str, Any]
    training_summary: Dict[str, Any]
    model_type: str
    explainer: Optional[Any] = None  # SHAP explainer


class ModelBuilder:
    """Construtor de modelos para avaliação imobiliária com Modo Especialista."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.explainer = None
        self.expert_mode = config.get('expert_mode', False)
        # Garantir Elastic Net como padrão (rejeitando métodos como stepwise)
        self.model_type = config.get('model_type', 'elastic_net')
        # Validar que não seja um método ultrapassado
        forbidden_methods = ['stepwise', 'backward', 'forward']
        if self.model_type in forbidden_methods:
            self.logger.warning(f"Método {self.model_type} rejeitado. Usando Elastic Net como padrão.")
            self.model_type = 'elastic_net'
        
    def _validate_numerical_stability(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Valida estabilidade numérica dos dados antes do treinamento.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dicionário com informações de validação
        """
        stability_info = {
            'is_stable': True,
            'issues': [],
            'recommendations': []
        }
        
        # Verificar multicolinearidade através de correlação
        corr_matrix = X.corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.95:  # Correlação muito alta
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            stability_info['issues'].append('multicolinearidade_detectada')
            stability_info['high_correlations'] = high_corr_pairs
            stability_info['recommendations'].append(
                f"Remover uma das variáveis dos {len(high_corr_pairs)} pares altamente correlacionados"
            )
        
        # Verificar variância das features
        variances = X.var()
        zero_var_features = variances[variances == 0].index.tolist()
        low_var_features = variances[(variances > 0) & (variances < 1e-10)].index.tolist()
        
        if zero_var_features:
            stability_info['issues'].append('features_variancia_zero')
            stability_info['zero_variance_features'] = zero_var_features
            stability_info['recommendations'].append(
                f"Remover {len(zero_var_features)} features com variância zero"
            )
        
        if low_var_features:
            stability_info['issues'].append('features_variancia_baixa')
            stability_info['low_variance_features'] = low_var_features
            stability_info['recommendations'].append(
                f"Considerar remoção de {len(low_var_features)} features com variância muito baixa"
            )
        
        # Verificar outliers extremos que podem causar instabilidade
        outlier_features = []
        for col in X.select_dtypes(include=[np.number]).columns:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                extreme_outliers = ((X[col] < q1 - 3 * iqr) | (X[col] > q3 + 3 * iqr)).sum()
                if extreme_outliers / len(X) > 0.05:  # Mais de 5% outliers extremos
                    outlier_features.append((col, extreme_outliers))
        
        if outlier_features:
            stability_info['issues'].append('outliers_extremos')
            stability_info['extreme_outliers'] = outlier_features
            stability_info['recommendations'].append(
                "Considerar tratamento de outliers extremos para estabilidade numérica"
            )
        
        # Verificar condição da matriz de correlação (aproximação do número de condição)
        try:
            eigenvals = np.linalg.eigvals(corr_matrix.fillna(0).values)
            condition_number = max(eigenvals) / min(eigenvals) if min(eigenvals) > 1e-10 else np.inf
            
            if condition_number > 1e12:
                stability_info['issues'].append('matriz_mal_condicionada')
                stability_info['condition_number'] = float(condition_number)
                stability_info['recommendations'].append(
                    "Matriz de correlação mal condicionada - considerar regularização mais forte"
                )
                stability_info['is_stable'] = False
        except Exception as e:
            self.logger.warning(f"Erro ao calcular número de condição: {e}")
        
        return stability_info
    
    def _apply_numerical_fixes(self, X: pd.DataFrame, y: pd.Series, 
                              stability_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica correções automáticas para problemas de estabilidade numérica.
        
        Args:
            X: Features
            y: Target
            stability_info: Informações de validação
            
        Returns:
            X e y corrigidos
        """
        X_fixed = X.copy()
        y_fixed = y.copy()
        
        # Remover features com variância zero
        if 'zero_variance_features' in stability_info:
            features_to_remove = stability_info['zero_variance_features']
            X_fixed = X_fixed.drop(columns=features_to_remove)
            self.logger.info(f"Removidas {len(features_to_remove)} features com variância zero")
        
        # Tratar multicolinearidade removendo uma das variáveis correlacionadas
        if 'high_correlations' in stability_info:
            features_to_remove = set()
            for feat1, feat2, corr in stability_info['high_correlations']:
                # Remover a feature com menor correlação com o target
                if feat1 in X_fixed.columns and feat2 in X_fixed.columns:
                    corr_target1 = abs(X_fixed[feat1].corr(y_fixed))
                    corr_target2 = abs(X_fixed[feat2].corr(y_fixed))
                    
                    if corr_target1 < corr_target2:
                        features_to_remove.add(feat1)
                    else:
                        features_to_remove.add(feat2)
            
            if features_to_remove:
                X_fixed = X_fixed.drop(columns=list(features_to_remove))
                self.logger.info(f"Removidas {len(features_to_remove)} features por multicolinearidade")
        
        # Aplicar winsorização para outliers extremos
        if 'extreme_outliers' in stability_info:
            for col, outlier_count in stability_info['extreme_outliers']:
                if col in X_fixed.columns:
                    # Winsorizar nos percentis 1% e 99%
                    lower_bound = X_fixed[col].quantile(0.01)
                    upper_bound = X_fixed[col].quantile(0.99)
                    
                    X_fixed[col] = X_fixed[col].clip(lower=lower_bound, upper=upper_bound)
                    self.logger.info(f"Aplicada winsorização na feature '{col}'")
        
        return X_fixed, y_fixed
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'valor', 
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara dados para treinamento com validações de estabilidade numérica.
        
        Args:
            df: DataFrame com dados
            target_col: Nome da coluna target
            test_size: Proporção para teste
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Validar estabilidade numérica
        stability_info = self._validate_numerical_stability(X, y)
        
        if not stability_info['is_stable']:
            self.logger.warning("Problemas de estabilidade numérica detectados. Aplicando correções automáticas.")
            for issue in stability_info['issues']:
                self.logger.warning(f"Problema detectado: {issue}")
            
            # Aplicar correções
            X, y = self._apply_numerical_fixes(X, y, stability_info)
        
        # Verificar se ainda temos features suficientes
        if len(X.columns) < 2:
            raise ValueError("Dados insuficientes após correções de estabilidade numérica")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Escolher scaler baseado na presença de outliers
        if 'extreme_outliers' in stability_info:
            self.scaler = RobustScaler()  # Mais robusto a outliers
            self.logger.info("Usando RobustScaler devido à presença de outliers")
        else:
            self.scaler = StandardScaler()
        
        # Escalar features com tratamento de erro
        try:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        except Exception as e:
            self.logger.error(f"Erro na padronização: {e}")
            raise ValueError(f"Falha na padronização dos dados: {e}")
        
        # Verificação final de sanidade
        if X_train_scaled.isnull().any().any():
            self.logger.error("Valores nulos detectados após padronização")
            raise ValueError("Dados inválidos após padronização")
        
        self.logger.info(f"Dados preparados com estabilidade - Treino: {len(X_train)}, Teste: {len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Otimiza hiperparâmetros do modelo selecionado.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Melhores parâmetros encontrados
        """
        if self.model_type == 'elastic_net':
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
            model = ElasticNet(random_state=42, max_iter=1000)
            
        elif self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42)
            
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
            
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")
        
        # Grid Search com cross-validation
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f"Melhores parâmetros ({self.model_type}): {grid_search.best_params_}")
        return grid_search.best_params_
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   hyperparams: Optional[Dict[str, Any]] = None) -> Any:
        """
        Treina o modelo selecionado com proteções contra instabilidade numérica.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            hyperparams: Hiperparâmetros (se None, usa otimização automática)
            
        Returns:
            Modelo treinado
        """
        if hyperparams is None:
            hyperparams = self.optimize_hyperparameters(X_train, y_train)
        
        # Treinar modelo conforme o tipo com tratamento robusto de erros
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                if self.model_type == 'elastic_net':
                    # Aumentar max_iter progressivamente se houver problemas de convergência
                    max_iter = 1000 * (2 ** attempt)
                    
                    self.model = ElasticNet(
                        alpha=hyperparams['alpha'],
                        l1_ratio=hyperparams['l1_ratio'],
                        random_state=42,
                        max_iter=max_iter,
                        tol=1e-4,
                        selection='random'  # Mais robusto que 'cyclic'
                    )
                    
                elif self.model_type == 'xgboost':
                    self.model = xgb.XGBRegressor(
                        n_estimators=hyperparams['n_estimators'],
                        max_depth=hyperparams['max_depth'],
                        learning_rate=hyperparams['learning_rate'],
                        subsample=hyperparams['subsample'],
                        random_state=42,
                        reg_alpha=0.1,  # Regularização L1 para estabilidade
                        reg_lambda=0.1,  # Regularização L2 para estabilidade
                        n_jobs=1  # Evitar problemas de concorrência
                    )
                    
                elif self.model_type == 'gradient_boosting':
                    self.model = GradientBoostingRegressor(
                        n_estimators=hyperparams['n_estimators'],
                        max_depth=hyperparams['max_depth'],
                        learning_rate=hyperparams['learning_rate'],
                        subsample=hyperparams['subsample'],
                        random_state=42,
                        validation_fraction=0.1,  # Early stopping
                        n_iter_no_change=10
                    )
                    
                elif self.model_type == 'random_forest':
                    self.model = RandomForestRegressor(
                        n_estimators=hyperparams['n_estimators'],
                        max_depth=hyperparams['max_depth'],
                        min_samples_split=hyperparams['min_samples_split'],
                        min_samples_leaf=hyperparams['min_samples_leaf'],
                        random_state=42,
                        n_jobs=1  # Evitar problemas de concorrência
                    )
                
                # Capturar warnings de convergência
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    start_time = time.time()
                    self.model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Verificar warnings de convergência
                    convergence_warnings = [warning for warning in w 
                                           if issubclass(warning.category, ConvergenceWarning)]
                    
                    if convergence_warnings and attempt < max_attempts - 1:
                        self.logger.warning(f"Aviso de convergência na tentativa {attempt + 1}. Tentando novamente...")
                        continue
                    
                    if convergence_warnings:
                        self.logger.warning("Modelo treinado com avisos de convergência - considere ajustar hiperparâmetros")
                
                # Verificar se o modelo foi treinado corretamente
                if hasattr(self.model, 'coef_') and np.any(np.isnan(self.model.coef_)):
                    raise ValueError("Coeficientes NaN detectados no modelo")
                
                if hasattr(self.model, 'feature_importances_') and np.any(np.isnan(self.model.feature_importances_)):
                    raise ValueError("Feature importances NaN detectadas")
                
                self.logger.info(f"Modelo {self.model_type} treinado com sucesso em {training_time:.2f}s (tentativa {attempt + 1})")
                return self.model
                
            except (ConvergenceWarning, LinAlgError, ValueError) as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(f"Erro no treinamento (tentativa {attempt + 1}): {e}. Tentando novamente...")
                    
                    # Ajustar hiperparâmetros para maior estabilidade
                    if self.model_type == 'elastic_net':
                        # Aumentar regularização
                        hyperparams['alpha'] = min(hyperparams['alpha'] * 2, 10.0)
                    elif self.model_type in ['xgboost', 'gradient_boosting']:
                        # Reduzir learning rate
                        hyperparams['learning_rate'] = max(hyperparams['learning_rate'] * 0.5, 0.01)
                    
                    continue
                else:
                    self.logger.error(f"Falha no treinamento após {max_attempts} tentativas: {e}")
                    raise ValueError(f"Não foi possível treinar o modelo estável: {e}")
            
            except Exception as e:
                self.logger.error(f"Erro inesperado no treinamento: {e}")
                raise
    
    def evaluate_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_test: pd.Series, feature_names: List[str]) -> ModelPerformance:
        """
        Avalia performance do modelo com interpretabilidade SHAP.
        
        Args:
            X_train: Features de treino (para SHAP)
            X_test: Features de teste
            y_test: Target de teste
            feature_names: Nomes das features
            
        Returns:
            Métricas de performance
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        
        # Predições
        y_pred = self.model.predict(X_test)
        
        # Métricas
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, 
                                   scoring='neg_mean_squared_error')
        cv_scores = np.sqrt(-cv_scores)  # Converter para RMSE
        
        # Feature importance/coefficients
        feature_coefficients = self._get_feature_importance(feature_names)
        
        # SHAP values para interpretabilidade - OBRIGATÓRIO no modo especialista
        shap_values = None
        shap_feature_importance = None
        if self.expert_mode:
            try:
                shap_values, shap_feature_importance = self._calculate_shap_values(X_train, X_test)
                self.logger.info("SHAP values calculados - interpretabilidade garantida no modo especialista")
            except Exception as e:
                self.logger.error(f"ERRO CRÍTICO: SHAP obrigatório no modo especialista falhou: {e}")
                raise ValueError(f"Modo especialista requer SHAP - {e}")
        
        # Permutation importance
        perm_importance = None
        try:
            perm_result = permutation_importance(self.model, X_test, y_test, 
                                               n_repeats=10, random_state=42)
            perm_importance = dict(zip(feature_names, perm_result.importances_mean))
        except Exception as e:
            self.logger.warning(f"Erro ao calcular permutation importance: {e}")
        
        return ModelPerformance(
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            mape=mape,
            cv_scores=cv_scores.tolist(),
            feature_coefficients=feature_coefficients,
            shap_values=shap_values,
            shap_feature_importance=shap_feature_importance,
            permutation_importance=perm_importance
        )
    
    def _get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Obtém importância das features baseada no tipo de modelo.
        """
        if hasattr(self.model, 'coef_'):
            # Modelos lineares (ElasticNet)
            return dict(zip(feature_names, self.model.coef_))
        elif hasattr(self.model, 'feature_importances_'):
            # Modelos baseados em árvore (XGBoost, GradientBoosting, RandomForest)
            return dict(zip(feature_names, self.model.feature_importances_))
        else:
            # Fallback
            return {name: 0.0 for name in feature_names}
    
    def _calculate_shap_values(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calcula SHAP values para interpretabilidade do modelo com fallbacks robustos.
        OBRIGATÓRIO no modo especialista para manter filosofia "glass-box".
        """
        shap_values = None
        shap_feature_importance = None
        
        # Usar amostra menor para evitar problemas de memória
        sample_size = min(100, len(X_train))
        X_train_sample = X_train.sample(n=sample_size, random_state=42)
        
        # Lista de explainers para tentar (ordem de preferência)
        explainer_attempts = []
        
        if self.model_type == 'elastic_net':
            explainer_attempts = [
                ('LinearExplainer', lambda: shap.LinearExplainer(self.model, X_train_sample)),
                ('KernelExplainer', lambda: shap.KernelExplainer(self.model.predict, X_train_sample))
            ]
        else:
            explainer_attempts = [
                ('TreeExplainer', lambda: shap.TreeExplainer(self.model)),
                ('KernelExplainer', lambda: shap.KernelExplainer(self.model.predict, X_train_sample))
            ]
        
        # Tentar cada explainer até um funcionar
        for explainer_name, explainer_func in explainer_attempts:
            try:
                self.logger.info(f"Tentando {explainer_name} para SHAP...")
                self.explainer = explainer_func()
                
                # Calcular SHAP values
                sample_size = min(50, len(X_test))
                X_test_sample = X_test.iloc[:sample_size]
                
                # Timeout para evitar travamento
                start_time = time.time()
                
                if explainer_name == 'KernelExplainer':
                    # KernelExplainer é mais lento, usar amostra menor
                    sample_size = min(20, sample_size)
                    X_test_sample = X_test.iloc[:sample_size]
                
                shap_values = self.explainer.shap_values(X_test_sample)
                calculation_time = time.time() - start_time
                
                if calculation_time > 300:  # 5 minutos
                    self.logger.warning(f"SHAP calculation took {calculation_time:.1f}s - consider optimization")
                
                # Verificar se os valores são válidos
                if np.any(np.isnan(shap_values)) or np.any(np.isinf(shap_values)):
                    raise ValueError("SHAP values contêm NaN ou Inf")
                
                # Calcular importância média das features
                if shap_values.ndim == 2:
                    shap_importance = np.abs(shap_values).mean(axis=0)
                else:
                    shap_importance = np.abs(shap_values).mean()
                
                # Verificar se a importância é válida
                if np.any(np.isnan(shap_importance)) or np.any(np.isinf(shap_importance)):
                    raise ValueError("SHAP importance contém NaN ou Inf")
                
                shap_feature_importance = dict(zip(X_test_sample.columns, shap_importance))
                
                self.logger.info(f"SHAP values calculados com sucesso usando {explainer_name} para {sample_size} amostras")
                break
                
            except Exception as e:
                self.logger.warning(f"Falha com {explainer_name}: {e}")
                if explainer_name == explainer_attempts[-1][0]:  # Último explainer
                    # Se todos os explainers falharam, calcular importância básica
                    self.logger.error("Todos os explainers SHAP falharam. Usando importância de features como fallback.")
                    
                    if self.expert_mode:
                        raise ValueError(f"SHAP obrigatório no modo especialista falhou com todos os métodos: {e}")
                    
                    # Fallback: usar feature importance do modelo
                    feature_importance = self._get_feature_importance(X_test.columns.tolist())
                    shap_feature_importance = feature_importance
                    shap_values = np.zeros((min(50, len(X_test)), len(X_test.columns)))
                    
                    self.logger.warning("Usando feature importance como substituto para SHAP")
                continue
        
        return shap_values, shap_feature_importance
    
    def get_shap_explanations(self, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera explicações SHAP detalhadas para o modo especialista.
        """
        if self.explainer is None:
            raise ValueError("Explainer SHAP não disponível")
        
        shap_values = self.explainer.shap_values(X_sample)
        
        explanations = {
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0,
            'feature_contributions': {},
            'sample_explanations': []
        }
        
        # Contribuições médias das features
        if shap_values.ndim == 2:
            avg_contributions = np.abs(shap_values).mean(axis=0)
            explanations['feature_contributions'] = dict(zip(X_sample.columns, avg_contributions))
            
            # Explicações por amostra
            for i, (idx, row) in enumerate(X_sample.iterrows()):
                sample_explanation = {
                    'sample_id': str(idx),
                    'features': row.to_dict(),
                    'shap_values': dict(zip(X_sample.columns, shap_values[i])),
                    'positive_contributions': {k: v for k, v in zip(X_sample.columns, shap_values[i]) if v > 0},
                    'negative_contributions': {k: v for k, v in zip(X_sample.columns, shap_values[i]) if v < 0}
                }
                explanations['sample_explanations'].append(sample_explanation)
        
        return explanations
    
    def build_model(self, df: pd.DataFrame, target_col: str = 'valor') -> ModelResult:
        """
        Constrói modelo completo.
        
        Args:
            df: DataFrame com dados
            target_col: Nome da coluna target
            
        Returns:
            Resultado completo do modelo
        """
        # Preparar dados
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Otimizar hiperparâmetros
        best_params = self.optimize_hyperparameters(X_train, y_train)
        
        # Treinar modelo
        model = self.train_model(X_train, y_train, best_params)
        
        # Avaliar modelo
        performance = self.evaluate_model(X_train, X_test, y_test, X_train.columns.tolist())
        
        # Resumo do treinamento
        training_summary = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features_count': len(X_train.columns),
            'target_mean': y_train.mean(),
            'target_std': y_train.std()
        }
        
        result = ModelResult(
            model=model,
            performance=performance,
            best_params=best_params,
            training_summary=training_summary,
            model_type=self.model_type,
            explainer=self.explainer
        )
        
        # Log de conformidade com padrões
        self.logger.info(f"Modelo {self.model_type} construído com sucesso")
        if self.expert_mode:
            self.logger.info("Modo especialista ativo - SHAP obrigatório garantido")
        
        return result
    
    def save_model(self, model_result: ModelResult, filepath: str) -> None:
        """
        Salva modelo treinado.
        
        Args:
            model_result: Resultado do modelo
            filepath: Caminho para salvar
        """
        model_data = {
            'model': model_result.model,
            'scaler': self.scaler,
            'performance': model_result.performance,
            'best_params': model_result.best_params,
            'training_summary': model_result.training_summary,
            'model_type': model_result.model_type,
            'explainer': model_result.explainer
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str) -> ModelResult:
        """
        Carrega modelo salvo.
        
        Args:
            filepath: Caminho do modelo
            
        Returns:
            Resultado do modelo carregado
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        
        return ModelResult(
            model=model_data['model'],
            performance=model_data['performance'],
            best_params=model_data['best_params'],
            training_summary=model_data['training_summary'],
            model_type=model_data.get('model_type', 'elastic_net'),
            explainer=model_data.get('explainer')
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições com o modelo.
        
        Args:
            X: Features para predição
            
        Returns:
            Predições
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)