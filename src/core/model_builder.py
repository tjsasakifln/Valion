"""
Fase 3: Construção e Treinamento do Modelo Elastic Net
Responsável por treinar o modelo de regressão com regularização.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path


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
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'valor', 
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara dados para treinamento.
        
        Args:
            df: DataFrame com dados
            target_col: Nome da coluna target
            test_size: Proporção para teste
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Escalar features
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
        
        self.logger.info(f"Dados preparados - Treino: {len(X_train)}, Teste: {len(X_test)}")
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
        Treina o modelo selecionado.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            hyperparams: Hiperparâmetros (se None, usa otimização automática)
            
        Returns:
            Modelo treinado
        """
        if hyperparams is None:
            hyperparams = self.optimize_hyperparameters(X_train, y_train)
        
        # Treinar modelo conforme o tipo
        if self.model_type == 'elastic_net':
            self.model = ElasticNet(
                alpha=hyperparams['alpha'],
                l1_ratio=hyperparams['l1_ratio'],
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'],
                learning_rate=hyperparams['learning_rate'],
                subsample=hyperparams['subsample'],
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'],
                learning_rate=hyperparams['learning_rate'],
                subsample=hyperparams['subsample'],
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'],
                min_samples_split=hyperparams['min_samples_split'],
                min_samples_leaf=hyperparams['min_samples_leaf'],
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        self.logger.info(f"Modelo {self.model_type} treinado com sucesso")
        return self.model
    
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
        Calcula SHAP values para interpretabilidade do modelo.
        OBRIGATÓRIO no modo especialista para manter filosofia "glass-box".
        """
        # Usar amostra menor para evitar problemas de memória
        sample_size = min(100, len(X_train))
        X_train_sample = X_train.sample(n=sample_size, random_state=42)
        
        if self.model_type == 'elastic_net':
            # Para modelos lineares
            self.explainer = shap.LinearExplainer(self.model, X_train_sample)
        else:
            # Para modelos baseados em árvore
            self.explainer = shap.TreeExplainer(self.model)
        
        # Calcular SHAP values
        sample_size = min(50, len(X_test))
        shap_values = self.explainer.shap_values(X_test.iloc[:sample_size])
        self.logger.info(f"SHAP values calculados para {sample_size} amostras no modo especialista")
        
        # Calcular importância média das features
        if shap_values.ndim == 2:
            shap_importance = np.abs(shap_values).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values).mean()
        
        shap_feature_importance = dict(zip(X_test.columns, shap_importance))
        
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