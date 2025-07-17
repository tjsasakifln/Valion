"""
Fase 3: Construção e Treinamento do Modelo Elastic Net
Responsável por treinar o modelo de regressão com regularização.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
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


@dataclass
class ModelResult:
    """Resultado do treinamento do modelo."""
    model: ElasticNet
    performance: ModelPerformance
    best_params: Dict[str, Any]
    training_summary: Dict[str, Any]


class ElasticNetModelBuilder:
    """Construtor de modelo Elastic Net para avaliação imobiliária."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        
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
        Otimiza hiperparâmetros do Elastic Net.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Melhores parâmetros encontrados
        """
        # Grid de parâmetros
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        # Grid Search com cross-validation
        elastic_net = ElasticNet(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(
            elastic_net, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
        return grid_search.best_params_
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   hyperparams: Optional[Dict[str, Any]] = None) -> ElasticNet:
        """
        Treina o modelo Elastic Net.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            hyperparams: Hiperparâmetros (se None, usa otimização automática)
            
        Returns:
            Modelo treinado
        """
        if hyperparams is None:
            hyperparams = self.optimize_hyperparameters(X_train, y_train)
        
        # Treinar modelo final
        self.model = ElasticNet(
            alpha=hyperparams['alpha'],
            l1_ratio=hyperparams['l1_ratio'],
            random_state=42,
            max_iter=1000
        )
        
        self.model.fit(X_train, y_train)
        
        self.logger.info("Modelo treinado com sucesso")
        return self.model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      feature_names: List[str]) -> ModelPerformance:
        """
        Avalia performance do modelo.
        
        Args:
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
        
        # Coeficientes das features
        feature_coefficients = dict(zip(feature_names, self.model.coef_))
        
        return ModelPerformance(
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            mape=mape,
            cv_scores=cv_scores.tolist(),
            feature_coefficients=feature_coefficients
        )
    
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
        performance = self.evaluate_model(X_test, y_test, X_train.columns.tolist())
        
        # Resumo do treinamento
        training_summary = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features_count': len(X_train.columns),
            'target_mean': y_train.mean(),
            'target_std': y_train.std()
        }
        
        return ModelResult(
            model=model,
            performance=performance,
            best_params=best_params,
            training_summary=training_summary
        )
    
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
            'training_summary': model_result.training_summary
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
            training_summary=model_data['training_summary']
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