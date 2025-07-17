"""
Fase 2: Preparação e Transformação de Variáveis
Responsável por transformar e preparar variáveis para modelagem.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from dataclasses import dataclass
import logging


@dataclass
class TransformationResult:
    """Resultado das transformações aplicadas."""
    transformed_data: pd.DataFrame
    feature_names: List[str]
    transformations_applied: Dict[str, Any]
    feature_importance: Dict[str, float]


class VariableTransformer:
    """Transformador de variáveis para avaliação imobiliária."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        
    def create_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria variáveis derivadas relevantes para avaliação imobiliária.
        
        Args:
            df: DataFrame com dados básicos
            
        Returns:
            DataFrame com variáveis derivadas
        """
        df_derived = df.copy()
        
        # Valor por metro quadrado
        if 'valor' in df.columns and 'area_total' in df.columns:
            df_derived['valor_m2'] = df['valor'] / df['area_total']
        
        # Variáveis de localização
        if 'localizacao' in df.columns:
            # Dummy variables para localização
            location_dummies = pd.get_dummies(df['localizacao'], prefix='loc')
            df_derived = pd.concat([df_derived, location_dummies], axis=1)
        
        # Variáveis de idade (se disponível)
        if 'ano_construcao' in df.columns:
            current_year = pd.Timestamp.now().year
            df_derived['idade'] = current_year - df['ano_construcao']
            df_derived['idade_quadrada'] = df_derived['idade'] ** 2
        
        # Variáveis de área
        if 'area_total' in df.columns:
            df_derived['log_area'] = np.log(df['area_total'])
            df_derived['area_quadrada'] = df['area_total'] ** 2
        
        # Variáveis de qualidade (se disponível)
        if 'padrao_construcao' in df.columns:
            # Codificação ordinal para padrão
            padrao_map = {'baixo': 1, 'medio': 2, 'alto': 3, 'luxo': 4}
            df_derived['padrao_num'] = df['padrao_construcao'].map(padrao_map)
        
        self.logger.info(f"Variáveis derivadas criadas. Total de colunas: {len(df_derived.columns)}")
        return df_derived
    
    def handle_categorical_variables(self, df: pd.DataFrame, 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        """
        Trata variáveis categóricas.
        
        Args:
            df: DataFrame com dados
            categorical_cols: Lista de colunas categóricas
            
        Returns:
            DataFrame com variáveis categóricas codificadas
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                # Verifica cardinalidade
                unique_values = df[col].nunique()
                
                if unique_values <= 10:  # One-hot encoding para baixa cardinalidade
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                    df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
                    
                    self.encoders[col] = encoder
                    
                else:  # Label encoding para alta cardinalidade
                    encoder = LabelEncoder()
                    df_encoded[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
                    df_encoded = df_encoded.drop(columns=[col])
                    
                    self.encoders[col] = encoder
        
        return df_encoded
    
    def normalize_numerical_variables(self, df: pd.DataFrame, 
                                    numerical_cols: List[str]) -> pd.DataFrame:
        """
        Normaliza variáveis numéricas.
        
        Args:
            df: DataFrame com dados
            numerical_cols: Lista de colunas numéricas
            
        Returns:
            DataFrame com variáveis normalizadas
        """
        df_normalized = df.copy()
        
        for col in numerical_cols:
            if col in df.columns:
                scaler = StandardScaler()
                df_normalized[f"{col}_scaled"] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
        
        return df_normalized
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       max_features: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Seleciona features mais relevantes.
        
        Args:
            X: DataFrame com features
            y: Série com variável target
            max_features: Número máximo de features
            
        Returns:
            DataFrame com features selecionadas e importâncias
        """
        if max_features is None:
            max_features = min(len(X.columns), 20)  # Máximo de 20 features
        
        # Seleção univariada
        selector = SelectKBest(score_func=f_regression, k=max_features)
        X_selected = selector.fit_transform(X, y)
        
        # Nomes das features selecionadas
        selected_features = X.columns[selector.get_support()].tolist()
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        # Calcula importâncias
        scores = selector.scores_
        feature_importance = dict(zip(selected_features, scores[selector.get_support()]))
        
        # Normaliza importâncias
        max_score = max(feature_importance.values())
        feature_importance = {k: v/max_score for k, v in feature_importance.items()}
        
        self.logger.info(f"Features selecionadas: {len(selected_features)}")
        return X_selected_df, feature_importance
    
    def transform_data(self, df: pd.DataFrame, target_col: str = 'valor') -> TransformationResult:
        """
        Aplica todas as transformações nos dados.
        
        Args:
            df: DataFrame com dados originais
            target_col: Nome da coluna target
            
        Returns:
            Resultado das transformações
        """
        transformations_applied = {}
        
        # Separar target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Identificar tipos de variáveis
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Criar variáveis derivadas
        X_derived = self.create_derived_variables(X)
        transformations_applied['derived_variables'] = True
        
        # Tratar categóricas
        if categorical_cols:
            X_encoded = self.handle_categorical_variables(X_derived, categorical_cols)
            transformations_applied['categorical_encoding'] = categorical_cols
        else:
            X_encoded = X_derived
        
        # Normalizar numéricas
        numerical_cols_updated = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols_updated:
            X_normalized = self.normalize_numerical_variables(X_encoded, numerical_cols_updated)
            transformations_applied['numerical_normalization'] = numerical_cols_updated
        else:
            X_normalized = X_encoded
        
        # Seleção de features
        X_final, feature_importance = self.select_features(X_normalized, y)
        transformations_applied['feature_selection'] = True
        
        # Recombinar com target
        final_data = pd.concat([X_final, y], axis=1)
        
        return TransformationResult(
            transformed_data=final_data,
            feature_names=X_final.columns.tolist(),
            transformations_applied=transformations_applied,
            feature_importance=feature_importance
        )