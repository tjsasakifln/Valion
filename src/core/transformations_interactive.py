# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Fase 2: Preparação e Transformação de Variáveis Interativa
Responsável por transformar e preparar variáveis com aprovação do usuário.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats
from dataclasses import dataclass
import logging
import warnings


@dataclass
class TransformationSuggestion:
    """Sugestão de transformação interativa."""
    column: str
    transformation_type: str
    reason: str
    impact_score: float
    preview_data: Optional[pd.Series] = None
    parameters: Optional[Dict[str, Any]] = None
    user_approved: bool = False


@dataclass
class OutlierDetectionResult:
    """Resultado da detecção de outliers."""
    outlier_indices: List[int]
    outlier_scores: np.ndarray
    method_used: str
    threshold: float
    affected_columns: List[str]
    user_approved_removal: bool = False


@dataclass
class TransformationResult:
    """Resultado das transformações aplicadas."""
    transformed_data: pd.DataFrame
    feature_names: List[str]
    transformations_applied: Dict[str, Any]
    feature_importance: Dict[str, float]
    summary: Dict[str, Any]
    suggestions_made: List[TransformationSuggestion]
    outliers_detected: Optional[OutlierDetectionResult] = None
    user_interactions: Dict[str, Any] = None


class InteractiveDataTransformer:
    """Transformador de dados interativo com sugestões e aprovação do usuário."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.interactive_mode = config.get('interactive_mode', True)
        self.auto_approve_threshold = config.get('auto_approve_threshold', 0.8)
        
        # Callbacks para interação com usuário
        self.suggestion_callback: Optional[Callable] = None
        self.outlier_callback: Optional[Callable] = None
    
    def detect_outliers_interactive(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> OutlierDetectionResult:
        """
        Detecta outliers com aprovação interativa do usuário.
        
        Args:
            df: DataFrame para análise
            columns: Colunas para análise (se None, usa todas numéricas)
            
        Returns:
            Resultado da detecção de outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        numeric_data = df[columns].copy()
        
        # Método 1: Isolation Forest
        isolation_forest = IsolationForest(
            contamination=self.config.get('outlier_contamination', 0.1),
            random_state=42
        )
        outlier_labels = isolation_forest.fit_predict(numeric_data.fillna(numeric_data.median()))
        outlier_scores = isolation_forest.score_samples(numeric_data.fillna(numeric_data.median()))
        
        outlier_indices = np.where(outlier_labels == -1)[0].tolist()
        
        # Calcular threshold
        threshold = np.percentile(outlier_scores, 
                                self.config.get('outlier_percentile', 10))
        
        result = OutlierDetectionResult(
            outlier_indices=outlier_indices,
            outlier_scores=outlier_scores,
            method_used='Isolation Forest',
            threshold=threshold,
            affected_columns=columns
        )
        
        self.logger.info(f"Detectados {len(outlier_indices)} outliers usando {result.method_used}")
        
        # Se em modo interativo, solicitar aprovação do usuário
        if self.interactive_mode and self.outlier_callback and len(outlier_indices) > 0:
            try:
                result.user_approved_removal = self.outlier_callback(result, df)
            except Exception as e:
                self.logger.warning(f"Erro no callback de outliers: {e}")
                result.user_approved_removal = False
        else:
            # Auto-aprovar se poucos outliers
            result.user_approved_removal = len(outlier_indices) / len(df) < 0.05
        
        return result
    
    def suggest_transformations(self, df: pd.DataFrame, target_col: str) -> List[TransformationSuggestion]:
        """
        Sugere transformações baseadas na análise dos dados.
        
        Args:
            df: DataFrame para análise
            target_col: Coluna target
            
        Returns:
            Lista de sugestões de transformação
        """
        suggestions = []
        
        # Análise de cada coluna
        for col in df.columns:
            if col == target_col:
                continue
                
            if df[col].dtype in ['object', 'category']:
                # Sugestões para variáveis categóricas
                suggestions.extend(self._suggest_categorical_transformations(df, col, target_col))
            else:
                # Sugestões para variáveis numéricas
                suggestions.extend(self._suggest_numerical_transformations(df, col, target_col))
        
        # Ordenar por impacto
        suggestions.sort(key=lambda x: x.impact_score, reverse=True)
        
        self.logger.info(f"Geradas {len(suggestions)} sugestões de transformação")
        return suggestions
    
    def _suggest_categorical_transformations(self, df: pd.DataFrame, col: str, target_col: str) -> List[TransformationSuggestion]:
        """
        Sugere transformações para variáveis categóricas.
        """
        suggestions = []
        unique_values = df[col].nunique()
        
        # One-hot encoding para baixa cardinalidade
        if unique_values <= 10:
            correlation_score = self._calculate_categorical_correlation(df, col, target_col)
            suggestions.append(TransformationSuggestion(
                column=col,
                transformation_type="one_hot_encoding",
                reason=f"Variável categórica com {unique_values} valores únicos. One-hot encoding pode melhorar a performance.",
                impact_score=correlation_score,
                parameters={'drop_first': True}
            ))
        
        # Label encoding para alta cardinalidade
        elif unique_values > 10:
            correlation_score = self._calculate_categorical_correlation(df, col, target_col)
            suggestions.append(TransformationSuggestion(
                column=col,
                transformation_type="label_encoding",
                reason=f"Variável categórica com alta cardinalidade ({unique_values} valores). Label encoding é mais eficiente.",
                impact_score=correlation_score * 0.8,  # Penalizar um pouco
                parameters={}
            ))
        
        return suggestions
    
    def _suggest_numerical_transformations(self, df: pd.DataFrame, col: str, target_col: str) -> List[TransformationSuggestion]:
        """
        Sugere transformações para variáveis numéricas.
        """
        suggestions = []
        data = df[col].dropna()
        
        if len(data) == 0:
            return suggestions
        
        # Testar normalidade
        _, p_value = stats.shapiro(data.sample(min(5000, len(data)), random_state=42))
        is_normal = p_value > 0.05
        
        # Calcular skewness
        skewness = stats.skew(data)
        
        # Transformação logarítmica para dados positivos com alta skewness
        if data.min() > 0 and abs(skewness) > 1:
            log_data = np.log(data)
            log_skewness = stats.skew(log_data)
            improvement = abs(skewness) - abs(log_skewness)
            
            suggestions.append(TransformationSuggestion(
                column=col,
                transformation_type="log_transform",
                reason=f"Dados com alta assimetria (skewness={skewness:.2f}). Transformação logarítmica pode normalizar.",
                impact_score=min(improvement * 0.5, 0.9),
                preview_data=log_data,
                parameters={}
            ))
        
        # Power transformation (Box-Cox/Yeo-Johnson)
        if not is_normal and len(data) > 10:
            try:
                # Yeo-Johnson funciona com valores negativos
                pt = PowerTransformer(method='yeo-johnson')
                transformed_data = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
                
                # Verificar melhoria na normalidade
                _, transformed_p = stats.shapiro(pd.Series(transformed_data).sample(min(5000, len(transformed_data)), random_state=42))
                
                if transformed_p > p_value:
                    suggestions.append(TransformationSuggestion(
                        column=col,
                        transformation_type="power_transform",
                        reason=f"Dados não normais (p-value={p_value:.4f}). Power transformation pode melhorar normalidade.",
                        impact_score=min((transformed_p - p_value) * 2, 0.8),
                        preview_data=pd.Series(transformed_data, index=data.index),
                        parameters={'method': 'yeo-johnson'}
                    ))
            except Exception as e:
                self.logger.debug(f"Erro na power transformation para {col}: {e}")
        
        # Padronização para variáveis com escalas diferentes
        correlation_with_target = abs(df[col].corr(df[target_col]))
        if correlation_with_target > 0.1:  # Apenas se tiver alguma correlação
            suggestions.append(TransformationSuggestion(
                column=col,
                transformation_type="standardization",
                reason=f"Variável com correlação significativa ({correlation_with_target:.3f}) com target. Padronização pode melhorar performance.",
                impact_score=correlation_with_target * 0.6,
                parameters={}
            ))
        
        return suggestions
    
    def _calculate_categorical_correlation(self, df: pd.DataFrame, cat_col: str, target_col: str) -> float:
        """
        Calcula correlação entre variável categórica e target numérico.
        """
        try:
            # ANOVA F-statistic
            groups = [group[target_col].values for name, group in df.groupby(cat_col) if len(group) > 1]
            if len(groups) < 2:
                return 0.0
            
            f_stat, p_value = stats.f_oneway(*groups)
            # Converter F-statistic para score normalizado
            return min(f_stat / (f_stat + 100), 0.9)  # Normalizar entre 0 e 0.9
        except Exception:
            return 0.0
    
    def apply_approved_transformations(self, df: pd.DataFrame, suggestions: List[TransformationSuggestion]) -> pd.DataFrame:
        """
        Aplica transformações aprovadas pelo usuário.
        
        Args:
            df: DataFrame original
            suggestions: Lista de sugestões com status de aprovação
            
        Returns:
            DataFrame transformado
        """
        transformed_df = df.copy()
        
        for suggestion in suggestions:
            if not suggestion.user_approved:
                continue
                
            col = suggestion.column
            trans_type = suggestion.transformation_type
            
            try:
                if trans_type == "log_transform":
                    transformed_df[f"{col}_log"] = np.log(transformed_df[col])
                    self.logger.info(f"Aplicada transformação logarítmica em {col}")
                    
                elif trans_type == "power_transform":
                    pt = PowerTransformer(method=suggestion.parameters['method'])
                    transformed_df[f"{col}_power"] = pt.fit_transform(transformed_df[[col]]).flatten()
                    self.logger.info(f"Aplicada power transformation em {col}")
                    
                elif trans_type == "one_hot_encoding":
                    dummies = pd.get_dummies(transformed_df[col], prefix=col, drop_first=suggestion.parameters.get('drop_first', True))
                    transformed_df = pd.concat([transformed_df, dummies], axis=1)
                    self.logger.info(f"Aplicado one-hot encoding em {col}")
                    
                elif trans_type == "label_encoding":
                    le = LabelEncoder()
                    transformed_df[f"{col}_encoded"] = le.fit_transform(transformed_df[col].astype(str))
                    self.encoders[col] = le
                    self.logger.info(f"Aplicado label encoding em {col}")
                    
                elif trans_type == "standardization":
                    scaler = StandardScaler()
                    transformed_df[f"{col}_scaled"] = scaler.fit_transform(transformed_df[[col]]).flatten()
                    self.scalers[col] = scaler
                    self.logger.info(f"Aplicada padronização em {col}")
                    
            except Exception as e:
                self.logger.error(f"Erro ao aplicar transformação {trans_type} em {col}: {e}")
        
        return transformed_df
    
    def transform_data_interactive(self, df: pd.DataFrame, target_col: str) -> TransformationResult:
        """
        Aplica transformações de forma interativa com aprovação do usuário.
        
        Args:
            df: DataFrame para transformar
            target_col: Nome da coluna target
            
        Returns:
            Resultado das transformações
        """
        self.logger.info("Iniciando transformação interativa de dados")
        
        # 1. Detectar outliers
        outliers_result = self.detect_outliers_interactive(df)
        
        # 2. Remover outliers se aprovado
        working_df = df.copy()
        if outliers_result.user_approved_removal and outliers_result.outlier_indices:
            working_df = working_df.drop(index=outliers_result.outlier_indices)
            self.logger.info(f"Removidos {len(outliers_result.outlier_indices)} outliers")
        
        # 3. Gerar sugestões de transformação
        suggestions = self.suggest_transformations(working_df, target_col)
        
        # 4. Solicitar aprovação do usuário para sugestões
        if self.interactive_mode and self.suggestion_callback and suggestions:
            try:
                approved_suggestions = self.suggestion_callback(suggestions, working_df)
                for i, suggestion in enumerate(suggestions):
                    suggestion.user_approved = i in approved_suggestions
            except Exception as e:
                self.logger.warning(f"Erro no callback de sugestões: {e}")
                # Auto-aprovar sugestões com alto impacto
                for suggestion in suggestions:
                    suggestion.user_approved = suggestion.impact_score > self.auto_approve_threshold
        else:
            # Auto-aprovar sugestões com alto impacto
            for suggestion in suggestions:
                suggestion.user_approved = suggestion.impact_score > self.auto_approve_threshold
        
        # 5. Aplicar transformações aprovadas
        transformed_df = self.apply_approved_transformations(working_df, suggestions)
        
        # 6. Continuar com transformações padrão (seleção de features, etc.)
        # Separar target
        y = transformed_df[target_col]
        X = transformed_df.drop(columns=[target_col])
        
        # Seleção de features
        from sklearn.feature_selection import SelectKBest, f_regression
        max_features = min(len(X.columns), self.config.get('max_features', 20))
        
        # Apenas features numéricas para seleção
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_features:
            selector = SelectKBest(score_func=f_regression, k=min(max_features, len(numeric_features)))
            X_selected = selector.fit_transform(X[numeric_features], y)
            
            selected_features = np.array(numeric_features)[selector.get_support()].tolist()
            X_final = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            # Calcular importâncias
            scores = selector.scores_
            feature_importance = dict(zip(selected_features, scores[selector.get_support()]))
            
            # Normalizar importâncias
            if feature_importance:
                max_score = max(feature_importance.values())
                feature_importance = {k: v/max_score for k, v in feature_importance.items()}
        else:
            X_final = X
            feature_importance = {}
            selected_features = X.columns.tolist()
        
        # Recombinar com target
        final_data = pd.concat([X_final, y], axis=1)
        
        # Resumo das transformações
        transformations_applied = {
            'outliers_removed': len(outliers_result.outlier_indices) if outliers_result.user_approved_removal else 0,
            'suggestions_applied': len([s for s in suggestions if s.user_approved]),
            'feature_selection': len(selected_features) < len(X.columns)
        }
        
        summary = {
            'original_shape': df.shape,
            'final_shape': final_data.shape,
            'features_created': len(transformed_df.columns) - len(df.columns),
            'features_selected': len(selected_features),
            'outliers_detected': len(outliers_result.outlier_indices),
            'outliers_removed': len(outliers_result.outlier_indices) if outliers_result.user_approved_removal else 0
        }
        
        return TransformationResult(
            transformed_data=final_data,
            feature_names=selected_features,
            transformations_applied=transformations_applied,
            feature_importance=feature_importance,
            summary=summary,
            suggestions_made=suggestions,
            outliers_detected=outliers_result,
            user_interactions={
                'suggestions_count': len(suggestions),
                'approved_suggestions': len([s for s in suggestions if s.user_approved]),
                'outliers_detected': len(outliers_result.outlier_indices),
                'outliers_removed': len(outliers_result.outlier_indices) if outliers_result.user_approved_removal else 0
            }
        )
    
    def set_suggestion_callback(self, callback: Callable):
        """Define callback para aprovação de sugestões."""
        self.suggestion_callback = callback
    
    def set_outlier_callback(self, callback: Callable):
        """Define callback para aprovação de remoção de outliers."""
        self.outlier_callback = callback


# Manter compatibilidade com versão original
from .transformations import VariableTransformer as OriginalVariableTransformer


class VariableTransformer(OriginalVariableTransformer):
    """Wrapper que adiciona funcionalidades interativas ao transformador original."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.interactive_transformer = InteractiveDataTransformer(config)
    
    def transform_data_interactive(self, df: pd.DataFrame, target_col: str = 'valor') -> TransformationResult:
        """Versão interativa da transformação de dados."""
        return self.interactive_transformer.transform_data_interactive(df, target_col)
    
    def suggest_transformations(self, df: pd.DataFrame, target_col: str) -> List[TransformationSuggestion]:
        """Gera sugestões de transformação."""
        return self.interactive_transformer.suggest_transformations(df, target_col)
    
    def detect_outliers_interactive(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> OutlierDetectionResult:
        """Detecta outliers com aprovação interativa."""
        return self.interactive_transformer.detect_outliers_interactive(df, columns)