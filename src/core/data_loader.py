"""
Fase 1: Ingestão e Validação de Dados
Responsável por carregar, validar e preparar dados imobiliários para análise.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass


@dataclass
class DataValidationResult:
    """Resultado da validação de dados."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]


class DataLoader:
    """Carregador e validador de dados imobiliários."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carrega dados de arquivo CSV, Excel ou outras fontes.
        
        Args:
            file_path: Caminho para o arquivo de dados
            
        Returns:
            DataFrame com os dados carregados
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {file_path.suffix}")
                
            self.logger.info(f"Dados carregados: {len(df)} registros")
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Valida os dados carregados segundo critérios da NBR 14653.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            Resultado da validação
        """
        errors = []
        warnings = []
        
        # Validações obrigatórias
        required_columns = ['valor', 'area_total', 'localizacao']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Colunas obrigatórias ausentes: {missing_columns}")
        
        # Validação de valores
        if 'valor' in df.columns:
            null_values = df['valor'].isnull().sum()
            if null_values > 0:
                errors.append(f"Valores nulos encontrados na coluna 'valor': {null_values}")
                
            negative_values = (df['valor'] <= 0).sum()
            if negative_values > 0:
                errors.append(f"Valores não positivos encontrados: {negative_values}")
        
        # Validação de tamanho da amostra
        min_sample_size = self.config.get('min_sample_size', 30)
        if len(df) < min_sample_size:
            warnings.append(f"Amostra pequena: {len(df)} registros (mínimo recomendado: {min_sample_size})")
        
        # Validação de outliers
        if 'valor' in df.columns and len(df) > 0:
            Q1 = df['valor'].quantile(0.25)
            Q3 = df['valor'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df['valor'] < (Q1 - 1.5 * IQR)) | (df['valor'] > (Q3 + 1.5 * IQR))).sum()
            
            if outliers > 0:
                warnings.append(f"Possíveis outliers detectados: {outliers} registros")
        
        is_valid = len(errors) == 0
        
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        return DataValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e padroniza os dados.
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame limpo
        """
        df_clean = df.copy()
        
        # Remove duplicatas
        df_clean = df_clean.drop_duplicates()
        
        # Padroniza tipos de dados
        if 'valor' in df_clean.columns:
            df_clean['valor'] = pd.to_numeric(df_clean['valor'], errors='coerce')
            
        if 'area_total' in df_clean.columns:
            df_clean['area_total'] = pd.to_numeric(df_clean['area_total'], errors='coerce')
        
        # Remove registros com valores críticos ausentes
        df_clean = df_clean.dropna(subset=['valor'])
        
        self.logger.info(f"Dados limpos: {len(df_clean)} registros restantes")
        return df_clean