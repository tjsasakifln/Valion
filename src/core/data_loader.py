"""
Phase 1: Data Ingestion and Validation
Responsible for loading, validating and preparing real estate data for analysis.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import csv
import magic
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings


@dataclass
class DataValidationResult:
    """Enhanced data validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]
    recommendations: Optional[List[str]] = None
    
    def __post_init__(self):
        """Gera recomendações automáticas baseadas nos resultados."""
        if self.recommendations is None:
            self.recommendations = self._generate_recommendations()
    
    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas nos erros e warnings."""
        recommendations = []
        
        # Baseado nos erros
        for error in self.errors:
            if "obrigatórias ausentes" in error:
                recommendations.append("Adicione as colunas obrigatórias ao dataset ou mapeie colunas existentes")
            elif "não positivos" in error:
                recommendations.append("Remova ou corrija registros com valores não positivos")
            elif "não numéricos" in error:
                recommendations.append("Padronize formato numérico das colunas ou remova caracteres inválidos")
        
        # Baseado nos warnings
        for warning in self.warnings:
            if "Amostra pequena" in warning:
                recommendations.append("Considere coletar mais dados para melhorar a confiabilidade do modelo")
            elif "outliers" in warning:
                recommendations.append("Analise outliers para decidir se devem ser mantidos ou removidos")
            elif "cardinalidade muito alta" in warning:
                recommendations.append("Considere agrupar categorias menos frequentes ou usar encoding especial")
            elif "variância zero" in warning:
                recommendations.append("Remova colunas com variância zero pois não contribuem para o modelo")
        
        # Recomendações gerais baseadas no summary
        if 'duplicate_rows' in self.summary and self.summary['duplicate_rows'] > 0:
            recommendations.append("Remova registros duplicados para evitar viés no modelo")
        
        if 'missing_values' in self.summary:
            total_missing = sum(self.summary['missing_values'].values())
            if total_missing > 0:
                recommendations.append("Implemente estratégia de tratamento para valores faltantes")
        
        return recommendations if recommendations else ["Dados em boa qualidade - prossiga para modelagem"]


class DataLoader:
    """Carregador e validador de dados imobiliários."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def _verify_file_integrity(self, file_path: Path) -> Dict[str, Any]:
        """
        Verifica integridade e tipo real do arquivo usando magic bytes.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Informações sobre o arquivo verificado
        """
        try:
            # Verificar se arquivo existe
            if not file_path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
            # Verificar tamanho do arquivo
            file_size = file_path.stat().st_size
            max_size = self.config.get('max_file_size_mb', 100) * 1024 * 1024  # 100MB padrão
            
            if file_size > max_size:
                raise ValueError(f"Arquivo muito grande: {file_size / (1024*1024):.1f}MB. Máximo: {max_size / (1024*1024)}MB")
            
            if file_size == 0:
                raise ValueError("Arquivo está vazio")
            
            # Detectar tipo real do arquivo usando magic bytes
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
            except:
                # Fallback se python-magic não estiver disponível
                mime_type = "application/octet-stream"
            
            expected_types = {
                '.csv': ['text/csv', 'text/plain', 'application/csv'],
                '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
                '.xls': ['application/vnd.ms-excel', 'application/x-ole-storage']
            }
            
            extension = file_path.suffix.lower()
            if extension in expected_types:
                valid_mime = mime_type in expected_types[extension] or mime_type.startswith('text/')
                if not valid_mime:
                    self.logger.warning(f"Possível incompatibilidade: extensão {extension} mas MIME type {mime_type}")
            
            return {
                'file_size': file_size,
                'mime_type': mime_type,
                'extension': extension,
                'integrity_ok': True
            }
            
        except Exception as e:
            self.logger.error(f"Erro na verificação de integridade: {str(e)}")
            raise
    
    def _detect_csv_delimiter(self, file_path: Path, sample_size: int = 1024) -> str:
        """
        Detecta automaticamente o delimitador de arquivo CSV.
        
        Args:
            file_path: Caminho para arquivo CSV
            sample_size: Tamanho da amostra para análise
            
        Returns:
            Delimitador detectado
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                sample = file.read(sample_size)
            
            # Usar csv.Sniffer para detectar delimitador
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample, delimiters=',;\t|').delimiter
            
            self.logger.info(f"Delimitador detectado: '{delimiter}'")
            return delimiter
            
        except Exception as e:
            self.logger.warning(f"Erro na detecção do delimitador: {str(e)}. Usando vírgula como padrão.")
            return ','
    
    def _handle_excel_sheets(self, file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Lida com arquivos Excel de múltiplas abas.
        
        Args:
            file_path: Caminho para arquivo Excel
            
        Returns:
            DataFrame da aba selecionada e metadados das abas
        """
        try:
            # Carregar informações sobre todas as abas
            xl_file = pd.ExcelFile(file_path)
            sheet_names = xl_file.sheet_names
            
            sheets_info = {}
            for sheet_name in sheet_names:
                try:
                    df_temp = pd.read_excel(xl_file, sheet_name=sheet_name, nrows=5)
                    sheets_info[sheet_name] = {
                        'rows_sample': len(df_temp),
                        'columns': len(df_temp.columns),
                        'column_names': list(df_temp.columns)
                    }
                except Exception as e:
                    sheets_info[sheet_name] = {'error': str(e)}
            
            # Se há múltiplas abas, usar a primeira por padrão, mas notificar
            if len(sheet_names) > 1:
                selected_sheet = sheet_names[0]
                self.logger.info(f"Arquivo Excel com {len(sheet_names)} abas. Usando '{selected_sheet}' por padrão.")
                self.logger.info(f"Abas disponíveis: {sheet_names}")
            else:
                selected_sheet = sheet_names[0]
            
            # Carregar dados da aba selecionada
            df = pd.read_excel(xl_file, sheet_name=selected_sheet)
            
            metadata = {
                'total_sheets': len(sheet_names),
                'selected_sheet': selected_sheet,
                'available_sheets': sheet_names,
                'sheets_info': sheets_info
            }
            
            return df, metadata
            
        except Exception as e:
            self.logger.error(f"Erro ao processar arquivo Excel: {str(e)}")
            raise
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carrega dados de arquivo CSV, Excel ou outras fontes com verificações robustas.
        
        Args:
            file_path: Caminho para o arquivo de dados
            
        Returns:
            DataFrame com os dados carregados
        """
        try:
            file_path = Path(file_path)
            
            # Verificar integridade do arquivo
            file_info = self._verify_file_integrity(file_path)
            self.logger.info(f"Arquivo verificado: {file_info['file_size']} bytes, MIME: {file_info['mime_type']}")
            
            # Carregar dados baseado na extensão
            if file_path.suffix.lower() == '.csv':
                # Detectar delimitador automaticamente
                delimiter = self._detect_csv_delimiter(file_path)
                
                # Tentar diferentes encodings se necessário
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                        self.logger.info(f"CSV carregado com encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise ValueError("Não foi possível decodificar o arquivo CSV com encodings suportados")
                    
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df, excel_metadata = self._handle_excel_sheets(file_path)
                
                # Armazenar metadados do Excel para uso posterior
                if not hasattr(self, 'file_metadata'):
                    self.file_metadata = {}
                self.file_metadata[str(file_path)] = excel_metadata
                
            else:
                raise ValueError(f"Formato de arquivo não suportado: {file_path.suffix}")
            
            # Verificação básica dos dados carregados
            if df.empty:
                raise ValueError("Arquivo carregado não contém dados")
            
            if len(df.columns) == 0:
                raise ValueError("Arquivo não contém colunas válidas")
            
            # Log de sucesso com estatísticas
            self.logger.info(f"Dados carregados com sucesso: {len(df)} registros, {len(df.columns)} colunas")
            
            # Detectar e reportar colunas sem nome
            unnamed_cols = [col for col in df.columns if 'Unnamed:' in str(col)]
            if unnamed_cols:
                self.logger.warning(f"Colunas sem nome detectadas: {unnamed_cols}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def _validate_mixed_data_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Valida tipos de dados mistos em colunas que deveriam ser numéricas.
        
        Args:
            df: DataFrame para validar
            
        Returns:
            Tupla com (erros, warnings) encontrados
        """
        errors = []
        warnings = []
        
        # Colunas que deveriam ser numéricas
        numeric_columns = ['valor', 'area_total', 'area_privativa', 'area_construida', 'vagas_garagem', 'quartos', 'banheiros']
        numeric_columns = [col for col in numeric_columns if col in df.columns]
        
        for col in numeric_columns:
            original_series = df[col].copy()
            
            # Tentar converter para numérico
            converted_series = pd.to_numeric(original_series, errors='coerce')
            
            # Detectar novos valores NaN (eram texto antes)
            new_nans = converted_series.isnull() & original_series.notnull()
            mixed_count = new_nans.sum()
            
            if mixed_count > 0:
                # Encontrar exemplos dos valores problemáticos
                problematic_values = original_series[new_nans].unique()[:5]
                
                if mixed_count / len(df) > 0.1:  # Mais de 10% problemáticos
                    errors.append(
                        f"Coluna '{col}': {mixed_count} valores não numéricos detectados "
                        f"({mixed_count/len(df)*100:.1f}%). Exemplos: {list(problematic_values)}"
                    )
                else:
                    warnings.append(
                        f"Coluna '{col}': {mixed_count} valores não numéricos encontrados. "
                        f"Exemplos: {list(problematic_values)}"
                    )
        
        return errors, warnings
    
    def _validate_zero_variance(self, df: pd.DataFrame) -> List[str]:
        """
        Identifica colunas numéricas com variância zero (todos os valores iguais).
        
        Args:
            df: DataFrame para validar
            
        Returns:
            Lista de warnings sobre colunas com variância zero
        """
        warnings = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].var() == 0 and not df[col].isnull().all():
                unique_val = df[col].iloc[0]
                warnings.append(
                    f"Coluna '{col}' tem variância zero (todos os valores = {unique_val}). "
                    "Esta coluna não será útil para modelagem."
                )
        
        return warnings
    
    def _validate_high_cardinality(self, df: pd.DataFrame) -> List[str]:
        """
        Identifica colunas categóricas com cardinalidade muito alta.
        
        Args:
            df: DataFrame para validar
            
        Returns:
            Lista de warnings sobre alta cardinalidade
        """
        warnings = []
        
        # Identificar colunas categóricas (não numéricas)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            total_count = len(df)
            
            # Se a cardinalidade é muito alta (>50% dos registros são únicos)
            if unique_count > min(100, total_count * 0.5):
                warnings.append(
                    f"Coluna '{col}' tem cardinalidade muito alta: {unique_count} valores únicos "
                    f"em {total_count} registros ({unique_count/total_count*100:.1f}%). "
                    "Considere agrupar categorias ou usar técnicas de encoding especiais."
                )
            
            # Verificar se há categorias dominantes
            value_counts = df[col].value_counts()
            if len(value_counts) > 0:
                most_common_pct = value_counts.iloc[0] / total_count * 100
                if most_common_pct > 90:
                    warnings.append(
                        f"Coluna '{col}': categoria dominante '{value_counts.index[0]}' "
                        f"representa {most_common_pct:.1f}% dos dados. Variabilidade limitada."
                    )
        
        return warnings
    
    def _validate_data_quality_advanced(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Validações avançadas de qualidade de dados.
        
        Args:
            df: DataFrame para validar
            
        Returns:
            Tupla com (erros, warnings) de validações avançadas
        """
        errors = []
        warnings = []
        
        # Detectar colunas completamente vazias
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        if empty_cols:
            warnings.append(f"Colunas completamente vazias: {empty_cols}")
        
        # Detectar duplicatas completas
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Registros duplicados encontrados: {duplicate_count}")
        
        # Validar consistência de dados relacionados
        if 'area_total' in df.columns and 'area_privativa' in df.columns:
            inconsistent = (df['area_privativa'] > df['area_total']).sum()
            if inconsistent > 0:
                errors.append(f"Inconsistência: {inconsistent} registros com área privativa > área total")
        
        # Validar ranges razoáveis para imóveis
        if 'valor' in df.columns:
            df_numeric = pd.to_numeric(df['valor'], errors='coerce')
            min_val, max_val = df_numeric.min(), df_numeric.max()
            
            if min_val < 1000:  # Muito baixo
                warnings.append(f"Valores muito baixos detectados (mínimo: R$ {min_val:.2f})")
            
            if max_val > 50000000:  # Muito alto para residencial típico
                warnings.append(f"Valores muito altos detectados (máximo: R$ {max_val:,.2f})")
        
        return errors, warnings
    
    def validate_data(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Valida os dados carregados com verificações robustas e abrangentes.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            Resultado da validação aprimorado
        """
        errors = []
        warnings = []
        
        # Validações obrigatórias básicas
        required_columns = ['valor', 'area_total', 'localizacao']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Colunas obrigatórias ausentes: {missing_columns}")
        
        # Validação de valores básica
        if 'valor' in df.columns:
            null_values = df['valor'].isnull().sum()
            if null_values > 0:
                if null_values / len(df) > 0.1:  # Mais de 10% nulos
                    errors.append(f"Muitos valores nulos na coluna 'valor': {null_values} ({null_values/len(df)*100:.1f}%)")
                else:
                    warnings.append(f"Valores nulos encontrados na coluna 'valor': {null_values}")
                
            # Converter para numérico para validações
            valor_numeric = pd.to_numeric(df['valor'], errors='coerce')
            negative_values = (valor_numeric <= 0).sum()
            if negative_values > 0:
                errors.append(f"Valores não positivos encontrados: {negative_values}")
        
        # Validações avançadas de tipos de dados mistos
        mixed_errors, mixed_warnings = self._validate_mixed_data_types(df)
        errors.extend(mixed_errors)
        warnings.extend(mixed_warnings)
        
        # Validação de variância zero
        variance_warnings = self._validate_zero_variance(df)
        warnings.extend(variance_warnings)
        
        # Validação de alta cardinalidade
        cardinality_warnings = self._validate_high_cardinality(df)
        warnings.extend(cardinality_warnings)
        
        # Validações avançadas de qualidade
        quality_errors, quality_warnings = self._validate_data_quality_advanced(df)
        errors.extend(quality_errors)
        warnings.extend(quality_warnings)
        
        # Validação de tamanho da amostra
        min_sample_size = self.config.get('min_sample_size', 30)
        if len(df) < min_sample_size:
            warnings.append(f"Amostra pequena: {len(df)} registros (mínimo recomendado: {min_sample_size})")
        
        # Validação de outliers melhorada
        if 'valor' in df.columns and len(df) > 4:  # Precisamos de pelo menos 5 valores para quartis
            valor_clean = pd.to_numeric(df['valor'], errors='coerce').dropna()
            if len(valor_clean) > 4:
                Q1 = valor_clean.quantile(0.25)
                Q3 = valor_clean.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Evitar divisão por zero
                    outliers_mask = (valor_clean < (Q1 - 1.5 * IQR)) | (valor_clean > (Q3 + 1.5 * IQR))
                    outliers = outliers_mask.sum()
                    
                    if outliers > 0:
                        outlier_pct = outliers / len(valor_clean) * 100
                        if outlier_pct > 15:  # Muitos outliers
                            warnings.append(f"Muitos outliers detectados: {outliers} registros ({outlier_pct:.1f}%). Revise os dados.")
                        else:
                            warnings.append(f"Outliers detectados: {outliers} registros ({outlier_pct:.1f}%)")
        
        is_valid = len(errors) == 0
        
        # Summary aprimorado
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Adicionar estatísticas descritivas para colunas numéricas principais
        if 'valor' in df.columns:
            valor_stats = pd.to_numeric(df['valor'], errors='coerce').describe()
            summary['valor_statistics'] = valor_stats.to_dict()
        
        return DataValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            summary=summary
        )
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e padroniza os dados com processo robusto.
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame limpo e padronizado
        """
        df_clean = df.copy()
        original_count = len(df_clean)
        
        self.logger.info(f"Iniciando limpeza de dados: {original_count} registros")
        
        # 1. Remove colunas completamente vazias
        empty_cols = [col for col in df_clean.columns if df_clean[col].isnull().all()]
        if empty_cols:
            df_clean = df_clean.drop(columns=empty_cols)
            self.logger.info(f"Removidas colunas vazias: {empty_cols}")
        
        # 2. Remove duplicatas completas
        duplicates_before = df_clean.duplicated().sum()
        if duplicates_before > 0:
            df_clean = df_clean.drop_duplicates()
            self.logger.info(f"Removidas {duplicates_before} linhas duplicadas")
        
        # 3. Padroniza tipos de dados numéricas principais
        numeric_columns = ['valor', 'area_total', 'area_privativa', 'area_construida', 'vagas_garagem', 'quartos', 'banheiros']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                original_series = df_clean[col].copy()
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Log de conversões que falharam
                conversion_failures = df_clean[col].isnull() & original_series.notnull()
                if conversion_failures.any():
                    failed_count = conversion_failures.sum()
                    self.logger.warning(f"Coluna '{col}': {failed_count} valores não puderam ser convertidos para numérico")
        
        # 4. Padroniza strings categóricas
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'valor':  # Não processar valor como string
                # Remove espaços extras e padroniza capitalização
                df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
                
                # Substitui valores claramente nulos por NaN
                null_values = ['', 'Nan', 'None', 'Null', '-', 'N/A', 'NA']
                df_clean[col] = df_clean[col].replace(null_values, np.nan)
        
        # 5. Remove registros com valores críticos ausentes
        critical_columns = ['valor']  # Colunas essenciais que não podem ser nulas
        
        for col in critical_columns:
            if col in df_clean.columns:
                before_count = len(df_clean)
                df_clean = df_clean.dropna(subset=[col])
                removed = before_count - len(df_clean)
                if removed > 0:
                    self.logger.info(f"Removidos {removed} registros com '{col}' nulo")
        
        # 6. Remove valores obviamente inválidos
        if 'valor' in df_clean.columns:
            # Remove valores não positivos
            before_count = len(df_clean)
            df_clean = df_clean[df_clean['valor'] > 0]
            removed = before_count - len(df_clean)
            if removed > 0:
                self.logger.info(f"Removidos {removed} registros com valor <= 0")
        
        # 7. Verificações de consistência
        if 'area_total' in df_clean.columns and 'area_privativa' in df_clean.columns:
            # Remove registros onde área privativa > área total
            inconsistent = df_clean['area_privativa'] > df_clean['area_total']
            if inconsistent.any():
                before_count = len(df_clean)
                df_clean = df_clean[~inconsistent]
                removed = before_count - len(df_clean)
                self.logger.warning(f"Removidos {removed} registros com área privativa > área total")
        
        # 8. Relatório final
        final_count = len(df_clean)
        removed_total = original_count - final_count
        removal_pct = (removed_total / original_count) * 100 if original_count > 0 else 0
        
        self.logger.info(
            f"Limpeza concluída: {final_count} registros restantes "
            f"({removed_total} removidos, {removal_pct:.1f}%)"
        )
        
        # Verificação final
        if final_count == 0:
            raise ValueError("Limpeza resultou em dataset vazio. Verifique a qualidade dos dados originais.")
        
        if removal_pct > 50:
            self.logger.warning(
                f"Mais de 50% dos dados foram removidos na limpeza ({removal_pct:.1f}%). "
                "Revise a qualidade dos dados de entrada."
            )
        
        return df_clean