"""
Testes unitários para o módulo data_loader.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
from pathlib import Path

from src.core.data_loader import DataLoader, DataValidationResult


class TestDataValidationResult:
    """Testes para a classe DataValidationResult"""
    
    def test_init_without_recommendations(self):
        """Testa inicialização sem recomendações"""
        result = DataValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            summary={'total_rows': 100}
        )
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations is not None
        assert len(result.recommendations) >= 1
    
    def test_generate_recommendations_for_errors(self):
        """Testa geração de recomendações baseada em erros"""
        result = DataValidationResult(
            is_valid=False,
            errors=[
                "Colunas obrigatórias ausentes: area",
                "Valores não positivos encontrados",
                "Valores não numéricos encontrados"
            ],
            warnings=[],
            summary={}
        )
        
        recommendations = result._generate_recommendations()
        
        assert any("colunas obrigatórias" in rec for rec in recommendations)
        assert any("valores não positivos" in rec for rec in recommendations)
        assert any("formato numérico" in rec for rec in recommendations)
    
    def test_generate_recommendations_for_warnings(self):
        """Testa geração de recomendações baseada em warnings"""
        result = DataValidationResult(
            is_valid=True,
            errors=[],
            warnings=[
                "Amostra pequena detectada",
                "outliers encontrados",
                "cardinalidade muito alta",
                "variância zero detectada"
            ],
            summary={}
        )
        
        recommendations = result._generate_recommendations()
        
        assert any("mais dados" in rec for rec in recommendations)
        assert any("outliers" in rec for rec in recommendations)
        assert any("categorias" in rec for rec in recommendations)
        assert any("variância zero" in rec for rec in recommendations)
    
    def test_generate_recommendations_for_summary_issues(self):
        """Testa geração de recomendações baseada no summary"""
        result = DataValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            summary={
                'duplicate_rows': 5,
                'missing_values': {'area': 3, 'valor': 2}
            }
        )
        
        recommendations = result._generate_recommendations()
        
        assert any("duplicados" in rec for rec in recommendations)
        assert any("valores faltantes" in rec for rec in recommendations)
    
    def test_generate_recommendations_clean_data(self):
        """Testa recomendações para dados limpos"""
        result = DataValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            summary={'duplicate_rows': 0, 'missing_values': {}}
        )
        
        recommendations = result._generate_recommendations()
        
        assert len(recommendations) == 1
        assert "boa qualidade" in recommendations[0]


class TestDataLoader:
    """Testes para a classe DataLoader"""
    
    @pytest.fixture
    def loader(self):
        """Fixture para instância do DataLoader"""
        config = {
            'valuation_standard': 'NBR 14653',
            'min_sample_size': 30,
            'max_outlier_ratio': 0.1
        }
        return DataLoader(config)
    
    @pytest.fixture
    def sample_valid_data(self):
        """Fixture para dados válidos"""
        return pd.DataFrame({
            'valor': [400000, 500000, 350000, 600000, 450000],
            'area_privativa': [100, 120, 80, 150, 110],
            'quartos': [2, 3, 2, 4, 3],
            'banheiros': [2, 2, 1, 3, 2],
            'localizacao': ['Centro', 'Zona Sul', 'Centro', 'Zona Norte', 'Zona Sul']
        })
    
    @pytest.fixture
    def sample_invalid_data(self):
        """Fixture para dados inválidos"""
        return pd.DataFrame({
            'valor': [400000, -500000, 0, 600000, np.nan],  # Valores inválidos
            'area_privativa': [100, 120, -80, 150, 110],    # Área negativa
            'quartos': [2, 3, 2, 4, 'abc'],                 # Valor não numérico
            'localizacao': ['Centro', 'Zona Sul', None, 'Zona Norte', 'Zona Sul']
        })
    
    def test_init(self):
        """Testa inicialização do DataLoader"""
        config = {'valuation_standard': 'NBR 14653'}
        loader = DataLoader(config)
        
        assert loader.config == config
        assert loader.logger is not None
        assert loader.regional_config is not None
        assert loader.regional_config['country'] == 'Brazil'
    
    def test_get_regional_config_nbr(self):
        """Testa configuração regional para NBR 14653"""
        config = {'valuation_standard': 'NBR 14653'}
        loader = DataLoader(config)
        
        regional = loader._get_regional_config()
        
        assert regional['country'] == 'Brazil'
        assert regional['decimal_separator'] == ','
        assert regional['thousands_separator'] == '.'
        assert regional['currency_symbol'] == 'R$'
        assert 'utf-8' in regional['encoding_priority']
        assert 'valor' in regional['column_mapping']
    
    def test_get_regional_config_uspap(self):
        """Testa configuração regional para USPAP"""
        config = {'valuation_standard': 'USPAP'}
        loader = DataLoader(config)
        
        regional = loader._get_regional_config()
        
        assert regional['country'] == 'USA'
        assert regional['decimal_separator'] == '.'
        assert regional['thousands_separator'] == ','
        assert regional['currency_symbol'] == '$'
    
    def test_get_regional_config_evs(self):
        """Testa configuração regional para EVS"""
        config = {'valuation_standard': 'EVS'}
        loader = DataLoader(config)
        
        regional = loader._get_regional_config()
        
        assert regional['country'] == 'Europe'
        assert regional['currency_symbol'] == '€'
    
    def test_get_regional_config_default(self):
        """Testa configuração regional padrão"""
        config = {}  # Sem valuation_standard
        loader = DataLoader(config)
        
        regional = loader._get_regional_config()
        
        assert regional['country'] == 'Brazil'  # Default para NBR 14653
    
    @patch('magic.from_file')
    def test_detect_file_type_excel(self, mock_magic, loader):
        """Testa detecção de arquivo Excel"""
        mock_magic.return_value = 'Microsoft Excel'
        
        file_type = loader._detect_file_type('/test/file.xlsx')
        
        assert file_type == 'excel'
        mock_magic.assert_called_once_with('/test/file.xlsx', mime=True)
    
    @patch('magic.from_file')
    def test_detect_file_type_csv(self, mock_magic, loader):
        """Testa detecção de arquivo CSV"""
        mock_magic.return_value = 'text/csv'
        
        file_type = loader._detect_file_type('/test/file.csv')
        
        assert file_type == 'csv'
    
    @patch('magic.from_file')
    def test_detect_file_type_unsupported(self, mock_magic, loader):
        """Testa detecção de arquivo não suportado"""
        mock_magic.return_value = 'application/pdf'
        
        with pytest.raises(ValueError, match="Tipo de arquivo não suportado"):
            loader._detect_file_type('/test/file.pdf')
    
    def test_detect_encoding_utf8(self, loader):
        """Testa detecção de encoding UTF-8"""
        test_content = "área,preço\n100,400000\n120,500000"
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            encoding = loader._detect_encoding('/test/file.csv')
            
        assert encoding == 'utf-8'
    
    @patch('builtins.open')
    def test_detect_encoding_fallback(self, mock_open_func, loader):
        """Testa fallback de encoding"""
        # Simular erro UTF-8 e sucesso com latin1
        mock_open_func.side_effect = [
            UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'),
            mock_open(read_data="test content").__enter__()
        ]
        
        encoding = loader._detect_encoding('/test/file.csv')
        
        assert encoding == 'latin1'
    
    def test_standardize_column_names(self, loader):
        """Testa padronização de nomes de colunas"""
        df = pd.DataFrame({
            'Preço (R$)': [400000, 500000],
            'Área Útil m²': [100, 120],
            'N° Quartos': [2, 3],
            'Local / Endereço': ['Centro', 'Zona Sul']
        })
        
        standardized = loader._standardize_column_names(df)
        
        expected_columns = ['preco_r', 'area_util_m', 'n_quartos', 'local_endereco']
        assert list(standardized.columns) == expected_columns
    
    def test_map_columns_to_standard(self, loader):
        """Testa mapeamento de colunas para padrão"""
        df = pd.DataFrame({
            'preco': [400000, 500000],
            'area_construida': [100, 120],
            'dormitorios': [2, 3],
            'endereco': ['Centro', 'Zona Sul']
        })
        
        mapped = loader._map_columns_to_standard(df)
        
        assert 'valor' in mapped.columns
        assert 'area_privativa' in mapped.columns
        assert 'quartos' in mapped.columns
        assert 'localizacao' in mapped.columns
    
    def test_map_columns_partial_match(self, loader):
        """Testa mapeamento parcial de colunas"""
        df = pd.DataFrame({
            'valor': [400000, 500000],
            'area_unknown': [100, 120],  # Não mapeada
            'quartos': [2, 3],
            'other_col': ['A', 'B']      # Não mapeada
        })
        
        mapped = loader._map_columns_to_standard(df)
        
        assert 'valor' in mapped.columns
        assert 'quartos' in mapped.columns
        assert 'area_unknown' in mapped.columns  # Mantém original
        assert 'other_col' in mapped.columns     # Mantém original
    
    def test_clean_numeric_values_brazilian(self, loader):
        """Testa limpeza de valores numéricos brasileiros"""
        df = pd.DataFrame({
            'valor': ['R$ 400.000,50', '500.000,00', 'R$350000'],
            'area': ['100,5 m²', '120,0', '80']
        })
        
        cleaned = loader._clean_numeric_values(df)
        
        assert cleaned['valor'].iloc[0] == 400000.50
        assert cleaned['valor'].iloc[1] == 500000.00
        assert cleaned['valor'].iloc[2] == 350000.00
        assert cleaned['area'].iloc[0] == 100.5
        assert cleaned['area'].iloc[1] == 120.0
        assert cleaned['area'].iloc[2] == 80.0
    
    def test_clean_numeric_values_with_text(self, loader):
        """Testa limpeza de valores com texto"""
        df = pd.DataFrame({
            'valor': ['Apartamento R$ 400.000', 'Casa 500000', 'N/A'],
            'area': ['100 metros quadrados', 'abc', '120.5']
        })
        
        cleaned = loader._clean_numeric_values(df)
        
        assert cleaned['valor'].iloc[0] == 400000.0
        assert cleaned['valor'].iloc[1] == 500000.0
        assert pd.isna(cleaned['valor'].iloc[2])
        assert pd.isna(cleaned['area'].iloc[0])
        assert pd.isna(cleaned['area'].iloc[1])
        assert cleaned['area'].iloc[2] == 120.5
    
    def test_validate_required_columns_present(self, loader, sample_valid_data):
        """Testa validação com colunas obrigatórias presentes"""
        result = loader._validate_required_columns(sample_valid_data)
        
        assert result['has_all_required'] is True
        assert len(result['missing_columns']) == 0
        assert len(result['warnings']) == 0
    
    def test_validate_required_columns_missing(self, loader):
        """Testa validação com colunas obrigatórias ausentes"""
        df = pd.DataFrame({
            'valor': [400000, 500000],
            # Falta area_privativa e outras colunas
        })
        
        result = loader._validate_required_columns(df)
        
        assert result['has_all_required'] is False
        assert len(result['missing_columns']) > 0
        assert 'area_privativa' in result['missing_columns']
    
    def test_validate_data_types_valid(self, loader, sample_valid_data):
        """Testa validação de tipos com dados válidos"""
        result = loader._validate_data_types(sample_valid_data)
        
        assert result['all_types_valid'] is True
        assert len(result['type_errors']) == 0
    
    def test_validate_data_types_invalid(self, loader, sample_invalid_data):
        """Testa validação de tipos com dados inválidos"""
        result = loader._validate_data_types(sample_invalid_data)
        
        assert result['all_types_valid'] is False
        assert len(result['type_errors']) > 0
    
    def test_validate_value_ranges_valid(self, loader, sample_valid_data):
        """Testa validação de intervalos com valores válidos"""
        result = loader._validate_value_ranges(sample_valid_data)
        
        assert result['all_ranges_valid'] is True
        assert len(result['range_violations']) == 0
    
    def test_validate_value_ranges_invalid(self, loader, sample_invalid_data):
        """Testa validação de intervalos com valores inválidos"""
        result = loader._validate_value_ranges(sample_invalid_data)
        
        assert result['all_ranges_valid'] is False
        assert len(result['range_violations']) > 0
    
    def test_check_data_quality_good(self, loader, sample_valid_data):
        """Testa verificação de qualidade com dados bons"""
        result = loader._check_data_quality(sample_valid_data)
        
        assert result['overall_quality'] == 'good'
        assert result['duplicate_rows'] == 0
        assert result['missing_percentage'] < 10
    
    def test_check_data_quality_with_issues(self, loader):
        """Testa verificação de qualidade com problemas"""
        df_with_issues = pd.DataFrame({
            'valor': [400000, 400000, np.nan, 600000, 450000],  # Duplicata e NaN
            'area_privativa': [100, 100, 80, 150, np.nan],     # Duplicata e NaN
            'quartos': [2, 2, 2, 4, 3],
            'localizacao': ['Centro', 'Centro', 'Centro', 'Norte', 'Sul']
        })
        
        result = loader._check_data_quality(df_with_issues)
        
        assert result['duplicate_rows'] > 0
        assert result['missing_percentage'] > 0
        assert result['overall_quality'] in ['poor', 'fair']
    
    def test_detect_outliers_iqr(self, loader):
        """Testa detecção de outliers usando IQR"""
        df = pd.DataFrame({
            'valor': [100, 200, 300, 400, 500, 1000000]  # 1000000 é outlier
        })
        
        outliers = loader._detect_outliers(df, 'valor', method='iqr')
        
        assert len(outliers) == 1
        assert 1000000 in outliers
    
    def test_detect_outliers_zscore(self, loader):
        """Testa detecção de outliers usando Z-score"""
        df = pd.DataFrame({
            'valor': [100, 200, 300, 400, 500, 2000]  # 2000 pode ser outlier
        })
        
        outliers = loader._detect_outliers(df, 'valor', method='zscore', threshold=2)
        
        assert isinstance(outliers, list)
    
    def test_perform_comprehensive_validation_valid(self, loader, sample_valid_data):
        """Testa validação completa com dados válidos"""
        result = loader.perform_comprehensive_validation(sample_valid_data)
        
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert 'total_rows' in result.summary
        assert 'total_columns' in result.summary
    
    def test_perform_comprehensive_validation_invalid(self, loader, sample_invalid_data):
        """Testa validação completa com dados inválidos"""
        result = loader.perform_comprehensive_validation(sample_invalid_data)
        
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert len(result.recommendations) > 0
    
    def test_perform_comprehensive_validation_small_sample(self, loader):
        """Testa validação com amostra pequena"""
        small_df = pd.DataFrame({
            'valor': [400000, 500000],  # Apenas 2 registros
            'area_privativa': [100, 120],
            'quartos': [2, 3],
            'localizacao': ['Centro', 'Sul']
        })
        
        result = loader.perform_comprehensive_validation(small_df)
        
        assert any('Amostra pequena' in warning for warning in result.warnings)
    
    @patch('pandas.read_excel')
    def test_load_from_file_excel(self, mock_read_excel, loader):
        """Testa carregamento de arquivo Excel"""
        mock_df = pd.DataFrame({'valor': [400000], 'area': [100]})
        mock_read_excel.return_value = mock_df
        
        with patch.object(loader, '_detect_file_type', return_value='excel'):
            df = loader._load_from_file('/test/file.xlsx')
        
        assert isinstance(df, pd.DataFrame)
        mock_read_excel.assert_called_once()
    
    @patch('pandas.read_csv')
    def test_load_from_file_csv(self, mock_read_csv, loader):
        """Testa carregamento de arquivo CSV"""
        mock_df = pd.DataFrame({'valor': [400000], 'area': [100]})
        mock_read_csv.return_value = mock_df
        
        with patch.object(loader, '_detect_file_type', return_value='csv'):
            with patch.object(loader, '_detect_encoding', return_value='utf-8'):
                df = loader._load_from_file('/test/file.csv')
        
        assert isinstance(df, pd.DataFrame)
        mock_read_csv.assert_called_once()
    
    def test_load_from_file_not_exists(self, loader):
        """Testa carregamento de arquivo inexistente"""
        with pytest.raises(FileNotFoundError):
            loader._load_from_file('/nonexistent/file.xlsx')
    
    @patch.object(DataLoader, '_load_from_file')
    def test_load_data_from_file(self, mock_load, loader, sample_valid_data):
        """Testa carregamento completo de dados de arquivo"""
        mock_load.return_value = sample_valid_data
        
        result = loader.load_data('/test/file.xlsx')
        
        assert isinstance(result, tuple)
        df, validation_result = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(validation_result, DataValidationResult)
        mock_load.assert_called_once_with('/test/file.xlsx')
    
    def test_load_data_from_dataframe(self, loader, sample_valid_data):
        """Testa carregamento de dados de DataFrame"""
        result = loader.load_data(sample_valid_data)
        
        assert isinstance(result, tuple)
        df, validation_result = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(validation_result, DataValidationResult)
        assert len(df) == len(sample_valid_data)
    
    def test_load_data_invalid_input(self, loader):
        """Testa carregamento com entrada inválida"""
        with pytest.raises(TypeError, match="deve ser um DataFrame ou caminho para arquivo"):
            loader.load_data(123)  # Tipo inválido
    
    def test_preprocessing_pipeline(self, loader):
        """Testa pipeline completo de pré-processamento"""
        raw_df = pd.DataFrame({
            'Preço': ['R$ 400.000,00', 'R$ 500.000,50'],
            'Área Construída': ['100,5 m²', '120,0 m²'],
            'Dormitórios': [2, 3],
            'Endereço': ['Centro SP', 'Zona Sul SP']
        })
        
        processed_df = loader._preprocessing_pipeline(raw_df)
        
        assert 'valor' in processed_df.columns
        assert 'area_privativa' in processed_df.columns
        assert 'quartos' in processed_df.columns
        assert 'localizacao' in processed_df.columns
        
        # Verificar conversão numérica
        assert processed_df['valor'].iloc[0] == 400000.00
        assert processed_df['valor'].iloc[1] == 500000.50
        assert processed_df['area_privativa'].iloc[0] == 100.5
        assert processed_df['area_privativa'].iloc[1] == 120.0


@pytest.mark.integration
class TestDataLoaderIntegration:
    """Testes de integração para DataLoader"""
    
    def test_load_real_excel_file(self):
        """Testa carregamento de arquivo Excel real"""
        # Criar arquivo Excel temporário
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            df = pd.DataFrame({
                'Valor (R$)': [400000, 500000, 350000],
                'Área (m²)': [100, 120, 80],
                'Quartos': [2, 3, 2],
                'Localização': ['Centro', 'Zona Sul', 'Centro']
            })
            df.to_excel(temp_file.name, index=False)
            temp_path = temp_file.name
        
        try:
            config = {'valuation_standard': 'NBR 14653'}
            loader = DataLoader(config)
            
            result_df, validation_result = loader.load_data(temp_path)
            
            assert isinstance(result_df, pd.DataFrame)
            assert isinstance(validation_result, DataValidationResult)
            assert len(result_df) == 3
            assert 'valor' in result_df.columns
            
        finally:
            os.unlink(temp_path)
    
    def test_load_real_csv_file(self):
        """Testa carregamento de arquivo CSV real"""
        # Criar arquivo CSV temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
            temp_file.write('valor,area_privativa,quartos,localizacao\n')
            temp_file.write('400000,100,2,Centro\n')
            temp_file.write('500000,120,3,Zona Sul\n')
            temp_file.write('350000,80,2,Centro\n')
            temp_path = temp_file.name
        
        try:
            config = {'valuation_standard': 'NBR 14653'}
            loader = DataLoader(config)
            
            result_df, validation_result = loader.load_data(temp_path)
            
            assert isinstance(result_df, pd.DataFrame)
            assert isinstance(validation_result, DataValidationResult)
            assert len(result_df) == 3
            assert 'valor' in result_df.columns
            
        finally:
            os.unlink(temp_path)
    
    def test_end_to_end_validation_workflow(self):
        """Testa fluxo completo de validação"""
        config = {
            'valuation_standard': 'NBR 14653',
            'min_sample_size': 3,
            'max_outlier_ratio': 0.2
        }
        loader = DataLoader(config)
        
        # Dados com vários problemas
        problematic_df = pd.DataFrame({
            'valor': [400000, -500000, np.nan, 600000, 1000000000],  # Negativo, NaN, outlier
            'area_privativa': [100, 120, 80, 150, 200],
            'quartos': [2, 'três', 2, 4, 3],                        # Valor não numérico
            'banheiros': [2, 2, 1, 3, 2],
            'localizacao': ['Centro', 'Centro', None, 'Norte', 'Sul']  # Valor faltante
        })
        
        result_df, validation_result = loader.load_data(problematic_df)
        
        assert isinstance(validation_result, DataValidationResult)
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
        assert len(validation_result.warnings) > 0
        assert len(validation_result.recommendations) > 0
        
        # Verificar que problemas foram detectados
        assert any('não positivos' in error for error in validation_result.errors)
        assert any('não numéricos' in error for error in validation_result.errors)
        assert validation_result.summary['missing_values']['localizacao'] > 0