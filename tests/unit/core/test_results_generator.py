"""
Testes unitários para o módulo results_generator.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, Any

from src.core.results_generator import ResultsGenerator, EvaluationReport


@dataclass
class MockValidationResult:
    """Mock para resultado de validação"""
    is_valid: bool = True
    errors: list = None
    warnings: list = None
    summary: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.summary is None:
            self.summary = {
                'missing_values': {'area': 0, 'valor': 2},
                'data_types': {'area': 'float64', 'valor': 'float64'}
            }


@dataclass
class MockTransformationResult:
    """Mock para resultado de transformação"""
    transformations_applied: list
    feature_names: list
    feature_importance: Dict[str, float]
    
    def __init__(self):
        self.transformations_applied = ['normalization', 'feature_selection']
        self.feature_names = ['area', 'quartos', 'localizacao_score']
        self.feature_importance = {
            'area': 0.45,
            'quartos': 0.30,
            'localizacao_score': 0.25
        }


@dataclass
class MockPerformance:
    """Mock para performance do modelo"""
    r2_score: float = 0.85
    rmse: float = 45000.0
    mae: float = 35000.0
    mape: float = 12.5
    cv_scores: list = None
    feature_coefficients: Dict[str, float] = None
    permutation_importance: Dict[str, float] = None
    shap_feature_importance: Dict[str, float] = None
    
    def __post_init__(self):
        if self.cv_scores is None:
            self.cv_scores = [42000, 48000, 43000, 46000, 44000]
        if self.feature_coefficients is None:
            self.feature_coefficients = {
                'area': 0.65,
                'quartos': 0.25,
                'localizacao_score': 0.45
            }
        if self.permutation_importance is None:
            self.permutation_importance = {
                'area': 0.62,
                'quartos': 0.28,
                'localizacao_score': 0.43
            }
        if self.shap_feature_importance is None:
            self.shap_feature_importance = {
                'area': 0.58,
                'quartos': 0.31,
                'localizacao_score': 0.41
            }


@dataclass
class MockModelResult:
    """Mock para resultado do modelo"""
    model_type: str = 'elastic_net'
    best_params: Dict[str, Any] = None
    training_summary: Dict[str, Any] = None
    performance: MockPerformance = None
    model: Mock = None
    n_train_samples: int = 1000
    
    def __post_init__(self):
        if self.best_params is None:
            self.best_params = {'alpha': 0.1, 'l1_ratio': 0.5}
        if self.training_summary is None:
            self.training_summary = {'training_time': 2.5, 'iterations': 100}
        if self.performance is None:
            self.performance = MockPerformance()
        if self.model is None:
            self.model = Mock()
            self.model.predict.return_value = np.array([400000, 500000, 350000])


@dataclass 
class MockIndividualTest:
    """Mock para teste individual"""
    test_name: str
    passed: bool
    value: float
    threshold: float
    description: str


@dataclass
class MockValidationTestResult:
    """Mock para resultado de validação NBR/USPAP/EVS"""
    standard: str = 'NBR 14653'
    overall_grade: str = 'Superior'
    compliance_score: float = 0.85
    summary: Dict[str, Any] = None
    individual_tests: list = None
    
    def __post_init__(self):
        if self.summary is None:
            self.summary = {'total_tests': 5, 'passed': 4}
        if self.individual_tests is None:
            self.individual_tests = [
                MockIndividualTest('Teste R²', True, 0.85, 0.80, 'R² adequado'),
                MockIndividualTest('Teste F', True, 25.5, 4.0, 'Significância global'),
                MockIndividualTest('Durbin-Watson', False, 1.2, 1.5, 'Autocorrelação'),
                MockIndividualTest('Normalidade', True, 0.95, 0.05, 'Resíduos normais')
            ]


class TestResultsGenerator:
    """Testes para a classe ResultsGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Fixture para instância do ResultsGenerator"""
        config = {
            'expert_mode': True,
            'output_format': 'json'
        }
        return ResultsGenerator(config)
    
    @pytest.fixture
    def sample_cleaned_data(self):
        """Fixture para dados limpos de exemplo"""
        return pd.DataFrame({
            'valor': [400000, 500000, 350000, 600000, 450000],
            'area': [100, 120, 80, 150, 110],
            'quartos': [2, 3, 2, 4, 3],
            'localizacao': ['Centro', 'Centro', 'Zona Sul', 'Centro', 'Zona Norte'],
            'data_transacao': ['2023-01-15', '2023-02-20', '2023-01-10', '2023-03-05', '2023-02-28']
        })
    
    def test_init(self):
        """Testa inicialização do ResultsGenerator"""
        config = {'expert_mode': True}
        generator = ResultsGenerator(config)
        
        assert generator.config == config
        assert generator.logger is not None
    
    def test_generate_data_summary(self, generator, sample_cleaned_data):
        """Testa geração do resumo de dados"""
        validation_result = MockValidationResult()
        
        summary = generator.generate_data_summary(validation_result, sample_cleaned_data)
        
        assert summary['total_records'] == 5
        assert summary['variables_count'] == 5
        assert summary['data_quality']['validation_passed'] is True
        assert summary['data_quality']['errors_count'] == 0
        assert 'descriptive_statistics' in summary
        assert 'valor' in summary['descriptive_statistics']
        assert summary['descriptive_statistics']['valor']['mean'] == 460000.0
        assert summary['descriptive_statistics']['valor']['median'] == 450000.0
    
    def test_analyze_geographic_distribution(self, generator, sample_cleaned_data):
        """Testa análise de distribuição geográfica"""
        result = generator._analyze_geographic_distribution(sample_cleaned_data)
        
        assert result['locations_count'] == 3
        assert 'Centro' in result['distribution']
        assert result['distribution']['Centro'] == 3
        assert 'concentration_index' in result
        assert 0 <= result['concentration_index'] <= 1
    
    def test_analyze_temporal_distribution(self, generator, sample_cleaned_data):
        """Testa análise de distribuição temporal"""
        result = generator._analyze_temporal_distribution(sample_cleaned_data)
        
        assert result['years_span'] == 1
        assert 2023 in result['distribution']
        assert result['distribution'][2023] == 5
        assert 'most_recent' in result
        assert 'oldest' in result
    
    def test_calculate_concentration_index(self, generator):
        """Testa cálculo do índice de concentração"""
        # Distribuição uniforme
        uniform_dist = {'A': 25, 'B': 25, 'C': 25, 'D': 25}
        hhi_uniform = generator._calculate_concentration_index(uniform_dist)
        assert 0.2 < hhi_uniform < 0.3  # Baixa concentração
        
        # Distribuição concentrada
        concentrated_dist = {'A': 90, 'B': 5, 'C': 3, 'D': 2}
        hhi_concentrated = generator._calculate_concentration_index(concentrated_dist)
        assert hhi_concentrated > 0.8  # Alta concentração
        
        # Distribuição vazia
        empty_dist = {}
        hhi_empty = generator._calculate_concentration_index(empty_dist)
        assert hhi_empty == 0
    
    def test_generate_transformation_summary(self, generator):
        """Testa geração do resumo de transformações"""
        transformation_result = MockTransformationResult()
        
        summary = generator.generate_transformation_summary(transformation_result)
        
        assert summary['transformations_applied'] == ['normalization', 'feature_selection']
        assert summary['final_features_count'] == 3
        assert summary['feature_names'] == ['area', 'quartos', 'localizacao_score']
        assert len(summary['top_features']) <= 10
        assert summary['top_features'][0][0] == 'area'  # Feature mais importante
        assert summary['top_features'][0][1] == 0.45   # Valor da importância
    
    def test_generate_model_summary(self, generator):
        """Testa geração do resumo do modelo"""
        model_result = MockModelResult()
        
        summary = generator.generate_model_summary(model_result)
        
        assert summary['model_type'] == 'Elastic Net Regression'
        assert summary['model_type_code'] == 'elastic_net'
        assert summary['hyperparameters'] == {'alpha': 0.1, 'l1_ratio': 0.5}
        assert 'performance_metrics' in summary
        assert summary['performance_metrics']['r2_score'] == 0.85
        assert summary['performance_metrics']['rmse'] == 45000.0
        assert 'shap_analysis' in summary  # Expert mode ativo
        assert 'interpretability_analysis' in summary
    
    def test_generate_model_summary_without_expert_mode(self):
        """Testa resumo do modelo sem modo expert"""
        config = {'expert_mode': False}
        generator = ResultsGenerator(config)
        model_result = MockModelResult()
        
        summary = generator.generate_model_summary(model_result)
        
        # Não deve ter erro SHAP no modo normal
        assert 'shap_error' not in summary
    
    def test_get_significant_features_elastic_net(self, generator):
        """Testa identificação de features significativas para Elastic Net"""
        feature_coefficients = {
            'area': 0.05,      # Significativa
            'quartos': 0.005,  # Não significativa
            'banheiros': -0.02, # Significativa
            'vagas': 0.001     # Não significativa
        }
        
        significant = generator._get_significant_features(feature_coefficients, 'elastic_net')
        
        assert 'area' in significant
        assert 'banheiros' in significant
        assert 'quartos' not in significant
        assert 'vagas' not in significant
    
    def test_get_significant_features_tree_based(self, generator):
        """Testa identificação de features significativas para modelos baseados em árvore"""
        feature_coefficients = {
            'area': 0.45,
            'quartos': 0.30,
            'banheiros': 0.15,
            'vagas': 0.10
        }
        
        significant = generator._get_significant_features(feature_coefficients, 'xgboost')
        
        # Para modelos baseados em árvore, pega top 50% ou pelo menos 10
        assert len(significant) >= 2  # Top 50%
        assert 'area' in significant
        assert 'quartos' in significant
    
    def test_generate_interpretability_analysis(self, generator):
        """Testa geração da análise de interpretabilidade"""
        performance = MockPerformance()
        
        analysis = generator._generate_interpretability_analysis(performance)
        
        assert analysis['glass_box_level'] == 'high'
        assert 'feature_coefficients' in analysis['available_explanations']
        assert 'permutation_importance' in analysis['available_explanations']
        assert 'shap_values' in analysis['available_explanations']
        assert 'explanation_consistency' in analysis
    
    def test_generate_shap_analysis(self, generator):
        """Testa geração da análise SHAP"""
        performance = MockPerformance()
        
        analysis = generator._generate_shap_analysis(performance)
        
        assert analysis['methodology'] == 'SHAP (SHapley Additive exPlanations)'
        assert 'guarantee' in analysis
        assert analysis['total_features'] == 3
        assert len(analysis['top_10_features']) == 3
        assert 'feature_explanations' in analysis
        assert 'contribution_analysis' in analysis
        assert 'interpretability_score' in analysis
        assert 0 <= analysis['interpretability_score'] <= 1
    
    def test_generate_feature_explanation(self, generator):
        """Testa geração de explicação de feature"""
        explanation = generator._generate_feature_explanation('area_privativa', 0.45, 1)
        
        assert explanation['rank'] == 1
        assert explanation['feature_name'] == 'Área Privativa'
        assert explanation['technical_name'] == 'area_privativa'
        assert explanation['shap_importance'] == 0.45
        assert explanation['impact_type'] == 'positivo'
        assert 'adicionou' in explanation['textual_explanation']
        assert explanation['estimated_monetary_impact'] > 0
    
    def test_calculate_interpretability_score(self, generator):
        """Testa cálculo do score de interpretabilidade"""
        # Cenário com alta concentração (mais interpretável)
        high_concentration = {
            'feature_1': 0.6,
            'feature_2': 0.3,
            'feature_3': 0.1
        }
        score_high = generator._calculate_interpretability_score(high_concentration)
        
        # Cenário com baixa concentração (menos interpretável)
        low_concentration = {
            'feature_1': 0.2,
            'feature_2': 0.2,
            'feature_3': 0.2,
            'feature_4': 0.2,
            'feature_5': 0.2
        }
        score_low = generator._calculate_interpretability_score(low_concentration)
        
        assert score_high > score_low
        assert 0 <= score_high <= 1
        assert 0 <= score_low <= 1
        
        # Caso extremo - vazio
        empty_dict = {}
        score_empty = generator._calculate_interpretability_score(empty_dict)
        assert score_empty == 0.0
    
    def test_calculate_explanation_consistency(self, generator):
        """Testa cálculo de consistência entre explicações"""
        coefficients = {'a': 0.5, 'b': 0.3, 'c': 0.2}
        permutation = {'a': 0.6, 'b': 0.25, 'c': 0.15}
        
        consistency = generator._calculate_explanation_consistency(coefficients, permutation)
        
        assert consistency['status'] == 'calculated'
        assert 0 <= consistency['consistency_score'] <= 1
        assert consistency['interpretation'] in ['high', 'medium', 'low']
        
        # Caso com poucas features
        few_features = {'a': 0.5, 'b': 0.3}
        consistency_few = generator._calculate_explanation_consistency(few_features, few_features)
        assert consistency_few['status'] == 'insufficient_features'
    
    def test_generate_validation_summary(self, generator):
        """Testa geração do resumo de validação"""
        validation_result = MockValidationTestResult()
        
        summary = generator.generate_validation_summary(validation_result)
        
        assert summary['standard'] == 'NBR 14653'
        assert summary['overall_grade'] == 'Superior'
        assert summary['compliance_score'] == 0.85
        assert len(summary['individual_tests']) == 4
        assert 'compliance_analysis' in summary
        assert len(summary['compliance_analysis']['passed_tests']) == 3
        assert len(summary['compliance_analysis']['failed_tests']) == 1
    
    def test_identify_critical_issues(self, generator):
        """Testa identificação de problemas críticos"""
        # NBR 14653
        validation_nbr = MockValidationTestResult()
        validation_nbr.individual_tests = [
            MockIndividualTest('Teste R²', False, 0.65, 0.70, 'R² insuficiente'),
            MockIndividualTest('Teste F', True, 25.5, 4.0, 'OK')
        ]
        
        critical_nbr = generator._identify_critical_issues(validation_nbr)
        assert 'Teste R²' in critical_nbr
        
        # USPAP
        validation_uspap = MockValidationTestResult()
        validation_uspap.standard = 'USPAP'
        validation_uspap.individual_tests = [
            MockIndividualTest('Bias Test', False, 0.15, 0.10, 'Bias alto'),
            MockIndividualTest('Other Test', False, 0.5, 0.6, 'Outro')
        ]
        
        critical_uspap = generator._identify_critical_issues(validation_uspap)
        assert 'Bias Test' in critical_uspap
        assert 'Other Test' not in critical_uspap  # Não é crítico
    
    def test_generate_predictions(self, generator, sample_cleaned_data):
        """Testa geração de predições"""
        model_result = MockModelResult()
        
        predictions = generator.generate_predictions(model_result, sample_cleaned_data)
        
        assert 'sample_predictions' in predictions
        assert 'prediction_summary' in predictions
        assert 'visualization_data' in predictions
        
        sample_preds = predictions['sample_predictions']
        assert len(sample_preds) <= 5  # Máximo 5 amostras
        
        for pred in sample_preds:
            assert 'predicted_value' in pred
            assert 'actual_value' in pred
            assert 'confidence_interval' in pred
            assert 'prediction_interval' in pred
            assert 'uncertainty_metrics' in pred
            assert pred['confidence_interval']['lower'] <= pred['confidence_interval']['upper']
            assert pred['prediction_interval']['lower'] <= pred['prediction_interval']['upper']
    
    def test_generate_predictions_empty_data(self, generator):
        """Testa geração de predições com dados vazios"""
        model_result = MockModelResult()
        empty_data = pd.DataFrame()
        
        predictions = generator.generate_predictions(model_result, empty_data)
        
        assert 'message' in predictions
        assert 'Nenhum dado disponível' in predictions['message']
    
    def test_generate_methodology_description(self, generator):
        """Testa geração da descrição metodológica"""
        methodology = generator.generate_methodology_description()
        
        assert methodology['approach'] == 'Machine Learning Avançado com validação NBR 14653 e SHAP obrigatório'
        assert len(methodology['phases']) == 5
        assert 'statistical_foundations' in methodology
        assert 'nbr_compliance' in methodology
        
        # Verificar fases
        phase_names = [phase['phase'] for phase in methodology['phases']]
        assert 'Fase 1 - Ingestão e Validação' in phase_names
        assert 'Fase 5 - Relatório' in phase_names
    
    def test_generate_conclusions(self, generator):
        """Testa geração das conclusões"""
        model_result = MockModelResult()
        validation_result = MockValidationTestResult()
        
        conclusions = generator.generate_conclusions(model_result, validation_result)
        
        assert 'model_adequacy' in conclusions
        assert 'key_findings' in conclusions
        assert 'recommendations' in conclusions
        assert 'limitations' in conclusions
        assert 'next_steps' in conclusions
        
        assert conclusions['model_adequacy']['overall_assessment'] == 'Superior'
        assert conclusions['model_adequacy']['standard_compliance'] is True
        assert len(conclusions['key_findings']) >= 4
        assert len(conclusions['limitations']) >= 3
        assert len(conclusions['next_steps']) >= 3
    
    def test_interpret_r2(self, generator):
        """Testa interpretação do R²"""
        assert generator._interpret_r2(0.95) == "Excelente capacidade explanatória"
        assert generator._interpret_r2(0.85) == "Boa capacidade explanatória"
        assert generator._interpret_r2(0.75) == "Capacidade explanatória adequada"
        assert generator._interpret_r2(0.65) == "Capacidade explanatória insuficiente"
    
    def test_assess_reliability(self, generator):
        """Testa avaliação de confiabilidade"""
        performance = MockPerformance()
        validation_high = MockValidationTestResult()
        validation_high.compliance_score = 0.85
        
        reliability = generator._assess_reliability(performance, validation_high)
        assert reliability == "Alta"
        
        validation_low = MockValidationTestResult()
        validation_low.compliance_score = 0.65
        performance.r2_score = 0.65
        
        reliability_low = generator._assess_reliability(performance, validation_low)
        assert reliability_low == "Baixa"
    
    def test_assess_precision(self, generator):
        """Testa avaliação de precisão"""
        assert generator._assess_precision(8.0) == "alta"
        assert generator._assess_precision(15.0) == "média"
        assert generator._assess_precision(25.0) == "baixa"
    
    def test_generate_recommendations(self, generator):
        """Testa geração de recomendações"""
        # Modelo com baixo R²
        performance_low_r2 = MockPerformance()
        performance_low_r2.r2_score = 0.75
        performance_low_r2.mape = 15.0
        
        validation_result = MockValidationTestResult()
        
        recommendations = generator._generate_recommendations(performance_low_r2, validation_result)
        
        assert any('variáveis adicionais' in rec for rec in recommendations)
        
        # Modelo com alto MAPE
        performance_high_mape = MockPerformance()
        performance_high_mape.r2_score = 0.85
        performance_high_mape.mape = 25.0
        
        recommendations_high_mape = generator._generate_recommendations(performance_high_mape, validation_result)
        
        assert any('outliers' in rec for rec in recommendations_high_mape)
    
    def test_generate_full_report(self, generator, sample_cleaned_data):
        """Testa geração do relatório completo"""
        validation_result = MockValidationResult()
        transformation_result = MockTransformationResult()
        model_result = MockModelResult()
        validation_test_result = MockValidationTestResult()
        
        report = generator.generate_full_report(
            validation_result, transformation_result, model_result,
            validation_test_result, sample_cleaned_data
        )
        
        assert isinstance(report, EvaluationReport)
        assert report.report_id.startswith('VALION_')
        assert isinstance(report.timestamp, datetime)
        assert report.data_summary is not None
        assert report.transformation_summary is not None
        assert report.model_performance is not None
        assert report.validation_summary is not None
        assert report.predictions is not None
        assert report.methodology is not None
        assert report.conclusions is not None
    
    def test_calculate_prediction_intervals(self, generator):
        """Testa cálculo de intervalos de predição"""
        model_result = MockModelResult()
        X_sample = pd.DataFrame({
            'area': [100, 120],
            'quartos': [2, 3]
        })
        predictions = np.array([400000, 500000])
        
        intervals = generator._calculate_prediction_intervals(model_result, X_sample, predictions)
        
        assert 'confidence_intervals' in intervals
        assert 'prediction_intervals' in intervals
        assert 'prediction_std' in intervals
        assert intervals['confidence_level'] == 0.95
        
        assert len(intervals['confidence_intervals']) == 2
        assert len(intervals['prediction_intervals']) == 2
        
        # Intervalos de predição devem ser mais largos que intervalos de confiança
        for i in range(2):
            conf_width = intervals['confidence_intervals'][i]['width']
            pred_width = intervals['prediction_intervals'][i]['width']
            assert pred_width > conf_width
    
    def test_prepare_uncertainty_visualization_data(self, generator):
        """Testa preparação dos dados de visualização de incerteza"""
        predictions = np.array([400000, 500000])
        actual_values = np.array([410000, 490000])
        uncertainty_data = {
            'confidence_intervals': [
                {'lower': 380000, 'upper': 420000, 'width': 40000},
                {'lower': 480000, 'upper': 520000, 'width': 40000}
            ],
            'prediction_intervals': [
                {'lower': 350000, 'upper': 450000, 'width': 100000},
                {'lower': 450000, 'upper': 550000, 'width': 100000}
            ],
            'confidence_level': 0.95,
            'z_score': 1.96,
            'rmse_used': 45000
        }
        
        viz_data = generator._prepare_uncertainty_visualization_data(
            predictions, uncertainty_data, actual_values
        )
        
        assert 'scatter_plot' in viz_data
        assert 'residual_plot' in viz_data
        assert 'interval_histograms' in viz_data
        assert 'metadata' in viz_data
        
        scatter = viz_data['scatter_plot']
        assert len(scatter['x']) == 2
        assert len(scatter['y_predicted']) == 2
        assert len(scatter['y_actual']) == 2
        assert len(scatter['confidence_lower']) == 2
        assert len(scatter['prediction_upper']) == 2
        
        residual = viz_data['residual_plot']
        assert len(residual['residuals']) == 2
        
        metadata = viz_data['metadata']
        assert metadata['confidence_level'] == 0.95
        assert metadata['n_samples'] == 2
    
    @patch('builtins.open')
    @patch('json.dump')
    def test_save_report(self, mock_json_dump, mock_open, generator):
        """Testa salvamento do relatório"""
        report = EvaluationReport(
            report_id='TEST_001',
            timestamp=datetime.now(),
            data_summary={},
            transformation_summary={},
            model_performance={},
            validation_summary={},
            predictions={},
            methodology={},
            conclusions={}
        )
        
        generator.save_report(report, '/test/path.json')
        
        mock_open.assert_called_once_with('/test/path.json', 'w', encoding='utf-8')
        mock_json_dump.assert_called_once()
    
    @patch('pandas.ExcelWriter')
    def test_export_to_excel(self, mock_excel_writer, generator):
        """Testa exportação para Excel"""
        mock_writer = Mock()
        mock_excel_writer.return_value.__enter__.return_value = mock_writer
        
        report = EvaluationReport(
            report_id='TEST_001',
            timestamp=datetime.now(),
            data_summary={},
            transformation_summary={},
            model_performance={
                'performance_metrics': {'r2_score': 0.85},
                'feature_coefficients': {'area': 0.5}
            },
            validation_summary={
                'overall_grade': 'Superior',
                'standard': 'NBR 14653',
                'individual_tests': []
            },
            predictions={},
            methodology={},
            conclusions={}
        )
        
        generator.export_to_excel(report, '/test/path.xlsx')
        
        mock_excel_writer.assert_called_once_with('/test/path.xlsx')
        # Verificar se DataFrames foram criados e salvos
        assert mock_writer.to_excel.call_count >= 3  # Pelo menos 3 sheets


class TestEvaluationReport:
    """Testes para a classe EvaluationReport"""
    
    def test_evaluation_report_creation(self):
        """Testa criação do EvaluationReport"""
        timestamp = datetime.now()
        
        report = EvaluationReport(
            report_id='TEST_001',
            timestamp=timestamp,
            data_summary={'total_records': 100},
            transformation_summary={'features': 10},
            model_performance={'r2': 0.85},
            validation_summary={'grade': 'Superior'},
            predictions={'accuracy': 0.90},
            methodology={'approach': 'ML'},
            conclusions={'reliable': True}
        )
        
        assert report.report_id == 'TEST_001'
        assert report.timestamp == timestamp
        assert report.data_summary['total_records'] == 100
        assert report.model_performance['r2'] == 0.85


@pytest.mark.integration
class TestResultsGeneratorIntegration:
    """Testes de integração para ResultsGenerator"""
    
    def test_full_pipeline_integration(self):
        """Testa pipeline completo de geração de relatório"""
        config = {'expert_mode': True}
        generator = ResultsGenerator(config)
        
        # Criar dados mock completos
        validation_result = MockValidationResult()
        transformation_result = MockTransformationResult()
        model_result = MockModelResult()
        validation_test_result = MockValidationTestResult()
        
        cleaned_data = pd.DataFrame({
            'valor': [400000, 500000, 350000],
            'area': [100, 120, 80],
            'quartos': [2, 3, 2],
            'localizacao': ['Centro', 'Centro', 'Zona Sul'],
            'data_transacao': ['2023-01-15', '2023-02-20', '2023-01-10']
        })
        
        # Executar pipeline completo
        report = generator.generate_full_report(
            validation_result, transformation_result, model_result,
            validation_test_result, cleaned_data
        )
        
        # Verificar integridade do relatório
        assert isinstance(report, EvaluationReport)
        assert 'total_records' in report.data_summary
        assert 'r2_score' in report.model_performance['performance_metrics']
        assert 'overall_grade' in report.validation_summary
        assert 'sample_predictions' in report.predictions
        assert 'approach' in report.methodology
        assert 'model_adequacy' in report.conclusions
        
        # Verificar consistência entre seções
        assert (report.data_summary['total_records'] == 
                len(cleaned_data))
        assert (report.model_performance['performance_metrics']['r2_score'] == 
                model_result.performance.r2_score)