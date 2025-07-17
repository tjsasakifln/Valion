"""
Fase 5: Gerador de Resultados
Consolida todas as informações em um relatório final defensável.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path


@dataclass
class EvaluationReport:
    """Relatório completo de avaliação."""
    report_id: str
    timestamp: datetime
    data_summary: Dict[str, Any]
    transformation_summary: Dict[str, Any]
    model_performance: Dict[str, Any]
    nbr_validation: Dict[str, Any]
    predictions: Dict[str, Any]
    methodology: Dict[str, Any]
    conclusions: Dict[str, Any]


class ResultsGenerator:
    """Gerador de relatórios de avaliação imobiliária."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_data_summary(self, validation_result, cleaned_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera resumo dos dados utilizados.
        
        Args:
            validation_result: Resultado da validação de dados
            cleaned_data: Dados limpos
            
        Returns:
            Resumo dos dados
        """
        summary = {
            'total_records': len(cleaned_data),
            'variables_count': len(cleaned_data.columns),
            'data_quality': {
                'validation_passed': validation_result.is_valid,
                'errors_count': len(validation_result.errors),
                'warnings_count': len(validation_result.warnings),
                'missing_values': validation_result.summary.get('missing_values', {}),
                'data_types': validation_result.summary.get('data_types', {})
            },
            'descriptive_statistics': {
                'valor': {
                    'mean': float(cleaned_data['valor'].mean()),
                    'median': float(cleaned_data['valor'].median()),
                    'std': float(cleaned_data['valor'].std()),
                    'min': float(cleaned_data['valor'].min()),
                    'max': float(cleaned_data['valor'].max()),
                    'q25': float(cleaned_data['valor'].quantile(0.25)),
                    'q75': float(cleaned_data['valor'].quantile(0.75))
                }
            },
            'sample_distribution': {
                'geographic_distribution': self._analyze_geographic_distribution(cleaned_data),
                'temporal_distribution': self._analyze_temporal_distribution(cleaned_data)
            }
        }
        
        return summary
    
    def _analyze_geographic_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa distribuição geográfica dos dados."""
        if 'localizacao' in df.columns:
            location_counts = df['localizacao'].value_counts().to_dict()
            return {
                'locations_count': len(location_counts),
                'distribution': location_counts,
                'concentration_index': self._calculate_concentration_index(location_counts)
            }
        return {'message': 'Informação de localização não disponível'}
    
    def _analyze_temporal_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa distribuição temporal dos dados."""
        if 'data_transacao' in df.columns:
            df['data_transacao'] = pd.to_datetime(df['data_transacao'])
            temporal_dist = df.groupby(df['data_transacao'].dt.year).size().to_dict()
            return {
                'years_span': len(temporal_dist),
                'distribution': temporal_dist,
                'most_recent': df['data_transacao'].max().isoformat(),
                'oldest': df['data_transacao'].min().isoformat()
            }
        return {'message': 'Informação temporal não disponível'}
    
    def _calculate_concentration_index(self, distribution: Dict[str, int]) -> float:
        """Calcula índice de concentração (Herfindahl-Hirschman)."""
        total = sum(distribution.values())
        if total == 0:
            return 0
        
        shares = [count/total for count in distribution.values()]
        hhi = sum(share**2 for share in shares)
        return hhi
    
    def generate_transformation_summary(self, transformation_result) -> Dict[str, Any]:
        """
        Gera resumo das transformações aplicadas.
        
        Args:
            transformation_result: Resultado das transformações
            
        Returns:
            Resumo das transformações
        """
        summary = {
            'transformations_applied': transformation_result.transformations_applied,
            'final_features_count': len(transformation_result.feature_names),
            'feature_names': transformation_result.feature_names,
            'feature_importance': transformation_result.feature_importance,
            'top_features': sorted(
                transformation_result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        
        return summary
    
    def generate_model_summary(self, model_result) -> Dict[str, Any]:
        """
        Gera resumo do modelo.
        
        Args:
            model_result: Resultado do modelo
            
        Returns:
            Resumo do modelo
        """
        performance = model_result.performance
        
        summary = {
            'model_type': 'Elastic Net Regression',
            'hyperparameters': model_result.best_params,
            'training_summary': model_result.training_summary,
            'performance_metrics': {
                'r2_score': performance.r2_score,
                'rmse': performance.rmse,
                'mae': performance.mae,
                'mape': performance.mape,
                'cv_rmse_mean': np.mean(performance.cv_scores),
                'cv_rmse_std': np.std(performance.cv_scores)
            },
            'feature_coefficients': performance.feature_coefficients,
            'significant_features': [
                feature for feature, coef in performance.feature_coefficients.items()
                if abs(coef) > 0.01  # Threshold para coeficientes significativos
            ]
        }
        
        return summary
    
    def generate_nbr_summary(self, nbr_result) -> Dict[str, Any]:
        """
        Gera resumo da validação NBR 14653.
        
        Args:
            nbr_result: Resultado da validação NBR
            
        Returns:
            Resumo da validação
        """
        summary = {
            'overall_grade': nbr_result.overall_grade,
            'compliance_score': nbr_result.compliance_score,
            'summary_statistics': nbr_result.summary,
            'individual_tests': [
                {
                    'test_name': test.test_name,
                    'passed': test.passed,
                    'value': test.value,
                    'threshold': test.threshold,
                    'description': test.description,
                    'recommendation': test.recommendation
                }
                for test in nbr_result.individual_tests
            ],
            'compliance_analysis': {
                'passed_tests': [test.test_name for test in nbr_result.individual_tests if test.passed],
                'failed_tests': [test.test_name for test in nbr_result.individual_tests if not test.passed],
                'critical_issues': [
                    test.test_name for test in nbr_result.individual_tests
                    if not test.passed and 'R²' in test.test_name
                ]
            }
        }
        
        return summary
    
    def generate_predictions(self, model_result, sample_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera exemplos de predições.
        
        Args:
            model_result: Resultado do modelo
            sample_data: Dados de exemplo
            
        Returns:
            Predições de exemplo
        """
        if len(sample_data) == 0:
            return {'message': 'Nenhum dado disponível para predição'}
        
        # Selecionar amostra para predição
        sample_size = min(5, len(sample_data))
        sample = sample_data.sample(n=sample_size, random_state=42)
        
        # Fazer predições (assumindo que model_result tem método predict)
        try:
            predictions = model_result.model.predict(sample.drop(columns=['valor']))
            actual_values = sample['valor'].values
            
            prediction_results = []
            for i in range(len(predictions)):
                prediction_results.append({
                    'sample_id': i + 1,
                    'predicted_value': float(predictions[i]),
                    'actual_value': float(actual_values[i]),
                    'absolute_error': float(abs(predictions[i] - actual_values[i])),
                    'percentage_error': float(abs(predictions[i] - actual_values[i]) / actual_values[i] * 100),
                    'features': sample.iloc[i].drop(['valor']).to_dict()
                })
            
            return {
                'sample_predictions': prediction_results,
                'prediction_summary': {
                    'mean_absolute_error': float(np.mean([p['absolute_error'] for p in prediction_results])),
                    'mean_percentage_error': float(np.mean([p['percentage_error'] for p in prediction_results])),
                    'prediction_range': {
                        'min': float(min(predictions)),
                        'max': float(max(predictions))
                    }
                }
            }
        except Exception as e:
            return {'error': f'Erro ao gerar predições: {str(e)}'}
    
    def generate_methodology_description(self) -> Dict[str, Any]:
        """
        Gera descrição da metodologia utilizada.
        
        Returns:
            Descrição da metodologia
        """
        methodology = {
            'approach': 'Regressão Elastic Net com validação NBR 14653',
            'phases': [
                {
                    'phase': 'Fase 1 - Ingestão e Validação',
                    'description': 'Carregamento e validação dos dados de entrada',
                    'techniques': ['Validação de tipos', 'Detecção de outliers', 'Análise de completude']
                },
                {
                    'phase': 'Fase 2 - Transformação',
                    'description': 'Preparação e engenharia de features',
                    'techniques': ['Normalização', 'Codificação categórica', 'Seleção de features']
                },
                {
                    'phase': 'Fase 3 - Modelagem',
                    'description': 'Treinamento do modelo Elastic Net',
                    'techniques': ['Grid Search', 'Cross-validation', 'Regularização L1/L2']
                },
                {
                    'phase': 'Fase 4 - Validação NBR',
                    'description': 'Bateria de testes estatísticos conforme NBR 14653',
                    'techniques': ['Teste F', 'Teste t', 'Durbin-Watson', 'Shapiro-Wilk']
                },
                {
                    'phase': 'Fase 5 - Relatório',
                    'description': 'Consolidação dos resultados',
                    'techniques': ['Análise de performance', 'Interpretação estatística']
                }
            ],
            'statistical_foundations': {
                'model_type': 'Elastic Net Regression',
                'regularization': 'Combinação L1 (Lasso) e L2 (Ridge)',
                'cross_validation': '5-fold cross-validation',
                'feature_selection': 'Seleção univariada + regularização'
            },
            'nbr_compliance': {
                'standard': 'NBR 14653 - Avaliação de bens',
                'precision_levels': ['Superior (R² ≥ 0.90)', 'Normal (R² ≥ 0.80)', 'Inferior (R² ≥ 0.70)'],
                'required_tests': ['Coeficiente de determinação', 'Teste F', 'Teste t', 'Normalidade', 'Autocorrelação']
            }
        }
        
        return methodology
    
    def generate_conclusions(self, model_result, nbr_result) -> Dict[str, Any]:
        """
        Gera conclusões da avaliação.
        
        Args:
            model_result: Resultado do modelo
            nbr_result: Resultado da validação NBR
            
        Returns:
            Conclusões da avaliação
        """
        performance = model_result.performance
        
        conclusions = {
            'model_adequacy': {
                'overall_assessment': nbr_result.overall_grade,
                'r2_interpretation': self._interpret_r2(performance.r2_score),
                'nbr_compliance': nbr_result.compliance_score >= 0.7,
                'reliability_level': self._assess_reliability(performance, nbr_result)
            },
            'key_findings': [
                f"Modelo apresenta R² de {performance.r2_score:.4f} ({nbr_result.overall_grade})",
                f"Aprovado em {len([t for t in nbr_result.individual_tests if t.passed])}/{len(nbr_result.individual_tests)} testes NBR 14653",
                f"RMSE de {performance.rmse:.2f} indica precisão {self._assess_precision(performance.mape)}",
                f"Top 3 variáveis mais importantes: {list(model_result.performance.feature_coefficients.keys())[:3]}"
            ],
            'recommendations': self._generate_recommendations(performance, nbr_result),
            'limitations': [
                "Modelo válido apenas para o mercado e período analisados",
                "Predições devem ser validadas com dados de mercado atuais",
                "Variáveis não incluídas no modelo podem afetar o valor real"
            ],
            'next_steps': [
                "Atualizar modelo com dados mais recentes",
                "Incluir variáveis adicionais se disponíveis",
                "Validar predições com transações recentes"
            ]
        }
        
        return conclusions
    
    def _interpret_r2(self, r2_score: float) -> str:
        """Interpreta o valor do R²."""
        if r2_score >= 0.90:
            return "Excelente capacidade explanatória"
        elif r2_score >= 0.80:
            return "Boa capacidade explanatória"
        elif r2_score >= 0.70:
            return "Capacidade explanatória adequada"
        else:
            return "Capacidade explanatória insuficiente"
    
    def _assess_reliability(self, performance, nbr_result) -> str:
        """Avalia confiabilidade geral do modelo."""
        if nbr_result.compliance_score >= 0.8 and performance.r2_score >= 0.80:
            return "Alta"
        elif nbr_result.compliance_score >= 0.7 and performance.r2_score >= 0.70:
            return "Média"
        else:
            return "Baixa"
    
    def _assess_precision(self, mape: float) -> str:
        """Avalia precisão baseada no MAPE."""
        if mape <= 10:
            return "alta"
        elif mape <= 20:
            return "média"
        else:
            return "baixa"
    
    def _generate_recommendations(self, performance, nbr_result) -> List[str]:
        """Gera recomendações baseadas nos resultados."""
        recommendations = []
        
        if performance.r2_score < 0.80:
            recommendations.append("Considerar inclusão de variáveis adicionais para melhorar R²")
        
        if performance.mape > 20:
            recommendations.append("Revisar outliers e qualidade dos dados para reduzir MAPE")
        
        failed_tests = [t for t in nbr_result.individual_tests if not t.passed]
        if failed_tests:
            recommendations.append(f"Endereçar falhas nos testes: {', '.join([t.test_name for t in failed_tests])}")
        
        if not recommendations:
            recommendations.append("Modelo atende aos critérios de qualidade estabelecidos")
        
        return recommendations
    
    def generate_full_report(self, validation_result, transformation_result, 
                           model_result, nbr_result, cleaned_data: pd.DataFrame) -> EvaluationReport:
        """
        Gera relatório completo de avaliação.
        
        Args:
            validation_result: Resultado da validação de dados
            transformation_result: Resultado das transformações
            model_result: Resultado do modelo
            nbr_result: Resultado da validação NBR
            cleaned_data: Dados limpos
            
        Returns:
            Relatório completo
        """
        report_id = f"VALION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report = EvaluationReport(
            report_id=report_id,
            timestamp=datetime.now(),
            data_summary=self.generate_data_summary(validation_result, cleaned_data),
            transformation_summary=self.generate_transformation_summary(transformation_result),
            model_performance=self.generate_model_summary(model_result),
            nbr_validation=self.generate_nbr_summary(nbr_result),
            predictions=self.generate_predictions(model_result, cleaned_data),
            methodology=self.generate_methodology_description(),
            conclusions=self.generate_conclusions(model_result, nbr_result)
        )
        
        self.logger.info(f"Relatório gerado: {report_id}")
        return report
    
    def save_report(self, report: EvaluationReport, filepath: str) -> None:
        """
        Salva relatório em arquivo JSON.
        
        Args:
            report: Relatório a ser salvo
            filepath: Caminho do arquivo
        """
        report_dict = asdict(report)
        report_dict['timestamp'] = report.timestamp.isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Relatório salvo em: {filepath}")
    
    def export_to_excel(self, report: EvaluationReport, filepath: str) -> None:
        """
        Exporta relatório para Excel.
        
        Args:
            report: Relatório a ser exportado
            filepath: Caminho do arquivo Excel
        """
        with pd.ExcelWriter(filepath) as writer:
            # Resumo geral
            summary_df = pd.DataFrame([{
                'Métrica': 'R²',
                'Valor': report.model_performance['performance_metrics']['r2_score'],
                'Grau NBR': report.nbr_validation['overall_grade']
            }])
            summary_df.to_excel(writer, sheet_name='Resumo', index=False)
            
            # Testes NBR
            nbr_df = pd.DataFrame(report.nbr_validation['individual_tests'])
            nbr_df.to_excel(writer, sheet_name='Testes NBR', index=False)
            
            # Coeficientes
            coef_df = pd.DataFrame(list(report.model_performance['feature_coefficients'].items()),
                                 columns=['Feature', 'Coeficiente'])
            coef_df.to_excel(writer, sheet_name='Coeficientes', index=False)
        
        self.logger.info(f"Relatório Excel salvo em: {filepath}")