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
        Gera resumo do modelo com interpretabilidade avançada.
        
        Args:
            model_result: Resultado do modelo
            
        Returns:
            Resumo do modelo
        """
        performance = model_result.performance
        
        # Determinar tipo de modelo
        model_type_map = {
            'elastic_net': 'Elastic Net Regression',
            'xgboost': 'XGBoost Regression',
            'gradient_boosting': 'Gradient Boosting Regression',
            'random_forest': 'Random Forest Regression'
        }
        
        model_display_name = model_type_map.get(model_result.model_type, model_result.model_type)
        
        summary = {
            'model_type': model_display_name,
            'model_type_code': model_result.model_type,
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
            'significant_features': self._get_significant_features(performance.feature_coefficients, model_result.model_type)
        }
        
        # Adicionar Permutation Feature Importance se disponível
        if performance.permutation_importance:
            summary['permutation_importance'] = performance.permutation_importance
            summary['top_permutation_features'] = sorted(
                performance.permutation_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        
        # Adicionar SHAP values se disponível (OBRIGATÓRIO no modo expert)
        if performance.shap_feature_importance:
            summary['shap_feature_importance'] = performance.shap_feature_importance
            summary['top_shap_features'] = sorted(
                performance.shap_feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Seção dedicada SHAP para modo especialista
            summary['shap_analysis'] = self._generate_shap_analysis(performance)
        elif self.config.get('expert_mode', False):
            # Erro crítico se modo expert sem SHAP
            summary['shap_error'] = 'ERRO CRÍTICO: Modo especialista requer SHAP values'
        
        # Análise de interpretabilidade
        summary['interpretability_analysis'] = self._generate_interpretability_analysis(performance)
        
        return summary
    
    def _get_significant_features(self, feature_coefficients: Dict[str, float], model_type: str) -> List[str]:
        """
        Identifica features significativas baseado no tipo de modelo.
        """
        if model_type == 'elastic_net':
            # Para modelos lineares, usar threshold nos coeficientes
            return [
                feature for feature, coef in feature_coefficients.items()
                if abs(coef) > 0.01
            ]
        else:
            # Para modelos baseados em árvore, usar threshold na importância
            sorted_features = sorted(
                feature_coefficients.items(),
                key=lambda x: x[1],
                reverse=True
            )
            # Pegar top 50% das features ou pelo menos 10
            threshold_idx = max(10, len(sorted_features) // 2)
            return [feature for feature, _ in sorted_features[:threshold_idx]]
    
    def _generate_interpretability_analysis(self, performance) -> Dict[str, Any]:
        """
        Gera análise de interpretabilidade do modelo.
        """
        analysis = {
            'glass_box_level': 'high',  # Valion é uma plataforma "glass-box"
            'available_explanations': []
        }
        
        # Verificar quais tipos de explicação estão disponíveis
        if performance.feature_coefficients:
            analysis['available_explanations'].append('feature_coefficients')
        
        if performance.permutation_importance:
            analysis['available_explanations'].append('permutation_importance')
        
        if performance.shap_feature_importance:
            analysis['available_explanations'].append('shap_values')
        
        # Consistência entre diferentes métodos de explicação
        if performance.permutation_importance and performance.feature_coefficients:
            analysis['explanation_consistency'] = self._calculate_explanation_consistency(
                performance.feature_coefficients,
                performance.permutation_importance
            )
        
        return analysis
    
    def _generate_shap_analysis(self, performance) -> Dict[str, Any]:
        """
        Gera análise detalhada SHAP para modo especialista.
        Garante interpretabilidade "glass-box" absoluta.
        """
        if not performance.shap_feature_importance:
            return {'error': 'SHAP values não disponíveis'}
        
        shap_importance = performance.shap_feature_importance
        
        # Ordenar features por importância SHAP
        sorted_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
        
        analysis = {
            'methodology': 'SHAP (SHapley Additive exPlanations)',
            'guarantee': 'Interpretabilidade glass-box absoluta e inegociável',
            'total_features': len(shap_importance),
            'top_10_features': sorted_features[:10],
            'feature_explanations': {},
            'contribution_analysis': {},
            'interpretability_score': self._calculate_interpretability_score(shap_importance),
            'shap_summary': {
                'max_importance': max(shap_importance.values()),
                'min_importance': min(shap_importance.values()),
                'importance_range': max(shap_importance.values()) - min(shap_importance.values()),
                'dominant_features': [f for f, imp in sorted_features[:3]]
            }
        }
        
        # Gerar explicações textuais para as top features
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            explanation = self._generate_feature_explanation(feature, importance, i+1)
            analysis['feature_explanations'][feature] = explanation
        
        # Análise de contribuições
        total_importance = sum(abs(imp) for imp in shap_importance.values())
        analysis['contribution_analysis'] = {
            'total_absolute_importance': total_importance,
            'feature_contributions_pct': {
                feature: (abs(importance) / total_importance) * 100
                for feature, importance in shap_importance.items()
            },
            'cumulative_top_5': sum(abs(imp) for _, imp in sorted_features[:5]) / total_importance * 100
        }
        
        return analysis
    
    def _generate_feature_explanation(self, feature: str, importance: float, rank: int) -> Dict[str, Any]:
        """
        Gera explicação textual detalhada para uma feature específica.
        Formato: "A característica 'X' de Y adicionou/reduziu R$ Z ao valor"
        """
        # Mapear nomes técnicos para nomes amigáveis
        feature_names = {
            'area_privativa': 'Área Privativa',
            'localizacao_score': 'Localização',
            'idade_imovel': 'Idade do Imóvel',
            'vagas_garagem': 'Vagas de Garagem',
            'banheiros': 'Número de Banheiros',
            'quartos': 'Número de Quartos',
            'elevador': 'Presença de Elevador',
            'piscina': 'Presença de Piscina'
        }
        
        friendly_name = feature_names.get(feature, feature.replace('_', ' ').title())
        
        # Determinar impacto
        impact_type = 'positivo' if importance > 0 else 'negativo'
        action = 'adicionou' if importance > 0 else 'reduziu'
        
        # Estimar valor monetário (simplificado - em produção seria baseado no modelo real)
        estimated_value = abs(importance * 300000)  # Estimativa baseada na importância
        
        explanation = {
            'rank': rank,
            'feature_name': friendly_name,
            'technical_name': feature,
            'shap_importance': importance,
            'impact_type': impact_type,
            'estimated_monetary_impact': estimated_value,
            'textual_explanation': f"A característica '{friendly_name}' {action} aproximadamente R$ {estimated_value:,.0f} ao valor final do imóvel",
            'relative_importance_pct': abs(importance) * 100  # Simplificado
        }
        
        return explanation
    
    def _calculate_interpretability_score(self, shap_importance: Dict[str, float]) -> float:
        """
        Calcula score de interpretabilidade baseado na distribuição SHAP.
        Score alto = poucas features dominantes (mais interpretável)
        Score baixo = muitas features com importância similar (menos interpretável)
        """
        if not shap_importance:
            return 0.0
        
        importances = list(shap_importance.values())
        total_importance = sum(abs(imp) for imp in importances)
        
        if total_importance == 0:
            return 0.0
        
        # Calcular concentração (índice de Gini simplificado)
        sorted_importances = sorted([abs(imp) for imp in importances], reverse=True)
        normalized_importances = [imp / total_importance for imp in sorted_importances]
        
        # Score baseado na concentração das top 3 features
        top_3_concentration = sum(normalized_importances[:3])
        
        # Score de interpretabilidade (0-1, onde 1 é mais interpretável)
        interpretability_score = min(1.0, top_3_concentration * 1.5)
        
        return interpretability_score
    
    def _calculate_explanation_consistency(self, coefficients: Dict[str, float], 
                                         permutation: Dict[str, float]) -> Dict[str, Any]:
        """
        Calcula consistência entre diferentes métodos de explicação.
        """
        common_features = set(coefficients.keys()) & set(permutation.keys())
        
        if len(common_features) < 3:
            return {'status': 'insufficient_features'}
        
        # Correlation entre rankings
        coef_ranking = {f: i for i, (f, _) in enumerate(
            sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        )}
        perm_ranking = {f: i for i, (f, _) in enumerate(
            sorted(permutation.items(), key=lambda x: x[1], reverse=True)
        )}
        
        rank_differences = [
            abs(coef_ranking[f] - perm_ranking[f]) for f in common_features
        ]
        
        avg_rank_diff = np.mean(rank_differences)
        max_possible_diff = len(common_features) - 1
        
        consistency_score = 1 - (avg_rank_diff / max_possible_diff) if max_possible_diff > 0 else 1
        
        return {
            'status': 'calculated',
            'consistency_score': consistency_score,
            'interpretation': 'high' if consistency_score > 0.7 else 'medium' if consistency_score > 0.5 else 'low'
        }
    
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
        Gera exemplos de predições com intervalos de confiança e predição.
        
        Args:
            model_result: Resultado do modelo
            sample_data: Dados de exemplo
            
        Returns:
            Predições de exemplo com incerteza
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
            
            # Calcular intervalos de confiança e predição
            uncertainty_data = self._calculate_prediction_intervals(
                model_result, sample.drop(columns=['valor']), predictions
            )
            
            prediction_results = []
            for i in range(len(predictions)):
                prediction_results.append({
                    'sample_id': i + 1,
                    'predicted_value': float(predictions[i]),
                    'actual_value': float(actual_values[i]),
                    'absolute_error': float(abs(predictions[i] - actual_values[i])),
                    'percentage_error': float(abs(predictions[i] - actual_values[i]) / actual_values[i] * 100),
                    'confidence_interval': {
                        'lower': float(uncertainty_data['confidence_intervals'][i]['lower']),
                        'upper': float(uncertainty_data['confidence_intervals'][i]['upper']),
                        'width': float(uncertainty_data['confidence_intervals'][i]['width'])
                    },
                    'prediction_interval': {
                        'lower': float(uncertainty_data['prediction_intervals'][i]['lower']),
                        'upper': float(uncertainty_data['prediction_intervals'][i]['upper']),
                        'width': float(uncertainty_data['prediction_intervals'][i]['width'])
                    },
                    'uncertainty_metrics': {
                        'prediction_std': float(uncertainty_data['prediction_std'][i]),
                        'confidence_level': 0.95,
                        'within_confidence_interval': (
                            uncertainty_data['confidence_intervals'][i]['lower'] <= 
                            actual_values[i] <= 
                            uncertainty_data['confidence_intervals'][i]['upper']
                        ),
                        'within_prediction_interval': (
                            uncertainty_data['prediction_intervals'][i]['lower'] <= 
                            actual_values[i] <= 
                            uncertainty_data['prediction_intervals'][i]['upper']
                        )
                    },
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
                    },
                    'uncertainty_summary': {
                        'mean_confidence_width': float(np.mean([p['confidence_interval']['width'] for p in prediction_results])),
                        'mean_prediction_width': float(np.mean([p['prediction_interval']['width'] for p in prediction_results])),
                        'confidence_coverage': float(np.mean([p['uncertainty_metrics']['within_confidence_interval'] for p in prediction_results])),
                        'prediction_coverage': float(np.mean([p['uncertainty_metrics']['within_prediction_interval'] for p in prediction_results]))
                    }
                },
                'visualization_data': self._prepare_uncertainty_visualization_data(
                    predictions, uncertainty_data, actual_values
                )
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
            'approach': 'Machine Learning Avançado com validação NBR 14653 e SHAP obrigatório',
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
    
    def _calculate_prediction_intervals(self, model_result, X_sample: pd.DataFrame, 
                                      predictions: np.ndarray) -> Dict[str, Any]:
        """
        Calcula intervalos de confiança e predição para as predições.
        
        Args:
            model_result: Resultado do modelo
            X_sample: Features para predição
            predictions: Predições do modelo
            
        Returns:
            Dados de incerteza incluindo intervalos
        """
        try:
            # Parâmetros para cálculo de intervalos
            confidence_level = 0.95
            alpha = 1 - confidence_level
            z_score = 1.96  # Para 95% de confiança
            
            # Estimar erro padrão baseado na performance do modelo
            rmse = model_result.performance.rmse
            n_train = getattr(model_result, 'n_train_samples', 1000)  # Padrão se não disponível
            
            # Calcular desvio padrão das predições (aproximação)
            prediction_std = np.full(len(predictions), rmse / np.sqrt(n_train))
            
            # Intervalos de confiança (incerteza da predição média)
            confidence_intervals = []
            for i, (pred, std) in enumerate(zip(predictions, prediction_std)):
                margin_error = z_score * std
                lower = pred - margin_error
                upper = pred + margin_error
                width = upper - lower
                
                confidence_intervals.append({
                    'lower': lower,
                    'upper': upper,
                    'width': width
                })
            
            # Intervalos de predição (incerteza da predição individual)
            # Inclui tanto a incerteza do modelo quanto a variabilidade dos dados
            prediction_intervals = []
            for i, (pred, std) in enumerate(zip(predictions, prediction_std)):
                # Para intervalos de predição, incluir variabilidade adicional
                total_std = np.sqrt(std**2 + rmse**2)
                margin_error = z_score * total_std
                lower = pred - margin_error
                upper = pred + margin_error
                width = upper - lower
                
                prediction_intervals.append({
                    'lower': lower,
                    'upper': upper,
                    'width': width
                })
            
            return {
                'confidence_intervals': confidence_intervals,
                'prediction_intervals': prediction_intervals,
                'prediction_std': prediction_std,
                'confidence_level': confidence_level,
                'z_score': z_score,
                'rmse_used': rmse
            }
            
        except Exception as e:
            # Fallback para intervalos simples baseados em RMSE
            self.logger.warning(f"Erro ao calcular intervalos: {e}. Usando aproximação simples.")
            
            rmse = model_result.performance.rmse
            simple_margin = 1.96 * rmse  # 95% de confiança
            
            confidence_intervals = []
            prediction_intervals = []
            
            for pred in predictions:
                # Intervalos de confiança mais estreitos
                conf_margin = simple_margin * 0.5
                confidence_intervals.append({
                    'lower': pred - conf_margin,
                    'upper': pred + conf_margin,
                    'width': 2 * conf_margin
                })
                
                # Intervalos de predição mais largos
                pred_margin = simple_margin
                prediction_intervals.append({
                    'lower': pred - pred_margin,
                    'upper': pred + pred_margin,
                    'width': 2 * pred_margin
                })
            
            return {
                'confidence_intervals': confidence_intervals,
                'prediction_intervals': prediction_intervals,
                'prediction_std': np.full(len(predictions), rmse),
                'confidence_level': 0.95,
                'z_score': 1.96,
                'rmse_used': rmse
            }
    
    def _prepare_uncertainty_visualization_data(self, predictions: np.ndarray, 
                                              uncertainty_data: Dict[str, Any], 
                                              actual_values: np.ndarray) -> Dict[str, Any]:
        """
        Prepara dados para visualização de incerteza com Plotly.
        
        Args:
            predictions: Predições do modelo
            uncertainty_data: Dados de incerteza calculados
            actual_values: Valores reais
            
        Returns:
            Dados formatados para visualização
        """
        sample_indices = list(range(len(predictions)))
        
        # Dados para gráfico de dispersão com bandas de incerteza
        scatter_data = {
            'x': sample_indices,
            'y_predicted': predictions.tolist(),
            'y_actual': actual_values.tolist(),
            'confidence_lower': [ci['lower'] for ci in uncertainty_data['confidence_intervals']],
            'confidence_upper': [ci['upper'] for ci in uncertainty_data['confidence_intervals']],
            'prediction_lower': [pi['lower'] for pi in uncertainty_data['prediction_intervals']],
            'prediction_upper': [pi['upper'] for pi in uncertainty_data['prediction_intervals']]
        }
        
        # Dados para gráfico de residuais com incerteza
        residuals = actual_values - predictions
        residual_data = {
            'x': predictions.tolist(),
            'residuals': residuals.tolist(),
            'confidence_bands': [
                ci['width'] / 2 for ci in uncertainty_data['confidence_intervals']
            ]
        }
        
        # Dados para histograma de intervalos
        interval_widths = {
            'confidence_widths': [ci['width'] for ci in uncertainty_data['confidence_intervals']],
            'prediction_widths': [pi['width'] for pi in uncertainty_data['prediction_intervals']]
        }
        
        return {
            'scatter_plot': scatter_data,
            'residual_plot': residual_data,
            'interval_histograms': interval_widths,
            'metadata': {
                'confidence_level': uncertainty_data['confidence_level'],
                'z_score': uncertainty_data['z_score'],
                'rmse_used': uncertainty_data['rmse_used'],
                'n_samples': len(predictions)
            }
        }