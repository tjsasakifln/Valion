# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
EVS (European Valuation Standards) Validation
Implements validation based on EVS standards focusing on market value assessment,
sustainability considerations, and European regulatory compliance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics import r2_score, mean_squared_error
import logging

from .validation_strategy import Validator, ValidationResult, ValidationTestResult


class EVSValidator(Validator):
    """
    Validator for EVS (European Valuation Standards) standards.
    
    EVS focuses on market value assessment, sustainability considerations,
    and compliance with European regulatory frameworks.
    """
    
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                 X_test: pd.DataFrame, y_test: pd.Series, config: Dict[str, Any]):
        super().__init__(model, X_train, y_train, X_test, y_test, config)
        self.logger = logging.getLogger(__name__)
        
        # EVS focuses on market value and sustainability
        self.thresholds = {
            'min_r2': 0.65,  # Reasonable accuracy for market value
            'max_cv': 0.30,   # Coefficient of variation threshold
            'min_sample_size': 15,  # Minimum sample for European markets
            'sustainability_weight': 0.3,  # Weight for sustainability considerations
            'market_value_accuracy': 0.75  # Threshold for market value accuracy
        }
    
    def test_market_value_basis(self) -> ValidationTestResult:
        """
        Test if the model provides a sound basis for market value assessment.
        
        Returns:
            ValidationTestResult: Result of market value basis test
        """
        try:
            # Check model performance
            y_pred = self.model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            
            # Check for reasonable prediction accuracy
            mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
            
            # Check for model stability (consistency across predictions)
            cv_predictions = np.std(y_pred) / np.mean(y_pred) if np.mean(y_pred) > 0 else float('inf')
            
            market_value_score = 0.0
            
            if r2 >= self.thresholds['min_r2']:
                market_value_score += 0.4
            
            if mape <= 25:  # Less than 25% mean absolute percentage error
                market_value_score += 0.3
            
            if cv_predictions <= self.thresholds['max_cv']:
                market_value_score += 0.3
            
            passed = market_value_score >= self.thresholds['market_value_accuracy']
            
            return ValidationTestResult(
                test_name="Market Value Basis",
                passed=passed,
                value=market_value_score,
                threshold=self.thresholds['market_value_accuracy'],
                description=f"Market value score: {market_value_score:.2f} (R²: {r2:.3f}, MAPE: {mape:.1f}%, CV: {cv_predictions:.3f})",
                recommendation="Model should provide reliable basis for market value assessment"
            )
            
        except Exception as e:
            self.logger.error(f"Error in market value basis test: {e}")
            return ValidationTestResult(
                test_name="Market Value Basis",
                passed=False,
                value="N/A",
                threshold=self.thresholds['market_value_accuracy'],
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify model performance and market value alignment",
                applicable=False
            )
    
    def test_sustainability_considerations(self) -> ValidationTestResult:
        """
        Test if sustainability factors are considered in the model.
        EVS emphasizes environmental and sustainability aspects.
        
        Returns:
            ValidationTestResult: Result of sustainability considerations test
        """
        try:
            # Check for sustainability-related features
            feature_names = self.X_train.columns.str.lower()
            
            # Look for sustainability indicators
            sustainability_indicators = [
                'energy', 'energia', 'energetic', 'energetico',
                'efficiency', 'eficiencia', 'efficient', 'eficiente',
                'green', 'verde', 'sustainable', 'sustentavel',
                'solar', 'heating', 'aquecimento', 'cooling', 'refrigeracao',
                'insulation', 'isolamento', 'thermal', 'termico',
                'water', 'agua', 'waste', 'residuo', 'recycling', 'reciclagem',
                'carbon', 'co2', 'emission', 'emissao',
                'certification', 'certificacao', 'rating', 'avaliacao'
            ]
            
            sustainability_features = sum(1 for indicator in sustainability_indicators 
                                        if any(indicator in feature for feature in feature_names))
            
            # Score based on sustainability feature presence
            sustainability_score = min(sustainability_features / 3, 1.0)  # Normalize to max 1.0
            
            # Check if sustainability features have meaningful impact
            feature_importance_score = 0.0
            if hasattr(self.model, 'feature_importances_'):
                # For tree-based models
                importance_threshold = 0.05
                for i, feature in enumerate(self.X_train.columns):
                    if any(indicator in feature.lower() for indicator in sustainability_indicators):
                        if i < len(self.model.feature_importances_) and self.model.feature_importances_[i] > importance_threshold:
                            feature_importance_score += 0.3
            elif hasattr(self.model, 'coef_'):
                # For linear models
                for i, feature in enumerate(self.X_train.columns):
                    if any(indicator in feature.lower() for indicator in sustainability_indicators):
                        if i < len(self.model.coef_) and abs(self.model.coef_[i]) > 0.1:
                            feature_importance_score += 0.3
            
            total_score = (sustainability_score * 0.7) + (min(feature_importance_score, 1.0) * 0.3)
            
            passed = total_score >= self.thresholds['sustainability_weight']
            
            warning = None
            if sustainability_features == 0:
                warning = "No sustainability features identified - consider including energy efficiency, environmental certifications, etc."
            
            return ValidationTestResult(
                test_name="Sustainability Considerations",
                passed=passed,
                value=total_score,
                threshold=self.thresholds['sustainability_weight'],
                description=f"Sustainability score: {total_score:.2f} ({sustainability_features} relevant features, importance score: {feature_importance_score:.2f})",
                recommendation="Model should consider sustainability factors as per EVS guidelines",
                warning=warning
            )
            
        except Exception as e:
            self.logger.error(f"Error in sustainability considerations test: {e}")
            return ValidationTestResult(
                test_name="Sustainability Considerations",
                passed=False,
                value="N/A",
                threshold=self.thresholds['sustainability_weight'],
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify inclusion of sustainability-related features",
                applicable=False
            )
    
    def test_european_market_compliance(self) -> ValidationTestResult:
        """
        Test compliance with European market characteristics and regulations.
        
        Returns:
            ValidationTestResult: Result of European market compliance test
        """
        try:
            # Check for European market characteristics
            feature_names = self.X_train.columns.str.lower()
            
            # Look for European market indicators
            eu_market_indicators = [
                'location', 'localizacao', 'distrito', 'region', 'regiao',
                'transport', 'transporte', 'metro', 'train', 'trem',
                'school', 'escola', 'university', 'universidade',
                'hospital', 'health', 'saude', 'medical', 'medico',
                'cultural', 'culture', 'cultura', 'historic', 'historico',
                'commercial', 'comercial', 'retail', 'varejo',
                'parking', 'estacionamento', 'garage', 'garagem'
            ]
            
            eu_features = sum(1 for indicator in eu_market_indicators 
                            if any(indicator in feature for feature in feature_names))
            
            # Check sample size adequacy for European markets
            total_samples = len(self.X_train) + len(self.X_test)
            adequate_sample = total_samples >= self.thresholds['min_sample_size']
            
            # Check for reasonable market coverage
            market_coverage_score = 0.0
            
            if eu_features >= 3:
                market_coverage_score += 0.4
            
            if adequate_sample:
                market_coverage_score += 0.3
            
            # Check for price reasonableness (European context)
            y_combined = pd.concat([self.y_train, self.y_test])
            price_cv = y_combined.std() / y_combined.mean() if y_combined.mean() > 0 else float('inf')
            
            if price_cv <= 1.5:  # Reasonable price variation
                market_coverage_score += 0.3
            
            passed = market_coverage_score >= 0.6
            
            return ValidationTestResult(
                test_name="European Market Compliance",
                passed=passed,
                value=market_coverage_score,
                threshold=0.6,
                description=f"EU market compliance score: {market_coverage_score:.2f} (EU features: {eu_features}, Sample size: {total_samples}, Price CV: {price_cv:.2f})",
                recommendation="Model should reflect European market characteristics and regulations"
            )
            
        except Exception as e:
            self.logger.error(f"Error in European market compliance test: {e}")
            return ValidationTestResult(
                test_name="European Market Compliance",
                passed=False,
                value="N/A",
                threshold=0.6,
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify model alignment with European market standards",
                applicable=False
            )
    
    def test_valuation_transparency(self) -> ValidationTestResult:
        """
        Test if the valuation methodology is transparent and explainable.
        
        Returns:
            ValidationTestResult: Result of valuation transparency test
        """
        try:
            # Check model interpretability
            interpretability_score = 0.0
            
            # Linear models are highly interpretable
            if hasattr(self.model, 'coef_'):
                interpretability_score += 0.4
                
                # Check for reasonable number of features (not overly complex)
                num_features = len(self.model.coef_)
                if num_features <= 20:  # Reasonable number of features
                    interpretability_score += 0.2
            
            # Tree-based models have feature importance
            if hasattr(self.model, 'feature_importances_'):
                interpretability_score += 0.3
            
            # Check for documentation potential (named features)
            named_features = sum(1 for col in self.X_train.columns if not col.startswith('feature_'))
            feature_naming_score = min(named_features / len(self.X_train.columns), 1.0)
            interpretability_score += feature_naming_score * 0.3
            
            # Check model performance stability
            try:
                y_pred = self.model.predict(self.X_test)
                residuals = self.y_test - y_pred
                
                # Check for consistent errors (not wildly varying)
                residual_cv = np.std(residuals) / np.mean(np.abs(residuals)) if np.mean(np.abs(residuals)) > 0 else float('inf')
                
                if residual_cv <= 2.0:  # Reasonable error consistency
                    interpretability_score += 0.2
            except:
                pass
            
            interpretability_score = min(interpretability_score, 1.0)
            
            passed = interpretability_score >= 0.6
            
            return ValidationTestResult(
                test_name="Valuation Transparency",
                passed=passed,
                value=interpretability_score,
                threshold=0.6,
                description=f"Transparency score: {interpretability_score:.2f} (Feature naming: {feature_naming_score:.2f}, Model type: {type(self.model).__name__})",
                recommendation="Valuation methodology should be transparent and explainable to stakeholders"
            )
            
        except Exception as e:
            self.logger.error(f"Error in valuation transparency test: {e}")
            return ValidationTestResult(
                test_name="Valuation Transparency",
                passed=False,
                value="N/A",
                threshold=0.6,
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify model interpretability and documentation capabilities",
                applicable=False
            )
    
    def test_professional_competence(self) -> ValidationTestResult:
        """
        Test indicators of professional competence in model development.
        
        Returns:
            ValidationTestResult: Result of professional competence test
        """
        try:
            competence_score = 0.0
            
            # Check for proper train/test split
            if len(self.X_test) > 0 and len(self.X_train) > 0:
                test_ratio = len(self.X_test) / (len(self.X_train) + len(self.X_test))
                if 0.1 <= test_ratio <= 0.4:  # Reasonable test split
                    competence_score += 0.3
            
            # Check for reasonable model selection
            model_type = type(self.model).__name__
            professional_models = [
                'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
                'RandomForestRegressor', 'GradientBoostingRegressor',
                'SVR', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'
            ]
            
            if model_type in professional_models:
                competence_score += 0.3
            
            # Check for data quality considerations
            missing_data_handled = (self.X_train.isnull().sum().sum() == 0 and 
                                  self.X_test.isnull().sum().sum() == 0)
            
            if missing_data_handled:
                competence_score += 0.2
            
            # Check for reasonable feature engineering
            feature_count = len(self.X_train.columns)
            if 3 <= feature_count <= 50:  # Reasonable feature count
                competence_score += 0.2
            
            passed = competence_score >= 0.6
            
            return ValidationTestResult(
                test_name="Professional Competence",
                passed=passed,
                value=competence_score,
                threshold=0.6,
                description=f"Professional competence score: {competence_score:.2f} (Model: {model_type}, Features: {feature_count}, Test split: {test_ratio:.2f})",
                recommendation="Model development should demonstrate professional competence and best practices"
            )
            
        except Exception as e:
            self.logger.error(f"Error in professional competence test: {e}")
            return ValidationTestResult(
                test_name="Professional Competence",
                passed=False,
                value="N/A",
                threshold=0.6,
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify professional modeling practices and methodologies",
                applicable=False
            )
    
    def validate(self) -> ValidationResult:
        """
        Execute complete EVS validation battery.
        
        Returns:
            ValidationResult: Complete validation result
        """
        tests = []
        
        # Execute all EVS tests
        test_functions = [
            self.test_market_value_basis,
            self.test_sustainability_considerations,
            self.test_european_market_compliance,
            self.test_valuation_transparency,
            self.test_professional_competence
        ]
        
        for test_func in test_functions:
            try:
                result = test_func()
                tests.append(result)
            except Exception as e:
                self.logger.error(f"Error in EVS test {test_func.__name__}: {e}")
                tests.append(ValidationTestResult(
                    test_name=f"EVS Test {test_func.__name__}",
                    passed=False,
                    value="N/A",
                    threshold=0.0,
                    description=f"Error in test execution: {str(e)}",
                    recommendation="Verify test implementation and data integrity",
                    applicable=False
                ))
        
        # Calculate compliance score
        applicable_tests = [t for t in tests if t.applicable]
        passed_applicable = sum(1 for test in applicable_tests if test.passed)
        
        if len(applicable_tests) == 0:
            compliance_score = 0.0
            overall_grade = "Non-Compliant"
        else:
            compliance_score = passed_applicable / len(applicable_tests)
            
            # Determine overall grade based on compliance
            if compliance_score >= 0.9:
                overall_grade = "Excellent"
            elif compliance_score >= 0.8:
                overall_grade = "Good"
            elif compliance_score >= 0.6:
                overall_grade = "Adequate"
            else:
                overall_grade = "Inadequate"
        
        # Summary
        summary = {
            'total_tests': len(tests),
            'applicable_tests': len(applicable_tests),
            'passed_tests': sum(1 for test in tests if test.passed),
            'passed_applicable': passed_applicable,
            'compliance_score': compliance_score,
            'overall_grade': overall_grade,
            'non_applicable_tests': [test.test_name for test in tests if not test.applicable],
            'failed_tests': [test.test_name for test in tests if not test.passed and test.applicable],
            'market_value_score': next((t.value for t in tests if t.test_name == "Market Value Basis"), "N/A"),
            'sustainability_score': next((t.value for t in tests if t.test_name == "Sustainability Considerations"), "N/A"),
            'transparency_score': next((t.value for t in tests if t.test_name == "Valuation Transparency"), "N/A"),
            'competence_score': next((t.value for t in tests if t.test_name == "Professional Competence"), "N/A")
        }
        
        self.logger.info(f"EVS validation completed: {overall_grade} ({compliance_score:.2%} compliance)")
        
        return ValidationResult(
            standard="EVS",
            overall_grade=overall_grade,
            compliance_score=compliance_score,
            individual_tests=tests,
            summary=summary
        )