# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
USPAP (Uniform Standards of Professional Appraisal Practice) Validation
Implements validation based on USPAP standards focusing on methodology defensibility,
market analysis, and professional judgment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics import r2_score, mean_squared_error
import logging

from .validation_strategy import Validator, ValidationResult, ValidationTestResult


class USPAPValidator(Validator):
    """
    Validator for USPAP (Uniform Standards of Professional Appraisal Practice) standards.
    
    USPAP focuses on professional judgment, methodology defensibility, and market analysis
    rather than strict statistical thresholds like NBR 14653.
    """
    
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                 X_test: pd.DataFrame, y_test: pd.Series, config: Dict[str, Any]):
        super().__init__(model, X_train, y_train, X_test, y_test, config)
        self.logger = logging.getLogger(__name__)
        
        # USPAP focuses on reasonableness and professional judgment
        self.thresholds = {
            'min_r2': 0.60,  # More lenient than NBR, focus on reasonableness
            'max_cv': 0.25,   # Coefficient of variation threshold
            'min_sample_size': 10,  # Minimum meaningful sample
            'max_outliers_percent': 10.0,  # More tolerant of outliers
            'methodology_score_threshold': 0.7
        }
    
    def test_methodology_defensibility(self) -> ValidationTestResult:
        """
        Test if the methodology is defensible based on model characteristics.
        
        Returns:
            ValidationTestResult: Result of methodology defensibility test
        """
        try:
            # Check if model has interpretable coefficients
            has_coefficients = hasattr(self.model, 'coef_')
            
            # Check if model type is commonly accepted in appraisal practice
            model_type = type(self.model).__name__
            acceptable_models = [
                'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
                'RandomForestRegressor', 'GradientBoostingRegressor',
                'XGBRegressor', 'LGBMRegressor'
            ]
            
            is_acceptable_model = model_type in acceptable_models
            
            # Check R² for basic model performance
            y_pred = self.model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            
            # Score methodology based on multiple factors
            methodology_score = 0.0
            
            if has_coefficients:
                methodology_score += 0.3  # Interpretability bonus
            
            if is_acceptable_model:
                methodology_score += 0.4  # Accepted model type
            
            if r2 >= self.thresholds['min_r2']:
                methodology_score += 0.3  # Reasonable performance
            
            passed = methodology_score >= self.thresholds['methodology_score_threshold']
            
            return ValidationTestResult(
                test_name="Methodology Defensibility",
                passed=passed,
                value=methodology_score,
                threshold=self.thresholds['methodology_score_threshold'],
                description=f"Methodology score: {methodology_score:.2f} (Model: {model_type}, R²: {r2:.3f})",
                recommendation="Methodology should be defensible and commonly accepted in appraisal practice"
            )
            
        except Exception as e:
            self.logger.error(f"Error in methodology defensibility test: {e}")
            return ValidationTestResult(
                test_name="Methodology Defensibility",
                passed=False,
                value="N/A",
                threshold=self.thresholds['methodology_score_threshold'],
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify model integrity and type",
                applicable=False
            )
    
    def test_market_analysis_adequacy(self) -> ValidationTestResult:
        """
        Test if the market analysis is adequate based on sample size and data quality.
        
        Returns:
            ValidationTestResult: Result of market analysis adequacy test
        """
        try:
            # Combine training and test data for market analysis
            total_samples = len(self.X_train) + len(self.X_test)
            
            # Check for reasonable sample size
            adequate_sample = total_samples >= self.thresholds['min_sample_size']
            
            # Check for data variability (not all values identical)
            y_combined = pd.concat([self.y_train, self.y_test])
            has_variability = y_combined.nunique() > 1 and y_combined.std() > 0
            
            # Check for reasonable coverage of market segments
            # (based on feature coverage)
            feature_coverage_score = 0.0
            for col in self.X_train.columns:
                if self.X_train[col].nunique() > 1:  # Feature has variation
                    feature_coverage_score += 1
            
            feature_coverage_score = feature_coverage_score / len(self.X_train.columns)
            
            market_adequacy_score = 0.0
            
            if adequate_sample:
                market_adequacy_score += 0.4
            
            if has_variability:
                market_adequacy_score += 0.3
            
            if feature_coverage_score > 0.5:
                market_adequacy_score += 0.3
            
            passed = market_adequacy_score >= 0.6
            
            return ValidationTestResult(
                test_name="Market Analysis Adequacy",
                passed=passed,
                value=market_adequacy_score,
                threshold=0.6,
                description=f"Market analysis score: {market_adequacy_score:.2f} (Sample size: {total_samples}, Features coverage: {feature_coverage_score:.2f})",
                recommendation="Market analysis should cover adequate sample size and market segments"
            )
            
        except Exception as e:
            self.logger.error(f"Error in market analysis adequacy test: {e}")
            return ValidationTestResult(
                test_name="Market Analysis Adequacy",
                passed=False,
                value="N/A",
                threshold=0.6,
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify data integrity and market coverage",
                applicable=False
            )
    
    def test_reasonableness_of_results(self) -> ValidationTestResult:
        """
        Test if the results are reasonable based on prediction accuracy and consistency.
        
        Returns:
            ValidationTestResult: Result of reasonableness test
        """
        try:
            y_pred = self.model.predict(self.X_test)
            
            # Calculate coefficient of variation of residuals
            residuals = self.y_test - y_pred
            cv_residuals = np.std(residuals) / np.mean(np.abs(y_pred)) if np.mean(np.abs(y_pred)) > 0 else float('inf')
            
            # Check for reasonable R²
            r2 = r2_score(self.y_test, y_pred)
            
            # Check for absence of extreme outliers
            residuals_std = np.std(residuals)
            outliers = np.abs(residuals) > 3 * residuals_std
            outlier_percentage = np.sum(outliers) / len(residuals) * 100
            
            reasonableness_score = 0.0
            
            if r2 >= self.thresholds['min_r2']:
                reasonableness_score += 0.4
            
            if cv_residuals <= self.thresholds['max_cv']:
                reasonableness_score += 0.3
            
            if outlier_percentage <= self.thresholds['max_outliers_percent']:
                reasonableness_score += 0.3
            
            passed = reasonableness_score >= 0.6
            
            return ValidationTestResult(
                test_name="Reasonableness of Results",
                passed=passed,
                value=reasonableness_score,
                threshold=0.6,
                description=f"Reasonableness score: {reasonableness_score:.2f} (R²: {r2:.3f}, CV: {cv_residuals:.3f}, Outliers: {outlier_percentage:.1f}%)",
                recommendation="Results should be reasonable and consistent with market expectations"
            )
            
        except Exception as e:
            self.logger.error(f"Error in reasonableness test: {e}")
            return ValidationTestResult(
                test_name="Reasonableness of Results",
                passed=False,
                value="N/A",
                threshold=0.6,
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify model predictions and result consistency",
                applicable=False
            )
    
    def test_highest_and_best_use_consideration(self) -> ValidationTestResult:
        """
        Test if highest and best use considerations are reflected in the model.
        This is a qualitative assessment based on feature selection.
        
        Returns:
            ValidationTestResult: Result of highest and best use test
        """
        try:
            # Check if model includes relevant features for highest and best use
            feature_names = self.X_train.columns.str.lower()
            
            # Look for features that might indicate highest and best use considerations
            hbu_indicators = [
                'zoning', 'zone', 'land_use', 'uso', 'tipo',
                'area', 'size', 'tamanho', 'metragem',
                'location', 'localizacao', 'bairro', 'neighborhood',
                'access', 'acesso', 'infrastructure', 'infraestrutura'
            ]
            
            hbu_feature_count = sum(1 for indicator in hbu_indicators 
                                   if any(indicator in feature for feature in feature_names))
            
            # Score based on presence of relevant features
            hbu_score = min(hbu_feature_count / 3, 1.0)  # Normalize to max 1.0
            
            passed = hbu_score >= 0.3  # At least some relevant features
            
            warning = None
            if hbu_score < 0.5:
                warning = "Consider including more features related to highest and best use analysis"
            
            return ValidationTestResult(
                test_name="Highest and Best Use Consideration",
                passed=passed,
                value=hbu_score,
                threshold=0.3,
                description=f"HBU feature score: {hbu_score:.2f} ({hbu_feature_count} relevant features identified)",
                recommendation="Model should consider factors relevant to highest and best use analysis",
                warning=warning
            )
            
        except Exception as e:
            self.logger.error(f"Error in highest and best use test: {e}")
            return ValidationTestResult(
                test_name="Highest and Best Use Consideration",
                passed=False,
                value="N/A",
                threshold=0.3,
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify feature selection includes relevant HBU factors",
                applicable=False
            )
    
    def test_data_verification_quality(self) -> ValidationTestResult:
        """
        Test the quality of data verification based on completeness and consistency.
        
        Returns:
            ValidationTestResult: Result of data verification test
        """
        try:
            # Check for missing values
            total_cells = self.X_train.size + self.X_test.size
            missing_cells = self.X_train.isnull().sum().sum() + self.X_test.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            
            # Check for data consistency (reasonable ranges)
            consistency_score = 0.0
            
            # Check target variable consistency
            y_combined = pd.concat([self.y_train, self.y_test])
            if y_combined.min() > 0:  # Prices should be positive
                consistency_score += 0.3
            
            # Check for reasonable data ranges (no extreme outliers in features)
            extreme_outliers = 0
            for col in self.X_train.columns:
                if self.X_train[col].dtype in ['int64', 'float64']:
                    Q1 = self.X_train[col].quantile(0.25)
                    Q3 = self.X_train[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        outliers = ((self.X_train[col] < Q1 - 3 * IQR) | 
                                   (self.X_train[col] > Q3 + 3 * IQR)).sum()
                        if outliers > len(self.X_train) * 0.1:  # More than 10% outliers
                            extreme_outliers += 1
            
            if extreme_outliers == 0:
                consistency_score += 0.4
            
            # Data completeness score
            completeness_score = max(0, 1 - missing_percentage / 100)
            
            overall_score = (consistency_score + completeness_score) / 2
            
            passed = overall_score >= 0.7 and missing_percentage <= 20
            
            return ValidationTestResult(
                test_name="Data Verification Quality",
                passed=passed,
                value=overall_score,
                threshold=0.7,
                description=f"Data quality score: {overall_score:.2f} (Missing: {missing_percentage:.1f}%, Extreme outliers in {extreme_outliers} features)",
                recommendation="Data should be complete, consistent, and properly verified"
            )
            
        except Exception as e:
            self.logger.error(f"Error in data verification test: {e}")
            return ValidationTestResult(
                test_name="Data Verification Quality",
                passed=False,
                value="N/A",
                threshold=0.7,
                description=f"Error in test execution: {str(e)}",
                recommendation="Verify data completeness and consistency",
                applicable=False
            )
    
    def validate(self) -> ValidationResult:
        """
        Execute complete USPAP validation battery.
        
        Returns:
            ValidationResult: Complete validation result
        """
        tests = []
        
        # Execute all USPAP tests
        test_functions = [
            self.test_methodology_defensibility,
            self.test_market_analysis_adequacy,
            self.test_reasonableness_of_results,
            self.test_highest_and_best_use_consideration,
            self.test_data_verification_quality
        ]
        
        for test_func in test_functions:
            try:
                result = test_func()
                tests.append(result)
            except Exception as e:
                self.logger.error(f"Error in USPAP test {test_func.__name__}: {e}")
                tests.append(ValidationTestResult(
                    test_name=f"USPAP Test {test_func.__name__}",
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
                overall_grade = "Exemplary"
            elif compliance_score >= 0.8:
                overall_grade = "Compliant"
            elif compliance_score >= 0.6:
                overall_grade = "Conditionally Compliant"
            else:
                overall_grade = "Non-Compliant"
        
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
            'methodology_score': next((t.value for t in tests if t.test_name == "Methodology Defensibility"), "N/A"),
            'market_analysis_score': next((t.value for t in tests if t.test_name == "Market Analysis Adequacy"), "N/A"),
            'reasonableness_score': next((t.value for t in tests if t.test_name == "Reasonableness of Results"), "N/A")
        }
        
        self.logger.info(f"USPAP validation completed: {overall_grade} ({compliance_score:.2%} compliance)")
        
        return ValidationResult(
            standard="USPAP",
            overall_grade=overall_grade,
            compliance_score=compliance_score,
            individual_tests=tests,
            summary=summary
        )