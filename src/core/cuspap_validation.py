# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
CUSPAP (Canadian Uniform Standards of Professional Appraisal Practice) Validation
Implements validation based on Canadian appraisal standards focusing on 
credible assignment results and professional competency.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .validation_strategy import Validator, ValidationResult, ValidationTestResult


class CUSPAPValidator(Validator):
    """
    Validator for CUSPAP standards.
    Focuses on credible assignment results, professional competency, and Canadian market conditions.
    """
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series, config: Dict[str, Any]):
        super().__init__(model, X_train, y_train, X_test, y_test, config)
        self.thresholds = {
            'min_r2_credibility': 0.70,
            'max_mape_canadian': 15.0,
            'min_competency_score': 0.75,
            'max_bias_tolerance': 0.05
        }

    def test_credible_assignment_results(self) -> ValidationTestResult:
        """
        Tests if the assignment results are credible per CUSPAP standards.
        """
        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
        
        # Credibility based on both accuracy and precision
        credibility_score = (r2 * 0.6) + ((100 - min(mape, 100)) / 100 * 0.4)
        passed = r2 >= self.thresholds['min_r2_credibility'] and mape <= self.thresholds['max_mape_canadian']
        
        return ValidationTestResult(
            test_name="Credible Assignment Results",
            passed=passed,
            value=credibility_score,
            threshold=0.7,
            description=f"Credibility score: {credibility_score:.3f} (R²: {r2:.3f}, MAPE: {mape:.1f}%)",
            recommendation="Results must be credible with R² ≥ 0.70 and MAPE ≤ 15%"
        )

    def test_professional_competency(self) -> ValidationTestResult:
        """
        Assesses professional competency through model selection and methodology.
        """
        model_type = type(self.model).__name__
        
        # Canadian standards favor transparent, well-established methods
        competent_models = {
            'LinearRegression': 1.0,
            'Ridge': 0.95,
            'Lasso': 0.95,
            'ElasticNet': 0.95,
            'RandomForestRegressor': 0.85,
            'XGBRegressor': 0.80,
            'GradientBoostingRegressor': 0.80
        }
        
        model_competency = competent_models.get(model_type, 0.5)
        
        # Feature engineering competency
        feature_count = len(self.X_train.columns)
        if 5 <= feature_count <= 20:
            feature_competency = 1.0
        elif feature_count <= 30:
            feature_competency = 0.8
        else:
            feature_competency = 0.6
        
        competency_score = (model_competency + feature_competency) / 2
        passed = competency_score >= self.thresholds['min_competency_score']
        
        return ValidationTestResult(
            test_name="Professional Competency",
            passed=passed,
            value=competency_score,
            threshold=self.thresholds['min_competency_score'],
            description=f"Competency score: {competency_score:.3f} (Model: {model_type}, Features: {feature_count})",
            recommendation="Use established models with appropriate feature engineering"
        )

    def test_canadian_market_relevance(self) -> ValidationTestResult:
        """
        Tests relevance to Canadian real estate market conditions.
        """
        # Check for Canadian-relevant features
        canadian_features = ['postal_code', 'province', 'mls', 'square_feet', 'lot_size']
        general_features = ['area', 'location', 'price', 'bedrooms', 'bathrooms']
        
        feature_names = [col.lower() for col in self.X_train.columns]
        canadian_score = sum(1 for cf in canadian_features if any(cf in fn for fn in feature_names))
        general_score = sum(1 for gf in general_features if any(gf in fn for fn in feature_names))
        
        relevance_score = (canadian_score * 0.6 + general_score * 0.4) / max(len(canadian_features), len(general_features))
        passed = relevance_score >= 0.4
        
        return ValidationTestResult(
            test_name="Canadian Market Relevance",
            passed=passed,
            value=relevance_score,
            threshold=0.4,
            description=f"Market relevance: {relevance_score:.3f} (Canadian: {canadian_score}, General: {general_score})",
            recommendation="Include features relevant to Canadian real estate market"
        )

    def test_unbiased_analysis(self) -> ValidationTestResult:
        """
        Tests for unbiased analysis as required by CUSPAP ethics.
        """
        # Check for systematic bias in predictions
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred
        
        # Calculate bias measures
        mean_bias = np.mean(residuals) / np.mean(self.y_test)
        median_bias = np.median(residuals) / np.median(self.y_test)
        
        # Test for systematic over/under-valuation
        bias_magnitude = max(abs(mean_bias), abs(median_bias))
        passed = bias_magnitude <= self.thresholds['max_bias_tolerance']
        
        return ValidationTestResult(
            test_name="Unbiased Analysis",
            passed=passed,
            value=bias_magnitude,
            threshold=self.thresholds['max_bias_tolerance'],
            description=f"Bias magnitude: {bias_magnitude:.4f} (Mean: {mean_bias:.4f}, Median: {median_bias:.4f})",
            recommendation="Maintain systematic bias ≤ 5% for ethical standards"
        )

    def test_due_diligence_documentation(self) -> ValidationTestResult:
        """
        Tests if the methodology provides adequate documentation for due diligence.
        """
        # Assess documentation readiness through model interpretability
        has_coefficients = hasattr(self.model, 'coef_')
        has_feature_importance = hasattr(self.model, 'feature_importances_')
        
        model_type = type(self.model).__name__
        feature_count = len(self.X_train.columns)
        
        # Documentation score based on interpretability
        interpretability_score = 0
        if has_coefficients:
            interpretability_score += 0.5
        if has_feature_importance:
            interpretability_score += 0.3
        if model_type in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
            interpretability_score += 0.3
        if feature_count <= 15:  # Manageable for documentation
            interpretability_score += 0.2
        
        interpretability_score = min(1.0, interpretability_score)
        passed = interpretability_score >= 0.7
        
        return ValidationTestResult(
            test_name="Due Diligence Documentation",
            passed=passed,
            value=interpretability_score,
            threshold=0.7,
            description=f"Documentation readiness: {interpretability_score:.3f}",
            recommendation="Use interpretable models that support comprehensive documentation"
        )

    def validate(self) -> ValidationResult:
        """
        Executes the complete test battery for CUSPAP standard.
        """
        tests = [
            self.test_credible_assignment_results(),
            self.test_professional_competency(),
            self.test_canadian_market_relevance(),
            self.test_unbiased_analysis(),
            self.test_due_diligence_documentation(),
        ]

        passed_count = sum(1 for test in tests if test.passed and test.applicable)
        applicable_count = sum(1 for test in tests if test.applicable)
        compliance_score = passed_count / applicable_count if applicable_count > 0 else 0

        if compliance_score >= 0.8:
            overall_grade = "Professional Standard Met"
        elif compliance_score >= 0.6:
            overall_grade = "Acceptable with Conditions"
        else:
            overall_grade = "Below Professional Standard"

        summary = {
            'total_tests': len(tests),
            'applicable_tests': applicable_count,
            'passed_tests': passed_count,
            'compliance_score': compliance_score,
            'standard_focus': 'Credible results, professional competency, Canadian market'
        }

        return ValidationResult(
            standard="CUSPAP",
            overall_grade=overall_grade,
            compliance_score=compliance_score,
            individual_tests=tests,
            summary=summary
        )