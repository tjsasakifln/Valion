# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
RICS (Royal Institution of Chartered Surveyors) "Red Book" Validation
Implements validation based on RICS standards focusing on market value,
professional judgment, and due diligence.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .validation_strategy import Validator, ValidationResult, ValidationTestResult


class RICSValidator(Validator):
    """
    Validator for RICS "Red Book" standards.
    Focuses on market value basis, professional competence, and reporting clarity.
    """
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series, config: Dict[str, Any]):
        super().__init__(model, X_train, y_train, X_test, y_test, config)
        self.thresholds = {
            'min_r2_reasonableness': 0.60,
            'max_mape_accuracy': 20.0,
            'professional_competence_score': 0.7
        }

    def test_market_value_basis(self) -> ValidationTestResult:
        """
        Tests if the valuation provides a clear and unambiguous basis for Market Value.
        """
        # Logic to verify if the model and data support a clear market value basis
        # per RICS standards - checking for market-defining features
        market_features = ['area', 'localizacao', 'valor_m2', 'location_score', 'area_total']
        present_features = [f for f in market_features if f in self.X_train.columns]
        
        market_coverage = len(present_features) / len(market_features)
        passed = market_coverage >= 0.6  # At least 60% of market features present
        
        return ValidationTestResult(
            test_name="Market Value Basis",
            passed=passed,
            value=f"{len(present_features)}/{len(market_features)} features",
            threshold=0.6,
            description=f"Market-defining features coverage: {market_coverage:.1%}",
            recommendation="Ensure features clearly related to market value are present."
        )

    def test_professional_competence_and_due_diligence(self) -> ValidationTestResult:
        """
        Assesses if the model development process reflects professional competence.
        """
        # Logic to assess the quality of the modeling process
        # Verify if the model is not overly complex and is interpretable
        model_type = type(self.model).__name__
        acceptable_models = ['LinearRegression', 'ElasticNet', 'Ridge', 'Lasso', 'XGBRegressor', 'RandomForestRegressor']
        passed = model_type in acceptable_models
        
        return ValidationTestResult(
            test_name="Professional Competence and Due Diligence",
            passed=passed,
            value=model_type,
            threshold=1.0,
            description=f"Model type '{model_type}' reflects standard industry practice.",
            recommendation="Use well-established and interpretable models."
        )

    def test_reporting_clarity(self) -> ValidationTestResult:
        """
        Tests if the model provides clear and understandable results.
        """
        # Check if the model has interpretable coefficients
        has_coefficients = hasattr(self.model, 'coef_')
        feature_count = len(self.X_train.columns)
        
        # Model is considered clear if it has coefficients or reasonable feature count
        clarity_score = 1.0 if has_coefficients else 0.5
        if feature_count > 20:  # Too many features reduce clarity
            clarity_score *= 0.7
            
        passed = clarity_score >= 0.6
        
        return ValidationTestResult(
            test_name="Reporting Clarity",
            passed=passed,
            value=clarity_score,
            threshold=0.6,
            description=f"Clarity score: {clarity_score:.2f} (Features: {feature_count}, Interpretable: {has_coefficients})",
            recommendation="Use interpretable models with reasonable feature counts for clarity."
        )

    def test_due_diligence_accuracy(self) -> ValidationTestResult:
        """
        Tests if the model meets RICS accuracy standards for due diligence.
        """
        from sklearn.metrics import mean_absolute_percentage_error
        
        y_pred = self.model.predict(self.X_test)
        mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
        
        passed = mape <= self.thresholds['max_mape_accuracy']
        
        return ValidationTestResult(
            test_name="Due Diligence Accuracy",
            passed=passed,
            value=mape,
            threshold=self.thresholds['max_mape_accuracy'],
            description=f"MAPE = {mape:.2f}%",
            recommendation="MAPE should be ≤ 20% for professional standards"
        )

    def validate(self) -> ValidationResult:
        """
        Executes the complete test battery for RICS standard.
        """
        tests = [
            self.test_market_value_basis(),
            self.test_professional_competence_and_due_diligence(),
            self.test_reporting_clarity(),
            self.test_due_diligence_accuracy(),
        ]

        passed_count = sum(1 for test in tests if test.passed)
        compliance_score = passed_count / len(tests)

        if compliance_score >= 0.8:
            overall_grade = "Compliant"
        elif compliance_score >= 0.6:
            overall_grade = "Conditionally Compliant"
        else:
            overall_grade = "Non-Compliant"

        summary = {
            'total_tests': len(tests),
            'passed_tests': passed_count,
            'compliance_score': compliance_score,
            'standard_focus': 'Market value, professional competence, due diligence'
        }

        return ValidationResult(
            standard="RICS Red Book",
            overall_grade=overall_grade,
            compliance_score=compliance_score,
            individual_tests=tests,
            summary=summary
        )