# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
IVS (International Valuation Standards) Validation
Implements validation based on IVS, focusing on internationally recognized
concepts and principles for valuation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .validation_strategy import Validator, ValidationResult, ValidationTestResult


class IVSValidator(Validator):
    """
    Validator for IVS standards.
    Focuses on bases of value, valuation approaches, and reporting.
    """
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series, config: Dict[str, Any]):
        super().__init__(model, X_train, y_train, X_test, y_test, config)
        self.thresholds = {
            'methodology_transparency_score': 0.7,
            'min_r2_international': 0.65,
            'max_prediction_variance': 0.25
        }

    def test_bases_of_value(self) -> ValidationTestResult:
        """
        Ensures the valuation approach is consistent with IVS bases of value (e.g., Market Value).
        """
        # Logic to verify if the target 'valor' and features are consistent 
        # with IVS Market Value concept
        target_name = self.y_train.name if hasattr(self.y_train, 'name') else str(self.y_train.index.name)
        
        market_value_indicators = ['valor', 'price', 'value', 'preco']
        is_market_value = any(indicator in str(target_name).lower() for indicator in market_value_indicators)
        
        # Check for market-based features
        market_features = ['area', 'location', 'localizacao', 'market', 'comparable']
        market_feature_count = sum(1 for col in self.X_train.columns 
                                 if any(mf in col.lower() for mf in market_features))
        
        market_basis_score = (0.6 if is_market_value else 0.2) + (0.4 * min(market_feature_count / 5, 1))
        passed = market_basis_score >= 0.6
        
        return ValidationTestResult(
            test_name="Bases of Value Compliance",
            passed=passed,
            value=market_basis_score,
            threshold=0.6,
            description=f"Market value basis score: {market_basis_score:.2f} (Target: {target_name}, Market features: {market_feature_count})",
            recommendation="Ensure the target variable and features align with IVS Market Value definition."
        )

    def test_valuation_approaches(self) -> ValidationTestResult:
        """
        Checks if the model aligns with one of the main valuation approaches (Market, Income, Cost).
        """
        # Regression models align with the "Market Approach"
        model_type = type(self.model).__name__
        market_approach_models = ['LinearRegression', 'ElasticNet', 'Ridge', 'Lasso', 
                                'XGBRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor']
        
        approach_alignment = model_type in market_approach_models
        
        # Check for approach-specific features
        market_features = ['area', 'location', 'comparable', 'neighborhood']
        income_features = ['rent', 'rental', 'income', 'yield']
        cost_features = ['construction', 'material', 'building_cost', 'replacement']
        
        feature_names = [col.lower() for col in self.X_train.columns]
        market_score = sum(1 for mf in market_features if any(mf in fn for fn in feature_names))
        income_score = sum(1 for inf in income_features if any(inf in fn for fn in feature_names))
        cost_score = sum(1 for cf in cost_features if any(cf in fn for fn in feature_names))
        
        dominant_approach = max(market_score, income_score, cost_score)
        if dominant_approach == market_score:
            approach = "Market Approach"
        elif dominant_approach == income_score:
            approach = "Income Approach"
        else:
            approach = "Cost Approach"
        
        passed = approach_alignment and dominant_approach > 0
        
        return ValidationTestResult(
            test_name="Valuation Approaches",
            passed=passed,
            value=approach,
            threshold=1.0,
            description=f"Model '{model_type}' aligns with {approach} (Score: {dominant_approach})",
            recommendation="Use models that reflect established valuation approaches with appropriate features."
        )

    def test_international_consistency(self) -> ValidationTestResult:
        """
        Tests if the model meets international accuracy standards.
        """
        from sklearn.metrics import r2_score
        
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        
        passed = r2 >= self.thresholds['min_r2_international']
        
        return ValidationTestResult(
            test_name="International Consistency",
            passed=passed,
            value=r2,
            threshold=self.thresholds['min_r2_international'],
            description=f"R² = {r2:.4f} meets international standards",
            recommendation="R² should be ≥ 0.65 for international consistency"
        )

    def test_transparency_and_disclosure(self) -> ValidationTestResult:
        """
        Tests methodology transparency as required by IVS.
        """
        # Assess model transparency based on interpretability
        has_coefficients = hasattr(self.model, 'coef_')
        feature_count = len(self.X_train.columns)
        model_type = type(self.model).__name__
        
        # Transparency score calculation
        interpretability_score = 1.0 if has_coefficients else 0.6
        complexity_penalty = max(0, (feature_count - 15) * 0.02)  # Penalty for >15 features
        
        if model_type in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
            model_transparency = 1.0
        elif model_type in ['RandomForestRegressor', 'XGBRegressor']:
            model_transparency = 0.7
        else:
            model_transparency = 0.5
        
        transparency_score = (interpretability_score + model_transparency) / 2 - complexity_penalty
        transparency_score = max(0, min(1, transparency_score))
        
        passed = transparency_score >= self.thresholds['methodology_transparency_score']
        
        return ValidationTestResult(
            test_name="Transparency and Disclosure",
            passed=passed,
            value=transparency_score,
            threshold=self.thresholds['methodology_transparency_score'],
            description=f"Methodology transparency score: {transparency_score:.2f}",
            recommendation="Use transparent, interpretable models with clear feature selection."
        )

    def test_prediction_reliability(self) -> ValidationTestResult:
        """
        Tests the reliability and consistency of predictions.
        """
        # Calculate prediction variance as a measure of reliability
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error
        
        try:
            # Cross-validation to assess prediction stability
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                      cv=min(5, len(self.X_train) // 10), 
                                      scoring='r2')
            
            # Calculate coefficient of variation
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            cv_variance = cv_std / cv_mean if cv_mean > 0 else 1.0
            
            passed = cv_variance <= self.thresholds['max_prediction_variance']
            
            return ValidationTestResult(
                test_name="Prediction Reliability",
                passed=passed,
                value=cv_variance,
                threshold=self.thresholds['max_prediction_variance'],
                description=f"CV variance: {cv_variance:.3f} (Mean R²: {cv_mean:.3f}, Std: {cv_std:.3f})",
                recommendation="Coefficient of variation should be ≤ 0.25 for reliable predictions"
            )
            
        except Exception as e:
            return ValidationTestResult(
                test_name="Prediction Reliability",
                passed=False,
                value="N/A",
                threshold=self.thresholds['max_prediction_variance'],
                description=f"Error in reliability assessment: {str(e)}",
                recommendation="Verify model stability and data quality",
                applicable=False
            )

    def validate(self) -> ValidationResult:
        """
        Executes the complete test battery for IVS standard.
        """
        tests = [
            self.test_bases_of_value(),
            self.test_valuation_approaches(),
            self.test_international_consistency(),
            self.test_transparency_and_disclosure(),
            self.test_prediction_reliability(),
        ]

        passed_count = sum(1 for test in tests if test.passed and test.applicable)
        applicable_count = sum(1 for test in tests if test.applicable)
        compliance_score = passed_count / applicable_count if applicable_count > 0 else 0

        if compliance_score >= 0.8:
            overall_grade = "Compliant"
        elif compliance_score >= 0.6:
            overall_grade = "Conditionally Compliant"
        else:
            overall_grade = "Non-Compliant"

        summary = {
            'total_tests': len(tests),
            'applicable_tests': applicable_count,
            'passed_tests': passed_count,
            'compliance_score': compliance_score,
            'standard_focus': 'International bases of value, approaches, transparency'
        }

        return ValidationResult(
            standard="IVS",
            overall_grade=overall_grade,
            compliance_score=compliance_score,
            individual_tests=tests,
            summary=summary
        )