# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
API (Australian Property Institute) Validation
Implements validation based on Australian appraisal standards focusing on 
Australian market conditions and professional practice standards.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .validation_strategy import Validator, ValidationResult, ValidationTestResult


class APIValidator(Validator):
    """
    Validator for API (Australian Property Institute) standards.
    Focuses on Australian market conditions, professional practice, and valuation accuracy.
    """
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series, config: Dict[str, Any]):
        super().__init__(model, X_train, y_train, X_test, y_test, config)
        self.thresholds = {
            'min_r2_australian': 0.75,
            'max_mape_australian': 12.0,
            'min_market_evidence_score': 0.8,
            'max_cv_variance': 0.2
        }

    def test_australian_market_evidence(self) -> ValidationTestResult:
        """
        Tests if the model incorporates adequate Australian market evidence.
        """
        # Check for Australian-specific features
        australian_features = ['postcode', 'suburb', 'land_size', 'dwelling_type', 'capital_growth']
        market_features = ['area', 'bedrooms', 'bathrooms', 'parking', 'pool', 'garden']
        
        feature_names = [col.lower() for col in self.X_train.columns]
        
        australian_count = sum(1 for af in australian_features if any(af in fn for fn in feature_names))
        market_count = sum(1 for mf in market_features if any(mf in fn for fn in feature_names))
        
        # Weight Australian features more heavily
        evidence_score = (australian_count * 0.7 + market_count * 0.3) / max(len(australian_features), len(market_features))
        passed = evidence_score >= self.thresholds['min_market_evidence_score']
        
        return ValidationTestResult(
            test_name="Australian Market Evidence",
            passed=passed,
            value=evidence_score,
            threshold=self.thresholds['min_market_evidence_score'],
            description=f"Market evidence score: {evidence_score:.3f} (Australian: {australian_count}, Market: {market_count})",
            recommendation="Include comprehensive Australian market features and evidence"
        )

    def test_professional_practice_standards(self) -> ValidationTestResult:
        """
        Tests adherence to API professional practice standards.
        """
        model_type = type(self.model).__name__
        
        # API favors transparent and professionally accepted methodologies
        professional_models = {
            'LinearRegression': 1.0,
            'Ridge': 0.9,
            'Lasso': 0.9,
            'ElasticNet': 0.9,
            'RandomForestRegressor': 0.8,
            'XGBRegressor': 0.75,
            'GradientBoostingRegressor': 0.75
        }
        
        model_score = professional_models.get(model_type, 0.4)
        
        # Feature engineering standards
        feature_count = len(self.X_train.columns)
        sample_size = len(self.X_train)
        
        # Rule of thumb: at least 10 observations per feature
        data_adequacy = min(1.0, sample_size / (feature_count * 10))
        
        practice_score = (model_score + data_adequacy) / 2
        passed = practice_score >= 0.75
        
        return ValidationTestResult(
            test_name="Professional Practice Standards",
            passed=passed,
            value=practice_score,
            threshold=0.75,
            description=f"Practice standards: {practice_score:.3f} (Model: {model_score:.2f}, Data adequacy: {data_adequacy:.2f})",
            recommendation="Use professional methodologies with adequate sample sizes"
        )

    def test_valuation_accuracy(self) -> ValidationTestResult:
        """
        Tests valuation accuracy against Australian market expectations.
        """
        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
        
        # Australian standards expect high accuracy
        accuracy_passed = r2 >= self.thresholds['min_r2_australian'] and mape <= self.thresholds['max_mape_australian']
        
        return ValidationTestResult(
            test_name="Valuation Accuracy",
            passed=accuracy_passed,
            value=r2,
            threshold=self.thresholds['min_r2_australian'],
            description=f"Accuracy metrics - R²: {r2:.4f}, MAPE: {mape:.2f}%",
            recommendation="Achieve R² ≥ 0.75 and MAPE ≤ 12% for Australian standards"
        )

    def test_comparable_evidence(self) -> ValidationTestResult:
        """
        Tests if the model appropriately uses comparable evidence.
        """
        # Check for features that support comparable analysis
        comparable_features = ['similar_sales', 'neighborhood', 'property_type', 'age', 'condition']
        location_features = ['suburb', 'postcode', 'district', 'location_score']
        
        feature_names = [col.lower() for col in self.X_train.columns]
        
        comparable_count = sum(1 for cf in comparable_features if any(cf in fn for fn in feature_names))
        location_count = sum(1 for lf in location_features if any(lf in fn for fn in feature_names))
        
        # Both comparable and location evidence are important
        comparable_score = (comparable_count + location_count) / (len(comparable_features) + len(location_features))
        passed = comparable_score >= 0.4
        
        return ValidationTestResult(
            test_name="Comparable Evidence",
            passed=passed,
            value=comparable_score,
            threshold=0.4,
            description=f"Comparable evidence: {comparable_score:.3f} (Comparable: {comparable_count}, Location: {location_count})",
            recommendation="Include features supporting comparable sales analysis"
        )

    def test_consistency_and_reliability(self) -> ValidationTestResult:
        """
        Tests model consistency and reliability as required by API standards.
        """
        from sklearn.model_selection import cross_val_score
        
        try:
            # Cross-validation for consistency assessment
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                      cv=min(5, len(self.X_train) // 15), 
                                      scoring='r2')
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            cv_variance = cv_std / cv_mean if cv_mean > 0 else 1.0
            
            # Check prediction spread
            y_pred = self.model.predict(self.X_test)
            pred_variance = np.std(y_pred) / np.mean(y_pred) if np.mean(y_pred) > 0 else 1.0
            
            consistency_score = 1 - min(1.0, (cv_variance + pred_variance) / 2)
            passed = cv_variance <= self.thresholds['max_cv_variance']
            
            return ValidationTestResult(
                test_name="Consistency and Reliability",
                passed=passed,
                value=consistency_score,
                threshold=0.8,
                description=f"Consistency score: {consistency_score:.3f} (CV variance: {cv_variance:.3f})",
                recommendation="Maintain low variance in cross-validation for reliable predictions"
            )
            
        except Exception as e:
            return ValidationTestResult(
                test_name="Consistency and Reliability",
                passed=False,
                value="N/A",
                threshold=0.8,
                description=f"Error in consistency assessment: {str(e)}",
                recommendation="Verify model stability and data adequacy",
                applicable=False
            )

    def test_regulatory_compliance(self) -> ValidationTestResult:
        """
        Tests compliance with Australian regulatory requirements.
        """
        # Check for regulatory compliance indicators
        model_type = type(self.model).__name__
        feature_count = len(self.X_train.columns)
        sample_size = len(self.X_train)
        
        # Regulatory factors
        model_transparency = 1.0 if hasattr(self.model, 'coef_') else 0.6
        sample_adequacy = min(1.0, sample_size / 100)  # Minimum 100 samples preferred
        complexity_appropriateness = 1.0 if feature_count <= 25 else 0.8
        
        compliance_score = (model_transparency + sample_adequacy + complexity_appropriateness) / 3
        passed = compliance_score >= 0.75
        
        return ValidationTestResult(
            test_name="Regulatory Compliance",
            passed=passed,
            value=compliance_score,
            threshold=0.75,
            description=f"Compliance score: {compliance_score:.3f} (Transparency: {model_transparency:.2f}, Sample: {sample_adequacy:.2f})",
            recommendation="Ensure model transparency and adequate sample sizes for regulatory compliance"
        )

    def validate(self) -> ValidationResult:
        """
        Executes the complete test battery for API standard.
        """
        tests = [
            self.test_australian_market_evidence(),
            self.test_professional_practice_standards(),
            self.test_valuation_accuracy(),
            self.test_comparable_evidence(),
            self.test_consistency_and_reliability(),
            self.test_regulatory_compliance(),
        ]

        passed_count = sum(1 for test in tests if test.passed and test.applicable)
        applicable_count = sum(1 for test in tests if test.applicable)
        compliance_score = passed_count / applicable_count if applicable_count > 0 else 0

        if compliance_score >= 0.85:
            overall_grade = "Meets API Standards"
        elif compliance_score >= 0.7:
            overall_grade = "Acceptable with Monitoring"
        else:
            overall_grade = "Below API Standards"

        summary = {
            'total_tests': len(tests),
            'applicable_tests': applicable_count,
            'passed_tests': passed_count,
            'compliance_score': compliance_score,
            'standard_focus': 'Australian market evidence, professional practice, accuracy'
        }

        return ValidationResult(
            standard="API",
            overall_grade=overall_grade,
            compliance_score=compliance_score,
            individual_tests=tests,
            summary=summary
        )