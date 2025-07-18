# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Base validation strategy for real estate appraisal models.
Provides abstract interface for different validation standards (NBR 14653, USPAP, EVS).
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class ValidationTestResult:
    """Result of a single validation test."""
    test_name: str
    passed: bool
    value: Union[float, str]
    threshold: float
    description: str
    recommendation: str
    applicable: bool = True
    warning: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result for any standard."""
    standard: str
    overall_grade: str
    compliance_score: float
    individual_tests: List[ValidationTestResult]
    summary: Dict[str, Any]


class Validator(ABC):
    """
    Abstract base class for validation strategies.
    
    Different appraisal standards (NBR 14653, USPAP, EVS) can implement
    their own validation logic by inheriting from this class.
    """
    
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                 X_test: pd.DataFrame, y_test: pd.Series, config: Dict[str, Any]):
        """
        Initialize validator with model and datasets.
        
        Args:
            model: Trained model to validate
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            config: Configuration parameters
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.config = config
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """
        Execute the validation battery and return standardized result.
        
        Returns:
            ValidationResult: Standardized validation result
        """
        pass
    
    def generate_report(self, validation_result: ValidationResult) -> str:
        """
        Generate a textual report of the validation results.
        
        Args:
            validation_result: Validation result to report
            
        Returns:
            str: Formatted validation report
        """
        report = []
        report.append("=" * 60)
        report.append(f"VALIDATION REPORT - {validation_result.standard}")
        report.append("=" * 60)
        report.append("")
        
        # General summary
        report.append(f"OVERALL GRADE: {validation_result.overall_grade}")
        report.append(f"COMPLIANCE SCORE: {validation_result.compliance_score:.1%}")
        
        passed_tests = sum(1 for test in validation_result.individual_tests if test.passed)
        total_tests = len(validation_result.individual_tests)
        report.append(f"TESTS PASSED: {passed_tests}/{total_tests}")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)
        
        for test in validation_result.individual_tests:
            status = "✓ PASSED" if test.passed else "✗ FAILED"
            if not test.applicable:
                status = "- NOT APPLICABLE"
            
            report.append(f"{test.test_name}: {status}")
            if test.applicable:
                if isinstance(test.value, (int, float)):
                    report.append(f"  Value: {test.value:.4f} (Threshold: {test.threshold:.4f})")
                else:
                    report.append(f"  Value: {test.value}")
                report.append(f"  {test.description}")
                
                if test.warning:
                    report.append(f"  Warning: {test.warning}")
                
                report.append(f"  Recommendation: {test.recommendation}")
            else:
                report.append(f"  {test.description}")
            report.append("")
        
        # Summary from validation result
        if validation_result.summary:
            report.append("SUMMARY STATISTICS:")
            report.append("-" * 40)
            for key, value in validation_result.summary.items():
                if isinstance(value, (int, float)):
                    report.append(f"{key}: {value:.4f}")
                else:
                    report.append(f"{key}: {value}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class ValidationContext:
    """
    Context class for managing different validation strategies.
    """
    
    def __init__(self, validator: Validator):
        """
        Initialize with a specific validator.
        
        Args:
            validator: Validator instance to use
        """
        self._validator = validator
    
    def set_validator(self, validator: Validator):
        """
        Change the validation strategy.
        
        Args:
            validator: New validator instance
        """
        self._validator = validator
    
    def execute_validation(self) -> ValidationResult:
        """
        Execute validation using the current strategy.
        
        Returns:
            ValidationResult: Result of the validation
        """
        return self._validator.validate()
    
    def generate_report(self, validation_result: ValidationResult) -> str:
        """
        Generate report using the current validator.
        
        Args:
            validation_result: Validation result to report
            
        Returns:
            str: Formatted report
        """
        return self._validator.generate_report(validation_result)


class ValidatorFactory:
    """Factory for creating appropriate validators based on standard."""
    
    @staticmethod
    def create_validator(standard: str, model, X_train: pd.DataFrame, 
                        y_train: pd.Series, X_test: pd.DataFrame, 
                        y_test: pd.Series, config: Dict[str, Any]) -> Validator:
        """
        Create appropriate validator based on standard.
        
        Args:
            standard: Validation standard ('NBR14653', 'USPAP', 'EVS')
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            config: Configuration parameters
            
        Returns:
            Validator: Appropriate validator instance
            
        Raises:
            ValueError: If standard is not supported
        """
        if standard.upper() == 'NBR14653':
            from .nbr14653_validation import NBR14653Validator
            return NBR14653Validator(model, X_train, y_train, X_test, y_test, config)
        elif standard.upper() == 'USPAP':
            from .uspap_validation import USPAPValidator
            return USPAPValidator(model, X_train, y_train, X_test, y_test, config)
        elif standard.upper() == 'EVS':
            from .evs_validation import EVSValidator
            return EVSValidator(model, X_train, y_train, X_test, y_test, config)
        else:
            raise ValueError(f"Unsupported validation standard: {standard}")