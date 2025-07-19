# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Data Anonymization Service for Valion - Privacy and GDPR Compliance
"""

import pandas as pd
import hashlib
import uuid
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import random
from faker import Faker
import re

logger = logging.getLogger(__name__)

class DataAnonymizer:
    """
    Service for anonymizing sensitive data while maintaining statistical integrity.
    Supports GDPR compliance and privacy protection.
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the anonymizer with consistent seeding for reproducible results.
        
        Args:
            seed: Random seed for consistent anonymization
        """
        self.seed = seed
        self.faker = Faker()
        if seed:
            Faker.seed(seed)
            random.seed(seed)
        
        # Mapping of sensitive fields to anonymization methods
        self.pii_fields = {
            # User model fields
            'full_name': 'fake_name',
            'email': 'fake_email',
            'username': 'hash_preserve_format',
            'contact_email': 'fake_email',
            'contact_phone': 'fake_phone',
            'billing_address': 'fake_address',
            
            # Potentially sensitive property data
            'address': 'fake_address',
            'owner_name': 'fake_name',
            'owner_email': 'fake_email',
            'owner_phone': 'fake_phone',
            'property_id': 'hash_preserve_format',
            
            # IP addresses and user agents
            'ip_address': 'fake_ip',
            'user_agent': 'fake_user_agent',
            
            # File paths that might contain sensitive info
            'data_file_path': 'anonymize_file_path',
            'file_path': 'anonymize_file_path'
        }
        
        # Fields that should be completely removed
        self.sensitive_fields_to_remove = {
            'password', 'token', 'secret', 'key', 'credential', 'private'
        }
        
        logger.info("Data anonymizer initialized with seed: %s", seed)
    
    def anonymize_dataframe(self, df: pd.DataFrame, 
                          preserve_relationships: bool = True,
                          preserve_statistics: bool = True) -> pd.DataFrame:
        """
        Anonymize a DataFrame while preserving statistical properties.
        
        Args:
            df: Input DataFrame
            preserve_relationships: Whether to maintain relationships between records
            preserve_statistics: Whether to preserve statistical distributions
            
        Returns:
            Anonymized DataFrame
        """
        logger.info("Starting anonymization of DataFrame with %d rows, %d columns", 
                   len(df), len(df.columns))
        
        anonymized_df = df.copy()
        anonymization_log = []
        
        # Identify and handle PII columns
        for column in df.columns:
            column_lower = column.lower()
            
            # Skip if column should be removed entirely
            if any(sensitive in column_lower for sensitive in self.sensitive_fields_to_remove):
                anonymized_df = anonymized_df.drop(columns=[column])
                anonymization_log.append(f"Removed sensitive column: {column}")
                continue
            
            # Apply anonymization based on field type
            anonymization_method = self._get_anonymization_method(column)
            if anonymization_method:
                try:
                    anonymized_df[column] = self._apply_anonymization(
                        anonymized_df[column], anonymization_method, preserve_relationships
                    )
                    anonymization_log.append(f"Anonymized column '{column}' using method '{anonymization_method}'")
                except Exception as e:
                    logger.warning("Failed to anonymize column %s: %s", column, e)
        
        # Preserve statistical distributions for numeric columns if requested
        if preserve_statistics:
            anonymized_df = self._preserve_statistical_distributions(df, anonymized_df)
        
        logger.info("Anonymization completed. Applied %d transformations", len(anonymization_log))
        return anonymized_df
    
    def anonymize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize user data dictionary for export/audit purposes.
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            Anonymized user data
        """
        anonymized_data = user_data.copy()
        
        for field, value in user_data.items():
            if field.lower() in ['full_name', 'email', 'username']:
                anonymization_method = self._get_anonymization_method(field)
                if anonymization_method:
                    if isinstance(value, str):
                        anonymized_data[field] = self._apply_anonymization_single(
                            value, anonymization_method
                        )
        
        return anonymized_data
    
    def _get_anonymization_method(self, field_name: str) -> Optional[str]:
        """Get the appropriate anonymization method for a field."""
        field_lower = field_name.lower()
        
        for pii_field, method in self.pii_fields.items():
            if pii_field in field_lower:
                return method
        
        return None
    
    def _apply_anonymization(self, series: pd.Series, method: str, 
                           preserve_relationships: bool = True) -> pd.Series:
        """Apply anonymization method to a pandas Series."""
        if method == 'fake_name':
            if preserve_relationships:
                # Create consistent mapping for names
                unique_values = series.unique()
                mapping = {val: self.faker.name() for val in unique_values if pd.notna(val)}
                return series.map(lambda x: mapping.get(x, x))
            else:
                return series.apply(lambda x: self.faker.name() if pd.notna(x) else x)
        
        elif method == 'fake_email':
            if preserve_relationships:
                unique_values = series.unique()
                mapping = {val: self.faker.email() for val in unique_values if pd.notna(val)}
                return series.map(lambda x: mapping.get(x, x))
            else:
                return series.apply(lambda x: self.faker.email() if pd.notna(x) else x)
        
        elif method == 'fake_phone':
            return series.apply(lambda x: self.faker.phone_number() if pd.notna(x) else x)
        
        elif method == 'fake_address':
            return series.apply(lambda x: self.faker.address() if pd.notna(x) else x)
        
        elif method == 'fake_ip':
            return series.apply(lambda x: self.faker.ipv4() if pd.notna(x) else x)
        
        elif method == 'fake_user_agent':
            return series.apply(lambda x: self.faker.user_agent() if pd.notna(x) else x)
        
        elif method == 'hash_preserve_format':
            return series.apply(lambda x: self._hash_preserve_format(x) if pd.notna(x) else x)
        
        elif method == 'anonymize_file_path':
            return series.apply(lambda x: self._anonymize_file_path(x) if pd.notna(x) else x)
        
        else:
            logger.warning("Unknown anonymization method: %s", method)
            return series
    
    def _apply_anonymization_single(self, value: str, method: str) -> str:
        """Apply anonymization to a single value."""
        if method == 'fake_name':
            return self.faker.name()
        elif method == 'fake_email':
            return self.faker.email()
        elif method == 'fake_phone':
            return self.faker.phone_number()
        elif method == 'fake_address':
            return self.faker.address()
        elif method == 'fake_ip':
            return self.faker.ipv4()
        elif method == 'fake_user_agent':
            return self.faker.user_agent()
        elif method == 'hash_preserve_format':
            return self._hash_preserve_format(value)
        elif method == 'anonymize_file_path':
            return self._anonymize_file_path(value)
        else:
            return value
    
    def _hash_preserve_format(self, value: str) -> str:
        """Create a hash that preserves the general format of the original."""
        if not value:
            return value
        
        # Create a deterministic hash
        hash_obj = hashlib.sha256(f"{value}{self.seed}".encode())
        hash_hex = hash_obj.hexdigest()[:len(value)]
        
        # Preserve character types (alpha, numeric, special)
        result = ""
        hash_index = 0
        
        for char in value:
            if hash_index >= len(hash_hex):
                hash_index = 0
            
            if char.isalpha():
                result += chr(ord('a') + (ord(hash_hex[hash_index]) % 26))
            elif char.isdigit():
                result += str(ord(hash_hex[hash_index]) % 10)
            else:
                result += char
            
            hash_index += 1
        
        return result
    
    def _anonymize_file_path(self, file_path: str) -> str:
        """Anonymize file paths while preserving directory structure."""
        if not file_path:
            return file_path
        
        # Replace filename with anonymized version
        import os
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Keep extension
        name, ext = os.path.splitext(filename)
        anonymized_name = f"file_{hashlib.md5(f'{name}{self.seed}'.encode()).hexdigest()[:8]}"
        
        return os.path.join(directory, f"{anonymized_name}{ext}")
    
    def _preserve_statistical_distributions(self, original_df: pd.DataFrame, 
                                          anonymized_df: pd.DataFrame) -> pd.DataFrame:
        """Preserve statistical distributions for numeric columns."""
        for column in original_df.select_dtypes(include=['number']).columns:
            if column in anonymized_df.columns:
                # Only preserve distributions for non-ID columns
                if not any(id_term in column.lower() for id_term in ['id', '_id', 'key']):
                    original_stats = original_df[column].describe()
                    
                    # Add noise while preserving distribution
                    if original_stats['std'] > 0:
                        noise_factor = 0.01  # 1% noise
                        noise = pd.Series([
                            random.gauss(0, original_stats['std'] * noise_factor) 
                            for _ in range(len(anonymized_df))
                        ])
                        anonymized_df[column] = anonymized_df[column] + noise
        
        return anonymized_df
    
    def create_anonymization_report(self, original_df: pd.DataFrame, 
                                  anonymized_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a report of the anonymization process.
        
        Args:
            original_df: Original DataFrame
            anonymized_df: Anonymized DataFrame
            
        Returns:
            Anonymization report
        """
        report = {
            'anonymization_timestamp': datetime.utcnow().isoformat(),
            'original_shape': original_df.shape,
            'anonymized_shape': anonymized_df.shape,
            'columns_processed': [],
            'columns_removed': [],
            'pii_fields_found': [],
            'statistical_preservation': {}
        }
        
        # Identify processed columns
        for column in original_df.columns:
            if column not in anonymized_df.columns:
                report['columns_removed'].append(column)
            else:
                anonymization_method = self._get_anonymization_method(column)
                if anonymization_method:
                    report['columns_processed'].append({
                        'column': column,
                        'method': anonymization_method
                    })
                    report['pii_fields_found'].append(column)
        
        # Statistical comparison for numeric columns
        numeric_columns = original_df.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            if column in anonymized_df.columns:
                orig_stats = original_df[column].describe()
                anon_stats = anonymized_df[column].describe()
                
                report['statistical_preservation'][column] = {
                    'mean_preservation': abs(orig_stats['mean'] - anon_stats['mean']) / orig_stats['std'] if orig_stats['std'] > 0 else 0,
                    'std_preservation': abs(orig_stats['std'] - anon_stats['std']) / orig_stats['std'] if orig_stats['std'] > 0 else 0
                }
        
        return report

    def is_pii_field(self, field_name: str) -> bool:
        """
        Check if a field contains personally identifiable information.
        
        Args:
            field_name: Name of the field to check
            
        Returns:
            True if field is considered PII
        """
        field_lower = field_name.lower()
        
        # Check against known PII fields
        for pii_field in self.pii_fields.keys():
            if pii_field in field_lower:
                return True
        
        # Check against sensitive fields
        for sensitive_field in self.sensitive_fields_to_remove:
            if sensitive_field in field_lower:
                return True
        
        return False
    
    def get_pii_fields_in_dataframe(self, df: pd.DataFrame) -> List[str]:
        """
        Identify all PII fields in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that contain PII
        """
        pii_columns = []
        
        for column in df.columns:
            if self.is_pii_field(column):
                pii_columns.append(column)
        
        return pii_columns


# Global anonymizer instance
_anonymizer_instance: Optional[DataAnonymizer] = None


def get_anonymizer() -> DataAnonymizer:
    """Get singleton instance of the data anonymizer."""
    global _anonymizer_instance
    
    if _anonymizer_instance is None:
        _anonymizer_instance = DataAnonymizer()
    
    return _anonymizer_instance


def anonymize_for_export(data: Union[Dict, pd.DataFrame], 
                        preserve_relationships: bool = True) -> Union[Dict, pd.DataFrame]:
    """
    Convenience function to anonymize data for export.
    
    Args:
        data: Data to anonymize (dict or DataFrame)
        preserve_relationships: Whether to preserve relationships
        
    Returns:
        Anonymized data
    """
    anonymizer = get_anonymizer()
    
    if isinstance(data, pd.DataFrame):
        return anonymizer.anonymize_dataframe(data, preserve_relationships=preserve_relationships)
    elif isinstance(data, dict):
        return anonymizer.anonymize_user_data(data)
    else:
        raise ValueError(f"Unsupported data type for anonymization: {type(data)}")