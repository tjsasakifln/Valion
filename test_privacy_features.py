#!/usr/bin/env python3
"""
Privacy and Data Governance features validation for Valion
Tests GDPR compliance and data governance implementation
"""

import ast
import re
from pathlib import Path

class PrivacyValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []
        
    def log_error(self, component, error):
        self.errors.append(f"❌ {component}: {error}")
        print(f"❌ {component}: {error}")
    
    def log_warning(self, component, warning):
        self.warnings.append(f"⚠️ {component}: {warning}")
        print(f"⚠️ {component}: {warning}")
    
    def log_success(self, component, message):
        self.successes.append(f"✅ {component}: {message}")
        print(f"✅ {component}: {message}")
    
    def validate_anonymizer_implementation(self):
        """Validate data anonymization implementation"""
        print("🔒 Validating data anonymization implementation...")
        
        anonymizer_file = Path('src/core/anonymizer.py')
        if not anonymizer_file.exists():
            self.log_error("Anonymizer", "Anonymizer module not found")
            return
        
        with open(anonymizer_file, 'r') as f:
            content = f.read()
        
        # Check for key classes and functions
        if 'class DataAnonymizer' in content:
            self.log_success("Anonymizer", "DataAnonymizer class found")
        else:
            self.log_error("Anonymizer", "DataAnonymizer class missing")
        
        # Check for PII field mapping
        if 'pii_fields' in content:
            self.log_success("Anonymizer", "PII fields mapping found")
        else:
            self.log_error("Anonymizer", "PII fields mapping missing")
        
        # Check for anonymization methods
        anonymization_methods = [
            'anonymize_dataframe',
            'anonymize_user_data',
            '_apply_anonymization',
            '_preserve_statistical_distributions'
        ]
        
        for method in anonymization_methods:
            if f'def {method}' in content:
                self.log_success("Anonymizer", f"Method {method} implemented")
            else:
                self.log_error("Anonymizer", f"Method {method} missing")
        
        # Check for Faker integration
        if 'from faker import Faker' in content:
            self.log_success("Anonymizer", "Faker library integration found")
        else:
            self.log_warning("Anonymizer", "Faker library not integrated")
    
    def validate_database_privacy_methods(self):
        """Validate database privacy and GDPR methods"""
        print("\n🗄️ Validating database privacy methods...")
        
        db_file = Path('src/database/database.py')
        if not db_file.exists():
            self.log_error("Database", "Database module not found")
            return
        
        with open(db_file, 'r') as f:
            content = f.read()
        
        # Check for privacy methods
        privacy_methods = [
            'log_pii_access',
            'export_user_data', 
            'delete_user_data',
            'get_pii_access_log',
            'generate_privacy_report'
        ]
        
        for method in privacy_methods:
            if f'def {method}' in content:
                self.log_success("Database", f"Privacy method {method} implemented")
            else:
                self.log_error("Database", f"Privacy method {method} missing")
        
        # Check for GDPR compliance features
        gdpr_features = [
            'gdpr_data_export',
            'gdpr_data_deletion', 
            'right_to_be_forgotten',
            'confirmation_token'
        ]
        
        found_gdpr = 0
        for feature in gdpr_features:
            if feature.lower() in content.lower():
                found_gdpr += 1
        
        if found_gdpr >= 2:
            self.log_success("Database", f"GDPR compliance features found ({found_gdpr}/4)")
        else:
            self.log_warning("Database", f"Limited GDPR features ({found_gdpr}/4)")
    
    def validate_audit_trail_implementation(self):
        """Validate audit trail implementation"""
        print("\n📋 Validating audit trail implementation...")
        
        # Check models
        models_file = Path('src/database/models.py')
        if models_file.exists():
            with open(models_file, 'r') as f:
                content = f.read()
            
            if 'class AuditTrail' in content:
                self.log_success("AuditTrail", "AuditTrail model found")
                
                # Check for immutability features
                if 'prevent_audit_modification' in content:
                    self.log_success("AuditTrail", "Immutability protection found")
                else:
                    self.log_warning("AuditTrail", "No immutability protection detected")
                
                # Check for PII access tracking
                if 'access_pii_data' in content:
                    self.log_success("AuditTrail", "PII access tracking found")
                else:
                    self.log_warning("AuditTrail", "No PII access tracking")
            else:
                self.log_error("AuditTrail", "AuditTrail model missing")
        
        # Check database implementation
        db_file = Path('src/database/database.py')
        if db_file.exists():
            with open(db_file, 'r') as f:
                content = f.read()
            
            if 'create_audit_trail' in content:
                self.log_success("AuditTrail", "Audit trail creation method found")
            else:
                self.log_error("AuditTrail", "Audit trail creation method missing")
    
    def validate_api_privacy_endpoints(self):
        """Validate API privacy endpoints"""
        print("\n🌐 Validating API privacy endpoints...")
        
        api_file = Path('src/api/main.py')
        if not api_file.exists():
            self.log_error("API", "API main file not found")
            return
        
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for privacy endpoints
        privacy_endpoints = [
            '/privacy/report',
            '/privacy/pii-access-log',
            '/privacy/export-user-data',
            '/privacy/export-my-data',
            '/privacy/delete-user-data',
            '/privacy/my-activity'
        ]
        
        for endpoint in privacy_endpoints:
            if endpoint in content:
                self.log_success("API", f"Privacy endpoint {endpoint} found")
            else:
                self.log_error("API", f"Privacy endpoint {endpoint} missing")
        
        # Check for security measures
        if 'require_admin' in content or 'admin' in content:
            self.log_success("API", "Admin-only endpoint protection found")
        else:
            self.log_warning("API", "No admin protection patterns detected")
        
        # Check for input validation
        privacy_models = ['ExportUserDataRequest', 'DeleteUserDataRequest', 'ExportMyDataRequest']
        found_models = 0
        for model in privacy_models:
            if model in content:
                found_models += 1
        
        if found_models >= 2:
            self.log_success("API", f"Input validation models found ({found_models}/3)")
        else:
            self.log_warning("API", f"Limited input validation ({found_models}/3)")
    
    def validate_frontend_privacy_interface(self):
        """Validate frontend privacy interface"""
        print("\n🖥️ Validating frontend privacy interface...")
        
        frontend_file = Path('frontend.py')
        if not frontend_file.exists():
            self.log_error("Frontend", "Frontend file not found")
            return
        
        with open(frontend_file, 'r') as f:
            content = f.read()
        
        # Check for privacy pages
        privacy_pages = [
            'show_privacy_governance_page',
            'show_my_data_page'
        ]
        
        for page in privacy_pages:
            if f'def {page}' in content:
                self.log_success("Frontend", f"Privacy page {page} found")
            else:
                self.log_error("Frontend", f"Privacy page {page} missing")
        
        # Check for privacy features in menu
        if 'Privacy & Data' in content:
            self.log_success("Frontend", "Privacy menu option found")
        else:
            self.log_error("Frontend", "Privacy menu option missing")
        
        # Check for GDPR features in UI
        gdpr_ui_features = [
            'Export My Data',
            'Delete User Data',
            'right of access',
            'right to be forgotten',
            'anonymize'
        ]
        
        found_ui_features = 0
        for feature in gdpr_ui_features:
            if feature in content:
                found_ui_features += 1
        
        if found_ui_features >= 3:
            self.log_success("Frontend", f"GDPR UI features found ({found_ui_features}/5)")
        else:
            self.log_warning("Frontend", f"Limited GDPR UI features ({found_ui_features}/5)")
    
    def validate_compliance_documentation(self):
        """Validate compliance documentation"""
        print("\n📚 Validating compliance documentation...")
        
        # Check README for privacy mentions
        readme_file = Path('README.md')
        if readme_file.exists():
            with open(readme_file, 'r') as f:
                content = f.read()
            
            privacy_keywords = ['GDPR', 'privacy', 'data governance', 'compliance', 'anonymization']
            found_keywords = [kw for kw in privacy_keywords if kw.lower() in content.lower()]
            
            if len(found_keywords) >= 3:
                self.log_success("Documentation", f"Privacy documentation found ({len(found_keywords)}/5 keywords)")
            else:
                self.log_warning("Documentation", f"Limited privacy documentation ({len(found_keywords)}/5 keywords)")
        else:
            self.log_warning("Documentation", "README file not found")
        
        # Check for privacy policy or compliance docs
        doc_files = ['PRIVACY.md', 'COMPLIANCE.md', 'GDPR.md', 'DATA_GOVERNANCE.md']
        found_docs = [doc for doc in doc_files if Path(doc).exists()]
        
        if found_docs:
            self.log_success("Documentation", f"Compliance documents found: {', '.join(found_docs)}")
        else:
            self.log_warning("Documentation", "No specific compliance documentation found")
    
    def validate_security_measures(self):
        """Validate security measures for privacy"""
        print("\n🛡️ Validating security measures...")
        
        # Check for secure deletion patterns
        files_to_check = [
            'src/database/database.py',
            'src/auth/auth_service.py',
            'src/auth/dependencies.py'
        ]
        
        security_patterns = [
            'confirmation_token',
            'hash',
            'encrypt',
            'sanitize',
            'validate',
            'permission',
            'authorization'
        ]
        
        total_security_score = 0
        for file_path in files_to_check:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                file_score = 0
                for pattern in security_patterns:
                    if pattern in content.lower():
                        file_score += 1
                
                total_security_score += file_score
                
                if file_score >= 3:
                    self.log_success("Security", f"{file_path} has good security patterns ({file_score}/7)")
                else:
                    self.log_warning("Security", f"{file_path} has limited security patterns ({file_score}/7)")
        
        # Overall security assessment
        max_possible = len(files_to_check) * len(security_patterns)
        security_percentage = (total_security_score / max_possible) * 100
        
        if security_percentage >= 60:
            self.log_success("Security", f"Overall security score: {security_percentage:.1f}%")
        else:
            self.log_warning("Security", f"Overall security score: {security_percentage:.1f}% - needs improvement")
    
    def run_validation(self):
        """Run complete privacy features validation"""
        print("🚀 Running Privacy & Data Governance Validation...")
        print("=" * 70)
        
        self.validate_anonymizer_implementation()
        self.validate_database_privacy_methods()
        self.validate_audit_trail_implementation()
        self.validate_api_privacy_endpoints()
        self.validate_frontend_privacy_interface()
        self.validate_compliance_documentation()
        self.validate_security_measures()
        
        # Generate summary
        print("\n" + "=" * 70)
        print("📊 PRIVACY VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"✅ Successes: {len(self.successes)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.errors:
            print("\n❌ CRITICAL ISSUES:")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print("\n⚠️ IMPROVEMENT OPPORTUNITIES:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # Calculate privacy compliance score
        total_checks = len(self.successes) + len(self.warnings) + len(self.errors)
        if total_checks > 0:
            compliance_score = (len(self.successes) / total_checks) * 100
            print(f"\n🎯 Privacy Compliance Score: {compliance_score:.1f}%")
            
            if compliance_score >= 90:
                print("🟢 EXCELLENT: Privacy implementation exceeds requirements")
            elif compliance_score >= 75:
                print("🟡 GOOD: Privacy implementation meets most requirements")
            elif compliance_score >= 60:
                print("🟠 ADEQUATE: Privacy implementation needs some improvements")
            else:
                print("🔴 POOR: Privacy implementation needs significant work")
        
        # GDPR readiness assessment
        gdpr_keywords = ['gdpr', 'right to be forgotten', 'right of access', 'data export', 'data deletion']
        all_content = ""
        
        for file_path in ['src/api/main.py', 'frontend.py', 'src/database/database.py', 'README.md']:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    all_content += f.read().lower()
        
        gdpr_features_found = sum(1 for keyword in gdpr_keywords if keyword in all_content)
        gdpr_percentage = (gdpr_features_found / len(gdpr_keywords)) * 100
        
        print(f"\n🇪🇺 GDPR Readiness: {gdpr_percentage:.1f}% ({gdpr_features_found}/{len(gdpr_keywords)} features)")
        
        return len(self.errors) <= 3 and compliance_score >= 60

if __name__ == "__main__":
    validator = PrivacyValidator()
    success = validator.run_validation()
    
    exit(0 if success else 1)