#!/usr/bin/env python3
"""
Simulation script for complete Valion system test
Tests all major components and identifies potential errors
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path

# Set environment variables for testing
os.environ['SECRET_KEY'] = 'test-secret-key-for-simulation-only'
os.environ['DATABASE_URL'] = 'sqlite:///./test_simulation.db'
os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
os.environ['ENVIRONMENT'] = 'testing'
os.environ['DEBUG'] = 'true'

class SystemSimulator:
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
    
    def test_imports(self):
        """Test all critical imports"""
        print("\n🧪 Testing Python imports...")
        
        # Test core modules
        try:
            from src.core.data_loader import DataLoader
            self.log_success("Core", "DataLoader imported successfully")
        except Exception as e:
            self.log_error("Core", f"DataLoader import failed: {e}")
        
        try:
            from src.core.anonymizer import get_anonymizer
            self.log_success("Privacy", "Anonymizer imported successfully")
        except Exception as e:
            self.log_error("Privacy", f"Anonymizer import failed: {e}")
        
        try:
            from src.database.database import get_database_manager
            self.log_success("Database", "DatabaseManager imported successfully")
        except Exception as e:
            self.log_error("Database", f"DatabaseManager import failed: {e}")
        
        try:
            from src.database.models import User, Tenant, AuditTrail
            self.log_success("Models", "Database models imported successfully")
        except Exception as e:
            self.log_error("Models", f"Database models import failed: {e}")
    
    def test_anonymizer(self):
        """Test data anonymization functionality"""
        print("\n🔒 Testing data anonymization...")
        
        try:
            from src.core.anonymizer import DataAnonymizer
            import pandas as pd
            
            # Create test data
            test_data = pd.DataFrame({
                'full_name': ['João Silva', 'Maria Santos'],
                'email': ['joao@email.com', 'maria@email.com'],
                'username': ['joao123', 'maria456'],
                'address': ['Rua A, 123', 'Rua B, 456'],
                'property_value': [500000, 750000]
            })
            
            anonymizer = DataAnonymizer(seed=42)
            anonymized_data = anonymizer.anonymize_dataframe(test_data)
            
            # Check if PII fields were anonymized
            if anonymized_data['full_name'].iloc[0] != test_data['full_name'].iloc[0]:
                self.log_success("Anonymization", "Names properly anonymized")
            else:
                self.log_error("Anonymization", "Names not anonymized")
            
            # Check if non-PII fields preserved
            if (anonymized_data['property_value'] == test_data['property_value']).all():
                self.log_success("Anonymization", "Non-PII data preserved")
            else:
                self.log_warning("Anonymization", "Non-PII data modified unexpectedly")
                
        except Exception as e:
            self.log_error("Anonymization", f"Anonymization test failed: {e}")
    
    def test_database_models(self):
        """Test database model definitions"""
        print("\n🗄️ Testing database models...")
        
        try:
            from src.database.models import Base, User, Tenant, Role, AuditTrail
            from sqlalchemy import create_engine
            
            # Create in-memory SQLite database
            engine = create_engine('sqlite:///:memory:')
            Base.metadata.create_all(engine)
            
            self.log_success("Database", "All tables created successfully")
            
            # Check table names
            table_names = list(Base.metadata.tables.keys())
            expected_tables = ['users', 'tenants', 'roles', 'audit_trails', 'projects', 'evaluations']
            
            for table in expected_tables:
                if table in table_names:
                    self.log_success("Database", f"Table '{table}' exists")
                else:
                    self.log_error("Database", f"Table '{table}' missing")
                    
        except Exception as e:
            self.log_error("Database", f"Database model test failed: {e}")
            traceback.print_exc()
    
    def test_api_structure(self):
        """Test API structure and endpoints"""
        print("\n🌐 Testing API structure...")
        
        try:
            import ast
            with open('src/api/main.py', 'r') as f:
                code = f.read()
            
            tree = ast.parse(code)
            
            # Count decorators with @app
            app_decorators = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if (isinstance(decorator, ast.Attribute) and 
                            isinstance(decorator.value, ast.Name) and 
                            decorator.value.id == 'app'):
                            app_decorators.append(f"{decorator.attr} /{node.name}")
            
            self.log_success("API", f"Found {len(app_decorators)} API endpoints")
            
            # Check for critical endpoints
            critical_endpoints = ['health', 'privacy/report', 'privacy/export-user-data']
            endpoint_content = open('src/api/main.py', 'r').read()
            
            for endpoint in critical_endpoints:
                if endpoint in endpoint_content:
                    self.log_success("API", f"Endpoint '{endpoint}' found")
                else:
                    self.log_error("API", f"Endpoint '{endpoint}' missing")
                    
        except Exception as e:
            self.log_error("API", f"API structure test failed: {e}")
    
    def test_configuration(self):
        """Test configuration files and environment"""
        print("\n⚙️ Testing configuration...")
        
        # Check required files
        required_files = [
            'requirements.txt',
            'docker-compose.yml',
            'Dockerfile',
            '.env.example',
            'frontend.py'
        ]
        
        for file in required_files:
            if Path(file).exists():
                self.log_success("Config", f"File '{file}' exists")
            else:
                self.log_error("Config", f"File '{file}' missing")
        
        # Check environment variables
        required_env = ['SECRET_KEY', 'DATABASE_URL']
        for env_var in required_env:
            if os.getenv(env_var):
                self.log_success("Config", f"Environment variable '{env_var}' set")
            else:
                self.log_error("Config", f"Environment variable '{env_var}' missing")
    
    def test_docker_config(self):
        """Test Docker configuration"""
        print("\n🐳 Testing Docker configuration...")
        
        try:
            # Check if docker-compose.yml is valid YAML
            import yaml
            with open('docker-compose.yml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required services
            required_services = ['postgres', 'redis', 'api', 'worker', 'frontend']
            services = config.get('services', {})
            
            for service in required_services:
                if service in services:
                    self.log_success("Docker", f"Service '{service}' configured")
                else:
                    self.log_error("Docker", f"Service '{service}' missing")
            
            # Check environment variables in services
            if 'api' in services:
                api_env = services['api'].get('environment', {})
                if 'DATABASE_URL' in str(api_env):
                    self.log_success("Docker", "API service has database configuration")
                else:
                    self.log_error("Docker", "API service missing database configuration")
                    
        except Exception as e:
            self.log_error("Docker", f"Docker configuration test failed: {e}")
    
    def test_frontend_structure(self):
        """Test frontend structure"""
        print("\n🖥️ Testing frontend structure...")
        
        try:
            with open('frontend.py', 'r') as f:
                content = f.read()
            
            # Check for critical functions
            critical_functions = [
                'show_privacy_governance_page',
                'show_my_data_page',
                'main',
                'login_user'
            ]
            
            for func in critical_functions:
                if f"def {func}" in content:
                    self.log_success("Frontend", f"Function '{func}' found")
                else:
                    self.log_error("Frontend", f"Function '{func}' missing")
            
            # Check for privacy features
            privacy_features = [
                'Privacy & Data',
                'export-user-data',
                'delete-user-data',
                'GDPR'
            ]
            
            for feature in privacy_features:
                if feature in content:
                    self.log_success("Frontend", f"Privacy feature '{feature}' implemented")
                else:
                    self.log_warning("Frontend", f"Privacy feature '{feature}' not found")
                    
        except Exception as e:
            self.log_error("Frontend", f"Frontend structure test failed: {e}")
    
    def run_full_simulation(self):
        """Run complete system simulation"""
        print("🚀 Starting Valion System Simulation...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        self.test_imports()
        self.test_configuration()
        self.test_database_models()
        self.test_anonymizer()
        self.test_api_structure()
        self.test_docker_config()
        self.test_frontend_structure()
        
        # Generate report
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("📊 SIMULATION RESULTS")
        print("=" * 60)
        
        print(f"⏱️ Total simulation time: {duration:.2f} seconds")
        print(f"✅ Successes: {len(self.successes)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.errors:
            print("\n🔥 CRITICAL ERRORS FOUND:")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # Calculate score
        total_tests = len(self.successes) + len(self.warnings) + len(self.errors)
        if total_tests > 0:
            score = (len(self.successes) / total_tests) * 100
            print(f"\n🎯 System Health Score: {score:.1f}%")
            
            if score >= 90:
                print("🟢 System is PRODUCTION READY")
            elif score >= 75:
                print("🟡 System is MOSTLY READY with minor issues")
            elif score >= 50:
                print("🟠 System has SIGNIFICANT ISSUES requiring attention")
            else:
                print("🔴 System has CRITICAL ISSUES and is NOT READY")
        
        return len(self.errors) == 0

if __name__ == "__main__":
    simulator = SystemSimulator()
    success = simulator.run_full_simulation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)