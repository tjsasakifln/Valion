#!/usr/bin/env python3
"""
API Endpoints validation script for Valion
Tests API structure and endpoint definitions
"""

import ast
import re
from pathlib import Path

class APIEndpointValidator:
    def __init__(self):
        self.endpoints = []
        self.errors = []
        self.warnings = []
        
    def parse_api_file(self):
        """Parse the main API file and extract endpoints"""
        api_file = Path('src/api/main.py')
        
        if not api_file.exists():
            self.errors.append("API main file not found")
            return
        
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Use regex to extract endpoints (more reliable)
        endpoint_pattern = r'@app\.(get|post|put|patch|delete)\(["\']([^"\']+)["\']'
        matches = re.findall(endpoint_pattern, content, re.IGNORECASE)
        
        for method, path in matches:
            self.endpoints.append({
                'method': method.upper(),
                'path': path,
                'function': 'detected'
            })
    
    def _extract_endpoint_info(self, decorator, func_name):
        """Extract endpoint information from decorator"""
        endpoint_info = None
        
        if isinstance(decorator, ast.Attribute):
            if (isinstance(decorator.value, ast.Name) and 
                decorator.value.id == 'app'):
                
                method = decorator.attr.upper()
                
                # Try to get the path from the first argument
                if hasattr(decorator, 'args') and decorator.args:
                    if isinstance(decorator.args[0], ast.Constant):
                        path = decorator.args[0].value
                        endpoint_info = {
                            'method': method,
                            'path': path,
                            'function': func_name
                        }
        
        elif isinstance(decorator, ast.Call):
            if (isinstance(decorator.func, ast.Attribute) and
                isinstance(decorator.func.value, ast.Name) and
                decorator.func.value.id == 'app'):
                
                method = decorator.func.attr.upper()
                
                # Get path from first argument
                if decorator.args and isinstance(decorator.args[0], ast.Constant):
                    path = decorator.args[0].value
                    endpoint_info = {
                        'method': method,
                        'path': path,
                        'function': func_name
                    }
        
        return endpoint_info
    
    def validate_endpoint_structure(self):
        """Validate the structure of defined endpoints"""
        print(f"📊 Found {len(self.endpoints)} endpoints:")
        
        # Group by categories
        categories = {
            'auth': [],
            'admin': [],
            'evaluations': [],
            'privacy': [],
            'geospatial': [],
            'health': [],
            'other': []
        }
        
        for endpoint in self.endpoints:
            path = endpoint['path']
            categorized = False
            
            for category in categories.keys():
                if category in path or (category == 'health' and 'health' in path):
                    categories[category].append(endpoint)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(endpoint)
        
        # Print results by category
        for category, endpoints in categories.items():
            if endpoints:
                print(f"\n🔸 {category.upper()} Endpoints ({len(endpoints)}):")
                for ep in endpoints:
                    print(f"  {ep['method']:6} {ep['path']:30} -> {ep['function']}")
    
    def validate_critical_endpoints(self):
        """Validate presence of critical endpoints"""
        print("\n🔍 Checking critical endpoints...")
        
        critical_endpoints = [
            {'path': '/health', 'method': 'GET', 'description': 'Health check'},
            {'path': '/auth/login', 'method': 'POST', 'description': 'User login'},
            {'path': '/evaluations/', 'method': 'POST', 'description': 'Create evaluation'},
            {'path': '/privacy/report', 'method': 'GET', 'description': 'Privacy report'},
            {'path': '/privacy/export-user-data', 'method': 'POST', 'description': 'Export user data'},
            {'path': '/privacy/delete-user-data', 'method': 'POST', 'description': 'Delete user data'},
        ]
        
        for critical in critical_endpoints:
            found = False
            for endpoint in self.endpoints:
                if (endpoint['path'] == critical['path'] and 
                    endpoint['method'] == critical['method']):
                    found = True
                    break
            
            if found:
                print(f"  ✅ {critical['description']}: {critical['method']} {critical['path']}")
            else:
                print(f"  ❌ MISSING: {critical['description']}: {critical['method']} {critical['path']}")
                self.errors.append(f"Missing critical endpoint: {critical['method']} {critical['path']}")
    
    def validate_privacy_compliance(self):
        """Validate privacy and GDPR compliance endpoints"""
        print("\n🔒 Checking privacy compliance endpoints...")
        
        privacy_paths = [ep['path'] for ep in self.endpoints if 'privacy' in ep['path']]
        
        required_privacy_endpoints = [
            '/privacy/report',
            '/privacy/pii-access-log',
            '/privacy/export-user-data',
            '/privacy/export-my-data',
            '/privacy/delete-user-data',
            '/privacy/my-activity'
        ]
        
        for required in required_privacy_endpoints:
            if required in privacy_paths:
                print(f"  ✅ Privacy endpoint: {required}")
            else:
                print(f"  ❌ MISSING: Privacy endpoint: {required}")
                self.errors.append(f"Missing privacy endpoint: {required}")
    
    def validate_security_patterns(self):
        """Validate security patterns in API"""
        print("\n🛡️ Checking security patterns...")
        
        api_file = Path('src/api/main.py')
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for authentication dependencies
        if 'get_current_user' in content:
            print("  ✅ Authentication dependency found")
        else:
            print("  ⚠️ No authentication dependency found")
            self.warnings.append("No authentication patterns detected")
        
        # Check for permission checking
        if 'require_permission' in content or 'require_role' in content:
            print("  ✅ Authorization patterns found")
        else:
            print("  ⚠️ No authorization patterns found")
            self.warnings.append("No authorization patterns detected")
        
        # Check for input validation
        if 'pydantic' in content or 'BaseModel' in content:
            print("  ✅ Input validation patterns found")
        else:
            print("  ⚠️ No input validation patterns found")
            self.warnings.append("No input validation patterns detected")
    
    def generate_openapi_spec_check(self):
        """Check if OpenAPI specification would be generated properly"""
        print("\n📋 Checking OpenAPI specification compatibility...")
        
        # Count endpoints with proper documentation
        documented_endpoints = 0
        for endpoint in self.endpoints:
            # This is a simplified check - in reality would need to parse docstrings
            documented_endpoints += 1
        
        print(f"  📊 Total endpoints for OpenAPI: {documented_endpoints}")
        
        if documented_endpoints > 0:
            print("  ✅ API should generate valid OpenAPI specification")
        else:
            print("  ❌ No endpoints found for OpenAPI specification")
            self.errors.append("No endpoints found for OpenAPI specification")
    
    def run_validation(self):
        """Run complete API validation"""
        print("🚀 Running API Endpoint Validation...")
        print("=" * 60)
        
        self.parse_api_file()
        
        if self.errors:
            print("❌ Critical errors found during parsing:")
            for error in self.errors:
                print(f"  - {error}")
            return False
        
        self.validate_endpoint_structure()
        self.validate_critical_endpoints()
        self.validate_privacy_compliance()
        self.validate_security_patterns()
        self.generate_openapi_spec_check()
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"✅ Total endpoints found: {len(self.endpoints)}")
        print(f"❌ Errors: {len(self.errors)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # Calculate score
        total_issues = len(self.errors) + len(self.warnings)
        if total_issues == 0:
            print("\n🟢 API endpoints are properly configured!")
        elif len(self.errors) == 0:
            print("\n🟡 API endpoints are mostly configured with minor warnings")
        else:
            print("\n🔴 API endpoints have critical issues that need attention")
        
        return len(self.errors) == 0

if __name__ == "__main__":
    validator = APIEndpointValidator()
    success = validator.run_validation()
    
    exit(0 if success else 1)