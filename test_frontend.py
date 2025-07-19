#!/usr/bin/env python3
"""
Frontend validation script for Valion
Tests frontend structure and functionality
"""

import ast
import re
from pathlib import Path

class FrontendValidator:
    def __init__(self):
        self.functions = []
        self.errors = []
        self.warnings = []
        self.privacy_features = []
        
    def parse_frontend_file(self):
        """Parse the frontend file and extract functions"""
        frontend_file = Path('frontend.py')
        
        if not frontend_file.exists():
            self.errors.append("Frontend file not found")
            return
        
        with open(frontend_file, 'r') as f:
            content = f.read()
        
        # Parse AST to get functions
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.functions.append(node.name)
        except SyntaxError as e:
            self.errors.append(f"Syntax error in frontend file: {e}")
            return
        
        # Search for privacy features in content
        privacy_keywords = [
            'Privacy & Data',
            'My Data',
            'GDPR',
            'export-user-data',
            'delete-user-data',
            'privacy/report',
            'anonymize',
            'audit trail'
        ]
        
        for keyword in privacy_keywords:
            if keyword in content:
                self.privacy_features.append(keyword)
    
    def validate_core_functions(self):
        """Validate presence of core frontend functions"""
        print("🔍 Checking core frontend functions...")
        
        required_functions = [
            ('main', 'Main application entry point'),
            ('init_session_state', 'Session state initialization'),
            ('login_user', 'User login functionality'),
            ('logout_user', 'User logout functionality'),
            ('get_auth_headers', 'Authentication headers'),
            ('check_user_permission', 'Permission checking'),
        ]
        
        for func_name, description in required_functions:
            if func_name in self.functions:
                print(f"  ✅ {description}: {func_name}")
            else:
                print(f"  ❌ MISSING: {description}: {func_name}")
                self.errors.append(f"Missing core function: {func_name}")
    
    def validate_privacy_functions(self):
        """Validate privacy and data governance functions"""
        print("\n🔒 Checking privacy and data governance functions...")
        
        privacy_functions = [
            ('show_privacy_governance_page', 'Privacy governance dashboard'),
            ('show_my_data_page', 'Personal data management'),
        ]
        
        for func_name, description in privacy_functions:
            if func_name in self.functions:
                print(f"  ✅ {description}: {func_name}")
            else:
                print(f"  ❌ MISSING: {description}: {func_name}")
                self.errors.append(f"Missing privacy function: {func_name}")
    
    def validate_page_functions(self):
        """Validate page rendering functions"""
        print("\n📄 Checking page functions...")
        
        page_functions = [
            'show_new_evaluation_page',
            'show_evaluation_tracking_page', 
            'show_results_page',
            'show_predictions_page',
            'show_shap_laboratory_page',
            'show_user_management_page',
            'show_about_page',
            'show_login_page'
        ]
        
        found_pages = []
        for func in page_functions:
            if func in self.functions:
                found_pages.append(func)
        
        print(f"  📊 Found {len(found_pages)} page functions:")
        for page in found_pages:
            print(f"    ✅ {page}")
        
        if len(found_pages) < 5:
            self.warnings.append("Few page functions found - might indicate incomplete frontend")
    
    def validate_privacy_features(self):
        """Validate privacy features in frontend"""
        print("\n🛡️ Checking privacy features implementation...")
        
        print(f"  📊 Found {len(self.privacy_features)} privacy features:")
        for feature in self.privacy_features:
            print(f"    ✅ {feature}")
        
        # Check for specific GDPR features
        gdpr_features = [
            'export-user-data',
            'delete-user-data', 
            'GDPR',
            'anonymize'
        ]
        
        found_gdpr = [f for f in gdpr_features if f in self.privacy_features]
        
        if len(found_gdpr) >= 3:
            print("  ✅ Strong GDPR compliance features detected")
        elif len(found_gdpr) >= 1:
            print("  ⚠️ Some GDPR features detected")
            self.warnings.append("Limited GDPR features in frontend")
        else:
            print("  ❌ No GDPR features detected")
            self.errors.append("No GDPR compliance features in frontend")
    
    def validate_streamlit_usage(self):
        """Validate Streamlit usage patterns"""
        print("\n📱 Checking Streamlit usage patterns...")
        
        frontend_file = Path('frontend.py')
        with open(frontend_file, 'r') as f:
            content = f.read()
        
        # Check for Streamlit imports
        if 'import streamlit as st' in content:
            print("  ✅ Streamlit properly imported")
        else:
            print("  ❌ Streamlit import not found")
            self.errors.append("Streamlit import missing")
        
        # Check for page configuration
        if 'st.set_page_config' in content:
            print("  ✅ Page configuration found")
        else:
            print("  ⚠️ No page configuration found")
            self.warnings.append("No Streamlit page configuration")
        
        # Check for session state usage
        if 'st.session_state' in content:
            print("  ✅ Session state usage found")
        else:
            print("  ❌ No session state usage found")
            self.errors.append("No session state management")
        
        # Check for common Streamlit components
        components = ['st.sidebar', 'st.button', 'st.selectbox', 'st.text_input', 'st.dataframe']
        found_components = [comp for comp in components if comp in content]
        
        print(f"  📊 Found {len(found_components)} Streamlit components:")
        for comp in found_components:
            print(f"    ✅ {comp}")
    
    def validate_api_integration(self):
        """Validate API integration patterns"""
        print("\n🌐 Checking API integration...")
        
        frontend_file = Path('frontend.py')
        with open(frontend_file, 'r') as f:
            content = f.read()
        
        # Check for requests usage
        if 'import requests' in content:
            print("  ✅ HTTP requests library imported")
        else:
            print("  ❌ No HTTP requests library found")
            self.errors.append("No HTTP client for API integration")
        
        # Check for API base URL configuration
        if 'API_BASE_URL' in content:
            print("  ✅ API base URL configuration found")
        else:
            print("  ⚠️ No API base URL configuration")
            self.warnings.append("No API base URL configuration")
        
        # Check for authentication headers
        if 'headers=' in content and 'Authorization' in content:
            print("  ✅ Authentication headers usage found")
        else:
            print("  ⚠️ No authentication headers found")
            self.warnings.append("No authentication patterns detected")
        
        # Check for WebSocket usage
        if 'websocket' in content.lower() or 'ws://' in content:
            print("  ✅ WebSocket integration found")
        else:
            print("  ⚠️ No WebSocket integration found")
            self.warnings.append("No real-time features detected")
    
    def check_menu_structure(self):
        """Check menu and navigation structure"""
        print("\n📋 Checking menu structure...")
        
        frontend_file = Path('frontend.py')
        with open(frontend_file, 'r') as f:
            content = f.read()
        
        # Look for menu options
        menu_patterns = [
            'Privacy & Data',
            'My Data',
            'New Evaluation',
            'Results',
            'User Management'
        ]
        
        found_menus = [menu for menu in menu_patterns if menu in content]
        
        print(f"  📊 Found {len(found_menus)} menu options:")
        for menu in found_menus:
            print(f"    ✅ {menu}")
        
        if len(found_menus) >= 4:
            print("  ✅ Comprehensive menu structure")
        elif len(found_menus) >= 2:
            print("  ⚠️ Basic menu structure")
            self.warnings.append("Limited menu options")
        else:
            print("  ❌ Poor menu structure")
            self.errors.append("Insufficient menu structure")
    
    def run_validation(self):
        """Run complete frontend validation"""
        print("🚀 Running Frontend Validation...")
        print("=" * 60)
        
        self.parse_frontend_file()
        
        if self.errors and "not found" in str(self.errors):
            print("❌ Critical error: Frontend file not found")
            return False
        
        print(f"📊 Found {len(self.functions)} functions in frontend")
        
        self.validate_core_functions()
        self.validate_privacy_functions() 
        self.validate_page_functions()
        self.validate_privacy_features()
        self.validate_streamlit_usage()
        self.validate_api_integration()
        self.check_menu_structure()
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"✅ Total functions found: {len(self.functions)}")
        print(f"🔒 Privacy features found: {len(self.privacy_features)}")
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
            print("\n🟢 Frontend is excellently configured!")
        elif len(self.errors) == 0:
            print("\n🟡 Frontend is well configured with minor warnings")
        elif len(self.errors) <= 2:
            print("\n🟠 Frontend has some issues that should be addressed")
        else:
            print("\n🔴 Frontend has critical issues that need immediate attention")
        
        return len(self.errors) <= 2

if __name__ == "__main__":
    validator = FrontendValidator()
    success = validator.run_validation()
    
    exit(0 if success else 1)