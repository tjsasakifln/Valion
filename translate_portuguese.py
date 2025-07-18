#!/usr/bin/env python3
"""
Script to translate Portuguese text to English in MLOps files.
"""

import os
import re
from pathlib import Path

# Portuguese to English translation dictionary
TRANSLATIONS = {
    # Docstrings
    "\"\"\"Versão do modelo.\"\"\"": "\"\"\"Model version.\"\"\"",
    "\"\"\"Converte para dicionário.\"\"\"": "\"\"\"Convert to dictionary.\"\"\"",
    "\"\"\"Cria instância a partir de dicionário.\"\"\"": "\"\"\"Create instance from dictionary.\"\"\"",
    "\"\"\"Registry central de modelos.\"\"\"": "\"\"\"Central model registry.\"\"\"",
    "\"\"\"Inicializa banco de dados SQLite.\"\"\"": "\"\"\"Initialize SQLite database.\"\"\"",
    "\"\"\"Context manager para conexão com banco.\"\"\"": "\"\"\"Context manager for database connection.\"\"\"",
    "\"\"\"Registra um novo modelo.\"\"\"": "\"\"\"Register a new model.\"\"\"",
    "\"\"\"Cria nova versão do modelo.\"\"\"": "\"\"\"Create new model version.\"\"\"",
    "\"\"\"Obtém versão específica do modelo.\"\"\"": "\"\"\"Get specific model version.\"\"\"",
    "\"\"\"Lista modelos com filtros opcionais.\"\"\"": "\"\"\"List models with optional filters.\"\"\"",
    "\"\"\"Lista todas as versões de um modelo.\"\"\"": "\"\"\"List all versions of a model.\"\"\"",
    "\"\"\"Promove versão para estágio superior.\"\"\"": "\"\"\"Promote version to higher stage.\"\"\"",
    "\"\"\"Carrega modelo do registry.\"\"\"": "\"\"\"Load model from registry.\"\"\"",
    "\"\"\"Remove versão específica.\"\"\"": "\"\"\"Remove specific version.\"\"\"",
    "\"\"\"Obtém próximo número de versão.\"\"\"": "\"\"\"Get next version number.\"\"\"",
    "\"\"\"Salva artefatos do modelo.\"\"\"": "\"\"\"Save model artifacts.\"\"\"",
    "\"\"\"Calcula checksum dos artefatos.\"\"\"": "\"\"\"Calculate artifacts checksum.\"\"\"",
    "\"\"\"Salva versão no banco de dados.\"\"\"": "\"\"\"Save version to database.\"\"\"",
    "\"\"\"Atualiza última versão do modelo.\"\"\"": "\"\"\"Update model latest version.\"\"\"",
    "\"\"\"Obtém estatísticas do registry.\"\"\"": "\"\"\"Get registry statistics.\"\"\"",
    "\"\"\"Cria instância do model registry.\"\"\"": "\"\"\"Create model registry instance.\"\"\"",
    
    # Comments
    "# Diretórios": "# Directories",
    "# Criar diretórios": "# Create directories",
    "# Thread lock para operações concorrentes": "# Thread lock for concurrent operations",
    "# Inicializar banco de dados": "# Initialize database",
    "# Gerar ID único": "# Generate unique ID",
    "# Verificar se modelo já existe": "# Check if model already exists",
    "# Inserir modelo": "# Insert model",
    "# Obter próxima versão": "# Get next version",
    "# Criar metadados": "# Create metadata",
    "# Detectar automaticamente": "# Detect automatically",
    "# Salvar artefatos": "# Save artifacts",
    "# Calcular checksum": "# Calculate checksum",
    "# Calcular tamanho": "# Calculate size",
    "# Criar versão": "# Create version",
    "# Salvar no banco": "# Save to database",
    "# Atualizar modelo principal": "# Update main model",
    "# Obter última versão": "# Get latest version",
    "# Obter versão específica": "# Get specific version",
    "# Adicionar informações da última versão": "# Add latest version information",
    "# Atualizar estágio": "# Update stage",
    "# Se promovendo para produção, despromover outras versões": "# If promoting to production, demote other versions",
    "# Carregar modelo principal": "# Load main model",
    "# Obter informações da versão": "# Get version information",
    "# Remover artefatos": "# Remove artifacts",
    "# Remover do banco": "# Remove from database",
    "# Incrementar versão (simplificado)": "# Increment version (simplified)",
    "# Incrementar patch": "# Increment patch",
    "# Salvar modelo principal": "# Save main model",
    "# Salvar scaler se existir": "# Save scaler if exists",
    "# Salvar explainer se existir": "# Save explainer if exists",
    "# Contadores básicos": "# Basic counters",
    "# Estatísticas por estágio": "# Statistics by stage",
    "# Tamanho total": "# Total size",
    
    # Class docstrings
    "\"\"\"Estágios do modelo.\"\"\"": "\"\"\"Model deployment stages.\"\"\"",
    "\"\"\"Status do modelo.\"\"\"": "\"\"\"Model status states.\"\"\"",
    "\"\"\"Metadados do modelo.\"\"\"": "\"\"\"Model metadata container.\"\"\"",
    
    # Module docstrings
    "\"\"\"Model Registry para MLOps": "\"\"\"Model Registry for MLOps",
    "Sistema de versionamento e gestão de modelos de ML.": "Model versioning and management system for ML lifecycle.",
    "\"\"\"Model Deployer para MLOps": "\"\"\"Model Deployer for MLOps",
    "Sistema de deployment e gerenciamento de modelos em produção.": "System for deployment and management of models in production.",
    "\"\"\"Model Validator para MLOps": "\"\"\"Model Validator for MLOps",
    "Sistema de validação e teste de modelos antes do deployment.": "System for validation and testing of models before deployment.",
    "\"\"\"Pipeline Orchestrator para MLOps": "\"\"\"Pipeline Orchestrator for MLOps",
    "Sistema de orquestração de pipelines de ML (treinamento, validação, deployment).": "System for orchestrating ML pipelines (training, validation, deployment).",
    "\"\"\"Version Manager para MLOps": "\"\"\"Version Manager for MLOps",
    "Sistema de gerenciamento de versões semânticas para modelos de ML.": "System for semantic version management of ML models.",
    "\"\"\"MLOps Pipeline para Valion": "\"\"\"MLOps Pipeline for Valion",
    "Sistema completo de versionamento, deployment e monitoramento de modelos.": "Complete system for versioning, deployment and monitoring of models.",
}

def translate_file(file_path: Path):
    """Translate Portuguese text to English in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply translations
    for portuguese, english in TRANSLATIONS.items():
        content = content.replace(portuguese, english)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Translated: {file_path}")

def main():
    """Main function to translate all MLOps files."""
    mlops_dir = Path("src/mlops")
    
    if not mlops_dir.exists():
        print(f"MLOps directory not found: {mlops_dir}")
        return
    
    # Find all Python files
    python_files = list(mlops_dir.glob("*.py"))
    
    for file_path in python_files:
        translate_file(file_path)
    
    print(f"Translated {len(python_files)} files")

if __name__ == "__main__":
    main()