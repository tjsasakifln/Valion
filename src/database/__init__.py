"""
Pacote de banco de dados para Valion
"""

from .models import (
    Base, User, Project, Evaluation, EvaluationResult, 
    ModelArtifact, AuditTrail, DataValidation, ModelPerformanceHistory,
    create_tables, drop_tables
)
from .database import DatabaseManager, get_database_manager

__all__ = [
    'Base', 'User', 'Project', 'Evaluation', 'EvaluationResult',
    'ModelArtifact', 'AuditTrail', 'DataValidation', 'ModelPerformanceHistory',
    'create_tables', 'drop_tables', 'DatabaseManager', 'get_database_manager'
]