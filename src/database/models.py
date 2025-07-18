# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Modelos de banco de dados para Valion - Sistema de persistência e auditoria
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, ForeignKey, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    """Modelo de usuário para autenticação e auditoria."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relacionamentos
    projects = relationship("Project", back_populates="owner")
    evaluations = relationship("Evaluation", back_populates="user")
    audit_trails = relationship("AuditTrail", back_populates="user")


class Project(Base):
    """Modelo de projeto para organizar avaliações."""
    __tablename__ = 'projects'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relacionamentos
    owner = relationship("User", back_populates="projects")
    evaluations = relationship("Evaluation", back_populates="project")
    
    # Índices
    __table_args__ = (
        Index('idx_project_owner', 'owner_id'),
        Index('idx_project_created', 'created_at'),
    )


class Evaluation(Base):
    """Modelo principal de avaliação imobiliária."""
    __tablename__ = 'evaluations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Metadados da avaliação
    name = Column(String(200), nullable=False)
    description = Column(Text)
    data_file_path = Column(String(500), nullable=False)
    target_column = Column(String(100), nullable=False)
    
    # Status e timing
    status = Column(String(50), nullable=False, default='pending')  # pending, processing, completed, failed
    current_phase = Column(String(100))
    progress = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Configurações
    config = Column(JSON)  # Configurações específicas da avaliação
    
    # Relacionamentos
    project = relationship("Project", back_populates="evaluations")
    user = relationship("User", back_populates="evaluations")
    results = relationship("EvaluationResult", back_populates="evaluation", uselist=False)
    model_artifacts = relationship("ModelArtifact", back_populates="evaluation")
    audit_trails = relationship("AuditTrail", back_populates="evaluation")
    
    # Índices
    __table_args__ = (
        Index('idx_evaluation_status', 'status'),
        Index('idx_evaluation_user', 'user_id'),
        Index('idx_evaluation_project', 'project_id'),
        Index('idx_evaluation_created', 'created_at'),
    )


class EvaluationResult(Base):
    """Resultado final da avaliação."""
    __tablename__ = 'evaluation_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_id = Column(UUID(as_uuid=True), ForeignKey('evaluations.id'), nullable=False, unique=True)
    
    # Resultados principais
    report_json = Column(JSON, nullable=False)  # Relatório completo em JSON
    model_type = Column(String(50), nullable=False)
    
    # Métricas principais
    r2_score = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    mape = Column(Float)
    nbr_grade = Column(String(20))  # Superior, Normal, Inferior, Inadequado
    compliance_score = Column(Float)
    valuation_standard = Column(String(50), default='NBR 14653')  # Novo campo
    
    # Metadados
    total_records = Column(Integer)
    features_count = Column(Integer)
    training_time_seconds = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relacionamentos
    evaluation = relationship("Evaluation", back_populates="results")
    
    # Índices
    __table_args__ = (
        Index('idx_result_evaluation', 'evaluation_id'),
        Index('idx_result_r2', 'r2_score'),
        Index('idx_result_grade', 'nbr_grade'),
    )


class ModelArtifact(Base):
    """Artefatos do modelo treinado."""
    __tablename__ = 'model_artifacts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_id = Column(UUID(as_uuid=True), ForeignKey('evaluations.id'), nullable=False)
    
    # Informações do artefato
    artifact_type = Column(String(50), nullable=False)  # model, scaler, explainer, report
    artifact_name = Column(String(200), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    checksum = Column(String(128))  # SHA-256 hash
    
    # Metadados
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relacionamentos
    evaluation = relationship("Evaluation", back_populates="model_artifacts")
    
    # Índices
    __table_args__ = (
        Index('idx_artifact_evaluation', 'evaluation_id'),
        Index('idx_artifact_type', 'artifact_type'),
    )


class AuditTrail(Base):
    """Trilha de auditoria imutável para todas as operações."""
    __tablename__ = 'audit_trails'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Contexto da operação
    evaluation_id = Column(UUID(as_uuid=True), ForeignKey('evaluations.id'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Detalhes da operação
    operation = Column(String(100), nullable=False)  # create_evaluation, start_processing, etc.
    entity_type = Column(String(50), nullable=False)  # evaluation, model, result
    entity_id = Column(String(100))
    
    # Dados da operação
    old_values = Column(JSON)  # Estado anterior (para updates)
    new_values = Column(JSON)  # Estado novo
    metadata = Column(JSON)    # Informações adicionais
    
    # Timestamp imutável
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # IP e User Agent para rastreabilidade
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Relacionamentos
    evaluation = relationship("Evaluation", back_populates="audit_trails")
    user = relationship("User", back_populates="audit_trails")
    
    # Índices para consultas eficientes
    __table_args__ = (
        Index('idx_audit_evaluation', 'evaluation_id'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_operation', 'operation'),
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
    )


class DataValidation(Base):
    """Histórico de validações de dados."""
    __tablename__ = 'data_validations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_id = Column(UUID(as_uuid=True), ForeignKey('evaluations.id'), nullable=False)
    
    # Resultados da validação
    is_valid = Column(Boolean, nullable=False)
    validation_summary = Column(JSON, nullable=False)
    errors = Column(JSON)
    warnings = Column(JSON)
    
    # Estatísticas dos dados
    total_records = Column(Integer)
    missing_values_count = Column(Integer)
    duplicate_records_count = Column(Integer)
    outliers_count = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relacionamento
    evaluation = relationship("Evaluation")
    
    # Índices
    __table_args__ = (
        Index('idx_validation_evaluation', 'evaluation_id'),
        Index('idx_validation_status', 'is_valid'),
    )


class ModelPerformanceHistory(Base):
    """Histórico de performance dos modelos para análise temporal."""
    __tablename__ = 'model_performance_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_id = Column(UUID(as_uuid=True), ForeignKey('evaluations.id'), nullable=False)
    
    # Métricas de performance
    model_type = Column(String(50), nullable=False)
    r2_score = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    mae = Column(Float, nullable=False)
    mape = Column(Float, nullable=False)
    
    # Contexto temporal
    data_period_start = Column(DateTime)
    data_period_end = Column(DateTime)
    model_trained_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Metadados para análise
    features_used = Column(JSON)
    hyperparameters = Column(JSON)
    
    # Relacionamento
    evaluation = relationship("Evaluation")
    
    # Índices para análise temporal
    __table_args__ = (
        Index('idx_perf_model_type', 'model_type'),
        Index('idx_perf_r2', 'r2_score'),
        Index('idx_perf_trained_at', 'model_trained_at'),
    )


# Função para criar todas as tabelas
def create_tables(engine):
    """Cria todas as tabelas no banco de dados."""
    Base.metadata.create_all(engine)


# Função para dropping (use com cuidado!)
def drop_tables(engine):
    """Remove todas as tabelas do banco de dados."""
    Base.metadata.drop_all(engine)