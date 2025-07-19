# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Modelos de banco de dados para Valion - Sistema de persistência e auditoria
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, ForeignKey, Float, Index, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

# Association table for many-to-many relationship between User and Role
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id'), primary_key=True),
    Column('assigned_at', DateTime, default=datetime.utcnow, nullable=False),
    Column('assigned_by', UUID(as_uuid=True), ForeignKey('users.id')),
    Index('idx_user_roles_user', 'user_id'),
    Index('idx_user_roles_role', 'role_id'),
)


class Tenant(Base):
    """Modelo de tenant para isolamento multi-inquilino (multi-tenancy)."""
    __tablename__ = 'tenants'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), unique=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    subdomain = Column(String(100), unique=True, nullable=False)
    
    # Contact and billing information
    contact_email = Column(String(100), nullable=False)
    contact_phone = Column(String(50))
    billing_address = Column(Text)
    
    # Subscription and limits
    subscription_plan = Column(String(50), default='basic', nullable=False)  # basic, premium, enterprise
    max_users = Column(Integer, default=10, nullable=False)
    max_projects = Column(Integer, default=50, nullable=False)
    max_evaluations_per_month = Column(Integer, default=100, nullable=False)
    
    # Configuration
    settings = Column(JSON, default=dict)  # Tenant-specific settings
    
    # Status and timestamps
    is_active = Column(Boolean, default=True, nullable=False)
    is_trial = Column(Boolean, default=True, nullable=False)
    trial_ends_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relacionamentos
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="tenant", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="tenant", cascade="all, delete-orphan")
    
    # Índices
    __table_args__ = (
        Index('idx_tenant_subdomain', 'subdomain'),
        Index('idx_tenant_active', 'is_active'),
        Index('idx_tenant_name', 'name'),
    )


class Role(Base):
    """Modelo de função/papel para controle de acesso baseado em função (RBAC)."""
    __tablename__ = 'roles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    display_name = Column(String(100), nullable=False)
    description = Column(Text)
    permissions = Column(JSON, nullable=False, default=list)  # Lista de permissões
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relacionamentos
    users = relationship("User", secondary=user_roles, back_populates="roles")
    
    # Índices
    __table_args__ = (
        Index('idx_role_name', 'name'),
        Index('idx_role_active', 'is_active'),
    )


class User(Base):
    """Modelo de usuário para autenticação e auditoria."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id'), nullable=False)
    username = Column(String(50), nullable=False)
    email = Column(String(100), nullable=False)
    full_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relacionamentos
    tenant = relationship("Tenant", back_populates="users")
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    projects = relationship("Project", back_populates="owner")
    evaluations = relationship("Evaluation", back_populates="user")
    audit_trails = relationship("AuditTrail", back_populates="user")
    
    # Índices compostos para garantir unicidade por tenant
    __table_args__ = (
        Index('idx_user_tenant', 'tenant_id'),
        Index('idx_user_username_tenant', 'username', 'tenant_id', unique=True),
        Index('idx_user_email_tenant', 'email', 'tenant_id', unique=True),
    )


class Project(Base):
    """Modelo de projeto para organizar avaliações."""
    __tablename__ = 'projects'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id'), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relacionamentos
    tenant = relationship("Tenant", back_populates="projects")
    owner = relationship("User", back_populates="projects")
    evaluations = relationship("Evaluation", back_populates="project")
    
    # Índices
    __table_args__ = (
        Index('idx_project_tenant', 'tenant_id'),
        Index('idx_project_owner', 'owner_id'),
        Index('idx_project_created', 'created_at'),
        Index('idx_project_name_tenant', 'name', 'tenant_id'),
    )


class Evaluation(Base):
    """Modelo principal de avaliação imobiliária."""
    __tablename__ = 'evaluations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id'), nullable=False)
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
    tenant = relationship("Tenant", back_populates="evaluations")
    project = relationship("Project", back_populates="evaluations")
    user = relationship("User", back_populates="evaluations")
    results = relationship("EvaluationResult", back_populates="evaluation", uselist=False)
    model_artifacts = relationship("ModelArtifact", back_populates="evaluation")
    audit_trails = relationship("AuditTrail", back_populates="evaluation")
    
    # Índices
    __table_args__ = (
        Index('idx_evaluation_tenant', 'tenant_id'),
        Index('idx_evaluation_status', 'status'),
        Index('idx_evaluation_user', 'user_id'),
        Index('idx_evaluation_project', 'project_id'),
        Index('idx_evaluation_created', 'created_at'),
        Index('idx_evaluation_tenant_status', 'tenant_id', 'status'),
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