"""
Gerenciador de banco de dados para Valion
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime
import json

from .models import Base, User, Project, Evaluation, EvaluationResult, AuditTrail, ModelArtifact


class DatabaseManager:
    """Gerenciador centralizado do banco de dados com auditoria automática."""
    
    def __init__(self, database_url: str):
        """
        Inicializa o gerenciador de banco de dados.
        
        Args:
            database_url: URL de conexão com o banco
        """
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)
        
        # Configurar engine
        if database_url.startswith('sqlite'):
            # Configuração especial para SQLite (desenvolvimento)
            self.engine = create_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False
            )
        else:
            # Configuração para PostgreSQL (produção)
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
        
        # Criar session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Criar tabelas se não existirem
        self._create_tables()
        
        self.logger.info("Database manager initialized successfully")
    
    def _create_tables(self):
        """Cria todas as tabelas no banco de dados."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created/verified successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Context manager para sessões de banco de dados.
        
        Yields:
            Session: Sessão do SQLAlchemy
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def create_audit_trail(self, session: Session, user_id: str, operation: str, 
                          entity_type: str, entity_id: Optional[str] = None,
                          old_values: Optional[Dict] = None, new_values: Optional[Dict] = None,
                          metadata: Optional[Dict] = None, evaluation_id: Optional[str] = None,
                          ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """
        Cria entrada de auditoria para operação.
        
        Args:
            session: Sessão do banco
            user_id: ID do usuário
            operation: Tipo de operação
            entity_type: Tipo de entidade
            entity_id: ID da entidade
            old_values: Valores anteriores
            new_values: Valores novos
            metadata: Metadados adicionais
            evaluation_id: ID da avaliação (se aplicável)
            ip_address: IP do cliente
            user_agent: User agent do cliente
        """
        audit = AuditTrail(
            user_id=user_id,
            evaluation_id=evaluation_id,
            operation=operation,
            entity_type=entity_type,
            entity_id=entity_id,
            old_values=old_values,
            new_values=new_values,
            metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        session.add(audit)
        session.flush()  # Para obter o ID
        
        self.logger.info(f"Audit trail created: {operation} on {entity_type} by user {user_id}")
        return audit
    
    # Métodos para User
    def create_user(self, session: Session, username: str, email: str, full_name: str) -> User:
        """Cria novo usuário."""
        user = User(username=username, email=email, full_name=full_name)
        session.add(user)
        session.flush()
        
        # Auditoria
        self.create_audit_trail(
            session, str(user.id), 'create_user', 'user', str(user.id),
            new_values={'username': username, 'email': email, 'full_name': full_name}
        )
        
        return user
    
    def get_user_by_id(self, session: Session, user_id: str) -> Optional[User]:
        """Busca usuário por ID."""
        return session.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, session: Session, username: str) -> Optional[User]:
        """Busca usuário por username."""
        return session.query(User).filter(User.username == username).first()
    
    # Métodos para Project
    def create_project(self, session: Session, name: str, description: str, 
                      owner_id: str, user_id: str) -> Project:
        """Cria novo projeto."""
        project = Project(name=name, description=description, owner_id=owner_id)
        session.add(project)
        session.flush()
        
        # Auditoria
        self.create_audit_trail(
            session, user_id, 'create_project', 'project', str(project.id),
            new_values={'name': name, 'description': description, 'owner_id': owner_id}
        )
        
        return project
    
    def get_projects_by_user(self, session: Session, user_id: str) -> List[Project]:
        """Lista projetos do usuário."""
        return session.query(Project).filter(
            Project.owner_id == user_id,
            Project.is_active == True
        ).order_by(Project.created_at.desc()).all()
    
    # Métodos para Evaluation
    def create_evaluation(self, session: Session, project_id: str, user_id: str,
                         name: str, description: str, data_file_path: str,
                         target_column: str, config: Optional[Dict] = None) -> Evaluation:
        """Cria nova avaliação."""
        evaluation = Evaluation(
            project_id=project_id,
            user_id=user_id,
            name=name,
            description=description,
            data_file_path=data_file_path,
            target_column=target_column,
            config=config or {}
        )
        session.add(evaluation)
        session.flush()
        
        # Auditoria
        self.create_audit_trail(
            session, user_id, 'create_evaluation', 'evaluation', str(evaluation.id),
            new_values={
                'project_id': project_id,
                'name': name,
                'data_file_path': data_file_path,
                'target_column': target_column
            },
            evaluation_id=str(evaluation.id)
        )
        
        return evaluation
    
    def update_evaluation_status(self, session: Session, evaluation_id: str, 
                                status: str, current_phase: Optional[str] = None,
                                progress: Optional[float] = None, user_id: str = None):
        """Atualiza status da avaliação."""
        evaluation = session.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_id} not found")
        
        old_values = {
            'status': evaluation.status,
            'current_phase': evaluation.current_phase,
            'progress': evaluation.progress
        }
        
        evaluation.status = status
        if current_phase:
            evaluation.current_phase = current_phase
        if progress is not None:
            evaluation.progress = progress
        
        if status == 'processing' and not evaluation.started_at:
            evaluation.started_at = datetime.utcnow()
        elif status == 'completed':
            evaluation.completed_at = datetime.utcnow()
        
        # Auditoria
        if user_id:
            self.create_audit_trail(
                session, user_id, 'update_evaluation_status', 'evaluation', evaluation_id,
                old_values=old_values,
                new_values={
                    'status': status,
                    'current_phase': current_phase,
                    'progress': progress
                },
                evaluation_id=evaluation_id
            )
    
    def get_evaluation_by_id(self, session: Session, evaluation_id: str) -> Optional[Evaluation]:
        """Busca avaliação por ID."""
        return session.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    
    def get_evaluations_by_project(self, session: Session, project_id: str) -> List[Evaluation]:
        """Lista avaliações do projeto."""
        return session.query(Evaluation).filter(
            Evaluation.project_id == project_id
        ).order_by(Evaluation.created_at.desc()).all()
    
    # Métodos para EvaluationResult
    def save_evaluation_result(self, session: Session, evaluation_id: str, 
                              report_json: Dict, model_type: str,
                              r2_score: float, rmse: float, mae: float, mape: float,
                              nbr_grade: str, compliance_score: float,
                              total_records: int, features_count: int,
                              training_time_seconds: float, user_id: str) -> EvaluationResult:
        """Salva resultado da avaliação."""
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            report_json=report_json,
            model_type=model_type,
            r2_score=r2_score,
            rmse=rmse,
            mae=mae,
            mape=mape,
            nbr_grade=nbr_grade,
            compliance_score=compliance_score,
            total_records=total_records,
            features_count=features_count,
            training_time_seconds=training_time_seconds
        )
        session.add(result)
        session.flush()
        
        # Auditoria
        self.create_audit_trail(
            session, user_id, 'save_evaluation_result', 'evaluation_result', str(result.id),
            new_values={
                'model_type': model_type,
                'r2_score': r2_score,
                'nbr_grade': nbr_grade,
                'compliance_score': compliance_score
            },
            evaluation_id=evaluation_id
        )
        
        return result
    
    def get_evaluation_result(self, session: Session, evaluation_id: str) -> Optional[EvaluationResult]:
        """Busca resultado da avaliação."""
        return session.query(EvaluationResult).filter(
            EvaluationResult.evaluation_id == evaluation_id
        ).first()
    
    # Métodos para ModelArtifact
    def save_model_artifact(self, session: Session, evaluation_id: str,
                           artifact_type: str, artifact_name: str, file_path: str,
                           file_size: int, checksum: str, metadata: Optional[Dict] = None,
                           user_id: str = None) -> ModelArtifact:
        """Salva artefato do modelo."""
        artifact = ModelArtifact(
            evaluation_id=evaluation_id,
            artifact_type=artifact_type,
            artifact_name=artifact_name,
            file_path=file_path,
            file_size=file_size,
            checksum=checksum,
            metadata=metadata
        )
        session.add(artifact)
        session.flush()
        
        # Auditoria
        if user_id:
            self.create_audit_trail(
                session, user_id, 'save_model_artifact', 'model_artifact', str(artifact.id),
                new_values={
                    'artifact_type': artifact_type,
                    'artifact_name': artifact_name,
                    'file_path': file_path
                },
                evaluation_id=evaluation_id
            )
        
        return artifact
    
    def get_model_artifacts(self, session: Session, evaluation_id: str) -> List[ModelArtifact]:
        """Lista artefatos do modelo."""
        return session.query(ModelArtifact).filter(
            ModelArtifact.evaluation_id == evaluation_id
        ).all()
    
    # Métodos para consulta de auditoria
    def get_audit_trail(self, session: Session, evaluation_id: Optional[str] = None,
                       user_id: Optional[str] = None, operation: Optional[str] = None,
                       limit: int = 100) -> List[AuditTrail]:
        """Consulta trilha de auditoria."""
        query = session.query(AuditTrail)
        
        if evaluation_id:
            query = query.filter(AuditTrail.evaluation_id == evaluation_id)
        if user_id:
            query = query.filter(AuditTrail.user_id == user_id)
        if operation:
            query = query.filter(AuditTrail.operation == operation)
        
        return query.order_by(AuditTrail.timestamp.desc()).limit(limit).all()
    
    def get_evaluation_timeline(self, session: Session, evaluation_id: str) -> List[AuditTrail]:
        """Obtém timeline completa de uma avaliação."""
        return session.query(AuditTrail).filter(
            AuditTrail.evaluation_id == evaluation_id
        ).order_by(AuditTrail.timestamp.asc()).all()


# Singleton para gerenciador de banco
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Obtém instância singleton do gerenciador de banco."""
    global _database_manager
    
    if _database_manager is None:
        database_url = os.getenv('DATABASE_URL', 'sqlite:///valion.db')
        _database_manager = DatabaseManager(database_url)
    
    return _database_manager


def initialize_database(database_url: str = None):
    """Inicializa o banco de dados."""
    global _database_manager
    
    if database_url is None:
        database_url = os.getenv('DATABASE_URL', 'sqlite:///valion.db')
    
    _database_manager = DatabaseManager(database_url)
    return _database_manager