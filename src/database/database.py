# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Database manager for Valion
"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import IntegrityError, OperationalError
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Callable
import logging
import os
from datetime import datetime
import json
import hashlib
import time
from functools import wraps

from .models import Base, User, Project, Evaluation, EvaluationResult, AuditTrail, ModelArtifact


class DatabaseManager:
    """Centralized database manager with automatic auditing."""
    
    def __init__(self, database_url: str):
        """
        Initializes database manager with integrity protections.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)
        
        # Configure engine
        if database_url.startswith('sqlite'):
            # Special configuration for SQLite (development)
            self.engine = create_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30,
                    "isolation_level": "SERIALIZABLE"  # Maior consistência
                },
                echo=False,
                pool_timeout=30,
                pool_recycle=3600
            )
        else:
            # Configuração para PostgreSQL (produção)
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_timeout=30,
                pool_recycle=3600,
                echo=False,
                isolation_level="READ_COMMITTED",
                connect_args={
                    "options": "-c timezone=utc",
                    "application_name": "valion_backend"
                }
            )
        
        # Criar session factory com configurações de segurança
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine,
            expire_on_commit=False  # Manter objetos acessíveis após commit
        )
        
        # Configurar triggers de segurança para auditoria
        self._setup_audit_protection()
        
        # Criar tabelas se não existirem
        self._create_tables()
        
        self.logger.info("Database manager initialized successfully with audit protection")
    
    def _create_tables(self):
        """Cria todas as tabelas no banco de dados."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created/verified successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    def _setup_audit_protection(self):
        """
        Configura proteções para a trilha de auditoria.
        Previne modificações e exclusões na tabela de auditoria.
        """
        @event.listens_for(self.engine, "connect")
        def setup_audit_triggers(dbapi_connection, connection_record):
            try:
                if self.database_url.startswith('postgresql'):
                    # Triggers PostgreSQL para proteger auditoria
                    cursor = dbapi_connection.cursor()
                    
                    # Trigger para prevenir updates
                    cursor.execute("""
                        CREATE OR REPLACE FUNCTION prevent_audit_modification()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            RAISE EXCEPTION 'Modification of audit trail is not allowed. Operation: %', TG_OP;
                            RETURN NULL;
                        END;
                        $$ LANGUAGE plpgsql;
                    """)
                    
                    # Trigger para prevenir deletes
                    cursor.execute("""
                        DROP TRIGGER IF EXISTS prevent_audit_update ON audit_trails;
                        CREATE TRIGGER prevent_audit_update
                            BEFORE UPDATE ON audit_trails
                            FOR EACH ROW
                            EXECUTE FUNCTION prevent_audit_modification();
                    """)
                    
                    cursor.execute("""
                        DROP TRIGGER IF EXISTS prevent_audit_delete ON audit_trails;
                        CREATE TRIGGER prevent_audit_delete
                            BEFORE DELETE ON audit_trails
                            FOR EACH ROW
                            EXECUTE FUNCTION prevent_audit_modification();
                    """)
                    
                    # Trigger para garantir checksums nos artefatos
                    cursor.execute("""
                        CREATE OR REPLACE FUNCTION validate_artifact_checksum()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            IF NEW.checksum IS NULL OR LENGTH(NEW.checksum) != 64 THEN
                                RAISE EXCEPTION 'Model artifact must have valid SHA-256 checksum';
                            END IF;
                            RETURN NEW;
                        END;
                        $$ LANGUAGE plpgsql;
                    """)
                    
                    cursor.execute("""
                        DROP TRIGGER IF EXISTS validate_checksum ON model_artifacts;
                        CREATE TRIGGER validate_checksum
                            BEFORE INSERT OR UPDATE ON model_artifacts
                            FOR EACH ROW
                            EXECUTE FUNCTION validate_artifact_checksum();
                    """)
                    
                    cursor.close()
                    self.logger.info("PostgreSQL audit protection triggers created")
                    
                elif self.database_url.startswith('sqlite'):
                    # Para SQLite, usar triggers mais simples
                    cursor = dbapi_connection.cursor()
                    
                    cursor.execute("""
                        CREATE TRIGGER IF NOT EXISTS prevent_audit_update
                        BEFORE UPDATE ON audit_trails
                        BEGIN
                            SELECT RAISE(ABORT, 'Modification of audit trail is not allowed');
                        END;
                    """)
                    
                    cursor.execute("""
                        CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
                        BEFORE DELETE ON audit_trails
                        BEGIN
                            SELECT RAISE(ABORT, 'Deletion of audit trail is not allowed');
                        END;
                    """)
                    
                    cursor.close()
                    self.logger.info("SQLite audit protection triggers created")
                    
            except Exception as e:
                self.logger.warning(f"Could not create audit protection triggers: {e}")
    
    @contextmanager
    def get_session(self):
        """
        Context manager para sessões de banco de dados com retry e logs de auditoria.
        
        Yields:
            Session: Sessão do SQLAlchemy
        """
        session = self.SessionLocal()
        start_time = time.time()
        transaction_id = hashlib.md5(f"{start_time}{id(session)}".encode()).hexdigest()[:8]
        
        try:
            self.logger.debug(f"Starting transaction {transaction_id}")
            yield session
            
            # Commit atômico
            session.commit()
            duration = time.time() - start_time
            self.logger.debug(f"Transaction {transaction_id} committed successfully in {duration:.3f}s")
            
        except IntegrityError as e:
            session.rollback()
            duration = time.time() - start_time
            self.logger.error(f"Integrity error in transaction {transaction_id} after {duration:.3f}s: {e}")
            raise
            
        except OperationalError as e:
            session.rollback()
            duration = time.time() - start_time
            self.logger.error(f"Operational error in transaction {transaction_id} after {duration:.3f}s: {e}")
            raise
            
        except Exception as e:
            session.rollback()
            duration = time.time() - start_time
            self.logger.error(f"Unexpected error in transaction {transaction_id} after {duration:.3f}s: {e}")
            raise
            
        finally:
            session.close()
    
    @contextmanager
    def get_atomic_transaction(self, isolation_level: str = "READ_COMMITTED"):
        """
        Context manager para transações atômicas com nível de isolamento configurável.
        
        Args:
            isolation_level: Nível de isolamento da transação
            
        Yields:
            Session: Sessão com transação atômica
        """
        session = self.SessionLocal()
        transaction_id = hashlib.md5(f"{time.time()}{id(session)}".encode()).hexdigest()[:8]
        
        try:
            # Configurar nível de isolamento se suportado
            if not self.database_url.startswith('sqlite'):
                session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}"))
            
            self.logger.debug(f"Starting atomic transaction {transaction_id} with isolation {isolation_level}")
            
            with session.begin():
                yield session
                # Commit automático pelo context manager do begin()
            
            self.logger.debug(f"Atomic transaction {transaction_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Atomic transaction {transaction_id} failed: {e}")
            # Rollback automático pelo context manager
            raise
        finally:
            session.close()
    
    def create_audit_trail(self, session: Session, user_id: str, operation: str, 
                          entity_type: str, entity_id: Optional[str] = None,
                          old_values: Optional[Dict] = None, new_values: Optional[Dict] = None,
                          metadata: Optional[Dict] = None, evaluation_id: Optional[str] = None,
                          ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> AuditTrail:
        """
        Cria entrada de auditoria imutável para operação.
        
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
            
        Returns:
            AuditTrail: Entrada de auditoria criada
        """
        # Sanitizar e validar dados antes de inserir
        sanitized_metadata = self._sanitize_audit_data(metadata)
        sanitized_old_values = self._sanitize_audit_data(old_values)
        sanitized_new_values = self._sanitize_audit_data(new_values)
        
        # Criar hash da operação para detecção de duplicatas
        operation_hash = self._calculate_operation_hash(
            user_id, operation, entity_type, entity_id, 
            sanitized_old_values, sanitized_new_values
        )
        
        # Adicionar hash aos metadados
        if sanitized_metadata is None:
            sanitized_metadata = {}
        sanitized_metadata['operation_hash'] = operation_hash
        sanitized_metadata['creation_timestamp'] = datetime.utcnow().isoformat()
        
        try:
            audit = AuditTrail(
                user_id=user_id,
                evaluation_id=evaluation_id,
                operation=operation,
                entity_type=entity_type,
                entity_id=entity_id,
                old_values=sanitized_old_values,
                new_values=sanitized_new_values,
                metadata=sanitized_metadata,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            session.add(audit)
            session.flush()  # Para obter o ID e validar constraints
            
            self.logger.info(
                f"Immutable audit trail created: {operation} on {entity_type} by user {user_id} "
                f"(hash: {operation_hash[:8]})"
            )
            return audit
            
        except IntegrityError as e:
            self.logger.error(f"Failed to create audit trail: {e}")
            raise ValueError(f"Audit trail creation failed: {e}")
    
    def _sanitize_audit_data(self, data: Optional[Dict]) -> Optional[Dict]:
        """
        Sanitiza dados de auditoria removendo informações sensíveis.
        
        Args:
            data: Dados para sanitizar
            
        Returns:
            Dados sanitizados
        """
        if data is None:
            return None
        
        # Lista de chaves sensíveis que devem ser removidas ou mascaradas
        sensitive_keys = {
            'password', 'senha', 'token', 'secret', 'key', 'auth',
            'credential', 'private', 'confidential'
        }
        
        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            
            # Verificar se a chave contém informação sensível
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[SANITIZED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_audit_data(value)
            elif isinstance(value, str) and len(value) > 1000:
                # Truncar strings muito longas
                sanitized[key] = value[:997] + "..."
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _calculate_operation_hash(self, user_id: str, operation: str, entity_type: str,
                                 entity_id: Optional[str], old_values: Optional[Dict],
                                 new_values: Optional[Dict]) -> str:
        """
        Calcula hash único da operação para detecção de duplicatas.
        
        Args:
            user_id: ID do usuário
            operation: Operação
            entity_type: Tipo de entidade
            entity_id: ID da entidade
            old_values: Valores antigos
            new_values: Valores novos
            
        Returns:
            Hash SHA-256 da operação
        """
        data_to_hash = {
            'user_id': user_id,
            'operation': operation,
            'entity_type': entity_type,
            'entity_id': entity_id,
            'old_values': old_values,
            'new_values': new_values,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        json_string = json.dumps(data_to_hash, sort_keys=True, default=str)
        return hashlib.sha256(json_string.encode()).hexdigest()
    
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
        """
        Salva artefato do modelo com verificação de integridade.
        
        Args:
            session: Sessão do banco
            evaluation_id: ID da avaliação
            artifact_type: Tipo do artefato
            artifact_name: Nome do artefato
            file_path: Caminho do arquivo
            file_size: Tamanho do arquivo
            checksum: Hash SHA-256 do arquivo
            metadata: Metadados adicionais
            user_id: ID do usuário
            
        Returns:
            ModelArtifact: Artefato salvo
        """
        # Validar checksum
        if not checksum or len(checksum) != 64:
            raise ValueError("Checksum SHA-256 válido é obrigatório para artefatos")
        
        # Verificar se já existe artefato com mesmo checksum
        existing_artifact = session.query(ModelArtifact).filter(
            ModelArtifact.checksum == checksum,
            ModelArtifact.evaluation_id == evaluation_id
        ).first()
        
        if existing_artifact:
            self.logger.warning(f"Artifact with checksum {checksum[:8]}... already exists for evaluation {evaluation_id}")
            return existing_artifact
        
        # Verificar integridade do arquivo se existir
        if os.path.exists(file_path):
            actual_size = os.path.getsize(file_path)
            if actual_size != file_size:
                raise ValueError(f"File size mismatch: expected {file_size}, got {actual_size}")
            
            # Calcular checksum do arquivo para verificação
            actual_checksum = self._calculate_file_checksum(file_path)
            if actual_checksum != checksum:
                raise ValueError(f"Checksum mismatch: expected {checksum}, got {actual_checksum}")
        
        # Adicionar metadados de integridade
        if metadata is None:
            metadata = {}
        metadata.update({
            'integrity_verified_at': datetime.utcnow().isoformat(),
            'file_verified': os.path.exists(file_path),
            'creation_method': 'database_manager'
        })
        
        try:
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
            
            # Auditoria com informações de integridade
            if user_id:
                self.create_audit_trail(
                    session, user_id, 'save_model_artifact', 'model_artifact', str(artifact.id),
                    new_values={
                        'artifact_type': artifact_type,
                        'artifact_name': artifact_name,
                        'file_path': file_path,
                        'file_size': file_size,
                        'checksum_verified': True
                    },
                    evaluation_id=evaluation_id
                )
            
            self.logger.info(f"Model artifact saved with verified integrity: {artifact_name} (checksum: {checksum[:8]}...)")
            return artifact
            
        except Exception as e:
            self.logger.error(f"Failed to save model artifact: {e}")
            raise
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """
        Calcula checksum SHA-256 de um arquivo.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Checksum SHA-256 do arquivo
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file checksum: {e}")
            raise ValueError(f"Cannot calculate checksum for {file_path}: {e}")
    
    def get_model_artifacts(self, session: Session, evaluation_id: str) -> List[ModelArtifact]:
        """Lista artefatos do modelo."""
        return session.query(ModelArtifact).filter(
            ModelArtifact.evaluation_id == evaluation_id
        ).all()
    
    # Métodos para consulta de auditoria
    def get_audit_trail(self, session: Session, evaluation_id: Optional[str] = None,
                       user_id: Optional[str] = None, operation: Optional[str] = None,
                       limit: int = 100, verify_integrity: bool = True) -> List[AuditTrail]:
        """
        Consulta trilha de auditoria com verificação opcional de integridade.
        
        Args:
            session: Sessão do banco
            evaluation_id: Filtrar por ID da avaliação
            user_id: Filtrar por ID do usuário
            operation: Filtrar por tipo de operação
            limit: Limite de registros
            verify_integrity: Se deve verificar integridade dos registros
            
        Returns:
            Lista de registros de auditoria
        """
        query = session.query(AuditTrail)
        
        if evaluation_id:
            query = query.filter(AuditTrail.evaluation_id == evaluation_id)
        if user_id:
            query = query.filter(AuditTrail.user_id == user_id)
        if operation:
            query = query.filter(AuditTrail.operation == operation)
        
        audit_records = query.order_by(AuditTrail.timestamp.desc()).limit(limit).all()
        
        if verify_integrity:
            self._verify_audit_integrity(audit_records)
        
        return audit_records
    
    def _verify_audit_integrity(self, audit_records: List[AuditTrail]) -> bool:
        """
        Verifica integridade dos registros de auditoria.
        
        Args:
            audit_records: Lista de registros para verificar
            
        Returns:
            True se todos os registros estão íntegros
        """
        for record in audit_records:
            try:
                # Verificar se o hash da operação confere
                if record.metadata and 'operation_hash' in record.metadata:
                    expected_hash = self._calculate_operation_hash(
                        record.user_id,
                        record.operation,
                        record.entity_type,
                        record.entity_id,
                        record.old_values,
                        record.new_values
                    )
                    
                    stored_hash = record.metadata['operation_hash']
                    if stored_hash != expected_hash:
                        self.logger.error(
                            f"Audit integrity violation detected in record {record.id}: "
                            f"hash mismatch (stored: {stored_hash[:8]}..., calculated: {expected_hash[:8]}...)"
                        )
                        return False
                        
            except Exception as e:
                self.logger.warning(f"Error verifying audit record {record.id}: {e}")
        
        return True
    
    def get_audit_summary(self, session: Session, evaluation_id: str) -> Dict[str, Any]:
        """
        Gera resumo de auditoria com verificações de integridade.
        
        Args:
            session: Sessão do banco
            evaluation_id: ID da avaliação
            
        Returns:
            Resumo da auditoria
        """
        audit_records = self.get_audit_trail(
            session, evaluation_id=evaluation_id, limit=1000, verify_integrity=True
        )
        
        operations = {}
        users = set()
        total_operations = 0
        integrity_verified = True
        
        for record in audit_records:
            operations[record.operation] = operations.get(record.operation, 0) + 1
            users.add(record.user_id)
            total_operations += 1
            
            # Verificar integridade individual
            if record.metadata and 'operation_hash' not in record.metadata:
                integrity_verified = False
        
        timeline_start = min(record.timestamp for record in audit_records) if audit_records else None
        timeline_end = max(record.timestamp for record in audit_records) if audit_records else None
        
        return {
            'evaluation_id': evaluation_id,
            'total_operations': total_operations,
            'unique_users': len(users),
            'operations_by_type': operations,
            'timeline_start': timeline_start.isoformat() if timeline_start else None,
            'timeline_end': timeline_end.isoformat() if timeline_end else None,
            'integrity_verified': integrity_verified,
            'verification_timestamp': datetime.utcnow().isoformat()
        }
    
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
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("A variável de ambiente DATABASE_URL não está configurada.")
        _database_manager = DatabaseManager(database_url)
    
    return _database_manager


def initialize_database(database_url: str = None):
    """Inicializa o banco de dados."""
    global _database_manager
    
    if database_url is None:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("A variável de ambiente DATABASE_URL não está configurada.")
    
    _database_manager = DatabaseManager(database_url)
    return _database_manager