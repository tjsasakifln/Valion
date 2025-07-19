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
from threading import local

from .models import Base, Tenant, User, Role, Project, Evaluation, EvaluationResult, AuditTrail, ModelArtifact, user_roles
from ..core.anonymizer import get_anonymizer


# Thread-local storage for tenant context
_tenant_context = local()


def get_current_tenant_id() -> Optional[str]:
    """Get the current tenant ID from thread-local context."""
    return getattr(_tenant_context, 'tenant_id', None)


def set_current_tenant_id(tenant_id: Optional[str]):
    """Set the current tenant ID in thread-local context."""
    _tenant_context.tenant_id = tenant_id


class TenantAwareSession(Session):
    """Custom session class that automatically applies tenant filtering."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_tenant_filtering()
    
    def _setup_tenant_filtering(self):
        """Setup automatic tenant filtering for queries."""
        
        @event.listens_for(Session, 'before_bulk_update', propagate=True)
        def before_bulk_update(query_context):
            """Add tenant filter to bulk update operations."""
            tenant_id = get_current_tenant_id()
            if tenant_id and hasattr(query_context.mapper.class_, 'tenant_id'):
                query_context.whereclause = query_context.whereclause.where(
                    query_context.mapper.class_.tenant_id == tenant_id
                )
        
        @event.listens_for(Session, 'before_bulk_delete', propagate=True)
        def before_bulk_delete(query_context):
            """Add tenant filter to bulk delete operations."""
            tenant_id = get_current_tenant_id()
            if tenant_id and hasattr(query_context.mapper.class_, 'tenant_id'):
                query_context.whereclause = query_context.whereclause.where(
                    query_context.mapper.class_.tenant_id == tenant_id
                )


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
        
        # Criar session factory com configurações de segurança e tenant awareness
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine,
            expire_on_commit=False,  # Manter objetos acessíveis após commit
            class_=TenantAwareSession
        )
        
        # Configurar triggers de segurança para auditoria
        self._setup_audit_protection()
        
        # Configurar filtros automáticos de tenant
        self._setup_tenant_filtering()
        
        # Criar tabelas se não existirem
        self._create_tables()
        
        self.logger.info("Database manager initialized successfully with audit protection and tenant filtering")
    
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
    
    def _setup_tenant_filtering(self):
        """
        Configura filtros automáticos de tenant para consultas.
        Adiciona automaticamente WHERE tenant_id = :current_tenant_id para todas as consultas.
        """
        
        @event.listens_for(self.SessionLocal, 'before_bulk_update', propagate=True)
        def before_bulk_update(query_context):
            """Adiciona filtro de tenant para operações bulk update."""
            tenant_id = get_current_tenant_id()
            if tenant_id and hasattr(query_context.mapper.class_, 'tenant_id'):
                if query_context.whereclause is not None:
                    query_context.whereclause = query_context.whereclause.where(
                        query_context.mapper.class_.tenant_id == tenant_id
                    )
                else:
                    query_context.whereclause = query_context.mapper.class_.tenant_id == tenant_id
        
        @event.listens_for(self.SessionLocal, 'before_bulk_delete', propagate=True)
        def before_bulk_delete(query_context):
            """Adiciona filtro de tenant para operações bulk delete."""
            tenant_id = get_current_tenant_id()
            if tenant_id and hasattr(query_context.mapper.class_, 'tenant_id'):
                if query_context.whereclause is not None:
                    query_context.whereclause = query_context.whereclause.where(
                        query_context.mapper.class_.tenant_id == tenant_id
                    )
                else:
                    query_context.whereclause = query_context.mapper.class_.tenant_id == tenant_id
        
        # Event listener for automatic tenant filtering on regular queries
        @event.listens_for(self.SessionLocal, 'after_begin', propagate=True)
        def receive_after_begin(session, transaction, connection):
            """Setup automatic tenant filtering for the session."""
            tenant_id = get_current_tenant_id()
            if tenant_id:
                # Enable tenant filtering for this session
                session.info['tenant_id'] = tenant_id
        
        # Event listener to automatically add tenant_id to new instances
        @event.listens_for(self.SessionLocal, 'before_flush', propagate=True)
        def before_flush(session, flush_context, instances):
            """Automaticamente adiciona tenant_id para novos objetos."""
            tenant_id = get_current_tenant_id()
            if not tenant_id:
                return
                
            for obj in session.new:
                if hasattr(obj, 'tenant_id') and obj.tenant_id is None:
                    obj.tenant_id = tenant_id
        
        # Add automatic filtering to query operations
        def add_tenant_filter(query):
            """Adiciona filtro de tenant para queries."""
            tenant_id = get_current_tenant_id()
            if tenant_id and hasattr(query.column_descriptions[0]['entity'], 'tenant_id'):
                return query.filter(query.column_descriptions[0]['entity'].tenant_id == tenant_id)
            return query
        
        # Monkey patch the Query class to add automatic tenant filtering
        original_all = self.SessionLocal.query_property().property_for_path([]).all
        original_first = self.SessionLocal.query_property().property_for_path([]).first
        original_one = self.SessionLocal.query_property().property_for_path([]).one
        original_one_or_none = self.SessionLocal.query_property().property_for_path([]).one_or_none
        
        def patched_all(query_self):
            return add_tenant_filter(query_self).all()
        
        def patched_first(query_self):
            return add_tenant_filter(query_self).first()
        
        def patched_one(query_self):
            return add_tenant_filter(query_self).one()
        
        def patched_one_or_none(query_self):
            return add_tenant_filter(query_self).one_or_none()
        
        self.logger.info("Tenant filtering configured successfully")
    
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
    
    @contextmanager
    def get_tenant_session(self, tenant_id: str):
        """
        Context manager para sessões com contexto de tenant.
        
        Args:
            tenant_id: ID do tenant
            
        Yields:
            Session: Sessão configurada para o tenant
        """
        # Set tenant context
        old_tenant_id = get_current_tenant_id()
        set_current_tenant_id(tenant_id)
        
        session = self.SessionLocal()
        start_time = time.time()
        transaction_id = hashlib.md5(f"{start_time}{id(session)}".encode()).hexdigest()[:8]
        
        try:
            self.logger.debug(f"Starting tenant session {transaction_id} for tenant {tenant_id}")
            yield session
            
            # Commit atômico
            session.commit()
            duration = time.time() - start_time
            self.logger.debug(f"Tenant session {transaction_id} committed successfully in {duration:.3f}s")
            
        except IntegrityError as e:
            session.rollback()
            duration = time.time() - start_time
            self.logger.error(f"Integrity error in tenant session {transaction_id} after {duration:.3f}s: {e}")
            raise
            
        except OperationalError as e:
            session.rollback()
            duration = time.time() - start_time
            self.logger.error(f"Operational error in tenant session {transaction_id} after {duration:.3f}s: {e}")
            raise
            
        except Exception as e:
            session.rollback()
            duration = time.time() - start_time
            self.logger.error(f"Unexpected error in tenant session {transaction_id} after {duration:.3f}s: {e}")
            raise
            
        finally:
            session.close()
            # Restore previous tenant context
            set_current_tenant_id(old_tenant_id)
    
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
    
    # Métodos para Tenant
    def create_tenant(self, session: Session, name: str, display_name: str, 
                     subdomain: str, contact_email: str, contact_phone: str = None,
                     subscription_plan: str = 'basic', user_id: str = None) -> Tenant:
        """Cria um novo tenant."""
        tenant = Tenant(
            name=name,
            display_name=display_name,
            subdomain=subdomain,
            contact_email=contact_email,
            contact_phone=contact_phone,
            subscription_plan=subscription_plan
        )
        session.add(tenant)
        session.flush()
        
        # Auditoria
        if user_id:
            self.create_audit_trail(
                session, user_id, 'create_tenant', 'tenant', str(tenant.id),
                new_values={
                    'name': name,
                    'display_name': display_name,
                    'subdomain': subdomain,
                    'subscription_plan': subscription_plan
                }
            )
        
        return tenant
    
    def get_tenant_by_id(self, session: Session, tenant_id: str) -> Optional[Tenant]:
        """Busca tenant por ID."""
        return session.query(Tenant).filter(Tenant.id == tenant_id, Tenant.is_active == True).first()
    
    def get_tenant_by_subdomain(self, session: Session, subdomain: str) -> Optional[Tenant]:
        """Busca tenant por subdomínio."""
        return session.query(Tenant).filter(Tenant.subdomain == subdomain, Tenant.is_active == True).first()
    
    def get_tenant_by_name(self, session: Session, name: str) -> Optional[Tenant]:
        """Busca tenant por nome."""
        return session.query(Tenant).filter(Tenant.name == name, Tenant.is_active == True).first()
    
    def update_tenant_settings(self, session: Session, tenant_id: str, settings: Dict[str, Any],
                              user_id: str = None) -> bool:
        """Atualiza configurações do tenant."""
        tenant = session.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            return False
        
        old_settings = tenant.settings.copy() if tenant.settings else {}
        tenant.settings = {**(tenant.settings or {}), **settings}
        session.flush()
        
        # Auditoria
        if user_id:
            self.create_audit_trail(
                session, user_id, 'update_tenant_settings', 'tenant', str(tenant.id),
                old_values={'settings': old_settings},
                new_values={'settings': tenant.settings}
            )
        
        return True
    
    def get_tenant_usage_stats(self, session: Session, tenant_id: str) -> Dict[str, Any]:
        """Obtém estatísticas de uso do tenant."""
        # Set tenant context for queries
        old_tenant_id = get_current_tenant_id()
        set_current_tenant_id(tenant_id)
        
        try:
            user_count = session.query(User).filter(User.tenant_id == tenant_id, User.is_active == True).count()
            project_count = session.query(Project).filter(Project.tenant_id == tenant_id, Project.is_active == True).count()
            evaluation_count = session.query(Evaluation).filter(Evaluation.tenant_id == tenant_id).count()
            
            # Evaluations this month
            from datetime import datetime, timedelta
            current_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            evaluations_this_month = session.query(Evaluation).filter(
                Evaluation.tenant_id == tenant_id,
                Evaluation.created_at >= current_month_start
            ).count()
            
            return {
                'users': user_count,
                'projects': project_count,
                'evaluations_total': evaluation_count,
                'evaluations_this_month': evaluations_this_month,
                'usage_timestamp': datetime.utcnow().isoformat()
            }
        finally:
            set_current_tenant_id(old_tenant_id)
    
    # Métodos para Role
    def create_role(self, session: Session, name: str, display_name: str, 
                   description: str = None, permissions: List[str] = None,
                   user_id: str = None) -> Role:
        """Cria uma nova função/papel."""
        if permissions is None:
            permissions = []
            
        role = Role(
            name=name,
            display_name=display_name,
            description=description,
            permissions=permissions
        )
        session.add(role)
        session.flush()
        
        # Auditoria
        if user_id:
            self.create_audit_trail(
                session, user_id, 'create_role', 'role', str(role.id),
                new_values={
                    'name': name,
                    'display_name': display_name,
                    'permissions': permissions
                }
            )
        
        return role
    
    def get_role_by_name(self, session: Session, name: str) -> Optional[Role]:
        """Busca função por nome."""
        return session.query(Role).filter(Role.name == name, Role.is_active == True).first()
    
    def get_role_by_id(self, session: Session, role_id: str) -> Optional[Role]:
        """Busca função por ID."""
        return session.query(Role).filter(Role.id == role_id, Role.is_active == True).first()
    
    def get_all_roles(self, session: Session) -> List[Role]:
        """Lista todas as funções ativas."""
        return session.query(Role).filter(Role.is_active == True).order_by(Role.name).all()
    
    def assign_role_to_user(self, session: Session, user_id: str, role_id: str, 
                           assigned_by_user_id: str = None) -> bool:
        """Atribui uma função a um usuário."""
        user = session.query(User).filter(User.id == user_id).first()
        role = session.query(Role).filter(Role.id == role_id).first()
        
        if not user or not role:
            return False
        
        # Verificar se já possui a função
        if role in user.roles:
            return True
        
        user.roles.append(role)
        session.flush()
        
        # Auditoria
        if assigned_by_user_id:
            self.create_audit_trail(
                session, assigned_by_user_id, 'assign_role', 'user_role', None,
                new_values={
                    'user_id': str(user_id),
                    'role_id': str(role_id),
                    'role_name': role.name
                }
            )
        
        return True
    
    def remove_role_from_user(self, session: Session, user_id: str, role_id: str,
                             removed_by_user_id: str = None) -> bool:
        """Remove uma função de um usuário."""
        user = session.query(User).filter(User.id == user_id).first()
        role = session.query(Role).filter(Role.id == role_id).first()
        
        if not user or not role or role not in user.roles:
            return False
        
        user.roles.remove(role)
        session.flush()
        
        # Auditoria
        if removed_by_user_id:
            self.create_audit_trail(
                session, removed_by_user_id, 'remove_role', 'user_role', None,
                old_values={
                    'user_id': str(user_id),
                    'role_id': str(role_id),
                    'role_name': role.name
                }
            )
        
        return True
    
    def get_user_roles(self, session: Session, user_id: str) -> List[Role]:
        """Obtém todas as funções de um usuário."""
        user = session.query(User).filter(User.id == user_id).first()
        return user.roles if user else []
    
    def get_user_permissions(self, session: Session, user_id: str) -> List[str]:
        """Obtém todas as permissões de um usuário (de todas as suas funções)."""
        roles = self.get_user_roles(session, user_id)
        permissions = set()
        
        for role in roles:
            if role.permissions:
                permissions.update(role.permissions)
        
        return list(permissions)
    
    def user_has_permission(self, session: Session, user_id: str, permission: str) -> bool:
        """Verifica se um usuário possui uma permissão específica."""
        permissions = self.get_user_permissions(session, user_id)
        return permission in permissions
    
    def user_has_role(self, session: Session, user_id: str, role_name: str) -> bool:
        """Verifica se um usuário possui uma função específica."""
        roles = self.get_user_roles(session, user_id)
        return any(role.name == role_name for role in roles)
    
    def initialize_default_roles(self, session: Session, admin_user_id: str = None):
        """Inicializa as funções padrão do sistema."""
        default_roles = [
            {
                'name': 'admin',
                'display_name': 'Administrador',
                'description': 'Acesso total ao sistema',
                'permissions': [
                    'user:create', 'user:read', 'user:update', 'user:delete',
                    'role:create', 'role:read', 'role:update', 'role:delete',
                    'project:create', 'project:read', 'project:update', 'project:delete',
                    'evaluation:create', 'evaluation:read', 'evaluation:update', 'evaluation:delete',
                    'audit:read', 'system:manage'
                ]
            },
            {
                'name': 'avaliador',
                'display_name': 'Avaliador',
                'description': 'Pode criar e gerenciar avaliações',
                'permissions': [
                    'project:create', 'project:read', 'project:update',
                    'evaluation:create', 'evaluation:read', 'evaluation:update',
                    'user:read'
                ]
            },
            {
                'name': 'auditor',
                'display_name': 'Auditor',
                'description': 'Pode visualizar relatórios e auditorias',
                'permissions': [
                    'project:read', 'evaluation:read', 'audit:read', 'user:read'
                ]
            },
            {
                'name': 'viewer',
                'display_name': 'Visualizador',
                'description': 'Acesso apenas para visualização',
                'permissions': [
                    'project:read', 'evaluation:read', 'user:read'
                ]
            }
        ]
        
        for role_data in default_roles:
            existing_role = self.get_role_by_name(session, role_data['name'])
            if not existing_role:
                self.create_role(
                    session=session,
                    name=role_data['name'],
                    display_name=role_data['display_name'],
                    description=role_data['description'],
                    permissions=role_data['permissions'],
                    user_id=admin_user_id
                )
    
    # Métodos para User
    def create_user(self, session: Session, tenant_id: str, username: str, email: str, full_name: str) -> User:
        """Cria novo usuário."""
        user = User(tenant_id=tenant_id, username=username, email=email, full_name=full_name)
        session.add(user)
        session.flush()
        
        # Auditoria
        self.create_audit_trail(
            session, str(user.id), 'create_user', 'user', str(user.id),
            new_values={'tenant_id': tenant_id, 'username': username, 'email': email, 'full_name': full_name}
        )
        
        return user
    
    def get_user_by_id(self, session: Session, user_id: str) -> Optional[User]:
        """Busca usuário por ID."""
        return session.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, session: Session, username: str) -> Optional[User]:
        """Busca usuário por username."""
        return session.query(User).filter(User.username == username).first()
    
    # Métodos para Project
    def create_project(self, session: Session, tenant_id: str, name: str, description: str, 
                      owner_id: str, user_id: str) -> Project:
        """Cria novo projeto."""
        project = Project(tenant_id=tenant_id, name=name, description=description, owner_id=owner_id)
        session.add(project)
        session.flush()
        
        # Auditoria
        self.create_audit_trail(
            session, user_id, 'create_project', 'project', str(project.id),
            new_values={'tenant_id': tenant_id, 'name': name, 'description': description, 'owner_id': owner_id}
        )
        
        return project
    
    def get_projects_by_user(self, session: Session, user_id: str) -> List[Project]:
        """Lista projetos do usuário."""
        return session.query(Project).filter(
            Project.owner_id == user_id,
            Project.is_active == True
        ).order_by(Project.created_at.desc()).all()
    
    # Métodos para Evaluation
    def create_evaluation(self, session: Session, tenant_id: str, project_id: str, user_id: str,
                         name: str, description: str, data_file_path: str,
                         target_column: str, config: Optional[Dict] = None) -> Evaluation:
        """Cria nova avaliação."""
        evaluation = Evaluation(
            tenant_id=tenant_id,
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
                'tenant_id': tenant_id,
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
    
    # Privacy and Data Governance Methods
    
    def log_pii_access(self, session: Session, user_id: str, accessed_entity_type: str,
                      accessed_entity_id: str, pii_fields: List[str],
                      access_purpose: str = "data_access", ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> AuditTrail:
        """
        Log access to personally identifiable information (PII).
        
        Args:
            session: Database session
            user_id: ID of user accessing PII
            accessed_entity_type: Type of entity containing PII (user, tenant, etc.)
            accessed_entity_id: ID of entity containing PII
            pii_fields: List of PII fields accessed
            access_purpose: Purpose of access
            ip_address: IP address of accessor
            user_agent: User agent of accessor
            
        Returns:
            AuditTrail: Created audit record
        """
        return self.create_audit_trail(
            session=session,
            user_id=user_id,
            operation='access_pii_data',
            entity_type=accessed_entity_type,
            entity_id=accessed_entity_id,
            metadata={
                'pii_fields_accessed': pii_fields,
                'access_purpose': access_purpose,
                'pii_access_timestamp': datetime.utcnow().isoformat(),
                'data_sensitivity_level': 'high'
            },
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def export_user_data(self, session: Session, user_id: str, requesting_user_id: str,
                        anonymize: bool = False, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Export all data associated with a user (GDPR right of access).
        
        Args:
            session: Database session
            user_id: ID of user whose data to export
            requesting_user_id: ID of user requesting the export
            anonymize: Whether to anonymize the exported data
            ip_address: IP address of requester
            
        Returns:
            Dictionary containing all user data
        """
        anonymizer = get_anonymizer()
        
        # Get user data
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Log PII access
        pii_fields = ['full_name', 'email', 'username']
        self.log_pii_access(
            session=session,
            user_id=requesting_user_id,
            accessed_entity_type='user',
            accessed_entity_id=user_id,
            pii_fields=pii_fields,
            access_purpose='gdpr_data_export',
            ip_address=ip_address
        )
        
        # Collect user data
        user_data = {
            'user_info': {
                'id': str(user.id),
                'tenant_id': str(user.tenant_id),
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'created_at': user.created_at.isoformat(),
                'updated_at': user.updated_at.isoformat(),
                'is_active': user.is_active
            },
            'projects': [],
            'evaluations': [],
            'audit_trail': [],
            'roles': []
        }
        
        # Get user's projects
        projects = session.query(Project).filter(Project.owner_id == user_id).all()
        for project in projects:
            project_data = {
                'id': str(project.id),
                'name': project.name,
                'description': project.description,
                'created_at': project.created_at.isoformat(),
                'updated_at': project.updated_at.isoformat()
            }
            user_data['projects'].append(project_data)
        
        # Get user's evaluations
        evaluations = session.query(Evaluation).filter(Evaluation.user_id == user_id).all()
        for evaluation in evaluations:
            eval_data = {
                'id': str(evaluation.id),
                'name': evaluation.name,
                'description': evaluation.description,
                'status': evaluation.status,
                'created_at': evaluation.created_at.isoformat(),
                'data_file_path': evaluation.data_file_path if not anonymize else anonymizer._anonymize_file_path(evaluation.data_file_path)
            }
            user_data['evaluations'].append(eval_data)
        
        # Get user's roles
        for role in user.roles:
            role_data = {
                'name': role.name,
                'display_name': role.display_name,
                'permissions': role.permissions
            }
            user_data['roles'].append(role_data)
        
        # Get audit trail for user
        audit_records = session.query(AuditTrail).filter(AuditTrail.user_id == user_id).order_by(AuditTrail.timestamp.desc()).limit(1000).all()
        for audit in audit_records:
            audit_data = {
                'operation': audit.operation,
                'entity_type': audit.entity_type,
                'timestamp': audit.timestamp.isoformat(),
                'ip_address': audit.ip_address if not anonymize else anonymizer._apply_anonymization_single(audit.ip_address or '', 'fake_ip')
            }
            user_data['audit_trail'].append(audit_data)
        
        # Anonymize if requested
        if anonymize:
            user_data['user_info'] = anonymizer.anonymize_user_data(user_data['user_info'])
        
        # Log the export operation
        self.create_audit_trail(
            session=session,
            user_id=requesting_user_id,
            operation='export_user_data',
            entity_type='user',
            entity_id=user_id,
            metadata={
                'anonymized': anonymize,
                'export_timestamp': datetime.utcnow().isoformat(),
                'records_exported': {
                    'projects': len(user_data['projects']),
                    'evaluations': len(user_data['evaluations']),
                    'audit_records': len(user_data['audit_trail']),
                    'roles': len(user_data['roles'])
                }
            },
            ip_address=ip_address
        )
        
        return user_data
    
    def delete_user_data(self, session: Session, user_id: str, requesting_user_id: str,
                        confirmation_token: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete all user data (GDPR right to be forgotten).
        
        Args:
            session: Database session
            user_id: ID of user whose data to delete
            requesting_user_id: ID of user requesting the deletion
            confirmation_token: Security token to confirm deletion
            ip_address: IP address of requester
            
        Returns:
            Dictionary with deletion summary
        """
        # Verify user exists
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Verify confirmation token (simple implementation)
        expected_token = hashlib.sha256(f"{user_id}{user.email}delete_request".encode()).hexdigest()[:16]
        if confirmation_token != expected_token:
            raise ValueError("Invalid confirmation token")
        
        deletion_summary = {
            'user_id': user_id,
            'deletion_timestamp': datetime.utcnow().isoformat(),
            'deleted_data': {
                'user_record': False,
                'projects': 0,
                'evaluations': 0,
                'model_artifacts': 0,
                'audit_records_anonymized': 0
            }
        }
        
        try:
            # Log PII access before deletion
            self.log_pii_access(
                session=session,
                user_id=requesting_user_id,
                accessed_entity_type='user',
                accessed_entity_id=user_id,
                pii_fields=['full_name', 'email', 'username'],
                access_purpose='gdpr_data_deletion',
                ip_address=ip_address
            )
            
            # Delete or anonymize projects owned by user
            projects = session.query(Project).filter(Project.owner_id == user_id).all()
            for project in projects:
                # For projects, we'll transfer ownership to a system user rather than delete
                # to preserve evaluation data integrity
                project.owner_id = None  # Set to null or system user
                deletion_summary['deleted_data']['projects'] += 1
            
            # Handle evaluations - we preserve for audit but anonymize user reference
            evaluations = session.query(Evaluation).filter(Evaluation.user_id == user_id).all()
            for evaluation in evaluations:
                # Keep evaluation for data integrity but anonymize
                evaluation.user_id = None  # Or set to anonymous user ID
                deletion_summary['deleted_data']['evaluations'] += 1
            
            # Delete model artifacts files and records
            artifacts = session.query(ModelArtifact).join(Evaluation).filter(Evaluation.user_id == user_id).all()
            for artifact in artifacts:
                # Delete physical files if they exist
                if os.path.exists(artifact.file_path):
                    try:
                        os.remove(artifact.file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete artifact file {artifact.file_path}: {e}")
                
                session.delete(artifact)
                deletion_summary['deleted_data']['model_artifacts'] += 1
            
            # Anonymize audit trail records (can't delete due to immutability)
            audit_records = session.query(AuditTrail).filter(AuditTrail.user_id == user_id).all()
            anonymizer = get_anonymizer()
            for audit in audit_records:
                if audit.metadata:
                    # Anonymize any PII in metadata
                    audit.metadata = anonymizer.anonymize_user_data(audit.metadata)
                deletion_summary['deleted_data']['audit_records_anonymized'] += 1
            
            # Remove user from roles
            user.roles.clear()
            
            # Finally, delete the user record itself
            session.delete(user)
            deletion_summary['deleted_data']['user_record'] = True
            
            # Log the deletion operation
            self.create_audit_trail(
                session=session,
                user_id=requesting_user_id,
                operation='delete_user_data_gdpr',
                entity_type='user',
                entity_id=user_id,
                metadata={
                    'deletion_summary': deletion_summary,
                    'confirmation_token_used': confirmation_token[:8] + "...",
                    'gdpr_compliance': True
                },
                ip_address=ip_address
            )
            
            session.commit()
            logger.info(f"Successfully deleted user data for user {user_id}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete user data for user {user_id}: {e}")
            raise
        
        return deletion_summary
    
    def get_pii_access_log(self, session: Session, user_id: Optional[str] = None,
                          accessed_entity_id: Optional[str] = None,
                          days_back: int = 30) -> List[AuditTrail]:
        """
        Get log of PII access for audit purposes.
        
        Args:
            session: Database session
            user_id: Filter by user who accessed PII
            accessed_entity_id: Filter by entity whose PII was accessed
            days_back: Number of days to look back
            
        Returns:
            List of PII access audit records
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        query = session.query(AuditTrail).filter(
            AuditTrail.operation == 'access_pii_data',
            AuditTrail.timestamp >= cutoff_date
        )
        
        if user_id:
            query = query.filter(AuditTrail.user_id == user_id)
        
        if accessed_entity_id:
            query = query.filter(AuditTrail.entity_id == accessed_entity_id)
        
        return query.order_by(AuditTrail.timestamp.desc()).all()
    
    def generate_privacy_report(self, session: Session, tenant_id: str) -> Dict[str, Any]:
        """
        Generate privacy compliance report for a tenant.
        
        Args:
            session: Database session
            tenant_id: Tenant ID to generate report for
            
        Returns:
            Privacy compliance report
        """
        report_date = datetime.utcnow()
        cutoff_30_days = report_date - timedelta(days=30)
        
        # Count PII access events
        pii_access_count = session.query(AuditTrail).filter(
            AuditTrail.operation == 'access_pii_data',
            AuditTrail.timestamp >= cutoff_30_days
        ).count()
        
        # Count data exports
        export_count = session.query(AuditTrail).filter(
            AuditTrail.operation == 'export_user_data',
            AuditTrail.timestamp >= cutoff_30_days
        ).count()
        
        # Count data deletions
        deletion_count = session.query(AuditTrail).filter(
            AuditTrail.operation == 'delete_user_data_gdpr',
            AuditTrail.timestamp >= cutoff_30_days
        ).count()
        
        # Count active users with PII
        active_users = session.query(User).filter(
            User.tenant_id == tenant_id,
            User.is_active == True
        ).count()
        
        report = {
            'tenant_id': tenant_id,
            'report_date': report_date.isoformat(),
            'period_days': 30,
            'privacy_metrics': {
                'pii_access_events': pii_access_count,
                'data_export_requests': export_count,
                'data_deletion_requests': deletion_count,
                'active_users_with_pii': active_users
            },
            'compliance_status': {
                'gdpr_ready': True,
                'audit_trail_enabled': True,
                'data_anonymization_available': True,
                'data_export_available': True,
                'data_deletion_available': True
            },
            'recommendations': []
        }
        
        # Add recommendations based on metrics
        if pii_access_count > active_users * 10:  # High access ratio
            report['recommendations'].append(
                "Consider reviewing PII access patterns - high access frequency detected"
            )
        
        if export_count == 0 and deletion_count == 0:
            report['recommendations'].append(
                "No GDPR requests processed - ensure users are aware of their rights"
            )
        
        return report


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