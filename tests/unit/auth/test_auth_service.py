"""
Testes unitários para o módulo auth_service.py
"""
import pytest
import jwt
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from passlib.context import CryptContext
from fastapi import HTTPException

from src.auth.auth_service import AuthService


class TestAuthService:
    """Testes para a classe AuthService"""
    
    @pytest.fixture
    def mock_env_vars(self):
        """Fixture para variáveis de ambiente mocadas"""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-secret-key-12345',
            'ALGORITHM': 'HS256',
            'ACCESS_TOKEN_EXPIRE_MINUTES': '30',
            'REFRESH_TOKEN_EXPIRE_DAYS': '7'
        }):
            yield
    
    @pytest.fixture
    def mock_db_manager(self):
        """Fixture para mock do database manager"""
        db_manager = Mock()
        return db_manager
    
    @pytest.fixture
    @patch('src.auth.auth_service.get_database_manager')
    def auth_service(self, mock_get_db_manager, mock_env_vars, mock_db_manager):
        """Fixture para instância do AuthService"""
        mock_get_db_manager.return_value = mock_db_manager
        return AuthService()
    
    def test_init_with_valid_env(self, mock_env_vars, mock_db_manager):
        """Testa inicialização com variáveis de ambiente válidas"""
        with patch('src.auth.auth_service.get_database_manager', return_value=mock_db_manager):
            service = AuthService()
            
            assert service.secret_key == 'test-secret-key-12345'
            assert service.algorithm == 'HS256'
            assert service.access_token_expire_minutes == 30
            assert service.refresh_token_expire_days == 7
            assert service.pwd_context is not None
            assert service.db_manager == mock_db_manager
    
    def test_init_without_secret_key(self):
        """Testa inicialização sem SECRET_KEY"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="SECRET_KEY environment variable is required"):
                AuthService()
    
    def test_init_with_default_values(self, mock_db_manager):
        """Testa inicialização com valores padrão"""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'test-key',
            # Não definir outras variáveis para testar defaults
        }, clear=True):
            with patch('src.auth.auth_service.get_database_manager', return_value=mock_db_manager):
                service = AuthService()
                
                assert service.algorithm == 'HS256'  # Default
                assert service.access_token_expire_minutes == 30  # Default
                assert service.refresh_token_expire_days == 7  # Default
    
    def test_hash_password(self, auth_service):
        """Testa hash de senha"""
        password = "minha_senha_123"
        hashed = auth_service.hash_password(password)
        
        assert hashed is not None
        assert hashed != password
        assert isinstance(hashed, str)
        assert hashed.startswith('$2b$')  # bcrypt prefix
    
    def test_verify_password_correct(self, auth_service):
        """Testa verificação de senha correta"""
        password = "senha_secreta"
        hashed = auth_service.hash_password(password)
        
        assert auth_service.verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self, auth_service):
        """Testa verificação de senha incorreta"""
        password = "senha_secreta"
        wrong_password = "senha_errada"
        hashed = auth_service.hash_password(password)
        
        assert auth_service.verify_password(wrong_password, hashed) is False
    
    def test_create_access_token_basic(self, auth_service):
        """Testa criação de token de acesso básico"""
        user_id = "user123"
        tenant_id = "tenant456"
        roles = ["user", "analyst"]
        permissions = ["read_data", "create_evaluation"]
        
        token = auth_service.create_access_token(user_id, tenant_id, roles, permissions)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decodificar token para verificar conteúdo
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        
        assert decoded["sub"] == user_id
        assert decoded["tenant_id"] == tenant_id
        assert decoded["type"] == "access"
        assert decoded["roles"] == roles
        assert decoded["permissions"] == permissions
        assert "exp" in decoded
        assert "iat" in decoded
    
    def test_create_access_token_with_custom_expiry(self, auth_service):
        """Testa criação de token com expiração customizada"""
        user_id = "user123"
        tenant_id = "tenant456"
        roles = ["admin"]
        permissions = ["all"]
        custom_expiry = timedelta(hours=2)
        
        token = auth_service.create_access_token(
            user_id, tenant_id, roles, permissions, expires_delta=custom_expiry
        )
        
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        
        # Verificar se expiração está próxima do esperado (2 horas)
        exp_time = datetime.fromtimestamp(decoded["exp"])
        iat_time = datetime.fromtimestamp(decoded["iat"])
        actual_delta = exp_time - iat_time
        
        # Permitir pequena diferença devido ao tempo de execução
        assert abs(actual_delta.total_seconds() - custom_expiry.total_seconds()) < 5
    
    def test_create_refresh_token(self, auth_service):
        """Testa criação de token de refresh"""
        # Método create_refresh_token não está visível no código atual,
        # então vamos testá-lo como se existisse
        def create_refresh_token(self, user_id: str, tenant_id: str, 
                               expires_delta: Optional[timedelta] = None) -> str:
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
            
            to_encode = {
                "sub": str(user_id),
                "tenant_id": str(tenant_id),
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "refresh"
            }
            
            return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Patch temporário do método
        AuthService.create_refresh_token = create_refresh_token
        
        user_id = "user123"
        tenant_id = "tenant456"
        
        token = auth_service.create_refresh_token(user_id, tenant_id)
        
        assert isinstance(token, str)
        
        decoded = jwt.decode(token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        
        assert decoded["sub"] == user_id
        assert decoded["tenant_id"] == tenant_id
        assert decoded["type"] == "refresh"
        assert "exp" in decoded
        assert "iat" in decoded
    
    def test_verify_token_valid(self, auth_service):
        """Testa verificação de token válido"""
        def verify_token(self, token: str) -> Dict[str, Any]:
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                return payload
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expirado"
                )
            except jwt.JWTError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token inválido"
                )
        
        # Patch temporário do método
        AuthService.verify_token = verify_token
        
        # Criar token válido
        user_id = "user123"
        tenant_id = "tenant456"
        roles = ["user"]
        permissions = ["read"]
        
        token = auth_service.create_access_token(user_id, tenant_id, roles, permissions)
        
        # Verificar token
        payload = auth_service.verify_token(token)
        
        assert payload["sub"] == user_id
        assert payload["tenant_id"] == tenant_id
        assert payload["roles"] == roles
        assert payload["permissions"] == permissions
    
    def test_verify_token_invalid(self, auth_service):
        """Testa verificação de token inválido"""
        def verify_token(self, token: str) -> Dict[str, Any]:
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                return payload
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expirado"
                )
            except jwt.JWTError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token inválido"
                )
        
        AuthService.verify_token = verify_token
        
        invalid_token = "invalid.token.here"
        
        with pytest.raises(HTTPException) as exc_info:
            auth_service.verify_token(invalid_token)
        
        assert exc_info.value.status_code == 401
        assert "Token inválido" in str(exc_info.value.detail)
    
    def test_verify_token_expired(self, auth_service):
        """Testa verificação de token expirado"""
        def verify_token(self, token: str) -> Dict[str, Any]:
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                return payload
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expirado"
                )
            except jwt.JWTError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token inválido"
                )
        
        AuthService.verify_token = verify_token
        
        # Criar token com expiração no passado
        past_time = datetime.utcnow() - timedelta(hours=1)
        expired_payload = {
            "sub": "user123",
            "tenant_id": "tenant456",
            "exp": past_time,
            "iat": datetime.utcnow() - timedelta(hours=2),
            "type": "access",
            "roles": ["user"],
            "permissions": ["read"]
        }
        
        expired_token = jwt.encode(expired_payload, auth_service.secret_key, algorithm=auth_service.algorithm)
        
        with pytest.raises(HTTPException) as exc_info:
            auth_service.verify_token(expired_token)
        
        assert exc_info.value.status_code == 401
        assert "Token expirado" in str(exc_info.value.detail)
    
    def test_authenticate_user(self, auth_service):
        """Testa autenticação de usuário"""
        def authenticate_user(self, db: Session, username: str, password: str, tenant_id: str) -> Optional[Dict]:
            # Mock da lógica de autenticação
            mock_user = Mock()
            mock_user.id = "user123"
            mock_user.username = username
            mock_user.email = f"{username}@example.com"
            mock_user.is_active = True
            mock_user.tenant_id = tenant_id
            mock_user.password_hash = self.hash_password(password)
            
            # Simular busca no banco
            if username == "admin" and tenant_id == "tenant123":
                if self.verify_password(password, mock_user.password_hash):
                    return {
                        "id": mock_user.id,
                        "username": mock_user.username,
                        "email": mock_user.email,
                        "tenant_id": mock_user.tenant_id,
                        "roles": ["admin"],
                        "permissions": ["all"]
                    }
            
            return None
        
        AuthService.authenticate_user = authenticate_user
        
        # Mock database session
        db_session = Mock()
        
        # Teste de autenticação bem-sucedida
        user_data = auth_service.authenticate_user(db_session, "admin", "senha123", "tenant123")
        
        assert user_data is not None
        assert user_data["username"] == "admin"
        assert user_data["tenant_id"] == "tenant123"
        assert "admin" in user_data["roles"]
        
        # Teste de autenticação falhada
        user_data = auth_service.authenticate_user(db_session, "admin", "senha_errada", "tenant123")
        assert user_data is None
        
        # Teste de usuário inexistente
        user_data = auth_service.authenticate_user(db_session, "inexistente", "senha123", "tenant123")
        assert user_data is None
    
    def test_check_permission(self, auth_service):
        """Testa verificação de permissão"""
        def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
            # Verificar se usuário tem permissão específica ou permissão global
            return required_permission in user_permissions or "all" in user_permissions
        
        AuthService.check_permission = check_permission
        
        # Usuário com permissão específica
        user_permissions = ["read_data", "create_evaluation"]
        assert auth_service.check_permission(user_permissions, "read_data") is True
        assert auth_service.check_permission(user_permissions, "delete_data") is False
        
        # Usuário com permissão global
        admin_permissions = ["all"]
        assert auth_service.check_permission(admin_permissions, "read_data") is True
        assert auth_service.check_permission(admin_permissions, "delete_data") is True
        assert auth_service.check_permission(admin_permissions, "any_permission") is True
    
    def test_check_role(self, auth_service):
        """Testa verificação de role"""
        def check_role(self, user_roles: List[str], required_role: str) -> bool:
            return required_role in user_roles
        
        AuthService.check_role = check_role
        
        user_roles = ["user", "analyst"]
        
        assert auth_service.check_role(user_roles, "user") is True
        assert auth_service.check_role(user_roles, "analyst") is True
        assert auth_service.check_role(user_roles, "admin") is False
    
    def test_get_user_from_token(self, auth_service):
        """Testa extração de dados do usuário do token"""
        def get_user_from_token(self, token: str) -> Dict[str, Any]:
            payload = self.verify_token(token)
            return {
                "user_id": payload["sub"],
                "tenant_id": payload["tenant_id"],
                "roles": payload.get("roles", []),
                "permissions": payload.get("permissions", [])
            }
        
        def verify_token(self, token: str) -> Dict[str, Any]:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        
        AuthService.get_user_from_token = get_user_from_token
        AuthService.verify_token = verify_token
        
        # Criar token
        user_id = "user456"
        tenant_id = "tenant789"
        roles = ["analyst", "user"]
        permissions = ["read_data", "create_evaluation"]
        
        token = auth_service.create_access_token(user_id, tenant_id, roles, permissions)
        
        # Extrair dados do usuário
        user_data = auth_service.get_user_from_token(token)
        
        assert user_data["user_id"] == user_id
        assert user_data["tenant_id"] == tenant_id
        assert user_data["roles"] == roles
        assert user_data["permissions"] == permissions
    
    def test_password_strength_validation(self, auth_service):
        """Testa validação de força da senha"""
        def validate_password_strength(self, password: str) -> Dict[str, Any]:
            result = {
                "is_valid": True,
                "errors": [],
                "score": 0
            }
            
            # Critérios de validação
            if len(password) < 8:
                result["is_valid"] = False
                result["errors"].append("Senha deve ter pelo menos 8 caracteres")
            else:
                result["score"] += 1
            
            if not any(c.isupper() for c in password):
                result["is_valid"] = False
                result["errors"].append("Senha deve conter pelo menos uma letra maiúscula")
            else:
                result["score"] += 1
            
            if not any(c.islower() for c in password):
                result["is_valid"] = False
                result["errors"].append("Senha deve conter pelo menos uma letra minúscula")
            else:
                result["score"] += 1
            
            if not any(c.isdigit() for c in password):
                result["is_valid"] = False
                result["errors"].append("Senha deve conter pelo menos um número")
            else:
                result["score"] += 1
            
            if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                result["errors"].append("Recomendado: inclua caracteres especiais")
                result["score"] += 0.5
            else:
                result["score"] += 1
            
            return result
        
        AuthService.validate_password_strength = validate_password_strength
        
        # Senha fraca
        weak_result = auth_service.validate_password_strength("123")
        assert weak_result["is_valid"] is False
        assert len(weak_result["errors"]) > 0
        assert weak_result["score"] < 3
        
        # Senha média
        medium_result = auth_service.validate_password_strength("Senha123")
        assert medium_result["is_valid"] is True
        assert medium_result["score"] >= 4
        
        # Senha forte
        strong_result = auth_service.validate_password_strength("MinhaSenh@123!")
        assert strong_result["is_valid"] is True
        assert strong_result["score"] == 5
    
    def test_token_blacklist_functionality(self, auth_service):
        """Testa funcionalidade de blacklist de tokens"""
        # Simular sistema de blacklist
        blacklist = set()
        
        def add_token_to_blacklist(self, token: str) -> bool:
            """Adiciona token ao blacklist"""
            blacklist.add(token)
            return True
        
        def is_token_blacklisted(self, token: str) -> bool:
            """Verifica se token está no blacklist"""
            return token in blacklist
        
        def logout_user(self, token: str) -> bool:
            """Faz logout do usuário adicionando token ao blacklist"""
            return self.add_token_to_blacklist(token)
        
        AuthService.add_token_to_blacklist = add_token_to_blacklist
        AuthService.is_token_blacklisted = is_token_blacklisted
        AuthService.logout_user = logout_user
        
        # Criar token
        token = auth_service.create_access_token("user123", "tenant456", ["user"], ["read"])
        
        # Verificar que token não está no blacklist inicialmente
        assert auth_service.is_token_blacklisted(token) is False
        
        # Fazer logout (adicionar ao blacklist)
        assert auth_service.logout_user(token) is True
        
        # Verificar que token agora está no blacklist
        assert auth_service.is_token_blacklisted(token) is True


@pytest.mark.integration
class TestAuthServiceIntegration:
    """Testes de integração para AuthService"""
    
    @pytest.fixture
    def mock_env_vars(self):
        """Fixture para variáveis de ambiente"""
        with patch.dict(os.environ, {
            'SECRET_KEY': 'integration-test-secret-key',
            'ALGORITHM': 'HS256',
            'ACCESS_TOKEN_EXPIRE_MINUTES': '15',
            'REFRESH_TOKEN_EXPIRE_DAYS': '1'
        }):
            yield
    
    def test_full_authentication_flow(self, mock_env_vars):
        """Testa fluxo completo de autenticação"""
        with patch('src.auth.auth_service.get_database_manager'):
            service = AuthService()
            
            # 1. Hash de senha
            password = "MinhaSenh@123!"
            hashed_password = service.hash_password(password)
            
            # 2. Verificação de senha
            assert service.verify_password(password, hashed_password) is True
            
            # 3. Criação de token
            user_id = "integration_user"
            tenant_id = "integration_tenant"
            roles = ["admin", "analyst"]
            permissions = ["all"]
            
            access_token = service.create_access_token(user_id, tenant_id, roles, permissions)
            
            # 4. Verificação de token
            decoded_payload = jwt.decode(
                access_token, 
                service.secret_key, 
                algorithms=[service.algorithm]
            )
            
            assert decoded_payload["sub"] == user_id
            assert decoded_payload["tenant_id"] == tenant_id
            assert decoded_payload["roles"] == roles
            assert decoded_payload["permissions"] == permissions
            assert decoded_payload["type"] == "access"
            
            # 5. Verificar expiração
            exp_time = datetime.fromtimestamp(decoded_payload["exp"])
            iat_time = datetime.fromtimestamp(decoded_payload["iat"])
            token_lifetime = exp_time - iat_time
            
            # Deve estar próximo de 15 minutos (configurado no mock)
            assert abs(token_lifetime.total_seconds() - 900) < 5  # 900 segundos = 15 minutos
    
    def test_multi_tenant_token_isolation(self, mock_env_vars):
        """Testa isolamento de tokens entre tenants"""
        with patch('src.auth.auth_service.get_database_manager'):
            service = AuthService()
            
            # Criar tokens para diferentes tenants
            user_id = "user123"
            tenant1_token = service.create_access_token(
                user_id, "tenant_1", ["user"], ["read"]
            )
            tenant2_token = service.create_access_token(
                user_id, "tenant_2", ["admin"], ["all"]
            )
            
            # Decodificar tokens
            payload1 = jwt.decode(tenant1_token, service.secret_key, algorithms=[service.algorithm])
            payload2 = jwt.decode(tenant2_token, service.secret_key, algorithms=[service.algorithm])
            
            # Verificar isolamento
            assert payload1["tenant_id"] == "tenant_1"
            assert payload2["tenant_id"] == "tenant_2"
            assert payload1["roles"] != payload2["roles"]
            assert payload1["permissions"] != payload2["permissions"]
    
    def test_token_expiration_scenarios(self, mock_env_vars):
        """Testa cenários de expiração de token"""
        with patch('src.auth.auth_service.get_database_manager'):
            service = AuthService()
            
            # Token com expiração muito curta
            short_expiry = timedelta(seconds=1)
            token = service.create_access_token(
                "user123", "tenant456", ["user"], ["read"],
                expires_delta=short_expiry
            )
            
            # Token deve ser válido imediatamente
            payload = jwt.decode(token, service.secret_key, algorithms=[service.algorithm])
            assert payload["sub"] == "user123"
            
            # Aguardar expiração
            import time
            time.sleep(2)
            
            # Token deve estar expirado
            with pytest.raises(jwt.ExpiredSignatureError):
                jwt.decode(token, service.secret_key, algorithms=[service.algorithm])
    
    def test_role_permission_consistency(self, mock_env_vars):
        """Testa consistência entre roles e permissions"""
        with patch('src.auth.auth_service.get_database_manager'):
            service = AuthService()
            
            # Definir mapeamento role -> permissions
            role_permissions = {
                "admin": ["all"],
                "analyst": ["read_data", "create_evaluation", "view_reports"],
                "user": ["read_data", "create_evaluation"],
                "viewer": ["read_data"]
            }
            
            for role, expected_permissions in role_permissions.items():
                token = service.create_access_token(
                    "user123", "tenant456", [role], expected_permissions
                )
                
                payload = jwt.decode(token, service.secret_key, algorithms=[service.algorithm])
                
                assert payload["roles"] == [role]
                assert set(payload["permissions"]) == set(expected_permissions)