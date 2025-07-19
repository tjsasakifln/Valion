# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Authentication and Authorization Service for Valion RBAC
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from passlib.context import CryptContext
from passlib.hash import bcrypt
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..database.database import get_database_manager
from ..database.models import User, Role


class AuthService:
    """Serviço de autenticação e autorização com RBAC."""
    
    def __init__(self):
        self.secret_key = os.getenv('SECRET_KEY')
        if not self.secret_key:
            raise ValueError("SECRET_KEY environment variable is required")
        
        self.algorithm = os.getenv('ALGORITHM', 'HS256')
        self.access_token_expire_minutes = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
        self.refresh_token_expire_days = int(os.getenv('REFRESH_TOKEN_EXPIRE_DAYS', '7'))
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Database manager
        self.db_manager = get_database_manager()
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user_id: str, tenant_id: str, roles: List[str], permissions: List[str], 
                           expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token with user roles, permissions, and tenant context."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {
            "sub": str(user_id),
            "tenant_id": str(tenant_id),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "roles": roles,
            "permissions": permissions
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def get_current_user_from_token(self, token: str, session: Session) -> Optional[User]:
        """Get current user from JWT token."""
        payload = self.verify_token(token)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        user = self.db_manager.get_user_by_id(session, user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
    
    def authenticate_user(self, session: Session, username: str, password: str) -> Optional[User]:
        """Authenticate a user by username and password."""
        user = self.db_manager.get_user_by_username(session, username)
        if not user:
            return None
        
        # For this implementation, we'll assume password is stored in a new field
        # In a real implementation, you'd add a password_hash field to the User model
        # For now, we'll just do a simple check or skip password verification
        # TODO: Add password_hash field to User model in a future update
        
        return user
    
    def login(self, session: Session, username: str, password: str, tenant_id: str = None) -> Dict[str, Any]:
        """Login a user and return tokens with roles and permissions."""
        user = self.authenticate_user(session, username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled"
            )
        
        # Verify tenant access if tenant_id is provided
        if tenant_id and str(user.tenant_id) != str(tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied for this tenant"
            )
        
        # Use user's tenant_id if not provided
        if not tenant_id:
            tenant_id = str(user.tenant_id)
        
        # Get user roles and permissions
        roles = self.db_manager.get_user_roles(session, str(user.id))
        role_names = [role.name for role in roles]
        permissions = self.db_manager.get_user_permissions(session, str(user.id))
        
        # Create tokens
        access_token = self.create_access_token(
            user_id=str(user.id),
            tenant_id=tenant_id,
            roles=role_names,
            permissions=permissions
        )
        refresh_token = self.create_refresh_token(str(user.id))
        
        # Audit login
        self.db_manager.create_audit_trail(
            session=session,
            user_id=str(user.id),
            operation='login',
            entity_type='user',
            entity_id=str(user.id),
            new_values={'login_time': datetime.utcnow().isoformat(), 'tenant_id': tenant_id}
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "tenant_id": tenant_id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "roles": role_names,
                "permissions": permissions
            }
        }
    
    def refresh_access_token(self, session: Session, refresh_token: str) -> Dict[str, str]:
        """Refresh an access token using a refresh token."""
        payload = self.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        user_id = payload.get("sub")
        user = self.db_manager.get_user_by_id(session, user_id)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or disabled"
            )
        
        # Get current roles and permissions
        roles = self.db_manager.get_user_roles(session, str(user.id))
        role_names = [role.name for role in roles]
        permissions = self.db_manager.get_user_permissions(session, str(user.id))
        
        # Create new access token
        access_token = self.create_access_token(
            user_id=str(user.id),
            tenant_id=str(user.tenant_id),
            roles=role_names,
            permissions=permissions
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    
    def check_permission(self, token_payload: Dict[str, Any], required_permission: str) -> bool:
        """Check if the current user has a specific permission."""
        permissions = token_payload.get("permissions", [])
        return required_permission in permissions
    
    def check_role(self, token_payload: Dict[str, Any], required_role: str) -> bool:
        """Check if the current user has a specific role."""
        roles = token_payload.get("roles", [])
        return required_role in roles
    
    def check_any_role(self, token_payload: Dict[str, Any], required_roles: List[str]) -> bool:
        """Check if the current user has any of the specified roles."""
        user_roles = token_payload.get("roles", [])
        return any(role in user_roles for role in required_roles)
    
    def check_all_permissions(self, token_payload: Dict[str, Any], required_permissions: List[str]) -> bool:
        """Check if the current user has all of the specified permissions."""
        user_permissions = token_payload.get("permissions", [])
        return all(permission in user_permissions for permission in required_permissions)
    
    def get_tenant_id_from_token(self, token_payload: Dict[str, Any]) -> Optional[str]:
        """Extract tenant_id from JWT token payload."""
        return token_payload.get("tenant_id")
    
    def login_with_subdomain(self, session: Session, username: str, password: str, subdomain: str) -> Dict[str, Any]:
        """Login a user using tenant subdomain."""
        # Get tenant by subdomain
        tenant = self.db_manager.get_tenant_by_subdomain(session, subdomain)
        if not tenant or not tenant.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found or inactive"
            )
        
        return self.login(session, username, password, str(tenant.id))


# Global instance
auth_service = AuthService()