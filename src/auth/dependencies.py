# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
FastAPI dependencies for authentication and authorization
"""

from typing import List, Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .auth_service import auth_service
from ..database.database import get_database_manager, set_current_tenant_id
from ..database.models import User, Tenant


security = HTTPBearer()


def get_db_session():
    """Dependency to get database session."""
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session

def get_tenant_db_session(
    tenant_id: str = Depends(get_current_tenant_id_from_token)
):
    """Dependency to get tenant-aware database session."""
    db_manager = get_database_manager()
    with db_manager.get_tenant_session(tenant_id) as session:
        yield session


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_db_session)
) -> User:
    """Get the current authenticated user from JWT token."""
    token = credentials.credentials
    return auth_service.get_current_user_from_token(token, session)


def get_current_token_payload(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get the current token payload for permission checking."""
    token = credentials.credentials
    payload = auth_service.verify_token(token)
    
    # Set tenant context for this request
    tenant_id = auth_service.get_tenant_id_from_token(payload)
    if tenant_id:
        set_current_tenant_id(tenant_id)
    
    return payload

def get_current_tenant_id_from_token(
    token_payload: Dict[str, Any] = Depends(get_current_token_payload)
) -> str:
    """Get the current tenant ID from JWT token."""
    tenant_id = auth_service.get_tenant_id_from_token(token_payload)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No tenant context in token"
        )
    return tenant_id

def get_current_tenant(
    tenant_id: str = Depends(get_current_tenant_id_from_token),
    session: Session = Depends(get_db_session)
) -> Tenant:
    """Get the current tenant object."""
    db_manager = get_database_manager()
    tenant = db_manager.get_tenant_by_id(session, tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    return tenant


def require_permission(permission: str):
    """Dependency factory to require a specific permission."""
    def check_permission(
        token_payload: Dict[str, Any] = Depends(get_current_token_payload)
    ):
        if not auth_service.check_permission(token_payload, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return token_payload
    
    return check_permission


def require_role(role: str):
    """Dependency factory to require a specific role."""
    def check_role(
        token_payload: Dict[str, Any] = Depends(get_current_token_payload)
    ):
        if not auth_service.check_role(token_payload, role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return token_payload
    
    return check_role


def require_any_role(roles: List[str]):
    """Dependency factory to require any of the specified roles."""
    def check_any_role(
        token_payload: Dict[str, Any] = Depends(get_current_token_payload)
    ):
        if not auth_service.check_any_role(token_payload, roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these roles required: {', '.join(roles)}"
            )
        return token_payload
    
    return check_any_role


def require_permissions(permissions: List[str]):
    """Dependency factory to require all specified permissions."""
    def check_permissions(
        token_payload: Dict[str, Any] = Depends(get_current_token_payload)
    ):
        if not auth_service.check_all_permissions(token_payload, permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"These permissions required: {', '.join(permissions)}"
            )
        return token_payload
    
    return check_permissions


def require_admin():
    """Convenience dependency to require admin role."""
    return require_role("admin")


def require_evaluator():
    """Convenience dependency to require evaluator role."""
    return require_any_role(["admin", "avaliador"])


def require_auditor():
    """Convenience dependency to require auditor role."""
    return require_any_role(["admin", "auditor"])


def optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session: Session = Depends(get_db_session)
) -> Optional[User]:
    """Get the current user if authenticated, otherwise return None."""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        return auth_service.get_current_user_from_token(token, session)
    except HTTPException:
        return None