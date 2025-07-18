# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Microserviços do Valion
Arquitetura modular para escalabilidade e manutenibilidade.
"""

from .api_gateway import APIGateway
from .data_processing_service import DataProcessingService
from .ml_service import MLService
from .geospatial_service import GeospatialService
from .reporting_service import ReportingService
from .audit_service import AuditService
from .service_registry import ServiceRegistry
from .base_service import BaseService

__all__ = [
    'APIGateway',
    'DataProcessingService',
    'MLService',
    'GeospatialService',
    'ReportingService',
    'AuditService',
    'ServiceRegistry',
    'BaseService'
]