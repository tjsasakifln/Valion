# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Classe Base para Microserviços
Define interface comum e funcionalidades compartilhadas.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from contextlib import asynccontextmanager
import aiohttp
from pydantic import BaseModel, Field
import structlog
from ..monitoring.metrics import get_metrics_collector
from ..monitoring.logging_config import get_logger


@dataclass
class ServiceInfo:
    """Informações do serviço."""
    name: str
    version: str
    host: str
    port: int
    health_endpoint: str
    status: str = "starting"
    started_at: datetime = None
    instance_id: str = None
    
    def __post_init__(self):
        if self.instance_id is None:
            self.instance_id = str(uuid.uuid4())
        if self.started_at is None:
            self.started_at = datetime.now()


@dataclass
class ServiceRequest:
    """Requisição entre serviços com timeout configurável."""
    service_name: str
    method: str
    endpoint: str
    payload: Dict[str, Any]
    headers: Dict[str, str] = None
    timeout: int = 30  # Timeout padrão, pode ser sobrescrito por operação
    request_id: str = None
    retry_count: int = 0  # Número de tentativas para retry
    
    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())
        if self.headers is None:
            self.headers = {}
        
        # Validar timeout mínimo para evitar valores muito baixos
        if self.timeout < 1:
            self.timeout = 1


@dataclass
class ServiceResponse:
    """Resposta entre serviços."""
    success: bool
    data: Any = None
    error: str = None
    status_code: int = 200
    headers: Dict[str, str] = None
    processing_time: float = 0.0
    request_id: str = None


class ServiceHealthCheck(BaseModel):
    """Modelo para health check."""
    status: str = Field(..., description="Status do serviço")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="Versão do serviço")
    uptime: float = Field(..., description="Uptime em segundos")
    dependencies: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class BaseService(ABC):
    """Classe base para todos os microserviços."""
    
    def __init__(self, service_name: str, version: str = "1.0.0", 
                 host: str = "localhost", port: int = 8000):
        self.service_info = ServiceInfo(
            name=service_name,
            version=version,
            host=host,
            port=port,
            health_endpoint=f"/health"
        )
        
        # Configurar logging
        self.logger = get_logger(f"service.{service_name}")
        self.struct_logger = structlog.get_logger(service_name)
        
        # Métricas
        self.metrics = get_metrics_collector()
        
        # Registry de serviços
        self.service_registry = None
        self.dependencies = {}
        
        # Configurações de timeout por serviço
        self.service_timeouts = self._get_default_service_timeouts()
        
        # Estado interno
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Callbacks
        self.startup_callbacks: List[Callable] = []
        self.shutdown_callbacks: List[Callable] = []
        
        # Client HTTP para comunicação entre serviços
        self.http_session = None
        
        self.logger.info(f"Service {service_name} initialized", 
                        service_info=asdict(self.service_info))
    
    def _get_default_service_timeouts(self) -> Dict[str, int]:
        """
        Define timeouts padrão por tipo de serviço baseado na natureza da operação.
        
        Returns:
            Dicionário com timeouts por serviço
        """
        return {
            # Serviços de ML/processamento pesado - timeouts longos
            "ml_service": 180,  # 3 minutos para treinamento/predição
            "model_service": 120,  # 2 minutos para operações de modelo
            "data_processing_service": 90,  # 1.5 minutos para transformações
            "analytics_service": 60,  # 1 minuto para análises
            
            # Serviços de dados - timeouts médios
            "database_service": 45,  # 45 segundos para queries complexas
            "cache_service": 10,  # 10 segundos para operações de cache
            "storage_service": 30,  # 30 segundos para I/O de arquivos
            
            # Serviços de comunicação - timeouts curtos
            "notification_service": 15,  # 15 segundos para notificações
            "websocket_service": 5,  # 5 segundos para WebSocket
            "auth_service": 10,  # 10 segundos para autenticação
            
            # Serviços de infraestrutura - timeouts muito curtos
            "health_service": 5,  # 5 segundos para health checks
            "metrics_service": 10,  # 10 segundos para métricas
            "logging_service": 5,  # 5 segundos para logs
            
            # Default para serviços não especificados
            "default": 30
        }
    
    async def start(self):
        """Inicia o serviço."""
        try:
            self.service_info.status = "starting"
            self.logger.info(f"Starting service {self.service_info.name}")
            
            # Configurar HTTP session
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": f"Valion-{self.service_info.name}/{self.service_info.version}"}
            )
            
            # Executar callbacks de startup
            for callback in self.startup_callbacks:
                await callback()
            
            # Inicializar serviço específico
            await self.initialize()
            
            # Registrar no service registry
            if self.service_registry:
                await self.service_registry.register_service(self.service_info)
            
            self.service_info.status = "running"
            self.is_running = True
            
            # Registrar métricas
            self.metrics.record_celery_task(
                task_name=f"{self.service_info.name}.startup",
                status="success",
                duration=0.0
            )
            
            self.logger.info(f"Service {self.service_info.name} started successfully")
            
        except Exception as e:
            self.service_info.status = "failed"
            self.logger.error(f"Failed to start service {self.service_info.name}: {e}")
            raise
    
    async def stop(self):
        """Para o serviço."""
        try:
            self.service_info.status = "stopping"
            self.logger.info(f"Stopping service {self.service_info.name}")
            
            # Sinalizar shutdown
            self.shutdown_event.set()
            
            # Desregistrar do service registry
            if self.service_registry:
                await self.service_registry.unregister_service(self.service_info.name)
            
            # Executar callbacks de shutdown
            for callback in self.shutdown_callbacks:
                await callback()
            
            # Finalizar serviço específico
            await self.cleanup()
            
            # Fechar HTTP session
            if self.http_session:
                await self.http_session.close()
            
            self.service_info.status = "stopped"
            self.is_running = False
            
            self.logger.info(f"Service {self.service_info.name} stopped successfully")
            
        except Exception as e:
            self.service_info.status = "error"
            self.logger.error(f"Error stopping service {self.service_info.name}: {e}")
            raise
    
    @abstractmethod
    async def initialize(self):
        """Inicialização específica do serviço."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Limpeza específica do serviço."""
        pass
    
    async def health_check(self) -> ServiceHealthCheck:
        """Verifica saúde do serviço."""
        uptime = (datetime.now() - self.service_info.started_at).total_seconds()
        
        # Verificar dependências
        dependencies_status = {}
        for dep_name, dep_url in self.dependencies.items():
            try:
                async with self.http_session.get(f"{dep_url}/health", timeout=5) as response:
                    if response.status == 200:
                        dependencies_status[dep_name] = "healthy"
                    else:
                        dependencies_status[dep_name] = "unhealthy"
            except Exception:
                dependencies_status[dep_name] = "unreachable"
        
        # Métricas básicas
        metrics = {
            "uptime_seconds": uptime,
            "status": self.service_info.status,
            "is_running": self.is_running
        }
        
        # Determinar status geral
        if not self.is_running:
            status = "down"
        elif any(dep_status != "healthy" for dep_status in dependencies_status.values()):
            status = "degraded"
        else:
            status = "healthy"
        
        return ServiceHealthCheck(
            status=status,
            version=self.service_info.version,
            uptime=uptime,
            dependencies=dependencies_status,
            metrics=metrics
        )
    
    async def call_service(self, request: ServiceRequest) -> ServiceResponse:
        """
        Chama outro serviço com timeout inteligente baseado no tipo de serviço.
        
        O timeout é determinado pela seguinte prioridade:
        1. Timeout explícito no ServiceRequest (se diferente do padrão)
        2. Timeout configurado para o serviço específico
        3. Timeout padrão (30 segundos)
        """
        start_time = datetime.now()
        
        # Determinar timeout apropriado
        effective_timeout = self._get_effective_timeout(request)
        
        try:
            # Buscar URL do serviço
            service_url = await self._get_service_url(request.service_name)
            
            if not service_url:
                return ServiceResponse(
                    success=False,
                    error=f"Service {request.service_name} not found",
                    status_code=404,
                    request_id=request.request_id
                )
            
            # Preparar requisição
            url = f"{service_url}{request.endpoint}"
            headers = {
                **request.headers,
                "Content-Type": "application/json",
                "X-Request-ID": request.request_id,
                "X-Service-Name": self.service_info.name,
                "X-Timeout": str(effective_timeout)  # Para debugging
            }
            
            # Log do timeout usado para debugging
            self.logger.debug(
                f"Calling {request.service_name}{request.endpoint} with timeout {effective_timeout}s",
                service=request.service_name,
                endpoint=request.endpoint,
                timeout=effective_timeout,
                request_id=request.request_id
            )
            
            # Fazer requisição com timeout específico
            timeout = aiohttp.ClientTimeout(total=effective_timeout)
            async with self.http_session.request(
                method=request.method,
                url=url,
                json=request.payload,
                headers=headers,
                timeout=timeout
            ) as response:
                
                response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Registrar métricas com informações de timeout
                self.metrics.record_api_request(
                    method=request.method,
                    endpoint=request.endpoint,
                    status_code=response.status,
                    duration=processing_time
                )
                
                # Log de sucesso com tempo de resposta
                self.logger.info(
                    f"Service call completed: {request.service_name}{request.endpoint}",
                    service=request.service_name,
                    endpoint=request.endpoint,
                    status_code=response.status,
                    duration=processing_time,
                    timeout_used=effective_timeout,
                    request_id=request.request_id
                )
                
                return ServiceResponse(
                    success=response.status < 400,
                    data=response_data,
                    status_code=response.status,
                    processing_time=processing_time,
                    request_id=request.request_id
                )
                
        except asyncio.TimeoutError:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(
                f"Timeout calling service {request.service_name}{request.endpoint}",
                service=request.service_name,
                endpoint=request.endpoint,
                timeout_used=effective_timeout,
                duration=processing_time,
                request_id=request.request_id
            )
            
            # Registrar timeout nas métricas
            self.metrics.record_api_request(
                method=request.method,
                endpoint=request.endpoint,
                status_code=408,
                duration=processing_time
            )
            
            return ServiceResponse(
                success=False,
                error=f"Service call timeout after {effective_timeout}s",
                status_code=408,
                processing_time=processing_time,
                request_id=request.request_id
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(
                f"Error calling service {request.service_name}{request.endpoint}: {e}",
                service=request.service_name,
                endpoint=request.endpoint,
                error=str(e),
                duration=processing_time,
                request_id=request.request_id
            )
            
            return ServiceResponse(
                success=False,
                error=str(e),
                status_code=500,
                processing_time=processing_time,
                request_id=request.request_id
            )
    
    def _get_effective_timeout(self, request: ServiceRequest) -> int:
        """
        Determina o timeout efetivo para a requisição.
        
        Args:
            request: Requisição do serviço
            
        Returns:
            Timeout em segundos
        """
        # Se o timeout foi explicitamente definido (diferente do padrão), usar ele
        if request.timeout != 30:  # 30 é o padrão
            return request.timeout
        
        # Buscar timeout específico para o serviço
        service_timeout = self.service_timeouts.get(request.service_name)
        if service_timeout:
            return service_timeout
        
        # Usar timeout padrão
        return self.service_timeouts.get("default", 30)
    
    async def _get_service_url(self, service_name: str) -> Optional[str]:
        """Obtém URL do serviço pelo nome."""
        if self.service_registry:
            service_info = await self.service_registry.get_service(service_name)
            if service_info:
                return f"http://{service_info.host}:{service_info.port}"
        
        # Fallback para configuração local
        return self.dependencies.get(service_name)
    
    def add_dependency(self, service_name: str, service_url: str):
        """Adiciona dependência de serviço."""
        self.dependencies[service_name] = service_url
        self.logger.info(f"Added dependency: {service_name} -> {service_url}")
    
    def configure_service_timeout(self, service_name: str, timeout: int):
        """
        Configura timeout específico para um serviço.
        
        Args:
            service_name: Nome do serviço
            timeout: Timeout em segundos
        """
        if timeout < 1:
            raise ValueError("Timeout deve ser pelo menos 1 segundo")
        
        self.service_timeouts[service_name] = timeout
        self.logger.info(f"Configured timeout for {service_name}: {timeout}s")
    
    def get_service_timeout(self, service_name: str) -> int:
        """
        Obtém timeout configurado para um serviço.
        
        Args:
            service_name: Nome do serviço
            
        Returns:
            Timeout em segundos
        """
        return self.service_timeouts.get(service_name, self.service_timeouts.get("default", 30))
    
    def create_service_request(self, service_name: str, method: str, endpoint: str, 
                             payload: Dict[str, Any], timeout: Optional[int] = None,
                             headers: Optional[Dict[str, str]] = None) -> ServiceRequest:
        """
        Factory method para criar ServiceRequest com timeout apropriado.
        
        Args:
            service_name: Nome do serviço
            method: Método HTTP
            endpoint: Endpoint do serviço
            payload: Dados da requisição
            timeout: Timeout customizado (opcional)
            headers: Headers customizados (opcional)
            
        Returns:
            ServiceRequest configurado
        """
        # Se não foi especificado timeout, usar o configurado para o serviço
        if timeout is None:
            timeout = self.get_service_timeout(service_name)
        
        return ServiceRequest(
            service_name=service_name,
            method=method,
            endpoint=endpoint,
            payload=payload,
            timeout=timeout,
            headers=headers or {}
        )
    
    def add_startup_callback(self, callback: Callable):
        """Adiciona callback de startup."""
        self.startup_callbacks.append(callback)
    
    def add_shutdown_callback(self, callback: Callable):
        """Adiciona callback de shutdown."""
        self.shutdown_callbacks.append(callback)
    
    @asynccontextmanager
    async def service_context(self):
        """Context manager para ciclo de vida do serviço."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    def set_service_registry(self, registry):
        """Define o registry de serviços."""
        self.service_registry = registry
        self.logger.info(f"Service registry set for {self.service_info.name}")
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Envia evento para outros serviços com timeout apropriado."""
        if self.service_registry:
            services = await self.service_registry.get_all_services()
            
            for service in services:
                if service.name != self.service_info.name:
                    try:
                        # Usar factory method que aplica timeout apropriado automaticamente
                        request = self.create_service_request(
                            service_name=service.name,
                            method="POST",
                            endpoint="/events",
                            payload={"event_type": event_type, "data": data}
                            # timeout será definido automaticamente baseado no tipo de serviço
                        )
                        
                        await self.call_service(request)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to send event to {service.name}: {e}")
    
    async def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Manipula eventos recebidos de outros serviços."""
        self.logger.info(f"Received event: {event_type}", data=data)
        
        # Implementar lógica específica nos serviços filhos
        pass
    
    def get_service_info(self) -> Dict[str, Any]:
        """Retorna informações do serviço incluindo configurações de timeout."""
        return {
            **asdict(self.service_info),
            "is_running": self.is_running,
            "dependencies": list(self.dependencies.keys()),
            "timeout_configuration": self.service_timeouts.copy()
        }
    
    def get_timeout_info(self) -> Dict[str, Any]:
        """
        Retorna informações detalhadas sobre configurações de timeout.
        
        Returns:
            Informações sobre timeouts configurados
        """
        return {
            "service_timeouts": self.service_timeouts.copy(),
            "default_timeout": self.service_timeouts.get("default", 30),
            "configured_services": [
                service for service in self.service_timeouts.keys() 
                if service != "default"
            ],
            "timeout_examples": {
                "ml_service": f"{self.get_service_timeout('ml_service')}s (ML operations)",
                "database_service": f"{self.get_service_timeout('database_service')}s (Database queries)",
                "cache_service": f"{self.get_service_timeout('cache_service')}s (Cache operations)",
                "notification_service": f"{self.get_service_timeout('notification_service')}s (Notifications)"
            }
        }
    
    def __str__(self) -> str:
        return f"{self.service_info.name}:{self.service_info.version} ({self.service_info.status})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self}>"