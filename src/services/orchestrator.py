# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Orquestrador de Microserviços
Gerencia o ciclo de vida de todos os microserviços do Valion.
"""

import asyncio
import signal
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import uvicorn
from contextlib import asynccontextmanager

from .service_registry import ServiceRegistry, get_service_registry
from .api_gateway import APIGateway, GatewayConfig, get_default_gateway_config
from .data_processing_service import DataProcessingService, create_data_processing_service
from .ml_service import MLService, create_ml_service
from .base_service import BaseService
from ..monitoring.metrics import get_metrics_collector, init_metrics_collector
from ..monitoring.logging_config import setup_structured_logging, get_logger
from ..config.settings import Settings
import structlog


@dataclass
class ServiceConfig:
    """Configuração de um serviço."""
    name: str
    host: str
    port: int
    enabled: bool = True
    dependencies: List[str] = None
    startup_delay: int = 0  # Delay em segundos antes de iniciar
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class OrchestrationConfig:
    """Configuração do orquestrador."""
    redis_url: str = "redis://localhost:6379/0"
    enable_service_registry: bool = True
    enable_api_gateway: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    graceful_shutdown_timeout: int = 30
    service_startup_timeout: int = 60
    health_check_interval: int = 30


class ServiceOrchestrator:
    """Orquestrador de microserviços."""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.services: Dict[str, BaseService] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.service_registry: Optional[ServiceRegistry] = None
        self.api_gateway: Optional[APIGateway] = None
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Logging
        self.logger = get_logger("orchestrator")
        self.struct_logger = structlog.get_logger("orchestrator")
        
        # Métricas
        self.metrics = get_metrics_collector()
        
        # Configurar services
        self._configure_services()
        
        # Configurar signal handlers
        self._setup_signal_handlers()
    
    def _configure_services(self):
        """Configura todos os serviços."""
        # Service Registry
        if self.config.enable_service_registry:
            self.service_configs["service_registry"] = ServiceConfig(
                name="service_registry",
                host="localhost",
                port=0,  # Não é um serviço HTTP
                enabled=True,
                dependencies=[]
            )
        
        # API Gateway
        if self.config.enable_api_gateway:
            self.service_configs["api_gateway"] = ServiceConfig(
                name="api_gateway",
                host="0.0.0.0",
                port=8000,
                enabled=True,
                dependencies=["service_registry"],
                startup_delay=2
            )
        
        # Data Processing Service
        self.service_configs["data_processing_service"] = ServiceConfig(
            name="data_processing_service",
            host="localhost",
            port=8001,
            enabled=True,
            dependencies=["service_registry"],
            startup_delay=1
        )
        
        # ML Service
        self.service_configs["ml_service"] = ServiceConfig(
            name="ml_service",
            host="localhost",
            port=8002,
            enabled=True,
            dependencies=["service_registry", "data_processing_service"],
            startup_delay=3
        )
        
        # Adicionar configurações de outros serviços conforme necessário
        # self.service_configs["geospatial_service"] = ServiceConfig(...)
        # self.service_configs["reporting_service"] = ServiceConfig(...)
    
    def _setup_signal_handlers(self):
        """Configura handlers para sinais do sistema."""
        def signal_handler(signum, frame):
            self.struct_logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Inicia todos os serviços."""
        try:
            self.struct_logger.info("Starting service orchestrator")
            
            # Inicializar service registry
            if self.config.enable_service_registry:
                await self._start_service_registry()
            
            # Inicializar serviços na ordem de dependência
            await self._start_services_in_order()
            
            # Inicializar API Gateway
            if self.config.enable_api_gateway:
                await self._start_api_gateway()
            
            # Iniciar servidor de métricas
            if self.config.enable_metrics:
                self.metrics.start_metrics_server(port=9090)
            
            self.running = True
            self.struct_logger.info("All services started successfully")
            
            # Aguardar shutdown
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.struct_logger.error(f"Error starting orchestrator: {e}")
            raise
    
    async def shutdown(self):
        """Para todos os serviços gracefully."""
        try:
            self.struct_logger.info("Shutting down orchestrator")
            self.running = False
            
            # Parar API Gateway primeiro
            if self.api_gateway:
                await self._stop_api_gateway()
            
            # Parar serviços em ordem reversa
            await self._stop_services_in_reverse_order()
            
            # Parar service registry por último
            if self.service_registry:
                await self._stop_service_registry()
            
            self.struct_logger.info("Orchestrator shutdown complete")
            self.shutdown_event.set()
            
        except Exception as e:
            self.struct_logger.error(f"Error during shutdown: {e}")
            self.shutdown_event.set()
    
    async def _start_service_registry(self):
        """Inicia o service registry."""
        try:
            self.struct_logger.info("Starting service registry")
            
            self.service_registry = ServiceRegistry(
                redis_url=self.config.redis_url,
                health_check_interval=self.config.health_check_interval
            )
            
            await self.service_registry.start()
            
            self.struct_logger.info("Service registry started")
            
        except Exception as e:
            self.struct_logger.error(f"Error starting service registry: {e}")
            raise
    
    async def _stop_service_registry(self):
        """Para o service registry."""
        try:
            if self.service_registry:
                await self.service_registry.stop()
                self.service_registry = None
                self.struct_logger.info("Service registry stopped")
                
        except Exception as e:
            self.struct_logger.error(f"Error stopping service registry: {e}")
    
    async def _start_api_gateway(self):
        """Inicia o API Gateway."""
        try:
            self.struct_logger.info("Starting API Gateway")
            
            gateway_config = get_default_gateway_config()
            gateway_config.host = self.service_configs["api_gateway"].host
            gateway_config.port = self.service_configs["api_gateway"].port
            
            self.api_gateway = APIGateway(gateway_config)
            
            if self.service_registry:
                self.api_gateway.set_service_registry(self.service_registry)
            
            # Configurar dependências
            for dep in self.service_configs["api_gateway"].dependencies:
                if dep != "service_registry":
                    service_config = self.service_configs[dep]
                    service_url = f"http://{service_config.host}:{service_config.port}"
                    self.api_gateway.add_dependency(dep, service_url)
            
            await self.api_gateway.start()
            
            self.struct_logger.info("API Gateway started")
            
        except Exception as e:
            self.struct_logger.error(f"Error starting API Gateway: {e}")
            raise
    
    async def _stop_api_gateway(self):
        """Para o API Gateway."""
        try:
            if self.api_gateway:
                await self.api_gateway.stop()
                self.api_gateway = None
                self.struct_logger.info("API Gateway stopped")
                
        except Exception as e:
            self.struct_logger.error(f"Error stopping API Gateway: {e}")
    
    async def _start_services_in_order(self):
        """Inicia serviços na ordem de dependência."""
        # Ordenar serviços por dependência
        ordered_services = self._resolve_service_order()
        
        for service_name in ordered_services:
            if service_name in ["service_registry", "api_gateway"]:
                continue  # Já tratados separadamente
            
            await self._start_service(service_name)
    
    async def _stop_services_in_reverse_order(self):
        """Para serviços em ordem reversa."""
        ordered_services = self._resolve_service_order()
        
        for service_name in reversed(ordered_services):
            if service_name in ["service_registry", "api_gateway"]:
                continue  # Já tratados separadamente
            
            await self._stop_service(service_name)
    
    async def _start_service(self, service_name: str):
        """Inicia um serviço específico."""
        try:
            service_config = self.service_configs[service_name]
            
            if not service_config.enabled:
                self.struct_logger.info(f"Service {service_name} is disabled, skipping")
                return
            
            self.struct_logger.info(f"Starting service: {service_name}")
            
            # Aguardar delay de startup
            if service_config.startup_delay > 0:
                await asyncio.sleep(service_config.startup_delay)
            
            # Criar serviço
            service = await self._create_service(service_name, service_config)
            
            if service:
                # Configurar service registry
                if self.service_registry:
                    service.set_service_registry(self.service_registry)
                
                # Configurar dependências
                for dep_name in service_config.dependencies:
                    if dep_name != "service_registry":
                        dep_config = self.service_configs[dep_name]
                        dep_url = f"http://{dep_config.host}:{dep_config.port}"
                        service.add_dependency(dep_name, dep_url)
                
                # Iniciar serviço
                await service.start()
                
                # Armazenar referência
                self.services[service_name] = service
                
                self.struct_logger.info(f"Service {service_name} started successfully")
            
        except Exception as e:
            self.struct_logger.error(f"Error starting service {service_name}: {e}")
            raise
    
    async def _stop_service(self, service_name: str):
        """Para um serviço específico."""
        try:
            if service_name in self.services:
                service = self.services[service_name]
                await service.stop()
                del self.services[service_name]
                
                self.struct_logger.info(f"Service {service_name} stopped")
                
        except Exception as e:
            self.struct_logger.error(f"Error stopping service {service_name}: {e}")
    
    async def _create_service(self, service_name: str, config: ServiceConfig) -> Optional[BaseService]:
        """Cria instância de um serviço."""
        try:
            if service_name == "data_processing_service":
                return create_data_processing_service(config.host, config.port)
            
            elif service_name == "ml_service":
                return create_ml_service(config.host, config.port)
            
            # Adicionar outros serviços conforme implementados
            # elif service_name == "geospatial_service":
            #     return create_geospatial_service(config.host, config.port)
            
            else:
                self.struct_logger.warning(f"Unknown service type: {service_name}")
                return None
                
        except Exception as e:
            self.struct_logger.error(f"Error creating service {service_name}: {e}")
            return None
    
    def _resolve_service_order(self) -> List[str]:
        """Resolve ordem de inicialização baseada em dependências."""
        ordered = []
        visited = set()
        
        def visit(service_name: str):
            if service_name in visited:
                return
            
            visited.add(service_name)
            
            # Visitar dependências primeiro
            config = self.service_configs.get(service_name)
            if config:
                for dep in config.dependencies:
                    if dep != "service_registry":  # Registry é tratado separadamente
                        visit(dep)
            
            ordered.append(service_name)
        
        # Visitar todos os serviços
        for service_name in self.service_configs:
            if service_name not in ["service_registry", "api_gateway"]:
                visit(service_name)
        
        return ordered
    
    def get_service_status(self) -> Dict[str, Any]:
        """Obtém status de todos os serviços."""
        status = {
            "orchestrator": {
                "running": self.running,
                "total_services": len(self.service_configs),
                "active_services": len(self.services),
                "timestamp": datetime.now().isoformat()
            },
            "services": {}
        }
        
        # Status dos serviços
        for service_name, service in self.services.items():
            try:
                status["services"][service_name] = {
                    "running": service.is_running,
                    "status": service.service_info.status,
                    "host": service.service_info.host,
                    "port": service.service_info.port,
                    "started_at": service.service_info.started_at.isoformat()
                }
            except Exception as e:
                status["services"][service_name] = {
                    "running": False,
                    "status": "error",
                    "error": str(e)
                }
        
        # Status do service registry
        if self.service_registry:
            status["service_registry"] = {
                "running": self.service_registry.running,
                "redis_url": self.service_registry.redis_url
            }
        
        # Status do API Gateway
        if self.api_gateway:
            status["api_gateway"] = {
                "running": self.api_gateway.is_running,
                "host": self.api_gateway.service_info.host,
                "port": self.api_gateway.service_info.port,
                "routes_count": len(self.api_gateway.routes)
            }
        
        return status
    
    async def restart_service(self, service_name: str):
        """Reinicia um serviço específico."""
        try:
            self.struct_logger.info(f"Restarting service: {service_name}")
            
            # Parar serviço
            await self._stop_service(service_name)
            
            # Aguardar um momento
            await asyncio.sleep(2)
            
            # Iniciar serviço
            await self._start_service(service_name)
            
            self.struct_logger.info(f"Service {service_name} restarted successfully")
            
        except Exception as e:
            self.struct_logger.error(f"Error restarting service {service_name}: {e}")
            raise
    
    async def scale_service(self, service_name: str, instances: int):
        """Escala um serviço (placeholder para implementação futura)."""
        # Implementar scaling horizontal
        self.struct_logger.info(f"Scaling service {service_name} to {instances} instances")
        # Por enquanto, apenas log
        pass
    
    @asynccontextmanager
    async def lifespan(self):
        """Context manager para ciclo de vida do orquestrador."""
        try:
            await self.start()
            yield self
        finally:
            await self.shutdown()


async def create_orchestrator(config: OrchestrationConfig = None) -> ServiceOrchestrator:
    """Cria e configura o orquestrador."""
    if config is None:
        config = OrchestrationConfig()
    
    # Configurar logging
    if config.enable_logging:
        logging_config = {
            "level": "INFO",
            "console_handler": True,
            "file_handler": True,
            "development_mode": False
        }
        setup_structured_logging(logging_config)
    
    # Configurar métricas
    if config.enable_metrics:
        init_metrics_collector()
    
    return ServiceOrchestrator(config)


async def main():
    """Função principal para executar o orquestrador."""
    try:
        # Criar configuração
        config = OrchestrationConfig(
            redis_url="redis://localhost:6379/0",
            enable_service_registry=True,
            enable_api_gateway=True,
            enable_metrics=True,
            enable_logging=True
        )
        
        # Criar e executar orquestrador
        orchestrator = await create_orchestrator(config)
        
        async with orchestrator.lifespan():
            # Manter rodando até receber sinal
            await orchestrator.shutdown_event.wait()
            
    except KeyboardInterrupt:
        print("Shutdown requested by user")
    except Exception as e:
        print(f"Error running orchestrator: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())