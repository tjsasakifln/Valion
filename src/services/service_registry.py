# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Service Registry para Microserviços
Gerencia descoberta e registro de serviços.
"""

import asyncio
import json
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import aioredis
import logging
from dataclasses import dataclass, asdict
from .base_service import ServiceInfo


logger = logging.getLogger(__name__)


@dataclass
class ServiceHealthStatus:
    """Status de saúde do serviço."""
    service_name: str
    status: str
    last_check: datetime
    consecutive_failures: int = 0
    
    def is_healthy(self) -> bool:
        """Verifica se o serviço está saudável."""
        if self.status != "healthy":
            return False
        
        # Considerar stale se não foi verificado há mais de 30 segundos
        if (datetime.now() - self.last_check).total_seconds() > 30:
            return False
        
        return True


class ServiceRegistry:
    """Registry centralizado de serviços."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0",
                 health_check_interval: int = 10):
        self.redis_url = redis_url
        self.health_check_interval = health_check_interval
        self.redis = None
        self.services: Dict[str, ServiceInfo] = {}
        self.health_status: Dict[str, ServiceHealthStatus] = {}
        self.running = False
        self.health_check_task = None
        
        # Callbacks
        self.on_service_registered = []
        self.on_service_unregistered = []
        self.on_service_health_changed = []
    
    async def start(self):
        """Inicia o service registry."""
        try:
            # Conectar ao Redis
            self.redis = await aioredis.from_url(self.redis_url)
            
            # Carregar serviços existentes
            await self._load_services_from_redis()
            
            # Iniciar health check
            self.running = True
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("Service registry started")
            
        except Exception as e:
            logger.error(f"Failed to start service registry: {e}")
            raise
    
    async def stop(self):
        """Para o service registry."""
        try:
            self.running = False
            
            # Cancelar health check
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Fechar conexão Redis
            if self.redis:
                await self.redis.close()
            
            logger.info("Service registry stopped")
            
        except Exception as e:
            logger.error(f"Error stopping service registry: {e}")
    
    async def register_service(self, service_info: ServiceInfo):
        """Registra um serviço."""
        try:
            # Salvar no Redis
            key = f"service:{service_info.name}"
            value = json.dumps(asdict(service_info), default=str)
            
            await self.redis.set(key, value)
            await self.redis.expire(key, 60)  # TTL de 60 segundos
            
            # Salvar localmente
            self.services[service_info.name] = service_info
            
            # Inicializar status de saúde
            self.health_status[service_info.name] = ServiceHealthStatus(
                service_name=service_info.name,
                status="healthy",
                last_check=datetime.now()
            )
            
            # Callbacks
            for callback in self.on_service_registered:
                try:
                    await callback(service_info)
                except Exception as e:
                    logger.error(f"Error in service registered callback: {e}")
            
            logger.info(f"Service registered: {service_info.name}")
            
        except Exception as e:
            logger.error(f"Failed to register service {service_info.name}: {e}")
            raise
    
    async def unregister_service(self, service_name: str):
        """Desregistra um serviço."""
        try:
            # Remover do Redis
            key = f"service:{service_name}"
            await self.redis.delete(key)
            
            # Remover localmente
            service_info = self.services.pop(service_name, None)
            self.health_status.pop(service_name, None)
            
            # Callbacks
            if service_info:
                for callback in self.on_service_unregistered:
                    try:
                        await callback(service_info)
                    except Exception as e:
                        logger.error(f"Error in service unregistered callback: {e}")
            
            logger.info(f"Service unregistered: {service_name}")
            
        except Exception as e:
            logger.error(f"Failed to unregister service {service_name}: {e}")
    
    async def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Obtém informações de um serviço."""
        try:
            # Verificar cache local primeiro
            if service_name in self.services:
                health_status = self.health_status.get(service_name)
                if health_status and health_status.is_healthy():
                    return self.services[service_name]
            
            # Buscar no Redis
            key = f"service:{service_name}"
            value = await self.redis.get(key)
            
            if value:
                service_data = json.loads(value)
                service_info = ServiceInfo(**service_data)
                
                # Atualizar cache local
                self.services[service_name] = service_info
                
                return service_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting service {service_name}: {e}")
            return None
    
    async def get_all_services(self) -> List[ServiceInfo]:
        """Obtém todos os serviços registrados."""
        try:
            services = []
            
            # Buscar todas as chaves de serviço
            keys = await self.redis.keys("service:*")
            
            for key in keys:
                try:
                    value = await self.redis.get(key)
                    if value:
                        service_data = json.loads(value)
                        service_info = ServiceInfo(**service_data)
                        services.append(service_info)
                except Exception as e:
                    logger.error(f"Error parsing service from {key}: {e}")
            
            return services
            
        except Exception as e:
            logger.error(f"Error getting all services: {e}")
            return []
    
    async def get_healthy_services(self) -> List[ServiceInfo]:
        """Obtém apenas os serviços saudáveis."""
        all_services = await self.get_all_services()
        healthy_services = []
        
        for service in all_services:
            health_status = self.health_status.get(service.name)
            if health_status and health_status.is_healthy():
                healthy_services.append(service)
        
        return healthy_services
    
    async def find_services_by_type(self, service_type: str) -> List[ServiceInfo]:
        """Encontra serviços por tipo."""
        all_services = await self.get_all_services()
        return [s for s in all_services if service_type in s.name.lower()]
    
    async def _load_services_from_redis(self):
        """Carrega serviços existentes do Redis."""
        try:
            services = await self.get_all_services()
            
            for service in services:
                self.services[service.name] = service
                self.health_status[service.name] = ServiceHealthStatus(
                    service_name=service.name,
                    status="unknown",
                    last_check=datetime.now()
                )
            
            logger.info(f"Loaded {len(services)} services from Redis")
            
        except Exception as e:
            logger.error(f"Error loading services from Redis: {e}")
    
    async def _health_check_loop(self):
        """Loop de verificação de saúde dos serviços."""
        import aiohttp
        
        while self.running:
            try:
                # Verificar saúde de todos os serviços
                services = list(self.services.values())
                
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as session:
                    
                    for service in services:
                        await self._check_service_health(session, service)
                
                # Limpar serviços mortos
                await self._cleanup_dead_services()
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            
            await asyncio.sleep(self.health_check_interval)
    
    async def _check_service_health(self, session: aiohttp.ClientSession, 
                                   service: ServiceInfo):
        """Verifica saúde de um serviço específico."""
        try:
            health_url = f"http://{service.host}:{service.port}{service.health_endpoint}"
            
            async with session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    status = health_data.get("status", "unknown")
                else:
                    status = "unhealthy"
            
            # Atualizar status
            old_status = self.health_status.get(service.name)
            
            self.health_status[service.name] = ServiceHealthStatus(
                service_name=service.name,
                status=status,
                last_check=datetime.now(),
                consecutive_failures=0
            )
            
            # Renovar TTL no Redis
            key = f"service:{service.name}"
            await self.redis.expire(key, 60)
            
            # Callback se status mudou
            if old_status and old_status.status != status:
                for callback in self.on_service_health_changed:
                    try:
                        await callback(service, old_status.status, status)
                    except Exception as e:
                        logger.error(f"Error in health changed callback: {e}")
            
        except Exception as e:
            # Marcar como não saudável
            old_status = self.health_status.get(service.name)
            failures = old_status.consecutive_failures + 1 if old_status else 1
            
            self.health_status[service.name] = ServiceHealthStatus(
                service_name=service.name,
                status="unhealthy",
                last_check=datetime.now(),
                consecutive_failures=failures
            )
            
            logger.warning(f"Health check failed for {service.name}: {e}")
    
    async def _cleanup_dead_services(self):
        """Remove serviços mortos do registry."""
        dead_services = []
        
        for service_name, health_status in self.health_status.items():
            # Considerar morto se falhou 3 vezes consecutivas
            if health_status.consecutive_failures >= 3:
                dead_services.append(service_name)
        
        for service_name in dead_services:
            logger.info(f"Removing dead service: {service_name}")
            await self.unregister_service(service_name)
    
    def add_service_registered_callback(self, callback):
        """Adiciona callback para quando um serviço é registrado."""
        self.on_service_registered.append(callback)
    
    def add_service_unregistered_callback(self, callback):
        """Adiciona callback para quando um serviço é desregistrado."""
        self.on_service_unregistered.append(callback)
    
    def add_service_health_changed_callback(self, callback):
        """Adiciona callback para quando a saúde de um serviço muda."""
        self.on_service_health_changed.append(callback)
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do registry."""
        all_services = await self.get_all_services()
        healthy_services = await self.get_healthy_services()
        
        status_counts = {}
        for health_status in self.health_status.values():
            status = health_status.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_services": len(all_services),
            "healthy_services": len(healthy_services),
            "status_breakdown": status_counts,
            "services": [
                {
                    "name": service.name,
                    "version": service.version,
                    "status": self.health_status.get(service.name, {}).status,
                    "host": service.host,
                    "port": service.port
                }
                for service in all_services
            ]
        }


# Instância global do registry
_service_registry = None


async def get_service_registry(redis_url: str = "redis://localhost:6379/0") -> ServiceRegistry:
    """Obtém instância global do service registry."""
    global _service_registry
    
    if _service_registry is None:
        _service_registry = ServiceRegistry(redis_url)
        await _service_registry.start()
    
    return _service_registry


async def cleanup_service_registry():
    """Limpa o service registry global."""
    global _service_registry
    
    if _service_registry:
        await _service_registry.stop()
        _service_registry = None