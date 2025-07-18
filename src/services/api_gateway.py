# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
API Gateway para Microserviços
Ponto de entrada unificado com roteamento, autenticação e rate limiting.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import aiohttp
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import redis.asyncio as redis
from .base_service import BaseService, ServiceRequest, ServiceResponse
from .service_registry import ServiceRegistry
from ..monitoring.metrics import get_metrics_collector
from ..monitoring.logging_config import get_logger
import structlog


# Modelos Pydantic
class RouteConfig(BaseModel):
    """Configuração de rota."""
    path: str = Field(..., description="Caminho da rota")
    service_name: str = Field(..., description="Nome do serviço de destino")
    method: str = Field(default="GET", description="Método HTTP")
    auth_required: bool = Field(default=True, description="Se autenticação é necessária")
    rate_limit: int = Field(default=100, description="Rate limit por minuto")
    timeout: int = Field(default=30, description="Timeout em segundos")
    cache_ttl: int = Field(default=0, description="TTL do cache em segundos")


class GatewayConfig(BaseModel):
    """Configuração do gateway."""
    host: str = Field(default="0.0.0.0", description="Host do gateway")
    port: int = Field(default=8000, description="Porta do gateway")
    debug: bool = Field(default=False, description="Modo debug")
    cors_origins: List[str] = Field(default=["*"], description="Origens CORS permitidas")
    rate_limit_storage: str = Field(default="redis://localhost:6379/1", description="Storage para rate limiting")
    jwt_secret: str = Field(default="your-secret-key", description="Chave secreta JWT")
    enable_metrics: bool = Field(default=True, description="Habilitar métricas")
    enable_logging: bool = Field(default=True, description="Habilitar logging")


class RateLimiter:
    """Rate limiter usando Redis."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
    
    async def init(self):
        """Inicializa conexão Redis."""
        self.redis_client = await redis.from_url(self.redis_url)
    
    async def is_allowed(self, key: str, limit: int, window: int = 60) -> bool:
        """Verifica se a requisição está dentro do limite."""
        if not self.redis_client:
            return True
        
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # Usar sliding window log
            pipe = self.redis_client.pipeline()
            
            # Remover entradas antigas
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Contar requisições no window
            pipe.zcard(key)
            
            # Adicionar requisição atual
            pipe.zadd(key, {str(current_time): current_time})
            
            # Definir TTL
            pipe.expire(key, window)
            
            results = await pipe.execute()
            
            current_count = results[1]
            
            return current_count < limit
            
        except Exception as e:
            # Em caso de erro, permitir requisição
            return True
    
    async def close(self):
        """Fecha conexão Redis."""
        if self.redis_client:
            await self.redis_client.close()


class APIGateway(BaseService):
    """API Gateway principal."""
    
    def __init__(self, config: GatewayConfig):
        super().__init__("api_gateway", "1.0.0", config.host, config.port)
        
        self.config = config
        self.app = FastAPI(
            title="Valion API Gateway",
            description="Gateway unificado para microserviços do Valion",
            version="1.0.0",
            debug=config.debug
        )
        
        # Componentes
        self.rate_limiter = RateLimiter(config.rate_limit_storage)
        self.security = HTTPBearer(auto_error=False)
        self.routes: Dict[str, RouteConfig] = {}
        self.circuit_breakers: Dict[str, Any] = {}
        
        # Métricas
        self.metrics = get_metrics_collector()
        
        # Configurar middleware
        self._setup_middleware()
        
        # Configurar rotas
        self._setup_routes()
        
        # Logger estruturado
        self.struct_logger = structlog.get_logger("api_gateway")
    
    def _setup_middleware(self):
        """Configura middleware do FastAPI."""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # GZip
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Middleware personalizado
        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Gerar request ID
            request_id = f"req_{int(time.time())}_{id(request)}"
            
            # Log da requisição
            self.struct_logger.info(
                "Request started",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else "unknown"
            )
            
            try:
                response = await call_next(request)
                
                # Calcular tempo de processamento
                process_time = time.time() - start_time
                
                # Métricas
                if self.config.enable_metrics:
                    self.metrics.record_api_request(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=response.status_code,
                        duration=process_time
                    )
                
                # Log da resposta
                self.struct_logger.info(
                    "Request completed",
                    request_id=request_id,
                    status_code=response.status_code,
                    duration=process_time
                )
                
                # Adicionar headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Process-Time"] = str(process_time)
                
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                
                self.struct_logger.error(
                    "Request failed",
                    request_id=request_id,
                    error=str(e),
                    duration=process_time
                )
                
                raise
    
    def _setup_routes(self):
        """Configura rotas do gateway."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check do gateway."""
            health = await super().health_check()
            return health.dict()
        
        @self.app.get("/routes")
        async def list_routes():
            """Lista todas as rotas configuradas."""
            return {"routes": list(self.routes.keys())}
        
        @self.app.get("/services")
        async def list_services():
            """Lista todos os serviços disponíveis."""
            if self.service_registry:
                services = await self.service_registry.get_all_services()
                return {"services": [s.dict() for s in services]}
            return {"services": []}
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Obtém métricas do gateway."""
            return {
                "routes_count": len(self.routes),
                "services_count": len(await self.service_registry.get_all_services()) if self.service_registry else 0,
                "uptime": (datetime.now() - self.service_info.started_at).total_seconds()
            }
        
        # Rota catch-all para proxy
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy_request(
            path: str,
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            return await self._handle_proxy_request(path, request, credentials)
    
    async def _handle_proxy_request(self, path: str, request: Request, 
                                   credentials: Optional[HTTPAuthorizationCredentials]) -> Response:
        """Processa requisição proxy."""
        try:
            # Encontrar rota correspondente
            route_config = self._find_route_config(path, request.method)
            
            if not route_config:
                raise HTTPException(status_code=404, detail="Route not found")
            
            # Verificar autenticação
            if route_config.auth_required and not credentials:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Rate limiting
            client_ip = request.client.host if request.client else "unknown"
            rate_limit_key = f"rate_limit:{client_ip}:{path}"
            
            if not await self.rate_limiter.is_allowed(rate_limit_key, route_config.rate_limit):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Verificar circuit breaker
            if await self._is_circuit_breaker_open(route_config.service_name):
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            
            # Verificar cache
            if request.method == "GET" and route_config.cache_ttl > 0:
                cached_response = await self._get_cached_response(path, request)
                if cached_response:
                    return cached_response
            
            # Preparar dados da requisição
            try:
                body = await request.json() if request.headers.get("content-type") == "application/json" else {}
            except:
                body = {}
            
            # Criar requisição para serviço
            service_request = ServiceRequest(
                service_name=route_config.service_name,
                method=request.method,
                endpoint=f"/{path}",
                payload=body,
                headers=dict(request.headers),
                timeout=route_config.timeout
            )
            
            # Fazer chamada para serviço
            service_response = await self.call_service(service_request)
            
            # Processar resposta
            if service_response.success:
                # Cachear resposta se necessário
                if request.method == "GET" and route_config.cache_ttl > 0:
                    await self._cache_response(path, request, service_response, route_config.cache_ttl)
                
                # Resetar circuit breaker em caso de sucesso
                await self._reset_circuit_breaker(route_config.service_name)
                
                return JSONResponse(
                    content=service_response.data,
                    status_code=service_response.status_code,
                    headers=service_response.headers or {}
                )
            else:
                # Registrar falha no circuit breaker
                await self._record_circuit_breaker_failure(route_config.service_name)
                
                raise HTTPException(
                    status_code=service_response.status_code,
                    detail=service_response.error
                )
                
        except HTTPException:
            raise
        except Exception as e:
            self.struct_logger.error(f"Proxy error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    def _find_route_config(self, path: str, method: str) -> Optional[RouteConfig]:
        """Encontra configuração de rota para o caminho."""
        # Busca exata primeiro
        route_key = f"{method}:{path}"
        if route_key in self.routes:
            return self.routes[route_key]
        
        # Busca por prefixo
        for route_path, config in self.routes.items():
            if path.startswith(config.path.rstrip("/")):
                return config
        
        # Rota padrão baseada no primeiro segmento
        path_parts = path.split("/")
        if len(path_parts) > 1:
            service_name = path_parts[1]
            
            # Criar configuração dinâmica
            return RouteConfig(
                path=f"/{service_name}",
                service_name=service_name,
                method=method,
                auth_required=True,
                rate_limit=100,
                timeout=30
            )
        
        return None
    
    async def _is_circuit_breaker_open(self, service_name: str) -> bool:
        """Verifica se o circuit breaker está aberto."""
        breaker = self.circuit_breakers.get(service_name)
        if not breaker:
            return False
        
        # Lógica simples de circuit breaker
        failure_count = breaker.get("failure_count", 0)
        last_failure = breaker.get("last_failure", 0)
        
        # Abrir se muitas falhas recentes
        if failure_count >= 5 and (time.time() - last_failure) < 60:
            return True
        
        # Resetar se passou tempo suficiente
        if (time.time() - last_failure) > 300:  # 5 minutos
            breaker["failure_count"] = 0
        
        return False
    
    async def _record_circuit_breaker_failure(self, service_name: str):
        """Registra falha no circuit breaker."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {}
        
        breaker = self.circuit_breakers[service_name]
        breaker["failure_count"] = breaker.get("failure_count", 0) + 1
        breaker["last_failure"] = time.time()
    
    async def _reset_circuit_breaker(self, service_name: str):
        """Reseta circuit breaker após sucesso."""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name]["failure_count"] = 0
    
    async def _get_cached_response(self, path: str, request: Request) -> Optional[Response]:
        """Obtém resposta do cache."""
        # Implementar cache usando Redis
        # Por simplicidade, retornando None aqui
        return None
    
    async def _cache_response(self, path: str, request: Request, 
                             response: ServiceResponse, ttl: int):
        """Cacheia resposta."""
        # Implementar cache usando Redis
        # Por simplicidade, fazendo nada aqui
        pass
    
    def add_route(self, route_config: RouteConfig):
        """Adiciona nova rota ao gateway."""
        route_key = f"{route_config.method}:{route_config.path}"
        self.routes[route_key] = route_config
        
        self.struct_logger.info(
            "Route added",
            path=route_config.path,
            service=route_config.service_name,
            method=route_config.method
        )
    
    def remove_route(self, path: str, method: str = "GET"):
        """Remove rota do gateway."""
        route_key = f"{method}:{path}"
        if route_key in self.routes:
            del self.routes[route_key]
            
            self.struct_logger.info(
                "Route removed",
                path=path,
                method=method
            )
    
    async def initialize(self):
        """Inicialização do gateway."""
        # Inicializar rate limiter
        await self.rate_limiter.init()
        
        # Configurar rotas padrão
        self._setup_default_routes()
        
        self.struct_logger.info("API Gateway initialized")
    
    async def cleanup(self):
        """Limpeza do gateway."""
        await self.rate_limiter.close()
        
        self.struct_logger.info("API Gateway cleaned up")
    
    def _setup_default_routes(self):
        """Configura rotas padrão."""
        default_routes = [
            RouteConfig(
                path="/data",
                service_name="data_processing_service",
                method="POST",
                auth_required=True,
                rate_limit=50,
                timeout=60
            ),
            RouteConfig(
                path="/ml",
                service_name="ml_service",
                method="POST",
                auth_required=True,
                rate_limit=20,
                timeout=300
            ),
            RouteConfig(
                path="/geospatial",
                service_name="geospatial_service",
                method="GET",
                auth_required=True,
                rate_limit=100,
                timeout=30,
                cache_ttl=3600
            ),
            RouteConfig(
                path="/reports",
                service_name="reporting_service",
                method="GET",
                auth_required=True,
                rate_limit=30,
                timeout=60
            )
        ]
        
        for route in default_routes:
            self.add_route(route)
    
    async def run(self):
        """Executa o gateway."""
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info" if self.config.debug else "warning"
        )
        
        server = uvicorn.Server(config)
        await server.serve()


def create_api_gateway(config: GatewayConfig) -> APIGateway:
    """Cria instância do API Gateway."""
    return APIGateway(config)


def get_default_gateway_config() -> GatewayConfig:
    """Obtém configuração padrão do gateway."""
    return GatewayConfig(
        host="0.0.0.0",
        port=8000,
        debug=False,
        cors_origins=["*"],
        rate_limit_storage="redis://localhost:6379/1",
        jwt_secret="your-secret-key-change-in-production",
        enable_metrics=True,
        enable_logging=True
    )