# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Model Deployer para MLOps
Sistema de deployment e gerenciamento de modelos em produção.
"""

import asyncio
import json
import shutil
import subprocess
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
from contextlib import asynccontextmanager
import aiohttp
import aiofiles
import joblib
import pandas as pd
import numpy as np
from ..monitoring.logging_config import get_logger
import structlog


class DeploymentStatus(Enum):
    """Status do deployment."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


class DeploymentStrategy(Enum):
    """Estratégias de deployment."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    REPLACE = "replace"


@dataclass
class DeploymentConfig:
    """Configuração de deployment."""
    model_id: str
    version: str
    environment: str
    strategy: DeploymentStrategy
    target_port: int
    health_check_path: str = "/health"
    health_check_timeout: int = 30
    max_replicas: int = 3
    min_replicas: int = 1
    cpu_limit: str = "500m"
    memory_limit: str = "512Mi"
    auto_rollback: bool = True
    canary_percentage: int = 10
    environment_variables: Dict[str, str] = None
    
    def __post_init__(self):
        if self.environment_variables is None:
            self.environment_variables = {}


@dataclass
class DeploymentInfo:
    """Informações do deployment."""
    deployment_id: str
    model_id: str
    version: str
    environment: str
    strategy: DeploymentStrategy
    status: DeploymentStatus
    endpoint: str
    replicas: int
    created_at: datetime
    updated_at: datetime
    health_status: Dict[str, Any]
    metrics: Dict[str, float]
    logs: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            **asdict(self),
            'strategy': self.strategy.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class ModelDeployer:
    """Sistema de deployment de modelos."""
    
    def __init__(self, registry, deployment_dir: str = "deployments"):
        self.registry = registry
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Diretórios
        self.configs_dir = self.deployment_dir / "configs"
        self.runtime_dir = self.deployment_dir / "runtime"
        self.logs_dir = self.deployment_dir / "logs"
        
        for dir_path in [self.configs_dir, self.runtime_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logger = get_logger("model_deployer")
        self.struct_logger = structlog.get_logger("model_deployer")
        
        # Deployments ativos
        self.active_deployments: Dict[str, DeploymentInfo] = {}
        self.deployment_processes: Dict[str, subprocess.Popen] = {}
        
        # Métricas
        self.deployment_metrics = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "rollbacks": 0,
            "average_deployment_time": 0.0
        }
    
    async def deploy_model(self, config: DeploymentConfig) -> str:
        """Deploya modelo usando estratégia especificada."""
        start_time = datetime.now()
        deployment_id = str(uuid.uuid4())
        
        try:
            # Validar modelo
            model_version = self.registry.get_model_version(config.model_id, config.version)
            if not model_version:
                raise ValueError(f"Model version {config.model_id}:{config.version} not found")
            
            # Criar deployment info
            deployment_info = DeploymentInfo(
                deployment_id=deployment_id,
                model_id=config.model_id,
                version=config.version,
                environment=config.environment,
                strategy=config.strategy,
                status=DeploymentStatus.PENDING,
                endpoint=f"http://localhost:{config.target_port}",
                replicas=config.min_replicas,
                created_at=start_time,
                updated_at=start_time,
                health_status={"status": "unknown"},
                metrics={},
                logs=[]
            )
            
            self.active_deployments[deployment_id] = deployment_info
            
            # Executar deployment baseado na estratégia
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._deploy_blue_green(config, deployment_info)
            elif config.strategy == DeploymentStrategy.CANARY:
                success = await self._deploy_canary(config, deployment_info)
            elif config.strategy == DeploymentStrategy.ROLLING:
                success = await self._deploy_rolling(config, deployment_info)
            else:  # REPLACE
                success = await self._deploy_replace(config, deployment_info)
            
            # Atualizar status
            if success:
                deployment_info.status = DeploymentStatus.RUNNING
                deployment_info.updated_at = datetime.now()
                self.deployment_metrics["successful_deployments"] += 1
            else:
                deployment_info.status = DeploymentStatus.FAILED
                self.deployment_metrics["failed_deployments"] += 1
                
                if config.auto_rollback:
                    await self._rollback_deployment(deployment_id)
            
            # Calcular tempo de deployment
            deployment_time = (datetime.now() - start_time).total_seconds()
            self._update_average_deployment_time(deployment_time)
            
            self.struct_logger.info(
                "Model deployment completed",
                deployment_id=deployment_id,
                model_id=config.model_id,
                version=config.version,
                strategy=config.strategy.value,
                success=success,
                deployment_time=deployment_time
            )
            
            return deployment_id
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            if deployment_id in self.active_deployments:
                self.active_deployments[deployment_id].status = DeploymentStatus.FAILED
            raise
    
    async def _deploy_blue_green(self, config: DeploymentConfig, deployment_info: DeploymentInfo) -> bool:
        """Deployment Blue-Green."""
        try:
            # Criar ambiente Green
            green_port = config.target_port + 1000
            green_config = DeploymentConfig(
                model_id=config.model_id,
                version=config.version,
                environment=f"{config.environment}_green",
                strategy=config.strategy,
                target_port=green_port,
                health_check_path=config.health_check_path,
                environment_variables=config.environment_variables
            )
            
            # Iniciar serviço Green
            deployment_info.status = DeploymentStatus.DEPLOYING
            success = await self._start_model_service(green_config, deployment_info)
            
            if not success:
                return False
            
            # Aguardar health check
            if not await self._wait_for_health_check(f"http://localhost:{green_port}", config.health_check_path):
                return False
            
            # Fazer switch (simular load balancer)
            await self._switch_traffic(config.target_port, green_port)
            
            # Parar ambiente Blue antigo
            await self._stop_old_deployment(config.environment)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Blue-Green deployment failed: {e}")
            return False
    
    async def _deploy_canary(self, config: DeploymentConfig, deployment_info: DeploymentInfo) -> bool:
        """Deployment Canary."""
        try:
            # Iniciar versão canary
            canary_port = config.target_port + 2000
            canary_config = DeploymentConfig(
                model_id=config.model_id,
                version=config.version,
                environment=f"{config.environment}_canary",
                strategy=config.strategy,
                target_port=canary_port,
                environment_variables=config.environment_variables
            )
            
            deployment_info.status = DeploymentStatus.DEPLOYING
            success = await self._start_model_service(canary_config, deployment_info)
            
            if not success:
                return False
            
            # Aguardar health check
            if not await self._wait_for_health_check(f"http://localhost:{canary_port}", config.health_check_path):
                return False
            
            # Configurar roteamento canary (simular)
            await self._configure_canary_routing(
                config.target_port, 
                canary_port, 
                config.canary_percentage
            )
            
            # Monitorar métricas por período
            await self._monitor_canary_metrics(config, deployment_info)
            
            # Se tudo OK, promover para produção
            await self._promote_canary(config.target_port, canary_port)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Canary deployment failed: {e}")
            return False
    
    async def _deploy_rolling(self, config: DeploymentConfig, deployment_info: DeploymentInfo) -> bool:
        """Deployment Rolling."""
        try:
            deployment_info.status = DeploymentStatus.DEPLOYING
            
            # Atualizar instâncias gradualmente
            for i in range(config.max_replicas):
                instance_port = config.target_port + i
                
                # Parar instância antiga
                await self._stop_instance(f"{config.environment}_instance_{i}")
                
                # Iniciar nova instância
                instance_config = DeploymentConfig(
                    model_id=config.model_id,
                    version=config.version,
                    environment=f"{config.environment}_instance_{i}",
                    strategy=config.strategy,
                    target_port=instance_port,
                    environment_variables=config.environment_variables
                )
                
                success = await self._start_model_service(instance_config, deployment_info)
                if not success:
                    return False
                
                # Aguardar health check
                if not await self._wait_for_health_check(f"http://localhost:{instance_port}", config.health_check_path):
                    return False
                
                # Aguardar antes da próxima instância
                await asyncio.sleep(5)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rolling deployment failed: {e}")
            return False
    
    async def _deploy_replace(self, config: DeploymentConfig, deployment_info: DeploymentInfo) -> bool:
        """Deployment Replace (simples)."""
        try:
            deployment_info.status = DeploymentStatus.DEPLOYING
            
            # Parar deployment anterior
            await self._stop_old_deployment(config.environment)
            
            # Iniciar novo deployment
            success = await self._start_model_service(config, deployment_info)
            if not success:
                return False
            
            # Aguardar health check
            if not await self._wait_for_health_check(f"http://localhost:{config.target_port}", config.health_check_path):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Replace deployment failed: {e}")
            return False
    
    async def _start_model_service(self, config: DeploymentConfig, deployment_info: DeploymentInfo) -> bool:
        """Inicia serviço do modelo."""
        try:
            # Carregar modelo
            model_version = self.registry.get_model_version(config.model_id, config.version)
            if not model_version:
                return False
            
            # Criar script de inicialização
            service_script = self._create_service_script(config, model_version)
            
            # Executar serviço
            process = await asyncio.create_subprocess_exec(
                "python", str(service_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.runtime_dir
            )
            
            self.deployment_processes[deployment_info.deployment_id] = process
            
            # Aguardar inicialização
            await asyncio.sleep(3)
            
            return process.returncode is None
            
        except Exception as e:
            self.logger.error(f"Failed to start model service: {e}")
            return False
    
    def _create_service_script(self, config: DeploymentConfig, model_version) -> Path:
        """Cria script de inicialização do serviço."""
        script_path = self.runtime_dir / f"service_{config.environment}.py"
        
        service_code = f'''
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import structlog

# Configurar logging
logger = structlog.get_logger("model_service")

# Carregar modelo
model = joblib.load(r"{model_version.artifacts.get('model')}")

# Criar app FastAPI
app = FastAPI(title="Model Service", version="{model_version.version}")

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str

@app.get("/health")
async def health_check():
    return {{
        "status": "healthy",
        "model_id": "{config.model_id}",
        "version": "{config.version}",
        "environment": "{config.environment}"
    }}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Converter para DataFrame
        df = pd.DataFrame([request.features])
        
        # Fazer predição
        prediction = model.predict(df)[0]
        
        # Calcular confiança (simplificado)
        confidence = 0.95  # Implementar cálculo real
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=confidence,
            model_version="{config.version}"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def model_info():
    return {{
        "model_id": "{config.model_id}",
        "version": "{config.version}",
        "environment": "{config.environment}",
        "algorithm": "{model_version.metadata.algorithm}",
        "features": {model_version.metadata.features},
        "performance": {model_version.performance_metrics}
    }}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port={config.target_port},
        log_level="info"
    )
'''
        
        with open(script_path, 'w') as f:
            f.write(service_code)
        
        return script_path
    
    async def _wait_for_health_check(self, endpoint: str, health_path: str, timeout: int = 30) -> bool:
        """Aguarda health check do serviço."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{endpoint}{health_path}") as response:
                        if response.status == 200:
                            return True
            except:
                pass
            
            await asyncio.sleep(2)
        
        return False
    
    async def _switch_traffic(self, blue_port: int, green_port: int):
        """Simula switch de tráfego."""
        self.logger.info(f"Switching traffic from port {blue_port} to {green_port}")
        # Implementar lógica de load balancer
        pass
    
    async def _configure_canary_routing(self, production_port: int, canary_port: int, percentage: int):
        """Configura roteamento canary."""
        self.logger.info(f"Routing {percentage}% traffic to canary on port {canary_port}")
        # Implementar lógica de roteamento
        pass
    
    async def _monitor_canary_metrics(self, config: DeploymentConfig, deployment_info: DeploymentInfo):
        """Monitora métricas do canary."""
        self.logger.info("Monitoring canary metrics...")
        # Implementar monitoramento por período
        await asyncio.sleep(30)  # Simular monitoramento
    
    async def _promote_canary(self, production_port: int, canary_port: int):
        """Promove canary para produção."""
        self.logger.info(f"Promoting canary from port {canary_port} to production")
        # Implementar promoção
        pass
    
    async def _stop_old_deployment(self, environment: str):
        """Para deployment antigo."""
        self.logger.info(f"Stopping old deployment for environment {environment}")
        # Implementar lógica de parada
        pass
    
    async def _stop_instance(self, instance_name: str):
        """Para instância específica."""
        self.logger.info(f"Stopping instance {instance_name}")
        # Implementar lógica de parada
        pass
    
    async def _rollback_deployment(self, deployment_id: str):
        """Executa rollback do deployment."""
        try:
            deployment_info = self.active_deployments.get(deployment_id)
            if not deployment_info:
                return
            
            deployment_info.status = DeploymentStatus.ROLLING_BACK
            
            # Parar deployment atual
            if deployment_id in self.deployment_processes:
                process = self.deployment_processes[deployment_id]
                process.terminate()
                await asyncio.sleep(2)
                if process.poll() is None:
                    process.kill()
                del self.deployment_processes[deployment_id]
            
            # Restaurar versão anterior (implementar lógica)
            self.logger.info(f"Rolling back deployment {deployment_id}")
            
            deployment_info.status = DeploymentStatus.STOPPED
            self.deployment_metrics["rollbacks"] += 1
            
            self.struct_logger.info(
                "Deployment rolled back",
                deployment_id=deployment_id,
                model_id=deployment_info.model_id,
                version=deployment_info.version
            )
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
    
    async def stop_deployment(self, deployment_id: str) -> bool:
        """Para deployment específico."""
        try:
            deployment_info = self.active_deployments.get(deployment_id)
            if not deployment_info:
                return False
            
            # Parar processo
            if deployment_id in self.deployment_processes:
                process = self.deployment_processes[deployment_id]
                process.terminate()
                await asyncio.sleep(2)
                if process.poll() is None:
                    process.kill()
                del self.deployment_processes[deployment_id]
            
            deployment_info.status = DeploymentStatus.STOPPED
            deployment_info.updated_at = datetime.now()
            
            self.struct_logger.info(
                "Deployment stopped",
                deployment_id=deployment_id,
                model_id=deployment_info.model_id,
                version=deployment_info.version
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop deployment: {e}")
            return False
    
    def get_deployment_info(self, deployment_id: str) -> Optional[DeploymentInfo]:
        """Obtém informações do deployment."""
        return self.active_deployments.get(deployment_id)
    
    def list_deployments(self, environment: str = None) -> List[DeploymentInfo]:
        """Lista deployments ativos."""
        deployments = list(self.active_deployments.values())
        
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
        
        return sorted(deployments, key=lambda d: d.created_at, reverse=True)
    
    def get_deployment_metrics(self) -> Dict[str, Any]:
        """Obtém métricas dos deployments."""
        active_count = len([d for d in self.active_deployments.values() if d.status == DeploymentStatus.RUNNING])
        
        return {
            **self.deployment_metrics,
            "active_deployments": active_count,
            "total_active": len(self.active_deployments)
        }
    
    def _update_average_deployment_time(self, deployment_time: float):
        """Atualiza tempo médio de deployment."""
        current_avg = self.deployment_metrics["average_deployment_time"]
        total_deployments = self.deployment_metrics["total_deployments"]
        
        if total_deployments == 0:
            new_avg = deployment_time
        else:
            new_avg = (current_avg * total_deployments + deployment_time) / (total_deployments + 1)
        
        self.deployment_metrics["average_deployment_time"] = new_avg
        self.deployment_metrics["total_deployments"] += 1
    
    async def cleanup_old_deployments(self, max_age_days: int = 7):
        """Limpa deployments antigos."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        to_remove = [
            deployment_id for deployment_id, info in self.active_deployments.items()
            if info.created_at < cutoff_date and info.status in [DeploymentStatus.STOPPED, DeploymentStatus.FAILED]
        ]
        
        for deployment_id in to_remove:
            del self.active_deployments[deployment_id]
            
            # Limpar arquivos
            try:
                service_script = self.runtime_dir / f"service_{deployment_id}.py"
                if service_script.exists():
                    service_script.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to clean up files for {deployment_id}: {e}")
        
        self.logger.info(f"Cleaned up {len(to_remove)} old deployments")


def create_model_deployer(registry, deployment_dir: str = "deployments") -> ModelDeployer:
    """Cria instância do model deployer."""
    return ModelDeployer(registry, deployment_dir)