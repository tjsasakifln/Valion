# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Pipeline Orchestrator para MLOps
Sistema de orquestração de pipelines de ML (treinamento, validação, deployment).
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .model_registry import ModelRegistry, ModelStage, ModelStatus
from .model_deployer import ModelDeployer, DeploymentConfig, DeploymentStrategy
from .model_validator import ModelValidator, ValidationStatus
from ..monitoring.logging_config import get_logger
from ..monitoring.metrics import MetricsCollector
import structlog

# Adicionar métrica de pipeline se não existir
try:
    from prometheus_client import Counter, Histogram
    pipeline_executions_total = Counter('valion_pipeline_executions_total', 'Total pipeline executions', ['pipeline', 'status'])
    pipeline_duration_seconds = Histogram('valion_pipeline_duration_seconds', 'Pipeline execution duration', ['pipeline'])
except ImportError:
    pipeline_executions_total = None
    pipeline_duration_seconds = None


class PipelineStatus(Enum):
    """Status do pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class PipelineStage(Enum):
    """Estágios do pipeline."""
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_DEPLOYMENT = "model_deployment"
    MONITORING = "monitoring"


@dataclass
class PipelineStep:
    """Passo do pipeline."""
    name: str
    stage: PipelineStage
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 3600  # 1 hora
    retry_count: int = 3
    retry_delay: int = 60
    skip_on_failure: bool = False
    required: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecution:
    """Execução do pipeline."""
    execution_id: str
    pipeline_name: str
    status: PipelineStatus
    current_stage: Optional[PipelineStage]
    steps_completed: List[str]
    steps_failed: List[str]
    steps_skipped: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    artifacts: Dict[str, Any]
    metrics: Dict[str, float]
    logs: List[str]
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            **asdict(self),
            'status': self.status.value,
            'current_stage': self.current_stage.value if self.current_stage else None,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class PipelineConfig:
    """Configuração do pipeline."""
    name: str
    description: str
    steps: List[PipelineStep]
    schedule: Optional[str] = None  # Cron expression
    max_concurrent_executions: int = 1
    timeout: int = 14400  # 4 horas
    auto_retry: bool = True
    notification_config: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)


class PipelineOrchestrator:
    """Orquestrador de pipelines MLOps."""
    
    def __init__(self, registry: ModelRegistry, deployer: ModelDeployer, 
                 validator: ModelValidator):
        self.registry = registry
        self.deployer = deployer
        self.validator = validator
        
        # Logging
        self.logger = get_logger("pipeline_orchestrator")
        self.struct_logger = structlog.get_logger("pipeline_orchestrator")
        
        # Métricas
        self.metrics_collector = MetricsCollector()
        
        # Pipelines e execuções
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Thread pool para execução de steps
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Scheduler
        self.scheduler_task: Optional[asyncio.Task] = None
        self.scheduler_running = False
        
        # Pipeline padrão
        self._create_default_pipeline()
    
    def _create_default_pipeline(self):
        """Cria pipeline padrão de ML."""
        steps = [
            PipelineStep(
                name="data_validation",
                stage=PipelineStage.DATA_PREPARATION,
                function=self._validate_data,
                dependencies=[],
                parameters={"quality_threshold": 0.9}
            ),
            PipelineStep(
                name="feature_engineering",
                stage=PipelineStage.FEATURE_ENGINEERING,
                function=self._engineer_features,
                dependencies=["data_validation"],
                parameters={"enable_scaling": True}
            ),
            PipelineStep(
                name="model_training",
                stage=PipelineStage.MODEL_TRAINING,
                function=self._train_model,
                dependencies=["feature_engineering"],
                parameters={"algorithms": ["elastic_net", "xgboost"]}
            ),
            PipelineStep(
                name="model_validation",
                stage=PipelineStage.MODEL_VALIDATION,
                function=self._validate_model,
                dependencies=["model_training"],
                parameters={"validation_split": 0.2}
            ),
            PipelineStep(
                name="model_deployment",
                stage=PipelineStage.MODEL_DEPLOYMENT,
                function=self._deploy_model,
                dependencies=["model_validation"],
                parameters={"strategy": "blue_green", "environment": "staging"}
            ),
            PipelineStep(
                name="monitoring_setup",
                stage=PipelineStage.MONITORING,
                function=self._setup_monitoring,
                dependencies=["model_deployment"],
                parameters={"alert_threshold": 0.1}
            )
        ]
        
        default_pipeline = PipelineConfig(
            name="default_ml_pipeline",
            description="Pipeline padrão de ML (treinamento, validação, deployment)",
            steps=steps,
            schedule="0 2 * * *",  # Todo dia às 2h
            max_concurrent_executions=1,
            timeout=14400,
            auto_retry=True
        )
        
        self.pipelines["default"] = default_pipeline
    
    def register_pipeline(self, config: PipelineConfig):
        """Registra pipeline."""
        self.pipelines[config.name] = config
        self.struct_logger.info(
            "Pipeline registered",
            pipeline_name=config.name,
            steps_count=len(config.steps),
            schedule=config.schedule
        )
    
    async def execute_pipeline(self, pipeline_name: str, 
                             parameters: Dict[str, Any] = None,
                             force: bool = False) -> str:
        """Executa pipeline."""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        config = self.pipelines[pipeline_name]
        
        # Verificar execuções concorrentes
        active_count = len([
            exec for exec in self.executions.values()
            if exec.pipeline_name == pipeline_name and 
            exec.status == PipelineStatus.RUNNING
        ])
        
        if active_count >= config.max_concurrent_executions and not force:
            raise RuntimeError(f"Maximum concurrent executions ({config.max_concurrent_executions}) reached")
        
        # Criar execução
        execution_id = str(uuid.uuid4())
        execution = PipelineExecution(
            execution_id=execution_id,
            pipeline_name=pipeline_name,
            status=PipelineStatus.PENDING,
            current_stage=None,
            steps_completed=[],
            steps_failed=[],
            steps_skipped=[],
            start_time=datetime.now(),
            end_time=None,
            artifacts={},
            metrics={},
            logs=[],
            error_message=None
        )
        
        self.executions[execution_id] = execution
        
        # Executar pipeline assincronamente
        task = asyncio.create_task(self._run_pipeline(execution_id, parameters or {}))
        self.active_executions[execution_id] = task
        
        self.struct_logger.info(
            "Pipeline execution started",
            execution_id=execution_id,
            pipeline_name=pipeline_name
        )
        
        return execution_id
    
    async def _run_pipeline(self, execution_id: str, parameters: Dict[str, Any]):
        """Executa pipeline completo."""
        execution = self.executions[execution_id]
        config = self.pipelines[execution.pipeline_name]
        
        try:
            execution.status = PipelineStatus.RUNNING
            
            # Resolver dependências e ordenar steps
            ordered_steps = self._resolve_dependencies(config.steps)
            
            # Executar steps em ordem
            for step in ordered_steps:
                if execution.status == PipelineStatus.CANCELLED:
                    break
                
                execution.current_stage = step.stage
                
                # Executar step
                success = await self._execute_step(execution, step, parameters)
                
                if success:
                    execution.steps_completed.append(step.name)
                    execution.logs.append(f"Step {step.name} completed successfully")
                else:
                    execution.steps_failed.append(step.name)
                    execution.logs.append(f"Step {step.name} failed")
                    
                    if not step.skip_on_failure and step.required:
                        execution.status = PipelineStatus.FAILED
                        execution.error_message = f"Required step {step.name} failed"
                        break
                    else:
                        execution.steps_skipped.append(step.name)
            
            # Finalizar execução
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.COMPLETED
            
            execution.end_time = datetime.now()
            
            # Atualizar métricas
            self._update_pipeline_metrics(execution)
            
            self.struct_logger.info(
                "Pipeline execution completed",
                execution_id=execution_id,
                pipeline_name=execution.pipeline_name,
                status=execution.status.value,
                steps_completed=len(execution.steps_completed),
                steps_failed=len(execution.steps_failed),
                duration=(execution.end_time - execution.start_time).total_seconds()
            )
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.now()
            
            self.logger.error(f"Pipeline execution failed: {e}")
            
        finally:
            # Remover da lista de execuções ativas
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    def _resolve_dependencies(self, steps: List[PipelineStep]) -> List[PipelineStep]:
        """Resolve dependências e ordena steps."""
        ordered = []
        remaining = steps.copy()
        
        while remaining:
            # Encontrar steps sem dependências não satisfeitas
            ready_steps = [
                step for step in remaining
                if all(dep in [s.name for s in ordered] for dep in step.dependencies)
            ]
            
            if not ready_steps:
                # Dependência circular ou dependência não encontrada
                raise ValueError("Circular dependency or missing dependency detected")
            
            # Adicionar steps prontos
            for step in ready_steps:
                ordered.append(step)
                remaining.remove(step)
        
        return ordered
    
    async def _execute_step(self, execution: PipelineExecution, 
                          step: PipelineStep, parameters: Dict[str, Any]) -> bool:
        """Executa step individual."""
        start_time = datetime.now()
        
        try:
            execution.logs.append(f"Starting step {step.name}")
            
            # Preparar parâmetros
            step_params = {**step.parameters, **parameters}
            step_params['execution_id'] = execution.execution_id
            step_params['artifacts'] = execution.artifacts
            
            # Executar step com retry
            success = False
            for attempt in range(step.retry_count + 1):
                try:
                    # Executar função do step
                    if asyncio.iscoroutinefunction(step.function):
                        result = await step.function(**step_params)
                    else:
                        # Executar em thread pool
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.executor, step.function, **step_params
                        )
                    
                    # Processar resultado
                    if isinstance(result, dict):
                        execution.artifacts.update(result)
                    
                    success = True
                    break
                    
                except Exception as e:
                    execution.logs.append(f"Step {step.name} attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt < step.retry_count:
                        await asyncio.sleep(step.retry_delay)
                    else:
                        execution.logs.append(f"Step {step.name} failed after {step.retry_count + 1} attempts")
                        break
            
            # Calcular tempo de execução
            execution_time = (datetime.now() - start_time).total_seconds()
            execution.metrics[f"step_{step.name}_duration"] = execution_time
            
            return success
            
        except Exception as e:
            execution.logs.append(f"Step {step.name} execution error: {str(e)}")
            return False
    
    async def _validate_data(self, **kwargs) -> Dict[str, Any]:
        """Step de validação de dados."""
        # Implementar validação de dados
        return {"data_validation": "completed"}
    
    async def _engineer_features(self, **kwargs) -> Dict[str, Any]:
        """Step de engenharia de features."""
        # Implementar engenharia de features
        return {"feature_engineering": "completed"}
    
    async def _train_model(self, **kwargs) -> Dict[str, Any]:
        """Step de treinamento de modelo."""
        # Implementar treinamento de modelo
        return {"model_training": "completed"}
    
    async def _validate_model(self, **kwargs) -> Dict[str, Any]:
        """Step de validação de modelo."""
        # Implementar validação de modelo
        return {"model_validation": "completed"}
    
    async def _deploy_model(self, **kwargs) -> Dict[str, Any]:
        """Step de deployment de modelo."""
        # Implementar deployment de modelo
        return {"model_deployment": "completed"}
    
    async def _setup_monitoring(self, **kwargs) -> Dict[str, Any]:
        """Step de configuração de monitoramento."""
        # Implementar setup de monitoramento
        return {"monitoring_setup": "completed"}
    
    def _update_pipeline_metrics(self, execution: PipelineExecution):
        """Atualiza métricas do pipeline."""
        try:
            # Atualizar métricas básicas
            if execution.status == PipelineStatus.COMPLETED and pipeline_executions_total:
                pipeline_executions_total.labels(
                    pipeline=execution.pipeline_name,
                    status="completed"
                ).inc()
            elif execution.status == PipelineStatus.FAILED and pipeline_executions_total:
                pipeline_executions_total.labels(
                    pipeline=execution.pipeline_name,
                    status="failed"
                ).inc()
            
            # Atualizar duração
            if execution.end_time and pipeline_duration_seconds:
                duration = (execution.end_time - execution.start_time).total_seconds()
                pipeline_duration_seconds.labels(
                    pipeline=execution.pipeline_name
                ).observe(duration)
            
        except Exception as e:
            self.logger.error(f"Error updating pipeline metrics: {e}")
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancela execução de pipeline."""
        if execution_id not in self.executions:
            return False
        
        execution = self.executions[execution_id]
        execution.status = PipelineStatus.CANCELLED
        
        # Cancelar task se ainda estiver ativa
        if execution_id in self.active_executions:
            task = self.active_executions[execution_id]
            task.cancel()
        
        self.struct_logger.info(
            "Pipeline execution cancelled",
            execution_id=execution_id,
            pipeline_name=execution.pipeline_name
        )
        
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[PipelineExecution]:
        """Obtém status da execução."""
        return self.executions.get(execution_id)
    
    def list_executions(self, pipeline_name: str = None, 
                       status: PipelineStatus = None) -> List[PipelineExecution]:
        """Lista execuções."""
        executions = list(self.executions.values())
        
        if pipeline_name:
            executions = [e for e in executions if e.pipeline_name == pipeline_name]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return sorted(executions, key=lambda e: e.start_time, reverse=True)
    
    def get_pipeline_metrics(self, pipeline_name: str = None) -> Dict[str, Any]:
        """Obtém métricas dos pipelines."""
        executions = self.list_executions(pipeline_name)
        
        if not executions:
            return {}
        
        completed = [e for e in executions if e.status == PipelineStatus.COMPLETED]
        failed = [e for e in executions if e.status == PipelineStatus.FAILED]
        
        durations = [
            (e.end_time - e.start_time).total_seconds()
            for e in executions if e.end_time
        ]
        
        return {
            "total_executions": len(executions),
            "completed_executions": len(completed),
            "failed_executions": len(failed),
            "success_rate": len(completed) / len(executions) if executions else 0,
            "average_duration": np.mean(durations) if durations else 0,
            "active_executions": len(self.active_executions)
        }
    
    async def start_scheduler(self):
        """Inicia scheduler de pipelines."""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        self.logger.info("Pipeline scheduler started")
    
    async def stop_scheduler(self):
        """Para scheduler de pipelines."""
        if not self.scheduler_running:
            return
        
        self.scheduler_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Pipeline scheduler stopped")
    
    async def _scheduler_loop(self):
        """Loop do scheduler."""
        while self.scheduler_running:
            try:
                # Verificar pipelines agendados
                for pipeline_name, config in self.pipelines.items():
                    if config.schedule:
                        # Implementar lógica de cron
                        # Por enquanto, apenas log
                        self.logger.debug(f"Checking schedule for pipeline {pipeline_name}")
                
                # Aguardar próxima verificação
                await asyncio.sleep(60)  # Verificar a cada minuto
                
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def cleanup_old_executions(self, max_age_days: int = 30):
        """Limpa execuções antigas."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        to_remove = [
            execution_id for execution_id, execution in self.executions.items()
            if execution.start_time < cutoff_date and 
            execution.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]
        ]
        
        for execution_id in to_remove:
            del self.executions[execution_id]
        
        self.logger.info(f"Cleaned up {len(to_remove)} old pipeline executions")
    
    def __del__(self):
        """Cleanup do orquestrador."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


def create_pipeline_orchestrator(registry: ModelRegistry, 
                                deployer: ModelDeployer,
                                validator: ModelValidator) -> PipelineOrchestrator:
    """Cria instância do pipeline orchestrator."""
    return PipelineOrchestrator(registry, deployer, validator)