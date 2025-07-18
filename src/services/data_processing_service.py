# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Serviço de Processamento de Dados
Microserviço responsável por carregar, validar e transformar dados.
"""

import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import os

from .base_service import BaseService
from ..core.data_loader import DataLoader
from ..core.transformations import VariableTransformer
from ..monitoring.data_drift import DataQualityMonitor, get_default_drift_config
from ..monitoring.metrics import get_metrics_collector, measure_time
from ..monitoring.logging_config import get_logger
import structlog


# Modelos Pydantic
class DataProcessingRequest(BaseModel):
    """Requisição de processamento de dados."""
    evaluation_id: str = Field(..., description="ID da avaliação")
    target_column: str = Field(default="valor", description="Coluna target")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configurações específicas")
    perform_validation: bool = Field(default=True, description="Realizar validação")
    perform_transformation: bool = Field(default=True, description="Realizar transformação")
    detect_drift: bool = Field(default=True, description="Detectar drift")


class DataProcessingResponse(BaseModel):
    """Resposta do processamento de dados."""
    evaluation_id: str
    status: str
    message: str
    data_summary: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]] = None
    transformation_results: Optional[Dict[str, Any]] = None
    drift_results: Optional[Dict[str, Any]] = None
    processing_time: float
    timestamp: datetime


class DataValidationResult(BaseModel):
    """Resultado da validação de dados."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]
    recommendations: List[str]


class DataTransformationResult(BaseModel):
    """Resultado da transformação de dados."""
    transformed_features: List[str]
    transformation_summary: Dict[str, Any]
    quality_metrics: Dict[str, Any]


class DataProcessingService(BaseService):
    """Serviço de processamento de dados."""
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        super().__init__("data_processing_service", "1.0.0", host, port)
        
        self.app = FastAPI(
            title="Valion Data Processing Service",
            description="Serviço para processamento, validação e transformação de dados",
            version="1.0.0"
        )
        
        # Componentes
        self.data_loader = DataLoader()
        self.variable_transformer = VariableTransformer()
        self.data_quality_monitor = DataQualityMonitor(get_default_drift_config())
        
        # Métricas
        self.metrics = get_metrics_collector()
        
        # Logger estruturado
        self.struct_logger = structlog.get_logger("data_processing_service")
        
        # Armazenamento temporário
        self.temp_dir = Path(tempfile.mkdtemp(prefix="valion_data_"))
        self.processed_datasets = {}
        
        # Configurar rotas
        self._setup_routes()
    
    def _setup_routes(self):
        """Configura rotas do serviço."""
        
        @self.app.post("/process", response_model=DataProcessingResponse)
        async def process_data(
            request: DataProcessingRequest,
            background_tasks: BackgroundTasks
        ):
            """Processa dados de avaliação."""
            return await self._process_data(request, background_tasks)
        
        @self.app.post("/upload")
        async def upload_file(
            file: UploadFile = File(...),
            evaluation_id: str = None
        ):
            """Faz upload de arquivo de dados."""
            return await self._upload_file(file, evaluation_id)
        
        @self.app.get("/validate/{evaluation_id}")
        async def validate_data(evaluation_id: str):
            """Valida dados de uma avaliação."""
            return await self._validate_data(evaluation_id)
        
        @self.app.get("/transform/{evaluation_id}")
        async def transform_data(evaluation_id: str):
            """Transforma dados de uma avaliação."""
            return await self._transform_data(evaluation_id)
        
        @self.app.get("/drift/{evaluation_id}")
        async def detect_drift(evaluation_id: str):
            """Detecta drift nos dados."""
            return await self._detect_drift(evaluation_id)
        
        @self.app.get("/summary/{evaluation_id}")
        async def get_data_summary(evaluation_id: str):
            """Obtém resumo dos dados."""
            return await self._get_data_summary(evaluation_id)
        
        @self.app.get("/health")
        async def health_check():
            """Health check do serviço."""
            health = await super().health_check()
            return health.dict()
    
    async def _process_data(self, request: DataProcessingRequest, 
                           background_tasks: BackgroundTasks) -> DataProcessingResponse:
        """Processa dados de avaliação."""
        start_time = datetime.now()
        
        try:
            self.struct_logger.info(
                "Starting data processing",
                evaluation_id=request.evaluation_id,
                target_column=request.target_column
            )
            
            # Verificar se dados existem
            if request.evaluation_id not in self.processed_datasets:
                raise HTTPException(
                    status_code=404,
                    detail="Dataset not found. Please upload data first."
                )
            
            dataset = self.processed_datasets[request.evaluation_id]
            
            # Resumo inicial dos dados
            data_summary = self._generate_data_summary(dataset)
            
            # Resultados
            validation_results = None
            transformation_results = None
            drift_results = None
            
            # Validação
            if request.perform_validation:
                with measure_time("data_validation_duration"):
                    validation_results = await self._perform_validation(dataset, request)
            
            # Transformação
            if request.perform_transformation:
                with measure_time("data_transformation_duration"):
                    transformation_results = await self._perform_transformation(
                        dataset, request, validation_results
                    )
            
            # Detecção de drift
            if request.detect_drift:
                with measure_time("drift_detection_duration"):
                    drift_results = await self._perform_drift_detection(dataset, request)
            
            # Calcular tempo de processamento
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Registrar métricas
            self.metrics.record_file_operation(
                file_type="dataset",
                operation="processing",
                status="success",
                duration=processing_time
            )
            
            # Criar resposta
            response = DataProcessingResponse(
                evaluation_id=request.evaluation_id,
                status="success",
                message="Data processing completed successfully",
                data_summary=data_summary,
                validation_results=validation_results,
                transformation_results=transformation_results,
                drift_results=drift_results,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            self.struct_logger.info(
                "Data processing completed",
                evaluation_id=request.evaluation_id,
                processing_time=processing_time,
                validation_status=validation_results["status"] if validation_results else "skipped",
                transformation_status=transformation_results["status"] if transformation_results else "skipped"
            )
            
            # Adicionar cleanup em background
            background_tasks.add_task(self._cleanup_old_datasets)
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Registrar erro
            self.metrics.record_file_operation(
                file_type="dataset",
                operation="processing",
                status="error",
                duration=processing_time
            )
            
            self.struct_logger.error(
                "Data processing failed",
                evaluation_id=request.evaluation_id,
                error=str(e),
                processing_time=processing_time
            )
            
            raise HTTPException(
                status_code=500,
                detail=f"Data processing failed: {str(e)}"
            )
    
    async def _upload_file(self, file: UploadFile, evaluation_id: str = None) -> Dict[str, Any]:
        """Faz upload de arquivo de dados."""
        try:
            # Gerar ID se não fornecido
            if not evaluation_id:
                evaluation_id = f"eval_{int(datetime.now().timestamp())}"
            
            # Verificar tipo de arquivo
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in ['.csv', '.xlsx', '.xls']:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Please upload CSV or Excel files."
                )
            
            # Salvar arquivo temporário
            file_path = self.temp_dir / f"{evaluation_id}_{file.filename}"
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Carregar dados
            with measure_time("file_upload_duration"):
                if file_extension == '.csv':
                    dataset = pd.read_csv(file_path)
                else:
                    dataset = pd.read_excel(file_path)
            
            # Armazenar dataset
            self.processed_datasets[evaluation_id] = dataset
            
            # Registrar métricas
            self.metrics.record_file_operation(
                file_type=file_extension[1:],
                operation="upload",
                status="success",
                duration=0.0
            )
            
            # Resumo dos dados
            data_summary = self._generate_data_summary(dataset)
            
            self.struct_logger.info(
                "File uploaded successfully",
                evaluation_id=evaluation_id,
                filename=file.filename,
                rows=len(dataset),
                columns=len(dataset.columns)
            )
            
            return {
                "evaluation_id": evaluation_id,
                "filename": file.filename,
                "status": "uploaded",
                "data_summary": data_summary
            }
            
        except Exception as e:
            self.struct_logger.error(f"File upload failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"File upload failed: {str(e)}"
            )
    
    async def _validate_data(self, evaluation_id: str) -> DataValidationResult:
        """Valida dados de uma avaliação."""
        try:
            if evaluation_id not in self.processed_datasets:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            dataset = self.processed_datasets[evaluation_id]
            
            # Realizar validação
            validation_result = self.data_loader.validate_data(dataset)
            
            return DataValidationResult(
                is_valid=validation_result.is_valid,
                errors=validation_result.errors,
                warnings=validation_result.warnings,
                summary=validation_result.summary,
                recommendations=validation_result.recommendations
            )
            
        except Exception as e:
            self.struct_logger.error(f"Data validation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Data validation failed: {str(e)}"
            )
    
    async def _transform_data(self, evaluation_id: str) -> DataTransformationResult:
        """Transforma dados de uma avaliação."""
        try:
            if evaluation_id not in self.processed_datasets:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            dataset = self.processed_datasets[evaluation_id]
            
            # Realizar transformação
            transformation_result = self.variable_transformer.transform_variables(dataset)
            
            # Atualizar dataset
            self.processed_datasets[evaluation_id] = transformation_result.transformed_data
            
            return DataTransformationResult(
                transformed_features=transformation_result.transformed_features,
                transformation_summary=transformation_result.transformation_summary,
                quality_metrics=transformation_result.quality_metrics
            )
            
        except Exception as e:
            self.struct_logger.error(f"Data transformation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Data transformation failed: {str(e)}"
            )
    
    async def _detect_drift(self, evaluation_id: str) -> Dict[str, Any]:
        """Detecta drift nos dados."""
        try:
            if evaluation_id not in self.processed_datasets:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            dataset = self.processed_datasets[evaluation_id]
            
            # Análise de qualidade
            report = self.data_quality_monitor.analyze_data_quality(dataset, evaluation_id)
            
            return {
                "drift_score": report.overall_drift_score,
                "anomaly_rate": report.overall_anomaly_rate,
                "quality_score": report.data_quality_score,
                "recommendations": report.recommendations,
                "drift_results": [
                    {
                        "feature": r.feature_name,
                        "test_type": r.test_type,
                        "is_drift": r.is_drift,
                        "severity": r.severity,
                        "recommendation": r.recommendation
                    }
                    for r in report.drift_results
                ]
            }
            
        except Exception as e:
            self.struct_logger.error(f"Drift detection failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Drift detection failed: {str(e)}"
            )
    
    async def _get_data_summary(self, evaluation_id: str) -> Dict[str, Any]:
        """Obtém resumo dos dados."""
        try:
            if evaluation_id not in self.processed_datasets:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            dataset = self.processed_datasets[evaluation_id]
            return self._generate_data_summary(dataset)
            
        except Exception as e:
            self.struct_logger.error(f"Get data summary failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Get data summary failed: {str(e)}"
            )
    
    def _generate_data_summary(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Gera resumo dos dados."""
        try:
            summary = {
                "total_rows": len(dataset),
                "total_columns": len(dataset.columns),
                "columns": dataset.columns.tolist(),
                "data_types": dataset.dtypes.astype(str).to_dict(),
                "missing_values": dataset.isnull().sum().to_dict(),
                "missing_percentage": (dataset.isnull().sum() / len(dataset) * 100).round(2).to_dict(),
                "numeric_columns": dataset.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": dataset.select_dtypes(include=['object']).columns.tolist(),
                "memory_usage": dataset.memory_usage(deep=True).sum(),
                "has_duplicates": dataset.duplicated().any(),
                "duplicate_count": dataset.duplicated().sum()
            }
            
            # Estatísticas descritivas para colunas numéricas
            if summary["numeric_columns"]:
                numeric_stats = dataset[summary["numeric_columns"]].describe()
                summary["numeric_statistics"] = numeric_stats.to_dict()
            
            # Informações sobre colunas categóricas
            if summary["categorical_columns"]:
                categorical_info = {}
                for col in summary["categorical_columns"]:
                    categorical_info[col] = {
                        "unique_values": dataset[col].nunique(),
                        "most_frequent": dataset[col].mode().iloc[0] if not dataset[col].mode().empty else None,
                        "unique_values_sample": dataset[col].value_counts().head(5).to_dict()
                    }
                summary["categorical_info"] = categorical_info
            
            return summary
            
        except Exception as e:
            self.struct_logger.error(f"Error generating data summary: {e}")
            return {
                "total_rows": len(dataset),
                "total_columns": len(dataset.columns),
                "error": str(e)
            }
    
    async def _perform_validation(self, dataset: pd.DataFrame, 
                                 request: DataProcessingRequest) -> Dict[str, Any]:
        """Realiza validação dos dados."""
        try:
            validation_result = self.data_loader.validate_data(dataset)
            
            return {
                "status": "valid" if validation_result.is_valid else "invalid",
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "summary": validation_result.summary,
                "recommendations": validation_result.recommendations
            }
            
        except Exception as e:
            return {
                "status": "error",
                "is_valid": False,
                "errors": [str(e)],
                "warnings": [],
                "summary": {},
                "recommendations": ["Fix validation errors before proceeding"]
            }
    
    async def _perform_transformation(self, dataset: pd.DataFrame, 
                                     request: DataProcessingRequest,
                                     validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza transformação dos dados."""
        try:
            # Só transformar se validação passou
            if validation_results and not validation_results.get("is_valid", True):
                return {
                    "status": "skipped",
                    "message": "Transformation skipped due to validation errors"
                }
            
            transformation_result = self.variable_transformer.transform_variables(dataset)
            
            # Atualizar dataset armazenado
            self.processed_datasets[request.evaluation_id] = transformation_result.transformed_data
            
            return {
                "status": "success",
                "transformed_features": transformation_result.transformed_features,
                "transformation_summary": transformation_result.transformation_summary,
                "quality_metrics": transformation_result.quality_metrics
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "transformed_features": [],
                "transformation_summary": {},
                "quality_metrics": {}
            }
    
    async def _perform_drift_detection(self, dataset: pd.DataFrame, 
                                      request: DataProcessingRequest) -> Dict[str, Any]:
        """Realiza detecção de drift."""
        try:
            # Verificar se há dados de referência
            if not hasattr(self.data_quality_monitor, 'reference_data') or self.data_quality_monitor.reference_data is None:
                # Usar dataset atual como referência se não houver
                self.data_quality_monitor.set_reference_data(dataset)
                
                return {
                    "status": "reference_set",
                    "message": "Dataset used as reference for future drift detection",
                    "drift_score": 0.0,
                    "anomaly_rate": 0.0,
                    "quality_score": 100.0,
                    "recommendations": ["Reference dataset established"]
                }
            
            # Realizar análise de qualidade
            report = self.data_quality_monitor.analyze_data_quality(dataset, request.evaluation_id)
            
            return {
                "status": "completed",
                "drift_score": report.overall_drift_score,
                "anomaly_rate": report.overall_anomaly_rate,
                "quality_score": report.data_quality_score,
                "recommendations": report.recommendations,
                "drift_details": [
                    {
                        "feature": r.feature_name,
                        "test_type": r.test_type,
                        "is_drift": r.is_drift,
                        "severity": r.severity,
                        "statistic": r.statistic,
                        "p_value": r.p_value,
                        "interpretation": r.interpretation,
                        "recommendation": r.recommendation
                    }
                    for r in report.drift_results
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "drift_score": 0.0,
                "anomaly_rate": 0.0,
                "quality_score": 0.0,
                "recommendations": ["Fix drift detection errors"]
            }
    
    async def _cleanup_old_datasets(self):
        """Limpa datasets antigos para liberar memória."""
        try:
            # Manter apenas 10 datasets mais recentes
            if len(self.processed_datasets) > 10:
                # Ordenar por timestamp (assumindo que IDs contêm timestamp)
                sorted_keys = sorted(self.processed_datasets.keys())
                
                # Remover mais antigos
                for key in sorted_keys[:-10]:
                    del self.processed_datasets[key]
                    
                self.struct_logger.info(
                    "Cleaned up old datasets",
                    remaining_count=len(self.processed_datasets)
                )
                
        except Exception as e:
            self.struct_logger.error(f"Error cleaning up datasets: {e}")
    
    async def initialize(self):
        """Inicialização do serviço."""
        # Criar diretório temporário
        self.temp_dir.mkdir(exist_ok=True)
        
        self.struct_logger.info("Data Processing Service initialized")
    
    async def cleanup(self):
        """Limpeza do serviço."""
        # Limpar diretório temporário
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            self.struct_logger.error(f"Error cleaning up temp directory: {e}")
        
        self.struct_logger.info("Data Processing Service cleaned up")


def create_data_processing_service(host: str = "localhost", port: int = 8001) -> DataProcessingService:
    """Cria instância do serviço de processamento de dados."""
    return DataProcessingService(host, port)