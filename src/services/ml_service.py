# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Serviço de Machine Learning
Microserviço responsável por treinamento e inferência de modelos.
"""

import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import json
import tempfile
from pathlib import Path
import joblib
import uuid

from .base_service import BaseService
from ..core.model_builder import ModelBuilder, ModelResult
from ..core.cache_system import ModelCache
from ..monitoring.metrics import get_metrics_collector, measure_time
from ..monitoring.logging_config import get_logger
import structlog


# Modelos Pydantic
class MLTrainingRequest(BaseModel):
    """Requisição de treinamento de modelo."""
    evaluation_id: str = Field(..., description="ID da avaliação")
    model_type: str = Field(default="elastic_net", description="Tipo do modelo")
    target_column: str = Field(default="valor", description="Coluna target")
    expert_mode: bool = Field(default=False, description="Modo especialista")
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Hiperparâmetros customizados")
    cross_validation: bool = Field(default=True, description="Usar validação cruzada")
    feature_selection: bool = Field(default=True, description="Realizar seleção de features")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configurações adicionais")


class MLInferenceRequest(BaseModel):
    """Requisição de inferência."""
    model_id: str = Field(..., description="ID do modelo")
    data: List[Dict[str, Any]] = Field(..., description="Dados para predição")
    return_confidence: bool = Field(default=True, description="Retornar intervalos de confiança")
    return_shap: bool = Field(default=False, description="Retornar valores SHAP")


class MLTrainingResponse(BaseModel):
    """Resposta do treinamento."""
    evaluation_id: str
    model_id: str
    status: str
    message: str
    model_type: str
    performance_metrics: Dict[str, Any]
    training_summary: Dict[str, Any]
    training_time: float
    expert_mode: bool
    timestamp: datetime


class MLInferenceResponse(BaseModel):
    """Resposta da inferência."""
    model_id: str
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    inference_time: float
    timestamp: datetime


class ModelInfo(BaseModel):
    """Informações do modelo."""
    model_id: str
    evaluation_id: str
    model_type: str
    performance_metrics: Dict[str, Any]
    training_timestamp: datetime
    features: List[str]
    target_column: str
    expert_mode: bool
    status: str


class MLService(BaseService):
    """Serviço de Machine Learning."""
    
    def __init__(self, host: str = "localhost", port: int = 8002):
        super().__init__("ml_service", "1.0.0", host, port)
        
        self.app = FastAPI(
            title="Valion ML Service",
            description="Serviço para treinamento e inferência de modelos de machine learning",
            version="1.0.0"
        )
        
        # Componentes
        self.model_cache = ModelCache(cache_dir="ml_service_cache", max_models=100)
        self.active_models = {}  # Modelos carregados na memória
        self.training_jobs = {}  # Jobs de treinamento ativos
        
        # Métricas
        self.metrics = get_metrics_collector()
        
        # Logger estruturado
        self.struct_logger = structlog.get_logger("ml_service")
        
        # Diretório temporário
        self.temp_dir = Path(tempfile.mkdtemp(prefix="valion_ml_"))
        
        # Configurar rotas
        self._setup_routes()
    
    def _setup_routes(self):
        """Configura rotas do serviço."""
        
        @self.app.post("/train", response_model=MLTrainingResponse)
        async def train_model(
            request: MLTrainingRequest,
            background_tasks: BackgroundTasks
        ):
            """Treina um modelo de machine learning."""
            return await self._train_model(request, background_tasks)
        
        @self.app.post("/predict", response_model=MLInferenceResponse)
        async def predict(request: MLInferenceRequest):
            """Faz predições com um modelo treinado."""
            return await self._predict(request)
        
        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """Lista todos os modelos disponíveis."""
            return await self._list_models()
        
        @self.app.get("/models/{model_id}", response_model=ModelInfo)
        async def get_model_info(model_id: str):
            """Obtém informações de um modelo específico."""
            return await self._get_model_info(model_id)
        
        @self.app.delete("/models/{model_id}")
        async def delete_model(model_id: str):
            """Remove um modelo."""
            return await self._delete_model(model_id)
        
        @self.app.get("/training/{evaluation_id}")
        async def get_training_status(evaluation_id: str):
            """Obtém status do treinamento."""
            return await self._get_training_status(evaluation_id)
        
        @self.app.post("/load/{model_id}")
        async def load_model(model_id: str):
            """Carrega modelo na memória."""
            return await self._load_model(model_id)
        
        @self.app.post("/unload/{model_id}")
        async def unload_model(model_id: str):
            """Descarrega modelo da memória."""
            return await self._unload_model(model_id)
        
        @self.app.get("/cache/stats")
        async def get_cache_stats():
            """Obtém estatísticas do cache de modelos."""
            return await self._get_cache_stats()
        
        @self.app.get("/health")
        async def health_check():
            """Health check do serviço."""
            health = await super().health_check()
            health_dict = health.dict()
            
            # Adicionar informações específicas do ML
            health_dict["active_models"] = len(self.active_models)
            health_dict["training_jobs"] = len(self.training_jobs)
            health_dict["cached_models"] = len(self.model_cache.models_registry)
            
            return health_dict
    
    async def _train_model(self, request: MLTrainingRequest, 
                          background_tasks: BackgroundTasks) -> MLTrainingResponse:
        """Treina um modelo de machine learning."""
        start_time = datetime.now()
        
        try:
            self.struct_logger.info(
                "Starting model training",
                evaluation_id=request.evaluation_id,
                model_type=request.model_type,
                expert_mode=request.expert_mode
            )
            
            # Buscar dados processados do data processing service
            dataset = await self._get_processed_data(request.evaluation_id)
            
            if dataset is None:
                raise HTTPException(
                    status_code=404,
                    detail="Processed data not found. Please process data first."
                )
            
            # Verificar se modelo similar já existe no cache
            features_list = [col for col in dataset.columns if col != request.target_column]
            
            config = {
                **request.config,
                "model_type": request.model_type,
                "expert_mode": request.expert_mode,
                "target_column": request.target_column
            }
            
            cached_model = self.model_cache.find_similar_model(
                model_type=request.model_type,
                features=features_list,
                config=config,
                similarity_threshold=0.95
            )
            
            if cached_model:
                self.struct_logger.info(
                    "Similar model found in cache",
                    evaluation_id=request.evaluation_id,
                    cached_model_id=cached_model.model_id
                )
                
                # Carregar modelo do cache
                cached_model_data = self.model_cache.load_model(cached_model.model_id)
                
                if cached_model_data:
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Registrar métricas
                    self.metrics.record_model_training(
                        model_type=request.model_type,
                        duration=training_time,
                        performance_metrics=cached_model.performance_metrics
                    )
                    
                    return MLTrainingResponse(
                        evaluation_id=request.evaluation_id,
                        model_id=cached_model.model_id,
                        status="cached",
                        message="Model loaded from cache",
                        model_type=request.model_type,
                        performance_metrics=cached_model.performance_metrics,
                        training_summary=cached_model.training_config,
                        training_time=training_time,
                        expert_mode=request.expert_mode,
                        timestamp=datetime.now()
                    )
            
            # Treinar novo modelo
            model_id = f"model_{uuid.uuid4().hex[:8]}_{request.evaluation_id}"
            
            # Marcar como em treinamento
            self.training_jobs[request.evaluation_id] = {
                "model_id": model_id,
                "status": "training",
                "start_time": start_time,
                "progress": 0.0
            }
            
            # Configurar model builder
            model_builder = ModelBuilder(config)
            
            # Treinar modelo
            with measure_time("model_training_duration"):
                model_result = model_builder.build_model(dataset, request.target_column)
            
            # Salvar modelo no cache
            model_data = {
                "model": model_result.model,
                "performance": model_result.performance,
                "best_params": model_result.best_params,
                "training_summary": model_result.training_summary,
                "scaler": model_builder.scaler,
                "explainer": model_result.explainer,
                "features": features_list,
                "target_column": request.target_column,
                "model_type": request.model_type,
                "expert_mode": request.expert_mode,
                "evaluation_id": request.evaluation_id
            }
            
            performance_metrics = {
                "r2_score": model_result.performance.r2_score,
                "rmse": model_result.performance.rmse,
                "mae": model_result.performance.mae,
                "mape": model_result.performance.mape
            }
            
            # Salvar no cache se performance for boa
            if model_result.performance.r2_score >= 0.6:
                self.model_cache.cache_model(
                    model_data=model_data,
                    model_type=request.model_type,
                    features=features_list,
                    config=config,
                    performance_metrics=performance_metrics
                )
            
            # Armazenar modelo ativo
            self.active_models[model_id] = {
                "model_data": model_data,
                "loaded_at": datetime.now(),
                "access_count": 0
            }
            
            # Calcular tempo de treinamento
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Atualizar status do job
            self.training_jobs[request.evaluation_id] = {
                "model_id": model_id,
                "status": "completed",
                "start_time": start_time,
                "end_time": datetime.now(),
                "progress": 100.0
            }
            
            # Registrar métricas
            self.metrics.record_model_training(
                model_type=request.model_type,
                duration=training_time,
                performance_metrics=performance_metrics
            )
            
            self.struct_logger.info(
                "Model training completed",
                evaluation_id=request.evaluation_id,
                model_id=model_id,
                r2_score=model_result.performance.r2_score,
                training_time=training_time
            )
            
            # Adicionar cleanup em background
            background_tasks.add_task(self._cleanup_old_models)
            
            return MLTrainingResponse(
                evaluation_id=request.evaluation_id,
                model_id=model_id,
                status="completed",
                message="Model trained successfully",
                model_type=request.model_type,
                performance_metrics=performance_metrics,
                training_summary=model_result.training_summary,
                training_time=training_time,
                expert_mode=request.expert_mode,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Atualizar status do job
            self.training_jobs[request.evaluation_id] = {
                "model_id": f"failed_{request.evaluation_id}",
                "status": "failed",
                "start_time": start_time,
                "end_time": datetime.now(),
                "error": str(e),
                "progress": 0.0
            }
            
            self.struct_logger.error(
                "Model training failed",
                evaluation_id=request.evaluation_id,
                error=str(e),
                training_time=training_time
            )
            
            raise HTTPException(
                status_code=500,
                detail=f"Model training failed: {str(e)}"
            )
    
    async def _predict(self, request: MLInferenceRequest) -> MLInferenceResponse:
        """Faz predições com um modelo treinado."""
        start_time = datetime.now()
        
        try:
            self.struct_logger.info(
                "Starting prediction",
                model_id=request.model_id,
                samples_count=len(request.data)
            )
            
            # Carregar modelo se necessário
            if request.model_id not in self.active_models:
                await self._load_model(request.model_id)
            
            model_info = self.active_models[request.model_id]
            model_data = model_info["model_data"]
            
            # Atualizar contador de acesso
            model_info["access_count"] += 1
            model_info["last_accessed"] = datetime.now()
            
            # Preparar dados para predição
            features = model_data["features"]
            df_input = pd.DataFrame(request.data)
            
            # Verificar se todas as features estão presentes
            missing_features = set(features) - set(df_input.columns)
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing features: {list(missing_features)}"
                )
            
            # Selecionar e ordenar features
            X = df_input[features]
            
            # Padronizar dados
            scaler = model_data["scaler"]
            X_scaled = scaler.transform(X)
            
            # Fazer predições
            model = model_data["model"]
            predictions = model.predict(X_scaled)
            
            # Preparar resposta
            predictions_list = []
            
            for i, prediction in enumerate(predictions):
                pred_dict = {
                    "sample_id": i,
                    "prediction": float(prediction),
                    "features": request.data[i]
                }
                
                # Adicionar intervalo de confiança se solicitado
                if request.return_confidence and hasattr(model, 'predict'):
                    # Simular intervalo de confiança (implementação simplificada)
                    std_error = model_data["performance"]["rmse"] * 0.1
                    pred_dict["confidence_interval"] = {
                        "lower": float(prediction - 1.96 * std_error),
                        "upper": float(prediction + 1.96 * std_error)
                    }
                
                # Adicionar valores SHAP se solicitado
                if request.return_shap and model_data.get("explainer"):
                    try:
                        explainer = model_data["explainer"]
                        shap_values = explainer.shap_values(X_scaled[i:i+1])
                        
                        if shap_values is not None:
                            pred_dict["shap_values"] = dict(zip(features, shap_values[0]))
                    except Exception as e:
                        self.struct_logger.warning(f"SHAP calculation failed: {e}")
                
                predictions_list.append(pred_dict)
            
            # Calcular tempo de inferência
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Registrar métricas
            self.metrics.record_api_request(
                method="POST",
                endpoint="/predict",
                status_code=200,
                duration=inference_time
            )
            
            self.struct_logger.info(
                "Prediction completed",
                model_id=request.model_id,
                samples_count=len(request.data),
                inference_time=inference_time
            )
            
            return MLInferenceResponse(
                model_id=request.model_id,
                predictions=predictions_list,
                model_info={
                    "model_type": model_data["model_type"],
                    "features": features,
                    "target_column": model_data["target_column"],
                    "performance": model_data["performance"].__dict__ if hasattr(model_data["performance"], '__dict__') else model_data["performance"],
                    "expert_mode": model_data["expert_mode"]
                },
                inference_time=inference_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            inference_time = (datetime.now() - start_time).total_seconds()
            
            self.struct_logger.error(
                "Prediction failed",
                model_id=request.model_id,
                error=str(e),
                inference_time=inference_time
            )
            
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    async def _get_processed_data(self, evaluation_id: str) -> Optional[pd.DataFrame]:
        """Obtém dados processados do data processing service."""
        try:
            # Chamar data processing service
            from .base_service import ServiceRequest
            
            request = ServiceRequest(
                service_name="data_processing_service",
                method="GET",
                endpoint=f"/summary/{evaluation_id}",
                payload={}
            )
            
            response = await self.call_service(request)
            
            if response.success:
                # Em um cenário real, retornaria os dados processados
                # Por simplicidade, retornando dados simulados
                return self._generate_sample_data()
            
            return None
            
        except Exception as e:
            self.struct_logger.error(f"Error getting processed data: {e}")
            return None
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Gera dados de exemplo para teste."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'area': np.random.normal(100, 25, n_samples),
            'quartos': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'banheiros': np.random.choice([1, 2, 3, 4], n_samples),
            'idade': np.random.exponential(10, n_samples),
            'preco_m2': np.random.normal(3000, 500, n_samples),
            'valor': np.random.normal(300000, 100000, n_samples)
        }
        
        return pd.DataFrame(data)
    
    async def _list_models(self) -> List[ModelInfo]:
        """Lista todos os modelos disponíveis."""
        try:
            models = []
            
            # Modelos ativos
            for model_id, model_info in self.active_models.items():
                model_data = model_info["model_data"]
                
                models.append(ModelInfo(
                    model_id=model_id,
                    evaluation_id=model_data["evaluation_id"],
                    model_type=model_data["model_type"],
                    performance_metrics=model_data["performance"].__dict__ if hasattr(model_data["performance"], '__dict__') else model_data["performance"],
                    training_timestamp=model_info["loaded_at"],
                    features=model_data["features"],
                    target_column=model_data["target_column"],
                    expert_mode=model_data["expert_mode"],
                    status="active"
                ))
            
            # Modelos em cache
            for model_id, cache_entry in self.model_cache.models_registry.items():
                if model_id not in self.active_models:
                    models.append(ModelInfo(
                        model_id=model_id,
                        evaluation_id="unknown",
                        model_type=cache_entry.model_type,
                        performance_metrics=cache_entry.performance_metrics,
                        training_timestamp=datetime.fromtimestamp(cache_entry.training_timestamp),
                        features=[],
                        target_column="unknown",
                        expert_mode=False,
                        status="cached"
                    ))
            
            return models
            
        except Exception as e:
            self.struct_logger.error(f"Error listing models: {e}")
            return []
    
    async def _get_model_info(self, model_id: str) -> ModelInfo:
        """Obtém informações de um modelo específico."""
        try:
            # Verificar modelos ativos
            if model_id in self.active_models:
                model_info = self.active_models[model_id]
                model_data = model_info["model_data"]
                
                return ModelInfo(
                    model_id=model_id,
                    evaluation_id=model_data["evaluation_id"],
                    model_type=model_data["model_type"],
                    performance_metrics=model_data["performance"].__dict__ if hasattr(model_data["performance"], '__dict__') else model_data["performance"],
                    training_timestamp=model_info["loaded_at"],
                    features=model_data["features"],
                    target_column=model_data["target_column"],
                    expert_mode=model_data["expert_mode"],
                    status="active"
                )
            
            # Verificar cache
            if model_id in self.model_cache.models_registry:
                cache_entry = self.model_cache.models_registry[model_id]
                
                return ModelInfo(
                    model_id=model_id,
                    evaluation_id="unknown",
                    model_type=cache_entry.model_type,
                    performance_metrics=cache_entry.performance_metrics,
                    training_timestamp=datetime.fromtimestamp(cache_entry.training_timestamp),
                    features=[],
                    target_column="unknown",
                    expert_mode=False,
                    status="cached"
                )
            
            raise HTTPException(status_code=404, detail="Model not found")
            
        except HTTPException:
            raise
        except Exception as e:
            self.struct_logger.error(f"Error getting model info: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _delete_model(self, model_id: str) -> Dict[str, Any]:
        """Remove um modelo."""
        try:
            deleted = False
            
            # Remover de modelos ativos
            if model_id in self.active_models:
                del self.active_models[model_id]
                deleted = True
            
            # Remover do cache
            if model_id in self.model_cache.models_registry:
                self.model_cache._remove_model(model_id)
                deleted = True
            
            if deleted:
                self.struct_logger.info(f"Model {model_id} deleted")
                return {"message": "Model deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Model not found")
                
        except HTTPException:
            raise
        except Exception as e:
            self.struct_logger.error(f"Error deleting model: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _get_training_status(self, evaluation_id: str) -> Dict[str, Any]:
        """Obtém status do treinamento."""
        try:
            if evaluation_id in self.training_jobs:
                job = self.training_jobs[evaluation_id]
                return {
                    "evaluation_id": evaluation_id,
                    "status": job["status"],
                    "progress": job["progress"],
                    "start_time": job["start_time"].isoformat(),
                    "end_time": job.get("end_time", {}).isoformat() if job.get("end_time") else None,
                    "model_id": job["model_id"],
                    "error": job.get("error")
                }
            else:
                return {
                    "evaluation_id": evaluation_id,
                    "status": "not_found",
                    "message": "Training job not found"
                }
                
        except Exception as e:
            self.struct_logger.error(f"Error getting training status: {e}")
            return {
                "evaluation_id": evaluation_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _load_model(self, model_id: str) -> Dict[str, Any]:
        """Carrega modelo na memória."""
        try:
            # Verificar se já está carregado
            if model_id in self.active_models:
                return {"message": "Model already loaded"}
            
            # Carregar do cache
            model_data = self.model_cache.load_model(model_id)
            
            if model_data:
                self.active_models[model_id] = {
                    "model_data": model_data,
                    "loaded_at": datetime.now(),
                    "access_count": 0
                }
                
                self.struct_logger.info(f"Model {model_id} loaded into memory")
                return {"message": "Model loaded successfully"}
            else:
                raise HTTPException(status_code=404, detail="Model not found in cache")
                
        except HTTPException:
            raise
        except Exception as e:
            self.struct_logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _unload_model(self, model_id: str) -> Dict[str, Any]:
        """Descarrega modelo da memória."""
        try:
            if model_id in self.active_models:
                del self.active_models[model_id]
                self.struct_logger.info(f"Model {model_id} unloaded from memory")
                return {"message": "Model unloaded successfully"}
            else:
                raise HTTPException(status_code=404, detail="Model not loaded")
                
        except HTTPException:
            raise
        except Exception as e:
            self.struct_logger.error(f"Error unloading model: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _get_cache_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do cache de modelos."""
        try:
            cache_stats = self.model_cache.get_cache_stats()
            
            return {
                "cache_stats": cache_stats,
                "active_models": len(self.active_models),
                "training_jobs": len(self.training_jobs),
                "memory_usage": {
                    "active_models_count": len(self.active_models),
                    "training_jobs_count": len(self.training_jobs)
                }
            }
            
        except Exception as e:
            self.struct_logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def _cleanup_old_models(self):
        """Limpa modelos antigos da memória."""
        try:
            # Manter apenas 5 modelos mais recentes na memória
            if len(self.active_models) > 5:
                # Ordenar por último acesso
                sorted_models = sorted(
                    self.active_models.items(),
                    key=lambda x: x[1].get("last_accessed", x[1]["loaded_at"])
                )
                
                # Remover modelos mais antigos
                for model_id, _ in sorted_models[:-5]:
                    del self.active_models[model_id]
                    self.struct_logger.info(f"Removed old model from memory: {model_id}")
            
            # Limpar jobs de treinamento antigos
            if len(self.training_jobs) > 20:
                # Manter apenas 20 jobs mais recentes
                sorted_jobs = sorted(
                    self.training_jobs.items(),
                    key=lambda x: x[1]["start_time"]
                )
                
                for eval_id, _ in sorted_jobs[:-20]:
                    del self.training_jobs[eval_id]
                    
        except Exception as e:
            self.struct_logger.error(f"Error cleaning up old models: {e}")
    
    async def initialize(self):
        """Inicialização do serviço."""
        self.struct_logger.info("ML Service initialized")
    
    async def cleanup(self):
        """Limpeza do serviço."""
        # Limpar diretório temporário
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            self.struct_logger.error(f"Error cleaning up temp directory: {e}")
        
        self.struct_logger.info("ML Service cleaned up")


def create_ml_service(host: str = "localhost", port: int = 8002) -> MLService:
    """Cria instância do serviço de ML."""
    return MLService(host, port)