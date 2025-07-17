"""
API FastAPI para Valion - Plataforma de Avaliação Imobiliária
Responsável por servir a API e gerenciar conexões WebSocket.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
from datetime import datetime
import pandas as pd
import uuid
from pathlib import Path
import os

# Imports dos módulos core
from src.core.data_loader import DataLoader, DataValidationResult
from src.core.transformations import VariableTransformer, TransformationResult
from src.core.model_builder import ElasticNetModelBuilder, ModelResult
from src.core.nbr14653_validation import NBR14653Validator, NBRValidationResult
from src.core.results_generator import ResultsGenerator, EvaluationReport
from src.config.settings import Settings

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização da aplicação
app = FastAPI(
    title="Valion API",
    description="API para avaliação imobiliária com transparência e auditabilidade",
    version="1.0.0"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações
settings = Settings()

# Modelos Pydantic
class EvaluationRequest(BaseModel):
    """Requisição de avaliação."""
    file_path: str
    target_column: str = "valor"
    config: Optional[Dict[str, Any]] = None

class EvaluationStatus(BaseModel):
    """Status da avaliação."""
    evaluation_id: str
    status: str
    current_phase: str
    progress: float
    message: str
    timestamp: datetime

class PredictionRequest(BaseModel):
    """Requisição de predição."""
    evaluation_id: str
    features: Dict[str, Any]

# Gerenciador de conexões WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, evaluation_id: str):
        await websocket.accept()
        self.active_connections[evaluation_id] = websocket
        logger.info(f"WebSocket conectado: {evaluation_id}")
    
    def disconnect(self, evaluation_id: str):
        if evaluation_id in self.active_connections:
            del self.active_connections[evaluation_id]
            logger.info(f"WebSocket desconectado: {evaluation_id}")
    
    async def send_status(self, evaluation_id: str, status: EvaluationStatus):
        if evaluation_id in self.active_connections:
            try:
                await self.active_connections[evaluation_id].send_text(
                    json.dumps(status.dict(), default=str)
                )
            except Exception as e:
                logger.error(f"Erro ao enviar status: {e}")
                self.disconnect(evaluation_id)

manager = ConnectionManager()

# Armazenamento em memória para resultados (em produção, usar Redis/Database)
evaluation_results: Dict[str, Dict[str, Any]] = {}

# Endpoints da API

@app.get("/")
async def root():
    """Endpoint raiz."""
    return {"message": "Valion API - Plataforma de Avaliação Imobiliária"}

@app.get("/health")
async def health_check():
    """Verifica saúde da API."""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/evaluations/", response_model=dict)
async def create_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Inicia uma nova avaliação imobiliária.
    
    Args:
        request: Dados da requisição
        background_tasks: Tarefas em background
        
    Returns:
        ID da avaliação iniciada
    """
    evaluation_id = str(uuid.uuid4())
    
    # Validar arquivo
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=400, detail="Arquivo não encontrado")
    
    # Inicializar status
    status = EvaluationStatus(
        evaluation_id=evaluation_id,
        status="iniciado",
        current_phase="Carregando dados",
        progress=0.0,
        message="Avaliação iniciada",
        timestamp=datetime.now()
    )
    
    evaluation_results[evaluation_id] = {
        "status": status,
        "request": request,
        "result": None
    }
    
    # Executar avaliação em background
    background_tasks.add_task(run_evaluation, evaluation_id, request)
    
    return {"evaluation_id": evaluation_id, "status": "iniciado"}

@app.get("/evaluations/{evaluation_id}")
async def get_evaluation_status(evaluation_id: str):
    """
    Obtém status da avaliação.
    
    Args:
        evaluation_id: ID da avaliação
        
    Returns:
        Status atual da avaliação
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    return evaluation_results[evaluation_id]["status"]

@app.get("/evaluations/{evaluation_id}/result")
async def get_evaluation_result(evaluation_id: str):
    """
    Obtém resultado da avaliação.
    
    Args:
        evaluation_id: ID da avaliação
        
    Returns:
        Resultado completo da avaliação
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    result = evaluation_results[evaluation_id]["result"]
    if result is None:
        raise HTTPException(status_code=202, detail="Avaliação ainda em andamento")
    
    return result

@app.post("/evaluations/{evaluation_id}/predict")
async def make_prediction(evaluation_id: str, request: PredictionRequest):
    """
    Faz predição usando modelo treinado.
    
    Args:
        evaluation_id: ID da avaliação
        request: Dados para predição
        
    Returns:
        Predição do valor
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    result = evaluation_results[evaluation_id]["result"]
    if result is None:
        raise HTTPException(status_code=202, detail="Modelo ainda não treinado")
    
    try:
        # Converter features para DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Fazer predição (assumindo que o modelo está disponível)
        # Em implementação real, seria necessário aplicar as mesmas transformações
        prediction = 0.0  # Placeholder
        
        return {
            "evaluation_id": evaluation_id,
            "features": request.features,
            "predicted_value": prediction,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Faz upload de arquivo de dados.
    
    Args:
        file: Arquivo a ser enviado
        
    Returns:
        Caminho do arquivo salvo
    """
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado")
    
    # Salvar arquivo
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return {"file_path": str(file_path), "filename": file.filename}

@app.websocket("/ws/{evaluation_id}")
async def websocket_endpoint(websocket: WebSocket, evaluation_id: str):
    """
    Endpoint WebSocket para feedback em tempo real.
    
    Args:
        websocket: Conexão WebSocket
        evaluation_id: ID da avaliação
    """
    await manager.connect(websocket, evaluation_id)
    
    try:
        while True:
            # Manter conexão ativa
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        manager.disconnect(evaluation_id)

# Função para executar avaliação
async def run_evaluation(evaluation_id: str, request: EvaluationRequest):
    """
    Executa processo completo de avaliação.
    
    Args:
        evaluation_id: ID da avaliação
        request: Dados da requisição
    """
    try:
        config = request.config or {}
        
        # Fase 1: Carregamento e validação
        await update_status(evaluation_id, "em_andamento", "Carregando dados", 10.0, "Carregando arquivo de dados")
        
        data_loader = DataLoader(config)
        raw_data = data_loader.load_data(request.file_path)
        
        await update_status(evaluation_id, "em_andamento", "Validando dados", 20.0, "Validando qualidade dos dados")
        
        validation_result = data_loader.validate_data(raw_data)
        if not validation_result.is_valid:
            await update_status(evaluation_id, "erro", "Validação falhou", 20.0, f"Erros encontrados: {validation_result.errors}")
            return
        
        cleaned_data = data_loader.clean_data(raw_data)
        
        # Fase 2: Transformações
        await update_status(evaluation_id, "em_andamento", "Transformando variáveis", 40.0, "Preparando features")
        
        transformer = VariableTransformer(config)
        transformation_result = transformer.transform_data(cleaned_data, request.target_column)
        
        # Fase 3: Modelagem
        await update_status(evaluation_id, "em_andamento", "Treinando modelo", 60.0, "Treinando modelo Elastic Net")
        
        model_builder = ElasticNetModelBuilder(config)
        model_result = model_builder.build_model(transformation_result.transformed_data, request.target_column)
        
        # Fase 4: Validação NBR
        await update_status(evaluation_id, "em_andamento", "Validando NBR 14653", 80.0, "Executando testes estatísticos")
        
        validator = NBR14653Validator(config)
        X_train, X_test, y_train, y_test = model_builder.prepare_data(transformation_result.transformed_data)
        nbr_result = validator.validate_model(model_result.model, X_train, X_test, y_train, y_test)
        
        # Fase 5: Geração de relatório
        await update_status(evaluation_id, "em_andamento", "Gerando relatório", 90.0, "Consolidando resultados")
        
        results_generator = ResultsGenerator(config)
        report = results_generator.generate_full_report(
            validation_result, transformation_result, model_result, nbr_result, cleaned_data
        )
        
        # Finalizar
        await update_status(evaluation_id, "concluido", "Avaliação concluída", 100.0, "Relatório gerado com sucesso")
        
        # Salvar resultado
        evaluation_results[evaluation_id]["result"] = {
            "report": report.__dict__,
            "model_performance": model_result.performance.__dict__,
            "nbr_validation": nbr_result.__dict__,
            "data_summary": validation_result.__dict__
        }
        
    except Exception as e:
        logger.error(f"Erro na avaliação {evaluation_id}: {str(e)}")
        await update_status(evaluation_id, "erro", "Erro na avaliação", 0.0, f"Erro: {str(e)}")

async def update_status(evaluation_id: str, status: str, phase: str, progress: float, message: str):
    """
    Atualiza status da avaliação.
    
    Args:
        evaluation_id: ID da avaliação
        status: Status atual
        phase: Fase atual
        progress: Progresso (0-100)
        message: Mensagem descritiva
    """
    status_obj = EvaluationStatus(
        evaluation_id=evaluation_id,
        status=status,
        current_phase=phase,
        progress=progress,
        message=message,
        timestamp=datetime.now()
    )
    
    evaluation_results[evaluation_id]["status"] = status_obj
    await manager.send_status(evaluation_id, status_obj)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)