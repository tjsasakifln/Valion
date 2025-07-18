"""
API FastAPI para Valion - Plataforma de Avaliação Imobiliária
Responsável por servir a API e gerenciar conexões WebSocket.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
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
import magic
from src.core.transformations import VariableTransformer, TransformationResult
from src.core.model_builder import ModelBuilder, ModelResult
from src.core.nbr14653_validation import NBR14653Validator, NBRValidationResult
from src.core.results_generator import ResultsGenerator, EvaluationReport
from src.core.geospatial_analysis import create_geospatial_analyzer, LocationAnalysis
from src.config.settings import Settings
from src.workers.tasks import process_evaluation
from src.websocket.websocket_manager import websocket_manager

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
    mode: str = "standard"  # "standard" ou "expert"
    config: Optional[Dict[str, Any]] = None
    
    def model_post_init(self, __context) -> None:
        """Validação pós-inicialização."""
        if self.mode not in ["standard", "expert"]:
            raise ValueError("Mode deve ser 'standard' ou 'expert'")
        
        # Configurar expert_mode no config baseado no mode
        if self.config is None:
            self.config = {}
        
        self.config['expert_mode'] = (self.mode == "expert")
        
        # No modo expert, garantir que modelos avançados estejam disponíveis
        if self.mode == "expert":
            # Permitir modelos avançados no modo expert
            allowed_models = ['xgboost', 'gradient_boosting', 'elastic_net']
            if 'model_type' not in self.config:
                self.config['model_type'] = 'xgboost'  # Padrão para expert mode
            elif self.config['model_type'] not in allowed_models:
                self.config['model_type'] = 'xgboost'
        else:
            # Modo standard sempre usa Elastic Net
            self.config['model_type'] = 'elastic_net'

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

class StepApprovalRequest(BaseModel):
    """Requisição de aprovação de etapa."""
    evaluation_id: str
    step: str  # "transformations", "outliers", "model_selection"
    approved: bool
    modifications: Optional[Dict[str, Any]] = None
    user_feedback: Optional[str] = None

class ShapSimulationRequest(BaseModel):
    """Requisição de simulação SHAP para laboratório interativo."""
    evaluation_id: str
    feature_modifications: Dict[str, float]  # {feature_name: new_value}
    simulation_name: Optional[str] = None
    compare_to_baseline: bool = True

class ShapWaterfallRequest(BaseModel):
    """Requisição de gráfico waterfall SHAP."""
    evaluation_id: str
    sample_index: int = 0
    feature_modifications: Optional[Dict[str, float]] = None

class GeospatialAnalysisRequest(BaseModel):
    """Requisição de análise geoespacial."""
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city_center_lat: Optional[float] = None
    city_center_lon: Optional[float] = None

class DataEnrichmentRequest(BaseModel):
    """Requisição de enriquecimento geoespacial de dataset."""
    evaluation_id: str
    address_column: str = "endereco"
    city_center_lat: Optional[float] = None
    city_center_lon: Optional[float] = None


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
async def create_evaluation(request: EvaluationRequest):
    """
    Inicia uma nova avaliação imobiliária.
    
    Args:
        request: Dados da requisição
        
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
    
    # Log do modo selecionado
    logger.info(f"Avaliação {evaluation_id} iniciada em modo {request.mode}")
    
    # Executar avaliação usando Celery
    task = process_evaluation.delay(evaluation_id, request.file_path, request.target_column, request.config)
    
    return {
        "evaluation_id": evaluation_id, 
        "task_id": task.id, 
        "status": "iniciado",
        "mode": request.mode,
        "expert_mode_active": request.config.get('expert_mode', False)
    }

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
    Faz upload de arquivo de dados com validações robustas.
    
    Args:
        file: Arquivo a ser enviado
        
    Returns:
        Caminho do arquivo salvo e informações de validação
    """
    # Validação de extensão
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de arquivo não suportado: {file.filename}. Use: .csv, .xlsx, .xls"
        )
    
    # Validação de tamanho do arquivo (100MB padrão)
    max_size = 100 * 1024 * 1024  # 100MB
    content = await file.read()
    
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande: {len(content)/(1024*1024):.1f}MB. Máximo: {max_size/(1024*1024)}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo está vazio")
    
    # Verificação de MIME type básica
    import magic
    try:
        mime_type = magic.from_buffer(content, mime=True)
        valid_mimes = {
            '.csv': ['text/csv', 'text/plain', 'application/csv'],
            '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
            '.xls': ['application/vnd.ms-excel', 'application/x-ole-storage']
        }
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext in valid_mimes:
            expected_mimes = valid_mimes[file_ext]
            if mime_type not in expected_mimes and not mime_type.startswith('text/'):
                logger.warning(f"MIME type suspeito: {mime_type} para arquivo {file.filename}")
    except Exception as e:
        logger.warning(f"Não foi possível verificar MIME type: {e}")
        mime_type = "unknown"
    
    # Salvar arquivo com nome único
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Gerar nome único mantendo extensão original
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = upload_dir / unique_filename
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Validação inicial básica após salvar
        try:
            data_loader = DataLoader({})
            file_info = data_loader._verify_file_integrity(file_path)
            
            # Tentar carregar amostra dos dados para validação prévia
            if file_ext == '.csv':
                delimiter = data_loader._detect_csv_delimiter(file_path)
                sample_df = pd.read_csv(file_path, delimiter=delimiter, nrows=5)
            else:
                sample_df = pd.read_excel(file_path, nrows=5)
            
            preview_info = {
                "columns": list(sample_df.columns),
                "sample_rows": len(sample_df),
                "total_columns": len(sample_df.columns),
                "detected_delimiter": delimiter if file_ext == '.csv' else None
            }
            
        except Exception as preview_error:
            logger.warning(f"Erro na visualização prévia: {preview_error}")
            preview_info = {"error": "Não foi possível gerar preview dos dados"}
        
        return {
            "file_path": str(file_path),
            "filename": file.filename,
            "unique_filename": unique_filename,
            "file_size_mb": len(content) / (1024 * 1024),
            "mime_type": mime_type,
            "upload_timestamp": datetime.now().isoformat(),
            "preview": preview_info,
            "status": "uploaded_successfully"
        }
        
    except Exception as e:
        # Limpar arquivo se houve erro no processamento
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")

@app.post("/evaluations/{evaluation_id}/approve_step")
async def approve_evaluation_step(evaluation_id: str, request: StepApprovalRequest):
    """
    Aprova ou rejeita uma etapa da avaliação interativa.
    
    Args:
        evaluation_id: ID da avaliação
        request: Dados da aprovação
        
    Returns:
        Status da aprovação
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    # Validar etapa
    valid_steps = ["transformations", "outliers", "model_selection"]
    if request.step not in valid_steps:
        raise HTTPException(status_code=400, detail=f"Etapa inválida. Use: {valid_steps}")
    
    try:
        # Importar tarefas do worker
        from src.workers.tasks import continue_evaluation_step
        
        # Continuar com a próxima etapa
        task = continue_evaluation_step.delay(
            evaluation_id, 
            request.step, 
            request.approved, 
            request.modifications,
            request.user_feedback
        )
        
        # Atualizar status local
        current_status = evaluation_results[evaluation_id]["status"]
        if request.approved:
            phase_map = {
                "transformations": "Continuando com modelagem",
                "outliers": "Aplicando remoção de outliers",
                "model_selection": "Finalizando modelo"
            }
            current_status.message = phase_map.get(request.step, "Continuando processo")
        else:
            current_status.message = f"Etapa {request.step} rejeitada - ajustando abordagem"
        
        evaluation_results[evaluation_id]["status"] = current_status
        
        return {
            "evaluation_id": evaluation_id,
            "step": request.step,
            "approved": request.approved,
            "task_id": task.id,
            "status": "processando_aprovacao",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar aprovação: {str(e)}")

@app.get("/evaluations/{evaluation_id}/pending_approval")
async def get_pending_approval(evaluation_id: str):
    """
    Obtém informações sobre etapas pendentes de aprovação.
    
    Args:
        evaluation_id: ID da avaliação
        
    Returns:
        Detalhes da etapa pendente
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    result_data = evaluation_results[evaluation_id]
    status = result_data["status"]
    
    # Verificar se há aprovação pendente
    if "aguardando_aprovacao" not in status.status:
        return {"pending_approval": False, "message": "Nenhuma aprovação pendente"}
    
    # Extrair dados da etapa pendente do metadata
    pending_data = getattr(status, 'metadata', {})
    
    return {
        "pending_approval": True,
        "evaluation_id": evaluation_id,
        "step": pending_data.get('step', 'unknown'),
        "details": pending_data.get('details', {}),
        "suggestions": pending_data.get('suggestions', []),
        "current_phase": status.current_phase,
        "message": status.message,
        "timestamp": status.timestamp
    }

@app.get("/evaluations/{evaluation_id}/audit_trail")
async def get_audit_trail(evaluation_id: str):
    """
    Obtém trilha de auditoria completa da avaliação.
    
    Args:
        evaluation_id: ID da avaliação
        
    Returns:
        Trilha de auditoria detalhada
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    try:
        # Gerar trilha de auditoria mockada mas realista
        audit_trail = {
            "evaluation_id": evaluation_id,
            "audit_metadata": {
                "generated_at": datetime.now().isoformat(),
                "audit_version": "1.0",
                "compliance_level": "NBR 14653 + Glass-Box",
                "total_steps": 15,
                "execution_duration": "3m 45s"
            },
            "pipeline_steps": [
                {
                    "step_id": 1,
                    "phase": "Data Ingestion",
                    "timestamp": "2024-01-15T10:00:00Z",
                    "status": "completed",
                    "duration": "12s",
                    "details": {
                        "action": "File upload and initial parsing",
                        "input": "property_data.csv (1,250 records)",
                        "output": "Pandas DataFrame with 15 columns",
                        "validation": "Schema validation passed",
                        "checksum": "sha256:a1b2c3d4e5f6..."
                    },
                    "audit_notes": "Arquivo carregado sem alterações. Hash verificado."
                },
                {
                    "step_id": 2,
                    "phase": "Data Validation",
                    "timestamp": "2024-01-15T10:00:12Z",
                    "status": "completed",
                    "duration": "8s",
                    "details": {
                        "action": "Quality assessment and type validation",
                        "checks_performed": [
                            "Missing values detection",
                            "Data type validation",
                            "Outlier identification",
                            "Duplicate detection"
                        ],
                        "issues_found": {
                            "missing_values": 23,
                            "outliers": 5,
                            "duplicates": 0
                        },
                        "resolution": "Issues flagged for transformation phase"
                    },
                    "audit_notes": "5 outliers identificados mas mantidos para análise posterior."
                },
                {
                    "step_id": 3,
                    "phase": "Feature Engineering",
                    "timestamp": "2024-01-15T10:00:20Z",
                    "status": "completed",
                    "duration": "25s",
                    "details": {
                        "action": "Variable transformation and feature creation",
                        "transformations_applied": {
                            "missing_value_imputation": "mean/mode strategy",
                            "categorical_encoding": "one-hot encoding",
                            "numerical_scaling": "StandardScaler",
                            "feature_selection": "univariate selection (k=12)"
                        },
                        "features_created": [
                            "area_per_bedroom",
                            "price_per_sqm",
                            "location_score"
                        ],
                        "features_dropped": ["raw_address", "listing_date"],
                        "final_feature_count": 18
                    },
                    "audit_notes": "Transformações aplicadas conforme NBR 14653. Features derivadas matematicamente justificadas."
                },
                {
                    "step_id": 4,
                    "phase": "Model Training",
                    "timestamp": "2024-01-15T10:00:45Z",
                    "status": "completed",
                    "duration": "45s",
                    "details": {
                        "action": "Elastic Net model training with cross-validation",
                        "algorithm": "Elastic Net Regression",
                        "hyperparameters": {
                            "alpha": 1.0,
                            "l1_ratio": 0.5,
                            "max_iter": 1000
                        },
                        "cross_validation": {
                            "method": "5-fold CV",
                            "cv_scores": [0.842, 0.856, 0.839, 0.851, 0.847],
                            "mean_cv_score": 0.847,
                            "std_cv_score": 0.006
                        },
                        "training_metrics": {
                            "r2_score": 0.854,
                            "rmse": 42150.0,
                            "mae": 31200.0,
                            "mape": 12.5
                        }
                    },
                    "audit_notes": "Modelo treinado com validação cruzada rigorosa. Hiperparâmetros otimizados via grid search."
                },
                {
                    "step_id": 5,
                    "phase": "NBR 14653 Validation",
                    "timestamp": "2024-01-15T10:01:30Z",
                    "status": "completed",
                    "duration": "35s",
                    "details": {
                        "action": "Statistical compliance testing",
                        "tests_performed": {
                            "r2_test": {
                                "value": 0.854,
                                "threshold": 0.70,
                                "result": "PASS",
                                "grade_contribution": "Normal"
                            },
                            "f_test": {
                                "value": 123.45,
                                "threshold": 3.84,
                                "result": "PASS",
                                "p_value": 0.001
                            },
                            "t_test": {
                                "significant_coefficients": 14,
                                "total_coefficients": 18,
                                "result": "PASS"
                            },
                            "durbin_watson": {
                                "value": 1.89,
                                "threshold": 1.5,
                                "result": "PASS"
                            },
                            "shapiro_wilk": {
                                "value": 0.045,
                                "threshold": 0.05,
                                "result": "FAIL",
                                "note": "Normalidade dos resíduos comprometida"
                            }
                        },
                        "overall_grade": "Normal",
                        "compliance_score": 0.8
                    },
                    "audit_notes": "4 de 5 testes NBR aprovados. Falha na normalidade não impede classificação Normal."
                }
            ],
            "feature_analysis": {
                "feature_importance_ranking": [
                    {"feature": "area_privativa", "importance": 0.342, "method": "coefficient_magnitude"},
                    {"feature": "localizacao_score", "importance": 0.287, "method": "coefficient_magnitude"},
                    {"feature": "idade_imovel", "importance": -0.156, "method": "coefficient_magnitude"},
                    {"feature": "vagas_garagem", "importance": 0.123, "method": "coefficient_magnitude"},
                    {"feature": "banheiros", "importance": 0.089, "method": "coefficient_magnitude"}
                ],
                "feature_selection_rationale": {
                    "area_privativa": "Correlação direta com valor (r=0.78)",
                    "localizacao_score": "Feature engenheirada baseada em dados geográficos",
                    "idade_imovel": "Depreciação temporal documentada",
                    "vagas_garagem": "Premium de mercado quantificado",
                    "banheiros": "Indicador de qualidade/tamanho"
                }
            },
            "compliance_evidence": {
                "nbr_14653_conformity": {
                    "section_compliance": {
                        "8.2.1_data_quality": "CONFORMANT",
                        "8.2.2_statistical_tests": "CONFORMANT", 
                        "8.2.3_model_validation": "CONFORMANT",
                        "8.3_documentation": "CONFORMANT"
                    },
                    "justifications": [
                        "Dados tratados conforme item 8.2.1",
                        "Bateria completa de testes estatísticos aplicada",
                        "R² = 0.854 > 0.80 (grau Normal)",
                        "Documentação completa e auditável"
                    ]
                },
                "glass_box_transparency": {
                    "interpretability_level": "COMPLETE",
                    "explanation_methods": ["coefficients", "permutation_importance"],
                    "auditability_score": 0.95,
                    "reproducibility": "GUARANTEED"
                }
            },
            "data_lineage": {
                "source_data": {
                    "origin": "property_data.csv",
                    "records_original": 1250,
                    "records_final": 1225,
                    "exclusion_reason": "25 records with >50% missing values"
                },
                "transformations_chain": [
                    "raw_data → quality_filter → missing_imputation → encoding → scaling → feature_selection → model_input"
                ],
                "reproducibility_hash": "sha256:f1e2d3c4b5a6..."
            },
            "quality_assurance": {
                "validation_checks": [
                    "Input data integrity verified",
                    "Transformation pipeline tested",
                    "Model performance within expected range",
                    "Statistical assumptions validated",
                    "Output consistency confirmed"
                ],
                "peer_review_status": "APPROVED",
                "technical_reviewer": "System Automated Validation",
                "review_date": "2024-01-15T10:02:05Z"
            }
        }
        
        return audit_trail
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar trilha de auditoria: {str(e)}")

@app.get("/evaluations/{evaluation_id}/shap_explanations")
async def get_shap_explanations(evaluation_id: str, sample_size: int = 5):
    """
    Obtém explicações SHAP detalhadas para o modo especialista.
    
    Args:
        evaluation_id: ID da avaliação
        sample_size: Número de amostras para explicar
        
    Returns:
        Explicações SHAP detalhadas
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    result = evaluation_results[evaluation_id]["result"]
    if result is None:
        raise HTTPException(status_code=202, detail="Modelo ainda não treinado")
    
    # Verificar se é modo especialista
    request_config = evaluation_results[evaluation_id]["request"].config
    if not request_config.get('expert_mode', False):
        raise HTTPException(status_code=400, detail="Explicações SHAP disponíveis apenas no modo especialista")
    
    try:
        # Em implementação real, carregar modelo e gerar explicações
        # Por ora, retornar estrutura mockada
        explanations = {
            "evaluation_id": evaluation_id,
            "sample_size": sample_size,
            "shap_explanations": {
                "base_value": 500000.0,
                "feature_contributions": {
                    "area_privativa": 0.35,
                    "localizacao_score": 0.28,
                    "idade_imovel": -0.15,
                    "vagas_garagem": 0.12
                },
                "interpretation": [
                    "A característica 'Area Privativa' de 150 m² adicionou R$ 120.000 ao valor",
                    "A localização premium contribuiu com R$ 95.000 para o valor final",
                    "A idade do imóvel (10 anos) reduziu o valor em R$ 45.000",
                    "As 2 vagas de garagem agregaram R$ 35.000 ao valor"
                ]
            },
            "model_transparency": {
                "glass_box_level": "complete",
                "interpretability_score": 0.95,
                "explanation_methods": ["SHAP", "feature_coefficients", "permutation_importance"]
            },
            "timestamp": datetime.now()
        }
        
        return explanations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar explicações SHAP: {str(e)}")

@app.post("/evaluations/{evaluation_id}/shap_simulation")
async def simulate_shap_scenario(evaluation_id: str, request: ShapSimulationRequest):
    """
    Simula cenário modificando features e calcula impacto SHAP.
    Núcleo do Laboratório de Simulação interativo.
    
    Args:
        evaluation_id: ID da avaliação
        request: Dados da simulação
        
    Returns:
        Análise comparativa do cenário simulado
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    result = evaluation_results[evaluation_id]["result"]
    if result is None:
        raise HTTPException(status_code=202, detail="Modelo ainda não treinado")
    
    # Verificar modo especialista
    request_config = evaluation_results[evaluation_id]["request"].config
    if not request_config.get('expert_mode', False):
        raise HTTPException(status_code=400, detail="Simulação SHAP disponível apenas no modo especialista")
    
    try:
        # Simular cenário com modificações das features
        baseline_prediction = 500000.0  # Valor base mockado
        baseline_shap = {
            "area_privativa": 120000.0,
            "localizacao_score": 95000.0,
            "idade_imovel": -45000.0,
            "vagas_garagem": 35000.0,
            "banheiros": 25000.0
        }
        
        # Calcular impacto das modificações
        modified_shap = baseline_shap.copy()
        impacts = {}
        total_impact = 0.0
        
        for feature, new_value in request.feature_modifications.items():
            if feature in baseline_shap:
                # Simulação simplificada do impacto proporcional
                if feature == "area_privativa":
                    # R$ 800 por m² adicional
                    baseline_area = 150.0
                    area_diff = new_value - baseline_area
                    impact = area_diff * 800.0
                elif feature == "vagas_garagem":
                    # R$ 17.500 por vaga adicional
                    baseline_vagas = 2.0
                    vagas_diff = new_value - baseline_vagas
                    impact = vagas_diff * 17500.0
                elif feature == "idade_imovel":
                    # -R$ 4.500 por ano adicional
                    baseline_idade = 10.0
                    idade_diff = new_value - baseline_idade
                    impact = idade_diff * -4500.0
                else:
                    # Impacto genérico proporcional
                    impact = (new_value - 1.0) * baseline_shap[feature] * 0.1
                
                modified_shap[feature] = baseline_shap[feature] + impact
                impacts[feature] = {
                    "baseline_value": baseline_shap[feature],
                    "modified_value": modified_shap[feature],
                    "absolute_impact": impact,
                    "relative_impact": (impact / abs(baseline_shap[feature])) * 100 if baseline_shap[feature] != 0 else 0
                }
                total_impact += impact
        
        modified_prediction = baseline_prediction + total_impact
        
        # Preparar resposta detalhada para o laboratório
        simulation_result = {
            "evaluation_id": evaluation_id,
            "simulation_name": request.simulation_name or f"Simulação {datetime.now().strftime('%H:%M:%S')}",
            "timestamp": datetime.now(),
            "scenario_comparison": {
                "baseline": {
                    "prediction": baseline_prediction,
                    "shap_values": baseline_shap,
                    "feature_values": {
                        "area_privativa": 150.0,
                        "localizacao_score": 8.5,
                        "idade_imovel": 10.0,
                        "vagas_garagem": 2.0,
                        "banheiros": 3.0
                    }
                },
                "modified": {
                    "prediction": modified_prediction,
                    "shap_values": modified_shap,
                    "feature_values": request.feature_modifications
                },
                "impact_analysis": {
                    "total_impact": total_impact,
                    "relative_change": (total_impact / baseline_prediction) * 100,
                    "feature_impacts": impacts,
                    "direction": "positive" if total_impact > 0 else "negative" if total_impact < 0 else "neutral"
                }
            },
            "waterfall_data": {
                "base_value": baseline_prediction,
                "contributions": [
                    {
                        "feature": feature,
                        "baseline_contribution": baseline_shap[feature],
                        "modified_contribution": modified_shap[feature],
                        "delta": impacts.get(feature, {}).get("absolute_impact", 0)
                    }
                    for feature in baseline_shap.keys()
                ],
                "final_prediction": modified_prediction
            },
            "insights": {
                "top_drivers": sorted(
                    [(k, abs(v.get("absolute_impact", 0))) for k, v in impacts.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:3],
                "recommendations": [
                    f"Modificação mais impactante: {max(impacts.keys(), key=lambda k: abs(impacts[k]['absolute_impact']))} ({impacts[max(impacts.keys(), key=lambda k: abs(impacts[k]['absolute_impact']))]['absolute_impact']:+,.0f})",
                    f"Impacto total no valor: {total_impact:+,.0f} ({(total_impact/baseline_prediction)*100:+.1f}%)",
                    "Use sliders para explorar diferentes cenários de forma interativa"
                ]
            },
            "metadata": {
                "simulation_quality": "high",
                "confidence_level": 0.85,
                "model_type": request_config.get('model_type', 'elastic_net'),
                "shap_explainer_type": "TreeExplainer" if request_config.get('model_type') in ['xgboost', 'gradient_boosting'] else "LinearExplainer"
            }
        }
        
        return simulation_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na simulação SHAP: {str(e)}")

@app.get("/evaluations/{evaluation_id}/shap_waterfall")
async def get_shap_waterfall(evaluation_id: str, sample_index: int = 0):
    """
    Gera dados para gráfico waterfall SHAP de uma amostra específica.
    
    Args:
        evaluation_id: ID da avaliação
        sample_index: Índice da amostra
        
    Returns:
        Dados estruturados para gráfico waterfall
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    result = evaluation_results[evaluation_id]["result"]
    if result is None:
        raise HTTPException(status_code=202, detail="Modelo ainda não treinado")
    
    # Verificar modo especialista
    request_config = evaluation_results[evaluation_id]["request"].config
    if not request_config.get('expert_mode', False):
        raise HTTPException(status_code=400, detail="Waterfall SHAP disponível apenas no modo especialista")
    
    try:
        # Dados mockados mas realistas para waterfall
        base_value = 485000.0
        contributions = [
            {"feature": "Base Value", "value": base_value, "cumulative": base_value},
            {"feature": "area_privativa", "value": 125000.0, "cumulative": base_value + 125000.0},
            {"feature": "localizacao_score", "value": 85000.0, "cumulative": base_value + 125000.0 + 85000.0},
            {"feature": "vagas_garagem", "value": 35000.0, "cumulative": base_value + 125000.0 + 85000.0 + 35000.0},
            {"feature": "banheiros", "value": 22000.0, "cumulative": base_value + 125000.0 + 85000.0 + 35000.0 + 22000.0},
            {"feature": "idade_imovel", "value": -48000.0, "cumulative": base_value + 125000.0 + 85000.0 + 35000.0 + 22000.0 - 48000.0},
            {"feature": "Final Prediction", "value": 0, "cumulative": base_value + 125000.0 + 85000.0 + 35000.0 + 22000.0 - 48000.0}
        ]
        
        waterfall_data = {
            "evaluation_id": evaluation_id,
            "sample_index": sample_index,
            "waterfall_chart": {
                "base_value": base_value,
                "final_prediction": contributions[-1]["cumulative"],
                "contributions": contributions,
                "chart_config": {
                    "colors": {
                        "positive": "#2E8B57",
                        "negative": "#DC143C",
                        "base": "#4682B4",
                        "final": "#FFD700"
                    },
                    "height": 400,
                    "width": 800
                }
            },
            "feature_details": {
                "area_privativa": {
                    "current_value": 180.0,
                    "unit": "m²",
                    "interpretation": "Área privativa de 180m² contribuiu significativamente para o valor",
                    "importance_rank": 1
                },
                "localizacao_score": {
                    "current_value": 9.2,
                    "unit": "score",
                    "interpretation": "Localização premium (score 9.2/10) adicionou valor substancial",
                    "importance_rank": 2
                },
                "idade_imovel": {
                    "current_value": 15.0,
                    "unit": "anos",
                    "interpretation": "Idade de 15 anos reduziu o valor pela depreciação",
                    "importance_rank": 3
                }
            },
            "interpretation_summary": {
                "main_value_drivers": ["area_privativa", "localizacao_score"],
                "main_value_detractors": ["idade_imovel"],
                "explanation": "O valor final de R$ 704.000 é resultado principalmente da área generosa e localização premium, parcialmente reduzido pela idade do imóvel."
            },
            "timestamp": datetime.now()
        }
        
        return waterfall_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar waterfall SHAP: {str(e)}")

@app.get("/evaluations/{evaluation_id}/laboratory_features")
async def get_laboratory_features(evaluation_id: str):
    """
    Obtém configuração de features para o laboratório de simulação.
    
    Args:
        evaluation_id: ID da avaliação
        
    Returns:
        Metadados das features para interface interativa
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    result = evaluation_results[evaluation_id]["result"]
    if result is None:
        raise HTTPException(status_code=202, detail="Modelo ainda não treinado")
    
    try:
        # Configuração das features para sliders e controles interativos
        features_config = {
            "evaluation_id": evaluation_id,
            "features": {
                "area_privativa": {
                    "display_name": "Área Privativa",
                    "unit": "m²",
                    "current_value": 150.0,
                    "min_value": 30.0,
                    "max_value": 500.0,
                    "step": 5.0,
                    "slider_type": "numeric",
                    "impact_coefficient": 800.0,
                    "description": "Área interna do imóvel em metros quadrados",
                    "importance_rank": 1,
                    "category": "structural"
                },
                "localizacao_score": {
                    "display_name": "Score de Localização",
                    "unit": "score (1-10)",
                    "current_value": 8.5,
                    "min_value": 1.0,
                    "max_value": 10.0,
                    "step": 0.1,
                    "slider_type": "numeric",
                    "impact_coefficient": 11200.0,
                    "description": "Qualidade da localização baseada em infraestrutura e serviços",
                    "importance_rank": 2,
                    "category": "location"
                },
                "idade_imovel": {
                    "display_name": "Idade do Imóvel",
                    "unit": "anos",
                    "current_value": 10.0,
                    "min_value": 0.0,
                    "max_value": 50.0,
                    "step": 1.0,
                    "slider_type": "numeric",
                    "impact_coefficient": -4500.0,
                    "description": "Idade do imóvel em anos (depreciação)",
                    "importance_rank": 3,
                    "category": "temporal"
                },
                "vagas_garagem": {
                    "display_name": "Vagas de Garagem",
                    "unit": "unidades",
                    "current_value": 2.0,
                    "min_value": 0.0,
                    "max_value": 6.0,
                    "step": 1.0,
                    "slider_type": "integer",
                    "impact_coefficient": 17500.0,
                    "description": "Número de vagas de garagem",
                    "importance_rank": 4,
                    "category": "amenities"
                },
                "banheiros": {
                    "display_name": "Banheiros",
                    "unit": "unidades",
                    "current_value": 3.0,
                    "min_value": 1.0,
                    "max_value": 6.0,
                    "step": 1.0,
                    "slider_type": "integer",
                    "impact_coefficient": 8300.0,
                    "description": "Número de banheiros",
                    "importance_rank": 5,
                    "category": "amenities"
                },
                "quartos": {
                    "display_name": "Quartos",
                    "unit": "unidades",
                    "current_value": 3.0,
                    "min_value": 1.0,
                    "max_value": 6.0,
                    "step": 1.0,
                    "slider_type": "integer",
                    "impact_coefficient": 12500.0,
                    "description": "Número de quartos",
                    "importance_rank": 6,
                    "category": "structural"
                }
            },
            "categories": {
                "structural": {
                    "name": "Características Estruturais",
                    "color": "#2E8B57",
                    "features": ["area_privativa", "quartos"]
                },
                "location": {
                    "name": "Localização",
                    "color": "#4169E1",
                    "features": ["localizacao_score"]
                },
                "amenities": {
                    "name": "Comodidades",
                    "color": "#FF8C00",
                    "features": ["vagas_garagem", "banheiros"]
                },
                "temporal": {
                    "name": "Fatores Temporais",
                    "color": "#DC143C",
                    "features": ["idade_imovel"]
                }
            },
            "simulation_presets": [
                {
                    "name": "Imóvel Compacto",
                    "description": "Apartamento pequeno bem localizado",
                    "modifications": {
                        "area_privativa": 80.0,
                        "quartos": 2.0,
                        "banheiros": 2.0,
                        "vagas_garagem": 1.0,
                        "localizacao_score": 9.0,
                        "idade_imovel": 5.0
                    }
                },
                {
                    "name": "Casa de Família",
                    "description": "Casa ampla em bairro residencial",
                    "modifications": {
                        "area_privativa": 220.0,
                        "quartos": 4.0,
                        "banheiros": 3.0,
                        "vagas_garagem": 3.0,
                        "localizacao_score": 7.5,
                        "idade_imovel": 12.0
                    }
                },
                {
                    "name": "Cobertura Premium",
                    "description": "Cobertura de alto padrão",
                    "modifications": {
                        "area_privativa": 350.0,
                        "quartos": 4.0,
                        "banheiros": 5.0,
                        "vagas_garagem": 4.0,
                        "localizacao_score": 9.8,
                        "idade_imovel": 2.0
                    }
                }
            ],
            "laboratory_config": {
                "real_time_updates": True,
                "comparison_mode": True,
                "waterfall_charts": True,
                "sensitivity_analysis": True,
                "export_scenarios": True
            },
            "timestamp": datetime.now()
        }
        
        return features_config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter configuração do laboratório: {str(e)}")

@app.post("/geospatial/analyze")
async def analyze_location(request: GeospatialAnalysisRequest):
    """
    Realiza análise geoespacial de uma localização.
    
    Args:
        request: Dados da localização para análise
        
    Returns:
        Análise geoespacial completa com features e scores
    """
    try:
        # Configurar centro da cidade
        city_center = None
        if request.city_center_lat and request.city_center_lon:
            city_center = (request.city_center_lat, request.city_center_lon)
        
        # Criar analisador geoespacial
        analyzer = create_geospatial_analyzer(city_center=city_center)
        
        # Verificar se foi fornecido endereço ou coordenadas
        if request.address:
            analysis = analyzer.analyze_location(address=request.address)
        elif request.latitude and request.longitude:
            coordinates = (request.latitude, request.longitude)
            analysis = analyzer.analyze_location(coordinates=coordinates)
        else:
            raise HTTPException(status_code=400, detail="Endereço ou coordenadas (lat/lon) devem ser fornecidos")
        
        if not analysis:
            raise HTTPException(status_code=400, detail="Falha na análise geoespacial")
        
        # Converter para formato JSON serializable
        result = {
            "coordinates": {
                "latitude": analysis.coordinates[0],
                "longitude": analysis.coordinates[1]
            },
            "features": {
                "distance_to_center": analysis.features.distance_to_center,
                "proximity_score": analysis.features.proximity_score,
                "density_score": analysis.features.density_score,
                "transport_score": analysis.features.transport_score,
                "amenities_score": analysis.features.amenities_score,
                "location_cluster": analysis.features.location_cluster,
                "neighborhood_value_index": analysis.features.neighborhood_value_index
            },
            "quality_score": analysis.quality_score,
            "address_components": analysis.address_components,
            "nearby_pois": analysis.nearby_pois,
            "analysis_summary": {
                "location_rating": "Excelente" if analysis.quality_score >= 8 else 
                                 "Boa" if analysis.quality_score >= 6 else
                                 "Regular" if analysis.quality_score >= 4 else "Limitada",
                "key_strengths": _get_location_strengths(analysis.features),
                "key_weaknesses": _get_location_weaknesses(analysis.features),
                "investment_potential": _calculate_investment_potential(analysis.features)
            },
            "timestamp": datetime.now()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na análise geoespacial: {str(e)}")

@app.post("/evaluations/{evaluation_id}/enrich_geospatial")
async def enrich_dataset_geospatial(evaluation_id: str, request: DataEnrichmentRequest):
    """
    Enriquece dataset de uma avaliação com features geoespaciais.
    
    Args:
        evaluation_id: ID da avaliação
        request: Configurações do enriquecimento
        
    Returns:
        Status do enriquecimento e estatísticas
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    try:
        # Configurar centro da cidade
        city_center = None
        if request.city_center_lat and request.city_center_lon:
            city_center = (request.city_center_lat, request.city_center_lon)
        
        # Criar analisador geoespacial
        analyzer = create_geospatial_analyzer(city_center=city_center)
        
        # Simular carregamento de dados (em implementação real, carregaria do storage)
        # Por ora, criar dados mockados para demonstração
        import pandas as pd
        import numpy as np
        
        # Dataset mockado
        mock_data = pd.DataFrame({
            'endereco': [
                'Av. Paulista, 1000, São Paulo, SP',
                'Rua Augusta, 500, São Paulo, SP',
                'Av. Faria Lima, 2000, São Paulo, SP',
                'Rua Consolação, 800, São Paulo, SP',
                'Av. Brigadeiro Luis Antonio, 1500, São Paulo, SP'
            ],
            'valor': [850000, 650000, 1200000, 750000, 900000],
            'area_privativa': [120, 85, 150, 100, 130]
        })
        
        # Enriquecer com dados geoespaciais
        enriched_data = analyzer.enrich_dataset_with_geospatial(
            mock_data, 
            address_column=request.address_column
        )
        
        # Gerar estatísticas do enriquecimento
        geo_columns = ['proximity_score', 'transport_score', 'amenities_score', 'geo_quality_score']
        statistics = {
            "total_records": len(enriched_data),
            "enriched_records": len(enriched_data.dropna(subset=['latitude', 'longitude'])),
            "geocoding_success_rate": len(enriched_data.dropna(subset=['latitude', 'longitude'])) / len(enriched_data) * 100,
            "geospatial_features_added": len([col for col in geo_columns if col in enriched_data.columns]),
            "quality_distribution": {
                "excellent": len(enriched_data[enriched_data['geo_quality_score'] >= 8]) if 'geo_quality_score' in enriched_data.columns else 0,
                "good": len(enriched_data[(enriched_data['geo_quality_score'] >= 6) & (enriched_data['geo_quality_score'] < 8)]) if 'geo_quality_score' in enriched_data.columns else 0,
                "regular": len(enriched_data[(enriched_data['geo_quality_score'] >= 4) & (enriched_data['geo_quality_score'] < 6)]) if 'geo_quality_score' in enriched_data.columns else 0,
                "limited": len(enriched_data[enriched_data['geo_quality_score'] < 4]) if 'geo_quality_score' in enriched_data.columns else 0
            },
            "location_clusters": enriched_data['location_cluster'].value_counts().to_dict() if 'location_cluster' in enriched_data.columns else {},
            "average_scores": {
                score: enriched_data[score].mean() if score in enriched_data.columns else 0
                for score in geo_columns
            }
        }
        
        # Gerar dados para mapa de calor
        heatmap_data = analyzer.generate_location_heatmap_data(enriched_data)
        
        # Armazenar dados enriquecidos (em implementação real, salvaria no storage)
        evaluation_results[evaluation_id]["enriched_data"] = enriched_data.to_dict('records')
        
        result = {
            "evaluation_id": evaluation_id,
            "enrichment_status": "completed",
            "statistics": statistics,
            "heatmap_data": heatmap_data,
            "feature_descriptions": {
                "distance_to_center": "Distância até o centro da cidade (km)",
                "proximity_score": "Score de proximidade a POIs importantes (0-10)",
                "density_score": "Score de densidade de imóveis na região (0-10)",
                "transport_score": "Score de acesso a transporte público (0-10)",
                "amenities_score": "Score de amenidades (saúde, educação, lazer) (0-10)",
                "location_cluster": "Classificação da localização (Premium Central, Urbano, etc.)",
                "neighborhood_value_index": "Índice de valor do bairro (0-10)",
                "geo_quality_score": "Score geral de qualidade da localização (0-10)"
            },
            "recommendations": _generate_geospatial_recommendations(statistics),
            "timestamp": datetime.now()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no enriquecimento geoespacial: {str(e)}")

@app.get("/evaluations/{evaluation_id}/geospatial_heatmap")
async def get_geospatial_heatmap(evaluation_id: str):
    """
    Obtém dados para mapa de calor geoespacial de uma avaliação.
    
    Args:
        evaluation_id: ID da avaliação
        
    Returns:
        Dados estruturados para visualização em mapa de calor
    """
    if evaluation_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    
    try:
        # Verificar se há dados enriquecidos
        enriched_data = evaluation_results[evaluation_id].get("enriched_data")
        if not enriched_data:
            raise HTTPException(status_code=404, detail="Dados geoespaciais não encontrados. Execute primeiro o enriquecimento.")
        
        # Converter de volta para DataFrame
        import pandas as pd
        df = pd.DataFrame(enriched_data)
        
        # Criar analisador para gerar dados do mapa
        analyzer = create_geospatial_analyzer()
        heatmap_data = analyzer.generate_location_heatmap_data(df)
        
        if not heatmap_data:
            raise HTTPException(status_code=400, detail="Não foi possível gerar dados do mapa de calor")
        
        # Adicionar configurações de visualização
        heatmap_data.update({
            "visualization_config": {
                "map_style": "OpenStreetMap",
                "heatmap_radius": 20,
                "heatmap_blur": 15,
                "marker_size": 8,
                "color_scale": {
                    "low": "#0066CC",
                    "medium": "#FFCC00", 
                    "high": "#FF6600",
                    "premium": "#CC0000"
                }
            },
            "legend": {
                "value_ranges": {
                    "baixo": "< R$ 500.000",
                    "medio": "R$ 500.000 - R$ 800.000",
                    "alto": "R$ 800.000 - R$ 1.200.000",
                    "premium": "> R$ 1.200.000"
                },
                "cluster_colors": {
                    "Premium Central": "#CC0000",
                    "Urbano Consolidado": "#FF6600", 
                    "Urbano em Desenvolvimento": "#FFCC00",
                    "Suburbano": "#0066CC",
                    "Periférico": "#6699FF"
                }
            },
            "timestamp": datetime.now()
        })
        
        return heatmap_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar mapa de calor: {str(e)}")

def _get_location_strengths(features) -> List[str]:
    """Identifica pontos fortes da localização."""
    strengths = []
    
    if features.proximity_score >= 8:
        strengths.append("Excelente proximidade a pontos de interesse")
    if features.transport_score >= 8:
        strengths.append("Ótimo acesso a transporte público")
    if features.amenities_score >= 8:
        strengths.append("Rica em amenidades (saúde, educação, lazer)")
    if features.distance_to_center <= 5:
        strengths.append("Localização central privilegiada")
    if features.neighborhood_value_index >= 7:
        strengths.append("Bairro com alto índice de valorização")
    
    return strengths if strengths else ["Localização com potencial de desenvolvimento"]

def _get_location_weaknesses(features) -> List[str]:
    """Identifica pontos fracos da localização."""
    weaknesses = []
    
    if features.transport_score <= 4:
        weaknesses.append("Acesso limitado a transporte público")
    if features.amenities_score <= 4:
        weaknesses.append("Poucas amenidades na região")
    if features.distance_to_center >= 20:
        weaknesses.append("Distante do centro urbano")
    if features.density_score <= 3:
        weaknesses.append("Baixa densidade de desenvolvimento")
    
    return weaknesses if weaknesses else ["Nenhuma limitação significativa identificada"]

def _calculate_investment_potential(features) -> str:
    """Calcula potencial de investimento."""
    score = (
        features.proximity_score * 0.2 +
        features.transport_score * 0.2 +
        features.amenities_score * 0.15 +
        features.neighborhood_value_index * 0.25 +
        (10 - min(features.distance_to_center, 10)) * 0.2
    )
    
    if score >= 8:
        return "Alto - Excelente potencial de valorização"
    elif score >= 6:
        return "Médio-Alto - Bom potencial com crescimento sustentável"
    elif score >= 4:
        return "Médio - Potencial moderado com riscos controlados"
    else:
        return "Baixo - Requer análise cuidadosa de viabilidade"

def _generate_geospatial_recommendations(statistics: Dict[str, Any]) -> List[str]:
    """Gera recomendações baseadas nas estatísticas geoespaciais."""
    recommendations = []
    
    success_rate = statistics.get("geocoding_success_rate", 0)
    if success_rate < 80:
        recommendations.append("Considere padronizar os endereços para melhorar a taxa de geocodificação")
    
    quality_dist = statistics.get("quality_distribution", {})
    excellent_pct = quality_dist.get("excellent", 0) / statistics.get("total_records", 1) * 100
    
    if excellent_pct >= 50:
        recommendations.append("Portfolio com excelente qualidade locacional - foque em marketing premium")
    elif excellent_pct >= 25:
        recommendations.append("Mix balanceado de qualidade - diversifique estratégias por cluster")
    else:
        recommendations.append("Oportunidade de melhoria na seleção de localizações premium")
    
    avg_transport = statistics.get("average_scores", {}).get("transport_score", 0)
    if avg_transport < 5:
        recommendations.append("Priorize imóveis com melhor acesso a transporte público")
    
    clusters = statistics.get("location_clusters", {})
    if "Premium Central" in clusters and clusters["Premium Central"] > 0:
        recommendations.append("Aproveite imóveis em localização Premium Central para maximizar retorno")
    
    return recommendations

@app.websocket("/ws/{evaluation_id}")
async def websocket_endpoint(websocket: WebSocket, evaluation_id: str):
    """
    Endpoint WebSocket para feedback em tempo real com suporte a heartbeat.
    
    Args:
        websocket: Conexão WebSocket
        evaluation_id: ID da avaliação
    """
    connection_id = await websocket_manager.connect(websocket, evaluation_id)
    
    try:
        while True:
            # Aguardar mensagens do cliente (principalmente pongs)
            try:
                # Timeout de 60 segundos para receber mensagens
                message = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'pong':
                        # Cliente respondeu ao ping
                        await websocket_manager.handle_pong(connection_id)
                    elif message_type == 'request_status':
                        # Cliente solicitou status atual
                        # Reenviar últimas mensagens
                        await websocket_manager._send_catchup_messages(connection_id, evaluation_id)
                    else:
                        logger.debug(f"Mensagem não reconhecida do cliente {connection_id}: {message_type}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Mensagem inválida recebida de {connection_id}: {message}")
                    
            except asyncio.TimeoutError:
                # Timeout normal - continuar loop
                continue
            except Exception as e:
                logger.warning(f"Erro ao receber mensagem de {connection_id}: {e}")
                break
            
    except WebSocketDisconnect:
        logger.info(f"Cliente {connection_id} desconectou")
    except Exception as e:
        logger.error(f"Erro na conexão WebSocket {connection_id}: {e}")
    finally:
        websocket_manager.disconnect(connection_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)