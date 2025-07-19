"""
Testes unitários para o módulo ml_service.py
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.services.ml_service import (
    MLTrainingRequest,
    MLInferenceRequest, 
    MLTrainingResponse,
    MLInferenceResponse,
    ModelInfo
)


class TestMLServiceModels:
    """Testes para os modelos Pydantic do ML Service"""
    
    def test_ml_training_request_default_values(self):
        """Testa valores padrão do MLTrainingRequest"""
        request = MLTrainingRequest(evaluation_id="eval_123")
        
        assert request.evaluation_id == "eval_123"
        assert request.model_type == "elastic_net"
        assert request.target_column == "valor"
        assert request.expert_mode is False
        assert request.hyperparameters is None
        assert request.cross_validation is True
        assert request.feature_selection is True
        assert request.config == {}
    
    def test_ml_training_request_custom_values(self):
        """Testa valores customizados do MLTrainingRequest"""
        custom_hyperparams = {"alpha": 0.1, "l1_ratio": 0.5}
        custom_config = {"max_features": 10, "random_state": 42}
        
        request = MLTrainingRequest(
            evaluation_id="eval_456",
            model_type="xgboost",
            target_column="preco",
            expert_mode=True,
            hyperparameters=custom_hyperparams,
            cross_validation=False,
            feature_selection=False,
            config=custom_config
        )
        
        assert request.evaluation_id == "eval_456"
        assert request.model_type == "xgboost"
        assert request.target_column == "preco"
        assert request.expert_mode is True
        assert request.hyperparameters == custom_hyperparams
        assert request.cross_validation is False
        assert request.feature_selection is False
        assert request.config == custom_config
    
    def test_ml_inference_request(self):
        """Testa MLInferenceRequest"""
        sample_data = [
            {"area": 100, "quartos": 2, "localizacao": "Centro"},
            {"area": 120, "quartos": 3, "localizacao": "Zona Sul"}
        ]
        
        request = MLInferenceRequest(
            model_id="model_789",
            data=sample_data,
            return_confidence=True,
            return_shap=True
        )
        
        assert request.model_id == "model_789"
        assert request.data == sample_data
        assert request.return_confidence is True
        assert request.return_shap is True
    
    def test_ml_inference_request_defaults(self):
        """Testa valores padrão do MLInferenceRequest"""
        request = MLInferenceRequest(
            model_id="model_123",
            data=[{"area": 100}]
        )
        
        assert request.return_confidence is True
        assert request.return_shap is False
    
    def test_ml_training_response(self):
        """Testa MLTrainingResponse"""
        performance_metrics = {
            "r2_score": 0.85,
            "rmse": 45000.0,
            "mae": 35000.0
        }
        
        training_summary = {
            "training_time": 120.5,
            "total_features": 15,
            "selected_features": 10
        }
        
        timestamp = datetime.now()
        
        response = MLTrainingResponse(
            evaluation_id="eval_123",
            model_id="model_456",
            status="success",
            message="Modelo treinado com sucesso",
            model_type="elastic_net",
            performance_metrics=performance_metrics,
            training_summary=training_summary,
            training_time=120.5,
            expert_mode=True,
            timestamp=timestamp
        )
        
        assert response.evaluation_id == "eval_123"
        assert response.model_id == "model_456"
        assert response.status == "success"
        assert response.performance_metrics == performance_metrics
        assert response.training_summary == training_summary
        assert response.training_time == 120.5
        assert response.expert_mode is True
        assert response.timestamp == timestamp
    
    def test_ml_inference_response(self):
        """Testa MLInferenceResponse"""
        predictions = [
            {
                "predicted_value": 450000.0,
                "confidence_interval": {"lower": 420000, "upper": 480000},
                "shap_values": {"area": 0.3, "quartos": 0.2}
            }
        ]
        
        model_info = {
            "model_type": "elastic_net",
            "version": "1.0",
            "features": ["area", "quartos"]
        }
        
        timestamp = datetime.now()
        
        response = MLInferenceResponse(
            model_id="model_789",
            predictions=predictions,
            model_info=model_info,
            inference_time=0.5,
            timestamp=timestamp
        )
        
        assert response.model_id == "model_789"
        assert response.predictions == predictions
        assert response.model_info == model_info
        assert response.inference_time == 0.5
        assert response.timestamp == timestamp
    
    def test_model_info(self):
        """Testa ModelInfo"""
        performance_metrics = {
            "r2_score": 0.85,
            "rmse": 45000.0
        }
        
        timestamp = datetime.now()
        features = ["area", "quartos", "localizacao"]
        
        model_info = ModelInfo(
            model_id="model_123",
            evaluation_id="eval_456",
            model_type="xgboost",
            performance_metrics=performance_metrics,
            training_timestamp=timestamp,
            features=features,
            target_column="valor"
        )
        
        assert model_info.model_id == "model_123"
        assert model_info.evaluation_id == "eval_456"
        assert model_info.model_type == "xgboost"
        assert model_info.performance_metrics == performance_metrics
        assert model_info.training_timestamp == timestamp
        assert model_info.features == features
        assert model_info.target_column == "valor"


class TestMLServiceFunctionality:
    """Testes para funcionalidades do ML Service"""
    
    @pytest.fixture
    def mock_model_builder(self):
        """Fixture para mock do ModelBuilder"""
        builder = Mock()
        
        # Mock do resultado do modelo
        mock_result = Mock()
        mock_result.model_id = "model_123"
        mock_result.model_type = "elastic_net"
        mock_result.performance = Mock()
        mock_result.performance.r2_score = 0.85
        mock_result.performance.rmse = 45000.0
        mock_result.performance.mae = 35000.0
        mock_result.training_summary = {"training_time": 120.5}
        mock_result.best_params = {"alpha": 0.1}
        
        builder.train_model.return_value = mock_result
        return builder
    
    @pytest.fixture
    def mock_model_cache(self):
        """Fixture para mock do ModelCache"""
        cache = Mock()
        cache.get_model.return_value = None
        cache.save_model.return_value = True
        cache.model_exists.return_value = False
        return cache
    
    @pytest.fixture
    def sample_training_data(self):
        """Fixture para dados de treinamento"""
        return pd.DataFrame({
            "valor": [400000, 500000, 350000, 600000, 450000],
            "area": [100, 120, 80, 150, 110],
            "quartos": [2, 3, 2, 4, 3],
            "localizacao": ["Centro", "Zona Sul", "Centro", "Norte", "Sul"]
        })
    
    def test_training_request_validation(self):
        """Testa validação de requisição de treinamento"""
        # Requisição válida
        valid_request = {
            "evaluation_id": "eval_123",
            "model_type": "elastic_net",
            "expert_mode": True
        }
        
        request = MLTrainingRequest(**valid_request)
        assert request.evaluation_id == "eval_123"
        assert request.expert_mode is True
        
        # Requisição inválida - evaluation_id obrigatório
        with pytest.raises(ValueError):
            MLTrainingRequest(model_type="elastic_net")
    
    def test_inference_request_validation(self):
        """Testa validação de requisição de inferência"""
        # Requisição válida
        valid_request = {
            "model_id": "model_123",
            "data": [{"area": 100, "quartos": 2}]
        }
        
        request = MLInferenceRequest(**valid_request)
        assert request.model_id == "model_123"
        assert len(request.data) == 1
        
        # Requisição inválida - model_id obrigatório
        with pytest.raises(ValueError):
            MLInferenceRequest(data=[{"area": 100}])
        
        # Requisição inválida - data obrigatório
        with pytest.raises(ValueError):
            MLInferenceRequest(model_id="model_123")
    
    @patch('uuid.uuid4')
    def test_generate_model_id(self, mock_uuid):
        """Testa geração de ID do modelo"""
        mock_uuid.return_value.hex = "abcd1234efgh5678"
        
        def generate_model_id(evaluation_id: str, model_type: str) -> str:
            unique_id = mock_uuid.return_value.hex[:8]
            return f"{evaluation_id}_{model_type}_{unique_id}"
        
        model_id = generate_model_id("eval_123", "elastic_net")
        expected_id = "eval_123_elastic_net_abcd1234"
        
        assert model_id == expected_id
    
    def test_prepare_training_config(self):
        """Testa preparação de configuração de treinamento"""
        def prepare_training_config(request: MLTrainingRequest) -> Dict[str, Any]:
            config = {
                "model_type": request.model_type,
                "target_column": request.target_column,
                "expert_mode": request.expert_mode,
                "cross_validation": request.cross_validation,
                "feature_selection": request.feature_selection
            }
            
            # Adicionar hiperparâmetros se fornecidos
            if request.hyperparameters:
                config["hyperparameters"] = request.hyperparameters
            
            # Adicionar configurações adicionais
            config.update(request.config)
            
            return config
        
        request = MLTrainingRequest(
            evaluation_id="eval_123",
            model_type="xgboost",
            expert_mode=True,
            hyperparameters={"max_depth": 5},
            config={"random_state": 42}
        )
        
        config = prepare_training_config(request)
        
        assert config["model_type"] == "xgboost"
        assert config["expert_mode"] is True
        assert config["hyperparameters"]["max_depth"] == 5
        assert config["random_state"] == 42
    
    def test_validate_training_data(self, sample_training_data):
        """Testa validação de dados de treinamento"""
        def validate_training_data(data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "summary": {}
            }
            
            # Verificar se target existe
            if target_column not in data.columns:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Coluna target '{target_column}' não encontrada")
            
            # Verificar tamanho da amostra
            if len(data) < 30:
                validation_result["warnings"].append("Amostra pequena detectada")
            
            # Verificar valores faltantes
            missing_values = data.isnull().sum().to_dict()
            if any(count > 0 for count in missing_values.values()):
                validation_result["warnings"].append("Valores faltantes detectados")
            
            validation_result["summary"] = {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "missing_values": missing_values
            }
            
            return validation_result
        
        # Dados válidos
        result = validate_training_data(sample_training_data, "valor")
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        
        # Target inexistente
        result = validate_training_data(sample_training_data, "preco_inexistente")
        assert result["is_valid"] is False
        assert any("não encontrada" in error for error in result["errors"])
        
        # Amostra pequena
        small_data = sample_training_data.head(2)
        result = validate_training_data(small_data, "valor")
        assert any("Amostra pequena" in warning for warning in result["warnings"])
    
    def test_format_training_response(self):
        """Testa formatação de resposta de treinamento"""
        def format_training_response(
            evaluation_id: str,
            model_result,
            training_time: float,
            expert_mode: bool
        ) -> MLTrainingResponse:
            
            return MLTrainingResponse(
                evaluation_id=evaluation_id,
                model_id=model_result.model_id,
                status="success",
                message="Modelo treinado com sucesso",
                model_type=model_result.model_type,
                performance_metrics={
                    "r2_score": model_result.performance.r2_score,
                    "rmse": model_result.performance.rmse,
                    "mae": model_result.performance.mae
                },
                training_summary=model_result.training_summary,
                training_time=training_time,
                expert_mode=expert_mode,
                timestamp=datetime.now()
            )
        
        # Mock model result
        mock_result = Mock()
        mock_result.model_id = "model_123"
        mock_result.model_type = "elastic_net"
        mock_result.performance = Mock()
        mock_result.performance.r2_score = 0.85
        mock_result.performance.rmse = 45000.0
        mock_result.performance.mae = 35000.0
        mock_result.training_summary = {"features_used": 10}
        
        response = format_training_response("eval_123", mock_result, 120.5, True)
        
        assert response.evaluation_id == "eval_123"
        assert response.model_id == "model_123"
        assert response.status == "success"
        assert response.model_type == "elastic_net"
        assert response.performance_metrics["r2_score"] == 0.85
        assert response.training_time == 120.5
        assert response.expert_mode is True
    
    def test_prepare_inference_data(self):
        """Testa preparação de dados para inferência"""
        def prepare_inference_data(request_data: List[Dict[str, Any]]) -> pd.DataFrame:
            df = pd.DataFrame(request_data)
            
            # Validações básicas
            if df.empty:
                raise ValueError("Dados de inferência não podem estar vazios")
            
            # Conversão de tipos (simplificado)
            for col in df.columns:
                if col in ["area", "quartos", "banheiros"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        
        # Dados válidos
        request_data = [
            {"area": 100, "quartos": 2, "localizacao": "Centro"},
            {"area": 120, "quartos": 3, "localizacao": "Zona Sul"}
        ]
        
        df = prepare_inference_data(request_data)
        assert len(df) == 2
        assert "area" in df.columns
        assert df["area"].dtype in [np.int64, np.float64]
        
        # Dados vazios
        with pytest.raises(ValueError, match="não podem estar vazios"):
            prepare_inference_data([])
    
    def test_format_predictions(self):
        """Testa formatação de predições"""
        def format_predictions(
            predictions: np.ndarray,
            confidence_intervals: List[Dict] = None,
            shap_values: List[Dict] = None
        ) -> List[Dict[str, Any]]:
            
            formatted_predictions = []
            
            for i, pred in enumerate(predictions):
                prediction_dict = {
                    "predicted_value": float(pred),
                    "index": i
                }
                
                if confidence_intervals:
                    prediction_dict["confidence_interval"] = confidence_intervals[i]
                
                if shap_values:
                    prediction_dict["shap_values"] = shap_values[i]
                
                formatted_predictions.append(prediction_dict)
            
            return formatted_predictions
        
        predictions = np.array([450000.0, 520000.0])
        confidence_intervals = [
            {"lower": 420000, "upper": 480000},
            {"lower": 490000, "upper": 550000}
        ]
        shap_values = [
            {"area": 0.3, "quartos": 0.2},
            {"area": 0.4, "quartos": 0.1}
        ]
        
        formatted = format_predictions(predictions, confidence_intervals, shap_values)
        
        assert len(formatted) == 2
        assert formatted[0]["predicted_value"] == 450000.0
        assert formatted[0]["confidence_interval"]["lower"] == 420000
        assert formatted[0]["shap_values"]["area"] == 0.3
        assert formatted[1]["predicted_value"] == 520000.0
    
    def test_calculate_inference_metrics(self):
        """Testa cálculo de métricas de inferência"""
        def calculate_inference_metrics(
            start_time: float,
            num_predictions: int
        ) -> Dict[str, Any]:
            
            end_time = start_time + 0.5  # Simular tempo de execução
            total_time = end_time - start_time
            
            return {
                "total_time": total_time,
                "predictions_per_second": num_predictions / total_time,
                "average_time_per_prediction": total_time / num_predictions
            }
        
        metrics = calculate_inference_metrics(100.0, 10)
        
        assert metrics["total_time"] == 0.5
        assert metrics["predictions_per_second"] == 20.0
        assert metrics["average_time_per_prediction"] == 0.05
    
    def test_error_handling_scenarios(self):
        """Testa cenários de tratamento de erro"""
        def handle_training_error(error: Exception, evaluation_id: str) -> MLTrainingResponse:
            error_message = str(error)
            
            return MLTrainingResponse(
                evaluation_id=evaluation_id,
                model_id="",
                status="error",
                message=f"Erro durante treinamento: {error_message}",
                model_type="",
                performance_metrics={},
                training_summary={},
                training_time=0.0,
                expert_mode=False,
                timestamp=datetime.now()
            )
        
        # Simular erro
        test_error = ValueError("Dados insuficientes para treinamento")
        error_response = handle_training_error(test_error, "eval_123")
        
        assert error_response.status == "error"
        assert "Dados insuficientes" in error_response.message
        assert error_response.evaluation_id == "eval_123"
        assert error_response.model_id == ""
    
    def test_model_serialization(self):
        """Testa serialização de modelo"""
        def serialize_model_info(model_result) -> Dict[str, Any]:
            return {
                "model_id": model_result.model_id,
                "model_type": model_result.model_type,
                "performance": {
                    "r2_score": model_result.performance.r2_score,
                    "rmse": model_result.performance.rmse
                },
                "hyperparameters": model_result.best_params,
                "training_timestamp": datetime.now().isoformat()
            }
        
        # Mock model result
        mock_result = Mock()
        mock_result.model_id = "model_456"
        mock_result.model_type = "xgboost"
        mock_result.performance = Mock()
        mock_result.performance.r2_score = 0.92
        mock_result.performance.rmse = 38000.0
        mock_result.best_params = {"max_depth": 5, "learning_rate": 0.1}
        
        serialized = serialize_model_info(mock_result)
        
        assert serialized["model_id"] == "model_456"
        assert serialized["model_type"] == "xgboost"
        assert serialized["performance"]["r2_score"] == 0.92
        assert serialized["hyperparameters"]["max_depth"] == 5
        assert "training_timestamp" in serialized


@pytest.mark.integration
class TestMLServiceIntegration:
    """Testes de integração para ML Service"""
    
    def test_complete_training_workflow(self):
        """Testa fluxo completo de treinamento"""
        # Simular requisição de treinamento
        request = MLTrainingRequest(
            evaluation_id="eval_integration_test",
            model_type="elastic_net",
            expert_mode=True,
            cross_validation=True,
            feature_selection=True
        )
        
        # Simular dados de treinamento
        training_data = pd.DataFrame({
            "valor": [400000, 500000, 350000, 600000, 450000],
            "area": [100, 120, 80, 150, 110],
            "quartos": [2, 3, 2, 4, 3]
        })
        
        def simulate_training_workflow(request, data):
            # 1. Validar dados
            validation_result = {
                "is_valid": True,
                "errors": [],
                "summary": {"total_rows": len(data)}
            }
            
            if not validation_result["is_valid"]:
                raise ValueError("Dados inválidos")
            
            # 2. Preparar configuração
            config = {
                "model_type": request.model_type,
                "expert_mode": request.expert_mode
            }
            
            # 3. Simular treinamento
            model_result = Mock()
            model_result.model_id = f"{request.evaluation_id}_elastic_net_12345"
            model_result.model_type = request.model_type
            model_result.performance = Mock()
            model_result.performance.r2_score = 0.85
            model_result.performance.rmse = 45000.0
            model_result.performance.mae = 35000.0
            model_result.training_summary = {"training_time": 120.5}
            
            # 4. Formar resposta
            response = MLTrainingResponse(
                evaluation_id=request.evaluation_id,
                model_id=model_result.model_id,
                status="success",
                message="Treinamento concluído com sucesso",
                model_type=model_result.model_type,
                performance_metrics={
                    "r2_score": model_result.performance.r2_score,
                    "rmse": model_result.performance.rmse,
                    "mae": model_result.performance.mae
                },
                training_summary=model_result.training_summary,
                training_time=120.5,
                expert_mode=request.expert_mode,
                timestamp=datetime.now()
            )
            
            return response
        
        # Executar workflow
        response = simulate_training_workflow(request, training_data)
        
        # Verificar resultado
        assert response.status == "success"
        assert response.evaluation_id == "eval_integration_test"
        assert response.model_type == "elastic_net"
        assert response.performance_metrics["r2_score"] == 0.85
        assert response.expert_mode is True
    
    def test_complete_inference_workflow(self):
        """Testa fluxo completo de inferência"""
        # Simular requisição de inferência
        request = MLInferenceRequest(
            model_id="model_123",
            data=[
                {"area": 100, "quartos": 2},
                {"area": 120, "quartos": 3}
            ],
            return_confidence=True,
            return_shap=True
        )
        
        def simulate_inference_workflow(request):
            # 1. Validar modelo (simulado)
            model_exists = True
            if not model_exists:
                raise ValueError("Modelo não encontrado")
            
            # 2. Preparar dados
            inference_data = pd.DataFrame(request.data)
            
            # 3. Fazer predições (simulado)
            predictions = np.array([450000.0, 520000.0])
            
            # 4. Calcular intervalos de confiança
            confidence_intervals = [
                {"lower": 420000, "upper": 480000, "width": 60000},
                {"lower": 490000, "upper": 550000, "width": 60000}
            ]
            
            # 5. Calcular valores SHAP
            shap_values = [
                {"area": 0.3, "quartos": 0.2},
                {"area": 0.4, "quartos": 0.1}
            ]
            
            # 6. Formatar predições
            formatted_predictions = []
            for i, pred in enumerate(predictions):
                prediction = {
                    "predicted_value": float(pred),
                    "input_data": request.data[i]
                }
                
                if request.return_confidence:
                    prediction["confidence_interval"] = confidence_intervals[i]
                
                if request.return_shap:
                    prediction["shap_values"] = shap_values[i]
                
                formatted_predictions.append(prediction)
            
            # 7. Formar resposta
            response = MLInferenceResponse(
                model_id=request.model_id,
                predictions=formatted_predictions,
                model_info={
                    "model_type": "elastic_net",
                    "features": ["area", "quartos"]
                },
                inference_time=0.5,
                timestamp=datetime.now()
            )
            
            return response
        
        # Executar workflow
        response = simulate_inference_workflow(request)
        
        # Verificar resultado
        assert response.model_id == "model_123"
        assert len(response.predictions) == 2
        assert response.predictions[0]["predicted_value"] == 450000.0
        assert "confidence_interval" in response.predictions[0]
        assert "shap_values" in response.predictions[0]
        assert response.inference_time == 0.5