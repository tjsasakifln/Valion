# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Model Registry para MLOps
Sistema de versionamento e gestão de modelos de ML.
"""

import json
import hashlib
import shutil
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import sqlite3
import threading
import logging
from contextlib import contextmanager
import joblib
import pandas as pd
import numpy as np
from ..monitoring.logging_config import get_logger
import structlog


class ModelStage(Enum):
    """Estágios do modelo."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(Enum):
    """Status do modelo."""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata:
    """Metadados do modelo."""
    name: str
    version: str
    algorithm: str
    framework: str
    author: str
    description: str
    tags: List[str]
    hyperparameters: Dict[str, Any]
    features: List[str]
    target: str
    dataset_info: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Cria instância a partir de dicionário."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ModelVersion:
    """Versão do modelo."""
    model_id: str
    version: str
    stage: ModelStage
    status: ModelStatus
    metadata: ModelMetadata
    performance_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    artifacts: Dict[str, str]  # artifact_type -> file_path
    size_bytes: int
    checksum: str
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'stage': self.stage.value,
            'status': self.status.value,
            'metadata': self.metadata.to_dict(),
            'performance_metrics': self.performance_metrics,
            'validation_results': self.validation_results,
            'artifacts': self.artifacts,
            'size_bytes': self.size_bytes,
            'checksum': self.checksum,
            'parent_version': self.parent_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Cria instância a partir de dicionário."""
        return cls(
            model_id=data['model_id'],
            version=data['version'],
            stage=ModelStage(data['stage']),
            status=ModelStatus(data['status']),
            metadata=ModelMetadata.from_dict(data['metadata']),
            performance_metrics=data['performance_metrics'],
            validation_results=data['validation_results'],
            artifacts=data['artifacts'],
            size_bytes=data['size_bytes'],
            checksum=data['checksum'],
            parent_version=data.get('parent_version')
        )


class ModelRegistry:
    """Registry central de modelos."""
    
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Diretórios
        self.models_dir = self.registry_dir / "models"
        self.artifacts_dir = self.registry_dir / "artifacts"
        self.metadata_dir = self.registry_dir / "metadata"
        self.db_path = self.registry_dir / "registry.db"
        
        # Criar diretórios
        for dir_path in [self.models_dir, self.artifacts_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logger = get_logger("model_registry")
        self.struct_logger = structlog.get_logger("model_registry")
        
        # Thread lock para operações concorrentes
        self.lock = threading.Lock()
        
        # Inicializar banco de dados
        self._init_database()
    
    def _init_database(self):
        """Inicializa banco de dados SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    latest_version TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    model_id TEXT,
                    version TEXT,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    validation_results TEXT NOT NULL,
                    artifacts TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    parent_version TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (model_id, version),
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    version TEXT,
                    environment TEXT NOT NULL,
                    endpoint TEXT,
                    status TEXT NOT NULL,
                    deployed_at TEXT NOT NULL,
                    FOREIGN KEY (model_id, version) REFERENCES model_versions (model_id, version)
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Context manager para conexão com banco."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def register_model(self, name: str, algorithm: str, framework: str = "sklearn",
                      author: str = "system", description: str = "",
                      tags: List[str] = None) -> str:
        """Registra um novo modelo."""
        with self.lock:
            # Gerar ID único
            model_id = f"{name}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
            
            try:
                with self._get_db_connection() as conn:
                    # Verificar se modelo já existe
                    existing = conn.execute(
                        "SELECT model_id FROM models WHERE model_id = ?",
                        (model_id,)
                    ).fetchone()
                    
                    if existing:
                        self.logger.info(f"Model {model_id} already exists")
                        return model_id
                    
                    # Inserir modelo
                    now = datetime.now().isoformat()
                    conn.execute('''
                        INSERT INTO models (model_id, name, latest_version, stage, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (model_id, name, "0.0.0", ModelStage.DEVELOPMENT.value, 
                          ModelStatus.READY.value, now, now))
                    
                    conn.commit()
                    
                    self.struct_logger.info(
                        "Model registered",
                        model_id=model_id,
                        name=name,
                        algorithm=algorithm,
                        framework=framework
                    )
                    
                    return model_id
                    
            except Exception as e:
                self.logger.error(f"Error registering model: {e}")
                raise
    
    def create_version(self, model_id: str, model_object: Any, 
                      performance_metrics: Dict[str, float],
                      hyperparameters: Dict[str, Any] = None,
                      features: List[str] = None,
                      target: str = None,
                      dataset_info: Dict[str, Any] = None,
                      validation_results: Dict[str, Any] = None,
                      description: str = "",
                      tags: List[str] = None) -> ModelVersion:
        """Cria nova versão do modelo."""
        with self.lock:
            try:
                # Obter próxima versão
                version = self._get_next_version(model_id)
                
                # Criar metadados
                with self._get_db_connection() as conn:
                    model_info = conn.execute(
                        "SELECT name FROM models WHERE model_id = ?",
                        (model_id,)
                    ).fetchone()
                    
                    if not model_info:
                        raise ValueError(f"Model {model_id} not found")
                    
                    model_name = model_info['name']
                
                metadata = ModelMetadata(
                    name=model_name,
                    version=version,
                    algorithm=type(model_object).__name__,
                    framework="sklearn",  # Detectar automaticamente
                    author="system",
                    description=description,
                    tags=tags or [],
                    hyperparameters=hyperparameters or {},
                    features=features or [],
                    target=target or "",
                    dataset_info=dataset_info or {},
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # Salvar artefatos
                artifacts = self._save_artifacts(model_id, version, model_object)
                
                # Calcular checksum
                checksum = self._calculate_checksum(artifacts)
                
                # Calcular tamanho
                size_bytes = sum(
                    Path(path).stat().st_size 
                    for path in artifacts.values() 
                    if Path(path).exists()
                )
                
                # Criar versão
                model_version = ModelVersion(
                    model_id=model_id,
                    version=version,
                    stage=ModelStage.DEVELOPMENT,
                    status=ModelStatus.READY,
                    metadata=metadata,
                    performance_metrics=performance_metrics,
                    validation_results=validation_results or {},
                    artifacts=artifacts,
                    size_bytes=size_bytes,
                    checksum=checksum
                )
                
                # Salvar no banco
                self._save_version_to_db(model_version)
                
                # Atualizar modelo principal
                self._update_model_latest_version(model_id, version)
                
                self.struct_logger.info(
                    "Model version created",
                    model_id=model_id,
                    version=version,
                    r2_score=performance_metrics.get('r2_score', 0),
                    size_mb=size_bytes / (1024 * 1024)
                )
                
                return model_version
                
            except Exception as e:
                self.logger.error(f"Error creating model version: {e}")
                raise
    
    def get_model_version(self, model_id: str, version: str = None) -> Optional[ModelVersion]:
        """Obtém versão específica do modelo."""
        try:
            with self._get_db_connection() as conn:
                if version is None:
                    # Obter última versão
                    model_info = conn.execute(
                        "SELECT latest_version FROM models WHERE model_id = ?",
                        (model_id,)
                    ).fetchone()
                    
                    if not model_info:
                        return None
                    
                    version = model_info['latest_version']
                
                # Obter versão específica
                version_data = conn.execute('''
                    SELECT * FROM model_versions 
                    WHERE model_id = ? AND version = ?
                ''', (model_id, version)).fetchone()
                
                if not version_data:
                    return None
                
                return ModelVersion(
                    model_id=version_data['model_id'],
                    version=version_data['version'],
                    stage=ModelStage(version_data['stage']),
                    status=ModelStatus(version_data['status']),
                    metadata=ModelMetadata.from_dict(json.loads(version_data['metadata'])),
                    performance_metrics=json.loads(version_data['performance_metrics']),
                    validation_results=json.loads(version_data['validation_results']),
                    artifacts=json.loads(version_data['artifacts']),
                    size_bytes=version_data['size_bytes'],
                    checksum=version_data['checksum'],
                    parent_version=version_data['parent_version']
                )
                
        except Exception as e:
            self.logger.error(f"Error getting model version: {e}")
            return None
    
    def list_models(self, stage: ModelStage = None, status: ModelStatus = None) -> List[Dict[str, Any]]:
        """Lista modelos com filtros opcionais."""
        try:
            with self._get_db_connection() as conn:
                query = "SELECT * FROM models WHERE 1=1"
                params = []
                
                if stage:
                    query += " AND stage = ?"
                    params.append(stage.value)
                
                if status:
                    query += " AND status = ?"
                    params.append(status.value)
                
                results = conn.execute(query, params).fetchall()
                
                models = []
                for row in results:
                    model_dict = dict(row)
                    
                    # Adicionar informações da última versão
                    latest_version = self.get_model_version(model_dict['model_id'])
                    if latest_version:
                        model_dict['latest_performance'] = latest_version.performance_metrics
                        model_dict['latest_size_mb'] = latest_version.size_bytes / (1024 * 1024)
                    
                    models.append(model_dict)
                
                return models
                
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """Lista todas as versões de um modelo."""
        try:
            with self._get_db_connection() as conn:
                results = conn.execute('''
                    SELECT * FROM model_versions 
                    WHERE model_id = ? 
                    ORDER BY version DESC
                ''', (model_id,)).fetchall()
                
                versions = []
                for row in results:
                    version = ModelVersion(
                        model_id=row['model_id'],
                        version=row['version'],
                        stage=ModelStage(row['stage']),
                        status=ModelStatus(row['status']),
                        metadata=ModelMetadata.from_dict(json.loads(row['metadata'])),
                        performance_metrics=json.loads(row['performance_metrics']),
                        validation_results=json.loads(row['validation_results']),
                        artifacts=json.loads(row['artifacts']),
                        size_bytes=row['size_bytes'],
                        checksum=row['checksum'],
                        parent_version=row['parent_version']
                    )
                    versions.append(version)
                
                return versions
                
        except Exception as e:
            self.logger.error(f"Error listing versions: {e}")
            return []
    
    def promote_version(self, model_id: str, version: str, 
                       target_stage: ModelStage) -> bool:
        """Promove versão para estágio superior."""
        try:
            with self._get_db_connection() as conn:
                # Atualizar estágio
                conn.execute('''
                    UPDATE model_versions 
                    SET stage = ? 
                    WHERE model_id = ? AND version = ?
                ''', (target_stage.value, model_id, version))
                
                # Se promovendo para produção, despromover outras versões
                if target_stage == ModelStage.PRODUCTION:
                    conn.execute('''
                        UPDATE model_versions 
                        SET stage = ? 
                        WHERE model_id = ? AND version != ? AND stage = ?
                    ''', (ModelStage.STAGING.value, model_id, version, ModelStage.PRODUCTION.value))
                
                conn.commit()
                
                self.struct_logger.info(
                    "Model version promoted",
                    model_id=model_id,
                    version=version,
                    target_stage=target_stage.value
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error promoting version: {e}")
            return False
    
    def load_model(self, model_id: str, version: str = None) -> Optional[Any]:
        """Carrega modelo do registry."""
        try:
            model_version = self.get_model_version(model_id, version)
            
            if not model_version:
                return None
            
            # Carregar modelo principal
            model_path = model_version.artifacts.get("model")
            if not model_path or not Path(model_path).exists():
                self.logger.error(f"Model file not found: {model_path}")
                return None
            
            model_data = joblib.load(model_path)
            
            self.struct_logger.info(
                "Model loaded",
                model_id=model_id,
                version=model_version.version,
                stage=model_version.stage.value
            )
            
            return model_data
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def delete_version(self, model_id: str, version: str) -> bool:
        """Remove versão específica."""
        try:
            with self._get_db_connection() as conn:
                # Obter informações da versão
                version_data = conn.execute('''
                    SELECT artifacts FROM model_versions 
                    WHERE model_id = ? AND version = ?
                ''', (model_id, version)).fetchone()
                
                if not version_data:
                    return False
                
                # Remover artefatos
                artifacts = json.loads(version_data['artifacts'])
                for artifact_path in artifacts.values():
                    try:
                        Path(artifact_path).unlink(missing_ok=True)
                    except Exception as e:
                        self.logger.warning(f"Could not delete artifact {artifact_path}: {e}")
                
                # Remover do banco
                conn.execute('''
                    DELETE FROM model_versions 
                    WHERE model_id = ? AND version = ?
                ''', (model_id, version))
                
                conn.commit()
                
                self.struct_logger.info(
                    "Model version deleted",
                    model_id=model_id,
                    version=version
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting version: {e}")
            return False
    
    def _get_next_version(self, model_id: str) -> str:
        """Obtém próximo número de versão."""
        try:
            with self._get_db_connection() as conn:
                result = conn.execute('''
                    SELECT version FROM model_versions 
                    WHERE model_id = ? 
                    ORDER BY version DESC 
                    LIMIT 1
                ''', (model_id,)).fetchone()
                
                if not result:
                    return "1.0.0"
                
                # Incrementar versão (simplificado)
                current_version = result['version']
                parts = current_version.split('.')
                major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                
                # Incrementar patch
                return f"{major}.{minor}.{patch + 1}"
                
        except Exception as e:
            self.logger.error(f"Error getting next version: {e}")
            return "1.0.0"
    
    def _save_artifacts(self, model_id: str, version: str, model_object: Any) -> Dict[str, str]:
        """Salva artefatos do modelo."""
        version_dir = self.artifacts_dir / model_id / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        # Salvar modelo principal
        model_path = version_dir / "model.joblib"
        joblib.dump(model_object, model_path)
        artifacts["model"] = str(model_path)
        
        # Salvar scaler se existir
        if hasattr(model_object, 'scaler') and model_object.scaler:
            scaler_path = version_dir / "scaler.joblib"
            joblib.dump(model_object.scaler, scaler_path)
            artifacts["scaler"] = str(scaler_path)
        
        # Salvar explainer se existir
        if hasattr(model_object, 'explainer') and model_object.explainer:
            explainer_path = version_dir / "explainer.joblib"
            joblib.dump(model_object.explainer, explainer_path)
            artifacts["explainer"] = str(explainer_path)
        
        return artifacts
    
    def _calculate_checksum(self, artifacts: Dict[str, str]) -> str:
        """Calcula checksum dos artefatos."""
        hash_md5 = hashlib.md5()
        
        for artifact_path in sorted(artifacts.values()):
            if Path(artifact_path).exists():
                with open(artifact_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _save_version_to_db(self, model_version: ModelVersion):
        """Salva versão no banco de dados."""
        with self._get_db_connection() as conn:
            conn.execute('''
                INSERT INTO model_versions 
                (model_id, version, stage, status, metadata, performance_metrics, 
                 validation_results, artifacts, size_bytes, checksum, parent_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_version.model_id,
                model_version.version,
                model_version.stage.value,
                model_version.status.value,
                json.dumps(model_version.metadata.to_dict()),
                json.dumps(model_version.performance_metrics),
                json.dumps(model_version.validation_results),
                json.dumps(model_version.artifacts),
                model_version.size_bytes,
                model_version.checksum,
                model_version.parent_version,
                datetime.now().isoformat()
            ))
            
            conn.commit()
    
    def _update_model_latest_version(self, model_id: str, version: str):
        """Atualiza última versão do modelo."""
        with self._get_db_connection() as conn:
            conn.execute('''
                UPDATE models 
                SET latest_version = ?, updated_at = ?
                WHERE model_id = ?
            ''', (version, datetime.now().isoformat(), model_id))
            
            conn.commit()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do registry."""
        try:
            with self._get_db_connection() as conn:
                # Contadores básicos
                models_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
                versions_count = conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0]
                
                # Estatísticas por estágio
                stage_stats = {}
                for stage in ModelStage:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM model_versions WHERE stage = ?",
                        (stage.value,)
                    ).fetchone()[0]
                    stage_stats[stage.value] = count
                
                # Tamanho total
                total_size = conn.execute(
                    "SELECT SUM(size_bytes) FROM model_versions"
                ).fetchone()[0] or 0
                
                return {
                    "models_count": models_count,
                    "versions_count": versions_count,
                    "stage_distribution": stage_stats,
                    "total_size_mb": total_size / (1024 * 1024),
                    "registry_path": str(self.registry_dir)
                }
                
        except Exception as e:
            self.logger.error(f"Error getting registry stats: {e}")
            return {}


def create_model_registry(registry_dir: str = "model_registry") -> ModelRegistry:
    """Cria instância do model registry."""
    return ModelRegistry(registry_dir)