# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Sistema de Cache Inteligente para Valion
Implementa cache para features geoespaciais e modelos treinados para otimização de performance.
"""

import pickle
import hashlib
import json
import time
from functools import lru_cache, wraps
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from threading import Lock
import redis
from contextlib import contextmanager
from abc import ABC, abstractmethod
import joblib
from datetime import datetime, timedelta
import os


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrada do cache com metadados."""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_accessed: float = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Verifica se a entrada expirou."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def is_stale(self, max_age: float = 3600) -> bool:
        """Verifica se a entrada está obsoleta."""
        return time.time() - self.timestamp > max_age


@dataclass
class ModelCacheEntry:
    """Entrada específica para cache de modelos."""
    model_id: str
    model_data: Any
    model_type: str
    features_hash: str
    performance_metrics: Dict[str, float]
    training_timestamp: float
    training_config: Dict[str, Any]
    file_path: Optional[str] = None
    
    def is_similar_features(self, other_hash: str, similarity_threshold: float = 0.9) -> bool:
        """Verifica se as features são similares o suficiente para reutilização."""
        if self.features_hash == other_hash:
            return True
        
        # Implementar comparação de similaridade mais sofisticada se necessário
        return False


class CacheBackend(ABC):
    """Interface abstrata para backends de cache."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Define valor no cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove valor do cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Limpa todo o cache."""
        pass


class MemoryCache(CacheBackend):
    """Cache em memória com LRU eviction."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self.cache[key]
                return None
            
            # Atualizar estatísticas de acesso
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self.lock:
            # Eviction se necessário
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Calcular tamanho aproximado
            size_bytes = len(pickle.dumps(value))
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size_bytes,
                last_accessed=time.time()
            )
            
            self.cache[key] = entry
    
    def delete(self, key: str) -> bool:
        with self.lock:
            return self.cache.pop(key, None) is not None
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
    
    def _evict_lru(self):
        """Remove o item menos recentemente usado."""
        if not self.cache:
            return
        
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].last_accessed or 0)
        del self.cache[lru_key]


class RedisCache(CacheBackend):
    """Cache usando Redis."""
    
    def __init__(self, host: str = 'redis', port: int = 6379,  # Docker service name 
                 db: int = 0, prefix: str = 'valion:cache'):
        self.prefix = prefix
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.redis_client.ping()
            logger.info(f"Redis cache conectado em {host}:{port}")
        except Exception as e:
            logger.warning(f"Falha ao conectar Redis: {e}. Usando cache em memória.")
            self.redis_client = None
    
    def _make_key(self, key: str) -> str:
        """Adiciona prefixo à chave."""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        if self.redis_client is None:
            return None
        
        try:
            data = self.redis_client.get(self._make_key(key))
            if data is None:
                return None
            
            return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Erro ao obter do Redis: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        if self.redis_client is None:
            return
        
        try:
            data = pickle.dumps(value)
            redis_key = self._make_key(key)
            
            if ttl:
                self.redis_client.setex(redis_key, int(ttl), data)
            else:
                self.redis_client.set(redis_key, data)
        except Exception as e:
            logger.warning(f"Erro ao definir no Redis: {e}")
    
    def delete(self, key: str) -> bool:
        if self.redis_client is None:
            return False
        
        try:
            return bool(self.redis_client.delete(self._make_key(key)))
        except Exception as e:
            logger.warning(f"Erro ao deletar do Redis: {e}")
            return False
    
    def clear(self) -> None:
        if self.redis_client is None:
            return
        
        try:
            pattern = f"{self.prefix}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Erro ao limpar Redis: {e}")


class GeospatialCache:
    """Cache especializado para features geoespaciais."""
    
    def __init__(self, backend: CacheBackend, ttl: float = 3600):
        self.backend = backend
        self.ttl = ttl
    
    def _make_location_key(self, coordinates: Tuple[float, float], 
                          precision: int = 4) -> str:
        """Cria chave baseada em coordenadas com precisão ajustável."""
        lat, lon = coordinates
        lat_rounded = round(lat, precision)
        lon_rounded = round(lon, precision)
        return f"geo:{lat_rounded},{lon_rounded}"
    
    def _make_address_key(self, address: str) -> str:
        """Cria chave baseada em endereço."""
        address_hash = hashlib.md5(address.lower().encode()).hexdigest()
        return f"addr:{address_hash}"
    
    def get_coordinates(self, address: str) -> Optional[Tuple[float, float]]:
        """Obtém coordenadas do cache."""
        key = self._make_address_key(address)
        return self.backend.get(key)
    
    def set_coordinates(self, address: str, coordinates: Tuple[float, float]) -> None:
        """Define coordenadas no cache."""
        key = self._make_address_key(address)
        self.backend.set(key, coordinates, self.ttl)
    
    def get_features(self, coordinates: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """Obtém features geoespaciais do cache."""
        key = self._make_location_key(coordinates)
        return self.backend.get(key)
    
    def set_features(self, coordinates: Tuple[float, float], features: Dict[str, Any]) -> None:
        """Define features geoespaciais no cache."""
        key = self._make_location_key(coordinates)
        self.backend.set(key, features, self.ttl)
    
    def get_poi_analysis(self, coordinates: Tuple[float, float], 
                        radius: float) -> Optional[Dict[str, Any]]:
        """Obtém análise de POIs do cache."""
        key = f"poi:{self._make_location_key(coordinates)}:{radius}"
        return self.backend.get(key)
    
    def set_poi_analysis(self, coordinates: Tuple[float, float], 
                        radius: float, analysis: Dict[str, Any]) -> None:
        """Define análise de POIs no cache."""
        key = f"poi:{self._make_location_key(coordinates)}:{radius}"
        self.backend.set(key, analysis, self.ttl)


class ModelCache:
    """Cache inteligente para modelos treinados."""
    
    def __init__(self, cache_dir: str = "model_cache", max_models: int = 50):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_models = max_models
        self.models_registry: Dict[str, ModelCacheEntry] = {}
        self.lock = Lock()
        self._load_registry()
    
    def _get_features_hash(self, features: List[str]) -> str:
        """Calcula hash das features para comparação."""
        features_sorted = sorted(features)
        features_str = ','.join(features_sorted)
        return hashlib.md5(features_str.encode()).hexdigest()
    
    def _get_model_id(self, model_type: str, features_hash: str, 
                     config_hash: str) -> str:
        """Gera ID único para o modelo."""
        combined = f"{model_type}:{features_hash}:{config_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """Calcula hash da configuração."""
        # Remover campos que não afetam o treinamento
        clean_config = {k: v for k, v in config.items() 
                       if k not in ['evaluation_id', 'timestamp', 'user_id']}
        config_str = json.dumps(clean_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def find_similar_model(self, model_type: str, features: List[str], 
                          config: Dict[str, Any],
                          similarity_threshold: float = 0.9) -> Optional[ModelCacheEntry]:
        """Encontra modelo similar que pode ser reutilizado."""
        features_hash = self._get_features_hash(features)
        config_hash = self._get_config_hash(config)
        
        with self.lock:
            # Buscar modelo exato primeiro
            model_id = self._get_model_id(model_type, features_hash, config_hash)
            if model_id in self.models_registry:
                entry = self.models_registry[model_id]
                if self._is_model_valid(entry):
                    logger.info(f"Modelo exato encontrado: {model_id}")
                    return entry
            
            # Buscar modelos similares
            for entry in self.models_registry.values():
                if (entry.model_type == model_type and 
                    entry.is_similar_features(features_hash, similarity_threshold) and
                    self._is_model_valid(entry)):
                    
                    logger.info(f"Modelo similar encontrado: {entry.model_id}")
                    return entry
        
        return None
    
    def _is_model_valid(self, entry: ModelCacheEntry) -> bool:
        """Verifica se o modelo ainda é válido."""
        # Verificar se o arquivo existe
        if entry.file_path and not Path(entry.file_path).exists():
            return False
        
        # Verificar idade do modelo (não usar modelos muito antigos)
        max_age = 7 * 24 * 3600  # 7 dias
        if time.time() - entry.training_timestamp > max_age:
            return False
        
        # Verificar métricas mínimas
        if entry.performance_metrics.get('r2_score', 0) < 0.5:
            return False
        
        return True
    
    def cache_model(self, model_data: Any, model_type: str, features: List[str],
                   config: Dict[str, Any], performance_metrics: Dict[str, float]) -> str:
        """Armazena modelo no cache."""
        features_hash = self._get_features_hash(features)
        config_hash = self._get_config_hash(config)
        model_id = self._get_model_id(model_type, features_hash, config_hash)
        
        with self.lock:
            # Verificar limite de modelos
            if len(self.models_registry) >= self.max_models:
                self._evict_oldest_model()
            
            # Salvar modelo no disco
            model_file = self.cache_dir / f"{model_id}.joblib"
            try:
                joblib.dump(model_data, model_file)
                logger.info(f"Modelo salvo: {model_file}")
            except Exception as e:
                logger.error(f"Erro ao salvar modelo: {e}")
                return None
            
            # Criar entrada no registry
            entry = ModelCacheEntry(
                model_id=model_id,
                model_data=model_data,
                model_type=model_type,
                features_hash=features_hash,
                performance_metrics=performance_metrics,
                training_timestamp=time.time(),
                training_config=config,
                file_path=str(model_file)
            )
            
            self.models_registry[model_id] = entry
            self._save_registry()
            
            logger.info(f"Modelo {model_id} adicionado ao cache")
            return model_id
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Carrega modelo do cache."""
        with self.lock:
            entry = self.models_registry.get(model_id)
            if entry is None:
                return None
            
            if not self._is_model_valid(entry):
                self._remove_model(model_id)
                return None
            
            try:
                if entry.file_path:
                    model_data = joblib.load(entry.file_path)
                    logger.info(f"Modelo {model_id} carregado do cache")
                    return model_data
                else:
                    return entry.model_data
            except Exception as e:
                logger.error(f"Erro ao carregar modelo {model_id}: {e}")
                self._remove_model(model_id)
                return None
    
    def _evict_oldest_model(self):
        """Remove o modelo mais antigo."""
        if not self.models_registry:
            return
        
        oldest_id = min(self.models_registry.keys(),
                       key=lambda k: self.models_registry[k].training_timestamp)
        self._remove_model(oldest_id)
    
    def _remove_model(self, model_id: str):
        """Remove modelo do cache."""
        entry = self.models_registry.get(model_id)
        if entry:
            # Remover arquivo
            if entry.file_path and Path(entry.file_path).exists():
                try:
                    Path(entry.file_path).unlink()
                except Exception as e:
                    logger.warning(f"Erro ao remover arquivo do modelo: {e}")
            
            # Remover do registry
            del self.models_registry[model_id]
            self._save_registry()
            logger.info(f"Modelo {model_id} removido do cache")
    
    def _load_registry(self):
        """Carrega registry do disco."""
        registry_file = self.cache_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                for model_id, entry_data in data.items():
                    entry = ModelCacheEntry(**entry_data)
                    self.models_registry[model_id] = entry
                
                logger.info(f"Registry carregado: {len(self.models_registry)} modelos")
            except Exception as e:
                logger.error(f"Erro ao carregar registry: {e}")
    
    def _save_registry(self):
        """Salva registry no disco."""
        registry_file = self.cache_dir / "registry.json"
        try:
            data = {}
            for model_id, entry in self.models_registry.items():
                data[model_id] = {
                    'model_id': entry.model_id,
                    'model_type': entry.model_type,
                    'features_hash': entry.features_hash,
                    'performance_metrics': entry.performance_metrics,
                    'training_timestamp': entry.training_timestamp,
                    'training_config': entry.training_config,
                    'file_path': entry.file_path
                }
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar registry: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do cache."""
        with self.lock:
            total_size = sum(Path(entry.file_path).stat().st_size 
                           for entry in self.models_registry.values() 
                           if entry.file_path and Path(entry.file_path).exists())
            
            return {
                'total_models': len(self.models_registry),
                'total_size_mb': total_size / (1024 * 1024),
                'model_types': list(set(entry.model_type for entry in self.models_registry.values())),
                'oldest_model': min(self.models_registry.values(), 
                                  key=lambda e: e.training_timestamp).training_timestamp if self.models_registry else None
            }


class CacheManager:
    """Gerenciador central de cache."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Inicializar backend
        cache_backend = config.get('cache_backend', 'memory')
        if cache_backend == 'redis':
            self.backend = RedisCache(
                host=config.get('redis_host', 'redis'),  # Docker service name
                port=config.get('redis_port', 6379),
                db=config.get('redis_db', 0)
            )
        else:
            self.backend = MemoryCache(max_size=config.get('max_cache_size', 1000))
        
        # Inicializar caches especializados
        self.geospatial_cache = GeospatialCache(
            backend=self.backend,
            ttl=config.get('geospatial_ttl', 3600)
        )
        
        self.model_cache = ModelCache(
            cache_dir=config.get('model_cache_dir', 'model_cache'),
            max_models=config.get('max_cached_models', 50)
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas completas do cache."""
        return {
            'model_cache': self.model_cache.get_cache_stats(),
            'backend_type': type(self.backend).__name__,
            'geospatial_ttl': self.geospatial_cache.ttl
        }
    
    def clear_all_caches(self):
        """Limpa todos os caches."""
        self.backend.clear()
        # Model cache tem sua própria limpeza
        logger.info("Todos os caches foram limpos")


# Decorators para cache automático
def cache_geospatial_feature(ttl: float = 3600):
    """Decorator para cache automático de features geoespaciais."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Tentar extrair coordenadas dos argumentos
            coordinates = None
            if len(args) >= 2 and isinstance(args[0], tuple) and len(args[0]) == 2:
                coordinates = args[0]
            elif 'coordinates' in kwargs:
                coordinates = kwargs['coordinates']
            
            if coordinates:
                cache_key = f"geo_feature:{func.__name__}:{coordinates[0]:.4f},{coordinates[1]:.4f}"
                
                # Tentar obter do cache global se existir
                try:
                    from src.config.settings import Settings
                    settings = Settings()
                    if hasattr(settings, 'cache_manager'):
                        cached_result = settings.cache_manager.backend.get(cache_key)
                        if cached_result is not None:
                            return cached_result
                except Exception:
                    pass
            
            # Executar função
            result = func(*args, **kwargs)
            
            # Salvar no cache se possível
            if coordinates:
                try:
                    from src.config.settings import Settings
                    settings = Settings()
                    if hasattr(settings, 'cache_manager'):
                        settings.cache_manager.backend.set(cache_key, result, ttl)
                except Exception:
                    pass
            
            return result
        return wrapper
    return decorator


# Funções auxiliares para integração
def create_cache_manager(config: Dict[str, Any]) -> CacheManager:
    """Cria gerenciador de cache com configuração."""
    return CacheManager(config)


def get_default_cache_config() -> Dict[str, Any]:
    """Retorna configuração padrão de cache."""
    return {
        'cache_backend': 'memory',
        'max_cache_size': 1000,
        'geospatial_ttl': 3600,
        'model_cache_dir': 'model_cache',
        'max_cached_models': 50,
        'redis_host': 'redis',  # Docker service name
        'redis_port': 6379,
        'redis_db': 0
    }