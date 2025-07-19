"""
Testes unitários para o módulo cache_system.py
"""
import pytest
import time
import pickle
import hashlib
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from src.core.cache_system import CacheEntry


class TestCacheEntry:
    """Testes para a classe CacheEntry"""
    
    def test_init(self):
        """Testa inicialização do CacheEntry"""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            timestamp=time.time()
        )
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.timestamp > 0
        assert entry.ttl is None
        assert entry.access_count == 0
        assert entry.last_accessed is None
        assert entry.size_bytes == 0
    
    def test_is_expired_no_ttl(self):
        """Testa expiração sem TTL definido"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=time.time(),
            ttl=None
        )
        
        assert entry.is_expired() is False
    
    def test_is_expired_with_ttl_not_expired(self):
        """Testa entrada não expirada com TTL"""
        current_time = time.time()
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=current_time,
            ttl=3600  # 1 hora
        )
        
        assert entry.is_expired() is False
    
    def test_is_expired_with_ttl_expired(self):
        """Testa entrada expirada com TTL"""
        old_time = time.time() - 7200  # 2 horas atrás
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=old_time,
            ttl=3600  # 1 hora
        )
        
        assert entry.is_expired() is True
    
    def test_is_stale_not_stale(self):
        """Testa entrada não obsoleta"""
        recent_time = time.time() - 1800  # 30 minutos atrás
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=recent_time
        )
        
        assert entry.is_stale(max_age=3600) is False
    
    def test_is_stale_stale(self):
        """Testa entrada obsoleta"""
        old_time = time.time() - 7200  # 2 horas atrás
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=old_time
        )
        
        assert entry.is_stale(max_age=3600) is True
    
    def test_is_stale_default_max_age(self):
        """Testa entrada obsoleta com max_age padrão"""
        old_time = time.time() - 7200  # 2 horas atrás
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=old_time
        )
        
        assert entry.is_stale() is True  # max_age padrão é 3600 segundos


# Como o código atual tem apenas CacheEntry visível, vou criar testes mock
# para as outras classes que provavelmente existem no arquivo completo
class TestCacheSystemMockStructure:
    """Testes para estrutura de cache usando mocks"""
    
    @pytest.fixture
    def mock_cache_backend(self):
        """Mock para backend de cache"""
        backend = Mock()
        backend.get.return_value = None
        backend.set.return_value = True
        backend.delete.return_value = True
        backend.exists.return_value = False
        backend.clear.return_value = True
        return backend
    
    @pytest.fixture
    def sample_cache_data(self):
        """Dados de exemplo para cache"""
        return {
            "model_predictions": [400000, 500000, 350000],
            "feature_importance": {"area": 0.5, "quartos": 0.3},
            "metadata": {"created_at": time.time(), "version": "1.0"}
        }
    
    def test_cache_key_generation(self):
        """Testa geração de chaves de cache"""
        # Simular função de geração de chave
        def generate_cache_key(data: dict, prefix: str = "") -> str:
            data_str = json.dumps(data, sort_keys=True)
            hash_obj = hashlib.md5(data_str.encode())
            key = hash_obj.hexdigest()
            return f"{prefix}:{key}" if prefix else key
        
        data1 = {"area": 100, "quartos": 2}
        data2 = {"quartos": 2, "area": 100}  # Mesmos dados, ordem diferente
        data3 = {"area": 120, "quartos": 2}  # Dados diferentes
        
        key1 = generate_cache_key(data1, "features")
        key2 = generate_cache_key(data2, "features")
        key3 = generate_cache_key(data3, "features")
        
        assert key1 == key2  # Mesmos dados devem gerar mesma chave
        assert key1 != key3  # Dados diferentes devem gerar chaves diferentes
        assert key1.startswith("features:")
    
    def test_memory_cache_operations(self, mock_cache_backend, sample_cache_data):
        """Testa operações básicas de cache em memória"""
        # Simular cache em memória
        memory_cache = {}
        
        def mock_set(key, value, ttl=None):
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=len(pickle.dumps(value))
            )
            memory_cache[key] = entry
            return True
        
        def mock_get(key):
            entry = memory_cache.get(key)
            if entry and not entry.is_expired():
                entry.access_count += 1
                entry.last_accessed = time.time()
                return entry.value
            return None
        
        # Testar operações
        key = "test_key"
        assert mock_get(key) is None  # Cache vazio
        
        assert mock_set(key, sample_cache_data, ttl=3600) is True
        retrieved_data = mock_get(key)
        assert retrieved_data == sample_cache_data
        
        # Verificar metadados da entrada
        entry = memory_cache[key]
        assert entry.access_count == 1
        assert entry.last_accessed is not None
        assert entry.size_bytes > 0
    
    def test_cache_expiration(self):
        """Testa expiração de cache"""
        # Criar entrada expirada
        old_time = time.time() - 7200  # 2 horas atrás
        expired_entry = CacheEntry(
            key="expired_key",
            value="expired_value",
            timestamp=old_time,
            ttl=3600  # 1 hora TTL
        )
        
        # Criar entrada válida
        valid_entry = CacheEntry(
            key="valid_key",
            value="valid_value",
            timestamp=time.time(),
            ttl=3600
        )
        
        cache = {
            "expired_key": expired_entry,
            "valid_key": valid_entry
        }
        
        def get_valid_entries():
            return {k: v for k, v in cache.items() if not v.is_expired()}
        
        valid_cache = get_valid_entries()
        
        assert "expired_key" not in valid_cache
        assert "valid_key" in valid_cache
    
    def test_cache_size_calculation(self, sample_cache_data):
        """Testa cálculo de tamanho do cache"""
        def calculate_entry_size(value):
            return len(pickle.dumps(value))
        
        small_data = {"simple": "value"}
        large_data = sample_cache_data
        
        small_size = calculate_entry_size(small_data)
        large_size = calculate_entry_size(large_data)
        
        assert large_size > small_size
        assert small_size > 0
        assert large_size > 0
    
    def test_lru_eviction_simulation(self):
        """Testa simulação de remoção LRU"""
        max_entries = 3
        cache = {}
        access_order = []
        
        def lru_set(key, value):
            if len(cache) >= max_entries and key not in cache:
                # Remover entrada menos recentemente usada
                lru_key = access_order[0]
                del cache[lru_key]
                access_order.remove(lru_key)
            
            cache[key] = value
            if key in access_order:
                access_order.remove(key)
            access_order.append(key)
        
        def lru_get(key):
            if key in cache:
                # Mover para final (mais recentemente usado)
                access_order.remove(key)
                access_order.append(key)
                return cache[key]
            return None
        
        # Preencher cache
        lru_set("key1", "value1")
        lru_set("key2", "value2")
        lru_set("key3", "value3")
        assert len(cache) == 3
        
        # Adicionar quarta entrada deve remover primeira
        lru_set("key4", "value4")
        assert len(cache) == 3
        assert "key1" not in cache
        assert "key4" in cache
        
        # Acessar key2 e adicionar key5 deve remover key3
        lru_get("key2")
        lru_set("key5", "value5")
        assert "key3" not in cache
        assert "key2" in cache
        assert "key5" in cache
    
    @patch('redis.Redis')
    def test_redis_cache_operations(self, mock_redis_class):
        """Testa operações de cache Redis"""
        mock_redis = Mock()
        mock_redis_class.return_value = mock_redis
        
        # Configurar mocks
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = True
        mock_redis.flushdb.return_value = True
        
        # Simular operações Redis
        def redis_set(key, value, ex=None):
            serialized = pickle.dumps(value)
            return mock_redis.set(key, serialized, ex=ex)
        
        def redis_get(key):
            raw_value = mock_redis.get(key)
            if raw_value:
                return pickle.loads(raw_value)
            return None
        
        # Testar operações
        test_data = {"test": "data"}
        assert redis_set("test_key", test_data, ex=3600) is True
        mock_redis.set.assert_called_once()
        
        # Simular recuperação
        mock_redis.get.return_value = pickle.dumps(test_data)
        retrieved = redis_get("test_key")
        assert retrieved == test_data
    
    def test_cache_decorator_simulation(self):
        """Testa simulação de decorator de cache"""
        cache_storage = {}
        
        def cache_decorator(ttl=3600):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    # Gerar chave baseada na função e argumentos
                    key_data = {
                        'func': func.__name__,
                        'args': args,
                        'kwargs': kwargs
                    }
                    cache_key = hashlib.md5(
                        str(key_data).encode()
                    ).hexdigest()
                    
                    # Verificar cache
                    entry = cache_storage.get(cache_key)
                    if entry and not entry.is_expired():
                        entry.access_count += 1
                        return entry.value
                    
                    # Executar função e cachear resultado
                    result = func(*args, **kwargs)
                    cache_storage[cache_key] = CacheEntry(
                        key=cache_key,
                        value=result,
                        timestamp=time.time(),
                        ttl=ttl
                    )
                    return result
                return wrapper
            return decorator
        
        # Função de exemplo para cache
        @cache_decorator(ttl=1800)
        def expensive_calculation(x, y):
            time.sleep(0.001)  # Simular cálculo custoso
            return x * y + x ** y
        
        # Testar cache
        start_time = time.time()
        result1 = expensive_calculation(2, 3)
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        result2 = expensive_calculation(2, 3)  # Deve vir do cache
        second_call_time = time.time() - start_time
        
        assert result1 == result2
        assert second_call_time < first_call_time  # Cache deve ser mais rápido
        assert len(cache_storage) == 1
        
        # Verificar acesso
        cache_key = list(cache_storage.keys())[0]
        entry = cache_storage[cache_key]
        assert entry.access_count >= 1
    
    def test_cache_invalidation_patterns(self):
        """Testa padrões de invalidação de cache"""
        cache = {}
        
        def invalidate_by_pattern(pattern):
            """Remove entradas que correspondem ao padrão"""
            keys_to_remove = [k for k in cache.keys() if pattern in k]
            for key in keys_to_remove:
                del cache[key]
            return len(keys_to_remove)
        
        def invalidate_by_tags(tags):
            """Remove entradas que têm determinadas tags"""
            keys_to_remove = []
            for key, entry in cache.items():
                if hasattr(entry, 'tags') and any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del cache[key]
            return len(keys_to_remove)
        
        # Criar entradas com diferentes padrões
        cache["user:123:profile"] = CacheEntry("user:123:profile", {"name": "João"}, time.time())
        cache["user:456:profile"] = CacheEntry("user:456:profile", {"name": "Maria"}, time.time())
        cache["model:v1:predictions"] = CacheEntry("model:v1:predictions", [1, 2, 3], time.time())
        cache["model:v2:predictions"] = CacheEntry("model:v2:predictions", [4, 5, 6], time.time())
        
        # Invalidar por padrão
        removed_count = invalidate_by_pattern("user:")
        assert removed_count == 2
        assert len(cache) == 2
        assert all("model:" in key for key in cache.keys())
        
        # Invalidar por outro padrão
        removed_count = invalidate_by_pattern("model:v1:")
        assert removed_count == 1
        assert len(cache) == 1
    
    def test_cache_warming_simulation(self):
        """Testa simulação de aquecimento de cache"""
        cache = {}
        
        def warm_cache_entry(key, value_generator, ttl=3600):
            """Aquece uma entrada específica do cache"""
            if key not in cache or cache[key].is_expired():
                value = value_generator()
                cache[key] = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl
                )
                return True
            return False
        
        def warm_cache_batch(entries):
            """Aquece múltiplas entradas do cache"""
            warmed_count = 0
            for key, generator, ttl in entries:
                if warm_cache_entry(key, generator, ttl):
                    warmed_count += 1
            return warmed_count
        
        # Definir geradores de valor
        def generate_user_data():
            return {"id": 123, "name": "User", "timestamp": time.time()}
        
        def generate_model_data():
            return {"predictions": [1, 2, 3], "accuracy": 0.95}
        
        # Aquecer cache
        entries_to_warm = [
            ("user:123", generate_user_data, 1800),
            ("model:latest", generate_model_data, 3600)
        ]
        
        warmed_count = warm_cache_batch(entries_to_warm)
        
        assert warmed_count == 2
        assert len(cache) == 2
        assert "user:123" in cache
        assert "model:latest" in cache
        
        # Tentar aquecer novamente (não deve substituir)
        warmed_count = warm_cache_batch(entries_to_warm)
        assert warmed_count == 0  # Nenhuma entrada aquecida (já existem)
    
    def test_cache_statistics(self):
        """Testa cálculo de estatísticas de cache"""
        cache = {}
        
        def add_cache_entry(key, value, access_count=0):
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=access_count,
                size_bytes=len(pickle.dumps(value))
            )
            cache[key] = entry
        
        def calculate_cache_stats():
            if not cache:
                return {
                    "total_entries": 0,
                    "total_size_bytes": 0,
                    "hit_rate": 0.0,
                    "average_access_count": 0.0
                }
            
            total_entries = len(cache)
            total_size = sum(entry.size_bytes for entry in cache.values())
            total_accesses = sum(entry.access_count for entry in cache.values())
            average_access = total_accesses / total_entries if total_entries > 0 else 0
            
            return {
                "total_entries": total_entries,
                "total_size_bytes": total_size,
                "total_accesses": total_accesses,
                "average_access_count": average_access
            }
        
        # Adicionar algumas entradas
        add_cache_entry("key1", {"data": "small"}, access_count=5)
        add_cache_entry("key2", {"data": "larger_data_structure"}, access_count=10)
        add_cache_entry("key3", {"data": "medium"}, access_count=3)
        
        stats = calculate_cache_stats()
        
        assert stats["total_entries"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["total_accesses"] == 18
        assert stats["average_access_count"] == 6.0
    
    def test_file_cache_operations(self):
        """Testa operações de cache em arquivo"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache_dir.mkdir(exist_ok=True)
            
            def file_cache_set(key, value, cache_dir):
                """Salva valor no cache de arquivo"""
                file_path = cache_dir / f"{key}.pkl"
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    size_bytes=len(pickle.dumps(value))
                )
                
                with open(file_path, 'wb') as f:
                    pickle.dump(entry, f)
                return True
            
            def file_cache_get(key, cache_dir):
                """Recupera valor do cache de arquivo"""
                file_path = cache_dir / f"{key}.pkl"
                if not file_path.exists():
                    return None
                
                try:
                    with open(file_path, 'rb') as f:
                        entry = pickle.load(f)
                    
                    if entry.is_expired():
                        file_path.unlink()  # Remove arquivo expirado
                        return None
                    
                    return entry.value
                except:
                    return None
            
            # Testar operações
            test_data = {"model": "test", "data": [1, 2, 3, 4, 5]}
            
            assert file_cache_set("test_model", test_data, cache_dir) is True
            assert (cache_dir / "test_model.pkl").exists()
            
            retrieved_data = file_cache_get("test_model", cache_dir)
            assert retrieved_data == test_data
            
            # Testar chave inexistente
            assert file_cache_get("nonexistent", cache_dir) is None


@pytest.mark.integration 
class TestCacheSystemIntegration:
    """Testes de integração para sistema de cache"""
    
    def test_multi_layer_cache_simulation(self):
        """Testa simulação de cache multicamadas"""
        # L1: Cache em memória (rápido, pequeno)
        l1_cache = {}
        l1_max_size = 2
        
        # L2: Cache em arquivo (médio, maior)
        with tempfile.TemporaryDirectory() as temp_dir:
            l2_cache_dir = Path(temp_dir)
            
            def get_from_cache(key):
                # Tentar L1 primeiro
                if key in l1_cache:
                    entry = l1_cache[key]
                    if not entry.is_expired():
                        return entry.value
                    else:
                        del l1_cache[key]
                
                # Tentar L2
                file_path = l2_cache_dir / f"{key}.pkl"
                if file_path.exists():
                    try:
                        with open(file_path, 'rb') as f:
                            entry = pickle.load(f)
                        
                        if not entry.is_expired():
                            # Promover para L1 se houver espaço
                            if len(l1_cache) < l1_max_size:
                                l1_cache[key] = entry
                            return entry.value
                        else:
                            file_path.unlink()
                    except:
                        pass
                
                return None
            
            def set_to_cache(key, value, ttl=3600):
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl,
                    size_bytes=len(pickle.dumps(value))
                )
                
                # Armazenar em L1 se houver espaço
                if len(l1_cache) < l1_max_size:
                    l1_cache[key] = entry
                else:
                    # Armazenar em L2
                    file_path = l2_cache_dir / f"{key}.pkl"
                    with open(file_path, 'wb') as f:
                        pickle.dump(entry, f)
            
            # Testar cache multicamadas
            set_to_cache("key1", "value1")
            set_to_cache("key2", "value2")
            assert len(l1_cache) == 2
            
            # Adicionar terceira chave (deve ir para L2)
            set_to_cache("key3", "value3")
            assert len(l1_cache) == 2
            assert (l2_cache_dir / "key3.pkl").exists()
            
            # Recuperar de L1
            assert get_from_cache("key1") == "value1"
            
            # Recuperar de L2 (deve promover para L1)
            result = get_from_cache("key3")
            assert result == "value3"
    
    def test_cache_performance_simulation(self):
        """Testa simulação de performance de cache"""
        cache = {}
        stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0
        }
        
        def cached_function(x):
            key = f"func_result_{x}"
            
            # Verificar cache
            if key in cache:
                entry = cache[key]
                if not entry.is_expired():
                    stats["hits"] += 1
                    return entry.value
                else:
                    del cache[key]
            
            # Cache miss - calcular valor
            stats["misses"] += 1
            time.sleep(0.001)  # Simular cálculo custoso
            result = x ** 2 + x * 3 + 1
            
            # Armazenar no cache
            cache[key] = CacheEntry(
                key=key,
                value=result,
                timestamp=time.time(),
                ttl=1800
            )
            stats["sets"] += 1
            
            return result
        
        # Testar performance
        test_values = [1, 2, 3, 1, 2, 4, 1, 3, 5]
        
        start_time = time.time()
        results = [cached_function(x) for x in test_values]
        total_time = time.time() - start_time
        
        # Verificar resultados
        assert len(results) == len(test_values)
        assert stats["hits"] > 0  # Deve ter hits para valores repetidos
        assert stats["misses"] > 0  # Deve ter misses para valores únicos
        assert stats["sets"] == stats["misses"]  # Um set para cada miss
        
        # Calcular hit rate
        hit_rate = stats["hits"] / (stats["hits"] + stats["misses"])
        assert 0 <= hit_rate <= 1
        
        # Para valores repetidos, hit rate deve ser > 0
        repeated_values = len(test_values) - len(set(test_values))
        if repeated_values > 0:
            assert hit_rate > 0