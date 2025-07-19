"""
Configuração global para testes do Valion
"""
import os
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configurar variáveis de ambiente para testes
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["SECRET_KEY"] = "test-secret-key-12345"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"


@pytest.fixture(scope="session")
def event_loop():
    """Criar loop de eventos para testes async"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame de exemplo para testes"""
    return pd.DataFrame({
        'area': [100, 150, 200, 120, 180],
        'quartos': [2, 3, 4, 2, 3],
        'preco': [300000, 450000, 600000, 350000, 520000],
        'latitude': [-23.5, -23.6, -23.4, -23.7, -23.3],
        'longitude': [-46.6, -46.7, -46.5, -46.8, -46.4]
    })


@pytest.fixture
def sample_geospatial_data() -> pd.DataFrame:
    """Dados geoespaciais de exemplo"""
    return pd.DataFrame({
        'id': range(1, 6),
        'lat': [-23.5505, -23.5629, -23.5433, -23.5751, -23.5329],
        'lng': [-46.6333, -46.6546, -46.6419, -46.6834, -46.6123],
        'value': [1000, 1200, 900, 1100, 950]
    })


@pytest.fixture
def mock_database_session():
    """Session mock do banco de dados"""
    session = Mock()
    session.query.return_value.filter.return_value.first.return_value = None
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    return session


@pytest.fixture
def mock_redis():
    """Mock do Redis"""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    return redis_mock


@pytest.fixture
def mock_model():
    """Mock de modelo ML"""
    model = Mock()
    model.predict.return_value = np.array([400000, 500000, 600000])
    model.fit.return_value = model
    model.score.return_value = 0.85
    return model


@pytest.fixture
def temp_file(tmp_path):
    """Arquivo temporário para testes"""
    file_path = tmp_path / "test_file.xlsx"
    return str(file_path)


@pytest.fixture
def sample_validation_data():
    """Dados para testes de validação"""
    return {
        'area': 120,
        'quartos': 3,
        'preco': 400000,
        'localizacao': 'São Paulo, SP',
        'tipo': 'apartamento'
    }


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Limpeza automática após cada teste"""
    yield
    # Limpar arquivos temporários, cache, etc.
    import tempfile
    import shutil
    temp_dir = tempfile.gettempdir()
    for item in os.listdir(temp_dir):
        if item.startswith('valion_test_'):
            path = os.path.join(temp_dir, item)
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)