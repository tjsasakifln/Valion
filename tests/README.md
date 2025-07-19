# Testes do Valion

Este diretório contém a suite completa de testes para o sistema Valion.

## Estrutura de Testes

```
tests/
├── unit/                    # Testes unitários
│   ├── core/               # Testes para módulos core
│   ├── services/           # Testes para services
│   ├── auth/               # Testes para autenticação
│   └── monitoring/         # Testes para monitoramento
├── integration/            # Testes de integração
├── conftest.py            # Configurações globais
└── README.md              # Este arquivo
```

## Tipos de Testes

### Testes Unitários
- **Localização**: `tests/unit/`
- **Propósito**: Testar funções e classes isoladamente
- **Execução**: `pytest tests/unit/`

### Testes de Integração
- **Localização**: `tests/integration/`
- **Propósito**: Testar interação entre componentes
- **Execução**: `pytest tests/integration/`

## Executando os Testes

### Comandos Básicos

```bash
# Todos os testes
pytest

# Apenas testes unitários
pytest tests/unit/

# Apenas testes de integração
pytest tests/integration/

# Com coverage
pytest --cov=src --cov-report=html

# Testes específicos
pytest tests/unit/core/test_results_generator.py

# Com verbose
pytest -v

# Parar no primeiro erro
pytest -x
```

### Usando o Makefile

```bash
# Instalar dependências de desenvolvimento
make install-dev

# Executar todos os testes
make test

# Apenas testes unitários
make test-unit

# Testes com coverage
make test-coverage

# Verificar qualidade do código
make check-quality
```

## Configuração de Coverage

O projeto está configurado para:
- **Meta de coverage**: 80%
- **Relatórios**: HTML, XML e terminal
- **Diretório HTML**: `htmlcov/`

### Arquivo .coveragerc

```ini
[run]
source = src
omit = */tests/*, */__pycache__/*

[report]
show_missing = True
skip_covered = False
precision = 2
```

## Fixtures Globais

### conftest.py

```python
@pytest.fixture
def sample_dataframe():
    """DataFrame de exemplo para testes"""
    return pd.DataFrame({
        'area': [100, 150, 200],
        'quartos': [2, 3, 4],
        'preco': [300000, 450000, 600000]
    })
```

## Marcadores de Teste

- `@pytest.mark.unit`: Testes unitários
- `@pytest.mark.integration`: Testes de integração
- `@pytest.mark.slow`: Testes lentos
- `@pytest.mark.requires_redis`: Requer Redis
- `@pytest.mark.requires_db`: Requer banco de dados

### Exemplo de uso:

```python
@pytest.mark.unit
def test_calculation():
    assert calculate_value(100) == 200

@pytest.mark.integration
@pytest.mark.requires_db
def test_database_integration():
    # Teste que requer banco
    pass
```

## Mocks e Fixtures

### Exemplo de Mock

```python
from unittest.mock import Mock, patch

@patch('src.core.model_builder.ModelBuilder')
def test_with_mock(mock_builder):
    mock_builder.train_model.return_value = "success"
    # Teste usando o mock
```

### Fixture de Dados

```python
@pytest.fixture
def sample_model_result():
    """Resultado de modelo de exemplo"""
    result = Mock()
    result.r2_score = 0.85
    result.rmse = 45000.0
    return result
```

## Testes Assíncronos

Para testar código assíncrono:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

## Testes de Performance

```python
def test_performance(benchmark):
    result = benchmark(expensive_function, arg1, arg2)
    assert result is not None
```

## Configuração de CI/CD

### GitHub Actions

```yaml
- name: Run tests
  run: |
    pytest tests/ --cov=src --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v1
```

## Padrões de Teste

### Estrutura de Teste

```python
class TestClassName:
    """Testes para ClassName"""
    
    @pytest.fixture
    def instance(self):
        """Fixture para instância da classe"""
        return ClassName()
    
    def test_method_name_scenario(self, instance):
        """Testa method_name em cenário específico"""
        # Arrange
        input_data = "test_data"
        
        # Act
        result = instance.method_name(input_data)
        
        # Assert
        assert result == expected_value
```

### Nomenclatura

- **Arquivos**: `test_*.py`
- **Classes**: `TestClassName`
- **Métodos**: `test_method_name_scenario`
- **Fixtures**: `nome_descritivo`

## Debugging

### Debugging com pdb

```python
def test_with_debug():
    import pdb; pdb.set_trace()
    # Código do teste
```

### Saída detalhada

```bash
pytest -v -s  # -s para mostrar prints
```

## Cobertura por Módulo

### Core Modules
- `results_generator.py`: 95%+
- `data_loader.py`: 90%+
- `cache_system.py`: 85%+

### Services
- `ml_service.py`: 90%+
- `auth_service.py`: 95%+

### Monitoring
- `metrics.py`: 85%+

## Contribuindo

1. **Novos testes**: Sempre adicione testes para código novo
2. **Coverage**: Mantenha coverage acima de 80%
3. **Nomenclatura**: Siga os padrões estabelecidos
4. **Documentação**: Documente fixtures e testes complexos

## Troubleshooting

### Problemas Comuns

1. **Import errors**: Verifique PYTHONPATH
2. **Mock não funciona**: Verifique patch path
3. **Fixtures não encontradas**: Verifique conftest.py
4. **Testes lentos**: Use marca `@pytest.mark.slow`

### Performance

```bash
# Testes mais rápidos primeiro
pytest --ff

# Apenas testes que falharam
pytest --lf

# Parallel execution
pytest -n 4  # Requer pytest-xdist
```