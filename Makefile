# Makefile para automação de testes e desenvolvimento

.PHONY: help test test-unit test-integration test-coverage clean lint format install-dev

# Variáveis
PYTHON := python
PIP := pip
PYTEST := pytest

help: ## Mostrar esta ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-dev: ## Instalar dependências de desenvolvimento
	$(PIP) install -r requirements/dev.txt
	$(PIP) install -e .

test: ## Executar todos os testes
	$(PYTEST) tests/ -v

test-unit: ## Executar apenas testes unitários
	$(PYTEST) tests/unit/ -v -m "not integration"

test-integration: ## Executar apenas testes de integração
	$(PYTEST) tests/integration/ -v -m "integration"

test-coverage: ## Executar testes com coverage
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=xml --cov-report=term-missing

test-fast: ## Executar testes rápidos (sem integração)
	$(PYTEST) tests/unit/ -v -x --ff

test-watch: ## Executar testes em modo watch
	$(PYTEST) tests/ -v --looponfail

lint: ## Executar linting
	flake8 src tests
	black --check src tests

format: ## Formatar código
	black src tests
	isort src tests

clean: ## Limpar arquivos temporários
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

check-quality: ## Verificar qualidade do código
	$(PYTEST) tests/ --cov=src --cov-fail-under=80
	flake8 src tests
	black --check src tests

ci-test: ## Testes para CI/CD
	$(PYTEST) tests/ --cov=src --cov-report=xml --cov-fail-under=80 --junitxml=test-results.xml

benchmark: ## Executar testes de performance
	$(PYTEST) tests/ -v -m "benchmark" --benchmark-only

security-scan: ## Executar verificação de segurança
	bandit -r src/
	safety check

requirements-update: ## Atualizar requirements
	pip-compile requirements/base.in
	pip-compile requirements/dev.in

docker-test: ## Executar testes no Docker
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

docs-test: ## Testar documentação
	$(PYTEST) --doctest-modules src/

pre-commit: ## Executar verificações pré-commit
	pre-commit run --all-files