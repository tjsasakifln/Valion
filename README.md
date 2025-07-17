# Valion - Plataforma de Avaliação Imobiliária

Uma plataforma "caixa de vidro" para avaliação imobiliária com foco em transparência, auditabilidade e rigor estatístico, seguindo a norma NBR 14653.

## 🏗️ Arquitetura

### Arquitetura Desacoplada e Escalável

- **Frontend**: Streamlit (Thin client)
- **Backend**: FastAPI (API REST + WebSocket)
- **Workers**: Celery (Processamento assíncrono)
- **Database**: PostgreSQL
- **Cache/Broker**: Redis
- **Containerização**: Docker

### Estrutura do Projeto

```
Valion/
├── src/
│   ├── core/                    # Motor analítico
│   │   ├── data_loader.py       # Fase 1: Ingestão e validação
│   │   ├── transformations.py   # Fase 2: Transformação de variáveis
│   │   ├── model_builder.py     # Fase 3: Modelo Elastic Net
│   │   ├── nbr14653_validation.py # Fase 4: Validação NBR 14653
│   │   └── results_generator.py # Fase 5: Geração de relatórios
│   ├── api/
│   │   └── main.py             # API FastAPI
│   ├── workers/
│   │   └── tasks.py            # Tasks Celery
│   └── config/
│       └── settings.py         # Configurações centralizadas
├── frontend.py                 # Interface Streamlit
├── requirements.txt            # Dependências Python
├── Dockerfile                  # Containerização
├── docker-compose.yml          # Orquestração
├── .env.example               # Exemplo de variáveis de ambiente
└── README.md                  # Este arquivo
```

## 🎯 Funcionalidades Principais

### 1. Transparência Total ("Caixa de Vidro")
- Todos os passos do processo são auditáveis
- Relatórios detalhados com fundamentos estatísticos
- Metodologia baseada em princípios científicos sólidos

### 2. Conformidade NBR 14653
- Bateria completa de testes estatísticos
- Classificação automática do grau de precisão
- Validação rigorosa dos resultados

### 3. Processamento Assíncrono
- Interface responsiva com feedback em tempo real
- Processamento paralelo de tarefas computacionalmente intensivas
- Monitoramento de progresso via WebSocket

### 4. Escalabilidade
- Arquitetura microserviços
- Containerização com Docker
- Balanceamento de carga automático

## 🔬 Metodologia Técnica

### Modelo Estatístico
- **Algoritmo**: Elastic Net Regression
- **Regularização**: Combinação L1 (Lasso) + L2 (Ridge)
- **Validação**: Cross-validation 5-fold
- **Otimização**: Grid Search para hiperparâmetros

### Fases do Processo

1. **Ingestão e Validação**
   - Carregamento de dados (CSV, Excel)
   - Validação de qualidade
   - Detecção de outliers
   - Análise de completude

2. **Transformação de Variáveis**
   - Engenharia de features
   - Normalização
   - Codificação de variáveis categóricas
   - Seleção de features

3. **Modelagem**
   - Treinamento Elastic Net
   - Otimização de hiperparâmetros
   - Validação cruzada
   - Análise de performance

4. **Validação NBR 14653**
   - Teste de coeficiente de determinação (R²)
   - Teste F de significância
   - Teste t dos coeficientes
   - Teste de normalidade dos resíduos
   - Teste de autocorrelação (Durbin-Watson)
   - Teste de multicolinearidade (VIF)

5. **Geração de Relatórios**
   - Consolidação de resultados
   - Análise de conclusões
   - Recomendações técnicas
   - Exportação em múltiplos formatos

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.11+
- Docker e Docker Compose
- Redis
- PostgreSQL (opcional para desenvolvimento)

### Instalação Local

1. **Clone o repositório**
   ```bash
   git clone https://github.com/seu-usuario/valion.git
   cd valion
   ```

2. **Configure o ambiente**
   ```bash
   cp .env.example .env
   # Edite as variáveis de ambiente conforme necessário
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute os serviços**
   ```bash
   # Terminal 1: API
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   
   # Terminal 2: Worker Celery
   celery -A src.workers.tasks worker --loglevel=info
   
   # Terminal 3: Frontend
   streamlit run frontend.py --server.port 8501
   ```

### Instalação com Docker

1. **Construir e executar**
   ```bash
   docker-compose up --build
   ```

2. **Acessar a aplicação**
   - Frontend: http://localhost:8501
   - API: http://localhost:8000
   - Flower (Monitor): http://localhost:5555

## 📊 Níveis de Precisão NBR 14653

| Grau | R² Mínimo | Descrição |
|------|-----------|-----------|
| Superior | ≥ 0,90 | Excelente capacidade explanatória |
| Normal | ≥ 0,80 | Boa capacidade explanatória |
| Inferior | ≥ 0,70 | Capacidade explanatória adequada |
| Inadequado | < 0,70 | Capacidade explanatória insuficiente |

## 🧪 Testes Estatísticos

### Testes Implementados

1. **Coeficiente de Determinação (R²)**
   - Mede a proporção da variância explicada
   - Critério principal para classificação NBR

2. **Teste F de Significância**
   - Testa significância global do modelo
   - H₀: Todos os coeficientes são zero

3. **Teste t dos Coeficientes**
   - Testa significância individual dos coeficientes
   - H₀: Coeficiente específico é zero

4. **Teste de Normalidade (Shapiro-Wilk)**
   - Verifica normalidade dos resíduos
   - H₀: Resíduos seguem distribuição normal

5. **Teste de Autocorrelação (Durbin-Watson)**
   - Detecta correlação serial nos resíduos
   - Valores entre 1,5 e 2,5 indicam ausência de autocorrelação

6. **Teste de Multicolinearidade (VIF)**
   - Variance Inflation Factor
   - VIF < 10 indica ausência de multicolinearidade severa

## 📈 Métricas de Performance

### Métricas Principais

- **R² (Coeficiente de Determinação)**: Proporção da variância explicada
- **RMSE (Root Mean Square Error)**: Erro quadrático médio
- **MAE (Mean Absolute Error)**: Erro absoluto médio
- **MAPE (Mean Absolute Percentage Error)**: Erro percentual absoluto médio

### Validação Cruzada

- **K-Fold Cross-Validation**: 5 folds
- **Métricas**: RMSE médio e desvio padrão
- **Objetivo**: Avaliar generalização do modelo

## 🔧 Configuração

### Variáveis de Ambiente

Consulte `.env.example` para lista completa de variáveis configuráveis.

### Configuração Avançada

O arquivo `src/config/settings.py` permite configuração detalhada de:

- Parâmetros do modelo
- Thresholds NBR 14653
- Configurações de API
- Configurações de logging
- Configurações de segurança

## 🛡️ Segurança

### Medidas Implementadas

- Validação rigorosa de entrada
- Sanitização de dados
- Autenticação JWT (implementação futura)
- Isolamento de containers
- Logs de auditoria

### Boas Práticas

- Usuário não-root em containers
- Validação de tipos com Pydantic
- Tratamento seguro de arquivos
- Configurações sensíveis via variáveis de ambiente

## 📋 API Documentation

### Endpoints Principais

- `POST /evaluations/`: Inicia nova avaliação
- `GET /evaluations/{id}`: Status da avaliação
- `GET /evaluations/{id}/result`: Resultado da avaliação
- `POST /evaluations/{id}/predict`: Fazer predição
- `POST /upload`: Upload de arquivo
- `GET /health`: Health check

### WebSocket

- `WS /ws/{evaluation_id}`: Feedback em tempo real

Acesse http://localhost:8000/docs para documentação interativa.

## 🧪 Testes

```bash
# Executar todos os testes
pytest

# Executar com coverage
pytest --cov=src

# Executar testes específicos
pytest tests/test_model_builder.py
```

## 📦 Deployment

### Produção

1. **Configure variáveis de ambiente**
   ```bash
   ENVIRONMENT=production
   DEBUG=false
   SECRET_KEY=your-production-secret-key
   ```

2. **Execute com Docker Compose**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Configure proxy reverso** (nginx, traefik, etc.)

### Monitoramento

- **Logs**: Configuráveis via variáveis de ambiente
- **Métricas**: Integração com Prometheus/Grafana
- **Health Checks**: Endpoints de monitoramento
- **Flower**: Monitor Celery em tempo real

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

## 🔗 Links Úteis

- [NBR 14653](https://www.abnt.org.br/normalizacao/lista-de-normas/nbr)
- [Elastic Net Regression](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Celery Documentation](https://docs.celeryproject.org/)

## 📞 Suporte

Para dúvidas e suporte:
- Abra uma issue no GitHub
- Consulte a documentação
- Entre em contato com a equipe de desenvolvimento

---

**Valion** - Transparência e rigor estatístico em avaliação imobiliária 🏠📊