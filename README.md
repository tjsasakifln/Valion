# Valion - Real Estate Evaluation Platform

[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](https://github.com/tjsasakifln/Valion/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)](https://www.docker.com/)

> **Advanced Glass-Box Real Estate Valuation Platform** - Transparent, auditable, and statistically rigorous property evaluation following NBR 14653 standards.

## 🎯 Key Features

✨ **Complete Transparency** - Every step of the evaluation process is auditable and explainable  
📊 **NBR 14653 Compliance** - Full battery of statistical tests meeting Brazilian technical standards  
⚡ **Real-time Processing** - Asynchronous architecture with live progress updates  
🔬 **Scientific Rigor** - Elastic Net regression with comprehensive statistical validation  
🏗️ **Enterprise Ready** - Scalable microservices architecture with Docker containerization  
📈 **Advanced Analytics** - SHAP analysis, geospatial intelligence, and interactive dashboards  

## 🚀 Quick Start

### 🐳 Docker (Recommended)

```bash
git clone https://github.com/tjsasakifln/Valion.git
cd Valion
cp .env.example .env
docker-compose up --build
```

**Access Points:**
- 🖥️ **Frontend**: http://localhost:8501
- 🔌 **API**: http://localhost:8000  
- 📊 **API Docs**: http://localhost:8000/docs
- 🌺 **Task Monitor**: http://localhost:5555

### 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start services (requires 3 terminals)
# Terminal 1: API Server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Celery Worker
celery -A src.workers.tasks worker --loglevel=info

# Terminal 3: Streamlit Frontend  
streamlit run frontend.py --server.port 8501
```

## 🏗️ Architecture

### Technology Stack
- **Frontend**: Streamlit (Interactive UI)
- **Backend**: FastAPI (REST API + WebSocket)
- **Workers**: Celery (Asynchronous processing)
- **Database**: PostgreSQL
- **Cache/Broker**: Redis
- **Containerization**: Docker + Docker Compose

### Project Structure
```
Valion/
├── src/
│   ├── core/                     # Analytics Engine
│   │   ├── data_loader.py        # Phase 1: Data ingestion & validation
│   │   ├── transformations.py    # Phase 2: Feature engineering
│   │   ├── model_builder.py      # Phase 3: Elastic Net modeling
│   │   ├── nbr14653_validation.py # Phase 4: NBR 14653 validation
│   │   ├── results_generator.py  # Phase 5: Report generation
│   │   └── geospatial_analysis.py # Geospatial intelligence
│   ├── api/main.py              # FastAPI application
│   ├── workers/tasks.py         # Celery background tasks
│   ├── websocket/               # Real-time communication
│   └── config/settings.py       # Centralized configuration
├── frontend.py                  # Streamlit interface
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Service orchestration
└── README.md                    # This file
```

## 🔬 Technical Methodology

### Statistical Model
- **Algorithm**: Elastic Net Regression (L1 + L2 regularization)
- **Validation**: 5-fold cross-validation
- **Optimization**: Grid search for hyperparameters
- **Interpretability**: SHAP (SHapley Additive exPlanations) values

### 5-Phase Evaluation Process

#### 📥 Phase 1: Data Ingestion & Validation
- Multi-format support (CSV, Excel, JSON)
- Automated data quality assessment
- Outlier detection and handling
- Missing data analysis and imputation

#### 🔧 Phase 2: Feature Engineering  
- Automated feature transformation
- Categorical variable encoding
- Feature selection with statistical tests
- Data normalization and scaling

#### 🤖 Phase 3: Model Training
- Elastic Net regression with hyperparameter optimization
- Cross-validation for model generalization
- Feature importance analysis
- Performance metric calculation

#### ✅ Phase 4: NBR 14653 Validation
- **R² Test**: Coefficient of determination
- **F-Test**: Overall model significance
- **t-Test**: Individual coefficient significance  
- **Shapiro-Wilk**: Residual normality
- **Durbin-Watson**: Autocorrelation detection
- **VIF**: Multicollinearity assessment

#### 📋 Phase 5: Report Generation
- Comprehensive statistical analysis
- Visual diagnostics and charts
- Technical recommendations
- Multi-format export (PDF, Excel, JSON)

## 📊 NBR 14653 Precision Levels

| Grade | Minimum R² | Description |
|-------|------------|-------------|
| **Superior** | ≥ 0.90 | Excellent explanatory capacity |
| **Normal** | ≥ 0.80 | Good explanatory capacity |
| **Inferior** | ≥ 0.70 | Adequate explanatory capacity |
| **Inadequate** | < 0.70 | Insufficient explanatory capacity |

## 🧪 Statistical Tests

### Core Validation Battery

1. **Coefficient of Determination (R²)**
   - Measures proportion of variance explained
   - Primary criterion for NBR classification

2. **F-Test for Significance**
   - Tests overall model significance
   - H₀: All coefficients equal zero

3. **t-Test for Coefficients**
   - Tests individual coefficient significance
   - H₀: Specific coefficient equals zero

4. **Shapiro-Wilk Normality Test**
   - Verifies residual normality assumption
   - H₀: Residuals follow normal distribution

5. **Durbin-Watson Autocorrelation**
   - Detects serial correlation in residuals
   - Values 1.5-2.5 indicate no autocorrelation

6. **Variance Inflation Factor (VIF)**
   - Measures multicollinearity severity
   - VIF < 10 indicates acceptable collinearity

## 📈 Performance Metrics

### Primary Metrics
- **R²**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error

### Cross-Validation
- **5-Fold CV**: Robust generalization assessment
- **Stability Analysis**: Performance consistency across folds

## 🔧 Configuration

### Environment Variables

Key configuration options (see `.env.example`):

```bash
# Application
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-secret-key

# Database
DATABASE_URL=postgresql://user:pass@localhost/valion
REDIS_URL=redis://localhost:6379

# Model Parameters
ELASTIC_NET_ALPHA=1.0
ELASTIC_NET_L1_RATIO=0.5
CV_FOLDS=5

# NBR Thresholds
NBR_R2_SUPERIOR=0.90
NBR_R2_NORMAL=0.80
NBR_R2_INFERIOR=0.70
```

### Advanced Configuration

The `src/config/settings.py` file enables detailed configuration of:
- Model hyperparameters and validation thresholds
- API security and rate limiting settings
- Logging levels and audit trail options
- Database connection pooling
- Celery worker configurations

## 🛡️ Security & Compliance

### Security Measures
- Input validation and sanitization
- SQL injection prevention
- XSS protection with Content Security Policy
- Rate limiting and DDoS protection
- Secure file upload handling
- Audit logging for all operations

### Best Practices
- Non-root container execution
- Pydantic type validation
- Environment-based secret management
- Secure communication protocols
- Regular dependency updates

## 📋 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/evaluations/` | Start new evaluation |
| `GET` | `/evaluations/{id}` | Get evaluation status |
| `GET` | `/evaluations/{id}/result` | Retrieve evaluation results |
| `POST` | `/evaluations/{id}/predict` | Make predictions |
| `POST` | `/upload` | Upload data files |
| `GET` | `/health` | Health check |

### WebSocket
- `WS /ws/{evaluation_id}`: Real-time progress updates

📖 **Interactive Documentation**: http://localhost:8000/docs

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_model_builder.py -v

# Run integration tests
pytest tests/integration/ -v
```

## 📦 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   ENVIRONMENT=production
   DEBUG=false
   SECRET_KEY=your-production-secret-key
   ```

2. **Docker Deployment**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Reverse Proxy Configuration** (nginx/traefik recommended)

### Monitoring & Observability

- **Application Logs**: Structured JSON logging
- **Metrics**: Prometheus/Grafana integration ready
- **Health Checks**: Kubernetes-compatible endpoints
- **Task Monitoring**: Flower dashboard for Celery
- **Error Tracking**: Sentry integration available

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **Business Source License 1.1** (BSL-1.1).

- ✅ **Free for development, testing, and non-production use**
- ✅ **Open source with full transparency**
- ⏰ **Converts to Apache 2.0 after 4 years**
- 📋 **Commercial use requires separate license**

See [LICENSE](LICENSE) file for complete details.

## 🔗 Resources

### Documentation
- [NBR 14653 Standards](https://www.abnt.org.br/normalizacao/lista-de-normas/nbr)
- [Elastic Net Regression](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)
- [SHAP Documentation](https://shap.readthedocs.io/)

### Framework Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Celery Documentation](https://docs.celeryproject.org/)

## 📞 Support

### Getting Help
- 📖 Check our [Documentation](docs/)
- 🐛 [Report Issues](https://github.com/tjsasakifln/Valion/issues)
- 💬 [Discussions](https://github.com/tjsasakifln/Valion/discussions)
- 📧 Contact: tiago@confenge.com.br

### Professional Services
For enterprise support, custom development, or consulting services, please contact our team.

---

<div align="center">

**🏠 Valion** - *Transparency and Statistical Rigor in Real Estate Evaluation*

Made with ❤️ for the real estate and data science community

[⭐ Star this repo](https://github.com/tjsasakifln/Valion) | [🐛 Report Bug](https://github.com/tjsasakifln/Valion/issues) | [💡 Request Feature](https://github.com/tjsasakifln/Valion/issues)

</div>