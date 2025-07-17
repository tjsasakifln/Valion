"""
Configurações centralizadas para a plataforma Valion
Gerencia todas as configurações do sistema de forma unificada.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging


@dataclass
class DatabaseSettings:
    """Configurações de banco de dados."""
    host: str = "localhost"
    port: int = 5432
    name: str = "valion"
    user: str = "valion_user"
    password: str = ""
    url: Optional[str] = None
    
    def __post_init__(self):
        if not self.url:
            self.url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RedisSettings:
    """Configurações do Redis."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    url: Optional[str] = None
    
    def __post_init__(self):
        if not self.url:
            auth = f":{self.password}@" if self.password else ""
            self.url = f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class CelerySettings:
    """Configurações do Celery."""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    task_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    result_serializer: str = "json"
    timezone: str = "UTC"
    enable_utc: bool = True
    task_track_started: bool = True
    task_time_limit: int = 3600
    task_soft_time_limit: int = 3300
    worker_prefetch_multiplier: int = 1
    worker_max_tasks_per_child: int = 1000


@dataclass
class ModelSettings:
    """Configurações do modelo."""
    model_type: str = "elastic_net"
    default_target_column: str = "valor"
    min_sample_size: int = 30
    test_size: float = 0.2
    cv_folds: int = 5
    max_features: int = 20
    random_state: int = 42
    
    # Parâmetros Elastic Net
    alpha_range: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    l1_ratio_range: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    max_iter: int = 1000
    
    # Thresholds NBR 14653
    r2_superior: float = 0.90
    r2_normal: float = 0.80
    r2_inferior: float = 0.70
    f_test_p_value: float = 0.05
    t_test_p_value: float = 0.05
    durbin_watson_lower: float = 1.5
    durbin_watson_upper: float = 2.5
    max_vif: float = 10.0
    max_outliers_percent: float = 5.0


@dataclass
class APISettings:
    """Configurações da API."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_extensions: List[str] = field(default_factory=lambda: [".csv", ".xlsx", ".xls"])


@dataclass
class StreamlitSettings:
    """Configurações do Streamlit."""
    port: int = 8501
    host: str = "0.0.0.0"
    theme: str = "light"
    page_title: str = "Valion - Avaliação Imobiliária"
    page_icon: str = "🏠"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    max_upload_size: int = 200  # MB


@dataclass
class LoggingSettings:
    """Configurações de logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_file: str = "logs/valion.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class StorageSettings:
    """Configurações de armazenamento."""
    upload_dir: str = "uploads"
    models_dir: str = "models"
    reports_dir: str = "reports"
    temp_dir: str = "temp"
    logs_dir: str = "logs"
    cleanup_days: int = 7


@dataclass
class SecuritySettings:
    """Configurações de segurança."""
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_hash_algorithm: str = "bcrypt"
    password_rounds: int = 12


class Settings:
    """Classe principal de configurações."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa configurações.
        
        Args:
            config_file: Caminho para arquivo de configuração personalizado
        """
        self._load_from_env()
        
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        self._setup_logging()
        self._create_directories()
    
    def _load_from_env(self):
        """Carrega configurações de variáveis de ambiente."""
        
        # Database
        self.database = DatabaseSettings(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "valion"),
            user=os.getenv("DB_USER", "valion_user"),
            password=os.getenv("DB_PASSWORD", ""),
            url=os.getenv("DATABASE_URL")
        )
        
        # Redis
        self.redis = RedisSettings(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            url=os.getenv("REDIS_URL")
        )
        
        # Celery
        self.celery = CelerySettings(
            broker_url=os.getenv("CELERY_BROKER_URL", self.redis.url),
            result_backend=os.getenv("CELERY_RESULT_BACKEND", self.redis.url)
        )
        
        # Model
        self.model = ModelSettings(
            min_sample_size=int(os.getenv("MIN_SAMPLE_SIZE", "30")),
            test_size=float(os.getenv("TEST_SIZE", "0.2")),
            cv_folds=int(os.getenv("CV_FOLDS", "5")),
            max_features=int(os.getenv("MAX_FEATURES", "20")),
            random_state=int(os.getenv("RANDOM_STATE", "42"))
        )
        
        # API
        self.api = APISettings(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            debug=os.getenv("API_DEBUG", "false").lower() == "true"
        )
        
        # Streamlit
        self.streamlit = StreamlitSettings(
            port=int(os.getenv("STREAMLIT_PORT", "8501")),
            host=os.getenv("STREAMLIT_HOST", "0.0.0.0"),
            theme=os.getenv("STREAMLIT_THEME", "light")
        )
        
        # Logging
        self.logging = LoggingSettings(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file_handler=os.getenv("LOG_FILE_HANDLER", "true").lower() == "true",
            console_handler=os.getenv("LOG_CONSOLE_HANDLER", "true").lower() == "true",
            log_file=os.getenv("LOG_FILE", "logs/valion.log")
        )
        
        # Storage
        self.storage = StorageSettings(
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            models_dir=os.getenv("MODELS_DIR", "models"),
            reports_dir=os.getenv("REPORTS_DIR", "reports"),
            temp_dir=os.getenv("TEMP_DIR", "temp"),
            logs_dir=os.getenv("LOGS_DIR", "logs"),
            cleanup_days=int(os.getenv("CLEANUP_DAYS", "7"))
        )
        
        # Security
        self.security = SecuritySettings(
            secret_key=os.getenv("SECRET_KEY", "your-secret-key-here"),
            algorithm=os.getenv("ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            refresh_token_expire_days=int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        )
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Propriedades de conveniência para compatibilidade
        self.CELERY_BROKER_URL = self.celery.broker_url
        self.CELERY_RESULT_BACKEND = self.celery.result_backend
    
    def _load_from_file(self, config_file: str):
        """
        Carrega configurações de arquivo JSON.
        
        Args:
            config_file: Caminho para arquivo de configuração
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Atualizar configurações com dados do arquivo
            for section, values in config_data.items():
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                            
        except Exception as e:
            logging.warning(f"Erro ao carregar arquivo de configuração {config_file}: {e}")
    
    def _setup_logging(self):
        """Configura sistema de logging."""
        
        # Criar diretório de logs
        log_dir = Path(self.storage.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, self.logging.level),
            format=self.logging.format,
            handlers=[]
        )
        
        logger = logging.getLogger()
        logger.handlers.clear()
        
        # Handler para console
        if self.logging.console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(self.logging.format))
            logger.addHandler(console_handler)
        
        # Handler para arquivo
        if self.logging.file_handler:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.log_file,
                maxBytes=self.logging.max_bytes,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            logger.addHandler(file_handler)
    
    def _create_directories(self):
        """Cria diretórios necessários."""
        directories = [
            self.storage.upload_dir,
            self.storage.models_dir,
            self.storage.reports_dir,
            self.storage.temp_dir,
            self.storage.logs_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Retorna configurações do modelo em formato dict.
        
        Returns:
            Configurações do modelo
        """
        return {
            'min_sample_size': self.model.min_sample_size,
            'test_size': self.model.test_size,
            'cv_folds': self.model.cv_folds,
            'max_features': self.model.max_features,
            'random_state': self.model.random_state,
            'alpha_range': self.model.alpha_range,
            'l1_ratio_range': self.model.l1_ratio_range,
            'max_iter': self.model.max_iter,
            'r2_superior': self.model.r2_superior,
            'r2_normal': self.model.r2_normal,
            'r2_inferior': self.model.r2_inferior,
            'f_test_p_value': self.model.f_test_p_value,
            't_test_p_value': self.model.t_test_p_value,
            'durbin_watson_lower': self.model.durbin_watson_lower,
            'durbin_watson_upper': self.model.durbin_watson_upper,
            'max_vif': self.model.max_vif,
            'max_outliers_percent': self.model.max_outliers_percent
        }
    
    def save_to_file(self, filepath: str):
        """
        Salva configurações em arquivo JSON.
        
        Args:
            filepath: Caminho para salvar arquivo
        """
        config_data = {
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'celery': self.celery.__dict__,
            'model': self.model.__dict__,
            'api': self.api.__dict__,
            'streamlit': self.streamlit.__dict__,
            'logging': self.logging.__dict__,
            'storage': self.storage.__dict__,
            'security': self.security.__dict__,
            'environment': self.environment,
            'debug': self.debug
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def validate(self) -> List[str]:
        """
        Valida configurações.
        
        Returns:
            Lista de erros encontrados
        """
        errors = []
        
        # Validar configurações do modelo
        if self.model.min_sample_size < 10:
            errors.append("Tamanho mínimo da amostra deve ser >= 10")
        
        if not (0.1 <= self.model.test_size <= 0.5):
            errors.append("Tamanho do teste deve estar entre 0.1 e 0.5")
        
        if self.model.cv_folds < 3:
            errors.append("Número de folds deve ser >= 3")
        
        # Validar thresholds NBR
        if not (0.0 <= self.model.r2_inferior <= self.model.r2_normal <= self.model.r2_superior <= 1.0):
            errors.append("Thresholds de R² devem estar em ordem crescente entre 0 e 1")
        
        # Validar configurações de API
        if not (1024 <= self.api.port <= 65535):
            errors.append("Porta da API deve estar entre 1024 e 65535")
        
        # Validar configurações de segurança
        if len(self.security.secret_key) < 32:
            errors.append("Secret key deve ter pelo menos 32 caracteres")
        
        return errors
    
    def __str__(self) -> str:
        """Representação string das configurações."""
        return f"Settings(environment={self.environment}, debug={self.debug})"
    
    def __repr__(self) -> str:
        """Representação detalhada das configurações."""
        return self.__str__()


# Instância global das configurações
settings = Settings()

# Função para recarregar configurações
def reload_settings(config_file: Optional[str] = None) -> Settings:
    """
    Recarrega configurações.
    
    Args:
        config_file: Arquivo de configuração opcional
        
    Returns:
        Nova instância das configurações
    """
    global settings
    settings = Settings(config_file)
    return settings