# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Configuração de Logging Estruturado
Sistema de logging avançado com suporte a logs estruturados e contextuais.
"""

import logging
import logging.handlers
import structlog
import sys
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path
import threading
import traceback


# Configuração global do structlog
_configured = False
_lock = threading.Lock()


class ValionLogRenderer:
    """Renderizador customizado para logs do Valion."""
    
    def __init__(self, colors: bool = True):
        self.colors = colors
        self.color_codes = {
            'debug': '\033[36m',     # Cyan
            'info': '\033[32m',      # Green
            'warning': '\033[33m',   # Yellow
            'error': '\033[31m',     # Red
            'critical': '\033[35m',  # Magenta
            'reset': '\033[0m'       # Reset
        }
    
    def __call__(self, logger, name, event_dict):
        """Renderiza log entry."""
        timestamp = datetime.now().isoformat()
        level = event_dict.get('level', 'info').upper()
        event = event_dict.get('event', '')
        
        # Extrair contexto
        context = {k: v for k, v in event_dict.items() 
                  if k not in ['level', 'event', 'timestamp']}
        
        # Formatação colorida para console
        if self.colors and sys.stdout.isatty():
            color = self.color_codes.get(level.lower(), '')
            reset = self.color_codes['reset']
            formatted_level = f"{color}{level:<8}{reset}"
        else:
            formatted_level = f"{level:<8}"
        
        # Linha principal
        main_line = f"{timestamp} {formatted_level} {event}"
        
        # Adicionar contexto se existir
        if context:
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
            main_line += f"\n{context_str}"
        
        return main_line


class ValionJSONRenderer:
    """Renderizador JSON para logs estruturados."""
    
    def __call__(self, logger, name, event_dict):
        """Renderiza log entry como JSON."""
        # Adicionar timestamp se não existir
        if 'timestamp' not in event_dict:
            event_dict['timestamp'] = datetime.now().isoformat()
        
        # Adicionar informações do logger
        event_dict['logger_name'] = name
        
        return json.dumps(event_dict, ensure_ascii=False, default=str)


class ContextualFilter(logging.Filter):
    """Filtro para adicionar contexto aos logs."""
    
    def __init__(self):
        super().__init__()
        self.context_vars = {}
    
    def set_context(self, **kwargs):
        """Define variáveis de contexto."""
        self.context_vars.update(kwargs)
    
    def clear_context(self):
        """Limpa contexto."""
        self.context_vars.clear()
    
    def filter(self, record):
        """Adiciona contexto ao record."""
        for key, value in self.context_vars.items():
            setattr(record, key, value)
        return True


class LoggingConfig:
    """Configuração centralizada de logging."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.contextual_filter = ContextualFilter()
        self.loggers = {}
    
    def setup_logging(self):
        """Configura sistema de logging."""
        global _configured
        with _lock:
            if _configured:
                return
            
            # Configurar structlog
            self._setup_structlog()
            
            # Configurar handlers
            self._setup_handlers()
            
            # Configurar loggers específicos
            self._setup_loggers()
            
            _configured = True
    
    def _setup_structlog(self):
        """Configura structlog."""
        # Processadores comuns
        shared_processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        # Processadores específicos para desenvolvimento
        if self.config.get('development_mode', False):
            shared_processors.append(
                structlog.dev.ConsoleRenderer(colors=True)
            )
        else:
            shared_processors.append(ValionJSONRenderer())
        
        # Configurar structlog
        structlog.configure(
            processors=shared_processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def _setup_handlers(self):
        """Configura handlers de logging."""
        # Handler para console
        if self.config.get('console_handler', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            if self.config.get('development_mode', False):
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            else:
                formatter = logging.Formatter(
                    '{"timestamp": "%(asctime)s", "logger": "%(name)s", '
                    '"level": "%(levelname)s", "message": "%(message)s"}'
                )
            
            console_handler.setFormatter(formatter)
            console_handler.addFilter(self.contextual_filter)
            
            # Adicionar ao root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(console_handler)
        
        # Handler para arquivo
        if self.config.get('file_handler', True):
            log_dir = Path(self.config.get('log_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / self.config.get('log_file', 'valion.log')
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_bytes', 10 * 1024 * 1024),
                backupCount=self.config.get('backup_count', 5)
            )
            file_handler.setLevel(logging.DEBUG)
            
            # Sempre usar JSON para arquivos
            json_formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "logger": "%(name)s", '
                '"level": "%(levelname)s", "message": "%(message)s", '
                '"module": "%(module)s", "function": "%(funcName)s", '
                '"line": %(lineno)d}'
            )
            
            file_handler.setFormatter(json_formatter)
            file_handler.addFilter(self.contextual_filter)
            
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
        
        # Handler para erros críticos
        if self.config.get('error_handler', True):
            error_dir = Path(self.config.get('log_dir', 'logs'))
            error_dir.mkdir(parents=True, exist_ok=True)
            
            error_file = error_dir / 'errors.log'
            
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=5 * 1024 * 1024,
                backupCount=3
            )
            error_handler.setLevel(logging.ERROR)
            
            error_formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "logger": "%(name)s", '
                '"level": "%(levelname)s", "message": "%(message)s", '
                '"module": "%(module)s", "function": "%(funcName)s", '
                '"line": %(lineno)d, "exception": "%(exc_info)s"}'
            )
            
            error_handler.setFormatter(error_formatter)
            error_handler.addFilter(self.contextual_filter)
            
            root_logger = logging.getLogger()
            root_logger.addHandler(error_handler)
    
    def _setup_loggers(self):
        """Configura loggers específicos."""
        # Logger principal do Valion
        valion_logger = logging.getLogger('valion')
        valion_logger.setLevel(logging.INFO)
        
        # Logger para métricas
        metrics_logger = logging.getLogger('valion.metrics')
        metrics_logger.setLevel(logging.INFO)
        
        # Logger para cache
        cache_logger = logging.getLogger('valion.cache')
        cache_logger.setLevel(logging.INFO)
        
        # Logger para API
        api_logger = logging.getLogger('valion.api')
        api_logger.setLevel(logging.INFO)
        
        # Logger para workers
        worker_logger = logging.getLogger('valion.worker')
        worker_logger.setLevel(logging.INFO)
        
        # Logger para banco de dados
        db_logger = logging.getLogger('valion.database')
        db_logger.setLevel(logging.INFO)
        
        # Configurar nível global
        root_logger = logging.getLogger()
        root_logger.setLevel(
            getattr(logging, self.config.get('level', 'INFO').upper())
        )
        
        # Silenciar loggers barulhentos
        noisy_loggers = [
            'urllib3.connectionpool',
            'requests.packages.urllib3.connectionpool',
            'werkzeug',
            'celery.worker.strategy',
            'celery.worker.consumer',
            'celery.app.trace'
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def set_context(self, **kwargs):
        """Define contexto para logs."""
        self.contextual_filter.set_context(**kwargs)
    
    def clear_context(self):
        """Limpa contexto."""
        self.contextual_filter.clear_context()
    
    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Obtém logger estruturado."""
        if name not in self.loggers:
            self.loggers[name] = structlog.get_logger(name)
        return self.loggers[name]


# Instância global
_logging_config = None


def setup_structured_logging(config: Dict[str, Any]) -> LoggingConfig:
    """
    Configura sistema de logging estruturado.
    
    Args:
        config: Configuração de logging
        
    Returns:
        Instância configurada
    """
    global _logging_config
    _logging_config = LoggingConfig(config)
    _logging_config.setup_logging()
    return _logging_config


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Obtém logger estruturado.
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger estruturado
    """
    if _logging_config is None:
        # Configuração padrão se não foi configurado
        default_config = {
            'level': 'INFO',
            'console_handler': True,
            'file_handler': True,
            'error_handler': True,
            'development_mode': True,
            'log_dir': 'logs',
            'log_file': 'valion.log'
        }
        setup_structured_logging(default_config)
    
    return _logging_config.get_logger(name)


def set_logging_context(**kwargs):
    """Define contexto para logs."""
    if _logging_config:
        _logging_config.set_context(**kwargs)


def clear_logging_context():
    """Limpa contexto de logs."""
    if _logging_config:
        _logging_config.clear_context()


def log_exception(logger: structlog.BoundLogger, exc: Exception, 
                 context: Optional[Dict[str, Any]] = None):
    """
    Registra exceção com contexto completo.
    
    Args:
        logger: Logger a ser usado
        exc: Exceção
        context: Contexto adicional
    """
    error_context = {
        'exception_type': type(exc).__name__,
        'exception_message': str(exc),
        'traceback': traceback.format_exc(),
        **(context or {})
    }
    
    logger.error("Exception occurred", **error_context)


def log_performance(logger: structlog.BoundLogger, operation: str, 
                   duration: float, context: Optional[Dict[str, Any]] = None):
    """
    Registra métricas de performance.
    
    Args:
        logger: Logger a ser usado
        operation: Nome da operação
        duration: Duração em segundos
        context: Contexto adicional
    """
    perf_context = {
        'operation': operation,
        'duration_seconds': duration,
        'performance_log': True,
        **(context or {})
    }
    
    logger.info("Performance metric", **perf_context)


def log_business_event(logger: structlog.BoundLogger, event: str, 
                      context: Optional[Dict[str, Any]] = None):
    """
    Registra evento de negócio.
    
    Args:
        logger: Logger a ser usado
        event: Nome do evento
        context: Contexto adicional
    """
    business_context = {
        'business_event': event,
        'event_timestamp': datetime.now().isoformat(),
        **(context or {})
    }
    
    logger.info("Business event", **business_context)


def get_default_logging_config(environment: str = 'development') -> Dict[str, Any]:
    """
    Obtém configuração padrão de logging.
    
    Args:
        environment: Ambiente (development, staging, production)
        
    Returns:
        Configuração de logging
    """
    base_config = {
        'console_handler': True,
        'file_handler': True,
        'error_handler': True,
        'log_dir': 'logs',
        'log_file': 'valion.log',
        'max_bytes': 10 * 1024 * 1024,
        'backup_count': 5
    }
    
    if environment == 'development':
        base_config.update({
            'level': 'DEBUG',
            'development_mode': True,
        })
    elif environment == 'staging':
        base_config.update({
            'level': 'INFO',
            'development_mode': False,
        })
    else:  # production
        base_config.update({
            'level': 'INFO',
            'development_mode': False,
        })
    
    return base_config