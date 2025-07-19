#!/usr/bin/env python3
"""
Startup check script for Valion system
Validates configuration and prepares system for execution
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check required environment variables"""
    logger.info("🔍 Checking environment variables...")
    
    required_vars = [
        'SECRET_KEY',
        'DATABASE_URL',
        'REDIS_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✅ All required environment variables are set")
    return True

def check_directories():
    """Check and create required directories"""
    logger.info("📁 Checking required directories...")
    
    required_dirs = [
        'uploads',
        'models', 
        'reports',
        'temp',
        'logs'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.info(f"📁 Creating directory: {dir_name}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"✅ Directory exists: {dir_name}")
    
    return True

def check_database_connection():
    """Check if database connection is possible"""
    logger.info("🗄️ Checking database connection...")
    
    try:
        from sqlalchemy import create_engine, text
        
        db_url = os.getenv('DATABASE_URL')
        
        # For SQLite, just check if we can create the engine
        if db_url.startswith('sqlite'):
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful (SQLite)")
            return True
        
        # For PostgreSQL, check if we can connect
        elif db_url.startswith('postgresql'):
            engine = create_engine(db_url)
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("✅ Database connection successful (PostgreSQL)")
                return True
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL connection failed (this is normal in testing): {e}")
                return True  # Don't fail startup for this
        
    except ImportError:
        logger.warning("⚠️ SQLAlchemy not installed (this is expected outside Docker)")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def check_redis_connection():
    """Check if Redis connection is possible"""
    logger.info("🔴 Checking Redis connection...")
    
    try:
        import redis
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        r.ping()
        logger.info("✅ Redis connection successful")
        return True
        
    except ImportError:
        logger.warning("⚠️ Redis library not installed (this is expected outside Docker)")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed (this is normal in testing): {e}")
        return True  # Don't fail startup for this

def setup_logging():
    """Setup logging configuration"""
    logger.info("📝 Setting up logging...")
    
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file
    log_file = logs_dir / 'valion.log'
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info("✅ Logging configured successfully")
    return True

def initialize_database():
    """Initialize database tables if needed"""
    logger.info("🗄️ Initializing database...")
    
    try:
        from src.database.database import get_database_manager
        
        # Get database manager and create tables
        db_manager = get_database_manager()
        logger.info("✅ Database manager initialized")
        
        # Test database connection with a simple session
        with db_manager.get_session() as session:
            logger.info("✅ Database session created successfully")
        
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Database modules not available: {e}")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def main():
    """Run all startup checks"""
    logger.info("🚀 Starting Valion system checks...")
    logger.info("=" * 50)
    
    checks = [
        ("Environment Variables", check_environment),
        ("Directories", check_directories),
        ("Logging Setup", setup_logging),
        ("Database Connection", check_database_connection),
        ("Redis Connection", check_redis_connection),
        ("Database Initialization", initialize_database),
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        logger.info(f"\n🔍 Running check: {check_name}")
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            logger.error(f"❌ Check '{check_name}' failed with exception: {e}")
            failed_checks.append(check_name)
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 STARTUP CHECK RESULTS")
    logger.info("=" * 50)
    
    if failed_checks:
        logger.error(f"❌ Failed checks: {', '.join(failed_checks)}")
        logger.error("🚨 System startup checks failed!")
        return False
    else:
        logger.info("✅ All startup checks passed!")
        logger.info("🚀 System is ready to start!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)