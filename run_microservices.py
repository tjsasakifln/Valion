#!/usr/bin/env python3
"""
Script para executar os microserviços do Valion
Permite executar todos os serviços ou serviços específicos.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.orchestrator import create_orchestrator, OrchestrationConfig
from src.services.api_gateway import create_api_gateway, get_default_gateway_config
from src.services.data_processing_service import create_data_processing_service
from src.services.ml_service import create_ml_service
from src.services.service_registry import ServiceRegistry
from src.monitoring.logging_config import setup_structured_logging
from src.monitoring.metrics import init_metrics_collector
import structlog


async def run_orchestrator(args):
    """Executa o orquestrador completo."""
    print("🚀 Starting Valion Microservices Orchestrator")
    print("=" * 50)
    
    config = OrchestrationConfig(
        redis_url=args.redis_url,
        enable_service_registry=not args.no_registry,
        enable_api_gateway=not args.no_gateway,
        enable_metrics=not args.no_metrics,
        enable_logging=not args.no_logging
    )
    
    orchestrator = await create_orchestrator(config)
    
    try:
        async with orchestrator.lifespan():
            print("✅ All services started successfully!")
            print("\nService URLs:")
            print(f"  • API Gateway: http://localhost:8000")
            print(f"  • Data Processing: http://localhost:8001")
            print(f"  • ML Service: http://localhost:8002")
            print(f"  • Metrics: http://localhost:9090")
            print("\nPress Ctrl+C to stop all services...")
            
            await orchestrator.shutdown_event.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


async def run_single_service(service_name: str, args):
    """Executa um serviço específico."""
    print(f"🚀 Starting {service_name}")
    print("=" * 50)
    
    # Configurar logging
    if not args.no_logging:
        setup_structured_logging({
            "level": "INFO",
            "console_handler": True,
            "file_handler": True,
            "development_mode": True
        })
    
    logger = structlog.get_logger(service_name)
    
    try:
        if service_name == "api_gateway":
            config = get_default_gateway_config()
            config.host = "0.0.0.0"
            config.port = 8000
            config.debug = args.debug
            
            service = create_api_gateway(config)
            await service.start()
            
            print(f"✅ {service_name} started at http://localhost:8000")
            
        elif service_name == "data_processing":
            service = create_data_processing_service("localhost", 8001)
            await service.start()
            
            print(f"✅ {service_name} started at http://localhost:8001")
            
        elif service_name == "ml_service":
            service = create_ml_service("localhost", 8002)
            await service.start()
            
            print(f"✅ {service_name} started at http://localhost:8002")
            
        elif service_name == "geospatial_service":
            import uvicorn
            print(f"✅ {service_name} starting at http://localhost:8101")
            await uvicorn.run(
                "src.microservices.geospatial_service.main:app",
                host="0.0.0.0",
                port=8101,
                log_level="info" if not args.debug else "debug"
            )
            
        elif service_name == "reporting_service":
            import uvicorn
            print(f"✅ {service_name} starting at http://localhost:8102")
            await uvicorn.run(
                "src.microservices.reporting_service.main:app",
                host="0.0.0.0",
                port=8102,
                log_level="info" if not args.debug else "debug"
            )
            
        elif service_name == "service_registry":
            service = ServiceRegistry(redis_url=args.redis_url)
            await service.start()
            
            print(f"✅ {service_name} started")
            
        else:
            print(f"❌ Unknown service: {service_name}")
            return
        
        print("Press Ctrl+C to stop the service...")
        
        # Aguardar interrupção
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping {service_name}...")
            await service.stop()
            print(f"✅ {service_name} stopped")
            
    except Exception as e:
        logger.error(f"Error running {service_name}: {e}")
        raise


async def show_status(args):
    """Mostra status dos serviços."""
    print("📊 Valion Services Status")
    print("=" * 50)
    
    import aiohttp
    import json
    
    services = [
        ("API Gateway", "http://localhost:8000/health"),
        ("Data Processing", "http://localhost:8001/health"),
        ("ML Service", "http://localhost:8002/health"),
        ("Metrics", "http://localhost:9090/metrics")
    ]
    
    async with aiohttp.ClientSession() as session:
        for name, url in services:
            try:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        print(f"✅ {name}: Running")
                        
                        if args.verbose and name != "Metrics":
                            data = await response.json()
                            print(f"   Status: {data.get('status', 'unknown')}")
                            print(f"   Uptime: {data.get('uptime', 0):.1f}s")
                    else:
                        print(f"⚠️  {name}: Unhealthy (HTTP {response.status})")
                        
            except Exception as e:
                print(f"❌ {name}: Not responding ({str(e)})")


async def run_tests(args):
    """Executa testes dos microserviços."""
    print("🧪 Running Microservices Tests")
    print("=" * 50)
    
    import aiohttp
    import json
    
    # Teste básico do API Gateway
    print("\n1. Testing API Gateway...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                if response.status == 200:
                    print("✅ API Gateway health check passed")
                else:
                    print(f"❌ API Gateway health check failed: {response.status}")
                    
            async with session.get("http://localhost:8000/services") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Found {len(data.get('services', []))} registered services")
                else:
                    print(f"❌ Services endpoint failed: {response.status}")
                    
    except Exception as e:
        print(f"❌ API Gateway test failed: {e}")
    
    # Teste do Data Processing Service
    print("\n2. Testing Data Processing Service...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8001/health") as response:
                if response.status == 200:
                    print("✅ Data Processing service health check passed")
                else:
                    print(f"❌ Data Processing service health check failed: {response.status}")
                    
    except Exception as e:
        print(f"❌ Data Processing service test failed: {e}")
    
    # Teste do ML Service
    print("\n3. Testing ML Service...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8002/health") as response:
                if response.status == 200:
                    print("✅ ML service health check passed")
                else:
                    print(f"❌ ML service health check failed: {response.status}")
                    
            async with session.get("http://localhost:8002/models") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Found {len(data)} models in ML service")
                else:
                    print(f"❌ ML models endpoint failed: {response.status}")
                    
    except Exception as e:
        print(f"❌ ML service test failed: {e}")
    
    print("\n✅ Tests completed!")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Valion Microservices Runner")
    
    # Comando principal
    parser.add_argument(
        "command",
        choices=["orchestrator", "api_gateway", "data_processing", "ml_service", "geospatial_service", "reporting_service", "service_registry", "status", "test"],
        help="Command to run"
    )
    
    # Opções gerais
    parser.add_argument("--redis-url", default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Opções do orquestrador
    parser.add_argument("--no-registry", action="store_true", help="Disable service registry")
    parser.add_argument("--no-gateway", action="store_true", help="Disable API gateway")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics")
    parser.add_argument("--no-logging", action="store_true", help="Disable structured logging")
    
    args = parser.parse_args()
    
    # Executar comando
    try:
        if args.command == "orchestrator":
            asyncio.run(run_orchestrator(args))
        elif args.command == "status":
            asyncio.run(show_status(args))
        elif args.command == "test":
            asyncio.run(run_tests(args))
        else:
            asyncio.run(run_single_service(args.command, args))
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()