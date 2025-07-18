#!/usr/bin/env python3
"""
Exemplo de uso do sistema de timeout aprimorado.
Este arquivo demonstra como usar os timeouts configuráveis por serviço.
"""

import asyncio
from src.services.base_service import BaseService, ServiceRequest

class ExampleService(BaseService):
    """Serviço de exemplo para demonstrar o uso de timeouts."""
    
    async def initialize(self):
        """Inicialização do serviço."""
        print(f"Initializing {self.service_info.name}")
        
        # Configurar timeouts customizados para serviços específicos
        self.configure_service_timeout("slow_ml_service", 300)  # 5 minutos
        self.configure_service_timeout("fast_cache_service", 5)  # 5 segundos
        self.configure_service_timeout("custom_api", 60)  # 1 minuto
        
        print("Timeout configuration:")
        timeout_info = self.get_timeout_info()
        for service, timeout in timeout_info["service_timeouts"].items():
            print(f"  {service}: {timeout}s")
    
    async def cleanup(self):
        """Limpeza do serviço."""
        print(f"Cleaning up {self.service_info.name}")
    
    async def demonstrate_timeout_usage(self):
        """Demonstra diferentes formas de usar timeouts."""
        
        print("\n=== Demonstração de Timeouts ===\n")
        
        # 1. Usar timeout padrão do serviço (baseado no nome)
        print("1. Usando timeout padrão baseado no nome do serviço:")
        ml_request = self.create_service_request(
            service_name="ml_service",
            method="POST",
            endpoint="/predict",
            payload={"data": "sample"}
        )
        print(f"   ml_service request timeout: {ml_request.timeout}s")
        
        # 2. Usar timeout customizado configurado
        print("2. Usando timeout customizado configurado:")
        slow_request = self.create_service_request(
            service_name="slow_ml_service",
            method="POST",
            endpoint="/train",
            payload={"model": "complex"}
        )
        print(f"   slow_ml_service request timeout: {slow_request.timeout}s")
        
        # 3. Sobrescrever timeout explicitamente
        print("3. Sobrescrevendo timeout explicitamente:")
        custom_request = self.create_service_request(
            service_name="ml_service",
            method="GET",
            endpoint="/quick_status",
            payload={},
            timeout=10  # Override para operação rápida
        )
        print(f"   ml_service override timeout: {custom_request.timeout}s")
        
        # 4. Serviço não configurado (usa default)
        print("4. Serviço não configurado (usa default):")
        unknown_request = self.create_service_request(
            service_name="unknown_service",
            method="GET",
            endpoint="/info",
            payload={}
        )
        print(f"   unknown_service timeout: {unknown_request.timeout}s")
        
        # 5. Requisição manual com ServiceRequest
        print("5. Requisição manual com ServiceRequest:")
        manual_request = ServiceRequest(
            service_name="manual_service",
            method="POST",
            endpoint="/process",
            payload={"task": "heavy"},
            timeout=120  # 2 minutos explícitos
        )
        print(f"   manual_service timeout: {manual_request.timeout}s")
        
        print("\n=== Configurações de Timeout Atuais ===")
        timeout_info = self.get_timeout_info()
        print(f"Default timeout: {timeout_info['default_timeout']}s")
        print("Examples:")
        for service, description in timeout_info["timeout_examples"].items():
            print(f"  {service}: {description}")
        
        print(f"\nConfigured services: {', '.join(timeout_info['configured_services'])}")

async def main():
    """Função principal de demonstração."""
    # Criar serviço de exemplo
    service = ExampleService("example_service", "1.0.0")
    
    try:
        # Inicializar serviço
        await service.start()
        
        # Demonstrar uso de timeouts
        await service.demonstrate_timeout_usage()
        
    finally:
        # Parar serviço
        await service.stop()

if __name__ == "__main__":
    print("=== Demonstração do Sistema de Timeout Aprimorado ===")
    print("Este exemplo mostra como configurar e usar timeouts específicos por serviço.")
    print()
    
    asyncio.run(main())