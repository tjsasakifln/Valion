#!/usr/bin/env python3
"""
Demonstração do Sistema de Cache Inteligente do Valion

Este script mostra como o sistema de cache pode melhorar significativamente
a performance da plataforma Valion.
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.cache_system import CacheManager, get_default_cache_config
from core.geospatial_analysis import GeospatialAnalyzer
from core.model_builder import ModelBuilder
from config.settings import Settings


def demonstrate_geospatial_cache():
    """Demonstra o cache geoespacial."""
    print("=== DEMONSTRAÇÃO DO CACHE GEOESPACIAL ===")
    
    # Configurar analisador com cache
    analyzer = GeospatialAnalyzer(cache_enabled=True)
    
    # Endereços para teste
    test_addresses = [
        "Rua Augusta, 1000, São Paulo, SP",
        "Avenida Paulista, 2000, São Paulo, SP",
        "Rua Oscar Freire, 500, São Paulo, SP",
        "Rua Augusta, 1000, São Paulo, SP",  # Repetido para demonstrar cache
        "Avenida Paulista, 2000, São Paulo, SP"  # Repetido para demonstrar cache
    ]
    
    print("Primeira execução (sem cache):")
    start_time = time.time()
    results_1 = []
    for address in test_addresses:
        coords = analyzer.geocode_address(address)
        if coords:
            proximity = analyzer.calculate_proximity_score(coords)
            results_1.append((address, coords, proximity))
            print(f"  {address}: {coords} (proximity: {proximity:.2f})")
    first_run_time = time.time() - start_time
    
    print(f"\nTempo da primeira execução: {first_run_time:.2f}s")
    
    print("\nSegunda execução (com cache):")
    start_time = time.time()
    results_2 = []
    for address in test_addresses:
        coords = analyzer.geocode_address(address)
        if coords:
            proximity = analyzer.calculate_proximity_score(coords)
            results_2.append((address, coords, proximity))
            print(f"  {address}: {coords} (proximity: {proximity:.2f})")
    second_run_time = time.time() - start_time
    
    print(f"\nTempo da segunda execução: {second_run_time:.2f}s")
    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    
    return results_1, results_2


def demonstrate_model_cache():
    """Demonstra o cache de modelos."""
    print("\n=== DEMONSTRAÇÃO DO CACHE DE MODELOS ===")
    
    # Gerar dataset sintético
    np.random.seed(42)
    n_samples = 1000
    
    # Features básicas
    area = np.random.normal(100, 30, n_samples)
    quartos = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    banheiros = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    idade = np.random.uniform(0, 50, n_samples)
    
    # Features geoespaciais simuladas
    distance_to_center = np.random.exponential(10, n_samples)
    proximity_score = np.random.uniform(0, 10, n_samples)
    transport_score = np.random.uniform(0, 10, n_samples)
    
    # Criar target com relação realista
    valor = (area * 3000 + 
             quartos * 50000 + 
             banheiros * 25000 - 
             idade * 1000 +
             (10 - distance_to_center) * 2000 +
             proximity_score * 5000 +
             transport_score * 3000 +
             np.random.normal(0, 20000, n_samples))
    
    # Criar DataFrame
    df = pd.DataFrame({
        'area': area,
        'quartos': quartos,
        'banheiros': banheiros,
        'idade': idade,
        'distance_to_center': distance_to_center,
        'proximity_score': proximity_score,
        'transport_score': transport_score,
        'valor': valor
    })
    
    print(f"Dataset criado com {len(df)} amostras")
    print(f"Features: {list(df.columns[:-1])}")
    
    # Configuração do modelo
    config = {
        'model_type': 'elastic_net',
        'expert_mode': False,
        'model_cache_enabled': True,
        'model_cache_dir': 'cache_demo_models',
        'max_cached_models': 10
    }
    
    print("\nPrimeiro treinamento (sem cache):")
    start_time = time.time()
    model_builder1 = ModelBuilder(config)
    result1 = model_builder1.build_model(df)
    first_training_time = time.time() - start_time
    
    print(f"  R² Score: {result1.performance.r2_score:.4f}")
    print(f"  RMSE: {result1.performance.rmse:.0f}")
    print(f"  Tempo de treinamento: {first_training_time:.2f}s")
    
    print("\nSegundo treinamento (com cache - mesmo dataset):")
    start_time = time.time()
    model_builder2 = ModelBuilder(config)
    result2 = model_builder2.build_model(df)
    second_training_time = time.time() - start_time
    
    print(f"  R² Score: {result2.performance.r2_score:.4f}")
    print(f"  RMSE: {result2.performance.rmse:.0f}")
    print(f"  Tempo de treinamento: {second_training_time:.2f}s")
    
    speedup = first_training_time / second_training_time if second_training_time > 0 else float('inf')
    print(f"  Speedup: {speedup:.2f}x")
    
    # Verificar se os resultados são consistentes
    r2_diff = abs(result1.performance.r2_score - result2.performance.r2_score)
    print(f"  Diferença no R²: {r2_diff:.6f} (deve ser próximo de 0)")
    
    return result1, result2


def demonstrate_cache_statistics():
    """Demonstra estatísticas do cache."""
    print("\n=== ESTATÍSTICAS DO CACHE ===")
    
    # Criar gerenciador de cache
    cache_config = get_default_cache_config()
    cache_manager = CacheManager(cache_config)
    
    # Obter estatísticas
    stats = cache_manager.get_cache_stats()
    
    print("Estatísticas do cache:")
    print(f"  Tipo de backend: {stats['backend_type']}")
    print(f"  TTL geoespacial: {stats['geospatial_ttl']}s")
    
    if 'model_cache' in stats:
        model_stats = stats['model_cache']
        print(f"  Modelos em cache: {model_stats['total_models']}")
        print(f"  Tamanho total: {model_stats['total_size_mb']:.2f} MB")
        print(f"  Tipos de modelo: {model_stats['model_types']}")
    
    return stats


def performance_comparison():
    """Compara performance com e sem cache."""
    print("\n=== COMPARAÇÃO DE PERFORMANCE ===")
    
    # Simular operações repetitivas
    coordinates_list = [
        (-23.5505, -46.6333),  # São Paulo centro
        (-23.5489, -46.6388),  # Próximo ao centro
        (-23.5574, -46.7311),  # USP
        (-23.5476, -46.6567),  # Ibirapuera
        (-23.5505, -46.6333),  # Repetido
    ]
    
    print("Teste de performance com múltiplas consultas geoespaciais:")
    
    # Sem cache
    analyzer_no_cache = GeospatialAnalyzer(cache_enabled=False)
    start_time = time.time()
    for coords in coordinates_list * 5:  # 25 consultas
        analyzer_no_cache.calculate_proximity_score(coords)
    no_cache_time = time.time() - start_time
    
    # Com cache
    analyzer_with_cache = GeospatialAnalyzer(cache_enabled=True)
    start_time = time.time()
    for coords in coordinates_list * 5:  # 25 consultas
        analyzer_with_cache.calculate_proximity_score(coords)
    with_cache_time = time.time() - start_time
    
    print(f"  Sem cache: {no_cache_time:.3f}s")
    print(f"  Com cache: {with_cache_time:.3f}s")
    speedup = no_cache_time / with_cache_time if with_cache_time > 0 else float('inf')
    print(f"  Speedup: {speedup:.2f}x")
    
    return no_cache_time, with_cache_time


def main():
    """Função principal da demonstração."""
    print("DEMONSTRAÇÃO DO SISTEMA DE CACHE INTELIGENTE DO VALION")
    print("=" * 60)
    
    try:
        # Demonstrar cache geoespacial
        demonstrate_geospatial_cache()
        
        # Demonstrar cache de modelos
        demonstrate_model_cache()
        
        # Mostrar estatísticas
        demonstrate_cache_statistics()
        
        # Comparação de performance
        performance_comparison()
        
        print("\n" + "=" * 60)
        print("RESUMO DOS BENEFÍCIOS DO CACHE:")
        print("• Redução significativa no tempo de geocodificação")
        print("• Reutilização de modelos treinados similares")
        print("• Economia de recursos computacionais")
        print("• Melhoria na experiência do usuário")
        print("• Escalabilidade aprimorada")
        
    except Exception as e:
        print(f"Erro durante a demonstração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()