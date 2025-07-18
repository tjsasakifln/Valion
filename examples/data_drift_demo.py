#!/usr/bin/env python3
"""
Demonstração do Sistema de Detecção de Data Drift
Mostra como usar os testes KS, PSI e detectores de anomalias.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitoring.data_drift import (
    DataQualityMonitor, 
    get_default_drift_config,
    KSTestDriftDetector,
    PSITestDriftDetector,
    AnomalyDetector
)


def generate_reference_data(n_samples: int = 1000) -> pd.DataFrame:
    """Gera dados de referência simulados."""
    np.random.seed(42)
    
    # Simular dados imobiliários
    data = {
        'area': np.random.normal(100, 25, n_samples),
        'quartos': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'banheiros': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'idade': np.random.exponential(10, n_samples),
        'preco_m2': np.random.normal(3000, 500, n_samples),
        'distancia_centro': np.random.exponential(8, n_samples),
        'renda_bairro': np.random.lognormal(8, 0.5, n_samples),
        'valor': np.random.normal(300000, 100000, n_samples)
    }
    
    # Adicionar algumas correlações realistas
    data['valor'] = (data['area'] * data['preco_m2'] + 
                     data['quartos'] * 20000 + 
                     data['banheiros'] * 15000 - 
                     data['idade'] * 2000 +
                     np.random.normal(0, 20000, n_samples))
    
    return pd.DataFrame(data)


def generate_drifted_data(reference_data: pd.DataFrame, drift_type: str = 'moderate') -> pd.DataFrame:
    """Gera dados com drift simulado."""
    n_samples = len(reference_data)
    data = reference_data.copy()
    
    if drift_type == 'moderate':
        # Drift moderado - mudança na média
        data['area'] = data['area'] + np.random.normal(15, 5, n_samples)
        data['preco_m2'] = data['preco_m2'] * 1.1 + np.random.normal(0, 100, n_samples)
        data['renda_bairro'] = data['renda_bairro'] * 1.05
        
    elif drift_type == 'severe':
        # Drift severo - mudança na distribuição
        data['area'] = np.random.normal(130, 30, n_samples)  # Mudança significativa
        data['preco_m2'] = np.random.normal(3500, 600, n_samples)
        data['quartos'] = np.random.choice([2, 3, 4, 5], n_samples, p=[0.3, 0.4, 0.2, 0.1])
        data['renda_bairro'] = np.random.lognormal(8.5, 0.6, n_samples)
        
    elif drift_type == 'anomalous':
        # Adicionar anomalias
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        data.loc[anomaly_indices, 'area'] = np.random.uniform(300, 500, len(anomaly_indices))
        data.loc[anomaly_indices, 'preco_m2'] = np.random.uniform(8000, 15000, len(anomaly_indices))
        data.loc[anomaly_indices, 'valor'] = np.random.uniform(1000000, 2000000, len(anomaly_indices))
    
    # Recalcular valor com as mudanças
    data['valor'] = (data['area'] * data['preco_m2'] + 
                     data['quartos'] * 20000 + 
                     data['banheiros'] * 15000 - 
                     data['idade'] * 2000 +
                     np.random.normal(0, 20000, n_samples))
    
    return data


def demonstrate_ks_test():
    """Demonstra teste Kolmogorov-Smirnov."""
    print("=== TESTE KOLMOGOROV-SMIRNOV ===")
    
    # Gerar dados
    reference_data = generate_reference_data(1000)
    moderate_drift = generate_drifted_data(reference_data, 'moderate')
    severe_drift = generate_drifted_data(reference_data, 'severe')
    
    # Criar detector
    ks_detector = KSTestDriftDetector(threshold=0.05)
    
    print("1. Teste com drift moderado:")
    results_moderate = ks_detector.detect_drift(reference_data, moderate_drift)
    
    for result in results_moderate:
        print(f"   {result.feature_name}: {result.interpretation}")
        if result.is_drift:
            print(f"   → {result.recommendation}")
    
    print("\n2. Teste com drift severo:")
    results_severe = ks_detector.detect_drift(reference_data, severe_drift)
    
    drift_count = 0
    for result in results_severe:
        if result.is_drift:
            drift_count += 1
            print(f"   {result.feature_name}: {result.interpretation}")
            print(f"   → {result.recommendation}")
    
    print(f"\nTotal de features com drift: {drift_count}/{len(results_severe)}")
    
    return results_moderate, results_severe


def demonstrate_psi_test():
    """Demonstra teste Population Stability Index."""
    print("\n=== TESTE POPULATION STABILITY INDEX (PSI) ===")
    
    # Gerar dados
    reference_data = generate_reference_data(1000)
    moderate_drift = generate_drifted_data(reference_data, 'moderate')
    severe_drift = generate_drifted_data(reference_data, 'severe')
    
    # Criar detector
    psi_detector = PSITestDriftDetector(threshold=0.1, bins=10)
    
    print("1. Teste com drift moderado:")
    results_moderate = psi_detector.detect_drift(reference_data, moderate_drift)
    
    for result in results_moderate:
        if result.is_drift:
            print(f"   {result.feature_name}: {result.interpretation}")
            print(f"   → {result.recommendation}")
    
    print("\n2. Teste com drift severo:")
    results_severe = psi_detector.detect_drift(reference_data, severe_drift)
    
    drift_count = 0
    for result in results_severe:
        if result.is_drift:
            drift_count += 1
            print(f"   {result.feature_name}: {result.interpretation}")
            print(f"   → {result.recommendation}")
    
    print(f"\nTotal de features com drift PSI: {drift_count}/{len(results_severe)}")
    
    return results_moderate, results_severe


def demonstrate_anomaly_detection():
    """Demonstra detecção de anomalias."""
    print("\n=== DETECÇÃO DE ANOMALIAS ===")
    
    # Gerar dados
    reference_data = generate_reference_data(1000)
    anomalous_data = generate_drifted_data(reference_data, 'anomalous')
    
    # Criar detector
    anomaly_detector = AnomalyDetector(methods=['isolation_forest', 'autoencoder'])
    
    print("Treinando detectores de anomalias...")
    anomaly_detector.fit(reference_data)
    
    print("Detectando anomalias...")
    anomaly_results = anomaly_detector.detect_anomalies(anomalous_data, threshold=0.5)
    
    # Analisar resultados
    anomalies = [r for r in anomaly_results if r.is_anomaly]
    severe_anomalies = [r for r in anomalies if r.anomaly_type == 'severe']
    
    print(f"\nTotal de amostras analisadas: {len(anomaly_results)}")
    print(f"Anomalias detectadas: {len(anomalies)} ({len(anomalies)/len(anomaly_results)*100:.1f}%)")
    print(f"Anomalias severas: {len(severe_anomalies)}")
    
    # Mostrar detalhes das anomalias mais severas
    if severe_anomalies:
        print("\nTop 5 anomalias severas:")
        severe_anomalies.sort(key=lambda x: x.anomaly_score, reverse=True)
        
        for i, anomaly in enumerate(severe_anomalies[:5]):
            print(f"\n{i+1}. Amostra {anomaly.sample_id}:")
            print(f"   Score: {anomaly.anomaly_score:.3f}")
            print(f"   Tipo: {anomaly.anomaly_type}")
            print(f"   Principais contribuições:")
            
            # Mostrar top 3 features contribuintes
            sorted_contributions = sorted(
                anomaly.feature_contributions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for feature, contribution in sorted_contributions[:3]:
                print(f"     {feature}: {contribution:.3f}")
    
    return anomaly_results


def demonstrate_complete_monitoring():
    """Demonstra monitoramento completo de qualidade."""
    print("\n=== MONITORAMENTO COMPLETO DE QUALIDADE ===")
    
    # Configurar monitor
    config = get_default_drift_config()
    monitor = DataQualityMonitor(config)
    
    # Dados de referência
    reference_data = generate_reference_data(1000)
    monitor.set_reference_data(reference_data)
    
    print("Monitor configurado com dados de referência.")
    
    # Simular monitoramento ao longo do tempo
    scenarios = [
        ('Dados normais', reference_data),
        ('Drift moderado', generate_drifted_data(reference_data, 'moderate')),
        ('Drift severo', generate_drifted_data(reference_data, 'severe')),
        ('Dados anômalos', generate_drifted_data(reference_data, 'anomalous'))
    ]
    
    for scenario_name, current_data in scenarios:
        print(f"\n--- {scenario_name} ---")
        
        # Analisar qualidade
        report = monitor.analyze_data_quality(current_data, scenario_name)
        
        print(f"Score de qualidade: {report.data_quality_score:.1f}/100")
        print(f"Score de drift: {report.overall_drift_score:.3f}")
        print(f"Taxa de anomalias: {report.overall_anomaly_rate:.3f}")
        
        print("\nRecomendações:")
        for rec in report.recommendations:
            print(f"• {rec}")
        
        # Drift details
        drift_features = [r for r in report.drift_results if r.is_drift]
        if drift_features:
            print(f"\nFeatures com drift ({len(drift_features)}):")
            for drift in drift_features:
                print(f"• {drift.feature_name} ({drift.severity}): {drift.test_type}")
    
    # Análise de tendências
    print("\n--- ANÁLISE DE TENDÊNCIAS ---")
    trends = monitor.get_trend_analysis(days=30)
    
    if 'message' not in trends:
        print(f"Relatórios analisados: {trends['reports_count']}")
        print(f"Tendência de drift: {trends['drift_trend']['trend']}")
        print(f"Tendência de anomalias: {trends['anomaly_trend']['trend']}")
        print(f"Tendência de qualidade: {trends['quality_trend']['trend']}")
    
    return monitor


def performance_comparison():
    """Compara performance dos diferentes métodos."""
    print("\n=== COMPARAÇÃO DE PERFORMANCE ===")
    
    # Gerar dados de tamanhos diferentes
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        print(f"\nTamanho do dataset: {size} amostras")
        
        reference_data = generate_reference_data(size)
        test_data = generate_drifted_data(reference_data, 'moderate')
        
        # Teste KS
        import time
        start_time = time.time()
        ks_detector = KSTestDriftDetector()
        ks_results = ks_detector.detect_drift(reference_data, test_data)
        ks_time = time.time() - start_time
        
        # Teste PSI
        start_time = time.time()
        psi_detector = PSITestDriftDetector()
        psi_results = psi_detector.detect_drift(reference_data, test_data)
        psi_time = time.time() - start_time
        
        # Detecção de anomalias
        start_time = time.time()
        anomaly_detector = AnomalyDetector(['isolation_forest'])
        anomaly_detector.fit(reference_data)
        anomaly_results = anomaly_detector.detect_anomalies(test_data)
        anomaly_time = time.time() - start_time
        
        print(f"  KS Test: {ks_time:.3f}s ({len([r for r in ks_results if r.is_drift])} drifts)")
        print(f"  PSI Test: {psi_time:.3f}s ({len([r for r in psi_results if r.is_drift])} drifts)")
        print(f"  Anomaly Detection: {anomaly_time:.3f}s ({len([r for r in anomaly_results if r.is_anomaly])} anomalies)")


def main():
    """Função principal da demonstração."""
    print("DEMONSTRAÇÃO DO SISTEMA DE DETECÇÃO DE DATA DRIFT")
    print("=" * 60)
    
    try:
        # Demonstrações individuais
        demonstrate_ks_test()
        demonstrate_psi_test()
        demonstrate_anomaly_detection()
        
        # Monitoramento completo
        monitor = demonstrate_complete_monitoring()
        
        # Comparação de performance
        performance_comparison()
        
        print("\n" + "=" * 60)
        print("RESUMO DOS BENEFÍCIOS:")
        print("• Detecção precoce de mudanças nos dados")
        print("• Identificação automática de anomalias")
        print("• Alertas para necessidade de retreino")
        print("• Monitoramento contínuo da qualidade")
        print("• Múltiplos métodos de detecção")
        print("• Análise de tendências temporais")
        
    except Exception as e:
        print(f"Erro durante a demonstração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()