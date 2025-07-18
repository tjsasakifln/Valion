#!/usr/bin/env python3
"""
Demonstração completa do MLOps Pipeline
Sistema integrado de versionamento, deployment e monitoramento de modelos.
"""

import asyncio
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mlops.model_registry import ModelRegistry, ModelStage, ModelStatus
from src.mlops.model_deployer import ModelDeployer, DeploymentConfig, DeploymentStrategy
from src.mlops.model_validator import ModelValidator, ValidationStatus
from src.mlops.pipeline_orchestrator import PipelineOrchestrator
from src.mlops.version_manager import VersionManager, VersioningStrategy
from src.monitoring.logging_config import setup_structured_logging
import structlog


class MLOpsDemo:
    """Demonstração completa do MLOps Pipeline."""
    
    def __init__(self):
        # Configurar logging
        setup_structured_logging({
            "level": "INFO",
            "console_handler": True,
            "file_handler": True,
            "development_mode": True
        })
        
        self.logger = structlog.get_logger("mlops_demo")
        
        # Criar componentes MLOps
        self.registry = ModelRegistry("demo_registry")
        self.deployer = ModelDeployer(self.registry, "demo_deployments")
        self.validator = ModelValidator(self.registry)
        self.orchestrator = PipelineOrchestrator(self.registry, self.deployer, self.validator)
        self.version_manager = VersionManager(VersioningStrategy.SEMANTIC)
        
        # Dados de exemplo
        self.sample_data = self._generate_sample_data()
        
        print("🚀 MLOps Pipeline Demo Initialized!")
        print("=" * 50)
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Gera dados de exemplo para demonstração."""
        np.random.seed(42)
        
        # Gerar dados sintéticos de imóveis
        n_samples = 1000
        
        data = {
            'area': np.random.normal(150, 50, n_samples),
            'quartos': np.random.randint(1, 6, n_samples),
            'banheiros': np.random.randint(1, 4, n_samples),
            'idade': np.random.randint(0, 50, n_samples),
            'garagem': np.random.randint(0, 3, n_samples),
            'distancia_centro': np.random.normal(10, 5, n_samples),
            'score_vizinhanca': np.random.normal(7, 2, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Gerar preço baseado nas features
        df['preco'] = (
            df['area'] * 1000 +
            df['quartos'] * 20000 +
            df['banheiros'] * 15000 +
            (50 - df['idade']) * 500 +
            df['garagem'] * 10000 +
            (20 - df['distancia_centro']) * 2000 +
            df['score_vizinhanca'] * 5000 +
            np.random.normal(0, 20000, n_samples)
        )
        
        # Garantir valores positivos
        df['preco'] = np.maximum(df['preco'], 50000)
        
        return df
    
    async def demo_model_registry(self):
        """Demonstra o Model Registry."""
        print("\n📚 Model Registry Demo")
        print("-" * 30)
        
        # Registrar modelo
        model_id = self.registry.register_model(
            name="RandomForest_Imobiliario",
            algorithm="RandomForestRegressor",
            framework="sklearn",
            description="Modelo de Random Forest para avaliação imobiliária",
            tags=["real_estate", "regression", "production"]
        )
        
        print(f"✅ Modelo registrado: {model_id}")
        
        # Preparar dados
        X = self.sample_data.drop('preco', axis=1)
        y = self.sample_data['preco']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar modelo
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Avaliar modelo
        predictions = model.predict(X_test_scaled)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"📊 Performance: R² = {r2:.3f}, MAE = {mae:.0f}")
        
        # Criar versão do modelo
        model_version = self.registry.create_version(
            model_id=model_id,
            model_object=model,
            performance_metrics={"r2_score": r2, "mae": mae, "rmse": np.sqrt(np.mean((y_test - predictions)**2))},
            hyperparameters={"n_estimators": 100, "random_state": 42},
            features=X.columns.tolist(),
            target="preco",
            dataset_info={"train_size": len(X_train), "test_size": len(X_test)},
            description="Primeira versão do modelo RandomForest"
        )
        
        print(f"📦 Versão criada: {model_version.version}")
        
        # Listar modelos
        models = self.registry.list_models()
        print(f"📋 Total de modelos no registry: {len(models)}")
        
        # Promover para staging
        self.registry.promote_version(model_id, model_version.version, ModelStage.STAGING)
        print(f"⬆️ Modelo promovido para staging")
        
        return model_id, model_version.version, X_test_scaled, y_test
    
    async def demo_model_validator(self, model_id: str, version: str, X_test: np.ndarray, y_test: np.ndarray):
        """Demonstra o Model Validator."""
        print("\n🔍 Model Validator Demo")
        print("-" * 30)
        
        # Converter para DataFrame
        feature_names = ['area', 'quartos', 'banheiros', 'idade', 'garagem', 'distancia_centro', 'score_vizinhanca']
        validation_data = pd.DataFrame(X_test, columns=feature_names)
        target = pd.Series(y_test)
        
        # Dados de referência (simulando dados históricos)
        reference_data = validation_data.copy()
        reference_data = reference_data + np.random.normal(0, 0.1, reference_data.shape)
        
        # Executar validação
        report = await self.validator.validate_model(
            model_id=model_id,
            version=version,
            validation_data=validation_data,
            target=target,
            reference_data=reference_data
        )
        
        print(f"📝 Relatório de validação gerado: {report.validation_id}")
        print(f"📊 Status geral: {report.overall_status.value}")
        print(f"🔄 Validadores executados: {len(report.results)}")
        
        # Mostrar resultados
        for result in report.results:
            status_emoji = "✅" if result.status == ValidationStatus.PASSED else "⚠️" if result.status == ValidationStatus.WARNING else "❌"
            print(f"{status_emoji} {result.validator_name}: {result.message}")
        
        # Mostrar métricas
        print(f"📈 Métricas: {report.metrics}")
        
        # Mostrar recomendações
        if report.recommendations:
            print("💡 Recomendações:")
            for rec in report.recommendations:
                print(f"  - {rec}")
        
        return report
    
    async def demo_model_deployer(self, model_id: str, version: str):
        """Demonstra o Model Deployer."""
        print("\n🚀 Model Deployer Demo")
        print("-" * 30)
        
        # Configurar deployment
        config = DeploymentConfig(
            model_id=model_id,
            version=version,
            environment="staging",
            strategy=DeploymentStrategy.BLUE_GREEN,
            target_port=8080,
            health_check_path="/health",
            max_replicas=2,
            min_replicas=1,
            auto_rollback=True,
            environment_variables={"MODEL_ENV": "staging"}
        )
        
        # Executar deployment
        deployment_id = await self.deployer.deploy_model(config)
        print(f"📦 Deployment iniciado: {deployment_id}")
        
        # Aguardar um pouco
        await asyncio.sleep(2)
        
        # Verificar status
        deployment_info = self.deployer.get_deployment_info(deployment_id)
        if deployment_info:
            print(f"📊 Status: {deployment_info.status.value}")
            print(f"🌐 Endpoint: {deployment_info.endpoint}")
            print(f"⚙️ Réplicas: {deployment_info.replicas}")
        
        # Listar deployments
        deployments = self.deployer.list_deployments()
        print(f"📋 Total de deployments ativos: {len(deployments)}")
        
        # Métricas de deployment
        metrics = self.deployer.get_deployment_metrics()
        print(f"📈 Métricas de deployment: {metrics}")
        
        return deployment_id
    
    async def demo_version_manager(self, model_id: str, current_version: str):
        """Demonstra o Version Manager."""
        print("\n🔢 Version Manager Demo")
        print("-" * 30)
        
        # Contexto de mudanças
        contexts = [
            {
                "performance_improvement": 0.05,
                "bug_fix": True,
                "breaking_change": False,
                "new_feature": False
            },
            {
                "performance_improvement": 0.15,
                "bug_fix": False,
                "breaking_change": False,
                "new_feature": True
            },
            {
                "performance_improvement": 0.02,
                "bug_fix": False,
                "breaking_change": True,
                "new_feature": False
            }
        ]
        
        print(f"🔄 Versão atual: {current_version}")
        
        for i, context in enumerate(contexts, 1):
            new_version, reason = self.version_manager.suggest_version(current_version, context)
            print(f"📊 Cenário {i}: {context}")
            print(f"  → Versão sugerida: {new_version}")
            print(f"  → Razão: {reason}")
        
        # Comparar versões
        version1 = "1.0.0"
        version2 = "1.2.0"
        comparison = self.version_manager.compare_versions(version1, version2)
        
        print(f"⚖️ Comparação {version1} vs {version2}: {comparison}")
        
        # Verificar compatibilidade
        compatibility = self.version_manager.is_compatible(version1, version2, "minor")
        print(f"🔗 Compatibilidade (minor): {compatibility}")
        
        # Gerar changelog
        version_history = [
            {"version": "1.0.0", "date": "2024-01-01", "changes": ["Initial release"]},
            {"version": "1.1.0", "date": "2024-02-01", "changes": ["Added new features", "Performance improvements"]},
            {"version": "1.2.0", "date": "2024-03-01", "changes": ["Bug fixes", "Enhanced validation"]}
        ]
        
        changelog = self.version_manager.generate_changelog(version_history)
        print(f"📖 Changelog gerado:\n{changelog[:200]}...")
    
    async def demo_pipeline_orchestrator(self, model_id: str):
        """Demonstra o Pipeline Orchestrator."""
        print("\n🎭 Pipeline Orchestrator Demo")
        print("-" * 30)
        
        # Listar pipelines disponíveis
        pipelines = list(self.orchestrator.pipelines.keys())
        print(f"📋 Pipelines disponíveis: {pipelines}")
        
        # Executar pipeline padrão
        execution_id = await self.orchestrator.execute_pipeline(
            "default",
            parameters={"model_id": model_id, "environment": "staging"}
        )
        
        print(f"🏃 Pipeline executado: {execution_id}")
        
        # Aguardar um pouco
        await asyncio.sleep(3)
        
        # Verificar status
        execution = self.orchestrator.get_execution_status(execution_id)
        if execution:
            print(f"📊 Status: {execution.status.value}")
            print(f"🎯 Estágio atual: {execution.current_stage.value if execution.current_stage else 'N/A'}")
            print(f"✅ Steps completados: {len(execution.steps_completed)}")
            print(f"❌ Steps falhados: {len(execution.steps_failed)}")
        
        # Listar execuções
        executions = self.orchestrator.list_executions()
        print(f"📋 Total de execuções: {len(executions)}")
        
        # Métricas do pipeline
        metrics = self.orchestrator.get_pipeline_metrics()
        print(f"📈 Métricas do pipeline: {metrics}")
        
        return execution_id
    
    async def demo_registry_stats(self):
        """Demonstra estatísticas do registry."""
        print("\n📊 Registry Statistics")
        print("-" * 30)
        
        stats = self.registry.get_registry_stats()
        print(f"📦 Modelos: {stats.get('models_count', 0)}")
        print(f"🔢 Versões: {stats.get('versions_count', 0)}")
        print(f"💾 Tamanho total: {stats.get('total_size_mb', 0):.2f} MB")
        print(f"📁 Localização: {stats.get('registry_path', 'N/A')}")
        
        # Distribuição por estágio
        distribution = stats.get('stage_distribution', {})
        print("📊 Distribuição por estágio:")
        for stage, count in distribution.items():
            print(f"  {stage}: {count}")
    
    async def run_complete_demo(self):
        """Executa demonstração completa."""
        print("🎬 Iniciando demonstração completa do MLOps Pipeline")
        print("=" * 60)
        
        try:
            # 1. Model Registry
            model_id, version, X_test, y_test = await self.demo_model_registry()
            
            # 2. Model Validator
            validation_report = await self.demo_model_validator(model_id, version, X_test, y_test)
            
            # 3. Model Deployer
            deployment_id = await self.demo_model_deployer(model_id, version)
            
            # 4. Version Manager
            await self.demo_version_manager(model_id, version)
            
            # 5. Pipeline Orchestrator
            pipeline_execution_id = await self.demo_pipeline_orchestrator(model_id)
            
            # 6. Registry Stats
            await self.demo_registry_stats()
            
            print("\n🎉 Demonstração completa finalizada!")
            print("=" * 60)
            
            # Resumo
            print("\n📋 Resumo da demonstração:")
            print(f"✅ Modelo registrado: {model_id}")
            print(f"✅ Versão criada: {version}")
            print(f"✅ Validação executada: {validation_report.validation_id}")
            print(f"✅ Deployment realizado: {deployment_id}")
            print(f"✅ Pipeline executado: {pipeline_execution_id}")
            print(f"✅ Versionamento demonstrado")
            
        except Exception as e:
            print(f"❌ Erro durante a demonstração: {e}")
            self.logger.error(f"Demo error: {e}")
    
    async def cleanup(self):
        """Limpeza após demonstração."""
        print("\n🧹 Limpando recursos...")
        
        # Parar deployments ativos
        deployments = self.deployer.list_deployments()
        for deployment in deployments:
            await self.deployer.stop_deployment(deployment.deployment_id)
        
        # Cancelar execuções ativas
        executions = self.orchestrator.list_executions()
        for execution in executions:
            if execution.status.value in ["running", "pending"]:
                await self.orchestrator.cancel_execution(execution.execution_id)
        
        print("✅ Limpeza concluída!")


async def main():
    """Função principal."""
    demo = MLOpsDemo()
    
    try:
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n⏹️ Demonstração interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())