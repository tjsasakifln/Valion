#!/usr/bin/env python3
"""
Script para validar a estrutura e sintaxe dos testes implementados
"""

import ast
import os
from pathlib import Path


def validate_python_syntax(file_path):
    """Valida sintaxe Python de um arquivo"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def count_test_functions(file_path):
    """Conta o número de funções de teste em um arquivo"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        test_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_count += 1
        
        return test_count
    except:
        return 0


def analyze_test_file(file_path):
    """Analisa um arquivo de teste"""
    is_valid, error = validate_python_syntax(file_path)
    test_count = count_test_functions(file_path)
    
    return {
        'file': file_path.name,
        'valid': is_valid,
        'error': error,
        'test_count': test_count,
        'size_lines': len(open(file_path, 'r').readlines()) if is_valid else 0
    }


def main():
    """Função principal"""
    print("🧪 Validando estrutura de testes do Valion")
    print("=" * 50)
    
    test_dir = Path("tests")
    
    if not test_dir.exists():
        print("❌ Diretório de testes não encontrado!")
        return
    
    # Buscar todos os arquivos de teste
    test_files = list(test_dir.rglob("test_*.py"))
    
    if not test_files:
        print("❌ Nenhum arquivo de teste encontrado!")
        return
    
    print(f"📁 Encontrados {len(test_files)} arquivos de teste")
    print()
    
    total_tests = 0
    total_lines = 0
    valid_files = 0
    
    results = []
    
    for test_file in sorted(test_files):
        result = analyze_test_file(test_file)
        results.append(result)
        
        status = "✅" if result['valid'] else "❌"
        print(f"{status} {result['file']}")
        
        if result['valid']:
            print(f"    └── {result['test_count']} testes, {result['size_lines']} linhas")
            total_tests += result['test_count']
            total_lines += result['size_lines']
            valid_files += 1
        else:
            print(f"    └── ERRO: {result['error']}")
        
        print()
    
    # Resumo
    print("📊 RESUMO")
    print("=" * 50)
    print(f"Arquivos de teste: {len(test_files)}")
    print(f"Arquivos válidos: {valid_files}")
    print(f"Total de testes: {total_tests}")
    print(f"Total de linhas: {total_lines}")
    print(f"Taxa de sucesso: {(valid_files/len(test_files)*100):.1f}%")
    
    # Verificar estrutura de diretórios
    print("\n📂 ESTRUTURA DE DIRETÓRIOS")
    print("=" * 50)
    
    expected_dirs = [
        "tests/unit",
        "tests/unit/core",
        "tests/unit/services", 
        "tests/unit/auth",
        "tests/unit/monitoring",
        "tests/integration"
    ]
    
    for dir_path in expected_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path}")
    
    # Verificar arquivos de configuração
    print("\n⚙️  ARQUIVOS DE CONFIGURAÇÃO")
    print("=" * 50)
    
    config_files = [
        "pytest.ini",
        ".coveragerc",
        "tests/conftest.py",
        "Makefile"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file}")
    
    # Recomendações
    print("\n💡 PRÓXIMOS PASSOS")
    print("=" * 50)
    print("1. Instalar dependências: pip install -r requirements/dev.txt")
    print("2. Executar testes: pytest tests/")
    print("3. Gerar coverage: pytest --cov=src --cov-report=html")
    print("4. Verificar qualidade: make check-quality")
    
    print(f"\n🎉 Implementação completa: {total_tests} testes em {valid_files} módulos!")


if __name__ == "__main__":
    main()