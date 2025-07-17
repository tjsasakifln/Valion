"""
Frontend Streamlit para Valion - Aplicação "Thin Client"
Interface de usuário responsável por renderizar a UI e orquestrar chamadas à API.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import websocket
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import threading
from datetime import datetime
import io
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Valion - Avaliação Imobiliária",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurações da API
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"

# Estilos CSS customizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .phase-indicator {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .nbr-grade {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .grade-superior {
        background: #28a745;
        color: white;
    }
    
    .grade-normal {
        background: #17a2b8;
        color: white;
    }
    
    .grade-inferior {
        background: #ffc107;
        color: black;
    }
    
    .grade-inadequado {
        background: #dc3545;
        color: white;
    }
    
    .transparency-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Estado da aplicação
if 'evaluation_id' not in st.session_state:
    st.session_state.evaluation_id = None
if 'evaluation_status' not in st.session_state:
    st.session_state.evaluation_status = None
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None

# Funções utilitárias
def call_api(endpoint: str, method: str = "GET", data: Optional[Dict] = None, files: Optional[Dict] = None):
    """
    Faz chamada para a API.
    
    Args:
        endpoint: Endpoint da API
        method: Método HTTP
        data: Dados para enviar
        files: Arquivos para enviar
        
    Returns:
        Resposta da API
    """
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Método não suportado: {method}")
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 202:
            return {"status": "processing", "message": "Processando..."}
        else:
            st.error(f"Erro na API: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Erro ao conectar com a API: {str(e)}")
        return None

def upload_file(file):
    """
    Faz upload de arquivo para a API.
    
    Args:
        file: Arquivo do Streamlit
        
    Returns:
        Caminho do arquivo no servidor
    """
    files = {"file": (file.name, file, file.type)}
    result = call_api("/upload", method="POST", files=files)
    
    if result:
        return result.get("file_path")
    return None

def start_evaluation(file_path: str, target_column: str = "valor"):
    """
    Inicia avaliação na API.
    
    Args:
        file_path: Caminho do arquivo
        target_column: Coluna target
        
    Returns:
        ID da avaliação
    """
    data = {
        "file_path": file_path,
        "target_column": target_column
    }
    
    result = call_api("/evaluations/", method="POST", data=data)
    
    if result:
        return result.get("evaluation_id")
    return None

def get_evaluation_status(evaluation_id: str):
    """
    Obtém status da avaliação.
    
    Args:
        evaluation_id: ID da avaliação
        
    Returns:
        Status da avaliação
    """
    return call_api(f"/evaluations/{evaluation_id}")

def get_evaluation_result(evaluation_id: str):
    """
    Obtém resultado da avaliação.
    
    Args:
        evaluation_id: ID da avaliação
        
    Returns:
        Resultado da avaliação
    """
    return call_api(f"/evaluations/{evaluation_id}/result")

def create_performance_chart(performance_data: Dict[str, Any]):
    """
    Cria gráfico de performance do modelo.
    
    Args:
        performance_data: Dados de performance
        
    Returns:
        Figura do Plotly
    """
    metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
    values = [
        performance_data.get('r2_score', 0),
        performance_data.get('rmse', 0),
        performance_data.get('mae', 0),
        performance_data.get('mape', 0)
    ]
    
    # Normalizar valores para visualização
    normalized_values = []
    for i, value in enumerate(values):
        if i == 0:  # R²
            normalized_values.append(value * 100)
        else:
            normalized_values.append(min(value, 100))
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=normalized_values,
            marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        )
    ])
    
    fig.update_layout(
        title="Métricas de Performance do Modelo",
        xaxis_title="Métricas",
        yaxis_title="Valores",
        height=400
    )
    
    return fig

def create_nbr_tests_chart(nbr_data: Dict[str, Any]):
    """
    Cria gráfico dos testes NBR 14653.
    
    Args:
        nbr_data: Dados dos testes NBR
        
    Returns:
        Figura do Plotly
    """
    tests = nbr_data.get('individual_tests', [])
    
    test_names = [test['test_name'] for test in tests]
    test_results = [1 if test['passed'] else 0 for test in tests]
    
    colors = ['#28a745' if result else '#dc3545' for result in test_results]
    
    fig = go.Figure(data=[
        go.Bar(
            x=test_names,
            y=test_results,
            marker_color=colors,
            text=['Passou' if result else 'Falhou' for result in test_results],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Resultados dos Testes NBR 14653",
        xaxis_title="Testes",
        yaxis_title="Resultado",
        height=400,
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Falhou', 'Passou'])
    )
    
    return fig

def create_feature_importance_chart(feature_importance: Dict[str, float]):
    """
    Cria gráfico de importância das features.
    
    Args:
        feature_importance: Importância das features
        
    Returns:
        Figura do Plotly
    """
    # Ordenar por importância
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Pegar top 10
    top_features = sorted_features[:10]
    
    feature_names = [f[0] for f in top_features]
    importance_values = [f[1] for f in top_features]
    
    fig = go.Figure(data=[
        go.Scatter(
            x=importance_values,
            y=feature_names,
            mode='markers',
            marker=dict(
                size=10,
                color=importance_values,
                colorscale='viridis',
                showscale=True
            )
        )
    ])
    
    fig.update_layout(
        title="Importância das Features (Top 10)",
        xaxis_title="Importância",
        yaxis_title="Features",
        height=400
    )
    
    return fig

# Interface principal
def main():
    """Interface principal da aplicação."""
    
    # Cabeçalho
    st.markdown('<div class="main-header">🏠 Valion - Avaliação Imobiliária</div>', unsafe_allow_html=True)
    
    # Sidebar com navegação
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Escolha uma opção", [
        "Nova Avaliação", 
        "Acompanhar Avaliação", 
        "Resultados", 
        "Predições",
        "Sobre"
    ])
    
    if page == "Nova Avaliação":
        show_new_evaluation_page()
    elif page == "Acompanhar Avaliação":
        show_evaluation_tracking_page()
    elif page == "Resultados":
        show_results_page()
    elif page == "Predições":
        show_predictions_page()
    elif page == "Sobre":
        show_about_page()

def show_new_evaluation_page():
    """Página para nova avaliação."""
    
    st.header("Nova Avaliação Imobiliária")
    
    # Seção de transparência
    st.markdown("""
    <div class="transparency-box">
        <h4>🔍 Transparência e Auditabilidade</h4>
        <p>A Valion é uma plataforma "caixa de vidro" que garante total transparência no processo de avaliação imobiliária. 
        Todos os passos são documentados e auditáveis, seguindo rigorosamente a norma NBR 14653.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload de arquivo
    st.subheader("1. Upload dos Dados")
    uploaded_file = st.file_uploader(
        "Escolha um arquivo com dados imobiliários",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos suportados: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        # Mostrar prévia dos dados
        st.subheader("2. Prévia dos Dados")
        
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Número de Registros", len(df))
                st.metric("Número de Colunas", len(df.columns))
            
            with col2:
                st.metric("Valores Ausentes", df.isnull().sum().sum())
                st.metric("Duplicatas", df.duplicated().sum())
            
            # Configurações da avaliação
            st.subheader("3. Configurações da Avaliação")
            
            target_column = st.selectbox(
                "Coluna Target (valor a ser predito)",
                df.columns,
                index=0 if 'valor' not in df.columns else list(df.columns).index('valor')
            )
            
            # Botão para iniciar avaliação
            if st.button("Iniciar Avaliação", type="primary"):
                with st.spinner("Fazendo upload do arquivo..."):
                    file_path = upload_file(uploaded_file)
                
                if file_path:
                    st.session_state.uploaded_file_path = file_path
                    
                    with st.spinner("Iniciando avaliação..."):
                        evaluation_id = start_evaluation(file_path, target_column)
                    
                    if evaluation_id:
                        st.session_state.evaluation_id = evaluation_id
                        st.success(f"Avaliação iniciada! ID: {evaluation_id}")
                        st.info("Vá para 'Acompanhar Avaliação' para ver o progresso.")
                    else:
                        st.error("Erro ao iniciar avaliação.")
                else:
                    st.error("Erro ao fazer upload do arquivo.")
                    
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

def show_evaluation_tracking_page():
    """Página para acompanhar avaliação."""
    
    st.header("Acompanhar Avaliação")
    
    # Input para ID da avaliação
    evaluation_id = st.text_input(
        "ID da Avaliação",
        value=st.session_state.evaluation_id or "",
        placeholder="Digite o ID da avaliação"
    )
    
    if evaluation_id:
        # Botão para atualizar status
        if st.button("Atualizar Status"):
            status = get_evaluation_status(evaluation_id)
            if status:
                st.session_state.evaluation_status = status
        
        # Mostrar status atual
        if st.session_state.evaluation_status:
            status = st.session_state.evaluation_status
            
            # Indicador de fase
            st.markdown(f"""
            <div class="phase-indicator">
                🔄 {status.get('current_phase', 'Processando')}
            </div>
            """, unsafe_allow_html=True)
            
            # Barra de progresso
            progress = status.get('progress', 0) / 100
            st.progress(progress)
            
            # Informações detalhadas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Status", status.get('status', 'Desconhecido'))
            
            with col2:
                st.metric("Progresso", f"{status.get('progress', 0):.1f}%")
            
            with col3:
                st.metric("Última Atualização", 
                         datetime.fromisoformat(status.get('timestamp', datetime.now().isoformat())).strftime('%H:%M:%S'))
            
            # Mensagem atual
            st.info(status.get('message', 'Processando...'))
            
            # Auto-refresh quando em progresso
            if status.get('status') == 'em_andamento':
                time.sleep(2)
                st.rerun()
            elif status.get('status') == 'concluido':
                st.success("Avaliação concluída! Vá para 'Resultados' para ver o relatório.")
            elif status.get('status') == 'erro':
                st.error("Erro na avaliação. Verifique os dados e tente novamente.")

def show_results_page():
    """Página para mostrar resultados."""
    
    st.header("Resultados da Avaliação")
    
    # Input para ID da avaliação
    evaluation_id = st.text_input(
        "ID da Avaliação",
        value=st.session_state.evaluation_id or "",
        placeholder="Digite o ID da avaliação"
    )
    
    if evaluation_id:
        if st.button("Carregar Resultados"):
            result = get_evaluation_result(evaluation_id)
            if result:
                st.session_state.evaluation_result = result
        
        if st.session_state.evaluation_result:
            result = st.session_state.evaluation_result
            
            # Grau NBR 14653
            nbr_grade = result.get('report', {}).get('nbr_validation', {}).get('overall_grade', 'Desconhecido')
            grade_class = f"grade-{nbr_grade.lower()}"
            
            st.markdown(f"""
            <div class="nbr-grade {grade_class}">
                Grau NBR 14653: {nbr_grade}
            </div>
            """, unsafe_allow_html=True)
            
            # Tabs para diferentes seções
            tabs = st.tabs([
                "Resumo Executivo", 
                "Performance do Modelo", 
                "Testes NBR 14653", 
                "Análise de Features",
                "Dados Utilizados",
                "Metodologia"
            ])
            
            with tabs[0]:
                show_executive_summary(result)
            
            with tabs[1]:
                show_model_performance(result)
            
            with tabs[2]:
                show_nbr_tests(result)
            
            with tabs[3]:
                show_feature_analysis(result)
            
            with tabs[4]:
                show_data_summary(result)
            
            with tabs[5]:
                show_methodology(result)

def show_executive_summary(result: Dict[str, Any]):
    """Mostra resumo executivo."""
    
    st.subheader("Resumo Executivo")
    
    report = result.get('report', {})
    performance = report.get('model_performance', {}).get('performance_metrics', {})
    nbr = report.get('nbr_validation', {})
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R²", f"{performance.get('r2_score', 0):.4f}")
    
    with col2:
        st.metric("RMSE", f"{performance.get('rmse', 0):.2f}")
    
    with col3:
        st.metric("MAE", f"{performance.get('mae', 0):.2f}")
    
    with col4:
        st.metric("MAPE", f"{performance.get('mape', 0):.2f}%")
    
    # Conclusões principais
    conclusions = report.get('conclusions', {})
    key_findings = conclusions.get('key_findings', [])
    
    st.subheader("Principais Conclusões")
    for finding in key_findings:
        st.write(f"• {finding}")
    
    # Recomendações
    recommendations = conclusions.get('recommendations', [])
    if recommendations:
        st.subheader("Recomendações")
        for rec in recommendations:
            st.write(f"• {rec}")

def show_model_performance(result: Dict[str, Any]):
    """Mostra performance do modelo."""
    
    st.subheader("Performance do Modelo")
    
    performance = result.get('report', {}).get('model_performance', {}).get('performance_metrics', {})
    
    # Gráfico de performance
    fig = create_performance_chart(performance)
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas detalhadas
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Coeficiente de Determinação (R²)", f"{performance.get('r2_score', 0):.4f}")
        st.metric("Erro Quadrático Médio (RMSE)", f"{performance.get('rmse', 0):.2f}")
    
    with col2:
        st.metric("Erro Absoluto Médio (MAE)", f"{performance.get('mae', 0):.2f}")
        st.metric("Erro Percentual Absoluto Médio (MAPE)", f"{performance.get('mape', 0):.2f}%")
    
    # Cross-validation
    cv_scores = performance.get('cv_scores', [])
    if cv_scores:
        st.subheader("Cross-Validation")
        st.write(f"Média RMSE: {np.mean(cv_scores):.2f}")
        st.write(f"Desvio Padrão: {np.std(cv_scores):.2f}")

def show_nbr_tests(result: Dict[str, Any]):
    """Mostra testes NBR 14653."""
    
    st.subheader("Testes NBR 14653")
    
    nbr_data = result.get('report', {}).get('nbr_validation', {})
    
    # Gráfico dos testes
    fig = create_nbr_tests_chart(nbr_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela detalhada dos testes
    tests = nbr_data.get('individual_tests', [])
    if tests:
        df_tests = pd.DataFrame(tests)
        st.dataframe(df_tests[['test_name', 'passed', 'value', 'threshold', 'description']])
    
    # Score de conformidade
    compliance_score = nbr_data.get('compliance_score', 0)
    st.metric("Score de Conformidade", f"{compliance_score:.2%}")

def show_feature_analysis(result: Dict[str, Any]):
    """Mostra análise de features."""
    
    st.subheader("Análise de Features")
    
    transformation = result.get('report', {}).get('transformation_summary', {})
    feature_importance = transformation.get('feature_importance', {})
    
    if feature_importance:
        # Gráfico de importância
        fig = create_feature_importance_chart(feature_importance)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de features
        df_features = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importância'])
        df_features = df_features.sort_values('Importância', ascending=False)
        st.dataframe(df_features)
    
    # Transformações aplicadas
    transformations = transformation.get('transformations_applied', {})
    if transformations:
        st.subheader("Transformações Aplicadas")
        for key, value in transformations.items():
            st.write(f"• {key}: {value}")

def show_data_summary(result: Dict[str, Any]):
    """Mostra resumo dos dados."""
    
    st.subheader("Resumo dos Dados")
    
    data_summary = result.get('report', {}).get('data_summary', {})
    
    # Estatísticas básicas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Registros", data_summary.get('total_records', 0))
    
    with col2:
        st.metric("Número de Variáveis", data_summary.get('variables_count', 0))
    
    with col3:
        quality = data_summary.get('data_quality', {})
        st.metric("Qualidade dos Dados", "✅" if quality.get('validation_passed') else "❌")
    
    # Estatísticas descritivas
    desc_stats = data_summary.get('descriptive_statistics', {}).get('valor', {})
    if desc_stats:
        st.subheader("Estatísticas Descritivas do Valor")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Média", f"R$ {desc_stats.get('mean', 0):,.2f}")
            st.metric("Mediana", f"R$ {desc_stats.get('median', 0):,.2f}")
        
        with col2:
            st.metric("Mínimo", f"R$ {desc_stats.get('min', 0):,.2f}")
            st.metric("Máximo", f"R$ {desc_stats.get('max', 0):,.2f}")
        
        with col3:
            st.metric("Q1", f"R$ {desc_stats.get('q25', 0):,.2f}")
            st.metric("Q3", f"R$ {desc_stats.get('q75', 0):,.2f}")
        
        with col4:
            st.metric("Desvio Padrão", f"R$ {desc_stats.get('std', 0):,.2f}")

def show_methodology(result: Dict[str, Any]):
    """Mostra metodologia utilizada."""
    
    st.subheader("Metodologia")
    
    methodology = result.get('report', {}).get('methodology', {})
    
    # Abordagem geral
    st.write(f"**Abordagem:** {methodology.get('approach', 'Não especificada')}")
    
    # Fases do processo
    phases = methodology.get('phases', [])
    if phases:
        st.subheader("Fases do Processo")
        for phase in phases:
            with st.expander(phase.get('phase', 'Fase')):
                st.write(phase.get('description', ''))
                techniques = phase.get('techniques', [])
                if techniques:
                    st.write("**Técnicas utilizadas:**")
                    for technique in techniques:
                        st.write(f"• {technique}")
    
    # Fundamentos estatísticos
    foundations = methodology.get('statistical_foundations', {})
    if foundations:
        st.subheader("Fundamentos Estatísticos")
        for key, value in foundations.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Conformidade NBR
    nbr_compliance = methodology.get('nbr_compliance', {})
    if nbr_compliance:
        st.subheader("Conformidade NBR 14653")
        st.write(f"**Norma:** {nbr_compliance.get('standard', '')}")
        
        precision_levels = nbr_compliance.get('precision_levels', [])
        if precision_levels:
            st.write("**Níveis de Precisão:**")
            for level in precision_levels:
                st.write(f"• {level}")

def show_predictions_page():
    """Página para fazer predições."""
    
    st.header("Fazer Predições")
    
    # Input para ID da avaliação
    evaluation_id = st.text_input(
        "ID da Avaliação",
        value=st.session_state.evaluation_id or "",
        placeholder="Digite o ID da avaliação com modelo treinado"
    )
    
    if evaluation_id:
        st.subheader("Características do Imóvel")
        
        # Formulário para características
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                area_total = st.number_input("Área Total (m²)", min_value=0.0, value=100.0)
                quartos = st.number_input("Quartos", min_value=0, value=2)
                banheiros = st.number_input("Banheiros", min_value=0, value=1)
            
            with col2:
                vagas = st.number_input("Vagas de Garagem", min_value=0, value=1)
                idade = st.number_input("Idade do Imóvel (anos)", min_value=0, value=10)
                localizacao = st.text_input("Localização", value="Centro")
            
            submitted = st.form_submit_button("Fazer Predição")
            
            if submitted:
                features = {
                    "area_total": area_total,
                    "quartos": quartos,
                    "banheiros": banheiros,
                    "vagas": vagas,
                    "idade": idade,
                    "localizacao": localizacao
                }
                
                # Fazer predição via API
                data = {
                    "evaluation_id": evaluation_id,
                    "features": features
                }
                
                result = call_api(f"/evaluations/{evaluation_id}/predict", method="POST", data=data)
                
                if result:
                    st.success("Predição realizada com sucesso!")
                    
                    # Mostrar resultado
                    predicted_value = result.get('predicted_value', 0)
                    st.metric("Valor Predito", f"R$ {predicted_value:,.2f}")
                    
                    # Mostrar características utilizadas
                    st.subheader("Características Utilizadas")
                    st.json(features)
                else:
                    st.error("Erro ao fazer predição. Verifique se o modelo foi treinado.")

def show_about_page():
    """Página sobre a aplicação."""
    
    st.header("Sobre a Valion")
    
    st.markdown("""
    ## 🎯 Missão
    
    A Valion é uma plataforma de avaliação imobiliária projetada para ser uma "caixa de vidro", 
    garantindo total transparência, auditabilidade e rigor estatístico em todas as etapas do processo.
    
    ## 🔍 Transparência
    
    - **Processo Auditável**: Cada etapa é documentada e pode ser auditada
    - **Metodologia Clara**: Baseada em princípios estatísticos sólidos
    - **Conformidade NBR**: Segue rigorosamente a norma NBR 14653
    - **Resultados Defensáveis**: Relatórios detalhados com fundamentos técnicos
    
    ## 🏗️ Arquitetura
    
    - **Frontend**: Streamlit (Interface thin client)
    - **Backend**: FastAPI (API REST com WebSocket)
    - **Workers**: Celery (Processamento assíncrono)
    - **Modelo**: Elastic Net Regression
    - **Validação**: Bateria completa de testes NBR 14653
    
    ## 📊 Fases do Processo
    
    1. **Ingestão e Validação**: Carregamento e validação dos dados
    2. **Transformação**: Preparação e engenharia de features
    3. **Modelagem**: Treinamento do modelo Elastic Net
    4. **Validação NBR**: Execução dos testes estatísticos
    5. **Relatório**: Consolidação dos resultados
    
    ## 🎓 Fundamentos Técnicos
    
    - **Modelo**: Elastic Net combina regularização L1 (Lasso) e L2 (Ridge)
    - **Validação**: Cross-validation com 5 folds
    - **Seleção de Features**: Seleção univariada + regularização
    - **Testes Estatísticos**: Teste F, Teste t, Durbin-Watson, Shapiro-Wilk
    
    ## 📈 Níveis de Precisão NBR 14653
    
    - **Superior**: R² ≥ 0,90
    - **Normal**: R² ≥ 0,80
    - **Inferior**: R² ≥ 0,70
    - **Inadequado**: R² < 0,70
    
    ## 🔧 Configuração
    
    A aplicação é altamente configurável através de variáveis de ambiente e arquivos de configuração,
    permitindo adaptação para diferentes contextos e necessidades.
    """)
    
    # Informações técnicas
    with st.expander("Informações Técnicas"):
        st.code("""
        # Configuração da API
        API_BASE_URL = "http://localhost:8000"
        
        # Dependências principais
        - streamlit >= 1.28.0
        - fastapi >= 0.104.0
        - celery >= 5.3.0
        - scikit-learn >= 1.3.0
        - pandas >= 2.0.0
        - plotly >= 5.17.0
        
        # Banco de dados
        - PostgreSQL (produção)
        - Redis (cache e broker)
        """)

if __name__ == "__main__":
    main()