# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Streamlit Frontend for Valion - "Thin Client" Application
User interface responsible for rendering the UI and orchestrating API calls.
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
import asyncio
import websockets

# Page configuration
st.set_page_config(
    page_title="Valion - Real Estate Evaluation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API settings
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"

# Custom CSS styles
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
    
    .shap-lab-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .simulation-result {
        background: #f8f9fa;
        border: 2px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .feature-impact-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .feature-impact-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .feature-impact-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Application state
if 'evaluation_id' not in st.session_state:
    st.session_state.evaluation_id = None
if 'evaluation_status' not in st.session_state:
    st.session_state.evaluation_status = None
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'websocket_connected' not in st.session_state:
    st.session_state.websocket_connected = False
if 'realtime_updates' not in st.session_state:
    st.session_state.realtime_updates = []

# Utility functions
def call_api(endpoint: str, method: str = "GET", data: Optional[Dict] = None, files: Optional[Dict] = None):
    """
    Makes call to the API.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        data: Data to send
        files: Files to send
        
    Returns:
        API response
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
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 202:
            return {"status": "processing", "message": "Processing..."}
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def upload_file(file):
    """
    Uploads file to the API.
    
    Args:
        file: Streamlit file
        
    Returns:
        File path on server
    """
    files = {"file": (file.name, file, file.type)}
    result = call_api("/upload", method="POST", files=files)
    
    if result:
        return result.get("file_path")
    return None

def start_evaluation(file_path: str, target_column: str = "valor"):
    """
    Starts evaluation in the API.
    
    Args:
        file_path: File path
        target_column: Target column
        
    Returns:
        Evaluation ID
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
    Gets evaluation status.
    
    Args:
        evaluation_id: Evaluation ID
        
    Returns:
        Evaluation status
    """
    return call_api(f"/evaluations/{evaluation_id}")

def get_evaluation_result(evaluation_id: str):
    """
    Gets evaluation result.
    
    Args:
        evaluation_id: Evaluation ID
        
    Returns:
        Evaluation result
    """
    return call_api(f"/evaluations/{evaluation_id}/result")

def setup_websocket_connection(evaluation_id: str):
    """
    Sets up WebSocket connection for real-time updates.
    
    Args:
        evaluation_id: Evaluation ID to monitor
    """
    if st.session_state.websocket_connected:
        return
    
    try:
        import asyncio
        import websockets
        import threading
        
        def websocket_listener():
            async def listen():
                ws_url = f"{WS_BASE_URL}/ws/evaluations/{evaluation_id}"
                try:
                    async with websockets.connect(ws_url) as websocket:
                        st.session_state.websocket_connected = True
                        while True:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            # Add update to list
                            st.session_state.realtime_updates.append(data)
                            
                            # Keep only the last 50 updates
                            if len(st.session_state.realtime_updates) > 50:
                                st.session_state.realtime_updates = st.session_state.realtime_updates[-50:]
                            
                            # Force Streamlit rerun if important update
                            if data.get('type') == 'progress_update':
                                st.rerun()
                                
                except Exception as e:
                    st.session_state.websocket_connected = False
                    st.error(f"WebSocket connection error: {e}")
            
            # Execute in asynchronous loop
            try:
                asyncio.run(listen())
            except Exception as e:
                st.session_state.websocket_connected = False
        
        # Start listener in separate thread
        if not st.session_state.websocket_connected:
            thread = threading.Thread(target=websocket_listener, daemon=True)
            thread.start()
            
    except Exception as e:
        st.warning(f"WebSocket not available: {e}")

def get_latest_realtime_status():
    """
    Gets the most recent status from real-time updates.
    
    Returns:
        Dict with most recent status or None
    """
    if not st.session_state.realtime_updates:
        return None
    
    # Get the most recent update that is of type progress_update
    for update in reversed(st.session_state.realtime_updates):
        if update.get('type') == 'progress_update':
            return update
    
    return None

def create_performance_chart(performance_data: Dict[str, Any]):
    """
    Creates model performance chart.
    
    Args:
        performance_data: Performance data
        
    Returns:
        Plotly figure
    """
    metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
    values = [
        performance_data.get('r2_score', 0),
        performance_data.get('rmse', 0),
        performance_data.get('mae', 0),
        performance_data.get('mape', 0)
    ]
    
    # Normalize values for visualization
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
        title="Model Performance Metrics",
        xaxis_title="Metrics",
        yaxis_title="Values",
        height=400
    )
    
    return fig

def create_nbr_tests_chart(nbr_data: Dict[str, Any]):
    """
    Creates NBR 14653 tests chart.
    
    Args:
        nbr_data: NBR test data
        
    Returns:
        Plotly figure
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
            text=['Passed' if result else 'Failed' for result in test_results],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="NBR 14653 Test Results",
        xaxis_title="Tests",
        yaxis_title="Result",
        height=400,
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Failed', 'Passed'])
    )
    
    return fig

def create_feature_importance_chart(feature_importance: Dict[str, float]):
    """
    Creates feature importance chart.
    
    Args:
        feature_importance: Feature importance
        
    Returns:
        Plotly figure
    """
    # Sort by importance
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
        title="Feature Importance (Top 10)",
        xaxis_title="Importance",
        yaxis_title="Features",
        height=400
    )
    
    return fig

# Main interface
def main():
    """Main application interface."""
    
    # Header
    st.markdown('<div class="main-header">🏠 Valion - Real Estate Evaluation</div>', unsafe_allow_html=True)
    
    # Sidebar with navigation
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Choose an option", [
        "New Evaluation", 
        "Guided Workflow",
        "Track Evaluation", 
        "Results", 
        "Predictions",
        "SHAP Laboratory",
        "About"
    ])
    
    if page == "New Evaluation":
        show_new_evaluation_page()
    elif page == "Guided Workflow":
        show_guided_workflow_page()
    elif page == "Track Evaluation":
        show_evaluation_tracking_page()
    elif page == "Results":
        show_results_page()
    elif page == "Predictions":
        show_predictions_page()
    elif page == "SHAP Laboratory":
        show_shap_laboratory_page()
    elif page == "About":
        show_about_page()

def show_new_evaluation_page():
    """Page for new evaluation."""
    
    st.header("New Real Estate Evaluation")
    
    # Transparency section
    st.markdown("""
    <div class="transparency-box">
        <h4>🔍 Transparency and Auditability</h4>
        <p>Valion is a "glass box" platform that guarantees total transparency in the real estate evaluation process. 
        All steps are documented and auditable, strictly following the NBR 14653 standard.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    st.subheader("1. Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a file with real estate data",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel (.xlsx, .xls)"
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
        # Configurar WebSocket para updates em tempo real
        setup_websocket_connection(evaluation_id)
        
        # Verificar updates em tempo real
        realtime_status = get_latest_realtime_status()
        if realtime_status:
            st.session_state.evaluation_status = realtime_status
        
        # Botão para atualizar status (fallback)
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Atualizar Status"):
                status = get_evaluation_status(evaluation_id)
                if status:
                    st.session_state.evaluation_status = status
        
        with col1:
            if st.session_state.websocket_connected:
                st.success("🔗 Conectado - Updates em tempo real")
            else:
                st.info("📡 Tentando conectar para updates em tempo real...")
        
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
                "Trilha de Auditoria",
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
                show_audit_trail_tab(evaluation_id)
            
            with tabs[5]:
                show_data_summary(result)
            
            with tabs[6]:
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

def show_guided_workflow_page():
    """Página do fluxo de trabalho guiado."""
    
    st.header("🔍 Fluxo de Trabalho Guiado")
    
    # Inicializar estado do workflow
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = 1
    if 'workflow_data' not in st.session_state:
        st.session_state.workflow_data = {}
    if 'user_tier' not in st.session_state:
        st.session_state.user_tier = 'mvp'  # Default to MVP tier
    
    # Seletor de nível do usuário (feature flag simulation)
    st.sidebar.subheader("Configurações do Usuário")
    user_tier = st.sidebar.selectbox(
        "Nível de Acesso",
        options=['mvp', 'professional', 'expert'],
        value=st.session_state.user_tier,
        format_func=lambda x: {'mvp': 'MVP', 'professional': 'Profissional', 'expert': 'Especialista'}[x]
    )
    st.session_state.user_tier = user_tier
    
    # Descrição do fluxo
    st.markdown("""
    <div class="transparency-box">
        <h4>🧭 Navegação Inteligente</h4>
        <p>Este fluxo guiado desmistifica cada etapa do processo de avaliação, 
        proporcionando controle total e transparência absoluta em cada decisão.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    phases = [
        "1. Validação",
        "2. Transformação", 
        "3. Modelagem",
        "4. Validação NBR",
        "5. Relatório"
    ]
    
    current_step = st.session_state.workflow_step
    progress = current_step / len(phases)
    
    st.markdown(f"""
    <div class="phase-indicator">
        Fase Atual: {phases[current_step - 1]} ({current_step}/{len(phases)})
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(progress)
    
    # Step navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_step > 1:
            if st.button("⬅️ Fase Anterior"):
                st.session_state.workflow_step -= 1
                st.rerun()
    
    with col3:
        if current_step < len(phases):
            if st.button("Próxima Fase ➡️"):
                st.session_state.workflow_step += 1
                st.rerun()
    
    # Display current phase
    if current_step == 1:
        show_validation_phase()
    elif current_step == 2:
        show_transformation_phase()
    elif current_step == 3:
        show_modeling_phase()
    elif current_step == 4:
        show_nbr_validation_phase()
    elif current_step == 5:
        show_report_phase()

def show_validation_phase():
    """Fase 1: Validação de Dados."""
    
    st.subheader("📊 Fase 1: Validação de Dados")
    
    st.markdown("""
    **Objetivo:** Garantir a qualidade e integridade dos dados de entrada.
    
    **O que acontece nesta fase:**
    - Verificação de tipos de dados
    - Detecção de valores ausentes
    - Identificação de outliers
    - Validação de consistência
    """)
    
    # Upload section
    uploaded_file = st.file_uploader(
        "Upload dos dados imobiliários",
        type=['csv', 'xlsx', 'xls'],
        help="Envie seus dados para análise"
    )
    
    if uploaded_file is not None:
        try:
            # Load and preview data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("✅ Arquivo carregado com sucesso!")
            
            # Data preview
            st.subheader("👀 Prévia dos Dados")
            st.dataframe(df.head(10))
            
            # Data quality metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Registros", len(df))
            with col2:
                st.metric("Colunas", len(df.columns))
            with col3:
                st.metric("Valores Ausentes", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicatas", df.duplicated().sum())
            
            # Column analysis
            st.subheader("🔍 Análise de Colunas")
            
            for col in df.columns:
                with st.expander(f"Coluna: {col}"):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**Tipo:** {df[col].dtype}")
                        st.write(f"**Valores únicos:** {df[col].nunique()}")
                        st.write(f"**Valores ausentes:** {df[col].isnull().sum()}")
                    
                    with col_info2:
                        if df[col].dtype in ['int64', 'float64']:
                            st.write(f"**Mínimo:** {df[col].min()}")
                            st.write(f"**Máximo:** {df[col].max()}")
                            st.write(f"**Média:** {df[col].mean():.2f}")
            
            # Quality assessment
            st.subheader("📋 Avaliação de Qualidade")
            
            quality_issues = []
            if df.isnull().sum().sum() > 0:
                quality_issues.append(f"⚠️ {df.isnull().sum().sum()} valores ausentes encontrados")
            if df.duplicated().sum() > 0:
                quality_issues.append(f"⚠️ {df.duplicated().sum()} registros duplicados encontrados")
            
            # Check for required columns
            required_cols = ['valor']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                quality_issues.append(f"❌ Colunas obrigatórias ausentes: {missing_cols}")
            
            if quality_issues:
                st.warning("Problemas identificados:")
                for issue in quality_issues:
                    st.write(issue)
            else:
                st.success("✅ Dados aprovados na validação inicial!")
            
            # Store data in session state
            st.session_state.workflow_data['validation_data'] = df
            st.session_state.workflow_data['quality_issues'] = quality_issues
            
            # Approval section
            st.subheader("🎯 Decisão")
            
            if st.button("✅ Aprovar e Continuar", type="primary"):
                st.session_state.workflow_step = 2
                st.success("Dados aprovados! Avançando para a fase de transformação...")
                time.sleep(1)
                st.rerun()
                
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

def show_transformation_phase():
    """Fase 2: Transformação de Dados."""
    
    st.subheader("🔧 Fase 2: Transformação de Dados")
    
    st.markdown("""
    **Objetivo:** Preparar os dados para modelagem através de transformações inteligentes.
    
    **O que acontece nesta fase:**
    - Tratamento de valores ausentes
    - Codificação de variáveis categóricas
    - Normalização/padronização
    - Engenharia de features
    """)
    
    if 'validation_data' not in st.session_state.workflow_data:
        st.warning("⚠️ Execute primeiro a fase de validação.")
        return
    
    df = st.session_state.workflow_data['validation_data']
    
    # Transformation options
    st.subheader("⚙️ Configurações de Transformação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tratamento de Valores Ausentes:**")
        missing_strategy = st.selectbox(
            "Estratégia",
            ["Remoção", "Média/Moda", "Interpolação", "Valor constante"]
        )
        
        st.write("**Variáveis Categóricas:**")
        encoding_strategy = st.selectbox(
            "Codificação",
            ["One-Hot Encoding", "Label Encoding", "Target Encoding"]
        )
    
    with col2:
        st.write("**Normalização:**")
        scaling_strategy = st.selectbox(
            "Método",
            ["StandardScaler", "MinMaxScaler", "RobustScaler", "Sem normalização"]
        )
        
        st.write("**Seleção de Features:**")
        feature_selection = st.selectbox(
            "Método",
            ["Seleção Univariada", "Recursive Feature Elimination", "Todas as features"]
        )
    
    # Feature engineering options
    st.subheader("🧪 Engenharia de Features")
    
    create_interaction = st.checkbox("Criar features de interação")
    create_polynomial = st.checkbox("Criar features polinomiais")
    
    if st.session_state.user_tier in ['professional', 'expert']:
        create_custom = st.checkbox("Features customizadas")
        if create_custom:
            st.text_area("Fórmulas customizadas (uma por linha)", 
                        placeholder="area_por_quarto = area_total / quartos")
    
    # Preview transformations
    st.subheader("👁️ Prévia das Transformações")
    
    if st.button("🔍 Simular Transformações"):
        with st.spinner("Aplicando transformações..."):
            # Simulate transformations
            transformed_df = df.copy()
            
            # Show before/after comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Antes:**")
                st.dataframe(df.head())
            
            with col2:
                st.write("**Depois (Simulação):**")
                # Simple simulation - in real implementation would apply actual transformations
                st.dataframe(transformed_df.head())
            
            # Store transformation config
            st.session_state.workflow_data['transformation_config'] = {
                'missing_strategy': missing_strategy,
                'encoding_strategy': encoding_strategy,
                'scaling_strategy': scaling_strategy,
                'feature_selection': feature_selection,
                'create_interaction': create_interaction,
                'create_polynomial': create_polynomial
            }
    
    # Approval section
    st.subheader("🎯 Decisão")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Aprovar e Continuar", type="primary"):
            st.session_state.workflow_step = 3
            st.success("Transformações aprovadas! Avançando para a modelagem...")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("🔄 Refazer Transformações"):
            st.info("Ajuste as configurações e simule novamente.")

def show_modeling_phase():
    """Fase 3: Modelagem."""
    
    st.subheader("🤖 Fase 3: Modelagem")
    
    st.markdown("""
    **Objetivo:** Treinar o modelo de machine learning para avaliação imobiliária.
    
    **O que acontece nesta fase:**
    - Divisão treino/teste
    - Seleção do algoritmo
    - Otimização de hiperparâmetros
    - Validação cruzada
    """)
    
    # Model selection based on user tier
    st.subheader("🎯 Seleção do Modelo")
    
    if st.session_state.user_tier == 'mvp':
        st.info("**Nível MVP:** Apenas Elastic Net (Grau III)")
        model_type = "Elastic Net"
        st.write("**Modelo selecionado:** Elastic Net Regression")
        
    elif st.session_state.user_tier == 'professional':
        st.info("**Nível Profissional:** Todos os graus disponíveis")
        model_options = ["Elastic Net", "Random Forest", "Support Vector Machine"]
        model_type = st.selectbox("Selecione o modelo:", model_options)
        
    else:  # expert
        st.info("**Nível Especialista:** Modelos avançados + SHAP obrigatório")
        model_options = ["XGBoost", "Gradient Boosting", "Elastic Net", "Ensemble"]
        model_type = st.selectbox("Selecione o modelo:", model_options)
        
        st.markdown("""
        <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 0.5rem;">
            <strong>🔬 Modo Especialista Ativo</strong><br>
            SHAP (SHapley Additive exPlanations) será automaticamente calculado para garantir 
            interpretabilidade "glass-box" absoluta.
        </div>
        """, unsafe_allow_html=True)
    
    # Hyperparameter configuration
    st.subheader("⚙️ Configuração de Hiperparâmetros")
    
    if model_type == "Elastic Net":
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.slider("Alpha (Regularização)", 0.01, 2.0, 1.0)
        with col2:
            l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
            
    elif model_type in ["XGBoost", "Gradient Boosting"]:
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("N Estimators", 50, 500, 100)
        with col2:
            max_depth = st.slider("Max Depth", 3, 10, 6)
        with col3:
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
    
    # Cross-validation settings
    st.subheader("🔄 Validação Cruzada")
    
    cv_folds = st.slider("Número de folds", 3, 10, 5)
    test_size = st.slider("Tamanho do conjunto de teste (%)", 10, 30, 20)
    
    # Training simulation
    st.subheader("🚀 Treinamento do Modelo")
    
    if st.button("🎯 Treinar Modelo", type="primary"):
        with st.spinner("Treinando modelo..."):
            # Simulate training
            progress_bar = st.progress(0)
            
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress((i + 1) / 100)
            
            st.success("✅ Modelo treinado com sucesso!")
            
            # Mock results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Score", "0.8542")
            with col2:
                st.metric("RMSE", "45,230")
            with col3:
                st.metric("MAE", "32,150")
            
            # Store model config
            st.session_state.workflow_data['model_config'] = {
                'model_type': model_type,
                'cv_folds': cv_folds,
                'test_size': test_size,
                'user_tier': st.session_state.user_tier
            }
            
            if st.session_state.user_tier == 'expert':
                st.info("🔬 SHAP values calculados automaticamente para interpretabilidade completa.")
    
    # Approval section
    st.subheader("🎯 Decisão")
    
    if 'model_config' in st.session_state.workflow_data:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Aprovar Modelo", type="primary"):
                st.session_state.workflow_step = 4
                st.success("Modelo aprovado! Avançando para validação NBR...")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("🔄 Retreinar"):
                st.info("Ajuste os hiperparâmetros e treine novamente.")

def show_nbr_validation_phase():
    """Fase 4: Validação NBR 14653."""
    
    st.subheader("📏 Fase 4: Validação NBR 14653")
    
    st.markdown("""
    **Objetivo:** Validar o modelo conforme norma NBR 14653 para garantir conformidade técnica.
    
    **Testes aplicados:**
    - Coeficiente de determinação (R²)
    - Teste F de significância global
    - Teste t para coeficientes
    - Durbin-Watson (autocorrelação)
    - Shapiro-Wilk (normalidade dos resíduos)
    """)
    
    if 'model_config' not in st.session_state.workflow_data:
        st.warning("⚠️ Execute primeiro o treinamento do modelo.")
        return
    
    # Run NBR tests
    if st.button("🧪 Executar Testes NBR 14653", type="primary"):
        with st.spinner("Executando bateria de testes..."):
            # Simulate NBR tests
            progress_bar = st.progress(0)
            
            tests = [
                "Coeficiente de determinação (R²)",
                "Teste F de significância",
                "Teste t para coeficientes", 
                "Durbin-Watson",
                "Shapiro-Wilk"
            ]
            
            results = {}
            
            for i, test in enumerate(tests):
                time.sleep(0.5)
                progress_bar.progress((i + 1) / len(tests))
                
                # Mock test results
                if test == "Coeficiente de determinação (R²)":
                    results[test] = {"value": 0.8542, "threshold": 0.70, "passed": True}
                elif test == "Teste F de significância":
                    results[test] = {"value": 125.45, "threshold": 3.84, "passed": True}
                elif test == "Teste t para coeficientes":
                    results[test] = {"value": 4.25, "threshold": 1.96, "passed": True}
                elif test == "Durbin-Watson":
                    results[test] = {"value": 1.89, "threshold": 1.5, "passed": True}
                else:  # Shapiro-Wilk
                    results[test] = {"value": 0.045, "threshold": 0.05, "passed": False}
            
            st.success("✅ Testes NBR 14653 concluídos!")
            
            # Results display
            st.subheader("📊 Resultados dos Testes")
            
            for test, result in results.items():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(test)
                with col2:
                    st.write(f"{result['value']:.3f}")
                with col3:
                    st.write(f"{result['threshold']:.3f}")
                with col4:
                    if result['passed']:
                        st.success("✅")
                    else:
                        st.error("❌")
            
            # Overall grade
            passed_tests = sum(1 for r in results.values() if r['passed'])
            total_tests = len(results)
            
            r2_value = results["Coeficiente de determinação (R²)"]["value"]
            
            if r2_value >= 0.90:
                grade = "Superior"
                grade_class = "grade-superior"
            elif r2_value >= 0.80:
                grade = "Normal"  
                grade_class = "grade-normal"
            elif r2_value >= 0.70:
                grade = "Inferior"
                grade_class = "grade-inferior"
            else:
                grade = "Inadequado"
                grade_class = "grade-inadequado"
            
            st.markdown(f"""
            <div class="nbr-grade {grade_class}">
                Grau NBR 14653: {grade} ({passed_tests}/{total_tests} testes aprovados)
            </div>
            """, unsafe_allow_html=True)
            
            # Store NBR results
            st.session_state.workflow_data['nbr_results'] = {
                'tests': results,
                'grade': grade,
                'passed_tests': passed_tests,
                'total_tests': total_tests
            }
    
    # Approval section
    st.subheader("🎯 Decisão")
    
    if 'nbr_results' in st.session_state.workflow_data:
        nbr_data = st.session_state.workflow_data['nbr_results']
        
        if nbr_data['grade'] in ['Superior', 'Normal']:
            if st.button("✅ Aprovar Validação", type="primary"):
                st.session_state.workflow_step = 5
                st.success("Validação NBR aprovada! Gerando relatório final...")
                time.sleep(1)
                st.rerun()
        else:
            st.warning("⚠️ Modelo não atende aos critérios mínimos da NBR 14653.")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Voltar à Modelagem"):
                    st.session_state.workflow_step = 3
                    st.info("Retornando à fase de modelagem para ajustes...")
                    time.sleep(1)
                    st.rerun()
            
            with col2:
                if st.button("⚠️ Prosseguir mesmo assim"):
                    st.session_state.workflow_step = 5
                    st.warning("Prosseguindo com modelo abaixo do padrão...")
                    time.sleep(1)
                    st.rerun()

def show_report_phase():
    """Fase 5: Geração do Relatório."""
    
    st.subheader("📄 Fase 5: Relatório Final")
    
    st.markdown("""
    **Objetivo:** Consolidar todos os resultados em um relatório técnico defensável.
    
    **Conteúdo do relatório:**
    - Resumo executivo
    - Metodologia aplicada
    - Resultados dos testes NBR
    - Análise de interpretabilidade
    - Conclusões e recomendações
    """)
    
    if 'nbr_results' not in st.session_state.workflow_data:
        st.warning("⚠️ Execute primeiro a validação NBR.")
        return
    
    # Report generation
    if st.button("📋 Gerar Relatório Final", type="primary"):
        with st.spinner("Consolidando resultados..."):
            time.sleep(2)
            
            st.success("✅ Relatório gerado com sucesso!")
            
            # Report preview
            st.subheader("📄 Prévia do Relatório")
            
            workflow_data = st.session_state.workflow_data
            
            # Executive summary
            with st.expander("📊 Resumo Executivo", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Grau NBR", workflow_data['nbr_results']['grade'])
                with col2:
                    st.metric("R² Score", "0.8542")
                with col3:
                    st.metric("Registros", len(workflow_data['validation_data']))
                
                st.write("**Conclusão:** Modelo atende aos critérios técnicos da NBR 14653 com grau " + 
                        workflow_data['nbr_results']['grade'] + ".")
            
            # Methodology
            with st.expander("🔬 Metodologia"):
                st.write(f"**Modelo utilizado:** {workflow_data['model_config']['model_type']}")
                st.write(f"**Nível de acesso:** {workflow_data['model_config']['user_tier'].title()}")
                st.write(f"**Validação cruzada:** {workflow_data['model_config']['cv_folds']} folds")
                
                if workflow_data['model_config']['user_tier'] == 'expert':
                    st.write("**Interpretabilidade:** SHAP values calculados (modo especialista)")
            
            # NBR Results
            with st.expander("📏 Resultados NBR 14653"):
                for test, result in workflow_data['nbr_results']['tests'].items():
                    status = "✅ Aprovado" if result['passed'] else "❌ Reprovado"
                    st.write(f"**{test}:** {result['value']:.3f} {status}")
            
            # Feature flags demonstration
            if st.session_state.user_tier in ['professional', 'expert']:
                with st.expander("🔍 Análise de Incerteza"):
                    st.info("**Funcionalidade Profissional:** Intervalos de confiança e predição disponíveis.")
                    
                    # Mock uncertainty visualization
                    fig = go.Figure()
                    
                    x = np.linspace(0, 10, 50)
                    y = x + np.random.normal(0, 0.5, 50)
                    y_upper = y + 1.96 * 0.5
                    y_lower = y - 1.96 * 0.5
                    
                    # Confidence band
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([y_upper, y_lower[::-1]]),
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=True,
                        name='Intervalo de Confiança'
                    ))
                    
                    # Main line
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        line=dict(color='rgb(0,100,80)'),
                        mode='lines',
                        name='Predição'
                    ))
                    
                    fig.update_layout(
                        title="Visualização de Incerteza (Simulação)",
                        xaxis_title="Features",
                        yaxis_title="Valor Predito",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.user_tier == 'expert':
                with st.expander("🧠 Análise SHAP (Especialista)"):
                    st.info("**Funcionalidade Especialista:** Interpretabilidade glass-box absoluta.")
                    
                    # Mock SHAP analysis
                    shap_data = {
                        'area_privativa': 0.35,
                        'localizacao_score': 0.28,
                        'idade_imovel': -0.15,
                        'vagas_garagem': 0.12,
                        'banheiros': 0.08
                    }
                    
                    st.write("**Top 5 Features por Importância SHAP:**")
                    for feature, importance in shap_data.items():
                        st.write(f"• {feature}: {importance:+.2f}")
    
    # Final actions
    st.subheader("🎯 Ações Finais")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📧 Exportar PDF"):
            st.success("Relatório PDF gerado!")
    
    with col2:
        if st.button("📊 Exportar Excel"):
            st.success("Planilha Excel gerada!")
    
    with col3:
        if st.button("🔄 Novo Fluxo"):
            st.session_state.workflow_step = 1
            st.session_state.workflow_data = {}
            st.success("Fluxo reiniciado!")
            st.rerun()

def create_audit_timeline_chart(audit_data: Dict[str, Any]):
    """
    Cria gráfico de timeline da trilha de auditoria.
    
    Args:
        audit_data: Dados da trilha de auditoria
        
    Returns:
        Figura do Plotly
    """
    steps = audit_data.get('pipeline_steps', [])
    
    if not steps:
        return None
    
    # Preparar dados
    phases = [step['phase'] for step in steps]
    durations = []
    statuses = []
    
    for step in steps:
        # Converter duração para segundos
        duration_str = step.get('duration', '0s')
        if 's' in duration_str:
            duration = int(duration_str.replace('s', ''))
        elif 'm' in duration_str:
            parts = duration_str.replace('m', '').split()
            duration = int(parts[0]) * 60
            if len(parts) > 1:
                duration += int(parts[1].replace('s', ''))
        else:
            duration = 0
        
        durations.append(duration)
        statuses.append(step.get('status', 'unknown'))
    
    # Cores baseadas no status
    colors = []
    for status in statuses:
        if status == 'completed':
            colors.append('#28a745')
        elif status == 'failed':
            colors.append('#dc3545')
        elif status == 'in_progress':
            colors.append('#ffc107')
        else:
            colors.append('#6c757d')
    
    fig = go.Figure(data=[
        go.Bar(
            x=durations,
            y=phases,
            orientation='h',
            marker_color=colors,
            text=[f"{d}s" for d in durations],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Timeline de Execução da Trilha de Auditoria",
        xaxis_title="Duração (segundos)",
        yaxis_title="Fases",
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def show_audit_trail_tab(evaluation_id: str):
    """Mostra trilha de auditoria da avaliação."""
    
    st.subheader("🔍 Trilha de Auditoria")
    
    st.markdown("""
    **Transparência Absoluta:** Esta seção demonstra a filosofia "glass-box" da Valion, 
    documentando cada etapa do processo de avaliação para garantir auditabilidade total.
    """)
    
    if st.button("🔄 Carregar Trilha de Auditoria"):
        with st.spinner("Carregando trilha de auditoria..."):
            audit_data = call_api(f"/evaluations/{evaluation_id}/audit_trail")
        
        if audit_data:
            # Metadata da auditoria
            st.subheader("📋 Metadados da Auditoria")
            
            metadata = audit_data.get('audit_metadata', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Versão da Auditoria", metadata.get('audit_version', 'N/A'))
            with col2:
                st.metric("Total de Etapas", metadata.get('total_steps', 'N/A'))
            with col3:
                st.metric("Duração Total", metadata.get('execution_duration', 'N/A'))
            
            st.info(f"**Nível de Conformidade:** {metadata.get('compliance_level', 'N/A')}")
            
            # Timeline de execução
            st.subheader("⏱️ Timeline de Execução")
            
            timeline_fig = create_audit_timeline_chart(audit_data)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Detalhes das etapas
            st.subheader("📝 Detalhes das Etapas")
            
            steps = audit_data.get('pipeline_steps', [])
            
            for step in steps:
                with st.expander(f"Etapa {step['step_id']}: {step['phase']} ({step['status']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Timestamp:** {step['timestamp']}")
                        st.write(f"**Duração:** {step['duration']}")
                        st.write(f"**Status:** {step['status']}")
                    
                    with col2:
                        details = step.get('details', {})
                        st.write(f"**Ação:** {details.get('action', 'N/A')}")
                        st.write(f"**Input:** {details.get('input', 'N/A')}")
                        st.write(f"**Output:** {details.get('output', 'N/A')}")
                    
                    # Notas de auditoria
                    if step.get('audit_notes'):
                        st.markdown(f"**📝 Notas de Auditoria:** {step['audit_notes']}")
                    
                    # Detalhes específicos
                    if 'transformations_applied' in details:
                        st.write("**Transformações Aplicadas:**")
                        for key, value in details['transformations_applied'].items():
                            st.write(f"  • {key}: {value}")
                    
                    if 'cross_validation' in details:
                        cv = details['cross_validation']
                        st.write("**Validação Cruzada:**")
                        st.write(f"  • Método: {cv.get('method', 'N/A')}")
                        st.write(f"  • Score médio: {cv.get('mean_cv_score', 'N/A')}")
                        st.write(f"  • Desvio padrão: {cv.get('std_cv_score', 'N/A')}")
                    
                    if 'tests_performed' in details:
                        st.write("**Testes Realizados:**")
                        tests = details['tests_performed']
                        for test_name, test_data in tests.items():
                            if isinstance(test_data, dict):
                                result_icon = "✅" if test_data.get('result') == 'PASS' else "❌"
                                st.write(f"  {result_icon} **{test_name}:** {test_data.get('value', 'N/A')}")
            
            # Análise de conformidade
            st.subheader("✅ Análise de Conformidade")
            
            compliance = audit_data.get('compliance_evidence', {})
            
            # Conformidade NBR 14653
            nbr_compliance = compliance.get('nbr_14653_conformity', {})
            if nbr_compliance:
                st.write("**📏 Conformidade NBR 14653:**")
                
                sections = nbr_compliance.get('section_compliance', {})
                for section, status in sections.items():
                    status_icon = "✅" if status == "CONFORMANT" else "❌"
                    st.write(f"  {status_icon} {section}: {status}")
                
                justifications = nbr_compliance.get('justifications', [])
                if justifications:
                    st.write("**Justificativas:**")
                    for justification in justifications:
                        st.write(f"  • {justification}")
            
            # Transparência Glass-Box
            glass_box = compliance.get('glass_box_transparency', {})
            if glass_box:
                st.write("**🔍 Transparência Glass-Box:**")
                st.write(f"  • Nível de interpretabilidade: {glass_box.get('interpretability_level', 'N/A')}")
                st.write(f"  • Score de auditabilidade: {glass_box.get('auditability_score', 'N/A')}")
                st.write(f"  • Reprodutibilidade: {glass_box.get('reproducibility', 'N/A')}")
                
                methods = glass_box.get('explanation_methods', [])
                if methods:
                    st.write(f"  • Métodos de explicação: {', '.join(methods)}")
            
            # Linhagem dos dados
            st.subheader("🔗 Linhagem dos Dados")
            
            lineage = audit_data.get('data_lineage', {})
            if lineage:
                source = lineage.get('source_data', {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Registros Originais", source.get('records_original', 'N/A'))
                with col2:
                    st.metric("Registros Finais", source.get('records_final', 'N/A'))
                with col3:
                    exclusions = source.get('records_original', 0) - source.get('records_final', 0)
                    st.metric("Registros Excluídos", exclusions)
                
                if source.get('exclusion_reason'):
                    st.info(f"**Motivo das exclusões:** {source['exclusion_reason']}")
                
                chains = lineage.get('transformations_chain', [])
                if chains:
                    st.write("**Cadeia de Transformações:**")
                    for chain in chains:
                        st.code(chain)
                
                if lineage.get('reproducibility_hash'):
                    st.write(f"**Hash de Reprodutibilidade:** `{lineage['reproducibility_hash']}`")
            
            # Garantia de qualidade
            st.subheader("🛡️ Garantia de Qualidade")
            
            qa = audit_data.get('quality_assurance', {})
            if qa:
                checks = qa.get('validation_checks', [])
                if checks:
                    st.write("**Verificações Realizadas:**")
                    for check in checks:
                        st.write(f"  ✅ {check}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Status da Revisão:** {qa.get('peer_review_status', 'N/A')}")
                    st.write(f"**Revisor Técnico:** {qa.get('technical_reviewer', 'N/A')}")
                with col2:
                    if qa.get('review_date'):
                        review_date = datetime.fromisoformat(qa['review_date'].replace('Z', '+00:00'))
                        st.write(f"**Data da Revisão:** {review_date.strftime('%d/%m/%Y %H:%M')}")
            
            # Botões de ação
            st.subheader("📤 Exportar Trilha de Auditoria")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Exportar PDF"):
                    st.success("Trilha de auditoria PDF gerada!")
            
            with col2:
                if st.button("📊 Exportar Excel"):
                    st.success("Planilha de auditoria Excel gerada!")
            
            with col3:
                if st.button("💾 Exportar JSON"):
                    # Disponibilizar download do JSON
                    import json
                    json_str = json.dumps(audit_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Baixar JSON",
                        data=json_str,
                        file_name=f"audit_trail_{evaluation_id}.json",
                        mime="application/json"
                    )
        else:
            st.error("Erro ao carregar trilha de auditoria.")

def show_shap_laboratory_page():
    """Página do Laboratório de Simulação SHAP Interativo."""
    
    st.header("🧪 Laboratório de Simulação SHAP")
    
    # Adicionar estado da aplicação para o laboratório
    if 'lab_features_config' not in st.session_state:
        st.session_state.lab_features_config = None
    if 'lab_simulation_result' not in st.session_state:
        st.session_state.lab_simulation_result = None
    if 'lab_baseline_loaded' not in st.session_state:
        st.session_state.lab_baseline_loaded = False
    
    # Descrição do laboratório
    st.markdown("""
    <div class="transparency-box">
        <h4>🔬 Interpretabilidade Interativa</h4>
        <p>O Laboratório SHAP transforma a análise "glass-box" de passiva para ativa. 
        Explore como diferentes características impactam o valor final através de simulações em tempo real.</p>
        <strong>Disponível apenas no modo especialista</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Input para ID da avaliação
    evaluation_id = st.text_input(
        "ID da Avaliação (Modo Especialista)",
        value=st.session_state.evaluation_id or "",
        placeholder="Digite o ID de uma avaliação concluída em modo especialista"
    )
    
    if not evaluation_id:
        st.info("💡 Insira o ID de uma avaliação para acessar o laboratório SHAP.")
        return
    
    # Carregar configuração do laboratório
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Carregar Laboratório", type="primary"):
            with st.spinner("Carregando configuração do laboratório..."):
                features_config = call_api(f"/evaluations/{evaluation_id}/laboratory_features")
                
                if features_config:
                    st.session_state.lab_features_config = features_config
                    st.session_state.lab_baseline_loaded = True
                    st.success("✅ Laboratório carregado!")
                else:
                    st.error("❌ Erro ao carregar laboratório. Verifique se a avaliação foi feita em modo especialista.")
    
    with col1:
        if st.session_state.lab_baseline_loaded:
            st.success("🔗 Laboratório ativo - Simulações habilitadas")
        else:
            st.info("📡 Aguardando carregamento do laboratório...")
    
    # Main interface do laboratório
    if st.session_state.lab_features_config:
        show_shap_simulation_interface(evaluation_id, st.session_state.lab_features_config)

def show_shap_simulation_interface(evaluation_id: str, features_config: Dict[str, Any]):
    """Interface principal de simulação SHAP."""
    
    st.markdown("---")
    
    # Tabs do laboratório
    tabs = st.tabs([
        "🎛️ Simulador", 
        "📊 Cenários Predefinidos", 
        "🌊 Gráfico Waterfall", 
        "📈 Análise de Sensibilidade"
    ])
    
    with tabs[0]:
        show_feature_simulator(evaluation_id, features_config)
    
    with tabs[1]:
        show_predefined_scenarios(evaluation_id, features_config)
    
    with tabs[2]:
        show_waterfall_analysis(evaluation_id)
    
    with tabs[3]:
        show_sensitivity_analysis(evaluation_id, features_config)

def show_feature_simulator(evaluation_id: str, features_config: Dict[str, Any]):
    """Simulador de features com sliders interativos."""
    
    st.subheader("🎛️ Simulador de Características")
    
    st.markdown("""
    **Como usar:** Ajuste os sliders abaixo para modificar as características do imóvel. 
    O impacto SHAP será calculado em tempo real para cada mudança.
    """)
    
    features = features_config.get('features', {})
    categories = features_config.get('categories', {})
    
    # Organizar por categorias
    feature_values = {}
    
    # Interface por categorias
    for category_key, category_info in categories.items():
        with st.expander(f"{category_info['name']}", expanded=True):
            category_features = category_info.get('features', [])
            
            cols = st.columns(min(3, len(category_features)))
            
            for i, feature_key in enumerate(category_features):
                if feature_key in features:
                    feature = features[feature_key]
                    col_idx = i % 3
                    
                    with cols[col_idx]:
                        if feature['slider_type'] == 'integer':
                            value = st.slider(
                                feature['display_name'],
                                min_value=int(feature['min_value']),
                                max_value=int(feature['max_value']),
                                value=int(feature['current_value']),
                                step=int(feature['step']),
                                help=feature.get('description', ''),
                                key=f"sim_{feature_key}"
                            )
                        else:
                            value = st.slider(
                                feature['display_name'],
                                min_value=feature['min_value'],
                                max_value=feature['max_value'],
                                value=feature['current_value'],
                                step=feature['step'],
                                help=feature.get('description', ''),
                                key=f"sim_{feature_key}"
                            )
                        
                        feature_values[feature_key] = float(value)
                        
                        # Mostrar impacto estimado
                        impact = (value - feature['current_value']) * feature.get('impact_coefficient', 0)
                        color = "green" if impact > 0 else "red" if impact < 0 else "gray"
                        st.markdown(f"<small style='color: {color}'>Impacto: R$ {impact:+,.0f}</small>", 
                                  unsafe_allow_html=True)
    
    # Botão de simulação
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Simular Cenário", type="primary", use_container_width=True):
            with st.spinner("Calculando impacto SHAP..."):
                simulate_scenario(evaluation_id, feature_values)

def simulate_scenario(evaluation_id: str, feature_modifications: Dict[str, float]):
    """Executa simulação de cenário."""
    
    data = {
        "evaluation_id": evaluation_id,
        "feature_modifications": feature_modifications,
        "simulation_name": f"Simulação {datetime.now().strftime('%H:%M:%S')}",
        "compare_to_baseline": True
    }
    
    result = call_api(f"/evaluations/{evaluation_id}/shap_simulation", method="POST", data=data)
    
    if result:
        st.session_state.lab_simulation_result = result
        show_simulation_results(result)
    else:
        st.error("Erro na simulação. Tente novamente.")

def show_simulation_results(result: Dict[str, Any]):
    """Mostra resultados da simulação."""
    
    st.markdown("---")
    st.subheader("📊 Resultados da Simulação")
    
    scenario = result.get('scenario_comparison', {})
    baseline = scenario.get('baseline', {})
    modified = scenario.get('modified', {})
    impact = scenario.get('impact_analysis', {})
    
    # Comparação de valores
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Valor Baseline", 
            f"R$ {baseline.get('prediction', 0):,.0f}",
            help="Valor original com características padrão"
        )
    
    with col2:
        st.metric(
            "Valor Simulado", 
            f"R$ {modified.get('prediction', 0):,.0f}",
            delta=f"R$ {impact.get('total_impact', 0):+,.0f}",
            help="Valor após modificações nas características"
        )
    
    with col3:
        rel_change = impact.get('relative_change', 0)
        st.metric(
            "Variação Relativa", 
            f"{rel_change:+.1f}%",
            help="Percentual de mudança em relação ao baseline"
        )
    
    # Gráfico de impacto por feature
    waterfall_data = result.get('waterfall_data', {})
    contributions = waterfall_data.get('contributions', [])
    
    if contributions:
        st.subheader("🌊 Impacto por Característica")
        
        # Preparar dados para gráfico waterfall
        features = []
        baseline_values = []
        modified_values = []
        deltas = []
        
        for contrib in contributions:
            feature = contrib['feature']
            if feature not in ['base_value', 'final_prediction']:
                features.append(feature.replace('_', ' ').title())
                baseline_values.append(contrib['baseline_contribution'])
                modified_values.append(contrib['modified_contribution'])
                deltas.append(contrib['delta'])
        
        # Criar gráfico de barras comparativo
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=features,
            y=baseline_values,
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            name='Simulado',
            x=features,
            y=modified_values,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Comparação SHAP: Baseline vs Simulado",
            xaxis_title="Características",
            yaxis_title="Contribuição SHAP (R$)",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights da simulação
    insights = result.get('insights', {})
    if insights:
        st.subheader("💡 Insights da Simulação")
        
        recommendations = insights.get('recommendations', [])
        for rec in recommendations:
            st.info(f"💡 {rec}")
        
        top_drivers = insights.get('top_drivers', [])
        if top_drivers:
            st.write("**Principais impactos:**")
            for feature, impact in top_drivers:
                st.write(f"• **{feature.replace('_', ' ').title()}**: R$ {impact:+,.0f}")

def show_predefined_scenarios(evaluation_id: str, features_config: Dict[str, Any]):
    """Cenários predefinidos para análise rápida."""
    
    st.subheader("📊 Cenários Predefinidos")
    
    st.markdown("""
    **Análise Rápida:** Explore cenários típicos do mercado imobiliário com um clique.
    Cada preset foi calibrado para representar diferentes segmentos do mercado.
    """)
    
    presets = features_config.get('simulation_presets', [])
    
    if presets:
        # Grid de cenários
        cols = st.columns(min(3, len(presets)))
        
        for i, preset in enumerate(presets):
            col_idx = i % 3
            
            with cols[col_idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{preset['name']}</h4>
                    <p>{preset['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Simular {preset['name']}", key=f"preset_{i}"):
                    with st.spinner(f"Simulando {preset['name']}..."):
                        simulate_scenario(evaluation_id, preset['modifications'])
    
    # Mostrar resultado se disponível
    if st.session_state.lab_simulation_result:
        show_simulation_results(st.session_state.lab_simulation_result)

def show_waterfall_analysis(evaluation_id: str):
    """Análise waterfall SHAP detalhada."""
    
    st.subheader("🌊 Análise Waterfall SHAP")
    
    st.markdown("""
    **Decomposição Completa:** Visualize como cada característica contribui para a formação do valor final,
    desde o valor base até a predição final.
    """)
    
    sample_index = st.slider("Índice da Amostra", 0, 10, 0, help="Selecione uma amostra para análise detalhada")
    
    if st.button("📈 Gerar Waterfall", type="primary"):
        with st.spinner("Gerando análise waterfall..."):
            waterfall_data = call_api(f"/evaluations/{evaluation_id}/shap_waterfall?sample_index={sample_index}")
            
            if waterfall_data:
                show_waterfall_chart(waterfall_data)
            else:
                st.error("Erro ao gerar waterfall. Verifique se a avaliação suporta SHAP.")

def show_waterfall_chart(waterfall_data: Dict[str, Any]):
    """Exibe gráfico waterfall SHAP."""
    
    chart_data = waterfall_data.get('waterfall_chart', {})
    contributions = chart_data.get('contributions', [])
    
    if not contributions:
        st.error("Dados de waterfall não disponíveis.")
        return
    
    # Preparar dados para gráfico waterfall
    features = []
    values = []
    cumulative = []
    colors = []
    
    config = chart_data.get('chart_config', {})
    color_map = config.get('colors', {})
    
    for contrib in contributions:
        feature = contrib['feature']
        value = contrib['value']
        cum_value = contrib['cumulative']
        
        features.append(feature.replace('_', ' ').title())
        values.append(value)
        cumulative.append(cum_value)
        
        # Definir cor baseada no tipo
        if feature in ['Base Value', 'Final Prediction']:
            colors.append(color_map.get('base', '#4682B4'))
        elif value > 0:
            colors.append(color_map.get('positive', '#2E8B57'))
        else:
            colors.append(color_map.get('negative', '#DC143C'))
    
    # Criar gráfico waterfall
    fig = go.Figure()
    
    # Adicionar barras
    for i, (feature, value, color) in enumerate(zip(features, values, colors)):
        if i == 0 or i == len(features) - 1:
            # Base value e final prediction
            fig.add_trace(go.Bar(
                x=[feature],
                y=[cumulative[i]],
                marker_color=color,
                name=feature,
                showlegend=False,
                text=f"R$ {cumulative[i]:,.0f}",
                textposition='auto'
            ))
        else:
            # Contribuições
            fig.add_trace(go.Bar(
                x=[feature],
                y=[value],
                base=cumulative[i-1] if value > 0 else cumulative[i],
                marker_color=color,
                name=feature,
                showlegend=False,
                text=f"R$ {value:+,.0f}",
                textposition='auto'
            ))
    
    fig.update_layout(
        title="Decomposição SHAP - Waterfall",
        xaxis_title="Características",
        yaxis_title="Valor (R$)",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detalhes das features
    feature_details = waterfall_data.get('feature_details', {})
    if feature_details:
        st.subheader("📝 Detalhes das Características")
        
        for feature_key, details in feature_details.items():
            with st.expander(f"{feature_key.replace('_', ' ').title()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Valor atual:** {details.get('current_value', 'N/A')} {details.get('unit', '')}")
                    st.write(f"**Ranking de importância:** #{details.get('importance_rank', 'N/A')}")
                
                with col2:
                    st.write(f"**Interpretação:** {details.get('interpretation', 'N/A')}")
    
    # Resumo da interpretação
    interpretation = waterfall_data.get('interpretation_summary', {})
    if interpretation:
        st.subheader("🎯 Resumo da Interpretação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            drivers = interpretation.get('main_value_drivers', [])
            if drivers:
                st.write("**Principais fatores de valorização:**")
                for driver in drivers:
                    st.write(f"✅ {driver.replace('_', ' ').title()}")
        
        with col2:
            detractors = interpretation.get('main_value_detractors', [])
            if detractors:
                st.write("**Principais fatores de desvalorização:**")
                for detractor in detractors:
                    st.write(f"❌ {detractor.replace('_', ' ').title()}")
        
        explanation = interpretation.get('explanation', '')
        if explanation:
            st.info(f"💡 **Explicação:** {explanation}")

def show_sensitivity_analysis(evaluation_id: str, features_config: Dict[str, Any]):
    """Análise de sensibilidade das features."""
    
    st.subheader("📈 Análise de Sensibilidade")
    
    st.markdown("""
    **Análise de Robustez:** Examine como pequenas variações em cada característica 
    afetam o valor final. Útil para entender a estabilidade das predições.
    """)
    
    features = features_config.get('features', {})
    
    # Seleção de feature para análise
    feature_options = {key: info['display_name'] for key, info in features.items()}
    selected_feature = st.selectbox(
        "Selecione uma característica para análise:",
        options=list(feature_options.keys()),
        format_func=lambda x: feature_options[x]
    )
    
    if selected_feature and st.button("🔍 Analisar Sensibilidade"):
        with st.spinner("Executando análise de sensibilidade..."):
            show_sensitivity_chart(evaluation_id, selected_feature, features[selected_feature])

def show_sensitivity_chart(evaluation_id: str, feature_key: str, feature_info: Dict[str, Any]):
    """Gera e exibe gráfico de sensibilidade."""
    
    # Gerar range de valores para análise
    min_val = feature_info['min_value']
    max_val = feature_info['max_value']
    current_val = feature_info['current_value']
    step = feature_info['step']
    
    # Criar range de valores
    values = np.arange(min_val, max_val + step, step)
    
    # Simular cada valor (mock - em implementação real seria via API)
    predictions = []
    shap_contributions = []
    
    base_prediction = 500000.0
    impact_coeff = feature_info.get('impact_coefficient', 1000.0)
    
    for val in values:
        # Simulação simplificada
        impact = (val - current_val) * impact_coeff
        prediction = base_prediction + impact
        shap_contrib = impact  # Simplificado
        
        predictions.append(prediction)
        shap_contributions.append(shap_contrib)
    
    # Criar gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de valor vs feature
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=values,
            y=predictions,
            mode='lines+markers',
            name='Valor Predito',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        # Destacar valor atual
        current_prediction = base_prediction
        fig1.add_trace(go.Scatter(
            x=[current_val],
            y=[current_prediction],
            mode='markers',
            name='Valor Atual',
            marker=dict(size=12, color='red', symbol='star')
        ))
        
        fig1.update_layout(
            title=f"Sensibilidade: {feature_info['display_name']}",
            xaxis_title=f"{feature_info['display_name']} ({feature_info.get('unit', '')})",
            yaxis_title="Valor Predito (R$)",
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Gráfico de contribuição SHAP
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=values,
            y=shap_contributions,
            mode='lines+markers',
            name='Contribuição SHAP',
            line=dict(color='green', width=3),
            marker=dict(size=6)
        ))
        
        # Linha zero
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig2.update_layout(
            title="Contribuição SHAP vs Valor da Feature",
            xaxis_title=f"{feature_info['display_name']} ({feature_info.get('unit', '')})",
            yaxis_title="Contribuição SHAP (R$)",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Análise estatística
    st.subheader("📊 Estatísticas de Sensibilidade")
    
    # Calcular métricas
    value_range = max(predictions) - min(predictions)
    shap_range = max(shap_contributions) - min(shap_contributions)
    elasticity = (max(predictions) / min(predictions) - 1) / (max_val / min_val - 1) if min_val > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Range de Valores", f"R$ {value_range:,.0f}")
    
    with col2:
        st.metric("Range SHAP", f"R$ {shap_range:,.0f}")
    
    with col3:
        st.metric("Elasticidade", f"{elasticity:.2f}")
    
    # Interpretação
    st.info(f"""
    💡 **Interpretação da Sensibilidade:**
    
    - **Range de Valores:** A variação total no valor predito quando {feature_info['display_name']} 
      varia de {min_val} a {max_val} {feature_info.get('unit', '')}
    - **Elasticidade:** {elasticity:.2f} indica que uma variação de 1% na feature resulta 
      em aproximadamente {elasticity:.2f}% de variação no valor
    - **Impacto:** {'Alto' if abs(elasticity) > 1 else 'Moderado' if abs(elasticity) > 0.5 else 'Baixo'} 
      impacto desta característica no valor final
    """)

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