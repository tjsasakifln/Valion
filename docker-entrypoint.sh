#!/bin/bash
set -e

# Função para aguardar serviços
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    
    echo "Aguardando $service_name em $host:$port..."
    while ! nc -z "$host" "$port"; do
        sleep 1
    done
    echo "$service_name disponível!"
}

# Aguardar dependências baseado no tipo de serviço
case "${SERVICE_TYPE:-api}" in
    "api")
        wait_for_service postgres 5432 "PostgreSQL"
        wait_for_service redis 6379 "Redis"
        echo "Executando verificações de startup..."
        python startup_check.py
        echo "Iniciando API FastAPI..."
        exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
        ;;
    "worker")
        wait_for_service postgres 5432 "PostgreSQL"
        wait_for_service redis 6379 "Redis"
        echo "Iniciando Celery Worker..."
        exec celery -A src.workers.tasks worker --loglevel=info --concurrency=4
        ;;
    "frontend")
        echo "Iniciando Frontend Streamlit..."
        exec streamlit run frontend.py --server.port=8501 --server.address=0.0.0.0
        ;;
    "scheduler")
        wait_for_service postgres 5432 "PostgreSQL"
        wait_for_service redis 6379 "Redis"
        echo "Iniciando Celery Beat Scheduler..."
        exec celery -A src.workers.tasks beat --loglevel=info
        ;;
    *)
        echo "Tipo de serviço não reconhecido: ${SERVICE_TYPE}"
        echo "Tipos válidos: api, worker, frontend, scheduler"
        exit 1
        ;;
esac