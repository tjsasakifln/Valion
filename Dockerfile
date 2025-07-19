# Multi-stage build para otimizar tamanho da imagem
FROM python:3.11-slim as builder

# Configurar variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar apenas requirements primeiro para aproveitar cache do Docker
COPY requirements.txt requirements.txt

# Install all requirements
RUN pip install --no-cache-dir -r requirements.txt

# Estágio final - runtime
FROM python:3.11-slim

# Re-declarar ARG para uso no estágio final
ARG SERVICE_TYPE=api

# Configurar variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV SERVICE_TYPE=${SERVICE_TYPE}

# Instalar dependências runtime mínimas
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r valion && useradd -r -g valion valion

# Copiar Python packages do builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Criar diretório de trabalho
WORKDIR /app

# Copiar apenas arquivos necessários para a aplicação
COPY src/ /app/src/
COPY frontend.py /app/frontend.py

# Criar diretórios necessários com permissões corretas
RUN mkdir -p uploads models reports temp logs && \
    chown -R valion:valion /app

# Trocar para usuário não-root
USER valion

# Expor portas
EXPOSE 8000 8501

# Healthcheck removed - will be configured per service in docker-compose.yml

# Script de entrada flexível baseado no serviço
COPY docker-entrypoint.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
USER valion

ENTRYPOINT ["docker-entrypoint.sh"]