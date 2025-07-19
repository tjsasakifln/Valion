FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create reports directory
RUN mkdir -p /app/reports

# Expose port
EXPOSE 8002

# Run the application
CMD ["uvicorn", "src.microservices.reporting_service.main:app", "--host", "0.0.0.0", "--port", "8002"]