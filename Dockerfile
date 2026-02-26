# KG-RAG QA Agent Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    # add frontend libraries used by Streamlit demo
    && pip install --no-cache-dir streamlit pyvis streamlit-agraph

# Copy source code
COPY src/ ./src/

# Copy project files
COPY pyproject.toml README.md ./

# Install the package in development mode
RUN pip install -e .

# Copy config and data files
COPY config/ ./config/
COPY data/ ./data/

# Expose ports (8080 for legacy, 8000 for backend compose)
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "kgrag.api.server:app", "--host", "0.0.0.0", "--port", "8080"]