FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by asyncpg (libpq) and DuckDB (build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached until requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project so the server can import db.py, memory/, etc. from root
COPY . .

# Ensure project root is on the Python path so all cross-module imports work
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "mcp_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
