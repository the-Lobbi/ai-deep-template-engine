FROM python:3.12-slim

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash deepagent

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY --chown=deepagent:deepagent src/ ./src/

# Switch to non-root user
USER deepagent

# Expose MCP server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.deep_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
