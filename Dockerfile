FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv securely
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Configure working directory
WORKDIR /app

# Copy dependency manifests
COPY pyproject.toml uv.lock ./

# Synchronize production dependencies (bypassing dev/test dependencies)
RUN uv sync --locked --no-dev

# Mount application source and ML models
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/

# Align with Cloud Run expectations
ENV PORT=8080
EXPOSE 8080

# Spin up via uv directly utilizing the virtual environment isolation
CMD ["uv", "run", "uvicorn", "claudio.server.claudio_server:app", "--host", "0.0.0.0", "--port", "8080"]
