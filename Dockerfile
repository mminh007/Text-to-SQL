# ─────────────────────────────────────────────────────────────────
# Dockerfile — text-to-sql-finetune (Phase 1: Data Generation)
# Dùng cho: sinh data (--generate, --all, --dedup-only, --split)
# KHÔNG dùng --validate-sql trong Docker (chạy trực tiếp trên Windows)
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="text-to-sql-finetune"
LABEL description="Text-to-SQL dataset generator using Groq LLM"
LABEL phase="1-data-generation"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install base requirements
COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

# Install optional torch CPU (for sentence-transformers dedup)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install data generation extras
RUN pip install --no-cache-dir \
    openai>=1.0.0 \
    sentence-transformers \
    scikit-learn \
    numpy

# Copy source
COPY src/data/ src/data/
COPY prompts/ prompts/
COPY src/__init__.py src/__init__.py

# Data output directory (mounted from host)
RUN mkdir -p data

# Default: run full pipeline
CMD ["python", "src/data/generate_data.py", "--all"]
