# ─────────────────────────────────────────────────────────────────
# Dockerfile — text-to-sql-finetune
#
# Dùng cho:
#   - Sinh data     : python -m src.data.generate_data --all
#   - Reprocess     : python scripts/reprocess_prompts.py
#   - CoT augment   : python -m src.data.cot_augment
#   - Schema tools  : python -m src.data.schema --diff
#
# KHÔNG dùng --validate-sql trong Docker (cần ODBC Driver trên Windows)
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="text-to-sql-finetune"
LABEL description="Text-to-SQL dataset generator using Groq LLM"

# System deps tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install deps theo 2 tầng để tận dụng Docker layer cache ──────
# Tầng 1: torch CPU-only (lớn, ít thay đổi)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Tầng 2: core requirements (thường xuyên thay đổi hơn)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tầng 3: optional deps cho semantic dedup + schema linker
RUN pip install --no-cache-dir \
    sentence-transformers \
    scikit-learn \
    numpy \
    rank-bm25

# ── Copy source code (package structure) ─────────────────────────
# Copy từng thư mục riêng để tận dụng cache layer tốt hơn
COPY setup.py .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY prompts/ ./prompts/

# Install package ở chế độ editable để src/ import được
RUN pip install --no-cache-dir -e .

# Tạo thư mục output (mount từ host qua volume)
RUN mkdir -p data logs

# ── Default command ───────────────────────────────────────────────
# Override khi chạy: docker compose run generate <command>
CMD ["python", "-m", "src.data.generate_data", "--all"]
