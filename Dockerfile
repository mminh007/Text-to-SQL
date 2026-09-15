# ─────────────────────────────────────────────────────────────────
# Dockerfile — text-to-sql-finetune
# Dùng cho: sinh data (--generate, --all, --dedup-only, --split)
# KHÔNG dùng --validate-sql trong Docker (chạy trực tiếp trên Windows)
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="text-to-sql-finetune"
LABEL description="Text-to-SQL dataset generator using Groq LLM"

# Cài system deps tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements trước để tận dụng Docker layer cache
COPY requirements.txt .

# Bước 1: Cài torch CPU-only từ PyTorch index riêng
# (tránh pip tự kéo bản CUDA mặc định ~1.1GB trên Linux)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Bước 2: Cài các package còn lại
# sentence-transformers sẽ reuse torch CPU đã cài ở trên
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        sentence-transformers \
        scikit-learn \
        numpy

# Copy source code
COPY generate_data.py .
COPY validate.py .

# Tạo thư mục output (sẽ được mount từ host qua volume)
RUN mkdir -p data

# Mặc định: chạy toàn bộ pipeline
CMD ["python", "generate_data.py", "--all"]
