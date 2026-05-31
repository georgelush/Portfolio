FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch first — saves ~1 GB vs the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the first request is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Download CV so the email agent can attach it (binary not stored in HF git)
RUN mkdir -p images && \
    wget -q -O images/RusuGeorgeCV.pdf \
    "https://raw.githubusercontent.com/georgelush/Portfolio/main/images/RusuGeorgeCV.pdf" || true

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
