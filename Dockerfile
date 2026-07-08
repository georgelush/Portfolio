FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as UID 1000 (non-root). Build as that user too,
# so the pip packages and the baked model cache are readable at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/torch/sentence_transformers

WORKDIR /home/user/app

COPY --chown=user:user requirements.txt .

# CPU-only torch first — saves ~1 GB vs the default CUDA build
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding model into the image so the first request is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user:user . .

# Download CV so the email agent can attach it (binary not stored in HF git)
RUN python -c "\
import urllib.request, os; \
os.makedirs('images', exist_ok=True); \
urllib.request.urlretrieve(\
'https://raw.githubusercontent.com/georgelush/Portfolio/main/images/RusuGeorgeCV.pdf', \
'images/RusuGeorgeCV.pdf'); \
print('CV downloaded:', os.path.getsize('images/RusuGeorgeCV.pdf'), 'bytes')"

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
