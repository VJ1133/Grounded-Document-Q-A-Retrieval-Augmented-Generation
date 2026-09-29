# Single image, two entrypoints (Streamlit app / FastAPI service) selected
# via CMD override in docker-compose.yml -- same code, same dependencies,
# no reason to build it twice.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # No Ollama inside the container, and a public deployment has no local
    # Ollama to reach anyway -- forced off here so it can't accidentally
    # ship enabled; docker-compose.yml does not override this.
    ALLOW_OLLAMA=false \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# sentence-transformers pulls in torch; plain `pip install` resolves the
# default CUDA build (pulls in gigabytes of nvidia-*/triton wheels) even
# though this image only ever runs CPU inference. Install the CPU-only
# wheel first so that requirement is already satisfied by the time the
# rest of requirements.txt is installed.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY api/ api/

# Pre-download the embedding and reranker models at build time, so the
# image works fully offline at runtime and the first real request isn't
# slowed down by a ~180MB download.
RUN python -c "from src.embeddings import get_embedding_model; from src.reranker import get_reranker; get_embedding_model(); get_reranker()"

# Actual command is set per-service in docker-compose.yml; this default lets
# the image still do something sensible if run standalone.
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
