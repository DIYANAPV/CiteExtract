FROM python:3.11-slim

WORKDIR /app

# curl is kept in the final image for HEALTHCHECK; gcc/g++ are build-only
# and stripped after pip install to keep the runtime image lean.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake models so first-request latency doesn't include a download.
# - spaCy en_core_web_sm (~12 MB) for sentence splitting
# - sentence-transformers all-MiniLM-L6-v2 (~80 MB) for dense retrieval
RUN python -m spacy download en_core_web_sm && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Strip build-only packages after compilation to shrink the image
RUN apt-get purge -y gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/cache

# Gradio default port
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "app.py"]
