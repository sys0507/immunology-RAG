# =============================================================================
# ImmunoBiology RAG — Semantic Chunking FastAPI Service
# =============================================================================
# Adapted from Tesla RAG: src/server/semantic_chunk.py
# Changes:
#   - Replaced M3E-small (Chinese) with all-MiniLM-L6-v2 (English, lightweight)
#   - Adjusted _min_doc_size from 256 → 100 (English words are shorter)
#   - English comments throughout
#   - Added "##" markdown heading split (common in textbooks) alongside "###"
#
# Service: FastAPI on port 6000
# Must be started BEFORE running build_index.py or chunker.py
#
# Prerequisites (AutoDL / China):
#   1. Pre-download embedding model to avoid network calls at startup:
#        huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
#            --local-dir models/all-MiniLM-L6-v2
#      (setup.sh does this automatically)
#   2. Set HF mirror if auto-download is ever needed:
#        export HF_ENDPOINT=https://hf-mirror.com
#
# Start commands (both work identically — model loaded via lifespan):
#   python -m uvicorn src.server.semantic_chunk:app --host 0.0.0.0 --port 6000
#   python src/server/semantic_chunk.py
#
# API endpoints:
#   GET  http://localhost:6000/health              → {"status":"ok"}
#   POST http://localhost:6000/v1/semantic-chunks
#        Body:     {"sentences": "<raw text>", "group_size": 15}
#        Response: {"chunks": ["semantic group 1", "semantic group 2", ...]}

import gc
import re
import math
import uvicorn
import torch
import pandas as pd
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Minimum document size (chars) to bother clustering; short texts returned as-is
_min_doc_size = 100       # English: shorter than Tesla's 256 (Chinese chars are dense)
# Minimum cluster size; tiny fragments get merged with neighbors
_min_chunk_size = 50      # chars

# ---------------------------------------------------------------------------
# App lifecycle: load model on startup, clean GPU on shutdown
# ---------------------------------------------------------------------------
# Model name: resolved from config.yaml when available, else sensible default
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
try:
    import sys, os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src import constant as _c
    _MODEL_NAME = _c.semantic_chunk_model_path
except Exception:
    pass  # Fall back to default above


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding model on startup; release memory on shutdown.

    Device choice: always CPU.
    all-MiniLM-L6-v2 is a 90 MB model — CPU inference is fast enough (<20 ms
    per batch) and keeps the entire GPU free for vLLM (Qwen3-8B needs ~34 GB).
    Putting this model on CUDA wastes ~35 GB of VRAM and causes vLLM to fail
    with "Free memory less than desired GPU memory utilization".
    """
    global embedding_model
    device = "cpu"   # intentional — keep GPU fully available for vLLM
    print(f"[SemanticChunk] Loading model '{_MODEL_NAME}' on {device} (CPU-only) …")
    embedding_model = SentenceTransformer(_MODEL_NAME, device=device)
    print(f"[SemanticChunk] Model ready. Serving on port 6000.")
    yield
    # Shutdown — free resources
    del embedding_model
    gc.collect()


app = FastAPI(
    title="ImmunoBiology Semantic Chunking Service",
    description="Splits English academic text into semantically coherent groups.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global embedding model (populated during lifespan startup)
embedding_model: Optional[SentenceTransformer] = None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class SemanticRequest(BaseModel):
    sentences: str
    group_size: Optional[int] = 15   # Default 15 for English (Tesla: 10 for Chinese)


class ChunkResponse(BaseModel):
    chunks: List[str]


# ---------------------------------------------------------------------------
# Health check endpoint  (GET /health)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Simple liveness probe.
    Returns 200 + {"status": "ok"} once the model has loaded.
    Returns 503 if the model is still loading (should not normally happen
    because lifespan blocks until load completes).
    """
    if embedding_model is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "model": _MODEL_NAME}


# ---------------------------------------------------------------------------
# Core endpoint
# ---------------------------------------------------------------------------
@app.post("/v1/semantic-chunks", response_model=ChunkResponse)
async def create_semantic_chunks(request: SemanticRequest):
    """
    Split input text into semantically coherent groups.

    Algorithm:
    1. If text is short enough, return as-is.
    2. Try splitting on markdown headings ("###" or "##").
    3. Fall back to paragraph splits ("\n\n").
    4. Embed paragraphs and cluster with AgglomerativeClustering (cosine, avg linkage).
    5. Merge tiny fragments (< _min_chunk_size chars) with adjacent groups.

    This is language-agnostic; works for English academic text by design.
    """
    global embedding_model

    if request.group_size < 1:
        raise HTTPException(status_code=400, detail="group_size must be >= 1")

    text = request.sentences

    # Step 1: Short text — return as single chunk
    if len(text) <= _min_doc_size:
        return ChunkResponse(chunks=[text])

    # Step 2: Split on markdown headings (textbook sections, "### Heading" or "## Heading")
    split_docs = re.split(r"(#{2,3}\s)", text)
    split_docs = [k for k in split_docs if k.strip()]
    if len(split_docs) > 1:
        # Reconstruct heading + content pairs
        result = []
        i = 0
        if not split_docs[0].startswith("#"):
            result.append(split_docs[0])
            i = 1
        while i < len(split_docs) - 1:
            result.append(split_docs[i] + split_docs[i + 1])
            i += 2
        if i < len(split_docs):
            result.append(split_docs[i])
        return ChunkResponse(chunks=[r for r in result if r.strip()])

    # Step 3: Paragraph split ("\n\n")
    split_docs = [s.strip() for s in text.split("\n\n") if s.strip()]
    if not split_docs:
        return ChunkResponse(chunks=[text])

    if len(split_docs) <= request.group_size:
        return ChunkResponse(chunks=split_docs)

    # Step 4: Agglomerative clustering on paragraph embeddings
    n_clusters = max(1, math.ceil(len(split_docs) / request.group_size))

    try:
        embeddings = embedding_model.encode(split_docs, show_progress_bar=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Clustering failed: {e}")

    df = pd.DataFrame({"sentence": split_docs, "label": labels})
    grouped = (
        df.groupby("label", sort=True)["sentence"]
        .agg(lambda x: " ".join(x))
        .to_dict()
    )
    docs = list(grouped.values())

    # Step 5: Merge tiny fragments with neighbors
    merged_docs = []
    idx = 0
    while idx < len(docs):
        cur = docs[idx]
        skip = 1
        for next_idx in range(idx + 1, len(docs)):
            if len(docs[next_idx]) < _min_chunk_size:
                cur += " " + docs[next_idx]
                skip += 1
            else:
                break
        idx += skip
        merged_docs.append(cur)

    return ChunkResponse(chunks=merged_docs)


# ---------------------------------------------------------------------------
# Entry point when run directly: python src/server/semantic_chunk.py
# (Model is loaded via lifespan, not here — same code path as uvicorn CLI)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "src.server.semantic_chunk:app",
        host="0.0.0.0",
        port=6000,
        workers=1,
    )
