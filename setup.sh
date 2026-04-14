#!/bin/bash
# =============================================================================
# ImmunoBiology RAG — One-Click Setup for AutoDL A100
# =============================================================================
# Run this after uploading/cloning the project to AutoDL:
#   cd /root/autodl-tmp/Immunology_RAG
#   bash setup.sh
#
# What it does:
#   1. Install core Python dependencies
#   2. Clone and install LLaMA-Factory (for LLM fine-tuning)
#   3. Clone and install RAG-Retrieval (for reranker fine-tuning)
#   4. Download NLTK data
#   5. Start MongoDB
#   6. Create data directories
# =============================================================================

set -e          # Exit on most errors
set -o pipefail # Catch errors in piped commands

echo "============================================="
echo " ImmunoBiology RAG — Setup Script"
echo "============================================="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# ------------------------------------------------------------------
# HuggingFace mirror (required for AutoDL in China)
# ------------------------------------------------------------------
# Persists across shell sessions; skip if you have direct HF access
if [ -z "$HF_ENDPOINT" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo "HF_ENDPOINT set to $HF_ENDPOINT"
    # Make permanent for future sessions
    grep -qxF 'export HF_ENDPOINT=https://hf-mirror.com' ~/.bashrc || \
        echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
    echo "(Added HF_ENDPOINT to ~/.bashrc for future sessions)"
fi

# ------------------------------------------------------------------
# Helper: clone from GitHub with automatic mirror fallback for AutoDL
# Usage: git_clone_with_fallback <repo-path> <dest-dir>
# ------------------------------------------------------------------
git_clone_with_fallback() {
    local REPO="$1"   # e.g. hiyouga/LLaMA-Factory
    local DEST="$2"   # e.g. LLaMA-Factory

    if git clone "https://github.com/${REPO}.git" "$DEST" 2>/dev/null; then
        echo "  Cloned via github.com"
    else
        echo "  github.com failed — trying gitclone.com mirror..."
        if git clone "https://gitclone.com/github.com/${REPO}.git" "$DEST" 2>/dev/null; then
            echo "  Cloned via gitclone.com"
        else
            echo "  gitclone.com failed — trying ghfast.top mirror..."
            git clone "https://ghfast.top/https://github.com/${REPO}.git" "$DEST" || {
                echo "ERROR: All clone attempts failed for ${REPO}."
                echo "       Try manually: git clone https://github.com/${REPO}.git ${DEST}"
                exit 1
            }
        fi
    fi
}

# ------------------------------------------------------------------
# 1. Core dependencies
# ------------------------------------------------------------------
echo ""
echo "[1/8] Installing core dependencies..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "[1/8b] Installing vLLM (for LLM serving)..."
pip install vllm>=0.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "NOTE: Flash Attention 2 is OPTIONAL and disabled by default (flash_attn: false in config.yaml)."
echo "      It is not installed here. To enable it (A100-40GB + tight VRAM only):"
echo "        pip install flash-attn --no-build-isolation"
echo "        Then set  training.llm_sft.flash_attn: true  in config.yaml"

# ------------------------------------------------------------------
# 2. LLaMA-Factory
# ------------------------------------------------------------------
echo ""
echo "[2/8] Setting up LLaMA-Factory..."
if [ -d "LLaMA-Factory" ]; then
    echo "LLaMA-Factory already exists, skipping clone."
else
    git_clone_with_fallback "hiyouga/LLaMA-Factory" "LLaMA-Factory"
fi
cd LLaMA-Factory
# --ignore-requires-python: LLaMA-Factory >=0.9 requires Python >=3.11.
# Safe to pass on 3.11+; also acts as a guard if somehow run on 3.10.
pip install -e ".[torch,metrics]" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --ignore-requires-python
cd "$PROJECT_DIR"
echo "LLaMA-Factory installed."

# ------------------------------------------------------------------
# 3. RAG-Retrieval
# ------------------------------------------------------------------
echo ""
echo "[3/8] Setting up RAG-Retrieval..."
if [ -d "RAG-Retrieval" ]; then
    echo "RAG-Retrieval already exists, skipping clone."
else
    git_clone_with_fallback "NLPJCL/RAG-Retrieval" "RAG-Retrieval"
fi
cd RAG-Retrieval
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd "$PROJECT_DIR"
echo "RAG-Retrieval installed."

# ------------------------------------------------------------------
# 4. NLTK data
# ------------------------------------------------------------------
echo ""
echo "[4/8] Installing system dependencies for OCR (required for scanned PDFs)..."
# tesseract-ocr is needed by pytesseract to extract text from image-based PDFs
apt-get install -y tesseract-ocr tesseract-ocr-eng 2>/dev/null || \
    echo "  WARNING: Could not install tesseract-ocr — scanned PDFs will not be parsed."
echo "  Tesseract: $(tesseract --version 2>&1 | head -1)"
echo ""
echo "[5/8] Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
print('NLTK data downloaded.')
"

# ------------------------------------------------------------------
# 5. Pre-download lightweight models (small — required before first run)
# ------------------------------------------------------------------
echo ""
echo "[6/8] Pre-downloading lightweight models (via HF mirror)..."
mkdir -p models

# Semantic chunking model (~90 MB) — downloaded to local path to avoid network calls at service startup
if [ ! -f "models/all-MiniLM-L6-v2/config.json" ]; then
    echo "  Downloading sentence-transformers/all-MiniLM-L6-v2 ..."
    huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
        --local-dir models/all-MiniLM-L6-v2
    echo "  all-MiniLM-L6-v2 downloaded."
else
    echo "  all-MiniLM-L6-v2 already present, skipping."
fi

echo ""
echo "  NOTE: Large models (Qwen3-8B ~16 GB, BGE-M3, BGE-reranker) are NOT downloaded"
echo "  here to avoid long blocking. Download them manually before use:"
echo "    huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B"
echo "    huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3"
echo "    huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3"

# ------------------------------------------------------------------
# 6. MongoDB
# ------------------------------------------------------------------
echo ""
echo "[7/8] Starting MongoDB..."
if command -v mongod &> /dev/null; then
    pkill mongod 2>/dev/null || true; sleep 1
    mkdir -p /data/db && chmod 777 /data/db
    rm -f /data/db/mongod.lock          # clear stale lock from any previous crash
    mongod --fork --logpath /var/log/mongodb.log --dbpath /data/db 2>/dev/null || echo "MongoDB failed to start — run: mongod --dbpath /data/db 2>&1 | head -20"
    echo "MongoDB started."
else
    echo "MongoDB not found. Installing mongodb-org from official repo (Ubuntu 22.04 Jammy)..."
    # Ubuntu 22.04 dropped the old 'mongodb' package; use the official MongoDB 7.0 repo
    apt-get install -y gnupg curl 2>/dev/null
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
        gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] \
        https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
        tee /etc/apt/sources.list.d/mongodb-org-7.0.list
    apt-get update -qq
    apt-get install -y mongodb-org
    mkdir -p /data/db
    pkill mongod 2>/dev/null || true; sleep 1
    rm -f /data/db/mongod.lock
    mongod --fork --logpath /var/log/mongodb.log --dbpath /data/db 2>/dev/null || true
    echo "MongoDB installed and started."
fi

# ------------------------------------------------------------------
# 6. Create data directories
# ------------------------------------------------------------------
echo ""
echo "[8/8] Creating data directories..."
mkdir -p data/raw data/processed data/train
mkdir -p outputs/diagnostics outputs/vectorstore outputs/system_eval
mkdir -p outputs/models/reranker_finetuned outputs/models/llm_finetuned
mkdir -p outputs/reranker_eval outputs/llm_eval
echo "Directories created."

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "============================================="
echo " Setup Complete!"
echo "============================================="
echo ""
echo "Project structure:"
echo "  $PROJECT_DIR/"
echo "  ├── LLaMA-Factory/    (installed)"
echo "  ├── RAG-Retrieval/     (installed)"
echo "  ├── src/               (pipeline code)"
echo "  ├── train/             (training scripts)"
echo "  ├── data/raw/          (put PDFs here)"
echo "  └── ..."
echo ""
echo "Next steps:"
echo "  1. Upload PDFs to: $PROJECT_DIR/data/raw/"
echo ""
echo "  2. Download large models (BEFORE starting services):"
echo "       # HF mirror is already set (HF_ENDPOINT=https://hf-mirror.com)"
echo "       huggingface-cli download Qwen/Qwen3-8B          --local-dir models/Qwen3-8B"
echo "       huggingface-cli download BAAI/bge-m3             --local-dir models/bge-m3"
echo "       huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3"
echo "       # all-MiniLM-L6-v2 was already downloaded by this script → models/all-MiniLM-L6-v2"
echo ""
echo "  3. Start services:"
echo "       # Semantic chunking (CPU-only — keeps full GPU free for vLLM)"
echo "       python -m uvicorn src.server.semantic_chunk:app --host 0.0.0.0 --port 6000 &"
echo "       sleep 8 && curl http://localhost:6000/health"
echo ""
echo "       # vLLM — absolute path + --served-model-name to match config.yaml"
echo "       python -m vllm.entrypoints.openai.api_server \\"
echo "           --model \$(pwd)/models/Qwen3-8B \\"
echo "           --served-model-name Qwen/Qwen3-8B \\"
echo "           --dtype bfloat16 --max-model-len 8192 \\"
echo "           --gpu-memory-utilization 0.85 \\"
echo "           --enable-prefix-caching --port 8000 &"
echo ""
echo "  4. Build index:  python build_index.py --test-retrieval"
echo "  5. Launch UI:    streamlit run app.py --server.port 6006 --server.address 0.0.0.0"
echo ""
