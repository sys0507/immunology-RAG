# ImmunoBiology RAG — AutoDL A100 Runbook

Complete step-by-step guide for deploying, training, and running the ImmunoBiology RAG
system on an AutoDL A100 GPU instance. All commands have been verified end-to-end.

---

## Table of Contents

1. [Instance Setup](#1-instance-setup)
2. [Environment Setup](#2-environment-setup)
3. [Data Upload](#3-data-upload)
4. [Execution Order — Training Path (Start from Scratch)](#4-execution-order--training-path-start-from-scratch)
5. [Execution Order — Inference Path (After Fine-Tuning)](#5-execution-order--inference-path-after-fine-tuning)
6. [LoRA Merge: Making Fine-Tuned Model Ready for vLLM](#6-lora-merge-making-fine-tuned-model-ready-for-vllm)
7. [Incremental PDF Addition](#7-incremental-pdf-addition)
8. [A100 Optimization Tips](#8-a100-optimization-tips)
9. [Port Mapping for Streamlit](#9-port-mapping-for-streamlit)
10. [Common Errors and Solutions](#10-common-errors-and-solutions)
11. [Quick Reference: Service Ports](#11-quick-reference-service-ports)

---

## 1. Instance Setup

### Image Selection

- **Recommended image:** PyTorch 2.1+ with CUDA 12.1
- **GPU:** A100-PCIE-40GB (minimum) or A100-SXM4-80GB (recommended)
- **Disk:** At least 120 GB (models ~50 GB + index + training data + checkpoints)

### Instance Creation

1. Log in to [AutoDL](https://www.autodl.com)
2. Select region (prefer nearby for lower latency)
3. Choose A100 GPU instance (40 GB or 80 GB)
4. Select PyTorch 2.1 + CUDA 12.1 image
5. Set disk space to 120 GB+
6. Create and start instance

### SSH Access

```bash
# AutoDL provides the exact SSH command in the instance dashboard
ssh -p <port> root@<host>
```

---

## 2. Environment Setup

### Step 2.1 — Clone Project

AutoDL persistent storage is at `/root/autodl-tmp/`. **Always put your project here**
so data and models survive instance restarts (unlike `/root/` which may reset).

> **GitHub is blocked on AutoDL** — the same network restriction as `huggingface.co`.
> Direct `https://github.com` connections time out after ~2 minutes.
> Use the `gitclone.com` mirror instead (see below).

**On AutoDL (China) — use the gitclone.com mirror:**
```bash
cd /root/autodl-tmp
git clone https://gitclone.com/github.com/<your-org>/Immunology_RAG.git Immunology_RAG
cd Immunology_RAG
export PROJECT_ROOT=$(pwd)
```

**On other cloud environments (AWS, GCP, Azure, local) — standard clone:**
```bash
cd /root/autodl-tmp    # or wherever you keep projects
git clone https://github.com/<your-org>/Immunology_RAG.git Immunology_RAG
cd Immunology_RAG
export PROJECT_ROOT=$(pwd)
```

### Step 2.2 — Create Conda Environment

> **Where should the env live?**
>
> | Disk | Path | Typical size | Recommended for |
> |------|------|-------------|-----------------|
> | System disk | `/root/` | ~30 GB total | OS, apt packages, miniconda base |
> | Data disk | `/root/autodl-tmp/` | 80+ GB (user-set) | Project code, models, training data |
>
> The conda env itself is **~6–8 GB** (PyTorch, transformers, etc.).
> Choose **Option A** (system disk, simpler) if your system disk has room.
> Choose **Option B** (data disk) if you want to keep the system disk lean or
> are on a small instance where system disk space is tight.

**Option A — System disk (recommended, simpler):**

```bash
# Initialize conda for bash if not already done (run once per instance)
conda init bash
source ~/.bashrc

# Create named env on system disk (/root/miniconda3/envs/immunorag/)
conda create -n immunorag python=3.11 -y
conda activate immunorag

# Auto-activate on every login
echo 'conda activate immunorag' >> ~/.bashrc
source ~/.bashrc
```

**Option B — Data disk (keeps system disk lean):**

```bash
# Initialize conda for bash if not already done (run once per instance)
conda init bash
source ~/.bashrc

# Create path-based env on data disk
mkdir -p /root/autodl-tmp/envs
conda create -p /root/autodl-tmp/envs/immunorag python=3.11 -y
conda activate /root/autodl-tmp/envs/immunorag

# Auto-activate on every login (use full path, not env name)
echo 'conda activate /root/autodl-tmp/envs/immunorag' >> ~/.bashrc
source ~/.bashrc
```

> **Verify the env is active** — your prompt should show `(immunorag)` or
> `(/root/autodl-tmp/envs/immunorag)` at the start of every line:
> ```bash
> python --version   # Expected: Python 3.11.x
> which python       # Expected: path inside the immunorag env
> ```

### Step 2.3 — Set HuggingFace Mirror (Required for AutoDL in China)

AutoDL instances **cannot reach `huggingface.co` directly**. Set the mirror **before**
downloading any model or starting any service that auto-downloads weights.
Forgetting this causes `MaxRetryError: Connection to huggingface.co timed out`.

```bash
# Set for current session AND make permanent
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# Verify
echo $HF_ENDPOINT  # should print: https://hf-mirror.com
```

### Step 2.4 — Install System Packages

```bash
# OCR support (required if your PDFs are scanned / image-based)
apt-get install -y tesseract-ocr tesseract-ocr-eng

# MongoDB 7.0 (Ubuntu 22.04 Jammy dropped old mongodb package)
apt-get install -y gnupg curl
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
    gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] \
    https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
    tee /etc/apt/sources.list.d/mongodb-org-7.0.list
apt-get update && apt-get install -y mongodb-org
```

### Step 2.5 — Install Python Dependencies

```bash
cd /root/autodl-tmp/Immunology_RAG

# Remove packages unavailable on newer Python versions
sed -i '/hashlib2/d' requirements.txt
sed -i '/pickle5/d' requirements.txt

# Install all project dependencies
pip install -r requirements.txt

# vLLM for LLM serving (not in requirements.txt to keep it optional)
pip install vllm>=0.4.0

# NLTK tokenizer data (required for BM25 English tokenization)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Step 2.6 — Install LLaMA-Factory & RAG-Retrieval (Required for Fine-Tuning Only)

> **Skip this step** if you are only running inference (not training). These two frameworks
> are only needed for Steps 4.7 (reranker fine-tuning) and 4.8 (LLM SFT).

> **GitHub is blocked on AutoDL.** Direct connections to `github.com` time out after
> ~2 minutes. Use the `gitclone.com` mirror (Option A below). On other cloud environments
> or local machines, use the standard GitHub URLs (Option B).

**Option A — AutoDL (China), use gitclone.com mirror:**

```bash
cd /root/autodl-tmp/Immunology_RAG

# --- LLaMA-Factory ---
git clone https://gitclone.com/github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# --- RAG-Retrieval ---
git clone https://gitclone.com/github.com/NLPJCL/RAG-Retrieval.git RAG-Retrieval
cd RAG-Retrieval
pip install -e .
cd ..
```

**Option B — Other cloud environments or local (standard GitHub URLs):**

```bash
cd /root/autodl-tmp/Immunology_RAG   # adjust path as needed

# --- LLaMA-Factory ---
git clone https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

# --- RAG-Retrieval ---
git clone https://github.com/NLPJCL/RAG-Retrieval.git RAG-Retrieval
cd RAG-Retrieval
pip install -e .
cd ..
```

**Verify both installed correctly (same for both options):**

```bash
python -c "import llamafactory; print('LLaMA-Factory OK')"
python -c "import rag_retrieval; print('RAG-Retrieval OK')"
```

After installation, your directory structure should look like:

```
/root/autodl-tmp/Immunology_RAG/
├── LLaMA-Factory/          <-- cloned + installed (only needed for fine-tuning)
├── RAG-Retrieval/          <-- cloned + installed (only needed for fine-tuning)
├── src/                    <-- RAG pipeline source code
├── train/                  <-- training scripts
├── pages/                  <-- Streamlit UI pages
├── config.yaml
├── build_index.py
├── evaluate.py
└── app.py
```

### Step 2.7 — Download Models

> **Download all models before starting any service.** If a model folder is missing
> when a service starts (vLLM, semantic chunker), it will try to download from HuggingFace
> at startup — which fails on AutoDL without `HF_ENDPOINT` set, and takes extra time.

```bash
cd /root/autodl-tmp/Immunology_RAG
mkdir -p models

# ── 1. Semantic chunking model (~90 MB, CPU-only inference) ──────────────────
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir models/all-MiniLM-L6-v2
echo "✓ all-MiniLM-L6-v2 ready"

# ── 2. BGE-M3 embedding model (~2.5 GB) ──────────────────────────────────────
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
echo "✓ BGE-M3 ready"

# ── 3. BGE Reranker v2-M3 (~1.1 GB) ─────────────────────────────────────────
huggingface-cli download BAAI/bge-reranker-v2-m3 \
    --local-dir models/bge-reranker-v2-m3
echo "✓ BGE Reranker ready"

# ── 4. Qwen3-8B (~16 GB — no HuggingFace login required) ────────────────────
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
echo "✓ Qwen3-8B ready"

# Verify all downloads
ls models/
# Expected: all-MiniLM-L6-v2/  bge-m3/  bge-reranker-v2-m3/  Qwen3-8B/

ls models/Qwen3-8B/
# Expected: config.json  model-*.safetensors  tokenizer.json  ...
```

---

## 3. Data Upload

### Option A: AutoDL File Manager (easiest)

1. Open AutoDL instance dashboard → click "File" tab
2. Navigate to `/root/autodl-tmp/Immunology_RAG/data/raw/`
3. Drag and drop your PDFs

### Option B: SCP from Local Machine

```bash
scp -P <port> /path/to/JanewaysImmunobiologyBiology10thEdition.pdf \
    root@<host>:/root/autodl-tmp/Immunology_RAG/data/raw/
```

### Option C: wget (if PDF is publicly hosted)

```bash
cd /root/autodl-tmp/Immunology_RAG/data/raw/
wget "https://example.com/immunology_textbook.pdf"
```

Verify upload:
```bash
ls /root/autodl-tmp/Immunology_RAG/data/raw/
```

---

## 4. Execution Order — Training Path (Start from Scratch)

This section covers the **complete workflow** from a fresh instance with no trained models:
parse PDFs → build index → generate training data → fine-tune reranker → fine-tune LLM →
merge LoRA → evaluate. Follow steps strictly in order.

### ⚠️ GPU Memory Conflict: vLLM vs Training

> **Critical rule:** vLLM occupies ~33–35 GB of the A100's 40 GB VRAM when running.
> Training (reranker and LLM SFT) also requires the full GPU.
> **You cannot run vLLM and training simultaneously on a 40 GB A100.**
>
> The workflow below reflects this:
> - vLLM is started for data generation (Step 4.4)
> - vLLM is **killed** before fine-tuning (Step 4.5)
> - After fine-tuning, the LoRA adapter is **merged** into the base model (Step 4.7)
> - vLLM is **restarted** with the merged model for evaluation and inference (Step 4.8)

---

### Step 4.1 — Start MongoDB

MongoDB must be running before building the index (chunker.py writes parent/child chunks
to MongoDB). Start it first and leave it running for the entire session.

```bash
cd /root/autodl-tmp/Immunology_RAG

pkill mongod 2>/dev/null; sleep 2          # kill any stale process from previous session
mkdir -p /data/db && chmod 777 /data/db    # ensure directory exists with correct permissions
rm -f /data/db/mongod.lock                 # remove stale lock file from any previous crash
mongod --fork --logpath /var/log/mongodb.log --dbpath /data/db

# Verify (~2 seconds to start)
sleep 2 && mongosh --eval "db.runCommand({ping: 1})"
# Expected output: { ok: 1 }
```

### Step 4.2 — Start Semantic Chunking Service

This FastAPI service (port 6000) is used by `build_index.py` to detect semantic
paragraph boundaries. It must be running before index building.

**It runs on CPU only** — this is intentional to keep the entire GPU free for vLLM.

```bash
cd /root/autodl-tmp/Immunology_RAG
python -m uvicorn src.server.semantic_chunk:app --host 0.0.0.0 --port 6000 &

# Wait for model to load from disk (~5-10 seconds)
sleep 12
curl http://localhost:6000/health
# Expected: {"status":"ok","model":"...all-MiniLM-L6-v2"}
```

**Expected startup log:**
```
[SemanticChunk] Loading model '...all-MiniLM-L6-v2' on cpu (CPU-only) …
[SemanticChunk] Model ready. Serving on port 6000.
INFO:     Application startup complete.
```

### Step 4.3 — Build the Knowledge Index (~10-30 min, scales with PDF size)

This step does everything needed to go from raw PDFs to a searchable knowledge base:
PDF parsing → semantic chunking → MongoDB storage → ChromaDB indexing → BM25 indexing.

```bash
cd /root/autodl-tmp/Immunology_RAG

# Optional but recommended: inspect PDF layout first (~1 min)
# This samples pages 1/50/100/200 and writes a report so you can verify
# header/footer cropping is correct before the full parse.
python build_index.py --inspect
cat outputs/diagnostics/pdf_layout_report.txt  # review output

# Full index build + retrieval test
# --test-retrieval runs a test query after indexing to verify everything works
python build_index.py --test-retrieval
```

Expected output when complete:
```
[Build] Parsing PDF: JanewaysImmunobiologyBiology10thEdition.pdf ...
[Build] Extracted 1200 pages, 847 figures
[Build] Semantic chunking 1200 page records ...
[Build] Inserted 4800 parent + 9600 child chunks into MongoDB
[Build] Embedding 9600 child docs with BGE-M3 ...
[Build] Chroma collection immunology_chunks: 9600 docs
[Build] BM25 index saved: outputs/vectorstore/bm25_index.pkl
--- Retrieval Test ---
Query: "What activates T cells?"
BM25 Top-3: [relevant snippets ...]
Dense Top-3: [relevant snippets ...]
[Build] Index build complete.
```

### Step 4.4 — Start vLLM Server (for Training Data Generation)

vLLM is the process that runs the Qwen3-8B language model and answers API requests.
It is needed by `build_train_data.py` to generate QA pairs from the indexed text.

> **Beginner note:** The `&` at the end of the command means "run in the background".
> vLLM keeps running while you continue working in the same terminal. You will see log
> lines printed occasionally — that is normal. Your prompt returns immediately.

```bash
cd /root/autodl-tmp/Immunology_RAG

# --served-model-name must match config.yaml: models.llm = "Qwen/Qwen3-8B"
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --port 8000 &

# Wait for the model to finish loading into GPU memory (~30-60 seconds)
sleep 60

# Confirm vLLM is ready — you should see the model listed
curl http://localhost:8000/v1/models
# Expected: {"data":[{"id":"Qwen/Qwen3-8B",...}]}
```

> **Check:** The model ID in the curl response must be `"Qwen/Qwen3-8B"` (matching
> `config.yaml`). If it shows a local path instead, add `--served-model-name Qwen/Qwen3-8B`.

**How to check if vLLM is already running:**
```bash
ps aux | grep vllm
# If running, you will see a line containing "vllm.entrypoints.openai.api_server"
# If not running, you will see only the grep command itself
```

**How to kill vLLM (you will need this in Step 4.6):**
```bash
pkill -f vllm
# pkill -f "pattern" finds every process whose command line contains that pattern
# and sends it a termination signal.
# It is safe to run even if vLLM is not currently running (no error is shown).

# Wait a few seconds for the process to fully exit and release GPU memory
sleep 5

# Confirm the GPU is now free (Memory-Usage should drop back to ~2000 MiB)
nvidia-smi
```

### Step 4.5 — Generate Training Data (~3.5 h for Stage 1; Stage 2 now < 5 min)

The script runs in **four checkpointed stages**. Each stage saves its output
immediately when done. If the connection drops or the job is interrupted,
simply re-run the same command — finished stages are detected and skipped.

| Stage | Output file | Est. time | Needs vLLM? |
|-------|-------------|-----------|-------------|
| 1 — QA generation | `data/train/qa_pairs_cache.jsonl` | ~3.4 h (actual: 202 min) | ✅ Yes |
| 2 — Reranker triplets | `data/train/reranker_train.jsonl` | **< 5 min** (BM25 only) | ❌ No |
| 3 — SFT formatting | `data/train/sft_train.jsonl` | < 1 min | ❌ No |
| 4 — Eval QA set | `data/train/eval_qa.jsonl` | < 1 min | ❌ No |

> **Stage 2 was rewritten** to use BM25 retrieval for hard negatives instead of LLM generation.
> The original LLM-generated negatives were hallucinated text — trivially distinguishable from
> real passages, causing the reranker to collapse to loss=0.0 in ~100 steps. BM25 negatives
> are actual textbook passages, making the task genuinely hard.
> Train/eval split is now stratified by chapter (no chapter in both sets).
>
> **Total for a full textbook (~16,900 QA pairs):** ~3.5 h (Stage 1 only; Stage 2 is now fast).
> Use `--limit 20` for a quick smoke test before committing to the full run.
> Use `--force` to delete all checkpoints and restart from scratch.

```bash
cd /root/autodl-tmp/Immunology_RAG
mkdir -p logs

# Run in background — safe to close terminal / disconnect SSH
# If the job is interrupted, re-run this exact command to resume from the last checkpoint
nohup python -u -m train.build_train_data --workers 4 \
    > logs/build_train_data.log 2>&1 &
echo "[+] PID $! — Job started. Follow progress:"
echo "    tail -f logs/build_train_data.log"
```

Monitor progress (each stage prints its own progress):
```bash
tail -f logs/build_train_data.log

# Stage 1 progress:  "[Stage 1] Progress: 500/4247 chunks, ... (~45 min remaining)"
# Stage 1 complete:  "[Stage 1] ✓ QA pairs saved to checkpoint: data/train/qa_pairs_cache.jsonl"
# Stage 2 progress:  "[Reranker] Progress: 500/16896 pairs, ... (~60 min remaining)"
# Stage 2 complete:  "[Reranker] Stage 2 done in X.X min."
# All done:          "[TrainData] Total wall time: X.X min"
```

If reconnecting after a disconnection:
```bash
# Check how far it got — look for the last checkpoint saved
grep "✓" logs/build_train_data.log

# If the process died, simply re-run — it will skip completed stages automatically
nohup python -u -m train.build_train_data --workers 4 \
    > logs/build_train_data.log 2>&1 &
```

Verify output files once all stages complete:
```bash
grep "wall time" logs/build_train_data.log   # confirms full completion
ls -lh data/train/                            # check all 5 files exist and have size > 0
wc -l data/train/reranker_train.jsonl         # actual: 15,242 triplets
wc -l data/train/sft_train.jsonl              # actual: 16,936 QA pairs
wc -l data/train/eval_qa.jsonl                # actual: 116 eval pairs
head -n 2 data/train/eval_qa.jsonl            # check QA format
```

### Step 4.6 — ⚠️ Kill vLLM Before Fine-Tuning

> **Why?** vLLM keeps the Qwen3-8B model loaded in GPU memory at all times — it
> pre-allocates about 33–35 GB out of the 40 GB available. Fine-tuning the reranker
> and LLM SFT also need the full GPU. If you try to train while vLLM is still running,
> you will immediately get `RuntimeError: CUDA out of memory` and training will crash.
> You **must** free the GPU first.

```bash
# Kill the vLLM background process and all its child workers
pkill -f vllm

# Wait for the process to fully exit and release all GPU memory
# (GPU memory is not freed instantly — the OS needs a moment to reclaim it)
sleep 5

# Verify the GPU is now free
nvidia-smi
```

After running `nvidia-smi` you should see:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI ...       Driver Version: ...       CUDA Version: ...                        |
|-------------------------------+----------------------+----------------------+
| GPU  Name        ...          | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-PCIE-40GB   |  ...                 |                  Off |
|  0%   35C    P0    60W / 400W |   2000MiB / 40960MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
```

The key line is `2000MiB / 40960MiB` — showing only ~2 GB used (system overhead), not 35 GB.
**Do not proceed to training until you see this.**

> **If `pkill -f vllm` reports nothing and `nvidia-smi` still shows high memory usage:**
> a different GPU process may be running. Find and kill it:
> ```bash
> nvidia-smi          # note the PID in the "Processes" section at the bottom
> kill <PID>          # replace <PID> with the number you saw
> sleep 3 && nvidia-smi   # re-check
> ```

### Step 4.7 — Fine-Tune Reranker (~4.7 hours on A100-40GB)

Fine-tunes `BAAI/bge-reranker-v2-m3` on immunology-specific relevance triplets.

> **Checkpoint/resume:** A `training_complete.json` sentinel is written when training
> finishes. If the job is interrupted and re-run, it resumes automatically from the
> `best/` checkpoint. Use `--force` to start from scratch.

> ⚠️ **`TRANSFORMERS_OFFLINE=1` is required on AutoDL.**
> `config.yaml` uses the HuggingFace model ID (`BAAI/bge-reranker-v2-m3`), so
> `from_pretrained()` will try to contact `hf.co` — which is blocked on AutoDL.
> Without this flag the script silently hangs for hours before training even starts.
> The model must already be in `~/.cache/huggingface/hub/` from the initial download.

```bash
cd /root/autodl-tmp/Immunology_RAG
mkdir -p logs

# TRANSFORMERS_OFFLINE=1 — prevents from_pretrained() from contacting hf.co (blocked on AutoDL)
# Model is loaded from ~/.cache/huggingface/hub/ (already downloaded in setup)
TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker \
    > logs/train_reranker.log 2>&1 &
echo "[+] PID $! — Job started. Follow progress:"
echo "    tail -f logs/train_reranker.log"
```

Monitor progress:
```bash
tail -f logs/train_reranker.log
# Progress line: "[Reranker] Epoch 1/3 Step 50/5715 | ... | ETA: ~XX min"
# Finished when: "[Reranker] ✓ Sentinel written: ...training_complete.json"
```

Expected log output (based on actual run: 14,741 train triplets × 2 = 29,482 pairs):
```
[Reranker] Starting from base model: BAAI/bge-reranker-v2-m3
[Reranker] Training on: cuda
[Reranker] Train samples: 29482 | Eval samples: 4390
[Reranker] Total steps: 5528 | Warmup steps: 552
[Reranker] Epoch 1/3 Step 50/5528  | Train loss: 0.8082 | Eval loss: 0.7043 | NDCG@10: 0.8828 | MRR@10: 0.8412 | ETA: ~102 min
[Reranker] ✓ New best NDCG@10=0.8828 → saved to outputs/models/reranker_finetuned/best
...
[Reranker] Epoch 3/3 complete | Avg loss: 0.2834 | Elapsed: 94.0 min
[Reranker] Training complete. Best NDCG@10: 0.9637
[Reranker] Total training time: 283.1 min
[Reranker] ✓ Sentinel written: outputs/models/reranker_finetuned/training_complete.json
[Reranker] Training curves saved: outputs/reranker_eval/reranker_training_curves.png
[Reranker] Pre-trained model not found at BAAI/bge-reranker-v2-m3, skipping.
  Fine-tuned: {'eval_loss': 0.4484, 'ndcg@10': 0.9637, 'mrr@10': 0.9508}
[Reranker] Not enough models for comparison.
```

> **Actual training time: ~4.7 h** (283 min, 3 epochs on A100-40GB with BM25 negatives).
> First progress line appears at step 50 (~25 min in).
>
> The last 3 lines are **expected non-fatal warnings** — the fine-tuned model is saved correctly:
> - `Pre-trained model not found` — comparison.png skipped (cosmetic only; `best/` is saved ✅)
> - `Not enough models for comparison` — same cause; ignore
> - Tokenizer `incorrect regex pattern` — false positive (BGE uses XLM-RoBERTa not Mistral); harmless

If interrupted and re-run:
```bash
# Re-run with the same TRANSFORMERS_OFFLINE=1 flag — auto-resumes from best/ checkpoint
TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker \
    > logs/train_reranker.log 2>&1 &

# Or explicitly resume:
TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker --resume \
    > logs/train_reranker.log 2>&1 &

# Force retrain from scratch:
TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker --force \
    > logs/train_reranker.log 2>&1 &
```

Output files (check after job completes):
```bash
cat outputs/models/reranker_finetuned/training_complete.json  # sentinel — best NDCG@10, elapsed time
ls  outputs/models/reranker_finetuned/best/                   # best checkpoint (NDCG@10=0.9637)
ls  outputs/models/reranker_finetuned/final/                  # final epoch checkpoint
ls  outputs/reranker_eval/                                     # reranker_training_curves.png
# Note: comparison.png is NOT generated (pre-trained base model skipped due to HF network block)
```

> **If you see NaN loss:** The current code defaults to `fp16=False` and casts logits
> with `.float()` before loss — this is already fixed. If you have an older version,
> see the Common Errors section.

**After training — enable the fine-tuned reranker:**
```yaml
# config.yaml → models section
reranker_use_finetuned: true   # was false during training
```
Then re-run evaluation (Step 4.11). If end-to-end metrics are **worse** than the base model,
see Step 4.7b below — the negatives need to be regenerated with hybrid retrieval.

### Step 4.7b — Reranker v2: Regenerate with Hybrid Negatives (if end-to-end regresses)

**When to run this:** After Step 4.7, if `evaluate.py` shows Recall@1 / MRR@10 lower with
the fine-tuned reranker than with the base model. Root cause: Stage 2 data used BM25-only
negatives, but in production the pipeline uses hybrid (BM25 + dense) retrieval. The reranker
learned to distinguish BM25 candidates only and re-ranks hybrid candidates poorly.

**Fix:** regenerate Stage 2 with negatives drawn from the full hybrid pipeline, then retrain.

> **Prerequisite:** MongoDB and Chroma vectorstore must be running (Step 4.1 + Step 4.3 already done).
> vLLM does NOT need to be running. BGE-M3 loads automatically inside `build_train_data.py`.

```bash
cd /root/autodl-tmp/Immunology_RAG

# Step 1 — revert to base reranker while retraining
# In config.yaml: reranker_use_finetuned: false  (already the default)

# Step 2 — delete old BM25-only Stage 2 data (Stage 1 QA cache is preserved)
# ⚠️ Do NOT use --force — that deletes qa_pairs_cache.jsonl too (3+ h re-run)
rm data/train/reranker_train.jsonl data/train/reranker_test.jsonl

# Step 3 — regenerate Stage 2 with hybrid negatives (~10-20 min)
# BGE-M3 encoder loads automatically; watch for the mode line
python -u -m train.build_train_data
# Expected: "[Reranker] Negative mining mode: hybrid (BM25 + dense)"
# If you see "BM25-only" → Chroma is unavailable; check MongoDB is running
```

Expected Stage 2 output:
```
[Reranker] Building BM25 index from 2288 chunks...
[Reranker] Loading BGE-M3 encoder for hybrid negative mining...
[ChromaRetriever] Collection has 2288 chunks.
[Reranker] Dense retriever ready (2288 chunks).
[Reranker] Negative mining mode: hybrid (BM25 + dense)
[Reranker] Building triplets for ~16900 QA pairs...
[Reranker] Train: ~14700 triplets → data/train/reranker_train.jsonl
[Reranker] Test:  ~2100 triplets  → data/train/reranker_test.jsonl
```

```bash
# Step 4 — retrain reranker with hybrid negatives (~4.7 h)
TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker --force \
    > logs/train_reranker_v2.log 2>&1 &
tail -f logs/train_reranker_v2.log
# Actual NDCG@10: 0.9669 (still high — harder negatives but model still overfits/forgets)
```

```bash
# Step 5 — enable fine-tuned reranker and re-evaluate
# In config.yaml: reranker_use_finetuned: true

nohup python -u evaluate.py > logs/evaluate_v2.log 2>&1 &
tail -f logs/evaluate_v2.log
```

> ⚠️ **Note: if evaluate.py crashes with CUDA OOM during `[Eval] Running per-document-type breakdown`:**
> vLLM (process B) holds ~34.6 GB and the evaluator (process A, reranker + encoder) holds ~3 GB,
> leaving only ~296 MiB free. Fix: `del inputs; torch.cuda.empty_cache()` was added to
> `src/reranker/bge_m3_reranker.py` after line `scores = scores.detach().cpu()`.
> Alternatively: `PYTORCH_ALLOC_CONF=expandable_segments:True python -m evaluate`

**Actual results — all three fine-tuning attempts regressed vs base model:**

| Metric | Base model | v1 BM25-only neg | v2 hybrid neg | v3 LoRA (actual) | v4 target |
|--------|-----------|------------------|---------------|-------------------|-----------|
| Recall@1 | **0.559** | 0.426 ⬇️ | 0.426 ⬇️ | 0.360 ⬇️ | **> 0.559** |
| Recall@3 | **0.693** | — | 0.630 ⬇️ | — | — |
| Recall@5 | **0.751** | — | 0.701 ⬇️ | — | — |
| MRR@10 | **0.613** | 0.401 ⬇️ | 0.401 ⬇️ | 0.332 ⬇️ | **> 0.613** |
| ROUGE-L | **0.398** | — | 0.382 ⬇️ | — | — |
| Reranker NDCG@10 | — | 0.9637 | 0.9669 | 0.9554 | — |
| Reranker latency | 101 ms | — | 94 ms ✅ | — | — |

**Root cause — three separate issues, one per version:**

- **v1 (BM25-only negatives):** Negative distribution mismatch — reranker trained on BM25 candidates only, but inference uses hybrid (BM25 + dense) retrieval.
- **v2 (hybrid negatives):** Fixed distribution mismatch but triggered **catastrophic forgetting** — full fine-tuning updated all ~560 M parameters on 14,741 triplets from one textbook, erasing broad ranking knowledge.
- **v3 (LoRA):** Fixed catastrophic forgetting, but revealed the deepest issue: **parent/child chunk mismatch**. Training data contained child chunk text for both positives and negatives, but the inference pipeline calls `merge_docs()` before reranking, which replaces child chunks with their parent chunks from MongoDB. The reranker trained on text it never actually sees at inference time.

**⚠️ Current state:** `config.yaml` remains `reranker_use_finetuned: false` (base model active).
All fine-tuned checkpoints preserved at `outputs/models/reranker_finetuned/` for reference.
See **Step 4.7c** (v3 LoRA) and **Step 4.7d** (v4 parent-chunk fix) below.

### Step 4.7c — Reranker v3: LoRA Fine-tuning ✅ (Completed — still regressed, see Step 4.7d)

**Why LoRA instead of full fine-tuning?**

Full fine-tuning (v1, v2) updated all 560 M parameters → catastrophic forgetting.
LoRA freezes base weights and trains only ~7 M adapter parameters (~1.3%), preserving the
base model's broad ranking knowledge while adapting to the immunology domain.
Same technique as LLM SFT (Step 4.8). Config and code are already updated.

**Config changes (already applied in `config.yaml`):**

```yaml
training:
  reranker:
    use_lora: true          # ← LoRA is now the default
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    lora_target_modules: [query, key, value]
    learning_rate: 1.0e-4   # higher LR safe for LoRA (base frozen)
    warmup_ratio: 0.2
    eval_steps: 200         # more frequent eval to catch overfitting
```

**Step 1 — Install PEFT on AutoDL (one-time)**

```bash
pip install peft
python -c "from peft import LoraConfig; print('peft OK')"
```

**Step 2 — Smoke test: verify trainable parameter count**

```bash
cd /root/autodl-tmp/Immunology_RAG

TRANSFORMERS_OFFLINE=1 python -m train.train_reranker --lora --eval-only
# Expected output:
# [Reranker] Mode: LoRA
# trainable params: X,XXX,XXX || all params: ~560,000,000 || trainable%: ~1.3%
# [Reranker] Eval metrics (base model + LoRA scaffold): {ndcg@10: ~0.88, ...}
```

**Step 3 — Kill vLLM (same requirement as v1/v2)**

```bash
pkill -f vllm; sleep 5
nvidia-smi   # confirm < 5 GB used before training
```

**Step 4 — Train reranker v3 (~4–5 h)**

```bash
cd /root/autodl-tmp/Immunology_RAG
mkdir -p logs

TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker --force --lora \
    > logs/train_reranker_v3.log 2>&1 &
echo "[+] PID $! — follow progress:"
echo "    tail -f logs/train_reranker_v3.log"
```

Monitor progress:
```bash
tail -f logs/train_reranker_v3.log
# [Reranker] Mode: LoRA
# trainable params: ~7M || all params: ~560M || trainable%: ~1.30%
# [Reranker] Epoch 1/3 Step 200/5529 | Train loss: 0.XXXX | NDCG@10: 0.XXXX | ETA: ~XX min
# [Reranker] ✓ Sentinel written: ...training_complete.json
```

> **Expected NDCG@10:** ~0.85–0.92 (lower than full fine-tuning's 0.9669 — LoRA is harder
> to overfit, so the metric reflects genuine ranking ability rather than memorisation).

**Step 5 — Check adapter outputs (NOT a full model — small files expected)**

> **No merge step required.** Unlike the LLM (Step 4.9), the reranker LoRA adapter is
> loaded directly at runtime by `bge_m3_reranker.py` via `PeftModel.from_pretrained()`.
> Merging is only needed for vLLM, which requires a standalone model directory and cannot
> load a `(base_model + adapter)` pair. The reranker is not served by vLLM, so the raw
> adapter files are used as-is.

```bash
ls -lh outputs/models/reranker_finetuned/best/
# Expected:
#   adapter_config.json         ~1 KB   ← LoRA config (r, alpha, target_modules)
#   adapter_model.safetensors   ~30-50 MB ← trained adapter weights only
#   tokenizer.*                 ~few MB ← tokenizer files (for convenience)

# NOT expected: model.safetensors, pytorch_model.bin (those are full model files)
# If you see model.safetensors, LoRA was not active — check --lora flag and config.yaml
```

**Step 6 — Enable the LoRA reranker in config.yaml**

> ⚠️ **This step is required.** Training saves the adapter to disk but does NOT
> automatically activate it. The pipeline loads whichever model `reranker_use_finetuned`
> points to. Without this change, `evaluate.py` will silently use the base model and
> your v3 results will be invisible.
>
> Two separate flags control training vs inference — they are independent:
>
> | Flag | Controls | Set when? |
> |------|----------|-----------|
> | `training.reranker.use_lora: true` | How the model is **trained** (LoRA vs full FT) | Before training — already set ✅ |
> | `models.reranker_use_finetuned: true` | Which model the **pipeline loads** at runtime | After training — set it now ⬇️ |

```bash
cd /root/autodl-tmp/Immunology_RAG

# Open config.yaml and change this line:
#   reranker_use_finetuned: false   ← current (base model)
# to:
#   reranker_use_finetuned: true    ← activates LoRA adapter

# Quick one-liner edit (or use nano/vim):
sed -i 's/reranker_use_finetuned: false/reranker_use_finetuned: true/' config.yaml

# Verify the change took effect:
grep "reranker_use_finetuned" config.yaml
# Expected: reranker_use_finetuned: true
```

**Step 7 — Verify the LoRA adapter loads correctly (quick sanity check)**

Before running the full evaluation (~15-30 min), confirm the pipeline actually picks up
the adapter with a one-line Python test:

```bash
cd /root/autodl-tmp/Immunology_RAG

python -c "
from src.reranker.bge_m3_reranker import BGEM3ReRanker
r = BGEM3ReRanker()
print('Reranker ready.')
"
# Expected log lines:
# [Reranker] Loading BGE reranker from: .../reranker_finetuned/best
# [Reranker] Detected LoRA adapter — loading base model + adapter.
# [Reranker] LoRA adapter loaded (base: BAAI/bge-reranker-v2-m3)
# [Reranker] BGE reranker loaded.
# Reranker ready.
#
# If you see "Loading BGE reranker from: BAAI/bge-reranker-v2-m3" instead,
# the flag was not set — re-check config.yaml.
```

**Step 8 — Restart vLLM (needed for generation metrics in evaluate.py)**

```bash
cd /root/autodl-tmp/Immunology_RAG

# vLLM was killed before training (Step 4.6) — restart it now
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/outputs/models/llm_finetuned_merged \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --port 8000 &

# Wait for model to load into GPU (~30-60 s)
sleep 60
curl http://localhost:8000/v1/models
# Expected: {"data":[{"id":"Qwen/Qwen3-8B",...}]}

# If llm_finetuned_merged does not exist yet, fall back to base model:
# --model $(pwd)/models/Qwen3-8B
```

**Step 9 — Run full evaluation**

```bash
cd /root/autodl-tmp/Immunology_RAG
mkdir -p logs

nohup python -u evaluate.py > logs/evaluate_v3.log 2>&1 &
echo "[+] PID $! — follow progress:"
echo "    tail -f logs/evaluate_v3.log"
```

Monitor and verify adapter is active at the top of the eval log:
```bash
tail -f logs/evaluate_v3.log

# First few lines should confirm LoRA is loaded:
# [Reranker] Detected LoRA adapter — loading base model + adapter.
# [Reranker] LoRA adapter loaded (base: BAAI/bge-reranker-v2-m3)
# [Eval] Computing retrieval metrics (Recall@K, MRR@10)...
```

**Actual v3 LoRA results:**

| Metric | Base model | v2 full FT | v3 LoRA |
|--------|-----------|------------|---------|
| Recall@1 | **0.559** | 0.426 ⬇️ | 0.360 ⬇️ |
| MRR@10 | **0.613** | 0.401 ⬇️ | 0.332 ⬇️ |
| Reranker NDCG@10 | — | 0.9669 | 0.9554 |
| Training time | — | ~4.7 h | ~1.95 h (117 min) |

**v3 also regressed — worse than both base model and v2 full fine-tuning.**

**Root cause of v3 regression: parent/child chunk mismatch**

LoRA solved catastrophic forgetting, but exposed a deeper data bug: training data used
**child chunk** text for both positives and negatives, while the inference pipeline resolves
to **parent chunks** via `merge_docs()` before the reranker runs. The v3 adapter trained on
text it never actually sees at inference — explaining the unprecedented drop to Recall@1=0.360.

→ **Fix: Step 4.7d below** patches `build_train_data.py` to resolve all chunks to their
parent before writing triplets, aligning training granularity with inference.

**Revert to base model (active while v4 is being trained):**
```bash
sed -i 's/reranker_use_finetuned: true/reranker_use_finetuned: false/' config.yaml
grep "reranker_use_finetuned" config.yaml   # verify: false
```

If v3 training was interrupted and needs resuming:
```bash
TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker --resume --lora \
    > logs/train_reranker_v3.log 2>&1 &
```

If you want to fall back to full fine-tuning (legacy):
```bash
TRANSFORMERS_OFFLINE=1 python -m train.train_reranker --force --no-lora
```

### Step 4.7d — Reranker v4: Parent-Chunk Data Fix + LoRA Retrain

**Root cause (confirmed):** All previous fine-tuning attempts used child chunk text as
positive/negative passages. The inference pipeline calls `merge_docs()` **before** the
reranker, which resolves child chunks to their parent chunks from MongoDB. The training
distribution never matched the inference distribution.

**Fix applied to `train/build_train_data.py` (Stage 2 only):**

| Change | Location | Detail |
|--------|----------|--------|
| New helper `_resolve_parent_content(uid, uid_to_chunk)` | after `_build_bm25_negative` | Looks up chunk by `unique_id`, follows `parent_id` to return parent's `page_content`; graceful fallback if parent missing |
| `_build_bm25_negative` | adds `uid_to_chunk` param | Returns parent-resolved text for the hard negative |
| `_build_hybrid_negative` | adds `uid_to_chunk` param | Returns parent-resolved text for the top hybrid candidate |
| `build_reranker_data` | 4 changes | Builds `uid_to_chunk` dict; resolves `"pos"` via `_resolve_parent_content(qa["unique_id"], ...)` instead of raw `qa["passage"]`; passes `uid_to_chunk` to both negative functions; random fallback also resolves to parent |

No other stages (SFT data, eval set, QA cache) were touched.

**Step 1 — Delete only Stage 2 outputs (preserve Stage 1 QA cache)**

```bash
cd /root/autodl-tmp/Immunology_RAG

# ⚠️ Do NOT use --force — that deletes qa_pairs_cache.jsonl (3-4 h to regenerate)
rm data/train/reranker_train.jsonl data/train/reranker_test.jsonl

# Verify Stage 1 cache still exists
ls -lh data/train/qa_pairs_cache.jsonl
# Expected: ~10-20 MB file
```

**Step 2 — Regenerate Stage 2 with parent-chunk resolution (< 10 min)**

```bash
cd /root/autodl-tmp/Immunology_RAG

# vLLM does NOT need to be running — Stage 2 uses BM25 + Chroma only
TRANSFORMERS_OFFLINE=1 python -u -m train.build_train_data
# Stage 1: ✓ QA cache found — skipping (uses existing qa_pairs_cache.jsonl)
# Stage 2: rebuilds reranker_train.jsonl + reranker_test.jsonl with parent text
# Stage 3+4: ✓ found — skipping

# Verify triplets now contain parent-length text (longer than child chunks)
python3 -c "
import json
with open('data/train/reranker_train.jsonl') as f:
    r = json.loads(f.readline())
print('Query:', r['query'][:80])
print('POS words:', len(r['pos'].split()))   # expect > 100 (parent chunk)
print('NEG words:', len(r['neg'].split()))   # expect > 100 (parent chunk)
print('POS snippet:', r['pos'][:150])
"
# Expected: POS/NEG word counts ~150-300 (parent paragraphs, not short child chunks)
```

**Step 3 — Kill vLLM + delete v3 sentinel**

```bash
pkill -f vllm; sleep 5
nvidia-smi   # confirm < 5 GB used

# Remove sentinel so train_reranker accepts --force
rm -f outputs/models/reranker_finetuned/training_complete.json
```

**Step 4 — Train reranker v4 with LoRA + parent-chunk data (~2 h)**

```bash
cd /root/autodl-tmp/Immunology_RAG
mkdir -p logs

TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker --force --lora \
    > logs/train_reranker_v4.log 2>&1 &
echo "[+] PID $! — follow progress:"
echo "    tail -f logs/train_reranker_v4.log"
```

Monitor:
```bash
tail -f logs/train_reranker_v4.log
# [Reranker] Mode: LoRA
# trainable params: ~7M || all params: ~560M || trainable%: ~1.30%
# [Reranker] Epoch 1/3 Step 200/XXXX | Train loss: 0.XXXX | NDCG@10: 0.XXXX | ETA: ~XX min
# [Reranker] ✓ Sentinel written: ...training_complete.json
```

**Step 5 — Verify adapter outputs**

```bash
ls -lh outputs/models/reranker_finetuned/best/
# adapter_config.json          ~1 KB
# adapter_model.safetensors    ~30-50 MB
# tokenizer.*                  few MB
```

**Step 6 — Enable the v4 reranker**

```bash
sed -i 's/reranker_use_finetuned: false/reranker_use_finetuned: true/' config.yaml
grep "reranker_use_finetuned" config.yaml   # verify: true
```

**Step 7 — Sanity check: confirm LoRA adapter loads**

```bash
python -c "
from src.reranker.bge_m3_reranker import BGEM3ReRanker
r = BGEM3ReRanker()
print('Reranker ready.')
"
# Expected:
# [Reranker] Detected LoRA adapter — loading base model + adapter.
# [Reranker] LoRA adapter loaded (base: BAAI/bge-reranker-v2-m3)
# Reranker ready.
```

**Step 8 — Restart vLLM**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/outputs/models/llm_finetuned_merged \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --port 8000 &
sleep 60
curl http://localhost:8000/v1/models
```

**Step 9 — Run evaluation**

```bash
nohup python -u evaluate.py > logs/evaluate_v4.log 2>&1 &
tail -f logs/evaluate_v4.log
```

**Target metrics (v4):**

| Metric | Base model | v3 LoRA (data bug) | v4 target |
|--------|-----------|-------------------|-----------|
| Recall@1 | 0.559 | 0.360 ⬇️ | **> 0.559** |
| MRR@10 | 0.613 | 0.332 ⬇️ | **> 0.613** |

**Revert to base model if v4 also underperforms:**
```bash
sed -i 's/reranker_use_finetuned: true/reranker_use_finetuned: false/' config.yaml
```

---

### Step 4.8 — Fine-Tune LLM (~4.6 h on A100-40GB — 274 min, 3 epochs × 16,936 samples, 6,351 steps)

Fine-tunes Qwen3-8B with LoRA using LLaMA-Factory. vLLM must still be killed.

> **Checkpoint/resume:** LLaMA-Factory saves a `checkpoint-N/` folder after each epoch.
> If interrupted, re-running auto-detects the latest checkpoint and resumes from there.
> A `training_complete.json` sentinel is written on success. Use `--force` to start over.

> ⚠️ **Check disk space before starting — LLM SFT needs ~50 GB free on the system disk.**
> LLaMA-Factory loads the full Qwen3-8B model (~16 GB) from the local `models/Qwen3-8B/`
> directory. If `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/` exists but is incomplete
> (only index file, missing shards), LLaMA-Factory will try to download ~16 GB more →
> disk fills → `OSError: [Errno 28] No space left on device`.
>
> **Before running, verify:**
> ```bash
> df -h   # system disk (overlay) must have ≥ 50 GB free
> ls -lh /root/autodl-tmp/Immunology_RAG/models/Qwen3-8B/
> # Must show: model-00001-of-00005.safetensors through model-00005-of-00005.safetensors
>
> # If ~/.cache has a partial download, remove it:
> rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B
> ```
>
> **The code uses `models/Qwen3-8B` (local path) — no HF download occurs.**
> `config.yaml` has `llm_local_path: models/Qwen3-8B`; `constant.llm_local_path` resolves
> this to the absolute path for LLaMA-Factory. `TRANSFORMERS_OFFLINE=1` is NOT needed here.

```bash
cd /root/autodl-tmp/Immunology_RAG
mkdir -p logs

# Run in background — LLaMA-Factory is auto-discovered from config.yaml
nohup python -u -m train.train_llm_sft \
    > logs/train_llm_sft.log 2>&1 &
echo "[+] PID $! — Job started. Follow progress:"
echo "    tail -f logs/train_llm_sft.log"
```

Monitor progress:
```bash
tail -f logs/train_llm_sft.log
# LLaMA-Factory progress: "{'loss': 1.82, 'epoch': 1.0, ...}"
# Finished when:          "[SFT] ✓ Sentinel written: ...training_complete.json"
```

Expected log output (based on actual run: 16,936 samples × 3 epochs on A100-40GB):
```
[SFT] Config written: outputs/models/llm_finetuned/sft_config.yaml
[SFT] Running LLaMA-Factory: ...
***** Running training *****
  Num examples = 16,936
  Num Epochs = 3
  Instantaneous batch size per device = 4
  Total train batch size (w. parallel, distributed & accumulation) = 8
  Gradient Accumulation steps = 2
  Total optimization steps = 6,351
  Number of trainable parameters = 7,667,712
  1%|█  | 59/6351 [02:30<4:06:05,  2.35s/it
...
***** train metrics *****
  epoch                    =          3.0
  total_flos               = 1597309824GF
  train_loss               =       0.2691
  train_runtime            =   4:32:57.41
  train_samples_per_second =        3.102
  train_steps_per_second   =        0.388

[SFT] Training finished in 274.4 min.
[SFT] ✓ Sentinel written: /root/autodl-tmp/Immunology_RAG/outputs/models/llm_finetuned/training_complete.json
[SFT] Training curves saved: .../outputs/llm_eval/sft_training_curves.png
[SFT] Evaluating Pre-trained: .../models/Qwen3-8B
[SFT Eval] Warning: pipeline failed: Connection error.
[SFT Eval] Warning: pipeline failed: Connection error.
```

> **Actual training time: ~4.6 h** (274 min, 3 epochs, A100-40GB, LoRA r=16).
> First progress bar appears within ~3 min at step ~59.
>
> The two `pipeline failed: Connection error` lines at the end are **expected and harmless.**
> The post-training comparison step tries to call vLLM at `localhost:8000` — but vLLM was
> killed before training (Step 4.6), so nothing is listening. The trained checkpoint is
> saved correctly regardless.

If interrupted and re-run:
```bash
# Re-run the same nohup command — auto-resumes from latest checkpoint-N/
nohup python -u -m train.train_llm_sft \
    > logs/train_llm_sft.log 2>&1 &

# Force retrain from scratch (ignores existing checkpoints):
nohup python -u -m train.train_llm_sft --force \
    > logs/train_llm_sft.log 2>&1 &
```

Output files (check after job completes):
```bash
cat outputs/models/llm_finetuned/training_complete.json  # sentinel — elapsed time, checkpoint path
ls  outputs/models/llm_finetuned/                        # checkpoint-6351/ (actual), checkpoint-N/ pattern
ls  outputs/llm_eval/                                    # sft_training_curves.png
# Note: comparison.png is NOT generated — post-training eval skipped (vLLM not running)
```

> **Important:** The output at this stage is a **LoRA adapter** (a small diff from the
> base model), NOT a standalone model. vLLM cannot load a raw LoRA adapter directly —
> you must merge it into the base model first (Step 4.9 below).

### Step 4.9 — Merge LoRA Adapter into Base Model

This step fuses the LoRA weights into the Qwen3-8B base model weights, producing a
standalone merged model that vLLM can load directly.

```bash
cd /root/autodl-tmp/Immunology_RAG

# Find the latest checkpoint
CHECKPOINT=$(ls -d outputs/models/llm_finetuned/checkpoint-* | sort -V | tail -1)
echo "Merging checkpoint: $CHECKPOINT"

# Create output directory for merged model
mkdir -p outputs/models/llm_finetuned_merged

# Merge using llamafactory-cli export
llamafactory-cli export \
    --model_name_or_path $(pwd)/models/Qwen3-8B \
    --adapter_name_or_path $CHECKPOINT \
    --template qwen \
    --finetuning_type lora \
    --export_dir $(pwd)/outputs/models/llm_finetuned_merged \
    --export_size 4 \
    --export_legacy_format false

# Verify merged model exists
ls outputs/models/llm_finetuned_merged/
# Expected: config.json  model-*.safetensors  tokenizer.json  generation_config.json
```

> **Why merge instead of loading LoRA on the fly?**
> vLLM does not support loading LoRA adapters at inference time in the standard workflow.
> The merged model contains exactly the same parameters as if you had done full fine-tuning,
> but it was produced efficiently via LoRA. After merging, `llm_finetuned_merged/` is a
> complete, self-contained model directory.

### Step 4.10 — Restart vLLM with the Fine-Tuned Merged Model

Now restart vLLM pointing to the **merged fine-tuned model** instead of the original.

```bash
# Kill any existing vLLM (safety measure)
pkill -f vllm; sleep 3

cd /root/autodl-tmp/Immunology_RAG

# Start vLLM with the MERGED fine-tuned model
# Note: --served-model-name stays "Qwen/Qwen3-8B" so config.yaml doesn't need to change
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/outputs/models/llm_finetuned_merged \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --port 8000 &

# Wait for model to load
sleep 60
curl http://localhost:8000/v1/models
# Expected: {"data":[{"id":"Qwen/Qwen3-8B",...}]}
```

### Step 4.11 — Run Full System Evaluation (~15-30 min)

#### Standard evaluation (reranker + generation quality)

```bash
cd /root/autodl-tmp/Immunology_RAG
mkdir -p logs

# Full evaluation (~15-30 min) — run in background
nohup python -u evaluate.py \
    > logs/evaluate.log 2>&1 &
echo "[+] PID $! — Job started. Follow progress:"
echo "    tail -f logs/evaluate.log"

# Quick version (20 random pairs, ~3 min) — safe to run while testing
# nohup python -u evaluate.py --quick > logs/evaluate_quick.log 2>&1 &
```

Monitor progress:
```bash
tail -f logs/evaluate.log
# Finished when you see: "[Eval] Evaluation complete in XXXs"
```

Output files (check after job completes):
```bash
ls outputs/system_eval/
#   evaluation_report.html    <-- full HTML report (open in browser)
#   retrieval_recall.png      <-- Recall@K + MRR@10 line chart
#   reranker_precision.png    <-- before/after reranking bar chart
#   generation_quality.png    <-- ROUGE-L + BERTScore by doc type
#   e2e_radar.png             <-- end-to-end radar chart
#   latency_breakdown.png     <-- per-module latency stacked bar
```

#### Pretrained vs Fine-tuned LLM comparison (optional, requires two vLLM servers)

Compares generation quality (ROUGE-L, BERTScore-F1) between the base `Qwen3-8B` and the
LoRA-fine-tuned model using the **same retrieved context** for both, so only the LLM
generation step differs.

**Setup: start two vLLM servers simultaneously**

> ⚠️ **VRAM requirement:** Two Qwen3-8B instances (~35 GB × 2 = 70 GB).
> Requires A100-SXM4-80GB. On A100-40GB, run the comparison with the quick flag (10 samples)
> or compare separately across two independent evaluation runs instead.

```bash
# Terminal 1 — pretrained base model on port 8000 (may already be running)
pkill -f vllm; sleep 5
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.40 \
    --enable-prefix-caching --port 8000 &

# Terminal 2 — fine-tuned merged model on port 8001
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/outputs/models/llm_finetuned_merged \
    --served-model-name finetuned \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.40 \
    --enable-prefix-caching --port 8001 &

# Wait for both to load (~60-90 s)
sleep 90
curl http://localhost:8000/v1/models   # → "id": "Qwen/Qwen3-8B"
curl http://localhost:8001/v1/models   # → "id": "finetuned"
```

**Run evaluation with LLM comparison:**

```bash
cd /root/autodl-tmp/Immunology_RAG
mkdir -p logs

nohup python -u evaluate.py \
    --compare-llm \
    --finetuned-llm-url http://localhost:8001/v1 \
    --finetuned-llm-model finetuned \
    > logs/evaluate_llm_compare.log 2>&1 &
echo "[+] PID $! — follow progress:"
echo "    tail -f logs/evaluate_llm_compare.log"
```

Monitor:
```bash
tail -f logs/evaluate_llm_compare.log
# [Eval] Running LLM comparison (pretrained vs fine-tuned)...
# [Eval]   Pretrained : http://localhost:8000/v1  model=Qwen/Qwen3-8B
# [Eval]   Fine-tuned : http://localhost:8001/v1  model=finetuned
# [LLM Compare] Progress: 10/50
# [LLM Compare] Progress: 20/50
# ...
# [Eval]   Pretrained : {'rouge_l': X.XXX, 'bertscore_f1': X.XXX}
# [Eval]   Fine-tuned : {'rouge_l': X.XXX, 'bertscore_f1': X.XXX}
```

Additional output file when `--compare-llm` is used:
```bash
ls outputs/system_eval/
#   ...                       (existing charts)
#   llm_comparison.png        <-- grouped bar chart: pretrained vs fine-tuned
#                                 with Δ ROUGE-L and Δ BERTScore annotation box
```

The HTML report (`evaluation_report.html`) will include a new **"LLM Comparison:
Pretrained vs Fine-tuned"** section with chart + table showing absolute deltas
(green = improvement, red = regression).

**A100-40GB alternative: sequential comparison (no dual-server VRAM needed)**

```bash
# Run 1 — base model (port 8000, standard vLLM)
nohup python -u evaluate.py > logs/evaluate_base_llm.log 2>&1 &

# After it finishes, swap vLLM to the fine-tuned model and re-run
pkill -f vllm; sleep 5
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/outputs/models/llm_finetuned_merged \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --port 8000 &
sleep 60

nohup python -u evaluate.py > logs/evaluate_finetuned_llm.log 2>&1 &
# Compare rouge_l and bertscore_f1 between the two logs
```

> **RAGAS metrics (faithfulness, context_precision, answer_relevancy)** require a real
> OpenAI API key and are skipped gracefully with a message. All other metrics are computed
> locally.

### Step 4.12 — Launch Streamlit UI

```bash
cd /root/autodl-tmp/Immunology_RAG
streamlit run app.py --server.port 6006 --server.address 0.0.0.0

# Access via AutoDL custom service URL → port 6006
```

Open the **Q&A** page and test: `"What is the role of MHC class II molecules?"` —
expect a detailed LLM answer with `[1]`, `[2]` citation markers.

---

## 5. Execution Order — Inference Path (After Fine-Tuning)

This section covers **daily use** after the system has been fully trained and the merged
model exists at `outputs/models/llm_finetuned_merged/`. Start services in this exact order.

### Step 5.1 — Start MongoDB

```bash
pkill mongod 2>/dev/null; sleep 2
mkdir -p /data/db && chmod 777 /data/db
rm -f /data/db/mongod.lock
mongod --fork --logpath /var/log/mongodb.log --dbpath /data/db
sleep 2 && mongosh --eval "db.runCommand({ping: 1})"
# Expected: { ok: 1 }
```

### Step 5.2 — Start Semantic Chunking Service (CPU)

> Only needed if you plan to add new PDFs or rebuild the index. You can skip this step
> if you are only doing inference (Q&A) on an already-built index.

```bash
cd /root/autodl-tmp/Immunology_RAG
python -m uvicorn src.server.semantic_chunk:app --host 0.0.0.0 --port 6000 &
sleep 12 && curl http://localhost:6000/health
# Expected: {"status":"ok",...}
```

### Step 5.3 — Start vLLM with Fine-Tuned Model

```bash
pkill -f vllm 2>/dev/null; sleep 3   # kill any previous vLLM instance

cd /root/autodl-tmp/Immunology_RAG

# Use the MERGED fine-tuned model (not the original models/Qwen3-8B)
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/outputs/models/llm_finetuned_merged \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --port 8000 &

# Wait for model to load (30-60 seconds)
sleep 60
curl http://localhost:8000/v1/models
# Must show: "id": "Qwen/Qwen3-8B"
```

> **If the merged model does not exist yet** (you haven't completed fine-tuning), fall
> back to the original base model during inference:
> ```bash
> python -m vllm.entrypoints.openai.api_server \
>     --model $(pwd)/models/Qwen3-8B \
>     --served-model-name Qwen/Qwen3-8B \
>     --dtype bfloat16 --max-model-len 8192 \
>     --gpu-memory-utilization 0.85 --enable-prefix-caching --port 8000 &
> ```

### Step 5.4 — (Optional) Run Quick Evaluation

```bash
cd /root/autodl-tmp/Immunology_RAG
python evaluate.py --quick
```

### Step 5.5 — Launch Streamlit UI

```bash
cd /root/autodl-tmp/Immunology_RAG
streamlit run app.py --server.port 6006 --server.address 0.0.0.0
```

Access via AutoDL Custom Service → port 6006, or SSH tunnel:
```bash
# On your local machine:
ssh -p <port> -L 6006:localhost:6006 root@<host>
# Then open http://localhost:6006 in browser
```

### Summary: Service Startup Order for Inference

```
1. mongod (always first — data layer)
2. uvicorn src.server.semantic_chunk:app (port 6000, CPU, optional)
3. vllm ...outputs/models/llm_finetuned_merged (port 8000, GPU)
4. streamlit run app.py (port 6006)
```

---

## 6. LoRA Merge: Making Fine-Tuned Model Ready for vLLM

### Why Merging Is Required

When `train_llm_sft.py` completes, the output in `outputs/models/llm_finetuned/checkpoint-N/`
is a **LoRA adapter** — it contains only the weight **differences** from the base model,
stored in small matrices. This is much smaller than the full model but:

- **vLLM cannot directly serve a LoRA adapter** without special merge support.
- **Merging** combines adapter weights back into the base model, producing a standard
  HuggingFace model directory that any inference engine can load.

### Quick Merge Command

```bash
cd /root/autodl-tmp/Immunology_RAG

# Find your latest checkpoint
CHECKPOINT=$(ls -d outputs/models/llm_finetuned/checkpoint-* | sort -V | tail -1)
echo "Latest checkpoint: $CHECKPOINT"

mkdir -p outputs/models/llm_finetuned_merged

llamafactory-cli export \
    --model_name_or_path $(pwd)/models/Qwen3-8B \
    --adapter_name_or_path $CHECKPOINT \
    --template qwen \
    --finetuning_type lora \
    --export_dir $(pwd)/outputs/models/llm_finetuned_merged \
    --export_size 4 \
    --export_legacy_format false
```

### Verifying the Merge

```bash
# Should contain ~4 safetensor shards + config + tokenizer
ls outputs/models/llm_finetuned_merged/

# Quick inference test (requires vLLM running with merged model)
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='dummy')
resp = client.chat.completions.create(
    model='Qwen/Qwen3-8B',
    messages=[{'role':'user','content':'What is an antibody?'}],
    max_tokens=100
)
print(resp.choices[0].message.content)
"
```

### What If Merge Fails?

```bash
# Verify llamafactory-cli is installed
llamafactory-cli --help

# If not found:
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..

# Common error: "adapter_name_or_path does not exist"
# → The checkpoint path is wrong. List available checkpoints:
ls outputs/models/llm_finetuned/
```

---

## 7. Incremental PDF Addition

To add new PDFs to the knowledge base without rebuilding from scratch:

```bash
# 1. Copy the new PDF to data/raw/
cp /path/to/new_immunology_paper.pdf data/raw/

# 2. Ensure services are running (MongoDB + semantic chunking)
# (see Steps 5.1 and 5.2 — vLLM is NOT needed for index building)

# 3. Index only the new PDF (auto-skips already-indexed chunks)
python build_index.py --pdf data/raw/new_immunology_paper.pdf --test-retrieval

# 4. Verify chunk count increased
python -c "
from src.client.mongodb_config import MongoConfig
from src import constant
c = MongoConfig.get_collection(constant.mongo_collection)
print(f'Total chunks: {c.count_documents({})}')
"
```

---

## 8. A100 Optimization Tips

### bf16 Everywhere (A100 Native)

A100 has dedicated bf16 tensor cores. All training uses bf16 by default:

```yaml
# config.yaml — these are the defaults:
training:
  reranker:
    bf16: true
    fp16: false   # keep false — fp16 causes NaN with noisy OCR text
  llm_sft:
    bf16: true
```

### Flash Attention 2 (Optional — off by default)

FA2 reduces VRAM and speeds up training on long sequences. Disabled by default
because installation requires CUDA compilation and is optional.

```bash
# Enable if you need it (e.g., 4096-token sequences with tight VRAM):
pip install flash-attn --no-build-isolation
# If compilation fails with your CUDA version:
pip install flash-attn==2.5.8 --no-build-isolation

# Then enable in config.yaml:
# training:
#   llm_sft:
#     flash_attn: true
```

**A100-80GB users:** Flash Attention 2 is not needed — you have plenty of VRAM.

### vLLM Memory Tuning

```bash
# A100-40GB: 0.85 leaves ~6 GB headroom for KV cache growth
--gpu-memory-utilization 0.85

# A100-80GB: can increase for more KV cache
--gpu-memory-utilization 0.90

# If running out of KV cache during long conversations:
--max-model-len 4096   # reduce context window
```

### Batch Size Guidelines

| Task | A100-40GB | A100-80GB |
|------|-----------|-----------|
| Reranker training | `batch_size=16` | `batch_size=32` |
| LLM SFT (LoRA) | `batch_size=4, grad_accum=4` | `batch_size=8, grad_accum=2` |
| Embedding (BGE-M3) | `batch_size=64` | `batch_size=128` |

### TensorBoard Monitoring

```bash
# Monitor training loss in real-time
tensorboard --logdir outputs/models/reranker_finetuned/runs --port 6007 &
tensorboard --logdir outputs/models/llm_finetuned/runs --port 6008 &

# Access via SSH tunnel:
ssh -p <port> -L 6007:localhost:6007 -L 6008:localhost:6008 root@<host>
```

---

## 9. Port Mapping for Streamlit

### AutoDL Custom Service

1. Go to instance dashboard → **Custom Service** tab
2. Set internal port: `6006`
3. AutoDL provides an external HTTPS URL like `https://<id>.autodl.run`

### SSH Tunnel (Alternative)

```bash
# Forward multiple ports in one command
ssh -p <port> \
    -L 6006:localhost:6006 \
    -L 6007:localhost:6007 \
    -L 8000:localhost:8000 \
    root@<host>

# Then access locally:
# Streamlit:  http://localhost:6006
# TensorBoard: http://localhost:6007
# vLLM API:   http://localhost:8000
```

### Streamlit Command

```bash
streamlit run app.py \
    --server.port 6006 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none   # optional: disable file watcher for faster startup
```

---

## 10. Common Errors and Solutions

### Reranker: loss collapses to 0.0000 and NDCG@10 hits 1.0000 after ~100 steps

**Symptom:**
```
[Reranker] Epoch 1/3 Step 50/5718  | Train loss: 0.8074 | NDCG@10: 0.8760 ...
[Reranker] Epoch 1/3 Step 100/5718 | Train loss: 0.0015 | NDCG@10: 1.0000 ...
[Reranker] Epoch 1/3 Step 150/5718 | Train loss: 0.0000 | NDCG@10: 1.0000 ...
# ... stays at 0.0000 / 1.0000 for all remaining steps
```

**Cause:** Hard negatives in `reranker_train.jsonl` were LLM-generated hallucinated text.
A cross-encoder trivially distinguishes hallucinated passages from real textbook text by
writing style alone — no domain knowledge required. Model memorises this artefact in ~100 steps.

**Solution:** Regenerate Stage 2 data using the fixed `build_train_data.py`, which now
uses BM25 retrieval for hard negatives (actual textbook passages, no LLM needed):

```bash
# Kill the wasted training run (best checkpoint already saved at step 100)
kill $(pgrep -f train_reranker)

# Delete Stage 2 files so the script regenerates them (Stage 1 cache is preserved)
# ⚠️ Do NOT use --force — that deletes qa_pairs_cache.jsonl too (3+ hour re-run)
cd /root/autodl-tmp/Immunology_RAG
rm data/train/reranker_train.jsonl data/train/reranker_test.jsonl

# Regenerate Stage 2 only (Stage 1 skipped — qa_pairs_cache.jsonl still exists)
# vLLM does NOT need to be running
python -u -m train.build_train_data --workers 4
# Completes in < 5 min (BM25-only, no LLM calls)

# Verify negatives are real textbook passages (not <think> blocks or LLM text)
head -n 1 data/train/reranker_train.jsonl | python3 -c "
import json, sys; r = json.load(sys.stdin)
print('NEG:', r['neg'][:200])
"

# Retrain reranker with new data
TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker \
    --force > logs/train_reranker.log 2>&1 &
tail -f logs/train_reranker.log
```

**Expected result after fix:** Loss decreases gradually over 3 epochs; NDCG@10 climbs
from ~0.88 → 0.93–0.97 rather than instantly collapsing to 1.0.

---

### `train_reranker.py` prints 3 lines then hangs for hours (no training output)

**Symptom:**
```
[Reranker] Starting from base model: BAAI/bge-reranker-v2-m3
[Reranker] Training on: cuda
[Reranker] Train file:  .../reranker_train.jsonl
```
…then nothing for hours. `[Reranker] Train samples:` never appears.

**Cause:** `config.yaml` sets `models.reranker: BAAI/bge-reranker-v2-m3` (a HuggingFace Hub
ID). `from_pretrained()` tries to contact `hf.co` to check for model updates — which is
blocked on AutoDL. The call hangs indefinitely before training starts.

**Solution:**
```bash
# Kill the frozen job
kill $(pgrep -f train_reranker)

# Restart with TRANSFORMERS_OFFLINE=1 — uses local HF cache, no network calls
TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker \
    > logs/train_reranker.log 2>&1 &
tail -f logs/train_reranker.log
```

Within 2–3 minutes you should see `[Reranker] Train samples: 30484 | Eval samples: 3388`.

**Prevention:** Always prefix `train_reranker.py` with `TRANSFORMERS_OFFLINE=1`
(already included in all Step 4.7 commands above).

---

### LLM SFT fails with `OSError: [Errno 28] No space left on device`

**Symptom:**
```
Fetching 5 files: 0%| | 0/5 [01:31<?, ?it/s]
OSError: [Errno 28] No space left on device
[SFT] Training failed with exit code 1.
```

**Cause:** LLaMA-Factory received the HuggingFace Hub ID `Qwen/Qwen3-8B` as `model_name_or_path`.
It found only the model index file in `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/` (partial
download) but not the 5 shard files (~16 GB), so it attempted to download them — filling the
system disk (typically 30 GB, already ~23 GB used).

**Solution:**
```bash
# 1. Free disk: remove the partial HF cache entry
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B

# 2. Verify the local model is intact
ls -lh /root/autodl-tmp/Immunology_RAG/models/Qwen3-8B/
# Must show: model-00001-of-00005.safetensors through model-00005-of-00005.safetensors

# 3. Check free space (need ≥ 10 GB free to run safely)
df -h   # look at the "overlay" row

# 4. Re-run SFT — now uses local path, no HF download
nohup python -u -m train.train_llm_sft > logs/train_llm_sft.log 2>&1 &
tail -f logs/train_llm_sft.log
```

**Why this is fixed:** `train_llm_sft.py` now uses `constant.llm_local_path` (which resolves
`models/Qwen3-8B` → `/root/autodl-tmp/Immunology_RAG/models/Qwen3-8B`), not the HF Hub ID.
LLaMA-Factory finds the model immediately — no network calls, no disk writes.

---

### Fine-tuned reranker performs worse than base model in end-to-end evaluation

**Symptom:**
```
# With reranker_use_finetuned: true  (observed with both v1 and v2 fine-tuning)
recall@1: 0.426   ← was 0.559 with base model  (−13.3 pp)
mrr@10:   0.401   ← was 0.613 with base model  (−21.3 pp)
```
Reranker training showed excellent NDCG@10 (0.9637 for v1, 0.9669 for v2) on its held-out
test set, but end-to-end pipeline metrics regressed significantly.

**Two causes were investigated:**

*v1 (BM25-only negatives):* Stage 2 training data used BM25-only hard negatives. In production
the pipeline uses hybrid retrieval (BM25 + dense Chroma). The reranker learned to distinguish
BM25 candidates but was never exposed to dense-retrieval candidates → poor reranking at inference.

*v2 (hybrid negatives, Step 4.7b):* Fixed the distribution mismatch — but metrics still regressed.
Root cause: **catastrophic forgetting**. Full fine-tuning updated all ~560 M parameters on
14,741 triplets from one textbook, overwriting the base model's broad ranking knowledge
(trained on hundreds of millions of diverse pairs). The high NDCG@10 was misleading — it only
measured within-distribution reranking; the base model generalises better to held-out queries.

**Current state (April 2026):** `reranker_use_finetuned: false` — running base model.
All fine-tuned checkpoints preserved at `outputs/models/reranker_finetuned/` for reference.

**Progression of fixes:**
- v1 → v2: fixed negative distribution (BM25-only → hybrid)
- v2 → v3: fixed catastrophic forgetting (full FT → LoRA)
- v3 → v4: fixed training/inference granularity mismatch (child chunks → parent chunks)

See Step 4.7d for the v4 fix.

```bash
# Immediate workaround — revert to base model:
# config.yaml → reranker_use_finetuned: false
```

---

### Fine-tuned reranker v3 (LoRA) performs even worse than v2 (Recall@1 dropped to 0.360)

**Symptom:**
```
# With reranker_use_finetuned: true, v3 LoRA adapter
recall@1: 0.360   ← was 0.559 with base model  (−19.9 pp)
mrr@10:   0.332   ← was 0.613 with base model  (−28.1 pp)
# Worse than v2 full fine-tuning (0.426 / 0.401) despite LoRA fixing catastrophic forgetting
```

**Cause: parent/child chunk mismatch in training data**

The inference pipeline resolves child chunks to their parent chunks via `merge_docs()` from
MongoDB **before** passing them to the reranker. Training data (`build_train_data.py` Stage 2)
wrote the raw child chunk `page_content` as both positives and negatives. The v3 LoRA adapter
trained on short child-chunk snippets (~60-100 words) but at inference only ever sees longer
parent-chunk paragraphs (~150-300 words). This distribution mismatch is more severe than the
distribution mismatches in v1/v2 — explaining the worst-ever Recall@1.

**Diagnosis: verify what the reranker actually receives at inference**

```bash
python3 -c "
from src.pipeline import RAGPipeline
from src.utils import format_context
p = RAGPipeline()
result = p.answer('What activates T cells?')
for i, doc in enumerate(result['retrieved_docs'][:2], 1):
    print(f'[{i}] words={len(doc.page_content.split())} snippet={doc.page_content[:120]}')
"
# If word counts are 150-300 → parent chunks (correct at inference)
# If word counts are 50-80  → child chunks (something wrong with merge_docs)
```

**Diagnosis: verify what the training data contains**

```bash
python3 -c "
import json
with open('data/train/reranker_train.jsonl') as f:
    r = json.loads(f.readline())
print('POS words:', len(r['pos'].split()))   # was ~60-80 in v1/v2/v3 (child chunks)
print('NEG words:', len(r['neg'].split()))   # was ~60-80 in v1/v2/v3 (child chunks)
"
# After v4 fix: POS/NEG should be 150-300 words (parent chunks)
# Before v4 fix: ~60-80 words (child chunks — mismatch)
```

**Fix:** Regenerate Stage 2 data with parent-chunk resolution and retrain (see Step 4.7d).
The fix is in `train/build_train_data.py`: `_resolve_parent_content()` is called for every
positive, negative, and random-fallback before writing to the triplet file.

```bash
# Quick fix path:
cd /root/autodl-tmp/Immunology_RAG
rm data/train/reranker_train.jsonl data/train/reranker_test.jsonl
rm -f outputs/models/reranker_finetuned/training_complete.json
TRANSFORMERS_OFFLINE=1 python -u -m train.build_train_data
TRANSFORMERS_OFFLINE=1 nohup python -u -m train.train_reranker --force --lora \
    > logs/train_reranker_v4.log 2>&1 &
```

---

### `MaxRetryError: Connection to huggingface.co timed out`

**Cause:** `HF_ENDPOINT` not set. AutoDL blocks direct `huggingface.co` access.

**Solution:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
# Then re-run the failing command
```

**Prevention:** Always run `source ~/.bashrc` after a new SSH login, or add
`export HF_ENDPOINT=https://hf-mirror.com` to your runbook first-line habit.

---

### `ValueError: Free memory on device cuda:0 less than desired GPU memory utilization`

**Cause:** Another process already loaded onto the GPU before vLLM started.
Most common: semantic chunking service on old code (before CPU-only fix) loading
`all-MiniLM-L6-v2` on CUDA (~35 GB consumed).

**Solution:**
```bash
nvidia-smi                              # identify what's using GPU
pkill -f uvicorn 2>/dev/null; sleep 3  # kill semantic chunking if it's the cause

# Restart semantic chunking on CPU (current code hard-codes device="cpu")
python -m uvicorn src.server.semantic_chunk:app --host 0.0.0.0 --port 6000 &
sleep 10 && curl http://localhost:6000/health
# Log must show: "on cpu (CPU-only)" — NOT "on cuda"

# Now start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --port 8000 &
```

---

### `CUDA out of memory` during `evaluate.py` generation / per-document-type breakdown

**Symptom:**
```
[Eval] Running per-document-type breakdown...
Generation failed: CUDA out of memory. Tried to allocate 490.00 MiB.
GPU 0 has a total capacity of 39.49 GiB of which 296.25 MiB is free.
Process 11924 has 3.08 GiB memory in use.
Process 20119 has 34.63 GiB memory in use.
```

**Cause:** vLLM (process 20119) holds ~34.6 GB. The evaluate.py process (process 11924) holds
~3 GB (BGE-M3 encoder + reranker model). The two together leave only ~296 MiB free.
The CUDA allocator cannot find 490 MiB of contiguous memory for the next reranker batch.

**Root cause in code:** `src/reranker/bge_m3_reranker.py`, `rank()` method — the `inputs` tensor
(tokenized query+doc pairs, can be 100–400 MB) was moved to CUDA with `.to("cuda")` but
never explicitly freed after scoring. Python's garbage collector reclaimed it eventually
but not before the next allocation needed the memory.

**Fix applied (permanent):** `del inputs; torch.cuda.empty_cache()` added after
`scores = scores.detach().cpu()` in `rank()`. Releases GPU tensors immediately after each batch.

**If you still hit OOM after the fix:**
```bash
# Option 1: Memory allocator hint (no code change, try first)
PYTORCH_ALLOC_CONF=expandable_segments:True python -m evaluate

# Option 2: Reduce vLLM GPU reservation (~2 GB freed)
# Restart vLLM with lower utilization:
pkill -f vllm; sleep 5
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/outputs/models/llm_finetuned_merged \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.82 \   # ← reduced from 0.85
    --enable-prefix-caching --port 8000 &
sleep 60 && python -m evaluate
```

---

### `CUDA out of memory` during training

**Cause:** Training started with vLLM still running, or batch size too large.

**Solution:**
```bash
pkill -f vllm; sleep 5     # free GPU memory first
nvidia-smi                  # confirm < 5 GB used

# Then retry training:
python -m train.train_reranker    # or train_llm_sft
```

If OOM persists with vLLM killed:
```bash
# Reduce batch size in config.yaml:
# training.reranker.batch_size: 8   (from 16)
# training.llm_sft.batch_size: 2    (from 4)
```

---

### `404 — The model 'Qwen/Qwen3-8B' does not exist`

**Cause:** vLLM was started without `--served-model-name`, so the API registered the
model under its disk path. `config.yaml` uses `Qwen/Qwen3-8B` as the model ID → mismatch.

**Solution:** Always use `--served-model-name Qwen/Qwen3-8B`:
```bash
pkill -f vllm; sleep 5
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \   # <-- this line is critical
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --port 8000 &
sleep 60 && curl http://localhost:8000/v1/models
# Verify: "id": "Qwen/Qwen3-8B"
```

---

### `400 Bad Request: context length exceeded`

**Cause:** `max_tokens` in `config.yaml` is too high. A prompt with 3000 tokens +
`max_tokens=4096` = 7096, which exceeds the 8192 token limit.

**Solution:**
```yaml
# config.yaml
llm:
  max_tokens: 1024   # not 4096 — keep room for the prompt
```

---

### `Connection refused: localhost:8000`

**Cause:** vLLM not running (crashed, killed, or not started yet).

**Solution:**
```bash
ps aux | grep vllm    # check if running
cd /root/autodl-tmp/Immunology_RAG
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/outputs/models/llm_finetuned_merged \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 --port 8000 &
sleep 60 && curl http://localhost:8000/v1/models
```

---

### `Connection refused: localhost:6000`

**Cause:** Semantic chunking service not running.

**Solution:**
```bash
cd /root/autodl-tmp/Immunology_RAG
python -m uvicorn src.server.semantic_chunk:app --host 0.0.0.0 --port 6000 &
sleep 12 && curl http://localhost:6000/health
```

---

### `MongoDB connection failed` / `child process failed, exited with 1`

**Cause:** Stale lock file from a previous crashed session, or MongoDB not installed.

**Solution:**
```bash
pkill mongod 2>/dev/null; sleep 2
mkdir -p /data/db && chmod 777 /data/db
rm -f /data/db/mongod.lock      # removes stale lock — the most common fix
mongod --fork --logpath /var/log/mongodb.log --dbpath /data/db
sleep 2 && mongosh --eval "db.runCommand({ping:1})"
# Expected: { ok: 1 }
```

If still failing, run without `--fork` to see the full error:
```bash
mongod --dbpath /data/db 2>&1 | head -30
```

---

### `E: Package 'mongodb' has no installation candidate`

**Cause:** Ubuntu 22.04 removed the old `mongodb` package. Use official MongoDB repo.

**Solution:** See Step 2.4 in this runbook (MongoDB 7.0 official APT repo setup).

---

### `LLM SFT completes with 0 training steps / no checkpoints`

**Cause:** Missing `do_train: true` in LLaMA-Factory config (old bug — fixed in current code).
With `do_train` absent, HuggingFace Trainer defaults to evaluation-only mode and exits
without training. Symptoms: no `checkpoint-N/` directories, `trainer_log.jsonl` has 1 entry.

**Solution (current code is fixed):**
```bash
# Verify fix is present:
grep "do_train" outputs/models/llm_finetuned/sft_config.yaml
# Must show: do_train: true

# If missing (old code), regenerate and retrain:
python -m train.train_llm_sft
```

---

### Reranker training produces NaN loss

**Cause:** `fp16=True` (old default) causes fp16 overflow with noisy OCR text.
Fixed in current code: `fp16=False` + logits cast to `.float()` before loss.

**Verify fix:**
```bash
grep "fp16\|logits.float" train/train_reranker.py | head -5
# Must show: fp16: bool = False
# Must show: loss_fn(logits.float(), labels.float())
```

---

### `AttributeError: module 'src.constant' has no attribute 'base_dir'`

**Cause:** Old code used lowercase `constant.base_dir`; `constant.py` uses `BASE_DIR`.
Fixed in all current files.

**If you see it in old scripts:**
```bash
sed -i 's/constant\.base_dir/constant.BASE_DIR/g' \
    evaluate.py pages/03_Evaluation.py pages/04_Settings.py
```

---

### `nltk.data.LookupError: punkt`

**Solution:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

### `ModuleNotFoundError: No module named 'flash_attn'`

**Solution A (recommended):** Keep `flash_attn: false` in `config.yaml` — not required.

**Solution B:** Install and enable:
```bash
pip install flash-attn --no-build-isolation
# In config.yaml: training.llm_sft.flash_attn: true
```

---

### `KeyError: 'from'` or `KeyError: 'messages'` (LLaMA-Factory)

**Cause:** Old `sft_train.jsonl` in wrong format. Fixed in current `build_train_data.py`
which generates native ShareGPT format with `messages` key and `from`/`value` fields.

**Solution:** Regenerate training data:
```bash
python -m train.build_train_data --workers 4
```

---

### RAGAS shows `NaN` or errors with `gpt-4o-mini`

**Cause:** RAGAS v0.2+ hardcodes `gpt-4o-mini` internally — it cannot be redirected to
a local vLLM endpoint regardless of any env var or LLM passed.

**Current behavior (by design):** RAGAS is gracefully skipped with a clear message.
All other metrics (Recall@K, MRR@10, ROUGE-L, BERTScore) are computed locally and
are not affected.

**If you have a real OpenAI API key:**
```bash
export OPENAI_API_KEY=sk-...
python evaluate.py   # RAGAS metrics will now compute successfully
```

---

### `OSError: models/Qwen3-8B is not a local folder`

**Cause:** Model not downloaded yet, or relative path passed to vLLM.

**Solution:**
```bash
# Download first (only needed once)
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B

# Always use absolute path: $(pwd)/models/Qwen3-8B
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/models/Qwen3-8B ...
```

---

### `LLaMA-Factory not found` / `ModuleNotFoundError: llamafactory`

**Solution:**
```bash
cd /root/autodl-tmp/Immunology_RAG
# On AutoDL use the mirror; on other clouds use the github.com URL directly
git clone https://gitclone.com/github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..
python -c "import llamafactory; print('OK')"
```

---

### `GET /health HTTP/1.1 404 Not Found` (semantic chunking service)

**Cause:** Old cached `.pyc` from before `/health` route was added.

**Solution:**
```bash
find src/server -name "*.pyc" -delete
pkill -f uvicorn; sleep 2
python -m uvicorn src.server.semantic_chunk:app --host 0.0.0.0 --port 6000 &
sleep 12 && curl http://localhost:6000/health   # should now return 200
```

---

## 11. Quick Reference: Service Ports

| Service | Port | Start Command | Notes |
|---------|------|---------------|-------|
| MongoDB | 27017 | `mongod --fork --logpath /var/log/mongodb.log --dbpath /data/db` | Always first |
| Semantic Chunking | 6000 | `uvicorn src.server.semantic_chunk:app --port 6000` | CPU only, optional for inference |
| vLLM (base model) | 8000 | `python -m vllm.entrypoints.openai.api_server --model $(pwd)/models/Qwen3-8B --served-model-name Qwen/Qwen3-8B ...` | For data gen only |
| vLLM (fine-tuned) | 8000 | `python -m vllm.entrypoints.openai.api_server --model $(pwd)/outputs/models/llm_finetuned_merged --served-model-name Qwen/Qwen3-8B ...` | For inference |
| Streamlit UI | 6006 | `streamlit run app.py --server.port 6006 --server.address 0.0.0.0` | After vLLM is ready |
| TensorBoard | 6007 | `tensorboard --logdir outputs/models/reranker_finetuned/runs --port 6007` | Training monitoring |

### GPU Conflict Summary

| Step | GPU needed by | vLLM state |
|------|--------------|-----------|
| Build index | BGE-M3 embedder (~2 GB) | Can run alongside |
| Generate training data | vLLM (Qwen3-8B, ~33 GB) | **Must be running** |
| Fine-tune reranker | PyTorch trainer (~8 GB) | **Must be killed first** |
| Fine-tune LLM SFT | LLaMA-Factory LoRA (~30 GB) | **Must be killed first** |
| Inference / UI | vLLM (fine-tuned, ~33 GB) | **Must be running** |

---

*This runbook is part of the ImmunoBiology RAG project documentation.
All commands verified on AutoDL A100-PCIE-40GB, PyTorch 2.1, CUDA 12.1, Ubuntu 22.04.*
