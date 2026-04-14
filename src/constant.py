# =============================================================================
# ImmunoBiology RAG Q&A System — Configuration Constants
# =============================================================================
# Single source of truth for all paths and model identifiers.
# All values are loaded from config.yaml at import time.
# Change config.yaml to update settings; do not hardcode paths here.
#
# Design pattern: mirrors Tesla RAG constant.py but reads from YAML
# instead of hardcoded strings, enabling portable deployment.

import os
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Load config.yaml — locate it relative to project root
# ---------------------------------------------------------------------------
_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.yaml"

if not _CONFIG_FILE.exists():
    raise FileNotFoundError(
        f"config.yaml not found at {_CONFIG_FILE}. "
        "Please ensure config.yaml is in the project root directory."
    )

with open(_CONFIG_FILE, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# Project root directory (parent of src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helper: resolve a config path string to an absolute Path
# ---------------------------------------------------------------------------
def _p(rel: str) -> str:
    """Convert a relative path string from config.yaml to an absolute path string."""
    return str(BASE_DIR / rel)


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
raw_dir            = _p(_cfg["data"]["raw_dir"])
processed_dir      = _p(_cfg["data"]["processed_dir"])
train_dir          = _p(_cfg["data"]["train_dir"])
stopwords_path     = _p(_cfg["data"]["stopwords_path"])

# Intermediate pickle caches (lazy-computation pattern from Tesla system)
processed_docs_dir  = _p("data/processed_docs")
raw_docs_path       = _p("data/processed_docs/raw_docs.pkl")
split_docs_path     = _p("data/processed_docs/split_docs.pkl")

# Training data files
reranker_train_path = _p("data/train/reranker_train.jsonl")
reranker_test_path  = _p("data/train/reranker_test.jsonl")
sft_train_path      = _p("data/train/sft_train.jsonl")
eval_qa_path        = _p("data/train/eval_qa.jsonl")

# Diagnostics
diagnostics_dir    = _p("outputs/diagnostics")
pdf_layout_report  = _p("outputs/diagnostics/pdf_layout_report.txt")
chunk_dist_plot    = _p("outputs/diagnostics/chunk_length_dist.png")
index_report       = _p("outputs/diagnostics/index_report.txt")

# ---------------------------------------------------------------------------
# Vector store paths
# ---------------------------------------------------------------------------
vectorstore_backend   = _cfg["vectorstore"]["backend"]
chroma_path           = _p(_cfg["vectorstore"]["chroma_path"])
chroma_collection     = _cfg["vectorstore"]["collection_name"]
bm25_pickle_path      = _p(_cfg["vectorstore"]["bm25_path"])
faiss_db_path         = _p(_cfg["vectorstore"]["faiss_path"])

# ---------------------------------------------------------------------------
# Model paths / identifiers
# ---------------------------------------------------------------------------
bge_m3_model_path          = _cfg["models"]["embedding"]
semantic_chunk_model_path  = _cfg["models"]["semantic_chunk_embedding"]
bge_reranker_model_path    = _cfg["models"]["reranker"]
bge_reranker_tuned_path    = _p(_cfg["models"]["reranker_finetuned"])
# reranker_use_finetuned: if True, BGEM3ReRanker loads from reranker_finetuned/best/.
# Toggle in config.yaml after retraining the reranker to compare base vs fine-tuned.
reranker_use_finetuned     = _cfg["models"].get("reranker_use_finetuned", False)
llm_model_name             = _cfg["models"]["llm"]
# llm_local_path: absolute path to local model directory for LLaMA-Factory training.
# LLaMA-Factory calls from_pretrained(path) — must be a real filesystem path.
# Falls back to "models/Qwen3-8B" if not set in config.yaml.
llm_local_path             = _p(_cfg["models"].get("llm_local_path", "models/Qwen3-8B"))
llm_finetuned_path         = _p(_cfg["models"]["llm_finetuned"])
eval_similarity_model      = _cfg["models"]["eval_similarity"]

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
bm25_topk       = _cfg["retrieval"]["bm25_topk"]
dense_topk      = _cfg["retrieval"]["dense_topk"]
rerank_topk     = _cfg["retrieval"]["rerank_topk"]
rrf_k           = _cfg["retrieval"]["rrf_k"]
rrf_bm25_weight = _cfg["retrieval"]["rrf_bm25_weight"]
rrf_dense_weight= _cfg["retrieval"]["rrf_dense_weight"]
hyde_enabled    = _cfg["retrieval"]["hyde_enabled"]

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
chunk_size          = _cfg["chunking"]["chunk_size"]
chunk_overlap       = _cfg["chunking"]["chunk_overlap"]
chunk_separators    = _cfg["chunking"]["separators"]
semantic_service_url= _cfg["chunking"]["semantic_service_url"]
semantic_group_size = _cfg["chunking"]["semantic_group_size"]
max_parent_size     = _cfg["chunking"]["max_parent_size"]

# ---------------------------------------------------------------------------
# MongoDB settings
# ---------------------------------------------------------------------------
mongo_host           = _cfg["mongodb"]["host"]
mongo_port           = _cfg["mongodb"]["port"]
mongo_db_name        = _cfg["mongodb"]["db_name"]
mongo_collection     = _cfg["mongodb"]["collection"]
mongo_max_pool_size  = _cfg["mongodb"]["max_pool_size"]

# ---------------------------------------------------------------------------
# LLM inference settings
# ---------------------------------------------------------------------------
vllm_base_url      = _cfg["llm"]["vllm_base_url"]
llm_max_tokens     = _cfg["llm"]["max_tokens"]
llm_temperature    = _cfg["llm"]["temperature"]
llm_top_p          = _cfg["llm"]["top_p"]
llm_freq_penalty   = _cfg["llm"]["frequency_penalty"]
llm_history_window = _cfg["llm"]["history_window"]

# ---------------------------------------------------------------------------
# LLM inference — Qwen3 / thinking mode
# ---------------------------------------------------------------------------
enable_thinking    = _cfg["llm"].get("enable_thinking", False)
# Qwen3 requires enable_thinking=False passed in extra_body to vLLM to suppress
# <think>...</think> chain-of-thought tokens in RAG responses.

# ---------------------------------------------------------------------------
# Training — reranker fine-tuning
# ---------------------------------------------------------------------------
reranker_epochs         = _cfg["training"]["reranker"]["epochs"]
reranker_batch_size     = _cfg["training"]["reranker"]["batch_size"]
reranker_lr             = _cfg["training"]["reranker"]["learning_rate"]
reranker_eval_steps     = _cfg["training"]["reranker"]["eval_steps"]
reranker_finetuned_path = _p(_cfg["training"]["reranker"]["output_dir"])

# LoRA v3 reranker config (use_lora: true in config.yaml enables PEFT LoRA)
reranker_use_lora            = _cfg["training"]["reranker"].get("use_lora", False)
reranker_lora_r              = _cfg["training"]["reranker"].get("lora_r", 16)
reranker_lora_alpha          = _cfg["training"]["reranker"].get("lora_alpha", 32)
reranker_lora_dropout        = _cfg["training"]["reranker"].get("lora_dropout", 0.05)
reranker_lora_target_modules = _cfg["training"]["reranker"].get(
    "lora_target_modules", ["query", "key", "value"]
)

# ---------------------------------------------------------------------------
# Training — LLM SFT (LoRA via LLaMA-Factory)
# ---------------------------------------------------------------------------
llm_sft_epochs      = _cfg["training"]["llm_sft"]["epochs"]
llm_sft_batch_size  = _cfg["training"]["llm_sft"]["batch_size"]
llm_sft_lr          = _cfg["training"]["llm_sft"]["learning_rate"]
llm_sft_bf16        = _cfg["training"]["llm_sft"]["bf16"]
llm_sft_flash_attn  = _cfg["training"]["llm_sft"]["flash_attn"]  # default False
lora_r              = _cfg["training"]["llm_sft"]["lora_r"]
lora_alpha          = _cfg["training"]["llm_sft"]["lora_alpha"]

# ---------------------------------------------------------------------------
# External training frameworks (subdirectories of project root)
# ---------------------------------------------------------------------------
llamafactory_dir   = _p(_cfg.get("external", {}).get("llamafactory_dir", "LLaMA-Factory"))
rag_retrieval_dir  = _p(_cfg.get("external", {}).get("rag_retrieval_dir", "RAG-Retrieval"))

# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------
eval_output_dir    = _p(_cfg["evaluation"]["output_dir"])
recall_k_values    = _cfg["evaluation"]["recall_k_values"]
quick_eval_n       = _cfg["evaluation"]["quick_eval_n"]

# ---------------------------------------------------------------------------
# Ensure critical output directories exist at import time
# ---------------------------------------------------------------------------
for _d in [processed_docs_dir, diagnostics_dir, chroma_path,
           str(Path(bm25_pickle_path).parent),
           train_dir, eval_output_dir,
           _p("outputs/reranker_eval"), _p("outputs/llm_eval"),
           _p("outputs/models/reranker_finetuned"),
           _p("outputs/models/llm_finetuned")]:
    os.makedirs(_d, exist_ok=True)
