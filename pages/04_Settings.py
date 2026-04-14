# =============================================================================
# ImmunoBiology RAG — Settings Page
# =============================================================================

import yaml
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Settings — ImmunoBiology RAG", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #880E4F 0%, #C2185B 60%, #E91E8C 100%);
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25) !important; }
[data-testid="stSidebarNav"] a { color: #FFE0EB !important; }
[data-testid="stSidebarNav"] [aria-selected="true"] {
    background: rgba(255,255,255,0.18) !important; border-radius: 8px !important;
}
h1 { color: #880E4F !important; font-weight: 700 !important; }
h2, h3 { color: #C2185B !important; }
.stButton > button {
    background: linear-gradient(135deg, #C2185B, #E91E8C) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
[data-testid="stMetric"] {
    background: #FDF0F5; border: 1px solid #F8BBD9; border-radius: 12px; padding: 1rem;
}
[data-testid="stExpander"] { border: 1px solid #F8BBD9 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #C2185B !important; font-weight: 600; }
[data-testid="stTabs"] [data-baseweb="tab"] { color: #C2185B !important; font-weight: 500; }
[data-testid="stTabs"] [aria-selected="true"] {
    border-bottom: 3px solid #C2185B !important;
    color: #880E4F !important; font-weight: 700 !important;
}
[data-testid="stNumberInput"] input { border-color: #F8BBD9 !important; }
[data-testid="stTextInput"] input { border-color: #F8BBD9 !important; }
hr { border-color: #F8BBD9 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
  <span style="font-size:2rem;">⚙️</span>
  <h1 style="margin:0; color:#880E4F !important;">Settings</h1>
</div>
<p style="color:#6D4C5E; margin-top:0; margin-bottom:1.5rem; font-size:0.95rem;">
  Tune retrieval, LLM, and chunking parameters. Changes are applied immediately — the pipeline
  reloads on the next query.
</p>
""", unsafe_allow_html=True)

from src import constant

config_path = Path(constant.BASE_DIR) / "config.yaml"


def load_config() -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


cfg = load_config()

# ---------------------------------------------------------------------------
# System info card — accurate model snapshot
# ---------------------------------------------------------------------------

_llm      = cfg.get("models", {}).get("llm", "—")
_embed    = cfg.get("models", {}).get("embedding", "—")
_reranker = cfg.get("models", {}).get("reranker", "—")
_vdb      = cfg.get("vectorstore", {}).get("backend", "chroma").capitalize()
_vllm_url = cfg.get("llm", {}).get("vllm_base_url", "http://localhost:8000/v1")

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #880E4F08, #C2185B05);
    border: 1px solid #F8BBD9;
    border-left: 5px solid #C2185B;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
">
  <div style="color:#880E4F; font-weight:700; font-size:0.95rem; margin-bottom:0.8rem;">
    🔬 Active System Configuration
  </div>
  <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem;">
    <div style="text-align:center; background:#fff; border:1px solid #F8BBD9; border-radius:8px; padding:0.7rem;">
      <div style="color:#6D4C5E; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px;">Embedding</div>
      <div style="color:#880E4F; font-weight:700; font-size:0.82rem; margin-top:3px;">{_embed}</div>
    </div>
    <div style="text-align:center; background:#fff; border:1px solid #F8BBD9; border-radius:8px; padding:0.7rem;">
      <div style="color:#6D4C5E; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px;">Reranker</div>
      <div style="color:#880E4F; font-weight:700; font-size:0.82rem; margin-top:3px;">{_reranker}</div>
    </div>
    <div style="text-align:center; background:#fff; border:1px solid #F8BBD9; border-radius:8px; padding:0.7rem;">
      <div style="color:#6D4C5E; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px;">LLM</div>
      <div style="color:#880E4F; font-weight:700; font-size:0.82rem; margin-top:3px;">{_llm}</div>
    </div>
    <div style="text-align:center; background:#fff; border:1px solid #F8BBD9; border-radius:8px; padding:0.7rem;">
      <div style="color:#6D4C5E; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px;">Vector DB</div>
      <div style="color:#880E4F; font-weight:700; font-size:0.82rem; margin-top:3px;">{_vdb}DB</div>
    </div>
  </div>
  <div style="margin-top:0.8rem; color:#6D4C5E; font-size:0.78rem;">
    🌐 vLLM endpoint: <code>{_vllm_url}</code>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabbed settings panels
# ---------------------------------------------------------------------------

tab_retrieval, tab_llm, tab_chunking, tab_models = st.tabs([
    "🔍 Retrieval", "🤖 LLM", "🧩 Chunking", "📦 Models"
])

# ── Retrieval tab ────────────────────────────────────────────────────────────
with tab_retrieval:
    st.markdown("""
<div style="background:#FDF0F5; border:1px solid #F8BBD9; border-radius:8px;
            padding:0.7rem 1rem; margin-bottom:1rem;">
  <span style="color:#C2185B; font-size:0.85rem; font-weight:600;">
    Adjust how many documents are fetched and how BM25 / dense retrieval are weighted.
  </span>
</div>
""", unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("**Top-K Controls**")
        bm25_topk   = st.number_input("BM25 Top-K",  value=int(cfg["retrieval"]["bm25_topk"]),
                                       min_value=1, max_value=50,
                                       help="Number of documents returned by BM25 sparse search")
        dense_topk  = st.number_input("Dense Top-K", value=int(cfg["retrieval"]["dense_topk"]),
                                       min_value=1, max_value=50,
                                       help="Number of documents returned by BGE-M3 dense search")
        rerank_topk = st.number_input("Rerank Top-K", value=int(cfg["retrieval"]["rerank_topk"]),
                                       min_value=1, max_value=20,
                                       help="Documents kept after cross-encoder reranking")

    with rc2:
        st.markdown("**RRF Fusion**")
        rrf_k = st.number_input("RRF k parameter", value=int(cfg["retrieval"]["rrf_k"]),
                                  min_value=1, max_value=200,
                                  help="Reciprocal Rank Fusion smoothing constant — higher = more uniform blending")
        rrf_bm25_w  = st.slider("BM25 Weight",  value=float(cfg["retrieval"]["rrf_bm25_weight"]),
                                 min_value=0.0, max_value=1.0, step=0.05,
                                 help="Weight applied to BM25 rankings in RRF fusion")
        rrf_dense_w = st.slider("Dense Weight", value=float(cfg["retrieval"]["rrf_dense_weight"]),
                                 min_value=0.0, max_value=1.0, step=0.05,
                                 help="Weight applied to dense (BGE-M3) rankings in RRF fusion")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    hyde_enabled = st.checkbox(
        "Enable HyDE Query Expansion",
        value=bool(cfg["retrieval"]["hyde_enabled"]),
        help="Generates a hypothetical document from the query before retrieval to improve recall"
    )

# ── LLM tab ──────────────────────────────────────────────────────────────────
with tab_llm:
    st.markdown("""
<div style="background:#FDF0F5; border:1px solid #F8BBD9; border-radius:8px;
            padding:0.7rem 1rem; margin-bottom:1rem;">
  <span style="color:#C2185B; font-size:0.85rem; font-weight:600;">
    Controls generation behaviour of the Qwen3-8B model served via vLLM.
  </span>
</div>
""", unsafe_allow_html=True)

    lc1, lc2 = st.columns(2)

    with lc1:
        st.markdown("**Generation**")
        max_tokens  = st.number_input("Max Tokens",
                                       value=int(cfg["llm"]["max_tokens"]),
                                       min_value=128, max_value=8192, step=128,
                                       help="Maximum tokens in the generated answer")
        temperature = st.slider("Temperature",
                                 value=float(cfg["llm"]["temperature"]),
                                 min_value=0.0, max_value=2.0, step=0.01,
                                 help="Sampling temperature — keep near 0 for factual RAG answers")
        history_window = st.number_input("History Window",
                                          value=int(cfg["llm"]["history_window"]),
                                          min_value=0, max_value=20,
                                          help="Number of prior turns kept in the conversation context")

    with lc2:
        st.markdown("**Sampling**")
        top_p = st.slider("Top-p (nucleus sampling)",
                           value=float(cfg["llm"].get("top_p", 0.95)),
                           min_value=0.0, max_value=1.0, step=0.01)
        freq_penalty = st.slider("Frequency Penalty",
                                  value=float(cfg["llm"].get("frequency_penalty", 0.0)),
                                  min_value=0.0, max_value=2.0, step=0.05,
                                  help="Penalises repetition — higher = less repetition")
        enable_thinking = st.checkbox(
            "Enable Thinking (Qwen3 CoT)",
            value=bool(cfg["llm"].get("enable_thinking", False)),
            help="Enable Qwen3's chain-of-thought reasoning — significantly slower, may improve complex questions"
        )

# ── Chunking tab ─────────────────────────────────────────────────────────────
with tab_chunking:
    st.markdown("""
<div style="background:#FDF0F5; border:1px solid #F8BBD9; border-radius:8px;
            padding:0.7rem 1rem; margin-bottom:1rem;">
  <span style="color:#C2185B; font-size:0.85rem; font-weight:600;">
    Chunking parameters affect document granularity. Changes take effect on the
    <em>next</em> index rebuild.
  </span>
</div>
""", unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown("**Chunk Sizing**")
        chunk_size    = st.number_input("Chunk Size (tokens)",
                                         value=int(cfg["chunking"]["chunk_size"]),
                                         min_value=64, max_value=2048, step=64,
                                         help="Target size for child chunks fed to the retriever")
        chunk_overlap = st.number_input("Chunk Overlap (tokens)",
                                         value=int(cfg["chunking"]["chunk_overlap"]),
                                         min_value=0, max_value=512, step=10,
                                         help="Token overlap between adjacent chunks to preserve context")

    with cc2:
        st.markdown("**Semantic Service**")
        sem_group_size = st.number_input("Semantic Group Size",
                                          value=int(cfg["chunking"]["semantic_group_size"]),
                                          min_value=3, max_value=50,
                                          help="Paragraph group size passed to the semantic chunking service (port 6000)")
        sem_url = st.text_input("Semantic Service URL",
                                 value=cfg["chunking"]["semantic_service_url"],
                                 help="FastAPI endpoint for MiniLM-based semantic boundary detection")

    st.markdown("""
<div style="background:#E8F5E9; border:1px solid #A5D6A7; border-radius:8px;
            padding:0.7rem 1rem; margin-top:0.6rem;">
  <span style="color:#2E7D32; font-size:0.83rem;">
    ℹ️ Chunking changes require re-running <code>python build_index.py</code> to take effect.
  </span>
</div>
""", unsafe_allow_html=True)

# ── Models tab ───────────────────────────────────────────────────────────────
with tab_models:
    st.markdown("""
<div style="background:#FDF0F5; border:1px solid #F8BBD9; border-radius:8px;
            padding:0.7rem 1rem; margin-bottom:1rem;">
  <span style="color:#C2185B; font-size:0.85rem; font-weight:600;">
    Model paths are read-only here — edit <code>config.yaml</code> directly to swap models.
  </span>
</div>
""", unsafe_allow_html=True)

    MODEL_INFO = [
        ("🔍 Embedding",         cfg["models"]["embedding"],
         "BGE-M3 — multilingual dense + sparse vectors (1024-dim). Top MTEB English performance."),
        ("🏆 Reranker",          cfg["models"]["reranker"],
         "BGE Reranker v2-M3 — cross-encoder, top BEIR English benchmark. Fine-tuned on immunology pairs."),
        ("🧠 LLM",               cfg["models"]["llm"],
         "Qwen3-8B — served via vLLM. LoRA fine-tuned on immunology SFT dataset. CoT disabled for RAG."),
        ("🔬 Semantic Chunker",   cfg["models"]["semantic_chunk_embedding"],
         "all-MiniLM-L6-v2 — lightweight 384-dim model for semantic boundary detection. Runs on CPU."),
        ("📦 Vector Store",      cfg["vectorstore"]["backend"].upper() + "DB",
         "Persistent Chroma collection at outputs/vectorstore/chroma — metadata filtering by doc_type and source_file."),
    ]

    for icon_label, model_id, description in MODEL_INFO:
        st.markdown(f"""
<div style="
    background:#FAFAFA;
    border:1px solid #F8BBD9;
    border-left:4px solid #C2185B;
    border-radius:8px;
    padding:0.9rem 1.1rem;
    margin-bottom:0.7rem;
    display:flex; align-items:flex-start; gap:12px;
">
  <div style="flex:1;">
    <div style="color:#880E4F; font-weight:700; font-size:0.88rem;">{icon_label}</div>
    <div style="
        display:inline-block;
        background:#FCE4EC; color:#C2185B;
        border:1px solid #F8BBD9; border-radius:6px;
        padding:2px 10px; font-size:0.82rem; font-weight:600;
        margin:4px 0 6px;
        font-family: monospace;
    ">{model_id}</div>
    <div style="color:#6D4C5E; font-size:0.8rem; line-height:1.4;">{description}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Fine-tuned paths (show existence status)
    ft_reranker = Path(constant.BASE_DIR) / cfg["models"]["reranker_finetuned"]
    ft_llm      = Path(constant.BASE_DIR) / cfg["models"]["llm_finetuned"]

    st.markdown("**Fine-tuned Checkpoints**")
    for label, fpath in [("Reranker (fine-tuned)", ft_reranker), ("LLM (fine-tuned)", ft_llm)]:
        exists = fpath.exists()
        color  = "#2E7D32" if exists else "#B71C1C"
        icon   = "✅" if exists else "❌"
        st.markdown(f"""
<div style="background:#{'E8F5E9' if exists else 'FFEBEE'}; border:1px solid #{'A5D6A7' if exists else 'EF9A9A'};
            border-radius:6px; padding:0.5rem 0.8rem; margin-bottom:0.4rem; font-size:0.83rem;">
  <span style="color:{color};">{icon} <strong>{label}</strong></span>
  <code style="color:{color}; margin-left:8px; font-size:0.78rem;">{fpath}</code>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Apply button
# ---------------------------------------------------------------------------

st.markdown("<hr/>", unsafe_allow_html=True)

apply_col, reset_col = st.columns([3, 1])
with apply_col:
    apply_btn = st.button("💾 Apply Settings", type="primary", use_container_width=True)
with reset_col:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.caption("Pipeline reloads on next query.")

if apply_btn:
    # Retrieval
    cfg["retrieval"]["bm25_topk"]        = int(bm25_topk)
    cfg["retrieval"]["dense_topk"]       = int(dense_topk)
    cfg["retrieval"]["rerank_topk"]      = int(rerank_topk)
    cfg["retrieval"]["rrf_k"]            = int(rrf_k)
    cfg["retrieval"]["rrf_bm25_weight"]  = float(rrf_bm25_w)
    cfg["retrieval"]["rrf_dense_weight"] = float(rrf_dense_w)
    cfg["retrieval"]["hyde_enabled"]     = bool(hyde_enabled)

    # LLM
    cfg["llm"]["max_tokens"]        = int(max_tokens)
    cfg["llm"]["temperature"]       = float(temperature)
    cfg["llm"]["top_p"]             = float(top_p)
    cfg["llm"]["frequency_penalty"] = float(freq_penalty)
    cfg["llm"]["history_window"]    = int(history_window)
    cfg["llm"]["enable_thinking"]   = bool(enable_thinking)

    # Chunking
    cfg["chunking"]["chunk_size"]           = int(chunk_size)
    cfg["chunking"]["chunk_overlap"]        = int(chunk_overlap)
    cfg["chunking"]["semantic_group_size"]  = int(sem_group_size)
    cfg["chunking"]["semantic_service_url"] = sem_url

    save_config(cfg)

    # Hot-reload constants
    import importlib
    from src import constant as const_mod
    importlib.reload(const_mod)

    # Evict cached pipeline so next query picks up new settings
    if "pipeline" in st.session_state:
        del st.session_state["pipeline"]

    st.success("✅ Settings saved and applied. The RAG pipeline will reload on the next query.")

# ---------------------------------------------------------------------------
# Raw config viewer
# ---------------------------------------------------------------------------

st.markdown("<hr/>", unsafe_allow_html=True)

with st.expander("📄 Raw config.yaml"):
    st.code(config_path.read_text(encoding="utf-8"), language="yaml")
