# =============================================================================
# ImmunoBiology RAG — Streamlit Entry Point
# =============================================================================

import streamlit as st

st.set_page_config(
    page_title="ImmunoBiology RAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global theme CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #FFFFFF; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #880E4F 0%, #C2185B 60%, #E91E8C 100%);
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label { color: #FCE4EC !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.25) !important; }

/* ── Nav links in sidebar ── */
[data-testid="stSidebarNav"] a { color: #FFE0EB !important; font-weight: 500; }
[data-testid="stSidebarNav"] a:hover { color: #FFFFFF !important; }
[data-testid="stSidebarNav"] [aria-selected="true"] {
    background: rgba(255,255,255,0.18) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}

/* ── Headings ── */
h1 { color: #880E4F !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { color: #C2185B !important; font-weight: 600 !important; }
h3 { color: #AD1457 !important; }

/* ── Primary buttons ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #C2185B, #E91E8C) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #FDF0F5;
    border: 1px solid #F8BBD9;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #C2185B !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #880E4F !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #F8BBD9 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #C2185B !important; font-weight: 600; }

/* ── Chat messages ── */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #FDF0F5 !important;
}
[data-testid="stChatMessageContent"] { color: #1A1A2E !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { color: #C2185B !important; font-weight: 500; }
.stTabs [aria-selected="true"] {
    border-bottom: 3px solid #C2185B !important;
    color: #880E4F !important;
    font-weight: 700 !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] thead th {
    background: #FCE4EC !important;
    color: #880E4F !important;
    font-weight: 700 !important;
}

/* ── Info / success / warning boxes ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: #F8BBD9 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: #C2185B !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

ANTIBODY_SVG = """
<svg width="72" height="72" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <line x1="100" y1="115" x2="100" y2="178" stroke="#C2185B" stroke-width="10" stroke-linecap="round"/>
  <line x1="58"  y1="78"  x2="100" y2="115" stroke="#C2185B" stroke-width="10" stroke-linecap="round"/>
  <line x1="142" y1="78"  x2="100" y2="115" stroke="#C2185B" stroke-width="10" stroke-linecap="round"/>
  <line x1="28"  y1="50"  x2="58"  y2="78"  stroke="#E91E8C" stroke-width="7" stroke-linecap="round"/>
  <line x1="36"  y1="26"  x2="58"  y2="78"  stroke="#9C27B0" stroke-width="7" stroke-linecap="round"/>
  <line x1="172" y1="50"  x2="142" y2="78"  stroke="#E91E8C" stroke-width="7" stroke-linecap="round"/>
  <line x1="164" y1="26"  x2="142" y2="78"  stroke="#9C27B0" stroke-width="7" stroke-linecap="round"/>
  <circle cx="28"  cy="50"  r="10" fill="#F48FB1" opacity="0.9"/>
  <circle cx="36"  cy="26"  r="10" fill="#F48FB1" opacity="0.9"/>
  <circle cx="172" cy="50"  r="10" fill="#F48FB1" opacity="0.9"/>
  <circle cx="164" cy="26"  r="10" fill="#F48FB1" opacity="0.9"/>
  <circle cx="100" cy="178" r="8"  fill="#CE93D8" opacity="0.8"/>
</svg>
"""

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #880E4F 0%, #C2185B 50%, #E91E8C 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    box-shadow: 0 8px 32px rgba(194,24,91,0.25);
">
  <div>{ANTIBODY_SVG}</div>
  <div>
    <div style="color:#FFFFFF; margin:0; font-size:2.2rem; font-weight:800; letter-spacing:-1px; line-height:1.1;">
      ImmunoBiology RAG
    </div>
    <p style="color:#FCE4EC; margin:0.4rem 0 0; font-size:1.05rem; font-weight:400;">
      Retrieval-Augmented Generation for Immunology Research
    </p>
    <div style="margin-top:0.8rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
      <span style="background:rgba(255,255,255,0.2); color:#fff; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.78rem; font-weight:600;">Qwen3-8B</span>
      <span style="background:rgba(255,255,255,0.2); color:#fff; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.78rem; font-weight:600;">BGE-M3</span>
      <span style="background:rgba(255,255,255,0.2); color:#fff; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.78rem; font-weight:600;">BGE Reranker v2-M3</span>
      <span style="background:rgba(255,255,255,0.2); color:#fff; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.78rem; font-weight:600;">ChromaDB</span>
      <span style="background:rgba(255,255,255,0.2); color:#fff; padding:0.25rem 0.75rem; border-radius:20px; font-size:0.78rem; font-weight:600;">LoRA Fine-Tuned</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline diagram
# ---------------------------------------------------------------------------

st.markdown("### RAG Pipeline")

STEPS = [
    ("🔍", "Query",       "User immunology question"),
    ("💡", "HyDE",        "Hypothetical document expansion"),
    ("📚", "Retrieval",   "BM25 + Dense hybrid search"),
    ("🔀", "RRF Fusion",  "Reciprocal rank fusion"),
    ("🏆", "Reranking",   "Cross-encoder reranker"),
    ("🤖", "Generation",  "Qwen3-8B LLM answer"),
    ("📎", "Citations",   "Inline refs + figures"),
]

cols = st.columns(len(STEPS))
for col, (icon, title, desc) in zip(cols, STEPS):
    with col:
        st.markdown(f"""
<div style="
    text-align:center;
    background:#FDF0F5;
    border:1px solid #F8BBD9;
    border-radius:12px;
    padding:0.9rem 0.4rem;
    height:110px;
    display:flex; flex-direction:column; justify-content:center;
">
  <div style="font-size:1.6rem;">{icon}</div>
  <div style="color:#C2185B; font-weight:700; font-size:0.82rem; margin-top:4px;">{title}</div>
  <div style="color:#6D4C5E; font-size:0.72rem; margin-top:2px; line-height:1.2;">{desc}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Architecture cards
# ---------------------------------------------------------------------------

st.markdown("### System Architecture")

ARCH = [
    ("🗂️",  "Knowledge Base",   "Janeway's Immunobiology 10e + research papers parsed with PyMuPDF",    "#FCE4EC"),
    ("🧩",  "Chunking",         "Semantic parent-child chunking via MiniLM clustering + MongoDB storage", "#F3E5F5"),
    ("🔎",  "Hybrid Retrieval", "BGE-M3 dense vectors (ChromaDB) + BM25 sparse index with RRF fusion",   "#E8EAF6"),
    ("⚡",  "Reranker",         "Fine-tuned BAAI/bge-reranker-v2-m3 cross-encoder (NDCG@10 = 1.0)",      "#E0F2F1"),
    ("🧠",  "LLM",              "LoRA fine-tuned Qwen3-8B via vLLM (ROUGE-L = 0.55, BERTScore = 0.92)", "#FFF3E0"),
    ("📊",  "Evaluation",       "Recall@K, MRR@10, ROUGE-L, BERTScore across 150 QA pairs",              "#FCE4EC"),
]

col1, col2, col3 = st.columns(3)
for i, (icon, title, desc, bg) in enumerate(ARCH):
    col = [col1, col2, col3][i % 3]
    with col:
        st.markdown(f"""
<div style="
    background:{bg};
    border-radius:12px;
    padding:1.2rem;
    margin-bottom:1rem;
    border-left:4px solid #C2185B;
    min-height:100px;
">
  <div style="font-size:1.5rem; margin-bottom:6px;">{icon}</div>
  <div style="color:#880E4F; font-weight:700; font-size:0.95rem;">{title}</div>
  <div style="color:#4A235A; font-size:0.82rem; margin-top:4px; line-height:1.4;">{desc}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Quick start
# ---------------------------------------------------------------------------

st.markdown("### Quick Start")
st.markdown("""
<div style="background:#FDF0F5; border:1px solid #F8BBD9; border-radius:12px; padding:1.5rem;">
<ol style="color:#1A1A2E; margin:0; padding-left:1.4rem; line-height:2;">
  <li>Navigate to <strong style="color:#C2185B;">Q&A</strong> in the sidebar to ask immunology questions</li>
  <li>View system performance on the <strong style="color:#C2185B;">Evaluation</strong> page</li>
  <li>Manage indexed documents on the <strong style="color:#C2185B;">Documents</strong> page</li>
  <li>Tune retrieval &amp; LLM parameters on the <strong style="color:#C2185B;">Settings</strong> page</li>
</ol>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"""
<div style="text-align:center; padding:1rem 0 0.5rem;">
  {ANTIBODY_SVG}
  <div style="font-weight:700; font-size:1.1rem; margin-top:0.5rem;">ImmunoBiology RAG</div>
  <div style="font-size:0.78rem; opacity:0.8; margin-top:2px;">Powered by Qwen3-8B</div>
</div>
<hr style="border-color:rgba(255,255,255,0.25); margin:0.8rem 0;"/>
""", unsafe_allow_html=True)
    st.markdown("**Navigate**")
    st.markdown("""
- 💬 **Q&A** — Ask questions
- 📄 **Documents** — Manage PDFs
- 📊 **Evaluation** — View metrics
- ⚙️ **Settings** — Configure system
""")
