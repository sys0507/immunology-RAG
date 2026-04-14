# =============================================================================
# ImmunoBiology RAG — Evaluation Page
# =============================================================================

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Evaluation — ImmunoBiology RAG", page_icon="📊", layout="wide")

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
[data-testid="stMetricValue"] { color: #C2185B !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #880E4F !important; }
[data-testid="stExpander"] { border: 1px solid #F8BBD9 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #C2185B !important; font-weight: 600; }
[data-testid="stDataFrame"] thead th {
    background: #FCE4EC !important; color: #880E4F !important; font-weight: 700 !important;
}
[data-testid="stSlider"] > div > div > div > div { background: #C2185B !important; }
hr { border-color: #F8BBD9 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
  <span style="font-size:2rem;">📊</span>
  <h1 style="margin:0; color:#880E4F !important;">System Evaluation</h1>
</div>
<p style="color:#6D4C5E; margin-top:0; margin-bottom:1.5rem; font-size:0.95rem;">
  Retrieval, reranker, and generation metrics — computed over 150 immunology QA pairs.
</p>
""", unsafe_allow_html=True)

from src import constant

eval_dir  = Path(constant.BASE_DIR) / "outputs" / "system_eval"
eval_path = Path(constant.BASE_DIR) / "outputs" / "system_eval" / "metrics.json"

# ---------------------------------------------------------------------------
# Load cached metrics if available
# ---------------------------------------------------------------------------

cached_metrics: dict = {}
if eval_path.exists():
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            cached_metrics = json.load(f)
    except Exception:
        cached_metrics = {}

# ---------------------------------------------------------------------------
# Top metric score-cards
# ---------------------------------------------------------------------------

st.markdown("### Key Performance Indicators")

METRIC_DEFS = [
    ("Recall@1",      "recall@1",       "retrieval", "#C2185B"),
    ("Recall@5",      "recall@5",       "retrieval", "#AD1457"),
    ("MRR@10",        "mrr@10",         "retrieval", "#880E4F"),
    ("ROUGE-L",       "rouge_l",        "generation","#7B1FA2"),
    ("BERTScore F1",  "bertscore_f1",   "generation","#6A1B9A"),
]

def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
        return f"{fv:.3f}"
    except Exception:
        return str(v)

def _pct(v) -> str:
    """Convert 0–1 float to percentage string."""
    if v is None:
        return ""
    try:
        return f"{float(v)*100:.1f}%"
    except Exception:
        return ""

m_cols = st.columns(5)
for col, (label, key, group, color) in zip(m_cols, METRIC_DEFS):
    raw = cached_metrics.get(group, {}).get(key) or cached_metrics.get(key)
    val = _fmt(raw)
    pct = _pct(raw)
    with col:
        st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {color}15, {color}08);
    border: 1px solid {color}44;
    border-top: 4px solid {color};
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
">
  <div style="color:{color}; font-weight:800; font-size:1.6rem; line-height:1;">{val}</div>
  {'<div style="color:' + color + '; font-size:0.75rem; font-weight:600; opacity:0.75;">' + pct + '</div>' if pct else ''}
  <div style="color:#6D4C5E; font-size:0.78rem; margin-top:6px; font-weight:600;">{label}</div>
</div>
""", unsafe_allow_html=True)

if not cached_metrics:
    st.markdown("""
<div style="background:#FFF8E1; border:1px solid #FFE082; border-radius:10px; padding:0.9rem 1.2rem; margin-top:0.8rem;">
  <span style="color:#E65100; font-size:0.88rem;">
    ⚠️ No cached metrics found. Run <code>python evaluate.py</code> to generate results,
    or use the Quick Evaluation below.
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Individual charts — 2-column grid
# ---------------------------------------------------------------------------

st.markdown("### Evaluation Charts")

CHARTS = [
    ("🎯 Retrieval Recall@K",   "retrieval_recall.png"),
    ("🏆 Reranker Precision",   "reranker_precision.png"),
    ("✍️ Generation Quality",   "generation_quality.png"),
    ("🕸️ End-to-End Radar",     "e2e_radar.png"),
    ("⏱️ Latency Breakdown",    "latency_breakdown.png"),
]

available = [(t, fn) for t, fn in CHARTS if (eval_dir / fn).exists()]
missing   = [(t, fn) for t, fn in CHARTS if not (eval_dir / fn).exists()]

if available:
    for i in range(0, len(available), 2):
        row = available[i:i+2]
        cols = st.columns(len(row))
        for col, (title, fname) in zip(cols, row):
            with col:
                st.markdown(f"""
<div style="background:#FDF0F5; border:1px solid #F8BBD9; border-radius:10px;
            padding:0.6rem 0.8rem; margin-bottom:0.5rem;">
  <div style="color:#C2185B; font-weight:700; font-size:0.9rem;">{title}</div>
</div>
""", unsafe_allow_html=True)
                st.image(str(eval_dir / fname), use_column_width=True)
else:
    st.markdown("""
<div style="background:#FDF0F5; border:1px dashed #F8BBD9; border-radius:12px; padding:2rem; text-align:center;">
  <div style="font-size:2rem; margin-bottom:0.5rem;">📈</div>
  <div style="color:#C2185B; font-weight:700;">No charts generated yet</div>
  <div style="color:#6D4C5E; font-size:0.85rem; margin-top:4px;">
    Run <code>python evaluate.py</code> to produce retrieval, reranker, and generation charts.
  </div>
</div>
""", unsafe_allow_html=True)

if missing:
    with st.expander(f"⚠️ {len(missing)} chart(s) not yet available"):
        for title, fname in missing:
            st.markdown(f"- {title} (`{fname}`)")

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Fine-tuning comparison cards
# ---------------------------------------------------------------------------

st.markdown("### Fine-Tuning Comparisons")

ft_cols = st.columns(2)

for col, (name, subdir) in zip(ft_cols, [("🔀 Reranker", "reranker_eval"), ("🧠 LLM SFT", "llm_eval")]):
    comp_dir = Path(constant.BASE_DIR) / "outputs" / subdir
    comp_img = comp_dir / "comparison.png"
    comp_csv = comp_dir / "comparison.csv"

    with col:
        st.markdown(f"""
<div style="background:#FDF0F5; border:1px solid #F8BBD9; border-top:4px solid #C2185B;
            border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.5rem;">
  <div style="color:#880E4F; font-weight:700; font-size:0.95rem;">{name}</div>
</div>
""", unsafe_allow_html=True)
        if comp_img.exists():
            st.image(str(comp_img), use_column_width=True)
        if comp_csv.exists():
            import pandas as pd
            df = pd.read_csv(str(comp_csv))
            st.dataframe(df, use_container_width=True)
        if not comp_img.exists() and not comp_csv.exists():
            st.markdown("""
<div style="background:#F3E5F5; border:1px dashed #CE93D8; border-radius:8px;
            padding:1.2rem; text-align:center;">
  <div style="color:#7B1FA2; font-size:0.85rem;">
    Fine-tuning comparison not available yet.<br/>
    Run the training pipeline first.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Quick evaluation runner
# ---------------------------------------------------------------------------

st.markdown("### Quick Evaluation")

st.markdown("""
<div style="background:#F3E5F5; border:1px dashed #CE93D8; border-radius:10px;
            padding:0.8rem 1.2rem; margin-bottom:0.8rem;">
  <span style="color:#7B1FA2; font-size:0.88rem;">
    ⚡ Runs a fast evaluation on a random sample from <code>data/train/eval_qa.jsonl</code>.
    Full evaluation: <code>python evaluate.py</code>
  </span>
</div>
""", unsafe_allow_html=True)

eval_qa_path = Path(constant.train_dir) / "eval_qa.jsonl"

if not eval_qa_path.exists():
    st.warning("No eval QA set found. Run `python -m train.build_train_data` first.")
else:
    # Count available pairs
    try:
        n_available = sum(1 for _ in open(eval_qa_path, encoding="utf-8"))
    except Exception:
        n_available = 50

    q_col1, q_col2 = st.columns([2, 1])
    with q_col1:
        n_pairs = st.slider("QA pairs to evaluate", min_value=5, max_value=min(50, n_available),
                            value=min(20, n_available), step=5)
    with q_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("⚡ Run Quick Eval", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Evaluating {n_pairs} QA pairs…"):
            try:
                from evaluate import run_evaluation
                run_evaluation(
                    eval_qa_file=str(eval_qa_path),
                    output_dir=str(eval_dir),
                    quick=True,
                    n_eval=n_pairs,
                )
                st.success("✅ Evaluation complete! Refresh to see updated charts and metrics.")
                st.rerun()
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Full HTML report embed
# ---------------------------------------------------------------------------

st.markdown("### Full Evaluation Report")

report_path = eval_dir / "evaluation_report.html"
if report_path.exists():
    with st.expander("📋 View HTML Report", expanded=False):
        st.components.v1.html(
            report_path.read_text(encoding="utf-8"),
            height=1200,
            scrolling=True,
        )
else:
    st.info("No HTML report found. Run `python evaluate.py` to generate the full report.")
