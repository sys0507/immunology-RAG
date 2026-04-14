# =============================================================================
# ImmunoBiology RAG — Document Management Page
# =============================================================================

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Documents — ImmunoBiology RAG", page_icon="📄", layout="wide")

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
hr { border-color: #F8BBD9 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
  <span style="font-size:2rem;">📄</span>
  <h1 style="margin:0; color:#880E4F !important;">Document Management</h1>
</div>
<p style="color:#6D4C5E; margin-top:0; margin-bottom:1.5rem; font-size:0.95rem;">
  Browse indexed documents, upload new PDFs, and view extraction diagnostics.
</p>
""", unsafe_allow_html=True)

from src import constant
from src.client.mongodb_config import MongoConfig

# ---------------------------------------------------------------------------
# Stats metrics
# ---------------------------------------------------------------------------

collection = MongoConfig.get_collection(constant.mongo_collection)

pipeline_agg = [
    {"$group": {
        "_id":         "$metadata.source_file",
        "chunk_count": {"$sum": 1},
        "pages":       {"$addToSet": "$metadata.page"},
        "chapters":    {"$addToSet": "$metadata.chapter"},
        "doc_type":    {"$first": "$metadata.doc_type"},
    }},
    {"$sort": {"_id": 1}},
]

try:
    docs_info = list(collection.aggregate(pipeline_agg))
except Exception as e:
    st.warning(f"Could not query MongoDB: {e}")
    docs_info = []

total_chunks = sum(d.get("chunk_count", 0) for d in docs_info)
total_pages  = sum(len([p for p in d.get("pages", []) if p]) for d in docs_info)
total_docs   = len(docs_info)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("📚 Documents Indexed", total_docs)
with m2:
    st.metric("🧩 Total Chunks", total_chunks)
with m3:
    st.metric("📃 Pages Covered", total_pages)

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Document cards
# ---------------------------------------------------------------------------

st.markdown("### Indexed Documents")

if docs_info:
    for info in docs_info:
        name     = info.get("_id", "Unknown") or "Unknown"
        dtype    = info.get("doc_type", "—") or "—"
        chunks   = info.get("chunk_count", 0)
        n_pages  = len([p for p in info.get("pages", []) if p])
        n_chaps  = len([c for c in info.get("chapters", []) if c])
        type_color = "#C2185B" if dtype == "textbook" else "#7B1FA2"
        icon       = "📖" if dtype == "textbook" else "📰"

        st.markdown(f"""
<div style="
    background:#FDF0F5;
    border:1px solid #F8BBD9;
    border-left:5px solid {type_color};
    border-radius:10px;
    padding:1rem 1.2rem;
    margin-bottom:0.8rem;
    display:flex; align-items:center; justify-content:space-between;
">
  <div style="display:flex; align-items:center; gap:12px;">
    <span style="font-size:1.6rem;">{icon}</span>
    <div>
      <div style="color:#1A1A2E; font-weight:700; font-size:0.95rem;">{name}</div>
      <div style="margin-top:4px;">
        <span style="background:{type_color}22; color:{type_color}; border:1px solid {type_color}44;
               padding:2px 8px; border-radius:20px; font-size:0.75rem; font-weight:600;">{dtype}</span>
      </div>
    </div>
  </div>
  <div style="display:flex; gap:2rem; text-align:center;">
    <div><div style="color:#C2185B; font-weight:700; font-size:1.2rem;">{chunks}</div>
         <div style="color:#6D4C5E; font-size:0.75rem;">chunks</div></div>
    <div><div style="color:#C2185B; font-weight:700; font-size:1.2rem;">{n_pages}</div>
         <div style="color:#6D4C5E; font-size:0.75rem;">pages</div></div>
    <div><div style="color:#C2185B; font-weight:700; font-size:1.2rem;">{n_chaps}</div>
         <div style="color:#6D4C5E; font-size:0.75rem;">chapters</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div style="background:#FDF0F5; border:1px solid #F8BBD9; border-radius:12px; padding:2rem; text-align:center;">
  <div style="font-size:2.5rem; margin-bottom:0.5rem;">📭</div>
  <div style="color:#C2185B; font-weight:700;">No documents indexed yet</div>
  <div style="color:#6D4C5E; font-size:0.88rem; margin-top:4px;">Run <code>python build_index.py</code> to index PDFs from <code>data/raw/</code></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("### Upload New Document")

st.markdown("""
<div style="background:#F3E5F5; border:1px dashed #CE93D8; border-radius:10px; padding:0.8rem 1.2rem; margin-bottom:0.8rem;">
  <span style="color:#7B1FA2; font-size:0.88rem;">
    📤 Upload a PDF to add it to the knowledge base. It will be saved to <code>data/raw/</code>.
  </span>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded is not None:
    save_path = Path(constant.raw_dir) / uploaded.name
    if save_path.exists():
        st.warning(f"`{uploaded.name}` already exists in data/raw/.")
    else:
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ Saved `{uploaded.name}` to `{save_path}`")

    if st.button("⚡ Index this document now", type="primary"):
        with st.spinner(f"Indexing {uploaded.name}…"):
            try:
                from build_index import build_index
                build_index(pdf_path=str(save_path))
                st.success(f"✅ Successfully indexed: {uploaded.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Indexing failed: {e}")

# ---------------------------------------------------------------------------
# Extraction report
# ---------------------------------------------------------------------------

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("### Extraction Report")

report_path = Path(constant.processed_dir) / "extraction_report.json"
if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    with st.expander("View extraction_report.json"):
        st.json(report)
else:
    st.info("No extraction report found. Run PDF parsing first.")

# ---------------------------------------------------------------------------
# Index diagnostics
# ---------------------------------------------------------------------------

st.markdown("### Index Diagnostics")

diag_dir = Path(constant.diagnostics_dir)
for diag_file in ["pdf_layout_report.txt", "index_report.txt"]:
    fpath = diag_dir / diag_file
    if fpath.exists():
        with st.expander(f"📋 {diag_file}"):
            st.code(fpath.read_text(encoding="utf-8"), language="text")

chunk_hist = diag_dir / "chunk_length_dist.png"
if chunk_hist.exists():
    with st.expander("📊 Chunk Length Distribution"):
        st.image(str(chunk_hist), use_column_width=True)
