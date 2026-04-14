# =============================================================================
# ImmunoBiology RAG — Q&A Page
# =============================================================================

import streamlit as st

st.set_page_config(page_title="Q&A — ImmunoBiology RAG", page_icon="💬", layout="wide")

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
[data-testid="stExpander"] { border: 1px solid #F8BBD9 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { color: #C2185B !important; font-weight: 600; }
hr { border-color: #F8BBD9 !important; }
/* Chat input */
[data-testid="stChatInput"] textarea {
    border: 2px solid #F8BBD9 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #C2185B !important;
    box-shadow: 0 0 0 3px rgba(194,24,91,0.15) !important;
}
/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #FDF0F5 !important;
    border-radius: 12px !important;
    border-left: 3px solid #C2185B !important;
}
</style>
""", unsafe_allow_html=True)

ANTIBODY_MINI = """<svg width="28" height="28" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <line x1="100" y1="115" x2="100" y2="178" stroke="#C2185B" stroke-width="14" stroke-linecap="round"/>
  <line x1="58" y1="78" x2="100" y2="115" stroke="#C2185B" stroke-width="14" stroke-linecap="round"/>
  <line x1="142" y1="78" x2="100" y2="115" stroke="#C2185B" stroke-width="14" stroke-linecap="round"/>
  <line x1="28" y1="50" x2="58" y2="78" stroke="#E91E8C" stroke-width="10" stroke-linecap="round"/>
  <line x1="36" y1="26" x2="58" y2="78" stroke="#9C27B0" stroke-width="10" stroke-linecap="round"/>
  <line x1="172" y1="50" x2="142" y2="78" stroke="#E91E8C" stroke-width="10" stroke-linecap="round"/>
  <line x1="164" y1="26" x2="142" y2="78" stroke="#9C27B0" stroke-width="10" stroke-linecap="round"/>
  <circle cx="28" cy="50" r="12" fill="#F48FB1"/>
  <circle cx="36" cy="26" r="12" fill="#F48FB1"/>
  <circle cx="172" cy="50" r="12" fill="#F48FB1"/>
  <circle cx="164" cy="26" r="12" fill="#F48FB1"/>
</svg>"""

st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
  {ANTIBODY_MINI}
  <h1 style="margin:0; color:#880E4F !important;">Immunology Q&A</h1>
</div>
<p style="color:#6D4C5E; margin-top:0; margin-bottom:1.5rem; font-size:0.95rem;">
  Ask questions about immunology — answers are grounded in indexed textbooks and papers with inline citations.
</p>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _get_pipeline():
    if "pipeline" not in st.session_state:
        with st.spinner("Loading RAG pipeline…"):
            from src.pipeline import RAGPipeline
            st.session_state.pipeline = RAGPipeline()
    return st.session_state.pipeline


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
<div style="text-align:center; padding:0.5rem 0 1rem;">
  <div style="font-size:1.8rem;">💬</div>
  <div style="font-weight:700; font-size:1rem;">Q&A Settings</div>
</div>
<hr/>
""", unsafe_allow_html=True)

    doc_type      = st.selectbox("Filter by document type", ["All", "textbook", "paper"])
    source_filter = st.text_input("Filter by source file", "", placeholder="e.g. janeway10e.pdf")
    hyde_toggle   = st.checkbox("Enable HyDE query expansion", value=False,
                                help="Generates a hypothetical answer to improve retrieval")

    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_messages = []
        _get_pipeline().reset_history()
        st.rerun()

    if st.session_state.chat_messages:
        turns = len([m for m in st.session_state.chat_messages if m["role"] == "user"])
        st.markdown(f"<div style='text-align:center; font-size:0.8rem; opacity:0.8;'>{turns} turn(s) in session</div>",
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Example questions
# ---------------------------------------------------------------------------

if not st.session_state.chat_messages:
    st.markdown("""
<div style="background:#FDF0F5; border:1px solid #F8BBD9; border-radius:12px; padding:1.2rem; margin-bottom:1rem;">
  <div style="color:#C2185B; font-weight:700; margin-bottom:0.6rem;">💡 Example questions</div>
  <div style="color:#4A235A; font-size:0.88rem; line-height:1.8;">
    • What is the role of MHC class II molecules in antigen presentation?<br/>
    • How do B cells differentiate into plasma cells?<br/>
    • Describe the mechanism of complement activation via the classical pathway.<br/>
    • What cytokines are secreted by Th1 vs Th2 cells?<br/>
    • How does somatic hypermutation contribute to antibody affinity maturation?
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"], avatar="🧬" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} references)"):
                for src in msg["sources"]:
                    st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; padding:6px 0; border-bottom:1px solid #F8BBD9;">
  <span style="background:#C2185B; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:700; white-space:nowrap;">{src['Ref']}</span>
  <span style="color:#1A1A2E; font-size:0.88rem;"><strong>{src.get('Source','—')}</strong> · {src.get('Chapter','—')} · p.{src.get('Page','—')}</span>
</div>
""", unsafe_allow_html=True)
        if msg.get("images"):
            with st.expander("🖼️ Related Figures"):
                for img in msg["images"]:
                    st.write(f"**{img.get('title', 'Figure')}**")
                    if img.get("path"):
                        st.image(img["path"])


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

query = st.chat_input("Ask an immunology question…")

if query:
    st.session_state.chat_messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    pipeline = _get_pipeline()
    from src import constant
    constant.hyde_enabled = hyde_toggle
    dt = doc_type if doc_type != "All" else None
    sf = source_filter if source_filter else None

    with st.chat_message("assistant", avatar="🧬"):
        with st.spinner("Retrieving passages and generating answer…"):
            try:
                result = pipeline.answer(query, doc_type=dt, source_file=sf)
            except Exception as e:
                err = str(e)
                if "Connection refused" in err and "8000" in err:
                    hint = "vLLM server is not running on port 8000."
                elif "Connection refused" in err and "6000" in err:
                    hint = "Semantic chunking service is not running on port 6000."
                elif "404" in err and "does not exist" in err:
                    hint = "vLLM model name mismatch — restart with `--served-model-name Qwen/Qwen3-8B`."
                else:
                    hint = err
                st.error(f"⚠️ {hint}")
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": f"Error: {hint}", "sources": [], "images": []}
                )
                st.stop()

        st.markdown(result["answer"])

        # Source references
        sources = []
        for i, doc in enumerate(result.get("retrieved_docs", []), 1):
            meta = doc.metadata
            sources.append({
                "Ref":     f"[{i}]",
                "Source":  meta.get("source_file", "—"),
                "Chapter": meta.get("chapter", "—"),
                "Page":    meta.get("page", "—"),
            })

        if sources:
            with st.expander(f"📚 Sources ({len(sources)} references)", expanded=True):
                for src in sources:
                    st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; padding:6px 0; border-bottom:1px solid #F8BBD9;">
  <span style="background:#C2185B; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:700; white-space:nowrap;">{src['Ref']}</span>
  <span style="color:#1A1A2E; font-size:0.88rem;"><strong>{src['Source']}</strong> · {src['Chapter']} · p.{src['Page']}</span>
</div>
""", unsafe_allow_html=True)

        # Related figures
        images = result.get("related_images", [])
        if images:
            with st.expander("🖼️ Related Figures"):
                img_cols = st.columns(min(len(images), 3))
                for i, img in enumerate(images):
                    with img_cols[i % 3]:
                        st.caption(img.get("title", "Figure"))
                        if img.get("path"):
                            try:
                                st.image(img["path"])
                            except Exception:
                                st.write(img["path"])

        # Latency breakdown
        latency = result.get("latency_ms", {})
        if latency:
            total_ms = sum(latency.values())
            with st.expander(f"⏱️ Latency — {total_ms:.0f} ms total"):
                import pandas as pd
                lat_df = pd.DataFrame(
                    [(k, round(v, 1)) for k, v in sorted(latency.items(), key=lambda x: -x[1])],
                    columns=["Module", "ms"]
                )
                # Simple bar display
                max_v = lat_df["ms"].max()
                for _, row in lat_df.iterrows():
                    pct = row["ms"] / max_v if max_v > 0 else 0
                    st.markdown(f"""
<div style="display:flex; align-items:center; gap:8px; margin:3px 0;">
  <div style="width:90px; color:#6D4C5E; font-size:0.82rem; font-weight:600;">{row['Module']}</div>
  <div style="flex:1; background:#FCE4EC; border-radius:4px; height:16px; overflow:hidden;">
    <div style="width:{pct*100:.0f}%; background:linear-gradient(90deg,#C2185B,#E91E8C); height:100%; border-radius:4px;"></div>
  </div>
  <div style="width:60px; text-align:right; color:#C2185B; font-size:0.82rem; font-weight:700;">{row['ms']:.0f} ms</div>
</div>
""", unsafe_allow_html=True)

    st.session_state.chat_messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": sources,
        "images":  images,
    })
