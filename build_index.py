# =============================================================================
# ImmunoBiology RAG — Index Builder (Entry Script)
# =============================================================================
# Adapted from Tesla RAG: build_index.py
# Key changes:
#   - Multi-PDF: glob("data/raw/*.pdf") instead of single hardcoded path
#   - Chroma (not Milvus) for vector store
#   - English BM25 tokenization (NLTK, not jieba)
#   - Semantic chunking service call for parent grouping
#   - MongoDB for parent-child metadata
#
# Usage:
#   python build_index.py                # full build (all PDFs in data/raw/)
#   python build_index.py --pdf data/raw/some_paper.pdf   # single PDF
#   python build_index.py --dry-run      # inspect chunks without indexing
#   python build_index.py --skip-parse   # skip PDF parsing (reuse existing JSON)
#   python build_index.py --test-retrieval  # run test query after indexing

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from src import constant
from src.pdf_parser import load_all_pdfs, inspect_pdf_layout
from src.chunker import texts_split, print_sample_chunks
from src.embedder import Embedder


# =============================================================================
# Helpers
# =============================================================================

def _discover_pdfs(raw_dir: str) -> list:
    """Find all PDF files in the data/raw directory."""
    pdfs = sorted(glob.glob(str(Path(raw_dir) / "*.pdf")))
    if not pdfs:
        print(f"[BuildIndex] WARNING: No PDFs found in {raw_dir}")
    else:
        print(f"[BuildIndex] Found {len(pdfs)} PDF(s):")
        for p in pdfs:
            print(f"  - {Path(p).name}")
    return pdfs


def _load_processed_json(processed_dir: str) -> dict:
    """
    Load all previously parsed JSON chunks from data/processed/.
    Returns dict: {doc_stem: list_of_chunk_dicts}
    """
    processed = {}
    base = Path(processed_dir)
    for doc_dir in base.iterdir():
        if not doc_dir.is_dir():
            continue
        text_dir = doc_dir / "text"
        if not text_dir.exists():
            continue
        chunks = []
        for json_file in sorted(text_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    chunks.extend(json.load(f))
            except Exception as e:
                print(f"[BuildIndex] Warning: Could not load {json_file}: {e}")
        if chunks:
            processed[doc_dir.name] = chunks
            print(f"[BuildIndex] Loaded {len(chunks)} pages from: {doc_dir.name}")
    return processed


def _test_retrieval(embedder: Embedder) -> None:
    """Run a quick retrieval test to verify the built index."""
    from src.retriever.bm25_retriever import BM25Retriever
    from src.retriever.chroma_retriever import ChromaRetriever

    test_queries = [
        "What activates T cells?",
        "How do B cells produce antibodies?",
        "What is MHC class II?",
    ]

    # Test BM25
    print("\n[Test] BM25 Retrieval:")
    bm25 = BM25Retriever(docs=None, retrieve=True)
    for q in test_queries[:1]:
        results = bm25.retrieve_topk(q, topk=3)
        print(f"  Query: {q}")
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("source_file", "?")
            pg  = doc.metadata.get("page", "?")
            print(f"  [{i}] (p{pg}, {src[:30]}) {doc.page_content[:80]}...")

    # Test Chroma (needs encoder)
    print("\n[Test] Dense Retrieval (Chroma):")
    try:
        encoder = embedder.bge.encode_query   # BGEEmbedder.encode_query callable
        chroma = ChromaRetriever(encode_query_fn=encoder)
        for q in test_queries[:1]:
            results = chroma.retrieve_topk(q, topk=3)
            print(f"  Query: {q}")
            for i, doc in enumerate(results, 1):
                src = doc.metadata.get("source_file", "?")
                pg  = doc.metadata.get("page", "?")
                print(f"  [{i}] (p{pg}, {src[:30]}) {doc.page_content[:80]}...")
    except Exception as e:
        print(f"  [Test] Dense retrieval test failed: {e}")


# =============================================================================
# Main build pipeline
# =============================================================================

def build_index(
    pdf_path: str = None,
    dry_run: bool = False,
    skip_parse: bool = False,
    test_retrieval: bool = False,
) -> None:
    """
    Full index build pipeline:
      1. Discover / inspect PDFs
      2. Parse PDFs → data/processed/{stem}/text/*.json
      3. Semantic chunk → parent + child documents → MongoDB
      4. Embed → Chroma (dense) + BM25 (sparse)
      5. (Optional) Test retrieval

    Args:
        pdf_path:       If set, process only this PDF; otherwise process all in data/raw/
        dry_run:        If True, stop after chunking and show diagnostics only
        skip_parse:     If True, skip PDF parsing and reuse existing JSON in data/processed/
        test_retrieval: If True, run a quick retrieval test after indexing
    """
    t_start = time.perf_counter()

    raw_dir       = constant.raw_dir
    processed_dir = constant.processed_dir

    # ------------------------------------------------------------------
    # Step 1 — Discover PDFs
    # ------------------------------------------------------------------
    if pdf_path:
        pdfs = [pdf_path]
        print(f"[BuildIndex] Processing single PDF: {pdf_path}")
    else:
        pdfs = _discover_pdfs(raw_dir)

    if not pdfs and not skip_parse:
        print("[BuildIndex] Nothing to do. Add PDFs to data/raw/ and rerun.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 2 — Inspect layout (first run only, for calibration)
    # ------------------------------------------------------------------
    if not skip_parse and pdfs:
        report_path = Path(constant.diagnostics_dir) / "pdf_layout_report.txt"
        if not report_path.exists():
            print("\n[BuildIndex] Running PDF layout inspection (first-time calibration)...")
            for pdf in pdfs[:1]:  # Inspect first PDF only
                inspect_pdf_layout(pdf)   # writes to constant.pdf_layout_report internally
            print(f"[BuildIndex] Layout report saved: {report_path}")
        else:
            print(f"[BuildIndex] Skipping layout inspection (report exists: {report_path})")

    # ------------------------------------------------------------------
    # Step 3 — Parse PDFs
    # ------------------------------------------------------------------
    if skip_parse:
        print("\n[BuildIndex] Skipping PDF parsing (--skip-parse). Loading existing JSON...")
        processed_data = _load_processed_json(processed_dir)
        all_page_records = []
        for records in processed_data.values():
            all_page_records.extend(records)
    else:
        print(f"\n[BuildIndex] Parsing {len(pdfs)} PDF(s)...")
        all_page_records, extraction_report = load_all_pdfs(pdfs)
        print(f"[BuildIndex] Parsed {len(all_page_records)} total page records.")

    if not all_page_records:
        print("[BuildIndex] No page records to process. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4 — Semantic chunking → parent + child docs → MongoDB
    # ------------------------------------------------------------------
    print(f"\n[BuildIndex] Chunking {len(all_page_records)} page records...")
    children = texts_split(all_page_records)
    print(f"[BuildIndex] Created {len(children)} child chunks (parents saved to MongoDB).")

    if dry_run:
        print("\n[BuildIndex] DRY RUN: showing sample chunks. Not writing to index.")
        print_sample_chunks(children, n=5)
        print(f"\n[BuildIndex] Dry run complete in {(time.perf_counter() - t_start)*1000:.0f} ms.")
        return

    # ------------------------------------------------------------------
    # Step 5 — Embed: Chroma + BM25
    # ------------------------------------------------------------------
    print("\n[BuildIndex] Building vector index (Chroma + BM25)...")
    embedder = Embedder()
    embedder.build_index(children)
    print("[BuildIndex] Indexing complete.")

    # ------------------------------------------------------------------
    # Step 6 — Report
    # ------------------------------------------------------------------
    report_path = Path(constant.diagnostics_dir) / "index_report.txt"
    print(f"[BuildIndex] Index report: {report_path}")

    total_ms = (time.perf_counter() - t_start) * 1000
    print(f"\n[BuildIndex] Total build time: {total_ms/1000:.1f} s")

    # ------------------------------------------------------------------
    # Step 7 — Optional retrieval test
    # ------------------------------------------------------------------
    if test_retrieval:
        print("\n[BuildIndex] Running retrieval sanity check...")
        _test_retrieval(embedder)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ImmunoBiology RAG — Build vector + BM25 index from PDFs."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to a single PDF to process. If omitted, processes all PDFs in data/raw/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + chunk only; show diagnostics without writing to index.",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip PDF parsing; reuse existing JSON in data/processed/.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run PDF layout inspection only (produces pdf_layout_report.txt).",
    )
    parser.add_argument(
        "--test-retrieval",
        action="store_true",
        help="Run a quick retrieval test query after indexing.",
    )

    args = parser.parse_args()

    if args.inspect:
        # Layout inspection only
        pdfs = _discover_pdfs(constant.raw_dir)
        if pdfs:
            inspect_pdf_layout(pdfs[0])   # writes to constant.pdf_layout_report internally
            print(f"Layout report saved to: {constant.pdf_layout_report}")
        sys.exit(0)

    build_index(
        pdf_path=args.pdf,
        dry_run=args.dry_run,
        skip_parse=args.skip_parse,
        test_retrieval=args.test_retrieval,
    )
