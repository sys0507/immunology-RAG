# =============================================================================
# ImmunoBiology RAG — Utility Functions
# =============================================================================
# Adapted from Tesla RAG: src/utils.py
# Key changes:
#   - merge_docs(): logic unchanged; uses immunology_rag MongoDB collection
#   - post_processing(): citation regex changed from 【1】 → [1] (English format)
#     Also returns source_file and chapter in output for richer UI citations
#   - Added timer() decorator for per-module latency tracking

import re
import time
import functools
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.client.mongodb_config import MongoConfig
from src import constant

# MongoDB collection for chunk lookups
chunk_collection = MongoConfig.get_collection(constant.mongo_collection)


# =============================================================================
# merge_docs: deduplicate and fetch parent documents
# =============================================================================

def merge_docs(
    docs1: List[Document],
    docs2: List[Document],
) -> List[Document]:
    """
    Merge two retrieval result lists, deduplicate, and fetch parent documents.

    Design (mirrors Tesla's merge_docs exactly):
    - Retrieval uses child chunks (precise matching)
    - For LLM context we want the larger parent chunk (more complete)
    - If a doc has parent_id → look up parent in MongoDB
    - If no parent_id → use the doc as-is (it is already a parent)
    - Deduplicate by unique_id to avoid repeated passages

    Args:
        docs1: Results from BM25 retrieval
        docs2: Results from dense (Chroma) retrieval

    Returns:
        Deduplicated list of parent Documents for reranking
    """
    merged = []
    seen_ids = set()

    for doc in docs1 + docs2:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            # Child doc → fetch parent from MongoDB
            parent_rec = chunk_collection.find_one({"unique_id": parent_id})
            if parent_rec:
                uid = parent_rec["unique_id"]
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    parent_doc = Document(
                        page_content=parent_rec["page_content"],
                        metadata=parent_rec["metadata"],
                    )
                    merged.append(parent_doc)
            # If parent not found in MongoDB, fall through to use child
            else:
                uid = doc.metadata.get("unique_id", "")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    merged.append(doc)
        else:
            # Already a parent doc (or standalone)
            uid = doc.metadata.get("unique_id", "")
            if uid and uid not in seen_ids:
                seen_ids.add(uid)
                merged.append(doc)

    return merged


# =============================================================================
# post_processing: extract citations and associated images/pages
# =============================================================================

def post_processing(
    response: str,
    docs: List[Document],
) -> Dict[str, Any]:
    """
    Extract citation numbers from LLM response and map to source metadata.

    Handles English citation format: [1], [2,3], [1, 2, 3]
    (Tesla used Chinese brackets: 【1,2】)

    Args:
        response: Raw LLM output containing [N] citation markers
        docs:     Ordered list of context Documents (1-indexed to match [N])

    Returns:
        {
          "answer":         str — answer text with citation markers stripped,
          "cite_pages":     List[int] — page numbers cited,
          "cite_sources":   List[str] — source filenames cited,
          "cite_chapters":  List[str] — chapters cited,
          "related_images": List[dict] — associated figure metadata,
        }
    """
    # Extract all citation indices from [1], [2,3], [1, 2, 3] patterns
    raw_cites = re.findall(r'\[(\d+(?:[,\s]+\d+)*)\]', response)
    cite_indices = []
    for cite_str in raw_cites:
        nums = re.findall(r'\d+', cite_str)
        cite_indices.extend(int(n) for n in nums)
    cite_indices = sorted(set(cite_indices))

    # Strip citation markers from answer text
    answer = re.sub(r'\[\d+(?:[,\s]+\d+)*\]', '', response).strip()

    # Gather metadata for cited passages
    pages, sources, chapters, related_images = [], [], [], []
    for idx in cite_indices:
        if idx < 1 or idx > len(docs):
            continue
        meta = docs[idx - 1].metadata

        page = meta.get("page")
        if page and page not in pages:
            pages.append(page)

        src = meta.get("source_file", "")
        if src and src not in sources:
            sources.append(src)

        ch = meta.get("chapter", "")
        if ch and ch not in chapters:
            chapters.append(ch)

        # Collect images that have a figure caption
        for img in meta.get("images_info", []):
            if img.get("title") or img.get("has_caption"):
                related_images.append(img)

    return {
        "answer":         answer,
        "cite_pages":     sorted(pages),
        "cite_sources":   sources,
        "cite_chapters":  chapters,
        "related_images": related_images,
    }


# =============================================================================
# format_context: prepare context string for LLM prompt
# =============================================================================

def format_context(ranked_docs: List[Document]) -> str:
    """
    Format ranked documents as numbered context passages for the LLM prompt.
    Example output:
        [1] T cell activation requires...
        [2] B cells differentiate into plasma cells...
    """
    lines = []
    for idx, doc in enumerate(ranked_docs, 1):
        lines.append(f"[{idx}] {doc.page_content}")
    return "\n\n".join(lines)


# =============================================================================
# timer: per-module latency decorator
# =============================================================================

def timer(func):
    """
    Decorator that measures and prints function execution time.
    Used by pipeline.py for per-module latency tracking.

    Usage:
        @timer
        def retrieve_topk(query, topk=10):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[Timer] {func.__qualname__}: {elapsed_ms:.1f} ms")
        return result
    return wrapper


class LatencyTracker:
    """
    Tracks per-module latency across a pipeline run.
    Results are included in the pipeline output for evaluation.
    """

    def __init__(self):
        self._times: Dict[str, float] = {}

    def start(self, module: str) -> None:
        self._times[f"{module}_start"] = time.perf_counter()

    def stop(self, module: str) -> float:
        start_key = f"{module}_start"
        if start_key not in self._times:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._times[start_key]) * 1000
        self._times[module] = elapsed_ms
        return elapsed_ms

    def get_report(self) -> Dict[str, float]:
        """Return dict of module_name → latency_ms."""
        return {k: v for k, v in self._times.items() if not k.endswith("_start")}
