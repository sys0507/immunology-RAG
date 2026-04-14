# =============================================================================
# ImmunoBiology RAG — Text Chunker (Parent-Child Architecture)
# =============================================================================
# Adapted from Tesla RAG: src/parser/pdf_parse.py (texts_split function)
# Key changes:
#   - Accepts pre-parsed Documents from pdf_parser.py (not raw PDF)
#   - Richer metadata: source_file, doc_type, chapter, chunk_id
#   - chunk_size: 512 tokens (English; Tesla used 256 for Chinese)
#   - chunk_overlap: 100 tokens (Tesla used 50)
#   - English separators: ["\n\n", "\n", ". ", " "]
#   - chunk_id format: "<doc_stem>_ch<N>_p<page>_<seq>"
#   - Dry-run mode: generates chunk-length histogram only
#
# Architecture (mirrors Tesla parent-child design):
#   semantic_group (parent, ~512 tokens each)
#       └── child_chunk_1  (child, ≤512 tokens with overlap)
#       └── child_chunk_2
#   Retrieval uses child chunks (precise matching)
#   LLM context uses parent chunks (complete context)
#   Parent-child relationships stored in MongoDB
#
# Usage:
#   python src/chunker.py --dry-run    # generate histogram, no MongoDB writes
#   python src/chunker.py              # full chunking + MongoDB storage

# %% [Cell 1: Imports and configuration]
import os
import re
import copy
import json
import hashlib
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import tiktoken
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import constant
from src.fields.chunk_info_mongo import ChunkInfo
from src.client.mongodb_config import MongoConfig
from src.client.semantic_chunk_client import request_semantic_chunk

# ---------------------------------------------------------------------------
# Configuration
# chunk_size: 512 tokens — calibrated for English academic text.
#   Typical English academic sentence: 20-30 words ≈ 25-35 tokens.
#   512 tokens ≈ 15-20 sentences — enough for one complete concept.
#   Tesla used 256 for Chinese (Chinese chars are ~1 token each; info-dense).
#
# chunk_overlap: 100 tokens — ensures context continuity across chunks.
#   Prevents cutting off mid-concept. ~20% of chunk_size.
# ---------------------------------------------------------------------------
_chunk_size    = constant.chunk_size      # 512
_chunk_overlap = constant.chunk_overlap   # 100
_separators    = constant.chunk_separators
_group_size    = constant.semantic_group_size  # 15
_max_parent_sz = constant.max_parent_size      # 512 chars

# Use cl100k_base tokenizer (same as GPT-4/LLaMA family) for accurate token counts
encoding = tiktoken.get_encoding("cl100k_base")

# MongoDB collection
chunk_collection = MongoConfig.get_collection(constant.mongo_collection)

# Text splitter for child chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_chunk_size,
    chunk_overlap=_chunk_overlap,
    separators=_separators,
    length_function=lambda text: len(encoding.encode(text)),
)


# %% [Cell 2: Helper — generate chunk_id]

def _make_chunk_id(source_file: str, chapter: str, page: int, seq: int) -> str:
    """
    Generate a human-readable chunk ID.
    Example: "janeway10e_ch3_p87_001"
    """
    stem = Path(source_file).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)[:20]
    ch_num = re.search(r"\d+", chapter)
    ch_str = f"ch{ch_num.group()}" if ch_num else "ch0"
    return f"{stem}_{ch_str}_p{page:04d}_{seq:03d}"


# %% [Cell 3: Save chunk to MongoDB]

def _save_to_mongo(docs: List[Document]) -> None:
    """
    Upsert a list of Documents into MongoDB chunk_metadata collection.
    Uses unique_id as the primary key; upsert=True prevents duplicates.
    """
    for doc in docs:
        metadata = doc.metadata
        unique_id = metadata.get("unique_id")
        if not unique_id:
            continue

        chunk_record = ChunkInfo(
            unique_id=unique_id,
            page_content=doc.page_content,
            metadata=metadata,
        )
        chunk_collection.update_one(
            {"unique_id": chunk_record.unique_id},
            {"$set": chunk_record.model_dump()},
            upsert=True,
        )


# %% [Cell 4: Main chunking function]

def texts_split(raw_docs: List[Document], dry_run: bool = False) -> List[Document]:
    """
    Split page-level Documents into parent + child chunks.

    Two-level hierarchy (mirrors Tesla parent-child design):
    1. PARENT chunks: semantic groups from the chunking service (~512 tokens).
       Stored in MongoDB. Used by merge_docs() to provide full context to LLM.
    2. CHILD chunks: sub-splits of parents with overlap (~256-512 tokens).
       Stored in MongoDB AND returned for vector indexing.
       Retrieval uses children (precise match); LLM receives parents.

    Args:
        raw_docs: Page-level Documents from pdf_parser.load_all_pdfs()
        dry_run:  If True, skip MongoDB writes and just return all chunks

    Returns:
        List of CHILD Documents (these go into Chroma + BM25)
    """
    all_split_docs = []
    chunk_lengths = []

    for doc in tqdm(raw_docs, desc="Chunking documents"):
        page_text = doc.page_content

        # Semantic grouping: call the FastAPI service to group paragraphs
        grouped_chunks = request_semantic_chunk(page_text, group_size=_group_size)

        parent_docs = []
        for group_text in grouped_chunks:
            # Create parent document
            parent_id = hashlib.md5(group_text.encode("utf-8")).hexdigest()
            parent_meta = copy.deepcopy(doc.metadata)
            parent_meta["unique_id"] = parent_id
            parent_meta.pop("parent_id", None)  # parents have no parent_id

            parent_doc = Document(page_content=group_text, metadata=parent_meta)
            parent_docs.append(parent_doc)

            # If parent is short enough, it is also a retrieval unit.
            # The 20-char minimum prevents trivial single-character parents
            # from the same hash-collision issue as child chunks.
            if len(group_text.strip()) >= 20 and len(group_text) < _max_parent_sz:
                all_split_docs.append(parent_doc)
                chunk_lengths.append(len(encoding.encode(group_text)))

        # Save parents to MongoDB
        if not dry_run:
            _save_to_mongo(parent_docs)

        # Create child documents from each parent
        child_seq = 0
        for parent_doc in parent_docs:
            child_docs_raw = text_splitter.create_documents(
                [parent_doc.page_content],
                metadatas=[parent_doc.metadata],
            )
            child_docs = []
            for child_doc in child_docs_raw:
                # Skip trivially short chunks (single letters, punctuation, whitespace).
                # These arise from near-empty pages (cover pages, full-page figures,
                # chapter openers) and cause MD5 hash collisions across the corpus,
                # which crashes ChromaDB with DuplicateIDError.
                stripped = child_doc.page_content.strip()
                if len(stripped) < 20:
                    continue

                # Skip if child == parent (no split happened)
                if stripped == parent_doc.page_content.strip():
                    continue

                child_id = hashlib.md5(
                    child_doc.page_content.encode("utf-8")
                ).hexdigest()
                child_meta = copy.deepcopy(parent_doc.metadata)
                child_meta["unique_id"] = child_id
                child_meta["parent_id"] = parent_doc.metadata["unique_id"]
                child_meta["chunk_id"] = _make_chunk_id(
                    child_meta["source_file"],
                    child_meta["chapter"],
                    child_meta["page"],
                    child_seq,
                )
                child_seq += 1

                child_doc_final = Document(
                    page_content=child_doc.page_content,
                    metadata=child_meta,
                )
                child_docs.append(child_doc_final)
                chunk_lengths.append(len(encoding.encode(child_doc.page_content)))

            if not dry_run:
                _save_to_mongo(child_docs)
            all_split_docs.extend(child_docs)

    print(
        f"[Chunker] Total chunks: {len(all_split_docs)} "
        f"(avg length: {sum(chunk_lengths) / max(len(chunk_lengths), 1):.0f} tokens)"
    )

    # Generate and save histogram
    _save_chunk_histogram(chunk_lengths)

    return all_split_docs


# %% [Cell 5: Chunk-length histogram]

def _save_chunk_histogram(chunk_lengths: List[int]) -> None:
    """
    Save a chunk-length distribution histogram to outputs/diagnostics/.
    Used to verify that chunk_size / chunk_overlap parameters are well-calibrated.
    """
    if not chunk_lengths:
        return

    os.makedirs(constant.diagnostics_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(chunk_lengths, bins=50, edgecolor="black", color="steelblue", alpha=0.8)
    axes[0].axvline(_chunk_size, color="red", linestyle="--", linewidth=1.5,
                    label=f"chunk_size={_chunk_size}")
    axes[0].set_title("Chunk Length Distribution (tokens)", fontsize=13)
    axes[0].set_xlabel("Tokens per chunk")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Cumulative distribution
    sorted_lengths = sorted(chunk_lengths)
    cumulative = [i / len(sorted_lengths) for i in range(len(sorted_lengths))]
    axes[1].plot(sorted_lengths, cumulative, color="steelblue")
    axes[1].axvline(_chunk_size, color="red", linestyle="--", linewidth=1.5,
                    label=f"chunk_size={_chunk_size}")
    axes[1].set_title("Cumulative Distribution", fontsize=13)
    axes[1].set_xlabel("Tokens per chunk")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].legend()

    # Statistics annotation
    stats_text = (
        f"n={len(chunk_lengths)}\n"
        f"min={min(chunk_lengths)}\n"
        f"median={sorted(chunk_lengths)[len(chunk_lengths)//2]}\n"
        f"max={max(chunk_lengths)}\n"
        f"mean={sum(chunk_lengths)/len(chunk_lengths):.0f}"
    )
    axes[0].text(0.98, 0.97, stats_text, transform=axes[0].transAxes,
                 verticalalignment="top", horizontalalignment="right",
                 fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(constant.chunk_dist_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Chunker] Chunk-length histogram saved to {constant.chunk_dist_plot}")


# %% [Cell 6: Print sample chunks for quality check]

def print_sample_chunks(split_docs: List[Document], n: int = 5) -> None:
    """Print n sample chunks with their metadata for quality inspection."""
    import random
    samples = random.sample(split_docs, min(n, len(split_docs)))
    for i, doc in enumerate(samples, 1):
        print(f"\n{'='*70}")
        print(f"Sample {i}/{n}")
        print(f"  source_file: {doc.metadata.get('source_file', 'N/A')}")
        print(f"  doc_type:    {doc.metadata.get('doc_type', 'N/A')}")
        print(f"  chapter:     {doc.metadata.get('chapter', 'N/A')}")
        print(f"  page:        {doc.metadata.get('page', 'N/A')}")
        print(f"  chunk_id:    {doc.metadata.get('chunk_id', 'N/A')}")
        print(f"  unique_id:   {doc.metadata.get('unique_id', 'N/A')[:16]}...")
        print(f"  parent_id:   {doc.metadata.get('parent_id', 'N/A')}")
        tokens = len(encoding.encode(doc.page_content))
        print(f"  tokens:      {tokens}")
        print(f"  content:     {doc.page_content[:300]}...")


# %% [Cell 7: CLI entry point]

def main():
    parser = argparse.ArgumentParser(description="ImmunoBiology RAG Chunker")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate chunk histogram only; no MongoDB writes"
    )
    args = parser.parse_args()

    # Load pre-parsed raw docs from pickle cache (written by build_index.py)
    raw_docs_path = constant.raw_docs_path
    if not os.path.exists(raw_docs_path):
        print(f"[Chunker] No cached raw docs at {raw_docs_path}. "
              "Run pdf_parser.py first or use build_index.py.")
        return

    print(f"[Chunker] Loading raw docs from {raw_docs_path}...")
    with open(raw_docs_path, "rb") as f:
        raw_docs = pickle.load(f)
    print(f"[Chunker] Loaded {len(raw_docs)} page documents.")

    split_docs = texts_split(raw_docs, dry_run=args.dry_run)

    if args.dry_run:
        print("[Chunker] Dry-run complete. Inspect the histogram before proceeding.")
        print_sample_chunks(split_docs, n=5)
    else:
        print(f"[Chunker] Chunking complete: {len(split_docs)} chunks stored in MongoDB.")
        print_sample_chunks(split_docs, n=3)

        # Cache split docs for embedder.py
        with open(constant.split_docs_path, "wb") as f:
            pickle.dump(split_docs, f)
        print(f"[Chunker] Split docs cached to {constant.split_docs_path}")


if __name__ == "__main__":
    main()
