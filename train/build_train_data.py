# =============================================================================
# ImmunoBiology RAG — Training Data Generation
# =============================================================================
# Adapted from Tesla RAG: src/gen_qa/run.py + generate_sft_data.py
# Key changes:
#   - All 4 prompt templates rewritten for English immunology domain
#   - Doubao API → local vLLM (same OpenAI-compatible interface)
#   - ThreadPoolExecutor retained for parallel generation
#   - Outputs both reranker triplets AND SFT ChatML data
#   - Checkpoint/resume: each stage saves its output and skips if already done
#   - Eval QA set: 100-200 QA pairs with ground-truth for evaluate.py
#
# Usage:
#   python -m train.build_train_data                    # full run (resumes if interrupted)
#   python -m train.build_train_data --force            # redo all stages from scratch
#   python -m train.build_train_data --workers 4        # parallel workers
#   python -m train.build_train_data --limit 200        # limit to N chunks (smoke test)
#
# Stage checkpoints (each stage is skipped if its output file already exists):
#   Stage 1: QA generation   → data/train/qa_pairs_cache.jsonl
#   Stage 2: Reranker data   → data/train/reranker_train.jsonl + reranker_test.jsonl
#   Stage 3: SFT data        → data/train/sft_train.jsonl
#   Stage 4: Eval QA set     → data/train/eval_qa.jsonl

import json
import random
import re
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.documents import Document

from src import constant
from src.client.llm_datagen_client import chat
from src.client.mongodb_config import MongoConfig
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.chroma_retriever import ChromaRetriever


# =============================================================================
# Prompt templates (English, immunology domain)
# =============================================================================

CONTEXT_PROMPT_TPL = """You are an expert immunology educator. Given the following passage from an immunology textbook,
generate {n_qa} distinct question-answer pairs that test understanding of the key concepts.

Requirements:
- Questions should be specific and scientifically accurate
- Answers should be concise (1-3 sentences) and directly answerable from the passage
- Cover different aspects: mechanisms, definitions, comparisons, clinical relevance
- Use proper immunological terminology

Passage:
{passage}

Output ONLY a JSON array with this exact format (no other text):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""


GENERALIZE_PROMPT_TPL = """Given the following immunology question, generate {n_para} paraphrase variants
that ask the same thing differently. Use synonyms, different phrasing, and varied question structures.

Original question: {question}

Output ONLY a JSON array with this exact format (no other text):
[
  {{"question": "..."}},
  {{"question": "..."}}
]"""


KEYWORDS_PROMPT_TPL = """Extract {n_kw} key immunological terms or concepts from this answer that
could be used as search keywords.

Answer: {answer}

Output ONLY a JSON array with this exact format (no other text):
[
  {{"keyword": "..."}},
  {{"keyword": "..."}}
]"""


# =============================================================================
# Data generation functions
# =============================================================================

def _parse_json_list(text: str) -> Optional[list]:
    """Safely parse a JSON list from LLM output (which may have extra text)."""
    if not text:
        return None
    # Find the JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_qa_pairs(chunk: dict, n_qa: int = 5) -> List[dict]:
    """
    Generate QA pairs from a single chunk.

    Args:
        chunk: MongoDB chunk document with 'page_content' and 'metadata'
        n_qa:  Number of QA pairs to generate

    Returns:
        List of {'question': str, 'answer': str, 'chunk_id': str, 'source_file': str}
    """
    passage = chunk.get("page_content", "").strip()
    if len(passage.split()) < 30:   # Skip very short passages
        return []

    prompt = CONTEXT_PROMPT_TPL.format(passage=passage, n_qa=n_qa)
    response = chat(prompt, temperature=0.85, debug=False)

    qa_list = _parse_json_list(response)
    if not qa_list:
        return []

    metadata = chunk.get("metadata", {})
    results = []
    for item in qa_list:
        if "question" in item and "answer" in item:
            results.append({
                "question":    item["question"].strip(),
                "answer":      item["answer"].strip(),
                "chunk_id":    metadata.get("chunk_id", ""),
                "source_file": metadata.get("source_file", ""),
                "chapter":     metadata.get("chapter", ""),
                "page":        metadata.get("page", ""),
                "passage":     passage,
                "unique_id":   chunk.get("unique_id", ""),
            })
    return results


def generate_paraphrases(qa: dict, n_para: int = 5) -> List[str]:
    """Generate paraphrase variants for a question."""
    prompt = GENERALIZE_PROMPT_TPL.format(
        question=qa["question"], n_para=n_para
    )
    response = chat(prompt, temperature=0.8)
    variants = _parse_json_list(response) or []
    paraphrases = [qa["question"]]  # Include original
    for item in variants:
        if "question" in item:
            paraphrases.append(item["question"].strip())
    return paraphrases


def generate_keywords(qa: dict, n_kw: int = 5) -> List[str]:
    """Extract keywords from an answer."""
    prompt = KEYWORDS_PROMPT_TPL.format(answer=qa["answer"], n_kw=n_kw)
    response = chat(prompt, temperature=0.3)
    kw_list = _parse_json_list(response) or []
    return [item["keyword"].strip() for item in kw_list if "keyword" in item]


def _resolve_parent_content(uid: str, uid_to_chunk: dict) -> Optional[str]:
    """
    Resolve a chunk's unique_id to its parent chunk's page_content.

    The inference pipeline calls merge_docs() before reranking, which replaces
    child chunks with their parent chunks from MongoDB.  Training data must use
    the same granularity so the reranker trains on the text it actually sees at
    inference time.

    Resolution rules:
    - If the chunk has a parent_id and the parent exists → return parent content
    - If the chunk has no parent_id (it IS a parent) → return its own content
    - If the chunk is unknown → return None (caller falls back to raw text)
    """
    chunk = uid_to_chunk.get(uid)
    if chunk is None:
        return None
    parent_id = chunk.get("metadata", {}).get("parent_id")
    if parent_id:
        parent = uid_to_chunk.get(parent_id)
        if parent:
            return parent.get("page_content")
        # parent_id set but parent missing — use child content as fallback
        return chunk.get("page_content")
    return chunk.get("page_content")


def _build_bm25_negative(
    qa: dict,
    bm25: "BM25Retriever",
    uid_to_chunk: dict,
    topk: int = 10,
) -> Optional[str]:
    """
    Find a hard negative for a QA pair using BM25 retrieval.

    Retrieves the top-K passages for the query, then returns the
    highest-ranked one that is NOT the positive passage (matched by unique_id).
    The returned text is resolved to the parent chunk so it matches the
    granularity the reranker sees at inference time (after merge_docs()).

    Using retrieval-based negatives instead of LLM-generated ones ensures:
    - Negatives are actual textbook passages (same distribution as inference)
    - They are topically related (genuinely hard to distinguish)
    - The reranker learns meaningful relevance signals, not style artefacts
    """
    results = bm25.retrieve_topk(qa["question"], topk=topk)
    pos_uid = qa.get("unique_id", "")
    for doc in results:
        uid = doc.metadata.get("unique_id", "")
        if uid != pos_uid:
            return _resolve_parent_content(uid, uid_to_chunk) or doc.page_content
    return None


def _load_dense_retriever() -> Optional[ChromaRetriever]:
    """
    Load the Chroma dense retriever with BGE-M3 encoder.

    Requires the Chroma vectorstore to be already built (Step 4.3 build_index.py).
    Returns None gracefully if the index is missing or FlagEmbedding is unavailable,
    falling back to BM25-only negatives.
    """
    try:
        from FlagEmbedding import FlagModel
        print("[Reranker] Loading BGE-M3 encoder for hybrid negative mining...")
        flag_model = FlagModel(
            constant.bge_m3_model_path,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            use_fp16=True,
        )

        def encode_query(query: str):
            return flag_model.encode(query).tolist()

        chroma = ChromaRetriever(encode_query_fn=encode_query)
        print(f"[Reranker] Dense retriever ready ({chroma.collection.count()} chunks).")
        return chroma
    except Exception as exc:
        print(f"[Reranker] ⚠ Dense retriever unavailable ({exc}). "
              "Falling back to BM25-only negatives.")
        return None


def _build_hybrid_negative(
    qa: dict,
    bm25: "BM25Retriever",
    chroma: Optional[ChromaRetriever],
    uid_to_chunk: dict,
    topk: int = 10,
) -> Optional[str]:
    """
    Find a hard negative using hybrid (BM25 + dense) retrieval.

    Retrieves top-K candidates from BM25 and (if available) from Chroma, merges
    and deduplicates them, excludes the positive passage, and returns the
    highest-ranked remaining candidate resolved to its parent chunk.

    Using both retrievers ensures the reranker learns to distinguish the same
    candidate distribution it faces in production, fixing the performance
    regression caused by BM25-only negatives (which the dense retriever may
    never surface in production).

    The returned text is resolved to the parent chunk so it matches the
    granularity the reranker sees at inference time (after merge_docs()).
    """
    pos_uid = qa.get("unique_id", "")

    bm25_docs  = bm25.retrieve_topk(qa["question"], topk=topk)
    dense_docs = chroma.retrieve_topk(qa["question"], topk=topk) if chroma else []

    seen_uids: set = set()
    candidates: List[Document] = []
    for doc in bm25_docs + dense_docs:
        uid = doc.metadata.get("unique_id", "")
        if uid != pos_uid and uid not in seen_uids:
            seen_uids.add(uid)
            candidates.append(doc)

    if not candidates:
        return None
    top_uid = candidates[0].metadata.get("unique_id", "")
    return _resolve_parent_content(top_uid, uid_to_chunk) or candidates[0].page_content


# =============================================================================
# Build reranker training data
# =============================================================================

def build_reranker_data(
    qa_pairs: List[dict],
    all_chunks: List[dict],
    output_train: str,
    output_test: str,
    test_ratio: float = 0.1,
) -> None:
    """
    Build reranker triplets: (query, positive_chunk, hard_negative_chunk).

    Hard negatives are sourced via hybrid retrieval (BM25 + dense Chroma) — actual
    textbook passages that are topically related to the query but do not answer it.
    Using both retrievers ensures negatives match the candidate distribution the
    reranker faces in production (hybrid pipeline), preventing the performance
    regression that occurred when BM25-only negatives were used.

    Falls back to BM25-only if the Chroma vectorstore is unavailable.

    Train/test split is stratified by chapter so no chapter appears in both
    sets, preventing data leakage from inflating eval metrics.

    Labels:
        2 = positive (directly answers the question)
        0 = hard negative (topically related but doesn't answer)

    Requires the Chroma vectorstore to be built (Step 4.3) for hybrid mode.
    Does NOT require vLLM.
    """
    # Build Document list for BM25 (used whether loading from pickle or building fresh)
    print(f"\n[Reranker] Building BM25 index from {len(all_chunks)} chunks...")
    docs = [
        Document(
            page_content=c["page_content"],
            metadata={**c.get("metadata", {}), "unique_id": c.get("unique_id", "")},
        )
        for c in all_chunks
        if c.get("page_content", "").strip()
    ]
    # retrieve=True loads existing pickle from build_index.py (Step 4.3) — fast path.
    # If pickle is missing, falls back to building from docs automatically.
    bm25 = BM25Retriever(docs, retrieve=True)

    # v4: build uid→chunk lookup for parent-chunk resolution.
    # The inference pipeline calls merge_docs() before reranking, replacing child
    # chunks with their parent chunks.  We must use the same granularity here so
    # the reranker trains on exactly the text it will see at inference time.
    uid_to_chunk: dict = {
        c["unique_id"]: c
        for c in all_chunks
        if c.get("unique_id")
    }

    # Load dense retriever for hybrid negatives (graceful fallback to BM25-only)
    chroma = _load_dense_retriever()
    mode = "hybrid (BM25 + dense)" if chroma else "BM25-only"
    print(f"[Reranker] Negative mining mode: {mode}")
    print(f"[Reranker] Building triplets for {len(qa_pairs)} QA pairs...")

    triplets = []
    skipped = 0
    t0 = time.perf_counter()

    for i, qa in enumerate(qa_pairs, 1):
        neg = _build_hybrid_negative(qa, bm25, chroma, uid_to_chunk)

        # Fallback: random chunk with different unique_id
        if neg is None:
            candidates = [
                c for c in all_chunks
                if c.get("unique_id") != qa.get("unique_id")
                and len(c.get("page_content", "").split()) > 30
            ]
            if candidates:
                fallback_chunk = random.choice(candidates)
                fallback_uid   = fallback_chunk.get("unique_id", "")
                neg = (_resolve_parent_content(fallback_uid, uid_to_chunk)
                       or fallback_chunk["page_content"])
            else:
                neg = None

        # v4: resolve positive to parent chunk (matches merge_docs() at inference)
        pos_content = (
            _resolve_parent_content(qa.get("unique_id", ""), uid_to_chunk)
            or qa["passage"]
        )

        if neg:
            triplets.append({
                "query":     qa["question"],
                "pos":       pos_content,
                "neg":       neg,
                "chapter":   qa.get("chapter", ""),
                "pos_label": 2,
                "neg_label": 0,
            })
        else:
            skipped += 1

        if i % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(qa_pairs) - i) / rate if rate > 0 else 0
            print(f"[Reranker] Progress: {i}/{len(qa_pairs)} pairs, "
                  f"{len(triplets)} triplets, {elapsed:.0f}s elapsed "
                  f"(~{remaining/60:.1f} min remaining)", flush=True)

    if skipped:
        print(f"[Reranker] Warning: {skipped} QA pairs skipped (no negative found).")
    print(f"\n[Reranker] Generated {len(triplets)} triplets total.")

    # Chapter-stratified split — eval covers held-out chapters only
    by_chapter: Dict[str, list] = {}
    for t in triplets:
        by_chapter.setdefault(t.get("chapter", "unknown"), []).append(t)

    chapters = sorted(by_chapter.keys())
    n_eval_chapters = max(1, int(len(chapters) * test_ratio))
    eval_chapters = set(chapters[-n_eval_chapters:])
    print(f"[Reranker] Eval chapters ({n_eval_chapters}/{len(chapters)}): "
          f"{sorted(eval_chapters)}")

    train_data = [t for t in triplets if t.get("chapter") not in eval_chapters]
    test_data  = [t for t in triplets if t.get("chapter") in eval_chapters]

    # Strip chapter field — not needed by the trainer
    for rec in train_data + test_data:
        rec.pop("chapter", None)

    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    with open(output_train, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(output_test, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    print(f"[Reranker] Train: {len(train_data)} triplets → {output_train}")
    print(f"[Reranker] Test:  {len(test_data)} triplets → {output_test}")
    print(f"[Reranker] Stage 2 done in {elapsed/60:.1f} min.", flush=True)


# =============================================================================
# Build SFT training data
# =============================================================================

def build_sft_data(
    qa_pairs: List[dict],
    output_file: str,
    system_prompt: str = None,
) -> None:
    """
    Build SFT training data in ChatML format for Llama fine-tuning.

    Each record uses LLaMA-Factory native ShareGPT format:
    {
        "messages": [
            {"from": "system", "value": "..."},
            {"from": "human",  "value": "Context:\n...\n\nQuestion: ..."},
            {"from": "gpt",    "value": "..."}
        ]
    }
    """
    if system_prompt is None:
        system_prompt = (
            "You are an expert immunology assistant. "
            "Answer questions accurately using the provided context. "
            "Cite relevant information and use proper immunological terminology."
        )

    print(f"\n[SFT] Building SFT data for {len(qa_pairs)} QA pairs...")
    records = []

    for qa in qa_pairs:
        user_content = (
            f"Context:\n{qa['passage']}\n\n"
            f"Question: {qa['question']}"
        )
        records.append({
            "messages": [
                {"from": "system", "value": system_prompt},
                {"from": "human",  "value": user_content},
                {"from": "gpt",    "value": qa["answer"]},
            ]
        })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[SFT] Stage 3 done. {len(records)} records → {output_file}")


# =============================================================================
# Build eval QA set
# =============================================================================

def build_eval_data(
    qa_pairs: List[dict],
    output_file: str,
    n_eval: int = 150,
) -> None:
    """
    Build evaluation QA set with ground-truth answers.
    Samples evenly across chapters/sources for coverage.

    Format:
    {
        "question":    str,
        "answer":      str,
        "source_file": str,
        "chapter":     str,
        "page":        int,
        "unique_id":   str,
    }
    """
    # Stratified sampling by chapter
    by_chapter: Dict[str, list] = {}
    for qa in qa_pairs:
        ch = qa.get("chapter", "unknown")
        by_chapter.setdefault(ch, []).append(qa)

    n_chapters = len(by_chapter)
    per_chapter = max(1, n_eval // n_chapters)

    eval_pairs = []
    for ch_qas in by_chapter.values():
        sampled = random.sample(ch_qas, min(per_chapter, len(ch_qas)))
        eval_pairs.extend(sampled)

    # Trim to n_eval
    random.shuffle(eval_pairs)
    eval_pairs = eval_pairs[:n_eval]

    # Save (exclude 'passage' field to reduce file size)
    records = []
    for qa in eval_pairs:
        records.append({
            "question":    qa["question"],
            "answer":      qa["answer"],
            "source_file": qa.get("source_file", ""),
            "chapter":     qa.get("chapter", ""),
            "page":        qa.get("page", ""),
            "unique_id":   qa.get("unique_id", ""),
        })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Eval] Stage 4 done. {len(records)} pairs → {output_file}")


# =============================================================================
# Main
# =============================================================================

def main(
    n_qa: int = 5,
    n_para: int = 5,
    n_kw: int = 5,
    workers: int = 4,
    limit: Optional[int] = None,
    spot_check: int = 10,
    force: bool = False,
) -> None:
    """
    Full training data generation pipeline with checkpoint/resume support.

    Each stage saves its output immediately when done. On re-run, any stage
    whose output file already exists is skipped automatically. Use --force
    to redo everything from scratch.

    Stages:
        1. QA generation   → data/train/qa_pairs_cache.jsonl   (~3-4 h, needs vLLM)
        2. Reranker data   → data/train/reranker_train.jsonl   (< 5 min, BM25 only)
        3. SFT data        → data/train/sft_train.jsonl        (< 1 min)
        4. Eval QA set     → data/train/eval_qa.jsonl          (< 1 min)

    Args:
        n_qa:       QA pairs per chunk
        n_para:     Paraphrase variants per question (unused in current pipeline)
        n_kw:       Keywords per answer (unused in current pipeline)
        workers:    Parallel workers for LLM calls (used in stages 1 and 2)
        limit:      Max chunks to process (None = all); for smoke tests
        spot_check: Number of random QA samples to print after stage 1
        force:      If True, redo all stages even if output files exist
    """
    train_dir = Path(constant.train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    qa_cache_path      = train_dir / "qa_pairs_cache.jsonl"
    reranker_train_path = train_dir / "reranker_train.jsonl"
    reranker_test_path  = train_dir / "reranker_test.jsonl"
    sft_train_path     = train_dir / "sft_train.jsonl"
    eval_qa_path       = train_dir / "eval_qa.jsonl"

    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 1 — QA pair generation (or load from checkpoint)
    # ------------------------------------------------------------------
    if not force and qa_cache_path.exists():
        print(f"[Stage 1] ✓ QA cache found: {qa_cache_path}")
        print(f"[Stage 1]   Loading from cache (skipping generation)...")
        with open(qa_cache_path, "r", encoding="utf-8") as f:
            all_qa = [json.loads(line) for line in f if line.strip()]
        print(f"[Stage 1] ✓ Loaded {len(all_qa)} QA pairs from cache.")
    else:
        if force and qa_cache_path.exists():
            print(f"[Stage 1] --force: removing existing cache and re-generating.")
            qa_cache_path.unlink()

        # Load chunks
        collection = MongoConfig.get_collection(constant.mongo_collection)
        all_chunks = list(collection.find({}, {"_id": 0}))
        print(f"[Stage 1] Loaded {len(all_chunks)} chunks from MongoDB.")

        if limit:
            all_chunks = all_chunks[:limit]
            print(f"[Stage 1] Limited to {limit} chunks.")

        if not all_chunks:
            print("[Stage 1] No chunks found. Run build_index.py first.")
            return

        print(f"\n[Stage 1] Generating QA pairs with {workers} workers...")
        all_qa: List[dict] = []
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(generate_qa_pairs, chunk, n_qa): chunk
                for chunk in all_chunks
            }
            completed = 0
            for future in as_completed(futures):
                try:
                    qa_list = future.result()
                    all_qa.extend(qa_list)
                except Exception as e:
                    print(f"[Stage 1] Warning: QA generation failed: {e}")
                completed += 1
                if completed % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(all_chunks) - completed) / rate if rate > 0 else 0
                    print(f"[Stage 1] Progress: {completed}/{len(all_chunks)} chunks, "
                          f"{len(all_qa)} QA pairs, {elapsed:.0f}s elapsed "
                          f"(~{remaining/60:.0f} min remaining)", flush=True)

        elapsed = time.perf_counter() - t0
        print(f"\n[Stage 1] Generated {len(all_qa)} QA pairs in {elapsed/60:.1f} min.")

        # Save checkpoint immediately
        with open(qa_cache_path, "w", encoding="utf-8") as f:
            for qa in all_qa:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        print(f"[Stage 1] ✓ QA pairs saved to checkpoint: {qa_cache_path}", flush=True)

    # Spot-check: print samples to stdout (non-blocking)
    if spot_check > 0 and all_qa:
        print(f"\n{'='*60}")
        print(f"SPOT CHECK: {min(spot_check, len(all_qa))} random samples")
        print("="*60)
        samples = random.sample(all_qa, min(spot_check, len(all_qa)))
        for i, qa in enumerate(samples, 1):
            print(f"\n[{i}] Source: {qa.get('source_file', '?')} | "
                  f"Chapter: {qa.get('chapter', '?')} | Page: {qa.get('page', '?')}")
            print(f"    Q: {qa['question']}")
            print(f"    A: {qa['answer'][:150]}...")
        print("="*60)

    # ------------------------------------------------------------------
    # Load all_chunks for hard-negative generation (stage 2)
    # Only needed if reranker files don't exist yet
    # ------------------------------------------------------------------
    need_reranker = force or not reranker_train_path.exists()
    if need_reranker:
        collection = MongoConfig.get_collection(constant.mongo_collection)
        all_chunks = list(collection.find({}, {"_id": 0}))

    # ------------------------------------------------------------------
    # Stage 2 — Reranker triplets
    # ------------------------------------------------------------------
    if not force and reranker_train_path.exists():
        print(f"\n[Stage 2] ✓ Reranker data found: {reranker_train_path} — skipping.")
    else:
        if force:
            for p in (reranker_train_path, reranker_test_path):
                if p.exists():
                    p.unlink()
        build_reranker_data(
            all_qa, all_chunks,
            output_train=str(reranker_train_path),
            output_test=str(reranker_test_path),
        )

    # ------------------------------------------------------------------
    # Stage 3 — SFT data
    # ------------------------------------------------------------------
    if not force and sft_train_path.exists():
        print(f"\n[Stage 3] ✓ SFT data found: {sft_train_path} — skipping.")
    else:
        if force and sft_train_path.exists():
            sft_train_path.unlink()
        build_sft_data(
            all_qa,
            output_file=str(sft_train_path),
        )

    # ------------------------------------------------------------------
    # Stage 4 — Eval QA set
    # ------------------------------------------------------------------
    if not force and eval_qa_path.exists():
        print(f"\n[Stage 4] ✓ Eval data found: {eval_qa_path} — skipping.")
    else:
        if force and eval_qa_path.exists():
            eval_qa_path.unlink()
        build_eval_data(
            all_qa,
            output_file=str(eval_qa_path),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"[TrainData] All stages complete.")
    print(f"[TrainData] Output directory: {train_dir}")
    for path in [qa_cache_path, reranker_train_path, reranker_test_path,
                 sft_train_path, eval_qa_path]:
        if path.exists():
            lines = sum(1 for _ in open(path, encoding="utf-8"))
            size_mb = path.stat().st_size / 1_048_576
            print(f"  {path.name:<30} {lines:>6} lines  {size_mb:.1f} MB")
    print(f"[TrainData] Total wall time: {total_elapsed/60:.1f} min")
    print(f"{'='*60}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ImmunoBiology RAG — Generate training data from indexed chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Checkpoint/resume behaviour
---------------------------
Each stage saves its output as soon as it completes.
If you re-run after a crash or disconnection, already-finished stages are
skipped automatically based on whether their output file exists.

  Stage 1  data/train/qa_pairs_cache.jsonl   (QA generation,   ~3-4 h, needs vLLM)
  Stage 2  data/train/reranker_train.jsonl   (BM25 negatives,  < 5 min, no vLLM)
  Stage 3  data/train/sft_train.jsonl        (SFT formatting,  < 1 min)
  Stage 4  data/train/eval_qa.jsonl          (eval set,        < 1 min)

Use --force to delete all checkpoints and start from scratch.
        """,
    )
    parser.add_argument("--n-qa",    type=int, default=5,    help="QA pairs per chunk (default: 5)")
    parser.add_argument("--n-para",  type=int, default=5,    help="Paraphrase variants (default: 5)")
    parser.add_argument("--n-kw",    type=int, default=5,    help="Keywords per answer (default: 5)")
    parser.add_argument("--workers", type=int, default=4,    help="Parallel worker threads (default: 4)")
    parser.add_argument("--limit",   type=int, default=None, help="Max chunks to process (smoke test)")
    parser.add_argument("--sample",  type=int, default=10,   help="Spot-check samples to print (0 = off)")
    parser.add_argument("--force",   action="store_true",    help="Re-run all stages even if output files exist")

    args = parser.parse_args()

    main(
        n_qa=args.n_qa,
        n_para=args.n_para,
        n_kw=args.n_kw,
        workers=args.workers,
        limit=args.limit,
        spot_check=args.sample,
        force=args.force,
    )
