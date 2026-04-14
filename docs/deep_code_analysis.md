# ImmunoBiology RAG — Deep Code Analysis Guide

A module-by-module walkthrough of the entire ImmunoBiology RAG system.
This guide covers beginner-friendly explanations, key classes and functions,
inter-module data flow, design decisions, and practical tips for every layer
of the pipeline.

---

## Table of Contents

1. [What is RAG? (Beginner Primer)](#1-what-is-rag-beginner-primer)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Configuration Layer](#3-configuration-layer)
4. [PDF Parsing](#4-pdf-parsing)
5. [Semantic Chunking Service](#5-semantic-chunking-service)
6. [Chunker — Parent-Child Strategy](#6-chunker--parent-child-strategy)
7. [Embedder & Index Building](#7-embedder--index-building)
8. [Hybrid Retrieval](#8-hybrid-retrieval)
9. [Reranker](#9-reranker)
10. [LLM Inference](#10-llm-inference)
11. [Pipeline Orchestration](#11-pipeline-orchestration)
12. [Utilities](#12-utilities)
13. [Training Data Generation](#13-training-data-generation)
14. [Fine-Tuning: Reranker](#14-fine-tuning-reranker)
15. [Fine-Tuning: LLM SFT (LoRA)](#15-fine-tuning-llm-sft-lora)
16. [Evaluation](#16-evaluation)
17. [Streamlit UI](#17-streamlit-ui)
18. [End-to-End Data Flow](#18-end-to-end-data-flow)
19. [Common Questions from Beginners](#19-common-questions-from-beginners)

---

## 1. What is RAG? (Beginner Primer)

### The Problem RAG Solves

Large Language Models (LLMs) like Qwen3-8B are trained on massive amounts of text and
can answer general questions. However, they have two critical limitations:

1. **Knowledge cutoff:** They don't know about papers or textbooks published after
   their training data was collected.
2. **Hallucination:** When asked about something they don't know well, they confidently
   generate plausible-sounding but incorrect answers.

### The RAG Solution

**Retrieval-Augmented Generation (RAG)** solves this by giving the LLM a "cheat sheet"
at question time:

```
User question: "What cytokines do Th2 cells secrete?"
                       │
              ┌────────▼─────────┐
              │  Search the      │  ← Retrieval step: find relevant passages
              │  knowledge base  │    from Janeway's Immunobiology
              └────────┬─────────┘
                       │
              Passages found:
              "Th2 cells produce IL-4, IL-5, IL-13..."
              "IL-4 promotes B cell class switching to IgE..."
                       │
              ┌────────▼──────────────┐
              │  LLM reads the        │  ← Generation step: LLM uses the retrieved
              │  passages + question  │    passages to write a grounded answer
              └────────┬──────────────┘
                       │
              Answer: "Th2 cells secrete IL-4, IL-5, and IL-13. [1][2]"
```

The LLM is now **grounded in real source text** and can cite specific pages from
Janeway's Immunobiology. Hallucinations are dramatically reduced.

### Why "Hybrid" Retrieval?

Two search strategies find complementary matches:

| Method | Finds | Example |
|--------|-------|---------|
| **BM25** (keyword) | Exact term matches | "MHC class II" → finds passages containing that exact phrase |
| **Dense** (semantic) | Meaning-level matches | "antigen presentation" → finds passages about the same concept even if the wording differs |

Combining both with **Reciprocal Rank Fusion (RRF)** gives better recall than either alone.

### Why Fine-Tune?

The base Qwen3-8B model knows general science but has never been specifically optimized
for immunology Q&A format or ranking immunology passages by relevance. Fine-tuning adapts:

- **Reranker:** Learns to distinguish highly relevant immunology passages from
  tangentially related ones
- **LLM:** Learns the expected answer format (inline citations `[1]`, academic register,
  concise but complete answers)

---

## 2. System Architecture Overview

```
data/raw/*.pdf
      │
      ▼ [src/pdf_parser.py]
data/processed/{stem}/text/*.json + images/
      │
      ▼ [src/chunker.py] ──────────────────────► MongoDB
                                                 (parent + child chunks)
      │
      ▼ [src/embedder.py]
      ├── Chroma (dense vectors, BGE-M3)
      └── BM25 pickle (sparse index)

                    ┌─────────────────────────────────────────────┐
                    │           RAGPipeline.answer(query)          │
                    │                                             │
  User query ──►   │  [Optional] HyDE expansion                  │
                    │       │                                     │
                    │  BM25 retrieval ──┐                         │
                    │  Dense retrieval ─┤─► RRF Fusion            │
                    │                   │       │                  │
                    │              merge_docs() (MongoDB lookup)  │
                    │                   │                         │
                    │           Cross-encoder Reranking           │
                    │                   │                         │
                    │           LLM Generation (vLLM)             │
                    │                   │                         │
                    │           Post-processing + Citations       │
                    └─────────────────────────────────────────────┘
                                        │
                              {answer, sources, images, latency}
```

### Key Design Principles

1. **Modular strategy pattern:** Every component (retriever, reranker, LLM) is a class
   with a defined interface. You can swap BGE-M3 for a different embedder, or Chroma for
   FAISS, without changing `pipeline.py`.

2. **Parent-child chunking:** Small child chunks (≤512 tokens) are retrieved for precise
   matching. Their larger parent chunks (~512 words of context) are passed to the LLM for
   more complete answers. This balances retrieval precision with generation quality.

3. **MongoDB as the source of truth:** All chunk metadata, parent-child relationships,
   and figure references are stored in MongoDB. ChromaDB stores only vector embeddings.
   This separation makes it easy to update metadata without re-embedding everything.

4. **CPU-GPU separation:** The semantic chunking service (lightweight MiniLM model) runs
   on CPU only, keeping the entire GPU free for vLLM during inference.

---

## 3. Configuration Layer

### `config.yaml` — Single Source of Truth

Every tunable parameter lives here. Modules never hardcode values.

```yaml
# Key sections and what they control:

models:
  llm: Qwen/Qwen3-8B           # which LLM vLLM serves
  embedding: BAAI/bge-m3       # which model encodes chunks to vectors
  reranker: BAAI/bge-reranker-v2-m3   # which cross-encoder reranks results

retrieval:
  bm25_topk: 10    # how many BM25 results to fetch
  dense_topk: 10   # how many dense results to fetch
  rerank_topk: 5   # how many to keep after reranking
  rrf_k: 60        # RRF smoothing constant (higher = more uniform blending)

llm:
  max_tokens: 1024   # IMPORTANT: keep this low enough so prompt+answer < 8192 tokens
  temperature: 0.001 # very low = deterministic factual answers (good for RAG)
```

### `src/constant.py` — Python Interface to config.yaml

```python
# How it works: loaded once at import time
import yaml
_cfg = yaml.safe_load(open("config.yaml"))

# Path helpers
BASE_DIR = Path(__file__).resolve().parents[1]
def _p(rel): return str(BASE_DIR / rel)

# All constants derived from config:
llm_model_name  = _cfg["models"]["llm"]          # "Qwen/Qwen3-8B"
bm25_topk       = _cfg["retrieval"]["bm25_topk"]  # 10
llm_temperature = _cfg["llm"]["temperature"]       # 0.001
```

**Why module-level variables?** Simple `from src import constant; constant.bm25_topk`
everywhere, without passing config objects through function calls.

**Hot-reload in Settings UI:** `pages/04_Settings.py` calls `importlib.reload(const_mod)`
after saving `config.yaml`, so pipeline parameters update without restarting Streamlit.

---

## 4. PDF Parsing

**File:** `src/pdf_parser.py`

### What It Does

Reads each PDF page-by-page using PyMuPDF (`fitz`), extracts text and images,
detects chapter headings, and writes structured JSON per page.

### Layout Inspection First

Before full parsing, `--inspect` mode samples 4 pages (1, 50, 100, 200) and writes
`outputs/diagnostics/pdf_layout_report.txt`. This tells you:
- Page dimensions in points (1 point = 1/72 inch)
- Where headers and footers appear (so they can be cropped out)
- Whether the PDF is single-column (textbook) or double-column (paper)
- Whether OCR is needed (scanned vs digital PDF)

Janeway's Immunobiology 10e is a single-column textbook at 595×842pt (A4 size) with
a 50pt header and 40pt footer — these crop values are set as defaults.

### Key Functions

| Function | What it does | Why it's needed |
|----------|-------------|-----------------|
| `inspect_pdf_layout()` | Sample pages, write layout report | Calibrate crop values before full parse |
| `detect_layout(page)` | Single vs double column | Different PDFs need different text extraction |
| `detect_chapter(blocks)` | Find chapter headings via regex | Metadata for citation: "Chapter 3" |
| `handle_image(page, img_info)` | Extract + filter images | Attach figures to nearby passages |
| `parse_pdf(pdf_path)` | Full extraction per page | Main parsing entry point |
| `load_all_pdfs(pdf_paths)` | Multi-PDF loop | Processes all PDFs in `data/raw/` |

### Image Extraction Thresholds (Calibrated for Janeway)

```python
IMAGE_MIN_WIDTH  = 100   # px — skips tiny inline icons and decorative elements
IMAGE_MIN_HEIGHT = 100   # px
IMAGE_MIN_BYTES  = 5000  # ~5 KB — skips low-quality compressed images

# Figure caption detection
FIGURE_CAPTION_RE = r'(Figure|FIGURE|Fig\.?)\s+\d+[\.\-]\d*'
```

### Output Structure

```
data/processed/JanewaysImmunobiologyBiology10thEdition/
  ├── text/
  │   ├── 0001.json    {"text": "...", "page": 1, "chapter": null, "doc_type": "textbook", ...}
  │   ├── 0002.json
  │   └── ...
  └── images/
      ├── fig_3_1.png
      └── ...
```

### OCR Support

If a page contains no extractable text (scanned PDF), the code falls back to
`pytesseract.image_to_string()` automatically. This requires `tesseract-ocr` to be
installed on the system (`apt-get install -y tesseract-ocr tesseract-ocr-eng`).

---

## 5. Semantic Chunking Service

**Files:** `src/server/semantic_chunk.py` (FastAPI service) + `src/client/semantic_chunk_client.py` (HTTP client)

### Why a Separate Service?

Simple character-based splitting (`"split every 512 chars"`) cuts text in the middle
of ideas. Semantic chunking detects natural topic boundaries before splitting.

Running this as a **separate microservice** (not inline in the main code) means:
- It stays running across multiple index builds without reloading the model
- It runs on CPU, keeping GPU free for vLLM
- It can be restarted independently if it crashes

### How the Algorithm Works

```
Input text: "T cells are activated by MHC-peptide complexes. This triggers
            clonal expansion. B cells, in contrast, recognize native antigen
            directly through BCR. They then undergo affinity maturation..."
                    │
          Step 1: Split by \n\n (paragraph boundaries)
          ["T cells are activated...", "B cells, in contrast..."]
                    │
          Step 2: Encode each paragraph with all-MiniLM-L6-v2 (384-dim vectors)
          [vector_1, vector_2]
                    │
          Step 3: Agglomerative clustering (cosine similarity)
          Nearby paragraphs in meaning-space are merged into groups
                    │
          Step 4: Merge tiny fragments < 100 characters
                    │
          Output: list of semantic chunks (typically 2-5 paragraphs each)
```

**Why `all-MiniLM-L6-v2` instead of BGE-M3 here?**
- MiniLM is 90 MB vs BGE-M3's 2.5 GB
- For boundary detection between adjacent paragraphs, the lightweight model is
  accurate enough
- The full BGE-M3 model is reserved for the final semantic search embeddings

**Key config:** `chunking.semantic_group_size: 15` — how many paragraphs to group
before clustering. English paragraphs are longer (more words) than Chinese sentences,
so 15 is appropriate (vs 10 for Chinese text).

### Service Endpoints

```
GET  /health    → {"status": "ok", "model": "...all-MiniLM-L6-v2"}
POST /v1/semantic-chunks
     Body: {"text": "...", "group_size": 15}
     Response: {"chunks": ["chunk1", "chunk2", ...]}
```

---

## 6. Chunker — Parent-Child Strategy

**File:** `src/chunker.py`

### The Parent-Child Concept

```
Semantic parent chunk (~512 words):
"T cells are activated when their T cell receptor (TCR) binds to an
MHC-peptide complex on the surface of an antigen-presenting cell (APC).
This interaction, combined with co-stimulatory signals from CD28-B7
binding, triggers intracellular signaling cascades that activate
transcription factors such as NF-κB and NFAT. These factors drive..."

        ├── Child chunk 1 (≤512 tokens):
        │   "T cells are activated when their T cell receptor (TCR)
        │    binds to an MHC-peptide complex on the surface of an
        │    antigen-presenting cell (APC)..."
        │
        ├── Child chunk 2 (≤512 tokens, 100-token overlap with child 1):
        │   "...antigen-presenting cell (APC). This interaction, combined
        │    with co-stimulatory signals from CD28-B7 binding, triggers..."
        │
        └── Child chunk 3:
            "...NF-κB and NFAT. These factors drive..."
```

**Retrieval** works on child chunks — they are small and precise, giving accurate
BM25/dense matches.

**LLM context** uses parent chunks — they provide more context so the LLM can write
a complete, coherent answer.

This is why `merge_docs()` in `utils.py` looks up the `parent_id` in MongoDB after
retrieval: it fetches the full parent passage for the LLM, not just the small child
chunk that matched.

### `texts_split()` Function

```python
def texts_split(page_records) -> List[Document]:
    """
    Full pipeline: page records → semantic parent chunks → child chunks → MongoDB storage.
    Returns child documents (used for indexing into Chroma + BM25).
    """
    # 1. Call semantic chunking service → parent documents
    parents = _semantic_chunk(page_records)

    # 2. Apply RecursiveCharacterTextSplitter → child documents
    children = _recursive_split(parents)

    # 3. Assign chunk_id: "{stem}_ch{N}_p{page}_{seq}"
    # 4. Compute unique_id = MD5(page_content) for deduplication
    # 5. Save ALL (parents + children) to MongoDB
    _save_to_mongodb(parents, children)

    return children  # only children go into the vector index
```

### MongoDB Document Schema

```json
{
  "_id": "ObjectId",
  "page_content": "T cells are activated when...",
  "metadata": {
    "chunk_id":    "janeway10e_ch3_p87_001",
    "unique_id":   "a3f8c2...",      // MD5 hash — used for dedup
    "parent_id":   "a1b2c3...",      // parent's unique_id (null if this IS the parent)
    "source_file": "JanewaysImmunobiologyBiology10thEdition.pdf",
    "doc_type":    "textbook",
    "chapter":     "Chapter 3",
    "page":        87,
    "is_parent":   false,
    "images_info": []
  }
}
```

---

## 7. Embedder & Index Building

**File:** `src/embedder.py`

### Three-Layer Index

The system maintains three synchronized indices. They must all be built together and
stay in sync:

| Index | File | Contains | Used for |
|-------|------|----------|---------|
| **Chroma** | `outputs/vectorstore/chroma/` | Dense 1024-dim vectors for each child chunk | Semantic similarity search |
| **BM25 pickle** | `outputs/vectorstore/bm25_index.pkl` | Inverted keyword index over all child chunks | Exact keyword search |
| **MongoDB** | `/data/db/` | Full text + metadata for all chunks | Parent lookup, metadata filtering |

### BGE-M3 Embedder

```python
class BGEEmbedder:
    def encode_docs(self, texts: List[str]) -> List[List[float]]:
        """Batch encode documents — called during index building."""
        return self.model.encode(texts, batch_size=64, max_length=512)["dense_vecs"]

    def encode_query(self, query: str) -> List[float]:
        """Single query encode — called during retrieval."""
        return self.model.encode_queries([query])["dense_vecs"][0]
```

**Why different methods for docs vs queries?** BGE-M3 supports asymmetric encoding —
queries and documents can be encoded slightly differently for better match accuracy.
`encode_queries()` prepends a special instruction token to the query.

### BM25 Tokenization (English-Specific)

```python
def english_tokenize(text: str) -> List[str]:
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stop_words]
```

This replaces Tesla's `jieba.lcut()` (Chinese word segmentation). NLTK's `word_tokenize`
handles English contractions, hyphenated terms, and immunology abbreviations correctly.

### Deduplication Strategy

When `build_index.py` is run on an already-indexed PDF, it should not create duplicate
embeddings. The `ChromaStore` checks existing `chunk_id` values before adding:

```python
existing_ids = set(collection.get(include=[])["ids"])
new_docs = [doc for doc in docs if doc.metadata["chunk_id"] not in existing_ids]
if new_docs:
    collection.add(...)  # only add truly new chunks
```

The same `chunk_id` is a deterministic hash of `(source_file, page, sequence_number)`,
so re-running on the same PDF always produces the same IDs.

---

## 8. Hybrid Retrieval

### BM25 Retriever (`src/retriever/bm25_retriever.py`)

BM25 (Best Matching 25) is a sophisticated variant of TF-IDF. For each term in the
query, it scores documents based on:
- **Term frequency** in the document (diminishing returns at high frequency)
- **Inverse document frequency** (rare terms are more discriminating)
- **Document length normalization** (penalizes very long documents slightly)

```python
class BM25Retriever(BaseRetriever):
    def retrieve_topk(self, query: str, topk: int = 10) -> List[Document]:
        query_tokens = english_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:topk]
        return [self.documents[i] for i in top_indices if scores[i] > 0]
```

**When BM25 wins:** Queries containing specific technical terms ("CD4+ T lymphocyte",
"MHC class II HLA-DR") that must appear verbatim.

### Chroma Dense Retriever (`src/retriever/chroma_retriever.py`)

Encodes the query with BGE-M3, then performs approximate nearest-neighbor search in the
1024-dimensional vector space stored in Chroma.

**When dense retrieval wins:** Queries phrased differently from the text ("immune cell
activation" → finds passages about "lymphocyte stimulation", "T cell triggering", etc.).

**Metadata filtering:** The retriever supports filtering results by `doc_type` (textbook
vs paper) or `source_file` — accessible via the Q&A sidebar filters.

### RRF Fusion (`src/pipeline.py`)

```python
def rrf_fuse(bm25_results, dense_results, k=60, bm25_w=0.5, dense_w=0.5):
    scores = {}
    for rank, doc in enumerate(bm25_results, 1):
        scores[doc.unique_id] += bm25_w * (1 / (k + rank))
    for rank, doc in enumerate(dense_results, 1):
        scores[doc.unique_id] += dense_w * (1 / (k + rank))
    return sorted(scores.items(), key=lambda x: -x[1])
```

**Why RRF instead of score normalization?**
BM25 and dense scores live on completely different scales (BM25 scores can be 5-50;
cosine similarity is -1 to 1). Normalizing them requires calibration. Rank-based
fusion (RRF) is scale-invariant — it only cares about the rank of each result, not
the raw score. This is more robust across different query types.

**The k=60 constant:** Prevents a rank-1 result from dominating too much.
At k=60, rank 1 contributes 1/61 ≈ 0.016; rank 10 contributes 1/70 ≈ 0.014.
Low k values make results more sensitive to top ranks; high k values blend results
more uniformly.

### HyDE — Hypothetical Document Embeddings

```
Original query: "how do NK cells recognize infected cells?"

HyDE expansion:
"NK cells recognize infected cells through a balance of activating
and inhibitory receptors. Activating receptors like NKG2D bind to
stress ligands such as MICA/MICB that are upregulated on infected
cells. Inhibitory receptors like KIR recognize MHC class I molecules..."

Dense search is now run on this hypothetical passage
instead of (or in addition to) the raw query.
```

HyDE bridges the vocabulary gap between short queries and longer document passages —
the expanded text uses academic vocabulary that matches the textbook's writing style.

---

## 9. Reranker

### Why Reranking?

The first-stage retrieval (BM25 + dense) fetches the top 10-20 documents quickly using
approximate methods. But these approximate methods can make mistakes:
- BM25 might rank a document highly just because it repeats the query terms many times
- Dense retrieval might retrieve semantically similar but factually irrelevant text

The **cross-encoder reranker** is a slower, more accurate model that reads the entire
(query, document) pair together — like a human reading both and deciding if this passage
actually answers the question.

### BGE Reranker v2-M3 (`src/reranker/bge_m3_reranker.py`)

```python
class BGEM3ReRanker(RerankerBase):
    def rank(self, query: str, candidate_docs: List[Document], top_k: int = 5) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in candidate_docs]
        inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                max_length=4096, return_tensors="pt").to("cuda")
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().float().numpy().flatten()

        # ⚠️ Critical: free GPU tensors immediately after scoring.
        # inputs can be 100–400 MB per batch; without this, evaluate.py OOMs
        # when vLLM is also running (~34.6 GB) and only ~296 MiB is free.
        del inputs
        torch.cuda.empty_cache()

        ranked = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)[:top_k]
        return [doc for _score, doc in ranked]
```

The model takes concatenated `[query] [SEP] [document]` as input and outputs a single
relevance score. It can capture subtle semantic mismatches that pure vector similarity misses.

**Speed trade-off:** Cross-encoder reranking is O(N × document_length) — much slower
than vector search. This is why it's run on only the top-N candidates from RRF fusion,
not all documents.

**GPU memory note:** `max_length=4096` allows long immunology passages but means each
batch of 10-20 pairs can hold 100–400 MB on CUDA. The explicit `del inputs;
torch.cuda.empty_cache()` after scoring is required to prevent OOM during `evaluate.py`
(which runs the reranker repeatedly while vLLM is also occupying ~34.6 GB of the same GPU).

---

## 10. LLM Inference

### vLLM as the Serving Layer

vLLM implements **continuous batching** and **PagedAttention** — two optimizations that
allow the same GPU to handle many requests simultaneously, even when each request has
different context lengths. For a RAG system where every query has a different prompt
(different retrieved passages), this is critical for throughput.

### System Prompt (English Immunology Domain)

```
You are an expert immunology assistant with deep knowledge of immunological principles,
cellular and molecular immunology, and clinical immunology. Answer questions using ONLY
the provided context passages (numbered [1]-[N]).

Rules:
- Cite passages inline: "T cells are activated [1]" or "multiple mechanisms [2,3]"
- If context is insufficient, say: "The knowledge base does not contain enough
  information to answer this question."
- Write in clear academic English. Be concise but complete.
- Do NOT invent mechanisms or cite papers not in the context.
```

### `llm_client.py` — How the API Call Works

```python
def generate(self, query: str, context: str, history: list) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        # Optional: previous turns from chat history
        *[{"role": m["role"], "content": m["content"]} for m in history],
        # Current query with retrieved context
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    response = self.client.chat.completions.create(
        model=constant.llm_model_name,    # "Qwen/Qwen3-8B"
        messages=messages,
        max_tokens=constant.llm_max_tokens,  # 1024
        temperature=constant.llm_temperature, # 0.001
        extra_body={"chat_template_kwargs": {"enable_thinking": constant.enable_thinking}}
    )
    return response.choices[0].message.content
```

**`enable_thinking: false`** — Qwen3 has a chain-of-thought mode where it outputs a
`<think>...</think>` block before the answer. For RAG, this is unnecessary and wastes
tokens. Setting it to `false` suppresses this behavior.

### HyDE Client (`llm_hyde_client.py`)

Uses a different prompt to generate a hypothetical passage rather than an answer:

```
Generate a detailed immunology textbook passage that would directly answer
this question: "{query}"
Write in the style of Janeway's Immunobiology. ~100 words.
```

Temperature is slightly higher (0.3) than the main client (0.001) to allow some
vocabulary variation — the goal is a passage with diverse immunology vocabulary that
will match more documents.

### Data Generation Client (`llm_datagen_client.py`)

Used only during `build_train_data.py`. Higher temperature (0.85) for diverse QA
generation, retry logic with exponential backoff for failed API calls.

---

## 11. Pipeline Orchestration

**File:** `src/pipeline.py`

### `RAGPipeline` Class — The Full Flow

```python
class RAGPipeline:
    def answer(self, query: str, doc_type=None, source_file=None) -> dict:
        tracker = LatencyTracker()

        # Step 1: Optional HyDE expansion
        search_query = query
        if constant.hyde_enabled:
            tracker.start("hyde")
            search_query = self.hyde_client.generate(query)
            tracker.stop("hyde")

        # Step 2-3: Parallel retrieval (BM25 + Dense)
        tracker.start("retrieval")
        bm25_docs  = self.bm25_retriever.retrieve_topk(search_query, constant.bm25_topk)
        dense_docs = self.dense_retriever.retrieve_topk(search_query, constant.dense_topk,
                                                         doc_type=doc_type, source_file=source_file)
        tracker.stop("retrieval")

        # Step 4: RRF fusion
        fused_docs = rrf_fuse(bm25_docs, dense_docs,
                               k=constant.rrf_k,
                               bm25_w=constant.rrf_bm25_weight,
                               dense_w=constant.rrf_dense_weight)

        # Step 5: Merge + parent lookup
        tracker.start("merge")
        merged = merge_docs(fused_docs, [])  # looks up parent chunks in MongoDB
        tracker.stop("merge")

        # Step 6: Cross-encoder reranking
        tracker.start("rerank")
        reranked = self.reranker.rerank(query, merged, topk=constant.rerank_topk)
        tracker.stop("rerank")

        # Step 7: Format context as numbered passages
        context = format_context(reranked)

        # Step 8: LLM generation
        tracker.start("llm")
        answer_text = self.llm_client.generate(query, context, self.chat_history)
        tracker.stop("llm")

        # Step 9: Post-processing — extract citations, map to metadata
        result = post_processing(answer_text, reranked)

        # Step 10: Update chat history for multi-turn conversation
        self._update_history(query, answer_text)

        result["latency_ms"] = tracker.get_all()
        result["retrieved_docs"] = reranked
        return result
```

### Multi-Turn Conversation

```python
def _update_history(self, query: str, answer: str):
    self.chat_history.append({"role": "user",    "content": query})
    self.chat_history.append({"role": "assistant","content": answer})
    # Keep only the last N turns
    window = constant.llm_history_window * 2  # pairs of user+assistant
    if len(self.chat_history) > window:
        self.chat_history = self.chat_history[-window:]
```

Each turn is appended to the `messages` list sent to vLLM, so the LLM can refer to
previous answers ("As I mentioned earlier about NK cells...").

---

## 12. Utilities

**File:** `src/utils.py`

### `merge_docs(docs1, docs2)` — Parent Lookup

```python
def merge_docs(docs1: List[Document], docs2: List[Document]) -> List[Document]:
    """
    1. Combine and deduplicate by unique_id
    2. For child chunks: fetch their parent from MongoDB
    3. Return parent documents (with richer context) for the LLM
    """
    all_docs = {doc.metadata["unique_id"]: doc for doc in docs1 + docs2}
    parents = []
    for doc in all_docs.values():
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            parent = mongodb_lookup(parent_id)  # fetch larger context chunk
            if parent: parents.append(parent)
        else:
            parents.append(doc)  # already a parent/standalone
    return deduplicate(parents)
```

### `post_processing(response, docs)` — Citation Extraction

```python
# Finds [1], [2,3], [1][4] citation patterns in the LLM output
CITATION_RE = r'\[(\d+(?:[,\s]+\d+)*)\]'

def post_processing(answer: str, docs: List[Document]) -> dict:
    cited_indices = extract_citation_indices(answer, CITATION_RE)
    cite_pages    = [docs[i-1].metadata.get("page") for i in cited_indices if i <= len(docs)]
    cite_sources  = [docs[i-1].metadata.get("source_file") for i in cited_indices]
    images        = collect_related_images(cited_indices, docs)
    return {
        "answer":        answer,
        "cite_pages":    cite_pages,
        "cite_sources":  cite_sources,
        "related_images": images,
    }
```

### `LatencyTracker` — Per-Module Timing

```python
tracker = LatencyTracker()
tracker.start("retrieval")
docs = retriever.search(query)
tracker.stop("retrieval")
# ...
print(tracker.get_all())
# {"retrieval": 45.2, "rerank": 312.8, "llm": 1820.5}
```

The Streamlit Q&A page renders these as an animated bar chart under each answer.

---

## 13. Training Data Generation

**File:** `train/build_train_data.py`

### Why Generate Synthetic Training Data?

We don't have thousands of pre-labeled immunology QA pairs with ground-truth retrieval
judgments. Instead, we use the same LLM we're fine-tuning to **generate its own training
signal** from the indexed text. This is called **self-supervised** or **bootstrapped**
training data generation.

### Four-Stage Pipeline Overview

| Stage | Output file | Time | Needs vLLM? |
|-------|-------------|------|-------------|
| 1 — QA generation | `data/train/qa_pairs_cache.jsonl` | ~3.4 h | ✅ Yes |
| 2 — Reranker triplets | `data/train/reranker_train.jsonl` + `reranker_test.jsonl` | < 5 min | ❌ No |
| 3 — SFT formatting | `data/train/sft_train.jsonl` | < 1 min | ❌ No |
| 4 — Eval QA set | `data/train/eval_qa.jsonl` | < 1 min | ❌ No |

Each stage is checkpointed — if its output file exists the stage is skipped on re-run.
Use `--force` to redo all stages, or delete individual files to redo just that stage.

---

### Stage 1 — Positive Pairs (QA Generation)

**Source:** Every chunk in MongoDB with ≥ 30 words.

**Process:** For each chunk, the script calls vLLM with `CONTEXT_PROMPT_TPL` and asks it
to generate 5 QA pairs. The LLM returns a JSON array; each item becomes one training record.

```
CONTEXT_PROMPT_TPL
──────────────────
You are an expert immunology educator. Given the following passage from an immunology
textbook, generate {n_qa} distinct question-answer pairs that test understanding of
the key concepts.

Requirements:
- Questions should be specific and scientifically accurate
- Answers should be concise (1-3 sentences) and directly answerable from the passage
- Cover different aspects: mechanisms, definitions, comparisons, clinical relevance
- Use proper immunological terminology

Passage:
{passage}

Output ONLY a JSON array with this exact format (no other text):
[
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."}
]
```

**Output record schema** (`qa_pairs_cache.jsonl`):
```json
{
  "question":    "What is the significance of KIR gene polymorphism in NK cell function?",
  "answer":      "KIR polymorphism creates individual variation in inhibitory/activating receptor balance...",
  "passage":     "The overall response of NK cells to differences in MHC expression...",
  "chunk_id":    "abc123",
  "source_file": "Janeway_Immunobiology_10e.pdf",
  "chapter":     "Chapter 3",
  "page":        "112",
  "unique_id":   "d41d8cd98f00b204e9800998ecf8427e"
}
```

**Key point:** The `passage` field is the **source chunk** — the exact textbook passage
the question was generated from. This makes `passage` the ground-truth **positive** for
the reranker: it is guaranteed to answer the question because the LLM wrote the question
specifically about it.

**Parallelism:** ThreadPoolExecutor runs multiple vLLM calls concurrently (default 4
workers). Stage 1 takes ~3.4 hours for a full textbook (~4,247 chunks × 5 QA pairs).
Actual result: **16,936 QA pairs** from Janeway's Immunobiology 10e.

---

### Stage 2 — Hard Negatives (Hybrid Retrieval + v4 Parent-Chunk Resolution)

This stage builds `(query, positive, negative)` triplets for training the cross-encoder
reranker. It requires no LLM — BM25 + Chroma retrieval is fast and completes in < 10 minutes.

#### What is a triplet?

```
{
  "query":     "What is the significance of KIR gene polymorphism in NK cell function?",
  "pos":       "<parent chunk text — the full paragraph the question was generated from>",
  "neg":       "<parent chunk text — topically related but non-answering paragraph>",
  "pos_label": 2,
  "neg_label": 0
}
```

The reranker learns: given a query and a passage, predict whether the passage answers
the query. Each triplet trains it on one positive example (it does) and one negative
example (it does not).

#### v4: Why "parent chunk"? The training/inference granularity problem

This is the most important design decision in Stage 2 — and the root cause of all three
previous fine-tuning regressions (v1, v2, v3).

The inference pipeline (Steps 3–6 in `RAGPipeline.answer()`) works like this:

```
Step 1–3: BM25 + dense retrieval returns CHILD chunks (~60-80 words each)
Step 4:   rrf_fuse() merges and reranks by score
Step 5:   merge_docs() looks up each chunk's parent_id in MongoDB
          REPLACES child chunks with their PARENT chunks (~150-300 words each)
Step 6:   BGEM3ReRanker.rank() receives PARENT chunks  ← what the reranker actually sees
```

Versions v1, v2, v3 all set `"pos": qa["passage"]` — the raw child chunk text from
Stage 1. The reranker was trained on short child-chunk snippets but at inference only
ever saw longer parent-chunk paragraphs. This distribution mismatch caused all three
runs to regress below the base model.

**v4 fix:** `_resolve_parent_content()` resolves every uid to its parent before writing:

```python
def _resolve_parent_content(uid: str, uid_to_chunk: dict) -> Optional[str]:
    """
    Resolve a chunk uid to its parent chunk's page_content.

    Rules:
    - chunk has parent_id AND parent exists  → return parent.page_content
    - chunk has no parent_id (IS a parent)   → return chunk.page_content
    - uid not found in lookup                → return None (caller falls back)
    """
    chunk = uid_to_chunk.get(uid)
    if chunk is None:
        return None
    parent_id = chunk.get("metadata", {}).get("parent_id")
    if parent_id:
        parent = uid_to_chunk.get(parent_id)
        if parent:
            return parent.get("page_content")
        return chunk.get("page_content")   # parent missing — use child as fallback
    return chunk.get("page_content")       # already a parent chunk
```

This is called for EVERY positive, negative, and random fallback before writing to the
triplet file. Training and inference now operate on identical text granularity.

#### How the positive is set (v4)

```python
# build_train_data.py — build_reranker_data()
uid_to_chunk: dict = {c["unique_id"]: c for c in all_chunks if c.get("unique_id")}

pos_content = (
    _resolve_parent_content(qa.get("unique_id", ""), uid_to_chunk)
    or qa["passage"]   # fallback: original child text if parent lookup fails
)
```

The `uid_to_chunk` dict maps every `unique_id` → MongoDB document, covering both child
and parent chunks. Resolution follows the `parent_id` pointer in the child's metadata.

#### How hard negatives are found (v4)

For each (question, positive_chunk) pair, Stage 2 runs hybrid retrieval and resolves to
parent chunks:

```
Step 1: bm25.retrieve_topk(question, k=10)  +  chroma.retrieve_topk(question, k=10)
Step 2: Merge + deduplicate by unique_id; exclude positive uid
Step 3: candidates[0] is the hardest negative (highest hybrid rank)
Step 4: _resolve_parent_content(candidates[0].metadata["unique_id"], uid_to_chunk)
        → returns the PARENT chunk text for that negative
```

**Why these negatives are genuinely hard:**
- They are **real textbook passages** — same domain, same writing style, same vocabulary
- They are **topically related** — BM25 + dense ranked them highly for this question
- They are **wrong answers** — they discuss a different mechanism, cell type, or concept
- They are **parent-length** (~150-300 words) — matching the granularity seen at inference

**Fallback:** If hybrid returns nothing, a random chunk is sampled and also resolved to
its parent before writing to the triplet.

#### Why NOT LLM-generated negatives (the original approach)

The original Stage 2 prompted the LLM to *write* a fake immunology passage as the
negative. This caused a fatal failure:

```
Step  50: Train loss=0.8074  NDCG@10=0.8828   ← normal start
Step 100: Train loss=0.0015  NDCG@10=1.0000   ← collapsed
Step 150: Train loss=0.0000  NDCG@10=1.0000   ← no learning
Step ...  (stays at 0.0000 for all 5,600 remaining steps)
```

The model memorised *"LLM-generated prose = negative, textbook prose = positive"*
in ~100 steps — zero immunology knowledge required. With BM25/hybrid negatives the loss
decreases gradually across all 3 epochs because the distinction is genuinely hard.

#### Chapter-stratified train/eval split

Instead of a random shuffle, triplets are split by chapter:

```
chapters = sorted(all_chapters)          # alphabetical order
eval_set = last 10% of chapters          # e.g. ["Chapter 9"]
train = triplets whose chapter ∉ eval_set
test  = triplets whose chapter ∈ eval_set
```

This ensures no chapter appears in both sets. The eval set measures whether the
reranker generalises to chapters it has never seen — a much stricter test than
a random split (which leaks queries from the same chapter into both sets).

**Actual result:** Chapter 9 held out → **14,741 train / 2,195 eval** triplets.

---

### Stage 3 — SFT Data

Each QA pair becomes one training record for LLM supervised fine-tuning. The format is
ShareGPT, required by LLaMA-Factory:

```json
{
  "messages": [
    {
      "from":  "system",
      "value": "You are an expert immunology assistant. Answer questions accurately
                using the provided context. Cite relevant information and use proper
                immunological terminology."
    },
    {
      "from":  "human",
      "value": "Context:\n<the source passage>\n\nQuestion: <the question>"
    },
    {
      "from":  "gpt",
      "value": "<the answer>"
    }
  ]
}
```

The `human` turn always includes the full passage as context so the LLM learns to
ground its answers in retrieved evidence — matching how it will be used at inference.

> **Why ShareGPT and not OpenAI format?** LLaMA-Factory expects `from`/`value` keys
> (ShareGPT) when `formatting: sharegpt` is set. Using `role`/`content` (OpenAI format)
> causes `KeyError: 'from'` at training time.

---

### Stage 4 — Eval QA Set

A held-out evaluation set for benchmarking the full RAG system after fine-tuning.

**Sampling:** Stratified across all chapters — `n_eval ÷ n_chapters` pairs per chapter
(default 150 total). This ensures even coverage of the textbook, not just popular chapters.

**Schema** (no `passage` field — reduces file size; `evaluate.py` retrieves fresh context):
```json
{
  "question":    "...",
  "answer":      "...",
  "source_file": "Janeway_Immunobiology_10e.pdf",
  "chapter":     "Chapter 3",
  "page":        "112",
  "unique_id":   "d41d8cd98f00b204e9800998ecf8427e"
}
```

**Actual result:** 116 pairs (textbook has 12 chapters; `min(per_chapter, available)` clipped some).

---

### Output Files (Actual Sizes)

| File | Lines | Size | Format | Used by |
|------|-------|------|--------|---------|
| `qa_pairs_cache.jsonl` | 16,936 | 36 MB | One QA record per line | Checkpoint; input to stages 2–4 |
| `reranker_train.jsonl` | 14,741 | 50 MB | `{query, pos, neg, pos_label, neg_label}` | `train_reranker.py` |
| `reranker_test.jsonl` | 2,195 | 7.5 MB | Same as train | Eval metrics during training |
| `sft_train.jsonl` | 16,936 | 37 MB | ShareGPT `{messages: [...]}` | LLaMA-Factory SFT trainer |
| `eval_qa.jsonl` | 116 | < 1 MB | `{question, answer, metadata}` | `evaluate.py` end-to-end eval |

---

## 14. Fine-Tuning: Reranker

**File:** `train/train_reranker.py`

### Training Objective

The reranker is a binary classifier: given (query, passage), predict relevance (0 or 1).
Training data: positive pairs (query, relevant passage) and negative pairs
(query, hard-negative passage) from `reranker_train.jsonl`.

```python
loss_fn = BCEWithLogitsLoss()

for step, (encoded, labels) in enumerate(train_loader):
    logits = model(**encoded).logits.squeeze(-1)  # shape: [batch_size]
    loss = loss_fn(logits.float(), labels.float())  # .float() prevents NaN in bf16/fp16
    ...
```

**Critical fix — `.float()` before loss:**
The model produces bf16 logits. BCEWithLogitsLoss computes `sigmoid(logit)`, which
overflows bf16 for large logit values, producing NaN. Casting to fp32 first prevents this.

### Evaluation Metrics

- **NDCG@10** (Normalized Discounted Cumulative Gain): measures if highly relevant
  results appear at the top of the ranking. NDCG@10 = 1.0 means perfect ranking.
- **MRR@10** (Mean Reciprocal Rank): the average of 1/rank for the first relevant result.
  MRR@10 = 1.0 means the relevant passage is always ranked #1.

### Training Config (from `config.yaml`)

```yaml
training:
  reranker:
    epochs: 3
    batch_size: 16
    learning_rate: 2.0e-5
    warmup_ratio: 0.1     # first 10% of steps: linear LR warmup
    fp16: false           # use bf16 instead (A100 native, no NaN risk)
    bf16: true
    max_length: 512       # truncate (query + passage) to 512 tokens
```

### Output

```
outputs/models/reranker_finetuned/
  ├── best/              ← saved when NDCG@10 improved (best checkpoint)
  ├── final/             ← end-of-training checkpoint
  └── runs/              ← TensorBoard event files

outputs/reranker_eval/
  ├── reranker_training_curves.png   ← loss + NDCG@10 curves per epoch
  └── comparison.png                 ← before vs after (only if base model accessible)
```

---

### Experimental Results (April 2026) — All Fine-tuning Attempts

Four fine-tuning iterations were attempted. All three completed runs regressed vs the
base model; v4 (parent-chunk data fix + LoRA) is the pending fix.

#### Run v1 — BM25-only Negatives (Full Fine-tuning)

| What | Value |
|------|-------|
| Training data | 14,741 triplets, Chapter 9 held out |
| Negative mining | BM25-only |
| Training time | ~4.7 h (283 min, 3 epochs, A100-40GB) |
| Reranker NDCG@10 (own eval) | **0.9637** |
| End-to-end Recall@1 | 0.426 ⬇️ (base: 0.559, −13.3 pp) |
| End-to-end MRR@10 | 0.401 ⬇️ (base: 0.613, −21.3 pp) |

**Root cause:** BM25-only negatives do not match production distribution (hybrid retrieval).
The reranker learned to discriminate BM25 candidates but re-ranked hybrid dense candidates poorly.

#### Run v2 — Hybrid Negatives (Full Fine-tuning)

| What | Value |
|------|-------|
| Training data | 14,741 triplets, Chapter 9 held out |
| Negative mining | BM25 + Chroma dense (hybrid) |
| Training time | ~4.7 h (283 min, 3 epochs, A100-40GB) |
| Reranker NDCG@10 (own eval) | **0.9669** |
| End-to-end Recall@1 | 0.426 ⬇️ (base: 0.559, −13.3 pp) |
| End-to-end MRR@10 | 0.401 ⬇️ (base: 0.613, −21.3 pp) |

**Root cause:** Fixing the negative distribution did not fix the regression. All 560 M
parameters were updated on 14,741 triplets → **catastrophic forgetting** of broad ranking
knowledge — see analysis below.

#### Why Full Fine-tuning Fails Here: Catastrophic Forgetting

```
Base model BAAI/bge-reranker-v2-m3:
  Pretrained on ~hundreds of millions of (query, doc) pairs
  Across many domains and languages
  560 M parameters all carry rich ranking knowledge

Full fine-tuning on 14,741 immunology triplets:
  Updates all 560 M parameters
  Gradient signal: "learn immunology ranking"
  Side effect: overwrites generalisation patterns from pretraining

Result:
  Reranker NDCG@10 on own test set: 0.9669  ← looks great (same distribution)
  Recall@1 on full eval_qa.jsonl:    0.426   ← regression (different distribution)
  The model has forgotten how to rank general immunological passages
  that differ from the training chapter distribution
```

The high NDCG@10 on the training eval set is **misleading**: the eval set is drawn from
the same BM25/hybrid negative distribution as training. It measures in-distribution
reranking, not generalisation. The base model's broad knowledge generalises better.

**Contrast with the LLM SFT (§15):** The LLM was fine-tuned with **LoRA** — only
~7 M of 8 B parameters were updated (0.09%). Base model knowledge was preserved.
→ Apply the same approach to the reranker.

#### Run v3 — LoRA Fine-tuning (child-chunk data, WRONG granularity)

| What | Value |
|------|-------|
| Training data | 14,741 triplets, Chapter 9 held out |
| Negative mining | BM25 + Chroma dense (hybrid) |
| Fine-tuning method | LoRA (r=16, α=32, target: query/key/value) |
| Trainable params | ~7 M / 560 M (~1.3%) |
| Training time | ~1.95 h (117 min, 3 epochs, A100-40GB) |
| Reranker NDCG@10 (own eval) | **0.9554** |
| End-to-end Recall@1 | 0.360 ⬇️ (base: 0.559, −19.9 pp) |
| End-to-end MRR@10 | 0.332 ⬇️ (base: 0.613, −28.1 pp) |

**LoRA fixed catastrophic forgetting but produced the worst result yet — why?**

v3 revealed a deeper data bug that was hidden in v1/v2 by the catastrophic forgetting
noise: **parent/child chunk granularity mismatch**.

```
Training (v1, v2, v3):
  "pos" = qa["passage"]  ← raw child chunk (~60-80 words)
  "neg" = retrieved doc.page_content  ← also child chunk (~60-80 words)
  Reranker learns: score short 60-80 word snippets

Inference pipeline:
  retrieve_topk() → child chunks
  merge_docs() → REPLACES with parent chunks (~150-300 words)
  BGEM3ReRanker.rank() receives parent chunks

The adapter trained on short text but only ever sees long text:
  word count mismatch → sentence structure mismatch → relevance signal mismatch
  → Adapter actively hurts ranking, explaining Recall@1 drop to 0.360 (worst ever)
```

The NDCG@10 of 0.9554 — while slightly lower than full fine-tuning — still looks
"good" on the training-distribution eval set, masking the inference distribution gap.

#### Run v4 — LoRA + Parent-Chunk Data Fix (pending)

| What | Value |
|------|-------|
| Training data | Same 14,741 triplets but positives/negatives resolved to **parent chunks** |
| Negative mining | BM25 + Chroma dense (hybrid), parent-resolved |
| Fine-tuning method | LoRA (r=16, α=32, target: query/key/value) |
| Data fix | `_resolve_parent_content()` in `train/build_train_data.py` Stage 2 |
| Status | Pending retrain after Stage 2 regeneration |
| Target Recall@1 | **> 0.559** (beat base model) |
| Target MRR@10 | **> 0.613** (beat base model) |

**Full history:**

| Version | Fix attempted | Recall@1 | MRR@10 | Root cause resolved? |
|---------|--------------|----------|--------|----------------------|
| Base model | — | **0.559** | **0.613** | — |
| v1 (full FT) | BM25 negatives | 0.426 ⬇️ | 0.401 ⬇️ | Negative distribution ✅, catastrophic forgetting ❌ |
| v2 (full FT) | Hybrid negatives | 0.426 ⬇️ | 0.401 ⬇️ | Negative distribution ✅, catastrophic forgetting ❌ |
| v3 (LoRA) | LoRA fine-tuning | 0.360 ⬇️ | 0.332 ⬇️ | Catastrophic forgetting ✅, chunk mismatch ❌ |
| v4 (LoRA + parent) | Parent-chunk resolution | pending | pending | All three ✅ |

**Current status:** `config.yaml` has `reranker_use_finetuned: false`.
Base model is active. All checkpoints preserved at `outputs/models/reranker_finetuned/`.

---

### 14b. Reranker LoRA Fine-tuning — Implementation Details (v3/v4)

> **Status: IMPLEMENTED.** LoRA fine-tuning is live in `train/train_reranker.py` and
> `src/reranker/bge_m3_reranker.py`. v3 training completed (117 min). v4 adds the
> parent-chunk data fix in `train/build_train_data.py` and is pending a retrain.

#### Why LoRA for the Reranker?

LoRA (Low-Rank Adaptation) adds small trainable matrices next to each attention layer
while **freezing all base weights**. For the reranker:

- Base XLM-RoBERTa weights: ~560 M parameters — **frozen** (general ranking knowledge preserved)
- LoRA adapters (r=16): ~7 M parameters — **trainable** (immunology domain adaptation)
- Result: the adapter learns "what makes immunology passages relevant" without forgetting
  how to rank non-immunology passages generally

This is exactly analogous to the LoRA SFT applied to the LLM, which preserved Qwen3-8B's
general capabilities while teaching it the immunology Q&A format.

#### Planned Code Changes — `train/train_reranker.py`

```python
# pip install peft  (already available from LLM SFT step)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 1. Load base model as before
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.half().cuda()

# 2. Wrap with LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query", "key", "value"],  # XLM-RoBERTa attention projections
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# "trainable params: ~7M | all params: ~560M | trainable%: ~1.3%"

# 3. Training loop unchanged — only the adapter weights receive gradients

# 4. Save adapter (not full model)
model.save_pretrained(adapter_output_path)
# Produces: adapter_config.json + adapter_model.safetensors (~30–50 MB)
```

#### Planned Config Changes — `config.yaml`

```yaml
training:
  reranker:
    use_lora: true          # NEW flag
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    lora_target_modules:
      - query
      - key
      - value
    learning_rate: 1.0e-4   # higher LR is safe for LoRA (base frozen)
    epochs: 3
    warmup_ratio: 0.2       # longer warmup for adapter stability
    batch_size: 16
    bf16: true
    fp16: false
```

#### Loading LoRA Reranker in Production — `src/reranker/bge_m3_reranker.py`

```python
from peft import PeftModel

if constant.reranker_use_finetuned and (best_path / "adapter_config.json").exists():
    # LoRA adapter path
    base = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    self.model = PeftModel.from_pretrained(base, str(best_path))
    print("[Reranker] Loaded LoRA fine-tuned reranker")
else:
    # Full model path (legacy v1/v2 or base)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
self.model.eval().half().cuda()
```

#### Additional Improvements to Try Alongside LoRA

| Improvement | Rationale |
|-------------|-----------|
| Learning rate 1e-4 | LoRA adapters are tiny; base is frozen → higher LR is safe |
| Warmup ratio 0.2 (vs 0.1) | More gradual warmup stabilises early LoRA convergence |
| Eval every 200 steps | Catch overfitting earlier than every 500 steps |
| KL regularisation loss | `loss += kl_div(student_logits, teacher_logits)` penalises divergence from base predictions |

#### Expected Outcome

```
Actual v3 results:  Recall@1=0.360 ⬇️, MRR@10=0.332 ⬇️ (chunk granularity mismatch)
v4 target:          Recall@1 > 0.559, MRR@10 > 0.613  (after parent-chunk data fix)
Adapter size:       ~30–50 MB vs ~1.1 GB (full model)  → easy to version-control
Training time:      ~2 h (117 min for v3; LoRA trains faster than full FT per step)
```

---

## 15. Fine-Tuning: LLM SFT (LoRA)

**File:** `train/train_llm_sft.py`

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique.
Instead of updating all ~8 billion parameters of Qwen3-8B (which requires 60+ GB VRAM
and takes days), LoRA:

1. **Freezes** all original model weights
2. **Adds** two small trainable matrices (A and B) next to each attention layer:
   - Matrix A: `d × r` (where r is the rank, e.g., 16)
   - Matrix B: `r × d`
   - `AB` represents the weight update: `ΔW = BA`
3. During inference: `W_effective = W_original + BA`

With r=16 and d=4096 (Qwen3-8B hidden size), the LoRA matrices for one attention
projection have `16 × 4096 + 16 × 4096 = 131,072` parameters vs `4096 × 4096 = 16.7M`
for the full weight matrix. **~128x fewer parameters to train.**

### LoRA Configuration

```yaml
# config.yaml
training:
  llm_sft:
    lora_r: 16          # rank — higher = more capacity, more VRAM
    lora_alpha: 32      # scaling factor: alpha/r = 2.0 (effective LR multiplier)
    lora_dropout: 0.05
    lora_target_modules:
      - q_proj    # query projection
      - v_proj    # value projection
```

**Why q_proj and v_proj?** These are the most influential attention layers for task
adaptation. Adding k_proj and o_proj (all 4 attention heads) gives marginal improvement
at 2× the VRAM cost.

### LLaMA-Factory Integration

`train_llm_sft.py` does NOT implement training directly. Instead, it:

1. **Builds** a LLaMA-Factory YAML config with all hyperparameters
2. **Writes** a `dataset_info.json` so LLaMA-Factory can find the data
3. **Launches** LLaMA-Factory as a subprocess: `llamafactory-cli train config.yaml`
4. **Parses** the resulting TensorBoard logs to produce visualization

Why use LLaMA-Factory instead of training directly?
- LLaMA-Factory handles the Qwen3 chat template correctly (`<|im_start|>` tokens)
- It implements gradient checkpointing, DeepSpeed, and other training optimizations
- It's actively maintained and tested on many models

### Critical YAML Settings

```yaml
# auto-generated by build_llamafactory_config()
do_train: true          # MUST be true — without this, Trainer runs 0 steps
eval_strategy: "no"     # quoted string — bare "no" in YAML is parsed as bool False
finetuning_type: lora   # not "method: lora" (old invalid key)
template: qwen          # Qwen3 chat template — NOT "llama3"
```

### After Training: The Merge Requirement

```
outputs/models/llm_finetuned/checkpoint-150/
  ├── adapter_config.json     ← LoRA config (base model path, r, alpha, target modules)
  ├── adapter_model.safetensors  ← the actual LoRA weight matrices (small, ~200 MB)
  └── ...
```

This is **NOT** a complete model. vLLM needs a full model directory. The merge step
(`llamafactory-cli export`) produces a complete, standalone model by applying
`W_effective = W_original + BA` for every LoRA-modified layer.

---

## 16. Evaluation

**File:** `evaluate.py`

### Metrics Explained

**Retrieval metrics** — does the system find the right passages?

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Recall@1 | #correct in top-1 / total | Is the answer passage always the very first result? |
| Recall@5 | #correct in top-5 / total | Is the answer passage somewhere in the top 5? |
| MRR@10 | mean(1/rank) for rank ≤ 10 | Average position of the first correct passage |

**Actual results (April 2026, base reranker + fine-tuned LLM):**

| Metric | Value | Notes |
|--------|-------|-------|
| Recall@1 | **0.559** | Base reranker (fine-tuned regressed — see §14b) |
| Recall@3 | 0.693 | |
| Recall@5 | 0.751 | = Recall@10 (rerank_topk=5 caps output at 5) |
| MRR@10 | **0.613** | |
| ROUGE-L | 0.398 | |
| BERTScore F1 | 0.907 | Computed on CPU (vLLM occupies GPU) |

**Generation metrics** — is the answer text correct?

| Metric | What it measures | Actual result |
|--------|-----------------|-----------|
| ROUGE-L | Longest common subsequence between generated and reference answer | **0.398** |
| BERTScore F1 | Semantic similarity using RoBERTa embeddings (0-1) | **0.907** |

ROUGE-L of 0.40 means about 40% of the generated answer's key phrases overlap with the
reference answer. BERTScore F1 of 0.907 means the generated answers are semantically
close to the reference answers ~91% of the time. ROUGE-L is lower than it could be
because the LLM paraphrases rather than reproducing the reference verbatim; BERTScore
(which measures meaning, not exact wording) is a more reliable quality signal for this task.

**RAGAS metrics** (faithfulness, context precision, answer relevancy) require a real
OpenAI API key — RAGAS v0.2+ cannot be redirected to local vLLM. They are skipped
gracefully with a printed message; all other metrics are computed locally.

### BERTScore on CPU

```python
# evaluate.py — BERTScore must run on CPU when vLLM occupies the GPU
P, R, F1 = bscore(predictions, references, lang="en", verbose=False, device="cpu")
```

BERTScore downloads `roberta-large` (~1.3 GB) on first run. On A100-40GB with vLLM
running, there is ~3-4 GB of free VRAM — not enough for roberta-large. CPU inference
is ~5× slower but works correctly.

### LLM Comparison: Pretrained vs Fine-tuned

`evaluate.py` supports a side-by-side comparison of the base Qwen3-8B and the fine-tuned
LoRA-merged model using the `--compare-llm` flag. This isolates the LLM's contribution
to generation quality from retrieval noise.

#### Design: same context, different LLM

```python
# compute_llm_comparison() in evaluate.py
for qa in samples:
    result = pipeline.answer(qa["question"])       # Step 1: retrieval (once)
    base_answer = result["answer"]                  # pretrained LLM answer
    context = format_context(result["retrieved_docs"])

    ft_answer = _request_chat_url(                 # Step 2: same context, different LLM
        query=qa["question"],
        context=context,
        base_url=finetuned_llm_url,               # e.g. http://localhost:8001/v1
        model_name=finetuned_model,               # e.g. "finetuned"
    )
```

Retrieval runs once; the ranked context is fed identically to both models. Only the
generation step differs. This is a controlled comparison: any metric difference is
attributable solely to LLM fine-tuning, not retrieval quality.

#### `_request_chat_url()` — calling a different vLLM endpoint

```python
def _request_chat_url(query, context, base_url, model_name):
    """
    Creates a one-off OpenAI client at any base_url.
    Uses the identical SYSTEM_PROMPT + LLM_CHAT_PROMPT as the main pipeline
    so the comparison is strictly model-only (same prompt template, same context).
    """
    from openai import OpenAI
    from src.client.llm_client import SYSTEM_PROMPT, LLM_CHAT_PROMPT
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=constant.llm_max_tokens,
        temperature=constant.llm_temperature,
        top_p=constant.llm_top_p,
        extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
        stream=False,
    )
    return completion.choices[0].message.content
```

A fresh `OpenAI` client is created per call (comparison-only path, not the hot path).
The `extra_body` disables Qwen3's `<think>...</think>` chain-of-thought prefix,
matching the main pipeline's behaviour.

#### CLI usage

```bash
# Requires two vLLM servers running simultaneously:
#   Port 8000 → pretrained Qwen3-8B  (started normally)
#   Port 8001 → fine-tuned merged model (started separately)

python evaluate.py \
    --compare-llm \
    --finetuned-llm-url http://localhost:8001/v1 \
    --finetuned-llm-model finetuned
```

The flag is strictly **opt-in** — omitting `--compare-llm` leaves the evaluation
identical to the standard flow. The section is silently absent from the HTML report.

#### HTML report output

The comparison adds a new section between "Generation Quality" and "Per-Document-Type
Breakdown" containing:
- `llm_comparison.png` — grouped bar chart (ROUGE-L + BERTScore-F1 side by side for
  pretrained vs fine-tuned), with a delta annotation box showing `Δ ROUGE` and `Δ BERT`
  in green (improvement) or red (regression)
- A 3-row summary table: pretrained metrics | fine-tuned metrics | delta row

#### VRAM requirement

Two Qwen3-8B instances require ~70 GB VRAM total. On A100-40GB: use
`--gpu-memory-utilization 0.40` for each server and limit to `n_samples=10` with `--quick`,
or compare sequentially by swapping the vLLM server between two evaluation runs.

---

### Chart Generation

Six matplotlib charts are saved to `outputs/system_eval/`:

1. **Retrieval Recall** — line chart: Recall@1, @3, @5, @10 + MRR@10 dashed line
2. **Reranker Precision** — grouped bar: Precision@1, NDCG@5 before vs after fine-tuning
3. **Generation Quality** — grouped bar: ROUGE-L and BERTScore by doc_type
4. **E2E Radar** — 7-axis radar chart: all metrics normalized to 0-1
5. **Latency Breakdown** — horizontal stacked bar: per-module milliseconds
6. **LLM Comparison** — grouped bar: pretrained vs fine-tuned ROUGE-L + BERTScore-F1
   with Δ annotation box *(only generated when `--compare-llm` is used)*

All charts are embedded as base64 in `evaluation_report.html` for a self-contained report.

---

## 17. Streamlit UI

### Page Structure

| Page | File | Key Feature |
|------|------|-------------|
| Home | `app.py` | Hero banner, pipeline diagram, architecture cards |
| Q&A | `pages/01_QA.py` | Chat interface, inline citations, latency bars |
| Documents | `pages/02_Documents.py` | Index stats, PDF upload, diagnostics |
| Evaluation | `pages/03_Evaluation.py` | KPI scorecards, chart grid, quick eval runner |
| Settings | `pages/04_Settings.py` | Live config editor, system info card |

### Pipeline Lifecycle in Streamlit

```python
# pages/01_QA.py — lazy loading pattern
def _get_pipeline():
    if "pipeline" not in st.session_state:
        with st.spinner("Loading RAG pipeline…"):
            from src.pipeline import RAGPipeline
            st.session_state.pipeline = RAGPipeline()  # loads all models once
    return st.session_state.pipeline
```

`RAGPipeline.__init__()` loads:
- BM25 index from pickle (~1 GB in RAM)
- Reranker model onto GPU (~1 GB VRAM)
- MongoDB connection (lightweight)
- Chroma client (lightweight — vectors stay on disk until queried)

This takes ~15-30 seconds on first load. Subsequent queries are fast because everything
stays in `st.session_state` for the session lifetime.

### Hot-Reload Settings

```python
# pages/04_Settings.py — apply button handler
save_config(cfg)                          # write updated config.yaml
importlib.reload(const_mod)               # reload constant.py with new values
if "pipeline" in st.session_state:
    del st.session_state["pipeline"]      # force pipeline to reinitialize next query
```

This allows changing `bm25_topk`, `temperature`, etc. without restarting Streamlit.

### Design System

All pages share a consistent magenta theme injected via CSS:
```python
st.markdown("<style>...</style>", unsafe_allow_html=True)
```

Color palette:
- Primary: `#C2185B` (magenta)
- Dark: `#880E4F`
- Bright: `#E91E8C`
- Accent: `#7B1FA2` (purple)
- Background: `#FDF0F5`
- Border: `#F8BBD9`
- Sidebar: `linear-gradient(180deg, #880E4F → #C2185B → #E91E8C)`

The antibody Y-shape SVG (IgG immunoglobulin) serves as the brand icon — rendered as
inline SVG in the hero banner and sidebar.

---

## 18. End-to-End Data Flow

```
INPUT: User query "How do NK cells kill infected cells?"
          │
          ▼ [Optional] HyDE client → hypothetical passage (if enabled)
          │
          ├──► BM25Retriever.retrieve_topk(query, k=10)
          │    Tokenize → score with BM25Okapi → top-10 child chunks
          │
          ├──► ChromaRetriever.retrieve_topk(query, k=10)
          │    BGEEmbedder.encode_query() → 1024-dim vector
          │    Chroma nearest-neighbor search → top-10 child chunks
          │
          ▼ rrf_fuse(bm25_results, dense_results, k=60)
          Combined ranking by reciprocal rank scores
          │
          ▼ merge_docs()
          For each chunk: look up parent_id in MongoDB
          Return parent chunks (larger context windows)
          │
          ▼ BGEReranker.rerank(query, parents, topk=5)
          Cross-encoder scores each (query, parent) pair
          Returns top-5 most relevant passages
          │
          ▼ format_context(reranked_docs)
          "[1] NK cells express activating receptors...\n\n[2] KIR receptors..."
          │
          ▼ LLMClient.generate(query, context, history)
          vLLM API call → OpenAI-compatible → Qwen3-8B
          "NK cells kill infected cells through... [1] activating receptor-ligand
           binding... [2] perforin and granzyme B release..."
          │
          ▼ post_processing(answer, docs)
          Extract [1][2] citations → map to page numbers
          Collect related figures from MongoDB metadata
          │
OUTPUT: {
  "answer":         "NK cells kill infected cells through...",
  "cite_pages":     [87, 92],
  "cite_sources":   ["Janeway10e.pdf", "Janeway10e.pdf"],
  "cite_chapters":  ["Chapter 3", "Chapter 3"],
  "related_images": [{"title": "Figure 3.4", "path": "..."}],
  "latency_ms":     {"retrieval": 45, "rerank": 312, "llm": 1820},
  "retrieved_docs": [Document(...), ...]
}
```

---

## 19. Common Questions from Beginners

### Q: Why do we need MongoDB if we already have ChromaDB?

**A:** ChromaDB stores only **vector embeddings** (arrays of floats). It's optimized for
fast nearest-neighbor search but is terrible at storing or querying metadata.

MongoDB stores the **full text** of every chunk plus all metadata (page numbers, chapter,
figure references, parent-child relationships). When the pipeline needs to fetch a parent
chunk for the LLM context, it can't do that from ChromaDB — it asks MongoDB.

Think of ChromaDB as a "search index" and MongoDB as the "actual database".

### Q: Why split into parent and child chunks? Why not just use one size?

**A:** There's a fundamental tension in RAG:
- **Small chunks** → better retrieval precision (the relevant sentence is easier to find)
- **Large chunks** → better generation quality (the LLM has more context to write a complete answer)

Parent-child chunking solves this by using small chunks for retrieval and large chunks
for generation. The child chunk finds the right "neighborhood" in the text; the parent
chunk gives the LLM the full paragraph needed for a complete answer.

### Q: Why does vLLM need `--served-model-name Qwen/Qwen3-8B`?

**A:** When you run `--model $(pwd)/models/Qwen3-8B`, vLLM serves the model under the
key `/root/autodl-tmp/.../models/Qwen3-8B` (the full disk path). But `config.yaml`
says `llm: Qwen/Qwen3-8B` — so our code requests model `"Qwen/Qwen3-8B"` from the API.
These don't match → 404 error.

`--served-model-name Qwen/Qwen3-8B` tells vLLM "register this model under the friendly
name `Qwen/Qwen3-8B` for API requests, regardless of where it lives on disk."

### Q: Why can't vLLM run during training?

**A:** A100-40GB has 40 GB of VRAM. vLLM pre-allocates `0.85 × 40 GB ≈ 34 GB` for
KV-cache and model weights. Training the reranker or LLM SFT also needs ~8-30 GB.
The total exceeds 40 GB → CUDA OOM.

On an A100-80GB, you might be able to run both simultaneously (vLLM at 0.50 utilization
+ LoRA SFT), but it's not tested or recommended.

### Q: What is the difference between LoRA adapter and merged model?

**A:**

| | LoRA Adapter (checkpoint-N/) | Merged Model (llm_finetuned_merged/) |
|-|-----|------|
| Size | ~200 MB | ~16 GB (same as base) |
| Contents | Weight *diffs* only (A and B matrices) | Full combined weights |
| Requires base model? | Yes — must be loaded with base model | No — standalone |
| vLLM compatible? | No (standard vLLM) | Yes |
| Recreatable? | Yes, from adapter + base | Yes, re-run merge |

### Q: Why does RAGAS always use `gpt-4o-mini` no matter what I set?

**A:** RAGAS v0.2+ hardcodes the OpenAI client internally for its metric scoring loop.
Even if you pass a custom LLM to the metric objects, the inner scoring loop reverts to
its default OpenAI configuration. This is a known limitation in RAGAS 0.2.x.

It's not possible to redirect RAGAS to a local vLLM without rewriting RAGAS internals.
Since all other metrics (Recall@K, MRR@10, ROUGE-L, BERTScore) are more meaningful for
RAG evaluation anyway, we skip RAGAS gracefully and print a clear message.

### Q: How do I add a second immunology textbook to the knowledge base?

**A:**
```bash
# 1. Copy the PDF
cp /path/to/Abbas_Cellular_Molecular_Immunology.pdf data/raw/

# 2. Services must be running (MongoDB + semantic chunking; vLLM NOT needed for indexing)

# 3. Index only the new PDF
python build_index.py --pdf data/raw/Abbas_Cellular_Molecular_Immunology.pdf --test-retrieval

# 4. (Optional) Regenerate training data to include the new book
python -m train.build_train_data --workers 4
# Kill vLLM → retrain reranker and LLM SFT → merge → restart vLLM
```

### Q: Why did all three reranker fine-tuning attempts (v1, v2, v3) regress below the base model?

Each run had a different root cause:

**v1 (BM25-only negatives):** The reranker was trained on negatives from BM25 retrieval
only, but the inference pipeline uses *hybrid* retrieval (BM25 + dense Chroma). The
reranker learned to distinguish BM25 candidates and performed poorly on dense candidates
it had never seen during training.

**v2 (hybrid negatives, full fine-tuning):** Fixed the negative distribution — but then
*catastrophic forgetting* struck. Updating all 560 M parameters on 14,741 immunology
triplets overwrote the base model's broad ranking knowledge. The reranker NDCG@10 looked
great (0.9669) on the training-distribution test set, but generalisation to held-out eval
queries collapsed. Lesson: a single-textbook dataset is orders of magnitude too small to
train a 560 M parameter model from scratch.

**v3 (LoRA fine-tuning):** Fixed catastrophic forgetting by freezing base weights — but
produced the worst result yet (Recall@1 = 0.360). The LoRA adapter trained on *child chunk*
text (60-80 words) but the inference pipeline calls `merge_docs()` before reranking, which
replaces child chunks with *parent chunks* (150-300 words). The adapter learned relevance
signals for short snippets and actively hurt ranking when presented with longer paragraphs.

**v4 (LoRA + parent-chunk data fix):** `build_train_data.py` Stage 2 now calls
`_resolve_parent_content()` for every positive, negative, and fallback before writing the
triplet. Training and inference now use identical text granularity. This is the pending fix.

The progression is a textbook example of *debugging by elimination*:
- Negative distribution → fixed in v2
- Catastrophic forgetting → fixed in v3
- Training/inference granularity mismatch → fixed in v4

### Q: How does the pretrained vs fine-tuned LLM comparison work in evaluate.py?

The `--compare-llm` flag enables a controlled comparison that isolates the LLM's
contribution to generation quality. The key design principle: **same retrieved context,
different LLM**.

For each eval query:
1. The main pipeline runs retrieval + reranking once → produces `ranked_docs`
2. The pretrained LLM (port 8000) generates an answer from those docs
3. `format_context(ranked_docs)` produces the identical context string
4. The fine-tuned LLM (port 8001) generates an answer from the *same* context string

Both answers are then scored against the reference answer using ROUGE-L and BERTScore-F1.
Because the context is identical, any difference in score is attributable only to the LLM,
not to retrieval. This is a much fairer comparison than running two separate evaluations
where retrieval results might vary by random seed.

The `_request_chat_url()` helper creates a fresh `OpenAI` client pointing at any
`base_url`. It imports `SYSTEM_PROMPT` and `LLM_CHAT_PROMPT` directly from
`src/client/llm_client.py` to guarantee the same prompt template — no risk of
accidentally giving the fine-tuned model a different instruction format.

### Q: How do I verify everything is working before running a full evaluation?

**A:** Quick smoke test:
```bash
# 1. Test MongoDB
mongosh --eval "db.runCommand({ping:1})"  # → { ok: 1 }

# 2. Test semantic chunking service
curl http://localhost:6000/health          # → {"status":"ok"}

# 3. Test vLLM
curl http://localhost:8000/v1/models       # → model "Qwen/Qwen3-8B"

# 4. Test full pipeline in Python
python -c "
from src.pipeline import RAGPipeline
p = RAGPipeline()
result = p.answer('What is an antibody?')
print(result['answer'][:200])
print('Sources:', result['cite_pages'])
print('Latency:', result['latency_ms'])
"
```

### Q: What should I do if the Q&A page shows "Connection refused: 8000"?

**A:** vLLM is not running. Start it:
```bash
cd /root/autodl-tmp/Immunology_RAG
pkill -f vllm 2>/dev/null; sleep 3

# If you have a fine-tuned merged model (preferred):
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/outputs/models/llm_finetuned_merged \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --port 8000 &

# Fallback to base model if merged model doesn't exist:
python -m vllm.entrypoints.openai.api_server \
    --model $(pwd)/models/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --port 8000 &

sleep 60 && curl http://localhost:8000/v1/models
```

---

*This guide covers all modules in the ImmunoBiology RAG system. For deployment
instructions, service startup order, and error solutions, see `docs/autodl_runbook.md`.*
