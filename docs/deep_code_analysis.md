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

### 📥📤 Data Structures & I/O Examples

**`config.yaml` → `constant.py` value resolution:**

```python
# Input: config.yaml (on disk)
retrieval:
  bm25_topk: 10
  dense_topk: 10
  rrf_k: 60
training:
  reranker:
    use_lora: true
    lora_r: 16

# Output: Python module-level variables (after import)
import src.constant as C

C.bm25_topk          # → 10         (int)
C.dense_topk         # → 10         (int)
C.rrf_k              # → 60         (int)
C.reranker_use_lora  # → True        (bool)
C.reranker_lora_r    # → 16         (int)
C.bge_reranker_model_path  # → "/root/autodl-tmp/.../models/bge-reranker-v2-m3"  (str, absolute path)
```

**Hot-reload example (Settings page):**

```python
# Before: bm25_topk = 10
# User changes config.yaml: bm25_topk: 15 and clicks "Apply"

importlib.reload(const_mod)
print(const_mod.bm25_topk)   # → 15  (updated without restarting Streamlit)
```

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

### 📥📤 Data Structures & I/O Examples

**Input:** Raw PDF file
```
data/raw/JanewaysImmunobiologyBiology10thEdition.pdf   (2,157 pages, ~170 MB)
```

**Output per page:** `data/processed/JanewaysImmunobiologyBiology10thEdition/text/0087.json`
```json
{
  "text": "T cells are activated when their T cell receptor (TCR) binds to an MHC-peptide complex on the surface of an antigen-presenting cell (APC). This interaction, combined with co-stimulatory signals from CD28-B7 binding, triggers intracellular signaling cascades...",
  "page": 87,
  "chapter": "Chapter 3",
  "doc_type": "textbook",
  "source_file": "JanewaysImmunobiologyBiology10thEdition.pdf",
  "images_info": [
    {
      "title": "Figure 3.4",
      "path": "data/processed/JanewaysImmunobiologyBiology10thEdition/images/fig_3_4.png",
      "caption": "Figure 3.4 T cell activation requires two signals..."
    }
  ]
}
```

**`detect_chapter()` input → output:**
```python
# Input: list of text blocks on a page
blocks = [
  {"text": "CHAPTER 3", "size": 18.0, "bold": True},
  {"text": "The Development of Lymphocytes", "size": 16.0, "bold": True},
  {"text": "T cells are activated when...", "size": 11.0, "bold": False}
]

# Output:
detect_chapter(blocks)   # → "Chapter 3"
```

**`handle_image()` filtering decision:**
```python
# Image 1: width=42px, height=38px, size=1200 bytes → SKIPPED (below threshold)
# Image 2: width=450px, height=320px, size=28000 bytes → KEPT, saved to images/
# Image 3: no caption nearby → saved but images_info caption = ""
# Image 4: caption "Figure 3.4 T cell activation..." → images_info populated ✅
```

**OCR fallback:**
```python
# Scanned page: fitz extracts "" (empty string)
# → pytesseract.image_to_string(page_image)
# → "T cells are activated when their TCR binds..."  (OCR output, may have typos)
```

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

### 📥📤 Data Structures & I/O Examples

**HTTP request to `/v1/semantic-chunks`:**
```json
POST http://localhost:6000/v1/semantic-chunks
Content-Type: application/json

{
  "text": "T cells are activated when their T cell receptor (TCR) binds to an MHC-peptide complex on the surface of an antigen-presenting cell (APC). This interaction triggers intracellular signaling cascades.\n\nB cells, in contrast, recognize native antigen directly through the B cell receptor (BCR). They then undergo clonal selection and affinity maturation in germinal centers.\n\nNK cells kill infected cells without prior sensitization. They use a balance of activating and inhibitory receptors.",
  "group_size": 15
}
```

**HTTP response:**
```json
{
  "chunks": [
    "T cells are activated when their T cell receptor (TCR) binds to an MHC-peptide complex on the surface of an antigen-presenting cell (APC). This interaction triggers intracellular signaling cascades.",
    "B cells, in contrast, recognize native antigen directly through the B cell receptor (BCR). They then undergo clonal selection and affinity maturation in germinal centers.\n\nNK cells kill infected cells without prior sensitization. They use a balance of activating and inhibitory receptors."
  ]
}
```

Note: paragraphs 2 and 3 were merged (similar cosine similarity in MiniLM vector space), paragraph 1 stayed separate (different topic — T cells vs B/NK cells).

**MiniLM encoding intermediate step:**
```python
# Input sentences → 384-dimensional vectors
"T cells are activated..." → [0.032, -0.156, 0.078, ..., 0.041]   # shape: (384,)
"B cells recognize..."     → [0.019, -0.143, 0.092, ..., 0.038]   # shape: (384,)
"NK cells kill..."         → [0.021, -0.139, 0.087, ..., 0.042]   # shape: (384,)

# Cosine similarity matrix
sim(T_cells, B_cells) = 0.61   # moderately similar (both are lymphocytes)
sim(B_cells, NK_cells) = 0.79  # highly similar → merged into one chunk
sim(T_cells, NK_cells) = 0.58  # lower → T cells stays separate
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

### 📥📤 Data Structures & I/O Examples

**`texts_split()` input → output:**

```python
# Input: list of page records from pdf_parser
page_records = [
  {
    "text": "T cells are activated when their T cell receptor (TCR) binds...\n\n"
            "This interaction triggers intracellular signaling...\n\n"
            "The transcription factors NF-κB and NFAT drive...",
    "page": 87,
    "chapter": "Chapter 3",
    "doc_type": "textbook",
    "source_file": "JanewaysImmunobiologyBiology10thEdition.pdf",
    "images_info": []
  },
  # ... more pages
]

# Output: list of child Document objects (for Chroma + BM25 indexing)
children = texts_split(page_records)
# Returns ~8,494 child Documents
# Each has: doc.page_content (str), doc.metadata (dict)
```

**MongoDB documents — parent vs child (actual schema with values):**

```json
// PARENT document (stored but NOT in Chroma/BM25 index)
{
  "_id": "ObjectId('65a3f2b8c4d5e6f7a8b9c0d1')",
  "page_content": "T cells are activated when their T cell receptor (TCR) binds to an MHC-peptide complex on the surface of an antigen-presenting cell (APC). This interaction, combined with co-stimulatory signals from CD28-B7 binding, triggers intracellular signaling cascades that activate transcription factors such as NF-κB and NFAT. These factors drive the expression of cytokines like IL-2, which promotes clonal expansion of the activated T cell.",
  "metadata": {
    "chunk_id":    "janeway10e_ch3_p87_000",
    "unique_id":   "a1b2c3d4e5f6a7b8c9d0e1f2",
    "parent_id":   null,
    "source_file": "JanewaysImmunobiologyBiology10thEdition.pdf",
    "doc_type":    "textbook",
    "chapter":     "Chapter 3",
    "page":        87,
    "is_parent":   true,
    "images_info": []
  }
}

// CHILD document 1 (stored in MongoDB + Chroma + BM25)
{
  "_id": "ObjectId('65a3f2b8c4d5e6f7a8b9c0d2')",
  "page_content": "T cells are activated when their T cell receptor (TCR) binds to an MHC-peptide complex on the surface of an antigen-presenting cell (APC). This interaction, combined with co-stimulatory signals from CD28-B7 binding, triggers intracellular signaling cascades.",
  "metadata": {
    "chunk_id":    "janeway10e_ch3_p87_001",
    "unique_id":   "b2c3d4e5f6a7b8c9d0e1f2a3",
    "parent_id":   "a1b2c3d4e5f6a7b8c9d0e1f2",  // ← points to parent's unique_id
    "source_file": "JanewaysImmunobiologyBiology10thEdition.pdf",
    "doc_type":    "textbook",
    "chapter":     "Chapter 3",
    "page":        87,
    "is_parent":   false,
    "images_info": []
  }
}

// CHILD document 2 (with 100-token overlap with child 1)
{
  "page_content": "...co-stimulatory signals from CD28-B7 binding, triggers intracellular signaling cascades that activate transcription factors such as NF-κB and NFAT. These factors drive IL-2 expression and clonal expansion.",
  "metadata": {
    "chunk_id": "janeway10e_ch3_p87_002",
    "unique_id": "c3d4e5f6a7b8c9d0e1f2a3b4",
    "parent_id": "a1b2c3d4e5f6a7b8c9d0e1f2",  // same parent
    ...
  }
}
```

**Word counts at each stage:**
```
Page text (raw):          ~800 words (full page)
Semantic parent chunk:    ~200-300 words (2-3 paragraphs merged by topic)
Child chunk:              ~60-80 words (≤512 tokens)
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

### 📥📤 Data Structures & I/O Examples

**`encode_docs()` — batch embedding child chunks:**
```python
# Input: list of text strings (child chunk page_content)
texts = [
    "T cells are activated when their T cell receptor (TCR) binds...",
    "NK cells use activating receptors like NKG2D to detect stress ligands...",
    # ... up to 4,247 child chunks total
]

# Output: 2D numpy array
vectors = embedder.encode_docs(texts)
print(vectors.shape)   # → (4247, 1024)   — 4,247 chunks × 1,024 dimensions
print(vectors.dtype)   # → float32
print(vectors[0][:5])  # → [0.032, -0.156, 0.078, 0.041, -0.093]  (first 5 of 1024 dims)
```

**`encode_query()` — single query encoding:**
```python
query = "How do NK cells recognize infected cells?"
vector = embedder.encode_query(query)
print(vector.shape)   # → (1024,)
print(vector[:5])     # → [0.028, -0.148, 0.082, 0.039, -0.087]
# Note: slightly different values than doc encoding — asymmetric encoding adds instruction token
```

**BM25 tokenization example:**
```python
text = "T cells are activated by MHC-peptide complexes on APCs"
english_tokenize(text)
# → ["cells", "activated", "mhc-peptide", "complexes", "apcs"]
# Removed: "T" (stopword "t"), "are" (stopword), "by" (stopword), "on" (stopword)
# Lowercased: "T" → "t", "MHC" → "mhc-peptide"

# BM25 score for this doc given query "NK cell activation":
# score ≈ 0.0   (no shared tokens after stopword removal)

# BM25 score for doc "NK cells are activated via receptor-ligand interactions":
# → tokens: ["nk", "cells", "activated", "via", "receptor-ligand", "interactions"]
# → shared: ["activated"]
# → score ≈ 4.2  (non-zero — ranked higher)
```

**Chroma collection structure:**
```python
# What's stored in ChromaDB per chunk:
collection.get(include=["documents", "metadatas", "embeddings"])
# {
#   "ids":        ["janeway10e_ch3_p87_001", "janeway10e_ch3_p87_002", ...],
#   "documents":  ["T cells are activated...", "NK cells use activating...", ...],
#   "metadatas":  [{"page": 87, "chapter": "Chapter 3", ...}, ...],
#   "embeddings": [[0.032, -0.156, ...], [0.028, -0.148, ...], ...]  # shape: (4247, 1024)
# }
print(collection.count())   # → 4247
```

**Index files on disk:**
```
outputs/vectorstore/
  ├── chroma/              # ~4.4 GB (4,247 chunks × 1,024 float32 = ~17 MB raw, plus Chroma overhead)
  │   ├── chroma.sqlite3
  │   └── <uuid>/
  │       ├── data_level0.bin
  │       └── header.bin
  └── bm25_index.pkl       # ~45 MB (compressed inverted index over 4,247 docs)
```

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

### 📥📤 Data Structures & I/O Examples

**BM25 retrieval — input/output:**
```python
query = "What cytokines do Th2 cells secrete?"

# BM25 tokenizes query: ["cytokines", "th2", "cells", "secrete"]
# Scores 4,247 docs, returns top-10

bm25_results = bm25_retriever.retrieve_topk(query, topk=10)
# Returns: list of Document objects, ordered by BM25 score (highest first)
# Example:
print(bm25_results[0].page_content[:80])
# → "Th2 cells produce the cytokines IL-4, IL-5, and IL-13. IL-4 is critical for..."
print(bm25_results[0].metadata["page"])   # → 284
print(bm25_results[0].metadata["chapter"])  # → "Chapter 9"
```

**Dense (Chroma) retrieval — input/output:**
```python
# Query is encoded to 1024-dim vector, then ANN search in Chroma
dense_results = chroma_retriever.retrieve_topk(query, topk=10)
# Finds semantically similar chunks even if they don't use the word "cytokines"
print(dense_results[0].page_content[:80])
# → "The differentiation of CD4+ T cells into Th2 effectors requires IL-4 signaling..."
# Note: "secrete" not in text, but semantically relevant ✅
```

**RRF fusion — step-by-step calculation:**
```python
# BM25 results ranking:    doc_A=rank1, doc_B=rank2, doc_C=rank3, doc_D=rank4 ...
# Dense results ranking:   doc_C=rank1, doc_A=rank2, doc_E=rank3, doc_B=rank4 ...

# RRF formula: score(doc) = Σ weight × (1 / (k + rank))
# With k=60, bm25_w=0.5, dense_w=0.5:

# doc_A: 0.5 × (1/61) + 0.5 × (1/62) = 0.00820 + 0.00806 = 0.01626
# doc_C: 0.5 × (1/63) + 0.5 × (1/61) = 0.00794 + 0.00820 = 0.01614
# doc_B: 0.5 × (1/62) + 0.5 × (1/64) = 0.00806 + 0.00781 = 0.01587
# doc_E: 0.5 × 0      + 0.5 × (1/63) = 0       + 0.00794 = 0.00794
# (doc_E not in BM25 top-10, so BM25 contribution = 0)

# Final ranking: doc_A(0.01626) > doc_C(0.01614) > doc_B(0.01587) > doc_E(0.00794)

fused_results = rrf_fuse(bm25_results, dense_results, k=60)
# Returns: list of (unique_id, score) tuples, sorted descending by score
# → [("uid_A", 0.01626), ("uid_C", 0.01614), ("uid_B", 0.01587), ...]
```

**HyDE expansion example:**
```python
# Input query (short, conversational):
query = "how do NK cells kill infected cells?"

# HyDE generates a hypothetical textbook passage:
hypothetical = "NK cells eliminate virus-infected and tumor cells through a mechanism..."
# + ~100 more words of academic immunology prose

# Dense search is then run on the hypothetical passage instead of the raw query:
dense_results = chroma.retrieve_topk(hypothetical, topk=10)
# The expanded passage matches textbook vocabulary more closely → better recall
```

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

### 📥📤 Data Structures & I/O Examples

**`rank()` — full input/output walkthrough:**

```python
query = "What cytokines do Th2 cells secrete?"

# Input: query (str) + list of parent Documents after merge_docs()
candidate_docs = [
    Document(page_content="Th2 cells produce IL-4, IL-5, and IL-13. IL-4 is critical...", metadata={...}),
    Document(page_content="T helper cells differentiate into subsets depending on cytokine...", metadata={...}),
    Document(page_content="NK cells use activating receptors to detect stress ligands...", metadata={...}),
    Document(page_content="B cell class switching to IgE is driven by IL-4 and IL-13...", metadata={...}),
    Document(page_content="Th1 cells secrete IFN-γ and TNF-α, driving macrophage activation...", metadata={...}),
    # ... up to 20 candidates
]

# Internally: builds (query, doc) pairs for cross-encoder
pairs = [
    ("What cytokines do Th2 cells secrete?", "Th2 cells produce IL-4, IL-5, and IL-13..."),
    ("What cytokines do Th2 cells secrete?", "T helper cells differentiate into subsets..."),
    ("What cytokines do Th2 cells secrete?", "NK cells use activating receptors..."),
    # ...
]
# Tokenized to: [CLS] query [SEP] document [SEP]  (max_length=4096)

# Model outputs logit scores (raw, before sigmoid):
scores = [ 8.34, 4.12, -2.87, 6.91, 1.23, ... ]
#          ↑ high relevance  ↑ moderate  ↑ irrelevant

# After sorting + top_k=5:
reranked = [
    Document("Th2 cells produce IL-4, IL-5, and IL-13...", score=8.34),   # rank 1 ✅
    Document("B cell class switching... IL-4 and IL-13...", score=6.91),  # rank 2 ✅
    Document("T helper cells differentiate...", score=4.12),              # rank 3
    Document("Th1 cells secrete IFN-γ...", score=1.23),                   # rank 4
    Document("CD4+ effector T cells...", score=0.88),                     # rank 5
]
# NK cells doc (score=-2.87) is dropped — irrelevant ✅
```

**GPU memory management:**
```python
# Before del inputs:
# CUDA memory: ~34.6 GB (vLLM) + ~0.4 GB (reranker batch) = ~35 GB used
# Free VRAM: ~5 GB

# After del inputs; torch.cuda.empty_cache():
# CUDA memory: ~34.6 GB (vLLM) + ~0.1 GB (reranker model weights only)
# Free VRAM: ~5.3 GB  ← safe for next query
```

**LoRA vs base model loading detection:**
```python
model_path = Path("outputs/models/reranker_finetuned/best/")

# Case 1: LoRA adapter (v3/v4)
(model_path / "adapter_config.json").exists()  # → True
# → loads base BAAI/bge-reranker-v2-m3, then wraps with PeftModel

# Case 2: Full fine-tuned model (v1/v2)
(model_path / "adapter_config.json").exists()  # → False
# → loads directly with AutoModelForSequenceClassification.from_pretrained()

# Case 3: reranker_use_finetuned=False in config.yaml
# → loads base model from constant.bge_reranker_model_path regardless
```

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

### 📥📤 Data Structures & I/O Examples

**Full `generate()` call — input messages list:**
```python
# After format_context(reranked_docs) produces:
context = """[1] Th2 cells produce IL-4, IL-5, and IL-13. IL-4 is critical for B cell class switching to IgE and for driving further Th2 differentiation. IL-5 promotes eosinophil development and activation. IL-13 is important in mucosal immunity and contributes to airway hyperreactivity. (p. 284, Chapter 9)

[2] B cell class switching to IgE is driven by IL-4 and IL-13 produced by Th2 cells in the presence of CD40L-CD40 interactions. This underlies the IgE-mediated allergic response. (p. 291, Chapter 9)

[3] T helper cell differentiation is driven by the cytokine environment during antigen presentation. Th2 differentiation requires IL-4 signaling through STAT6 and the transcription factor GATA3. (p. 280, Chapter 9)"""

# messages list sent to vLLM:
messages = [
    {
        "role": "system",
        "content": "You are an expert immunology assistant with deep knowledge of immunological principles..."
    },
    {   # optional: previous turn (if multi-turn conversation)
        "role": "user",
        "content": "What are T helper cell subsets?"
    },
    {
        "role": "assistant",
        "content": "T helper cells differentiate into distinct subsets including Th1, Th2, Th17..."
    },
    {   # current query
        "role": "user",
        "content": "Context:\n[1] Th2 cells produce IL-4, IL-5...\n\nQuestion: What cytokines do Th2 cells secrete?"
    }
]
```

**vLLM API response:**
```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=messages,
    max_tokens=1024,
    temperature=0.001,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)

# response.choices[0].message.content:
answer = "Th2 cells secrete three primary cytokines: IL-4, IL-5, and IL-13 [1]. IL-4 plays a dual role in promoting B cell class switching to IgE and driving further Th2 differentiation [1][2]. IL-5 is responsible for eosinophil development and activation [1]. IL-13 contributes to mucosal immunity and airway hyperreactivity [1]. The differentiation of CD4+ T cells into Th2 effectors is initiated by IL-4 signaling through STAT6 and the transcription factor GATA3 [3]."

# Note: citations [1][2][3] map to the numbered passages in context
# Note: no <think>...</think> prefix because enable_thinking=False ✅
```

**Token counts (typical query):**
```
System prompt:     ~120 tokens
Chat history:      0-200 tokens (2 previous turns × ~100 tokens each)
Context passages:  ~600-900 tokens (5 parent chunks × ~120-180 words each)
Question:          ~15-25 tokens
Total input:       ~750-1250 tokens   (well under 8,192 context limit)
Generated answer:  ~150-300 tokens
```

**HyDE client difference:**
```python
# HyDE prompt (temperature=0.3 for vocabulary variety):
hyde_prompt = "Generate a detailed immunology textbook passage that would directly answer:\n'What cytokines do Th2 cells secrete?'\n\nWrite in the style of Janeway's Immunobiology. ~100 words."

# HyDE output (used as search query, NOT shown to user):
hypothetical = "Th2 cells, also known as type 2 helper T cells, are characterized by their production of IL-4, IL-5, IL-10, and IL-13. These cytokines collectively promote humoral immunity, eosinophil activation, and IgE class switching. IL-4 is the signature cytokine that initiates Th2 differentiation through STAT6..."
```

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

### 📥📤 Data Structures & I/O Examples

**`RAGPipeline.answer()` — full input/output:**

```python
# Input
pipeline = RAGPipeline()
result = pipeline.answer(
    query="What cytokines do Th2 cells secrete?",
    doc_type=None,        # no filter: search all doc types
    source_file=None      # no filter: search all PDFs
)

# Output dict (full structure):
{
  "answer": "Th2 cells secrete three primary cytokines: IL-4, IL-5, and IL-13 [1]. IL-4 plays a dual role in promoting B cell class switching to IgE and driving further Th2 differentiation [1][2]...",

  "cite_pages": [284, 291, 280],          # page numbers for [1], [2], [3]
  "cite_sources": [
    "JanewaysImmunobiologyBiology10thEdition.pdf",
    "JanewaysImmunobiologyBiology10thEdition.pdf",
    "JanewaysImmunobiologyBiology10thEdition.pdf"
  ],
  "cite_chapters": ["Chapter 9", "Chapter 9", "Chapter 9"],

  "related_images": [
    {
      "title": "Figure 9.3",
      "path": "data/processed/JanewaysImmunobiologyBiology10thEdition/images/fig_9_3.png",
      "caption": "Figure 9.3 The differentiation of CD4 T cells into Th1 and Th2 effector cells..."
    }
  ],

  "latency_ms": {
    "hyde":      0,       # HyDE disabled
    "retrieval": 48,      # BM25 + Chroma search combined
    "merge":     12,      # MongoDB parent lookup
    "rerank":    318,     # cross-encoder scoring
    "llm":       1854     # vLLM generation time
  },

  "retrieved_docs": [
    Document(
      page_content="Th2 cells produce IL-4, IL-5, and IL-13...",
      metadata={
        "chunk_id": "janeway10e_ch9_p284_003",
        "unique_id": "a1b2c3d4...",
        "parent_id": null,
        "page": 284, "chapter": "Chapter 9",
        "source_file": "JanewaysImmunobiologyBiology10thEdition.pdf",
        "is_parent": true
      }
    ),
    # ... 4 more Documents (top-5 after reranking)
  ]
}
```

**Intermediate data at each pipeline step:**
```python
# After BM25: 10 child chunks, scored
# After dense: 10 child chunks, scored (some overlap with BM25)
# After RRF: up to 20 unique child chunks with combined scores

# After merge_docs(): all child chunks → replaced with their parent chunks
# child (60 words) → parent (200 words) via MongoDB lookup
# Example: "...TCR binds to MHC..." (child) → "T cells are activated when TCR binds...
#           triggers NF-κB and NFAT... drives IL-2 expression..." (parent)

# After reranker: top 5 parent chunks (from 20 candidates)
# After format_context(): numbered string passed to LLM
```

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

### 📥📤 Data Structures & I/O Examples

**`merge_docs()` — before vs after parent resolution:**
```python
# Input: fused child chunks from RRF
input_docs = [
    Document(
        page_content="TCR binds MHC-peptide complex on APC surface.",  # 60 words
        metadata={"unique_id": "b2c3d4", "parent_id": "a1b2c3", "page": 87}
    ),
    Document(
        page_content="IL-4 promotes class switching to IgE.",           # 55 words
        metadata={"unique_id": "c3d4e5", "parent_id": "d4e5f6", "page": 291}
    ),
    # 18 more child chunks...
]

# MongoDB lookup for parent_id "a1b2c3":
parent_doc = Document(
    page_content="T cells are activated when their TCR binds to an MHC-peptide complex on the surface of an APC. This interaction triggers intracellular signaling cascades that activate NF-κB and NFAT. These factors drive IL-2 expression and clonal expansion.",  # 200 words
    metadata={"unique_id": "a1b2c3", "parent_id": None, "is_parent": True, "page": 87}
)

# Output: list of parent Documents (deduplicated by unique_id)
merged = merge_docs(input_docs, [])
# len(merged) <= len(input_docs)  (dedup removes siblings sharing same parent)
# merged[0].page_content → the 200-word parent text (not the 60-word child)
```

**`post_processing()` — citation extraction:**
```python
# Input:
answer = "Th2 cells secrete IL-4 and IL-5 [1]. IL-4 drives IgE switching [1][2]. Multiple mechanisms are involved [2,3]."
docs = [parent_doc_1, parent_doc_2, parent_doc_3, parent_doc_4, parent_doc_5]

# Regex matches: ["1", "1", "2", "2,3"] → unique indices: {1, 2, 3}
# Output:
post_processing(answer, docs)
# {
#   "answer": "Th2 cells secrete IL-4 and IL-5 [1]...",
#   "cite_pages":   [284, 291, 280],     # docs[0].metadata["page"], docs[1], docs[2]
#   "cite_sources": ["Janeway10e.pdf", "Janeway10e.pdf", "Janeway10e.pdf"],
#   "related_images": [{"title": "Figure 9.3", "path": "..."}]  # from docs[0] images_info
# }
```

**`LatencyTracker` — timing example:**
```python
tracker = LatencyTracker()

tracker.start("retrieval")
# ... BM25 + Chroma search (~48 ms)
tracker.stop("retrieval")

tracker.start("rerank")
# ... cross-encoder on 20 docs (~318 ms)
tracker.stop("rerank")

tracker.start("llm")
# ... vLLM generation (~1854 ms)
tracker.stop("llm")

tracker.get_all()
# → {"retrieval": 48.3, "rerank": 318.7, "llm": 1854.2}  # all in milliseconds
# Streamlit renders this as a horizontal stacked bar chart
```

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

### 📥📤 Data Structures & I/O Examples

**Stage 1 — QA generation, actual vLLM request/response:**

```python
# Prompt sent to vLLM (CONTEXT_PROMPT_TPL filled in):
prompt = """You are an expert immunology educator. Given the following passage from an immunology textbook, generate 5 distinct question-answer pairs...

Passage:
NK cells use a balance of activating and inhibitory receptors to distinguish infected or tumor cells from healthy cells. Activating receptors such as NKG2D bind to stress ligands (MICA, MICB, ULBP) that are upregulated on infected cells but absent on healthy cells. Inhibitory receptors, particularly KIR receptors, recognize MHC class I molecules. Healthy cells express normal levels of MHC class I and thus inhibit NK cell killing.

Output ONLY a JSON array..."""

# vLLM raw response:
raw = '[{"question": "What is the role of NKG2D in NK cell activation?", "answer": "NKG2D is an activating receptor on NK cells that binds stress ligands such as MICA and MICB, which are upregulated on infected or tumor cells, triggering NK cell cytotoxicity."}, ...]'

# After JSON parse + one record saved to qa_pairs_cache.jsonl:
{
  "question": "What is the role of NKG2D in NK cell activation?",
  "answer": "NKG2D is an activating receptor on NK cells that binds stress ligands such as MICA and MICB, which are upregulated on infected or tumor cells, triggering NK cell cytotoxicity.",
  "passage": "NK cells use a balance of activating and inhibitory receptors...",
  "chunk_id": "janeway10e_ch3_p112_004",
  "source_file": "JanewaysImmunobiologyBiology10thEdition.pdf",
  "chapter": "Chapter 3",
  "page": "112",
  "unique_id": "f7a8b9c0d1e2f3a4b5c6d7e8"
}
```

**Stage 2 — triplet with parent resolution (v4):**
```python
# qa["unique_id"] = "f7a8b9c0d1e2f3a4b5c6d7e8"  (child chunk)
# Child chunk text: "NK cells use a balance..." (~65 words)
# Child's parent_id: "e6f7a8b9c0d1e2f3a4b5c6d7"

# _resolve_parent_content("f7a8b9c0d1e2f3a4b5c6d7e8", uid_to_chunk):
# → finds child → finds parent via parent_id
# → returns parent page_content: "NK cells are innate immune lymphocytes that can kill..." (~230 words)

# Hard negative: hybrid retrieval returns "KIR gene polymorphism creates individual variation..."
# Negative uid: "a9b0c1d2e3f4a5b6c7d8e9f0"  (also a child, resolved to its parent)
# → _resolve_parent_content("a9b0c1d2e3f4a5b6c7d8e9f0", uid_to_chunk) → parent (~220 words)

# Written to reranker_train.jsonl:
{
  "query":     "What is the role of NKG2D in NK cell activation?",
  "pos":       "NK cells are innate immune lymphocytes that can kill virus-infected and tumor cells without prior sensitization. They express activating receptors such as NKG2D that bind to stress ligands MICA, MICB, and ULBP proteins upregulated on infected cells...",
  "neg":       "The KIR gene locus shows remarkable polymorphism between individuals, creating variation in the inhibitory and activating receptor repertoire. Some individuals have predominantly inhibitory KIR haplotypes while others have activating haplotypes...",
  "pos_label": 2,
  "neg_label": 0
}
```

**Stage 3 — SFT record (ShareGPT format for LLaMA-Factory):**
```json
{
  "messages": [
    {
      "from": "system",
      "value": "You are an expert immunology assistant. Answer questions accurately using the provided context. Cite relevant information and use proper immunological terminology."
    },
    {
      "from": "human",
      "value": "Context:\nNK cells use a balance of activating and inhibitory receptors to distinguish infected or tumor cells from healthy cells. Activating receptors such as NKG2D bind to stress ligands (MICA, MICB, ULBP) that are upregulated on infected cells but absent on healthy cells...\n\nQuestion: What is the role of NKG2D in NK cell activation?"
    },
    {
      "from": "gpt",
      "value": "NKG2D is an activating receptor on NK cells that binds stress ligands such as MICA and MICB, which are upregulated on infected or tumor cells, triggering NK cell cytotoxicity."
    }
  ]
}
```

**Stage 4 — eval_qa record (no passage field):**
```json
{
  "question": "What is the role of NKG2D in NK cell activation?",
  "answer":   "NKG2D is an activating receptor on NK cells that binds stress ligands such as MICA and MICB, which are upregulated on infected or tumor cells, triggering NK cell cytotoxicity.",
  "source_file": "JanewaysImmunobiologyBiology10thEdition.pdf",
  "chapter":     "Chapter 3",
  "page":        "112",
  "unique_id":   "f7a8b9c0d1e2f3a4b5c6d7e8"
}
```

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

### 📥📤 Data Structures & I/O Examples

**Training DataLoader — one batch:**
```python
# DataCollator processes one batch of 16 triplets:
# For each triplet: 2 pairs → (query, pos) and (query, neg)
# Batch of 16 triplets → 32 encoded pairs

# Tokenized batch:
encoded = {
  "input_ids":      tensor of shape (32, 512),   # dtype=torch.long
  "attention_mask": tensor of shape (32, 512),   # dtype=torch.long
  "token_type_ids": tensor of shape (32, 512)    # if model uses them
}
labels = tensor([1, 0, 1, 0, 1, 0, ...])   # shape: (32,)  alternating pos/neg

# Forward pass:
logits = model(**encoded).logits.squeeze(-1)   # shape: (32,)
# Example values:
logits = tensor([ 8.34, -3.12,  6.91, -1.87,  7.23, -2.45, ...])
#                  pos    neg    pos    neg    pos    neg

# Loss (BCEWithLogitsLoss, cast to float32):
loss = BCEWithLogitsLoss()(logits.float(), labels.float())
# Step 1 loss: ~0.8074
# Step 100 (with real negatives): ~0.3200  (still decreasing ✅)
# Step 100 (with LLM-generated negatives): ~0.0015  (collapsed ❌)
```

**Training progress output (actual from v3 run):**
```
Epoch 1/3 [Step 50/1640]:   loss=0.8074  NDCG@10=0.8828
Epoch 1/3 [Step 100/1640]:  loss=0.6312  NDCG@10=0.9103
Epoch 1/3 [Step 200/1640]:  loss=0.4891  NDCG@10=0.9341
Epoch 2/3 [Step 500/1640]:  loss=0.3104  NDCG@10=0.9487
Epoch 3/3 [Step 1640/1640]: loss=0.2201  NDCG@10=0.9554
Training complete. Best checkpoint at step 1500 (NDCG@10=0.9554)
```

**Saved adapter files (LoRA checkpoint):**
```
outputs/models/reranker_finetuned/best/
  ├── adapter_config.json          # ~1 KB
  │   {
  │     "peft_type": "LORA",
  │     "task_type": "SEQ_CLS",
  │     "base_model_name_or_path": "/root/.../models/bge-reranker-v2-m3",
  │     "r": 16,
  │     "lora_alpha": 32,
  │     "lora_dropout": 0.05,
  │     "target_modules": ["query", "key", "value"],
  │     "bias": "none"
  │   }
  ├── adapter_model.safetensors    # ~40 MB (only LoRA A/B matrices)
  └── training_complete.json       # sentinel: {"completed": true, "step": 1500, ...}
```

**compare_pre_post() output:**
```
=== Reranker Comparison ===
Base model:         Recall@1=0.559  MRR@10=0.613
Fine-tuned (v3):    Recall@1=0.360  MRR@10=0.332  ← REGRESSED
Delta:              Recall@1=-0.199  MRR@10=-0.281
→ Base model still active (reranker_use_finetuned: false in config.yaml)
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

### 📥📤 Data Structures & I/O Examples

**Auto-generated LLaMA-Factory YAML config:**
```yaml
# Written to LLaMA-Factory/immunology_sft_config.yaml by build_llamafactory_config()
model_name_or_path: /root/autodl-tmp/Immunology_RAG/models/Qwen3-8B
do_train: true
finetuning_type: lora
template: qwen
dataset: immunology_sft
dataset_dir: /root/autodl-tmp/Immunology_RAG/data/train
output_dir: /root/autodl-tmp/Immunology_RAG/outputs/models/llm_finetuned
logging_steps: 10
save_steps: 500
eval_strategy: "no"
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
```

**LLaMA-Factory training progress (6,351 steps over ~4.6 h):**
```
[2026-04-01 08:15] Step  100/6351: loss=2.3142  lr=1.8e-05
[2026-04-01 09:22] Step  500/6351: loss=1.4871  lr=1.6e-05
[2026-04-01 10:45] Step 1000/6351: loss=1.1203  lr=1.3e-05
[2026-04-01 11:58] Step 2000/6351: loss=0.9341  lr=8.5e-06
[2026-04-01 12:47] Step 3000/6351: loss=0.8102  lr=4.8e-06
[2026-04-01 13:22] Step 4000/6351: loss=0.7654  lr=2.1e-06
[2026-04-01 13:44] Step 5000/6351: loss=0.7234  lr=5.4e-07
[2026-04-01 13:56] Step 6351/6351: loss=0.7089
Training complete. Total: 274 min (4.57 h)
```

**LoRA adapter checkpoint structure:**
```
outputs/models/llm_finetuned/checkpoint-6351/
  ├── adapter_config.json          # LoRA config (base model path, r=16, α=32)
  ├── adapter_model.safetensors    # ~200 MB (q_proj + v_proj for all 32 layers)
  ├── tokenizer_config.json
  └── special_tokens_map.json
```

**LoRA math — parameter count:**
```python
# Qwen3-8B architecture:
# - 32 transformer layers
# - Hidden dimension d = 4096
# - q_proj: (4096, 4096)
# - v_proj: (4096, 4096)

# LoRA matrices per projection per layer:
# A: (4096, 16)  = 65,536 parameters
# B: (16, 4096)  = 65,536 parameters
# Per projection: 131,072 parameters

# Total LoRA parameters:
32 layers × 2 projections × 131,072 = 8,388,608 ≈ 8M trainable params
# vs 8,000,000,000 total params in Qwen3-8B
# → 0.10% of parameters trained
```

**Post-merge model structure (vLLM-compatible):**
```
outputs/models/llm_finetuned_merged/
  ├── config.json                  # standard Qwen3 config
  ├── model-00001-of-00004.safetensors   # ~4 GB each
  ├── model-00002-of-00004.safetensors
  ├── model-00003-of-00004.safetensors
  ├── model-00004-of-00004.safetensors
  ├── tokenizer.json
  └── tokenizer_config.json
# Total: ~16 GB (same size as base — LoRA weights merged into W_effective)
```

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

### 📥📤 Data Structures & I/O Examples

**eval_qa.jsonl — one input record:**
```json
{
  "question": "What is the role of NKG2D in NK cell activation?",
  "answer": "NKG2D is an activating receptor on NK cells that binds stress ligands such as MICA and MICB, which are upregulated on infected or tumor cells, triggering NK cell cytotoxicity.",
  "source_file": "JanewaysImmunobiologyBiology10thEdition.pdf",
  "chapter": "Chapter 3",
  "page": "112",
  "unique_id": "f7a8b9c0d1e2f3a4b5c6d7e8"
}
```

**Per-sample evaluation output:**
```python
# For the query above:
result = pipeline.answer("What is the role of NKG2D in NK cell activation?")

# Ground truth unique_id: "f7a8b9c0d1e2f3a4b5c6d7e8"
# Retrieved doc unique_ids (after reranking, top-5):
retrieved_ids = [
    "e6f7a8b9...",   # parent of the gold chunk ← this one matches!
    "d5e6f7a8...",
    "c4d5e6f7...",
    "b3c4d5e6...",
    "a2b3c4d5..."
]

# Recall@1: gold found at rank 1? → parent_id of gold == retrieved_ids[0]? → YES
# Recall@1 for this sample: 1.0
# MRR@10 for this sample: 1/1 = 1.0

# Generated answer:
gen_answer = "NKG2D is an activating receptor expressed on NK cells that recognizes stress-induced ligands MICA and MICB on infected or tumor cells [1], triggering cytotoxic responses including perforin-mediated killing [2]."

# Reference answer:
ref_answer = "NKG2D is an activating receptor on NK cells that binds stress ligands such as MICA and MICB..."

# ROUGE-L for this sample: 0.52  (good phrase overlap)
# BERTScore F1 for this sample: 0.93  (high semantic similarity)
```

**Aggregate metrics dict (returned by evaluate.py):**
```python
metrics = {
    # Retrieval
    "recall_at_1": 0.559,
    "recall_at_3": 0.693,
    "recall_at_5": 0.751,
    "recall_at_10": 0.751,    # = recall@5 since rerank_topk=5
    "mrr_at_10": 0.613,

    # Generation
    "rouge_l": 0.398,
    "bertscore_f1": 0.907,
    "bertscore_precision": 0.901,
    "bertscore_recall": 0.914,

    # Per doc_type breakdown
    "by_doc_type": {
        "textbook": {"rouge_l": 0.398, "bertscore_f1": 0.907, "count": 116}
    },

    # Latency (average over 116 queries)
    "avg_latency_ms": {
        "retrieval": 45,
        "merge": 11,
        "rerank": 308,
        "llm": 1872
    },

    # Sample count
    "n_samples": 116
}
```

**HTML report structure:**
```
outputs/system_eval/
  ├── evaluation_report.html     # self-contained report (all charts embedded as base64)
  ├── retrieval_recall.png       # chart 1: Recall@K line + MRR dashed
  ├── reranker_precision.png     # chart 2: grouped bar before/after
  ├── generation_quality.png     # chart 3: ROUGE-L + BERTScore by doc_type
  ├── e2e_radar.png              # chart 4: radar across 7 normalized metrics
  ├── latency_breakdown.png      # chart 5: stacked horizontal bar
  └── llm_comparison.png         # chart 6: only if --compare-llm used
```

**LLM comparison output sample:**
```python
# _request_chat_url() called for fine-tuned model (port 8001):
ft_answer = "NKG2D is an activating receptor on NK cells that recognizes MICA and MICB stress ligands upregulated on infected cells [1]. This recognition triggers perforin and granzyme B release, directly killing the target cell [2]."

# Scores:
base_rouge    = 0.38   # pretrained model
ft_rouge      = 0.47   # fine-tuned model  (+0.09 ✅)
base_bert_f1  = 0.899
ft_bert_f1    = 0.921  # (+0.022 ✅)

# Aggregate across 116 samples:
{
  "pretrained":  {"rouge_l": 0.381, "bertscore_f1": 0.899},
  "finetuned":   {"rouge_l": 0.398, "bertscore_f1": 0.907},
  "delta":       {"rouge_l": +0.017, "bertscore_f1": +0.008}
}
```

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

### 📥📤 Data Structures & I/O Examples

**`st.session_state` — what's stored across requests:**
```python
# After first query in the Q&A page:
st.session_state = {
    "pipeline": <RAGPipeline object>,          # loaded once, reused
    "messages": [                               # chat history for display
        {
            "role": "user",
            "content": "What cytokines do Th2 cells secrete?"
        },
        {
            "role": "assistant",
            "content": "Th2 cells secrete IL-4, IL-5, and IL-13 [1]...",
            "cite_pages": [284, 291, 280],
            "cite_chapters": ["Chapter 9", ...],
            "related_images": [{"title": "Figure 9.3", "path": "..."}],
            "latency_ms": {"retrieval": 48, "rerank": 318, "llm": 1854}
        }
    ]
}

# After Settings page "Apply":
del st.session_state["pipeline"]   # force reinit with new config
# Next query call: pipeline is recreated with new bm25_topk, temperature, etc.
```

**Q&A page rendering — what each result field maps to in the UI:**
```python
result = pipeline.answer(query)

# result["answer"]         → displayed in chat bubble with markdown rendering
# result["cite_pages"]     → "Sources: p.284, p.291, p.280" footer under answer
# result["cite_chapters"]  → "Chapter 9" labels on source chips
# result["related_images"] → expandable "Related Figures" section with image previews
# result["latency_ms"]     → animated bar chart: retrieval=48ms, rerank=318ms, llm=1854ms

# Example rendering:
st.markdown(result["answer"])
# → "Th2 cells secrete three primary cytokines: **IL-4**, **IL-5**, and **IL-13** [1]..."

for img in result["related_images"]:
    st.image(img["path"], caption=img["title"])
# → displays Figure 9.3 thumbnail with caption
```

**Documents page — index stats display:**
```python
# MongoDB query for stats:
stats = {
    "total_chunks": collection.count_documents({}),   # → 8,494 (parents + children)
    "child_chunks": collection.count_documents({"metadata.is_parent": False}),  # → 4,247
    "parent_chunks": collection.count_documents({"metadata.is_parent": True}),  # → 4,247
    "sources": collection.distinct("metadata.source_file"),  # → ["JanewaysImmunobiologyBiology10thEdition.pdf"]
    "chroma_vectors": chroma_collection.count(),      # → 4,247
}

# Displayed as metric cards in the UI:
# 📚 Total Chunks: 8,494
# 🔍 Vector Index: 4,247 embeddings
# 📄 Sources: 1 document
```

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

### 📥📤 Data Structures & I/O Examples

**Complete type trace for `"What cytokines do Th2 cells secrete?"`:**

```python
# ─── INPUT ───────────────────────────────────────────────────────
query: str = "What cytokines do Th2 cells secrete?"

# ─── STEP 1: BM25 ─────────────────────────────────────────────────
query_tokens: list[str] = ["cytokines", "th2", "cells", "secrete"]
bm25_scores: np.ndarray shape=(4247,)  dtype=float64
# top-10 indices selected, returned as:
bm25_results: list[Document]  len=10

# ─── STEP 2: DENSE ────────────────────────────────────────────────
query_vector: np.ndarray shape=(1024,)  dtype=float32
# Chroma cosine ANN search, returned as:
dense_results: list[Document]  len=10

# ─── STEP 3: RRF ──────────────────────────────────────────────────
fused: list[tuple[str, float]]  len≤20  # (unique_id, rrf_score)
# e.g. [("uid_A", 0.01626), ("uid_C", 0.01614), ...]

# ─── STEP 4: merge_docs ───────────────────────────────────────────
# Input:  20 child Documents  (avg 65 words each)
# MongoDB lookups: 20 parent_id queries
# Output: ≤20 parent Documents  (avg 220 words each, deduplicated)
merged: list[Document]  len≤20

# ─── STEP 5: RERANKER ─────────────────────────────────────────────
# Input: query (str) + merged (list[Document] len≤20)
# Tokenize 20 pairs → tensor shape (20, 512)
# Forward pass → logits tensor shape (20,)
# Sort, keep top 5
reranked: list[Document]  len=5

# ─── STEP 6: FORMAT CONTEXT ───────────────────────────────────────
context: str  # numbered passages
# "[1] Th2 cells produce IL-4, IL-5...\n\n[2] B cell class switching...\n\n..."
# ~700 tokens total

# ─── STEP 7: LLM ──────────────────────────────────────────────────
messages: list[dict]  len=2  # system + user (no history on first turn)
# vLLM API call, ~1.8 seconds
answer_text: str  # ~200 tokens

# ─── STEP 8: POST-PROCESSING ──────────────────────────────────────
result: dict = {
    "answer":         str,           # ~200 words
    "cite_pages":     list[int],     # len=2-3
    "cite_sources":   list[str],     # len=2-3
    "cite_chapters":  list[str],     # len=2-3
    "related_images": list[dict],    # len=0-2 depending on nearby figures
    "latency_ms":     dict,          # 4 keys: retrieval, merge, rerank, llm
    "retrieved_docs": list[Document] # len=5
}

# ─── TOTAL LATENCY ────────────────────────────────────────────────
# retrieval: ~48 ms
# merge:     ~12 ms
# rerank:    ~318 ms
# llm:       ~1854 ms
# TOTAL:     ~2.2 seconds end-to-end (GPU-accelerated reranker + vLLM)
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
