# Contributing to ImmunoBiology RAG

Thank you for your interest in contributing! This document explains how to get
started, what kinds of contributions are welcome, and how to submit them.

---

## Table of Contents

1. [What Can I Contribute?](#what-can-i-contribute)
2. [Getting Started (Local Dev Setup)](#getting-started-local-dev-setup)
3. [Project Structure at a Glance](#project-structure-at-a-glance)
4. [Contribution Guidelines](#contribution-guidelines)
5. [Submitting a Pull Request](#submitting-a-pull-request)
6. [Reporting Bugs](#reporting-bugs)
7. [Requesting Features](#requesting-features)

---

## What Can I Contribute?

| Type | Examples |
|------|---------|
| **Bug fixes** | Incorrect citation extraction, broken Streamlit page, BM25 tokenization error |
| **New retrievers** | FAISS-HNSW backend, Qdrant, Weaviate |
| **New rerankers** | ColBERT, Cohere rerank API integration |
| **PDF parser improvements** | Better double-column detection, table extraction |
| **Evaluation metrics** | G-Eval, faithfulness via local LLM |
| **UI improvements** | New chart types on Evaluation page, dark mode |
| **Documentation** | Clearer explanations, examples, translations |
| **Notebooks** | Additional analysis notebooks (e.g. embedding space visualization) |
| **Domain expansion** | Support for other biomedical literature beyond immunology |

---

## Getting Started (Local Dev Setup)

### Prerequisites

- Python 3.11+ (LLaMA-Factory requires ≥ 3.11)
- MongoDB running locally (`mongod`)
- GPU with ≥8 GB VRAM for reranker (CPU fallback available but slow)
- vLLM-compatible GPU (≥20 GB VRAM) for LLM inference

### Clone and Install

```bash
git clone https://github.com/<your-org>/immunology-rag.git
cd immunology-rag

# Create environment
conda create -n immunorag python=3.11 -y
conda activate immunorag

# Install dependencies
pip install -r requirements.txt
pip install vllm>=0.4.0   # for LLM serving

# NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Provide Your Own PDF

Place an immunology PDF in `data/raw/` — the project ships without any PDFs because
source textbooks are copyrighted.

```bash
# Run the index builder
python build_index.py --test-retrieval
```

---

## Project Structure at a Glance

```
src/
  constant.py          # All config loaded from config.yaml — edit there first
  pdf_parser.py        # PDF text + image extraction
  chunker.py           # Parent-child chunking + MongoDB
  embedder.py          # BGE-M3 dense indexing + BM25
  pipeline.py          # Full RAG pipeline orchestrator
  retriever/           # BM25, Chroma, FAISS retrievers
  reranker/            # BGE cross-encoder reranker
  client/              # vLLM, HyDE, data-gen, MongoDB clients
  server/              # FastAPI semantic chunking service (port 6000)
train/
  build_train_data.py  # Synthetic QA generation
  train_reranker.py    # Reranker fine-tuning
  train_llm_sft.py     # LLM LoRA SFT via LLaMA-Factory
pages/                 # Streamlit multi-page UI
app.py                 # Streamlit entry point
build_index.py         # One-command index builder
evaluate.py            # System evaluation + charts
```

---

## Contribution Guidelines

### Code Style

- Follow existing code style (no strict linter enforced, but be consistent)
- Keep functions focused — one function, one responsibility
- Add a short docstring to every new public function
- Use type hints for function signatures

### Config Changes

- All tunable parameters belong in `config.yaml`, not hardcoded in source files
- Expose new config keys via `src/constant.py` following the existing pattern

### Tests

There is no formal test suite yet. For now:
- Run `python build_index.py --test-retrieval` to verify end-to-end indexing
- Run `python evaluate.py --quick` to verify the pipeline produces sensible scores
- If you add a new retriever or reranker, include a minimal usage example in your PR

### Commit Messages

Use short, imperative present-tense subject lines:
```
fix: handle empty page_content in post_processing
feat: add Qdrant vector store backend
docs: clarify LoRA merge step in runbook
refactor: extract citation parsing into separate module
```

---

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/my-improvement
   ```

2. **Make your changes** and verify the pipeline still works end-to-end

3. **Update documentation** if your change affects behaviour:
   - `docs/deep_code_analysis.md` for code-level changes
   - `docs/autodl_runbook.md` for deployment changes
   - Inline docstrings for API changes

4. **Push and open a PR** against `main` with a clear description:
   - What problem does this solve?
   - How did you test it?
   - Any known limitations?

---

## Reporting Bugs

Open a GitHub Issue using the **Bug Report** template. Include:

- Operating system and Python version
- GPU model and VRAM
- Full error traceback
- Minimal reproduction steps
- Which services were running (MongoDB, vLLM, semantic chunking)?

---

## Requesting Features

Open a GitHub Issue using the **Feature Request** template. Describe:

- The use case or problem you are trying to solve
- Your proposed solution (if you have one)
- Any alternative approaches you considered

---

## Questions?

Open a GitHub Discussion or an Issue tagged `question`. There are no silly questions —
RAG systems have many moving parts and we are happy to help.
