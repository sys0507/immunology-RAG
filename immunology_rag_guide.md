# ImmunoBiology RAG Q&A System — Claude Code Task Brief

## Project Background & Objective

Following the engineering paradigm of the Tesla Manual RAG Q&A system, build a fully English-language RAG-based question-answering system for immunology textbooks (ImmunoBiology Q&A System) from scratch in the directory `E:\LLM\notebook\Immunology_RAG`.

- **Reference source code**: `E:\LLM\notebook\Tesla知识引擎车书问答系统实战—源码\车书问答系统源码`
- **Supporting documentation**:
  - `车书问答系统完整学习与复现指南.md` (Chapters 1–3 only; **skip** Chapter 4 "Domain Data Reproduction Guide / TCR Discovery")
  - `深度代码解析指南.md`

---

## Input Data

- **Current data**: A single English immunology textbook PDF (e.g., *Janeway's Immunobiology*) placed in `data/raw/`
- **Scalability requirement**: The system architecture must support multiple documents from day one (multiple PDF textbooks + multiple PDF papers). Single-file paths must never be hardcoded anywhere. Specifically:
  - All `.pdf` files under `data/raw/` must be automatically discovered and batch-processed
  - Each chunk's metadata must include: source filename (`source_file`), document type (`doc_type: textbook | paper`), chapter, and page number
  - The vector store must support **incremental updates** — adding a new PDF should only append new embeddings without rebuilding the entire index
  - Answer citations in the UI must display the specific book/paper title, chapter, and page number
- **Language**: The entire system must be built in English — no Chinese in code comments, prompts, UI text, or log output

---

## Section 1 — PDF Parsing & Preprocessing

1. You may reference the Tesla RAG system's PDF parsing approach (text extraction + image extraction), but **do not blindly copy its parameters**. Re-calibrate for the actual layout of the immunology textbook, including:
   - Coordinate crop ranges for headers and footers
   - Image filtering thresholds (minimum width/height in pixels, minimum file size)
   - Figure caption detection rules (typically starting with `"Figure X.X"` or `"FIGURE"`)
   - Table detection and extraction strategy (consider `pdfplumber` or `camelot`)
2. The parsing logic must run independently per PDF file and automatically handle layout differences between textbooks (single-column) and papers (double-column)
3. **Document all chosen parameters in comments** at the top of the script, with justification (e.g., "body font ~11pt; header height ~50px based on visual inspection")
4. Extracted text and images must be organized by document and chapter under `data/processed/`, with a per-document extraction quality report (page count, image count, empty page warnings)

---

## Section 2 — Text Chunking

1. As the source material is English academic text, use a **semantically aware** chunking strategy rather than naive fixed-length splitting:
   - Preferred: paragraph-boundary + sliding window (`RecursiveCharacterTextSplitter` with English separators)
   - Alternative: semantic chunking (dynamic splits based on cosine similarity of sentence embeddings)
2. **Chunk parameters** (`chunk_size`, `chunk_overlap`) must be tuned for long English academic sentences; justify the chosen values in comments
3. Every chunk must carry complete metadata:
   ```json
   {
     "source_file": "Janeway_Immunobiology_10e.pdf",
     "doc_type": "textbook",
     "chapter": "Chapter 3",
     "page": 87,
     "has_figure_caption": false,
     "chunk_id": "janeway_ch3_p87_001"
   }
   ```
4. Run a dry-run chunking pass and output a chunk-length distribution histogram saved to `outputs/diagnostics/chunk_length_dist.png` for parameter tuning reference

---

## Section 3 — Embedding Model & Vector Store

1. Select an embedding model with strong English academic text performance. Recommended candidates (A100 VRAM is not a constraint):
   - `BAAI/bge-m3` (**first choice** — multilingual, top MTEB English scores, supports dense + sparse + ColBERT multi-path retrieval)
   - `BAAI/bge-large-en-v1.5` (English-only, strong alternative)
   - `intfloat/e5-mistral-7b-instruct` (excellent on English academic text; fully fits on A100)
2. Use **Chroma** as the vector store (supports persistence + metadata filtering + incremental document addition; preferred over FAISS); provide a FAISS fallback option in `config.yaml`
3. **Incremental update support**: implement an `add_documents(pdf_path)` interface in `embedder.py` that checks existing `chunk_id` values to prevent duplicate insertion
4. After index construction, output a summary report (total chunk count, per-document chunk counts, embedding dimension, index disk size)

---

## Section 4 — Reranker: Model Selection & Fine-Tuning

### 4.1 Model Selection
For English tasks on A100:
- `BAAI/bge-reranker-v2-m3` (**first choice** — multilingual, top BEIR benchmark English scores)
- `BAAI/bge-reranker-large` (English-only, lightweight alternative)
- If another model is chosen, document the reasoning in comments

### 4.2 Fine-Tuning Requirements
- Training data must be **automatically constructed** from immunology textbook content:
  - Positive pairs: LLM-generated questions for a passage → (question, relevant_chunk)
  - Hard negatives: top-K retrieved chunks that are not the ground-truth answer
- **Real-time training visualization is required**, including at minimum:
  - Train loss vs. eval loss curves (updated every N steps)
  - Learning rate schedule curve
  - Eval metric curves (MRR@10, NDCG@10)
  - Use both `matplotlib` and `TensorBoard`; checkpoint strategy configured in `config.yaml`
- **Pre- vs. post-fine-tuning performance comparison is required**:
  - Metrics: MRR@10, NDCG@10, Recall@5 on the held-out test set
  - Visualized as a grouped bar chart, saved to `outputs/reranker_eval/comparison.png`
  - Numeric results also saved as `outputs/reranker_eval/comparison.csv`

---

## Section 5 — LLM: Model Selection & (Optional) SFT

### 5.1 Model Selection
A100 VRAM supports larger models. Recommended candidates:
- `Qwen3-8B` (strong bilingual instruction following; **may be used as-is**)
- `Meta-Llama-3.1-8B-Instruct` (English-native; excellent academic QA generation quality; **recommended**)
- `Mistral-7B-Instruct-v0.3` (lightweight English model; fast inference)
- If VRAM allows: `Llama-3.1-70B-Instruct` (4-bit quantized ~40GB; viable on 80GB A100)
- Make the model choice configurable in `config.yaml`; document the final selection reasoning in `docs/deep_code_analysis.md`

### 5.2 Supervised Fine-Tuning (recommended for domain adaptation)
- Training data format: ChatML / Alpaca, auto-generated QA pairs from textbook content
- **Training visualization** requirements: same as Section 4.2
- **Pre- vs. post-SFT performance comparison**:
  - Metrics: ROUGE-L, BERTScore-F1, Answer Faithfulness
  - Visualized and saved to `outputs/llm_eval/`
- On A100, use `bf16` + Flash Attention 2 for training acceleration; quantization is not required

---

## Section 6 — RAG Pipeline Integration

1. Full pipeline flow: `Query → Hybrid Retrieval → Reranker Re-ranking → LLM Generation → Answer + Source Output`
2. **Hybrid Search** (dense + BM25 sparse fusion):
   - Dense: Chroma / FAISS vector retrieval
   - Sparse: `rank_bm25` library
   - Fusion strategy: Reciprocal Rank Fusion (RRF); weights tunable in `config.yaml`
3. **Multi-document awareness**: retrieval must support filtering by `doc_type` (textbook / paper) or `source_file`, enabling queries like "search textbooks only" or "search papers only"
4. All pipeline modules must be **independently replaceable** (strategy pattern / dependency injection) for ablation experiments
5. Multi-turn conversation support: chat history appended to system prompt with a configurable history window length
6. Per-module latency must be automatically recorded for use by the evaluation module

---

## Section 7 — Frontend UI (Streamlit)

Build a **multi-page Streamlit application** (replacing Gradio) for a more flexible layout and better document citation display.

**Page 1 — Q&A (`app.py` / `pages/01_QA.py`)**:
- Question input box + "Ask" button
- Answer display area (Markdown rendering with inline citation markers like `[1]`)
- Source display: Top-K retrieved passages, each showing `Document | Chapter | Page | Relevance Score`
- Related image display (when retrieved results include figure captions)
- Multi-turn conversation history panel (managed via `st.session_state`)

**Page 2 — Document Management (`pages/02_Documents.py`)**:
- List all currently indexed documents (filename, doc_type, chunk count, indexing timestamp)
- Support **uploading a new PDF and triggering incremental indexing** (calls `embedder.add_documents()`)
- View per-document extraction quality report

**Page 3 — Evaluation (`pages/03_Evaluation.py`)**:
- Display results from the latest `evaluate.py` run (loaded from `outputs/system_eval/`)
- Support running a lightweight quick evaluation online (random sample of 20 QA pairs)

**Page 4 — Settings (`pages/04_Settings.py`)**:
- Visual interface for editing key parameters in `config.yaml` (Top-K, RRF weights, LLM temperature, etc.)
- Hot-reload configuration without restarting the service

---

## Section 8 — System Evaluation (with Visualization Output)

`evaluate.py` must run as a **standalone script**, saving all results to `outputs/system_eval/`:

| Dimension | Metrics | Visualization | Output File |
|---|---|---|---|
| Retrieval quality | Recall@K (K=1,3,5,10), MRR@10 | Line chart (K vs Recall) | `retrieval_recall.png` |
| Re-ranking quality | Precision@1, NDCG@5 | Bar chart | `reranker_precision.png` |
| Generation quality | ROUGE-L, BERTScore-F1 | Grouped bar chart | `generation_quality.png` |
| End-to-end | Answer Relevance, Faithfulness, Context Precision | Radar chart | `e2e_radar.png` |
| Latency | Per-module response time (ms) | Stacked bar chart | `latency_breakdown.png` |
| Summary | All metrics aggregated | — | `evaluation_report.html` |

- Evaluation QA set: auto-generate 100–200 QA pairs with ground-truth answers, saved to `data/train/eval_qa.jsonl`
- Support **multi-document evaluation**: report retrieval performance separately for textbook-sourced vs. paper-sourced chunks
- All charts use consistent English labeling and are aggregated into `evaluation_report.html`

---

## Section 9 — Project Structure

Generate a **complete, fully executable** project scaffold. All scripts must contain complete implementations (no placeholder stubs):

```
Immunology_RAG/
├── data/
│   ├── raw/                              # Raw PDFs (placed here manually; multi-file supported)
│   ├── processed/
│   │   ├── {doc_name}/
│   │   │   ├── text/                     # Chapter-level extracted text (.json with metadata)
│   │   │   └── images/                   # Extracted figures (.png)
│   │   └── extraction_report.json        # Per-document parsing quality summary
│   └── train/
│       ├── reranker_train.jsonl
│       ├── reranker_test.jsonl
│       ├── sft_train.jsonl
│       └── eval_qa.jsonl
├── src/
│   ├── pdf_parser.py                     # PDF parsing & image extraction (batch multi-doc)
│   ├── chunker.py                        # Semantic chunking + metadata injection
│   ├── embedder.py                       # Embeddings + Chroma vector store (incremental)
│   ├── retriever.py                      # Hybrid search (Dense + BM25 + RRF)
│   ├── reranker.py                       # Reranker inference wrapper
│   ├── llm.py                            # LLM loading, inference, prompt management
│   ├── pipeline.py                       # Full modular RAG pipeline
│   └── utils.py                          # Logging, timing, config loading utilities
├── train/
│   ├── build_train_data.py               # Auto-build reranker + SFT + eval datasets
│   ├── train_reranker.py                 # Reranker fine-tuning (visualization + comparison)
│   └── train_llm_sft.py                  # LLM SFT (visualization + comparison)
├── evaluate.py                           # End-to-end system evaluation (full visualization)
├── app.py                                # Streamlit multi-page UI entry point
├── pages/
│   ├── 01_QA.py
│   ├── 02_Documents.py
│   ├── 03_Evaluation.py
│   └── 04_Settings.py
├── notebooks/
│   ├── 01_pdf_parser.ipynb
│   ├── 02_chunker.ipynb
│   ├── 03_embedder.ipynb
│   ├── 04_retriever.ipynb
│   ├── 05_reranker.ipynb
│   ├── 06_llm.ipynb
│   ├── 07_pipeline.ipynb
│   ├── 08_build_train_data.ipynb
│   ├── 09_train_reranker.ipynb
│   ├── 10_train_llm_sft.ipynb
│   └── 11_evaluate.ipynb
├── config.yaml                           # Global config (model paths, hyperparameters, etc.)
├── requirements.txt
├── outputs/
│   ├── diagnostics/                      # Chunk distribution plots, layout inspection reports
│   ├── vectorstore/                      # Persisted Chroma vector store
│   ├── models/
│   │   ├── reranker_finetuned/
│   │   └── llm_finetuned/
│   ├── reranker_eval/
│   ├── llm_eval/
│   └── system_eval/
└── docs/
    ├── deep_code_analysis.md             # Deep code analysis guide
    └── autodl_runbook.md                 # AutoDL A100 step-by-step runbook
```

---

## Section 10 — Documentation Requirements

**`docs/deep_code_analysis.md`** — modeled after the original `深度代码解析指南.md`:
- Module overview and its position in the overall pipeline
- Step-by-step explanation of key functions/classes (with parameter descriptions)
- Inter-module data flow and interface contracts
- Design decision rationale (chosen approach vs. alternatives considered)
- Notes on multi-document extension behavior

**`docs/autodl_runbook.md`** — complete step-by-step guide for **AutoDL A100 (40/80GB) single-GPU** environments:
- Image selection (recommended: PyTorch 2.x + CUDA 12.x)
- Environment setup (`conda` / `pip`; Flash Attention 2 compilation instructions)
- Data upload methods (AutoDL file manager / `scp` / OSS)
- **Execution order** (each step with estimated runtime):
  1. PDF parsing & preprocessing
  2. Chunking dry-run & diagnostics
  3. Vector store construction
  4. Training dataset generation
  5. Reranker fine-tuning + evaluation comparison
  6. LLM SFT + evaluation comparison
  7. End-to-end system evaluation
  8. Launch Streamlit UI
- A100 performance optimization tips (`bf16`, Flash Attention 2, `torch.compile`)
- AutoDL port mapping & Streamlit public access configuration (`--server.port`, `--server.address`)
- Incremental PDF addition workflow
- Common errors and solutions

---

## Section 11 — Notebook Companion Requirements

Every independently executable `.py` script must have a corresponding `.ipynb` notebook in `notebooks/`, for step-by-step learning and execution.

> Streamlit UI scripts (`app.py` and `pages/`) do not require notebook companions.

**Notebook structure and conventions** (each `.ipynb` must follow):

1. **Opening Markdown cell**: describe the corresponding `.py` script, its position in the pipeline, and the learning objectives of this notebook

2. **Cell decomposition principle**:
   - Each logical step (imports, configuration, function definition, execution, result verification) must be its own cell or group of cells
   - **Never paste the entire script into a single cell**
   - Function definition cells and function call/demo cells must be separate

3. **Markdown cell before each code cell** explaining:
   - What this step does (What)
   - Why it is done this way (Why) — including parameter justification and method trade-offs

4. **Intermediate result visualization** after key steps, for example:
   - After PDF parsing: print extracted text samples; display extracted images
   - After chunking: print 3–5 example chunks with metadata; display chunk-length histogram
   - After embedding: print vector dimensions; display chunk count statistics
   - After retrieval: print Top-K results with relevance scores
   - During training: display live loss curves and evaluation metrics

5. **Cross-reference markers**: in the corresponding `.py` script, mark each logical block with `# %% [Cell N: description]` so that the `.py` and `.ipynb` can be read side-by-side

6. **Closing Markdown cell**: summarize the outputs produced by this notebook (file paths) and indicate which notebook to run next

---

## Section 12 — Pre-Execution Checklist (Claude Code — Read Before Starting)

Complete the following checks before generating any code. **Ask the user for clarification on any uncertain item before proceeding.**

1. **PDF layout inspection (required first)**: Open the PDF in `data/raw/` using `PyMuPDF`, inspect pages 1, 50, 100, and 200. Record: page dimensions (pt), header/footer coordinate ranges, body text block boundaries, image size distribution (min/median/max). Use these measurements to set parsing parameters. Save the inspection results to `outputs/diagnostics/pdf_layout_report.txt`.

2. **Multi-document architecture first**: All module interfaces must accept list inputs (`List[Path]`) from the start. No hardcoded single-file paths anywhere in the codebase.

3. **Chunking dry-run**: Run a dry-run chunking pass on the extracted text, output a chunk-length histogram, and only then finalize `chunk_size` and `chunk_overlap`.

4. **Training data quality spot-check**: After generating training data, randomly sample and print 10 examples for the user to review. Wait for confirmation before starting any training run.

5. **Visualization pre-test**: Before any full training run, test all visualization code with mock data to confirm plots render correctly — avoid discovering broken plotting code only after a long training job completes.

6. **Incremental indexing validation**: After the initial vector store is built, simulate adding a small test PDF and verify the incremental update logic is correct (no duplicate insertions, metadata fields populated correctly).
