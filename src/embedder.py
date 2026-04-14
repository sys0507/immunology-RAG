# =============================================================================
# ImmunoBiology RAG — Embedding + Vector Store (Chroma + BM25)
# =============================================================================
# Replaces Tesla's Milvus-based milvus_retriever.py with Chroma.
# Key design decisions:
#   - Chroma (persistent) chosen over Milvus: simpler deployment, no server,
#     supports metadata filtering natively, incremental document addition.
#   - BAAI/bge-m3 generates 1024-dim dense vectors stored in Chroma.
#   - BM25 index uses NLTK English tokenization (replaces jieba for Chinese).
#   - add_documents() checks existing chunk_id values to prevent duplicates.
#   - Metadata filtering: supports doc_type and source_file filters.
#
# Usage:
#   python src/embedder.py              # build full index from split_docs.pkl
#   python src/embedder.py --add path/to/new.pdf  # incremental add

# %% [Cell 1: Imports and configuration]
import os
import re
import pickle
import hashlib
import argparse
from pathlib import Path
from typing import List, Optional

import nltk
import chromadb
from chromadb.config import Settings
from FlagEmbedding import FlagModel
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from tqdm import tqdm

from src import constant
from src.client.mongodb_config import MongoConfig

# Ensure NLTK stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize

_EN_STOPWORDS = set(nltk_stopwords.words("english"))

# Save English stopwords file for BM25Retriever
_STOPWORDS_PATH = Path(constant.stopwords_path)
if not _STOPWORDS_PATH.exists():
    _STOPWORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STOPWORDS_PATH.write_text("\n".join(sorted(_EN_STOPWORDS)), encoding="utf-8")

# Batch size for BGE-M3 encoding (reduce if OOM on smaller GPUs)
EMB_BATCH = 32


# %% [Cell 2: English tokenizer (replaces jieba)]

def english_tokenize(text: str) -> List[str]:
    """
    Tokenize English text for BM25 retrieval.
    Uses NLTK word_tokenize + stopword filtering.
    Retains scientific terms (multi-word not split by hyphens).
    """
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in _EN_STOPWORDS and len(t) > 1]


# %% [Cell 3: Chroma vector store wrapper]

class ChromaStore:
    """
    Wraps Chroma PersistentClient for immunology RAG vector storage.

    Supports:
    - Batch upsert of Document chunks with metadata
    - Dense similarity search with optional metadata filters
    - Incremental add (checks existing chunk_ids to avoid duplicates)
    - Metadata filtering by doc_type and source_file
    """

    def __init__(self, persist_path: str = None, collection_name: str = None):
        persist_path = persist_path or constant.chroma_path
        collection_name = collection_name or constant.chroma_collection

        os.makedirs(persist_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        print(f"[Chroma] Collection '{collection_name}' has "
              f"{self.collection.count()} existing chunks.")

    def get_existing_ids(self) -> set:
        """Return all chunk_ids already in the collection (for dedup)."""
        result = self.collection.get(include=[])
        return set(result["ids"])

    def add_documents(self, docs: List[Document], embeddings: List[List[float]]) -> int:
        """
        Upsert documents into Chroma. Checks for existing IDs to prevent duplicates.

        Args:
            docs:       List of LangChain Documents (must have unique_id in metadata)
            embeddings: Pre-computed dense vectors (len must match docs)

        Returns:
            Number of NEW documents actually inserted.
        """
        existing_ids = self.get_existing_ids()
        new_docs, new_embeds = [], []
        for doc, emb in zip(docs, embeddings):
            uid = doc.metadata.get("unique_id", "")
            if uid and uid not in existing_ids:
                new_docs.append(doc)
                new_embeds.append(emb)

        if not new_docs:
            print("[Chroma] No new documents to add (all already indexed).")
            return 0

        # Within-batch deduplication: if two chunks share the same unique_id
        # (e.g. two near-empty pages both reduced to a single letter), keep only
        # the first occurrence. ChromaDB raises DuplicateIDError on batch upserts
        # that contain repeated IDs even if none of them are already in the store.
        seen_ids: set = set()
        deduped_docs, deduped_embeds = [], []
        for doc, emb in zip(new_docs, new_embeds):
            uid = doc.metadata["unique_id"]
            if uid not in seen_ids:
                seen_ids.add(uid)
                deduped_docs.append(doc)
                deduped_embeds.append(emb)
        if len(deduped_docs) < len(new_docs):
            dropped = len(new_docs) - len(deduped_docs)
            print(f"[Chroma] Dropped {dropped} within-batch duplicate ID(s).")
        new_docs, new_embeds = deduped_docs, deduped_embeds

        # Prepare metadata (Chroma only supports str/int/float/bool metadata values)
        ids       = [d.metadata["unique_id"] for d in new_docs]
        documents = [d.page_content for d in new_docs]
        metadatas = []
        for d in new_docs:
            m = d.metadata.copy()
            # Serialize list fields to JSON strings for Chroma compatibility
            m["images_info"] = str(m.get("images_info", []))
            metadatas.append(m)

        # Insert in batches
        batch = 100
        for i in range(0, len(new_docs), batch):
            self.collection.upsert(
                ids=ids[i:i+batch],
                documents=documents[i:i+batch],
                embeddings=new_embeds[i:i+batch],
                metadatas=metadatas[i:i+batch],
            )

        print(f"[Chroma] Inserted {len(new_docs)} new documents "
              f"(total: {self.collection.count()}).")
        return len(new_docs)

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> List[Document]:
        """
        Perform a dense similarity search.

        Args:
            query_embedding: 1024-dim BGE-M3 dense vector
            n_results:       Number of top results to return
            where:           Optional Chroma metadata filter, e.g.
                             {"doc_type": "textbook"} or
                             {"source_file": "janeway.pdf"}

        Returns:
            List of LangChain Documents sorted by cosine similarity.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        docs = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            meta = dict(meta)
            meta["_chroma_distance"] = dist
            docs.append(Document(page_content=text, metadata=meta))
        return docs


# %% [Cell 4: BGE-M3 embedding wrapper]

class BGEEmbedder:
    """
    Wraps BAAI/bge-m3 for dense vector generation.
    Uses FlagEmbedding (official library) for accurate dense vectors.
    """

    def __init__(self, model_path: str = None):
        model_path = model_path or constant.bge_m3_model_path
        print(f"[Embedder] Loading BGE-M3 from: {model_path}")
        self.model = FlagModel(
            model_path,
            query_instruction_for_retrieval="Represent this immunology text for retrieval: ",
            use_fp16=True,  # Halved precision for speed; minimal accuracy loss
        )
        print("[Embedder] BGE-M3 loaded.")

    def encode_docs(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of document texts into dense vectors (batch mode)."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), EMB_BATCH), desc="Encoding docs"):
            batch = texts[i:i+EMB_BATCH]
            embs = self.model.encode(batch)
            all_embeddings.extend(embs.tolist())
        return all_embeddings

    def encode_query(self, query: str) -> List[float]:
        """Encode a single query string into a dense vector."""
        emb = self.model.encode_queries([query])
        return emb[0].tolist()


# %% [Cell 5: Full indexer class]

class Embedder:
    """
    Main indexing orchestrator.
    Builds and manages both Chroma (dense) and BM25 (sparse) indices.
    """

    def __init__(self):
        self.bge = BGEEmbedder()
        self.chroma = ChromaStore()
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._load_bm25()

    def _load_bm25(self) -> None:
        """Load persisted BM25 index if it exists."""
        if os.path.exists(constant.bm25_pickle_path):
            with open(constant.bm25_pickle_path, "rb") as f:
                self.bm25_retriever = pickle.load(f)
            print(f"[Embedder] BM25 index loaded from {constant.bm25_pickle_path}")

    def _save_bm25(self) -> None:
        """Persist BM25 index to disk."""
        os.makedirs(Path(constant.bm25_pickle_path).parent, exist_ok=True)
        with open(constant.bm25_pickle_path, "wb") as f:
            pickle.dump(self.bm25_retriever, f)
        print(f"[Embedder] BM25 index saved to {constant.bm25_pickle_path}")

    def build_index(self, split_docs: List[Document]) -> None:
        """
        Build full index from a list of chunk Documents.
        1. Encode all texts with BGE-M3
        2. Upsert into Chroma
        3. Build BM25 index from all chunks
        4. Save BM25 to disk
        5. Write index report
        """
        print(f"[Embedder] Building index for {len(split_docs)} chunks...")

        # Dense encoding
        texts = [d.page_content for d in split_docs]
        embeddings = self.bge.encode_docs(texts)

        # Chroma upsert
        inserted = self.chroma.add_documents(split_docs, embeddings)

        # BM25 (rebuild from all chunks — needed after any new addition)
        print("[Embedder] Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_documents(
            split_docs,
            preprocess_func=english_tokenize,
        )
        self._save_bm25()

        # Index report
        self._write_index_report(split_docs)
        print(f"[Embedder] Index build complete. {inserted} new docs added.")

    def add_documents(self, pdf_path: str) -> None:
        """
        Incremental indexing: parse + chunk + embed a new PDF and add to existing index.
        Checks chunk_ids to prevent duplicates.

        This is called from the Streamlit Document Management page.
        """
        from src.pdf_parser import parse_pdf
        from src.chunker import texts_split

        pdf_path = Path(pdf_path)
        print(f"[Embedder] Incrementally adding: {pdf_path.name}")

        raw_docs = parse_pdf(pdf_path)
        new_chunks = texts_split(raw_docs, dry_run=False)

        # Dense encoding
        texts = [d.page_content for d in new_chunks]
        embeddings = self.bge.encode_docs(texts)

        # Chroma upsert (dedup handled inside)
        inserted = self.chroma.add_documents(new_chunks, embeddings)

        # Rebuild BM25 from ALL chunks (BM25 requires full corpus rebuild)
        print("[Embedder] Rebuilding BM25 with new documents...")
        all_chunks = self._load_all_chunks_from_mongo()
        self.bm25_retriever = BM25Retriever.from_documents(
            all_chunks,
            preprocess_func=english_tokenize,
        )
        self._save_bm25()
        print(f"[Embedder] Incremental add complete: {inserted} new chunks indexed.")

    def _load_all_chunks_from_mongo(self) -> List[Document]:
        """Load all chunks from MongoDB for BM25 rebuild."""
        collection = MongoConfig.get_collection(constant.mongo_collection)
        cursor = collection.find({}, {"page_content": 1, "metadata": 1, "_id": 0})
        docs = []
        for rec in cursor:
            if rec.get("page_content"):
                docs.append(Document(
                    page_content=rec["page_content"],
                    metadata=rec.get("metadata", {}),
                ))
        return docs

    def _write_index_report(self, split_docs: List[Document]) -> None:
        """Write a text summary of the index to outputs/diagnostics/."""
        from collections import Counter
        source_counts = Counter(d.metadata.get("source_file", "unknown") for d in split_docs)
        chroma_total = self.chroma.collection.count()

        # Estimate index size
        chroma_size_mb = sum(
            f.stat().st_size for f in Path(constant.chroma_path).rglob("*") if f.is_file()
        ) / (1024 * 1024)
        bm25_size_mb = os.path.getsize(constant.bm25_pickle_path) / (1024 * 1024) \
            if os.path.exists(constant.bm25_pickle_path) else 0

        lines = [
            "ImmunoBiology RAG — Index Report",
            "=" * 50,
            f"Total chunks (Chroma): {chroma_total}",
            f"Total chunks (this batch): {len(split_docs)}",
            f"Embedding dimension: 1024 (BGE-M3 dense)",
            f"Chroma index size: {chroma_size_mb:.1f} MB",
            f"BM25 index size:   {bm25_size_mb:.1f} MB",
            "",
            "Per-document chunk counts:",
        ]
        for src, cnt in sorted(source_counts.items()):
            lines.append(f"  {src}: {cnt} chunks")

        report_text = "\n".join(lines)
        os.makedirs(constant.diagnostics_dir, exist_ok=True)
        with open(constant.index_report, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n[Embedder] Index report:\n{report_text}")
        print(f"[Embedder] Report saved to {constant.index_report}")


# %% [Cell 6: CLI entry point]

def main():
    parser = argparse.ArgumentParser(description="ImmunoBiology RAG Embedder")
    parser.add_argument(
        "--add", type=str, default=None,
        help="Incrementally add a single PDF to the existing index"
    )
    args = parser.parse_args()

    embedder = Embedder()

    if args.add:
        embedder.add_documents(args.add)
    else:
        # Full build from cached split docs
        split_docs_path = constant.split_docs_path
        if not os.path.exists(split_docs_path):
            print(f"[Embedder] No split_docs.pkl found at {split_docs_path}. "
                  "Run build_index.py first.")
            return
        with open(split_docs_path, "rb") as f:
            split_docs = pickle.load(f)
        print(f"[Embedder] Loaded {len(split_docs)} split docs from cache.")
        embedder.build_index(split_docs)


if __name__ == "__main__":
    main()
