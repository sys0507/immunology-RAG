# =============================================================================
# ImmunoBiology RAG — Chroma Dense Retriever
# =============================================================================
# Replaces Tesla's milvus_retriever.py with Chroma-based dense retrieval.
# Key design decisions:
#   - Chroma PersistentClient (no server needed, unlike Milvus)
#   - BGE-M3 dense vectors (1024-dim, same as Tesla's Milvus setup)
#   - RRF fusion implemented in Python (was built-in to Milvus)
#   - Metadata filtering by doc_type and source_file
#   - add_documents() for incremental indexing (called from embedder.py)
#
# Note: This class handles ONLY the dense (Chroma) retrieval.
# Hybrid retrieval (BM25 + Chroma + RRF) is in src/retriever.py (HybridRetriever).

import os
import pickle
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from src import constant
from src.retriever.retriever import BaseRetriever
from src.client.mongodb_config import MongoConfig


class ChromaRetriever(BaseRetriever):
    """
    Dense vector retriever backed by Chroma.

    Requires:
    - embedder.py has already built the Chroma index
    - BGE-M3 model loaded separately (passed as encode_query callable)
    """

    def __init__(self, docs=None, retrieve: bool = True,
                 encode_query_fn=None):
        """
        Args:
            docs:            Unused (index is always loaded from disk)
            retrieve:        Always True for Chroma (index pre-built by embedder.py)
            encode_query_fn: Callable(query: str) -> List[float]
                             If None, requires external call to set_encoder()
        """
        super().__init__(docs, retrieve)
        self._encode_query = encode_query_fn

        self.chroma_client = chromadb.PersistentClient(
            path=constant.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=constant.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.mongo_collection = MongoConfig.get_collection(constant.mongo_collection)
        print(f"[ChromaRetriever] Collection has {self.collection.count()} chunks.")

    def set_encoder(self, encode_query_fn) -> None:
        """Set the query encoding function (callable: str -> List[float])."""
        self._encode_query = encode_query_fn

    def retrieve_topk(
        self,
        query: str,
        topk: int = 10,
        doc_type: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve top-k documents by dense cosine similarity.

        Args:
            query:       User query string
            topk:        Number of results
            doc_type:    Optional filter: "textbook" or "paper"
            source_file: Optional filter: specific PDF filename

        Returns:
            List of LangChain Documents with metadata from MongoDB.
        """
        if self._encode_query is None:
            raise RuntimeError(
                "ChromaRetriever: encode_query_fn not set. "
                "Call set_encoder() or pass encode_query_fn to __init__."
            )

        query_emb = self._encode_query(query)

        # Build optional where filter
        where_filter = None
        if doc_type and source_file:
            where_filter = {"$and": [
                {"doc_type": {"$eq": doc_type}},
                {"source_file": {"$eq": source_file}},
            ]}
        elif doc_type:
            where_filter = {"doc_type": {"$eq": doc_type}}
        elif source_file:
            where_filter = {"source_file": {"$eq": source_file}}

        kwargs = {
            "query_embeddings": [query_emb],
            "n_results": min(topk, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        # Enrich with full metadata from MongoDB (mirrors Tesla's MongoDB lookup)
        docs = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            uid = meta.get("unique_id")
            if uid:
                mongo_rec = self.mongo_collection.find_one({"unique_id": uid})
                if mongo_rec:
                    full_meta = mongo_rec.get("metadata", meta)
                    docs.append(Document(
                        page_content=mongo_rec.get("page_content", text),
                        metadata={**full_meta, "_chroma_distance": dist},
                    ))
                    continue
            # Fallback: use Chroma metadata directly
            docs.append(Document(
                page_content=text,
                metadata={**dict(meta), "_chroma_distance": dist},
            ))
        return docs
