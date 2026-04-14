# =============================================================================
# ImmunoBiology RAG — FAISS Dense Retriever (Fallback Option)
# =============================================================================
# Adapted from Tesla RAG: src/retriever/faiss_retriever.py
# Changes: English comments; uses BGE-M3 embeddings (not BCE model).
# Activated when config.yaml: vectorstore.backend = "faiss"

import os
import pickle
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src import constant
from src.retriever.retriever import BaseRetriever


class FaissRetriever(BaseRetriever):
    """
    Dense vector retriever backed by FAISS.
    Fallback option when Chroma is not preferred.
    """

    def __init__(self, docs=None, retrieve: bool = False):
        super().__init__(docs, retrieve)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=constant.bge_m3_model_path,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if retrieve and os.path.exists(constant.faiss_db_path):
            self.vector_store = FAISS.load_local(
                constant.faiss_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"[FAISS] Index loaded from {constant.faiss_db_path}")
        else:
            print(f"[FAISS] Building index from {len(docs)} documents...")
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(constant.faiss_db_path)
            print(f"[FAISS] Index saved to {constant.faiss_db_path}")

    def retrieve_topk(self, query: str, topk: int = 10) -> List[Document]:
        """Return top-k documents by L2 distance."""
        results = self.vector_store.similarity_search_with_score(query, k=topk)
        return [doc for doc, _score in results]
