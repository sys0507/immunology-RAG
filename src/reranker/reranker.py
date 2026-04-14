# =============================================================================
# ImmunoBiology RAG — Abstract Base Reranker
# =============================================================================
# Copied verbatim from Tesla RAG: src/reranker/reranker.py
# Changes: English comments only.

from abc import ABC, abstractmethod
from typing import List, Tuple


class RerankerBase(ABC):
    """
    Abstract base class for all reranker implementations.
    Ensures a consistent interface for swapping rerankers (strategy pattern).
    """

    def __init__(self, model_path: str, max_length: int = 512) -> None:
        super().__init__()
        self.model_path = model_path
        self.max_length = max_length

    @abstractmethod
    def rank(self, query: str, candidate_docs: List, top_k: int = 10) -> List:
        """
        Rerank candidate documents by relevance to the query.

        Args:
            query:          User query string
            candidate_docs: List of LangChain Documents to rerank
            top_k:          Number of top documents to return

        Returns:
            Top-k Documents sorted by relevance score (highest first)
        """
        raise NotImplementedError
