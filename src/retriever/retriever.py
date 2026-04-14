# =============================================================================
# ImmunoBiology RAG — Abstract Base Retriever
# =============================================================================
# Copied verbatim from Tesla RAG: src/retriever/retriever.py
# Changes: English comments only.

from abc import ABC, abstractmethod
from typing import List


class BaseRetriever(ABC):
    """
    Abstract base class for all retriever implementations.
    Ensures a consistent interface for swapping retrievers (strategy pattern).
    """

    def __init__(self, docs, retrieve: bool = False) -> None:
        super().__init__()
        self.docs = docs
        self.retrieve = retrieve  # True = load existing index; False = build new

    @abstractmethod
    def retrieve_topk(self, query: str, topk: int = 3) -> List:
        """Return top-k most relevant Documents for the given query."""
        raise NotImplementedError
