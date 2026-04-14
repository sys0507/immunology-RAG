# =============================================================================
# ImmunoBiology RAG — BM25 Sparse Retriever
# =============================================================================
# Adapted from Tesla RAG: src/retriever/bm25_retriever.py
# Key changes:
#   - Replaced jieba (Chinese) tokenizer with NLTK English tokenizer
#   - English stopwords from NLTK instead of custom stopwords.txt
#   - Same BM25Retriever class structure and pickle persistence

import os
import pickle
import hashlib
from typing import List

import nltk
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as _LangchainBM25

from src import constant
from src.retriever.retriever import BaseRetriever

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize

_EN_STOPWORDS = set(nltk_stopwords.words("english"))


def english_tokenize(text: str) -> List[str]:
    """
    Tokenize English text for BM25.
    Replaces jieba Chinese tokenizer from Tesla system.
    Lowercases, removes stopwords, keeps only alphabetic tokens of length > 1.
    """
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in _EN_STOPWORDS and len(t) > 1]


class BM25Retriever(BaseRetriever):
    """
    BM25 sparse retriever with English tokenization.

    Design mirrors Tesla's BM25 class exactly:
    - Build mode: creates index from documents, persists to pickle
    - Retrieve mode: loads existing index from pickle
    """

    def __init__(self, docs, retrieve: bool = False):
        super().__init__(docs, retrieve)
        self.retriever = self._get_or_build_retriever(retrieve)

    def _get_or_build_retriever(self, retrieve: bool) -> _LangchainBM25:
        """Load existing BM25 index or build a new one."""
        if retrieve and os.path.exists(constant.bm25_pickle_path):
            with open(constant.bm25_pickle_path, "rb") as f:
                bm25 = pickle.load(f)
            print(f"[BM25] Loaded index from {constant.bm25_pickle_path}")
            return bm25
        else:
            print(f"[BM25] Building index from {len(self.docs)} documents...")
            bm25 = _LangchainBM25.from_documents(
                self.documents,
                preprocess_func=english_tokenize,
            )
            os.makedirs(os.path.dirname(constant.bm25_pickle_path), exist_ok=True)
            with open(constant.bm25_pickle_path, "wb") as f:
                pickle.dump(bm25, f)
            print(f"[BM25] Index saved to {constant.bm25_pickle_path}")
            return bm25

    @property
    def documents(self):
        return self.docs

    def retrieve_topk(self, query: str, topk: int = 10) -> List[Document]:
        """Return top-k BM25-scored documents for the query."""
        self.retriever.k = topk
        return self.retriever.invoke(query)


if __name__ == "__main__":
    texts = [
        "T cell activation requires MHC-peptide complex recognition.",
        "B cells produce antibodies that neutralize pathogens.",
        "Natural killer cells destroy virus-infected cells without prior sensitization.",
    ]
    docs = [
        Document(
            page_content=t,
            metadata={"unique_id": hashlib.md5(t.encode()).hexdigest()}
        )
        for t in texts
    ]
    bm25 = BM25Retriever(docs)
    results = bm25.retrieve_topk("How do T cells recognize antigens?", topk=2)
    for r in results:
        print(r.page_content)
