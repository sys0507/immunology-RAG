# =============================================================================
# ImmunoBiology RAG — Full Pipeline
# =============================================================================
# Adapted from Tesla RAG: infer.py
# Key changes:
#   - Converted interactive loop → RAGPipeline class (strategy pattern)
#   - MilvusRetriever → ChromaRetriever + BM25Retriever (HybridRetriever)
#   - Multi-turn conversation via chat_history with configurable window
#   - HyDE query expansion (optional, config-controlled)
#   - Per-module latency tracking via LatencyTracker
#   - Structured output dict instead of raw string

from typing import Dict, Any, List, Optional

from src import constant
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.chroma_retriever import ChromaRetriever
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.client.llm_client import request_chat
from src.client.llm_hyde_client import request_hyde
from src.utils import merge_docs, post_processing, format_context, LatencyTracker


# =============================================================================
# HybridRetriever: BM25 + Chroma dense + RRF fusion
# =============================================================================

def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a single rank position."""
    return 1.0 / (k + rank)


def rrf_fuse(
    bm25_docs: List,
    dense_docs: List,
    k: int = 60,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
) -> List:
    """
    Fuse two ranked lists via Reciprocal Rank Fusion (RRF).

    Score for a document = bm25_weight * 1/(k+rank_bm25)
                         + dense_weight * 1/(k+rank_dense)
    Documents not present in a list are treated as rank = len+1 (unranked).

    Args:
        bm25_docs:    Ranked list from BM25
        dense_docs:   Ranked list from dense retrieval
        k:            RRF smoothing constant (default 60)
        bm25_weight:  Weight applied to BM25 scores
        dense_weight: Weight applied to dense scores

    Returns:
        Combined list sorted by fused score descending (unique docs)
    """
    from langchain_core.documents import Document

    # uid → (doc, fused_score)
    score_map: Dict[str, List] = {}

    def uid(doc) -> str:
        return doc.metadata.get("unique_id") or doc.page_content[:80]

    # BM25 ranks
    for rank, doc in enumerate(bm25_docs, 1):
        key = uid(doc)
        score_map.setdefault(key, [doc, 0.0])
        score_map[key][1] += bm25_weight * _rrf_score(rank, k)

    # Dense ranks
    for rank, doc in enumerate(dense_docs, 1):
        key = uid(doc)
        score_map.setdefault(key, [doc, 0.0])
        score_map[key][1] += dense_weight * _rrf_score(rank, k)

    ranked = sorted(score_map.values(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _score in ranked]


# =============================================================================
# RAGPipeline: end-to-end orchestrator
# =============================================================================

class RAGPipeline:
    """
    Full RAG pipeline: Query → Retrieval → Reranking → LLM → Post-process.

    Designed to be modular: retriever / reranker / LLM can each be swapped
    at construction time.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.answer("What activates T cells?")
        print(result["answer"])
        print(result["cite_pages"])

    Multi-turn:
        pipeline = RAGPipeline()
        r1 = pipeline.answer("What are T cells?")
        r2 = pipeline.answer("How are they activated?")   # uses history
    """

    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[ChromaRetriever] = None,
        reranker=None,
        encode_query_fn=None,
    ):
        """
        Initialize pipeline components.

        Args:
            bm25_retriever:  Pre-built BM25Retriever (loads from pickle if None)
            dense_retriever: Pre-built ChromaRetriever (loads from Chroma path if None)
            reranker:        Reranker instance (loads BGEM3ReRanker if None)
            encode_query_fn: Query embedding callable for dense retriever
                             If None, BGE-M3 embedder is loaded automatically
        """
        # ----- BM25 -----
        if bm25_retriever is not None:
            self.bm25 = bm25_retriever
        else:
            print("[Pipeline] Loading BM25 retriever from pickle...")
            self.bm25 = BM25Retriever(docs=None, retrieve=True)

        # ----- Dense (Chroma) -----
        if dense_retriever is not None:
            self.dense = dense_retriever
        else:
            print("[Pipeline] Loading Chroma retriever...")
            # Load BGE-M3 encoder if not provided
            if encode_query_fn is None:
                encode_query_fn = self._load_default_encoder()
            self.dense = ChromaRetriever(encode_query_fn=encode_query_fn)

        # ----- Reranker -----
        if reranker is not None:
            self.reranker = reranker
        else:
            print("[Pipeline] Loading BGE reranker...")
            self.reranker = BGEM3ReRanker()

        # ----- Config -----
        self.bm25_topk     = constant.bm25_topk
        self.dense_topk    = constant.dense_topk
        self.rerank_topk   = constant.rerank_topk
        self.rrf_k         = constant.rrf_k
        self.rrf_bm25_w    = constant.rrf_bm25_weight
        self.rrf_dense_w   = constant.rrf_dense_weight
        self.hyde_enabled  = constant.hyde_enabled
        self.history_window = constant.llm_history_window

        # ----- State -----
        self._chat_history: List[dict] = []   # multi-turn history

        print("[Pipeline] Ready.")

    # ------------------------------------------------------------------
    # Encoder loader (lazy — only called when dense retriever not provided)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_default_encoder():
        """Load BGE-M3 and return a query encoding callable."""
        from FlagEmbedding import FlagModel
        print("[Pipeline] Loading BGE-M3 encoder for dense retrieval...")
        model = FlagModel(
            constant.bge_m3_model_path,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            use_fp16=True,
        )

        def encode_query(query: str):
            return model.encode(query).tolist()

        return encode_query

    # ------------------------------------------------------------------
    # Core answer method
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        doc_type: Optional[str] = None,
        source_file: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a single query.

        Args:
            query:       User question
            doc_type:    Optional metadata filter: "textbook" or "paper"
            source_file: Optional metadata filter: specific PDF filename
            stream:      If True, LLM response streams incrementally

        Returns:
            {
              "answer":         str,
              "cite_pages":     List[int],
              "cite_sources":   List[str],
              "cite_chapters":  List[str],
              "related_images": List[dict],
              "latency_ms":     Dict[str, float],
              "retrieved_docs": List[Document],
            }
        """
        tracker = LatencyTracker()

        # ----------------------------------------------------------
        # Step 1 — HyDE query expansion (optional)
        # ----------------------------------------------------------
        tracker.start("hyde")
        if self.hyde_enabled:
            try:
                hyde_text = request_hyde(query)
                expanded_query = f"{query}\n{hyde_text}"
            except Exception as e:
                print(f"[Pipeline] HyDE failed ({e}), using original query.")
                expanded_query = query
        else:
            expanded_query = query
        tracker.stop("hyde")

        # ----------------------------------------------------------
        # Step 2 — BM25 retrieval
        # ----------------------------------------------------------
        tracker.start("bm25")
        bm25_docs = self.bm25.retrieve_topk(expanded_query, topk=self.bm25_topk)
        tracker.stop("bm25")

        # ----------------------------------------------------------
        # Step 3 — Dense retrieval
        # ----------------------------------------------------------
        tracker.start("dense")
        dense_docs = self.dense.retrieve_topk(
            expanded_query,
            topk=self.dense_topk,
            doc_type=doc_type,
            source_file=source_file,
        )
        tracker.stop("dense")

        # ----------------------------------------------------------
        # Step 4 — RRF fusion
        # ----------------------------------------------------------
        tracker.start("rrf")
        fused = rrf_fuse(
            bm25_docs, dense_docs,
            k=self.rrf_k,
            bm25_weight=self.rrf_bm25_w,
            dense_weight=self.rrf_dense_w,
        )
        tracker.stop("rrf")

        # ----------------------------------------------------------
        # Step 5 — Merge (deduplicate + fetch parent docs from MongoDB)
        # Uses RRF-fused ordering so parent lookup preserves relevance rank
        # ----------------------------------------------------------
        tracker.start("merge")
        merged_docs = merge_docs(fused, [])
        tracker.stop("merge")

        if not merged_docs:
            return {
                "answer":         "No relevant documents found in the knowledge base.",
                "cite_pages":     [],
                "cite_sources":   [],
                "cite_chapters":  [],
                "related_images": [],
                "latency_ms":     tracker.get_report(),
                "retrieved_docs": [],
            }

        # ----------------------------------------------------------
        # Step 6 — Reranking
        # ----------------------------------------------------------
        tracker.start("rerank")
        ranked_docs = self.reranker.rank(query, merged_docs, top_k=self.rerank_topk)
        tracker.stop("rerank")

        # ----------------------------------------------------------
        # Step 7 — Format context for LLM prompt
        # ----------------------------------------------------------
        context = format_context(ranked_docs)

        # ----------------------------------------------------------
        # Step 8 — LLM generation
        # ----------------------------------------------------------
        tracker.start("llm")
        # Trim history to sliding window
        history_window = self._chat_history[-(self.history_window * 2):]

        if stream:
            # Streaming: return raw completion iterator + partial result
            completion = request_chat(
                query, context,
                stream=True,
                chat_history=history_window,
            )
            # Collect streamed content
            raw_response = ""
            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    raw_response += delta
        else:
            raw_response = request_chat(
                query, context,
                stream=False,
                chat_history=history_window,
            )
        tracker.stop("llm")

        # ----------------------------------------------------------
        # Step 9 — Post-processing: extract citations
        # ----------------------------------------------------------
        tracker.start("postproc")
        result = post_processing(raw_response, ranked_docs)
        tracker.stop("postproc")

        # ----------------------------------------------------------
        # Step 10 — Update multi-turn chat history
        # ----------------------------------------------------------
        self._chat_history.append({"role": "user", "content": query})
        self._chat_history.append({"role": "assistant", "content": result["answer"]})

        result["latency_ms"]     = tracker.get_report()
        result["retrieved_docs"] = ranked_docs

        return result

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        """Clear multi-turn conversation history."""
        self._chat_history = []

    def get_history(self) -> List[dict]:
        """Return current conversation history."""
        return list(self._chat_history)


# =============================================================================
# CLI entry point for quick testing
# =============================================================================

if __name__ == "__main__":
    pipeline = RAGPipeline()

    test_queries = [
        "What signals are required for T cell activation?",
        "How do B cells differentiate into plasma cells?",
        "What is the role of MHC class II in antigen presentation?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("="*60)
        result = pipeline.answer(q)
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Pages cited: {result['cite_pages']}")
        print(f"Sources: {result['cite_sources']}")
        print(f"Latency: {result['latency_ms']}")
