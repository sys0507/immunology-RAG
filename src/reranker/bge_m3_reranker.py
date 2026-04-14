# =============================================================================
# ImmunoBiology RAG — BGE-M3 Cross-Encoder Reranker
# =============================================================================
# Adapted from Tesla RAG: src/reranker/bge_m3_reranker.py
# Changes: English comments; model default → bge-reranker-v2-m3
# Logic: identical — cross-encoder scores (query, doc) pairs via logits.

import torch
import warnings
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.reranker.reranker import RerankerBase
from src import constant


class BGEM3ReRanker(RerankerBase):
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    How it works:
    - Concatenates (query, doc) as a single input
    - Runs through a classification head → logit score
    - Higher logit = more relevant
    - Returns top-k documents sorted by score descending

    This is more accurate than dual-encoder retrieval but slower,
    so it is only applied to the top-N candidates from hybrid retrieval.
    """

    def __init__(self, model_path: str = None, max_length: int = 4096):
        if model_path is None:
            # config.yaml: reranker_use_finetuned controls which model is loaded.
            # false (default) → base BAAI/bge-reranker-v2-m3 from HF cache
            # true            → outputs/models/reranker_finetuned/best/ (local fine-tuned)
            if constant.reranker_use_finetuned:
                best_path = Path(constant.bge_reranker_tuned_path) / "best"
                if best_path.exists():
                    model_path = str(best_path)
                else:
                    print("[Reranker] ⚠ reranker_use_finetuned=true but fine-tuned model not found "
                          f"at {best_path} — falling back to base model.")
                    model_path = constant.bge_reranker_model_path
            else:
                model_path = constant.bge_reranker_model_path
        super().__init__(model_path, max_length)

        print(f"[Reranker] Loading BGE reranker from: {model_path}")
        # Suppress false-positive "incorrect regex pattern" warning:
        # BGE uses a Metaspace pre-tokenizer (XLM-RoBERTa), not Mistral's Sequence/Split.
        # fix_mistral_regex=True crashes on Metaspace, so we suppress the warning instead.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
            # Tokenizer always loads from the base model — LoRA adapters don't change vocab.
            _tok_path = (constant.bge_reranker_model_path
                         if (Path(model_path) / "adapter_config.json").exists()
                         else model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(_tok_path)

        # ── LoRA-aware model loading ─────────────────────────────────────────
        # If model_path contains adapter_config.json it is a LoRA adapter (v3+).
        # We must load the base model first, then apply the adapter on top.
        # Legacy full fine-tuned paths (v1/v2) and the base model path are loaded
        # directly with AutoModelForSequenceClassification.
        _adapter_cfg = Path(model_path) / "adapter_config.json"
        if _adapter_cfg.exists():
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    "[Reranker] LoRA adapter found but 'peft' is not installed. "
                    "Run: pip install peft"
                )
            print(f"[Reranker] Detected LoRA adapter — loading base model + adapter.")
            _base = AutoModelForSequenceClassification.from_pretrained(
                constant.bge_reranker_model_path
            )
            self.model = PeftModel.from_pretrained(_base, model_path)
            print(f"[Reranker] LoRA adapter loaded (base: {constant.bge_reranker_model_path})")
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model.eval()
        self.model.half()   # fp16 for efficiency
        self.model.cuda()
        print("[Reranker] BGE reranker loaded.")

    def rank(
        self,
        query: str,
        candidate_docs: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """
        Score each (query, doc) pair and return top_k by score.

        Args:
            query:          User query string
            candidate_docs: List of retrieved Documents (typically 10-20)
            top_k:          Number to return after reranking (default 5)

        Returns:
            Top-k Documents sorted by cross-encoder score descending
        """
        if not candidate_docs:
            return []

        pairs = [(query, doc.page_content) for doc in candidate_docs]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to("cuda")

        with torch.no_grad():
            scores = self.model(**inputs).logits

        scores = scores.detach().cpu().float().numpy().flatten()

        # Free GPU tensors immediately — inputs can be large (batch × max_length)
        del inputs
        torch.cuda.empty_cache()

        ranked = sorted(
            zip(scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return [doc for _score, doc in ranked]


if __name__ == "__main__":
    reranker = BGEM3ReRanker()
    query = "What is the role of T helper cells in adaptive immunity?"
    docs = [
        Document(page_content="T helper cells coordinate adaptive immune responses.", metadata={}),
        Document(page_content="B cells produce antibodies in response to antigens.", metadata={}),
        Document(page_content="Natural killer cells are part of innate immunity.", metadata={}),
    ]
    ranked = reranker.rank(query, docs, top_k=2)
    for doc in ranked:
        print(doc.page_content)
