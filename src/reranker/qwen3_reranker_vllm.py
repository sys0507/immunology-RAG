# =============================================================================
# ImmunoBiology RAG — Qwen3 vLLM Reranker (Optional Alternative)
# =============================================================================
# Copied verbatim from Tesla RAG: src/reranker/qwen3_reranker_vllm.py
# Changes: English comments only. Use as alternative to BGE cross-encoder
# when higher reranking quality is needed (at the cost of higher latency).
#
# Activation: set config.yaml reranker model to "models/Qwen3-Reranker-4B"

import gc
import math
import torch
from typing import List
from langchain_core.documents import Document
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from sentence_transformers import CrossEncoder
from vllm.inputs.data import TokensPrompt

from src.reranker.reranker import RerankerBase
from src import constant


class Qwen3ReRankervLLM(RerankerBase):
    """
    LLM-based reranker using Qwen3-Reranker-4B via vLLM batch inference.

    How it works:
    - For each (query, doc) pair, asks the LLM: "Is this document relevant? yes/no"
    - Computes P(yes) / (P(yes) + P(no)) from logprobs
    - Returns top-k documents sorted by yes-probability

    More powerful than cross-encoder but requires more VRAM and is slower.
    Recommended for final evaluation or when top-precision is needed.
    """

    def __init__(
        self,
        model_path: str = None,
        instruction: str = "Given the user query, retrieve the relevant passages",
        **kwargs,
    ):
        model_path = model_path or constant.bge_reranker_model_path
        super().__init__(model_path)

        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_length = kwargs.get("max_length", 8192)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.true_token  = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no",  add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )
        n_gpus = torch.cuda.device_count()
        self.lm = LLM(
            model=model_path,
            tensor_parallel_size=n_gpus,
            max_model_len=10000,
            enable_prefix_caching=True,
            distributed_executor_backend="ray",
            gpu_memory_utilization=0.8,
        )

    def _format_instruction(self, instruction: str, query: str, doc: str) -> list:
        """Format a single (query, doc) pair as a chat message."""
        return [
            {
                "role": "system",
                "content": (
                    "Judge whether the Document meets the requirements based on "
                    "the Query and the Instruct provided. "
                    "Note that the answer can only be \"yes\" or \"no\"."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<Instruct>: {instruction}\n\n"
                    f"<Query>: {query}\n\n"
                    f"<Document>: {doc}"
                ),
            },
        ]

    def rank(
        self,
        query: str,
        candidate_docs: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """Score each (query, doc) pair and return top_k by yes-probability."""
        messages = [
            self._format_instruction(self.instruction, query, doc.page_content)
            for doc in candidate_docs
        ]
        tokenized = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        tokenized = [t[:self.max_length] + self.suffix_tokens for t in tokenized]
        prompts   = [TokensPrompt(prompt_token_ids=t) for t in tokenized]
        outputs   = self.lm.generate(prompts, self.sampling_params, use_tqdm=False)

        scores = []
        for out in outputs:
            logits = out.outputs[0].logprobs[-1]
            true_logit  = logits[self.true_token].logprob  if self.true_token  in logits else -10
            false_logit = logits[self.false_token].logprob if self.false_token in logits else -10
            true_score  = math.exp(true_logit)
            false_score = math.exp(false_logit)
            scores.append(true_score / (true_score + false_score))

        ranked = sorted(
            zip(scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]
        return [doc for _score, doc in ranked]

    def stop(self) -> None:
        """Release vLLM GPU resources."""
        destroy_model_parallel()
