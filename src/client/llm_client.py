# =============================================================================
# ImmunoBiology RAG — Local LLM Inference Client
# =============================================================================
# Adapted from Tesla RAG: src/client/llm_local_client.py
# Changes:
#   - English LLM_CHAT_PROMPT (immunology domain)
#   - Citation format: [1], [2] instead of Chinese 【1】【2】
#   - Model: Qwen/Qwen3-8B (via vLLM) — mirrors Tesla RAG model choice
#   - enable_thinking=False passed via extra_body to suppress Qwen3 <think> tokens
#   - Multi-turn conversation support via chat_history parameter
#   - Reads model name and vLLM URL from constant.py

import os
from typing import Iterator, List, Optional, Union
from openai import OpenAI
from src import constant


# ---------------------------------------------------------------------------
# System prompt: English, immunology domain
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert immunology assistant. "
    "Answer questions using ONLY the provided numbered context passages. "
    "Cite relevant passages inline using [N] notation (e.g., [1], [2,3]). "
    "If the provided context does not contain sufficient information, respond with: "
    "'The knowledge base does not contain enough information to answer this question.' "
    "Never fabricate information. Format answers in clear academic English."
)

# ---------------------------------------------------------------------------
# User message prompt template
# ---------------------------------------------------------------------------
LLM_CHAT_PROMPT = """### Context
{context}

### Question
{query}

Please answer the question based strictly on the context above. Cite passages with [N] notation.
Answer:"""


# ---------------------------------------------------------------------------
# OpenAI-compatible client pointing to local vLLM server
# ---------------------------------------------------------------------------
llm_client = OpenAI(
    api_key="EMPTY",  # vLLM does not require a real key
    base_url=constant.vllm_base_url,
)


def request_chat(
    query: str,
    context: str,
    stream: bool = False,
    chat_history: Optional[List[dict]] = None,
    model_name: Optional[str] = None,
) -> Union[str, Iterator]:
    """
    Generate an answer using the local LLM via vLLM's OpenAI-compatible API.

    Args:
        query:        User question
        context:      Formatted retrieved context (numbered passages [1]...[N])
        stream:       If True, returns a streaming iterator; otherwise full response
        chat_history: Optional list of {"role": ..., "content": ...} dicts
                      for multi-turn conversation (up to history_window turns)
        model_name:   Override model; defaults to config value

    Returns:
        str (stream=False) or streaming completion iterator (stream=True)
    """
    model = model_name or constant.llm_model_name
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (multi-turn)
    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": prompt})

    completion = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=constant.llm_max_tokens,
        temperature=constant.llm_temperature,
        top_p=constant.llm_top_p,
        frequency_penalty=constant.llm_freq_penalty,
        stream=stream,
        extra_body={
            "top_k": 1,
            # Qwen3 requires enable_thinking=False to suppress <think>...</think>
            # chain-of-thought tokens in the output. Mirrors Tesla llm_local_client.py.
            "chat_template_kwargs": {"enable_thinking": constant.enable_thinking},
        },
    )

    if not stream:
        return completion.choices[0].message.content
    return completion


if __name__ == "__main__":
    context = (
        "[1] T cell activation requires recognition of peptide-MHC complexes "
        "by the T cell receptor (TCR), combined with co-stimulatory signals.\n"
        "[2] Without co-stimulation, T cells may become anergic rather than activated."
    )
    query = "What signals are required for T cell activation?"
    result = request_chat(query, context, stream=True)
    for chunk in result:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()
