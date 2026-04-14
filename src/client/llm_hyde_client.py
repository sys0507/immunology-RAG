# =============================================================================
# ImmunoBiology RAG — HyDE Query Expansion Client
# =============================================================================
# Adapted from Tesla RAG: src/client/llm_hyde_client.py
# Changes:
#   - Replaced Doubao API with same local vLLM (no separate API key needed)
#   - English HyDE prompt for immunology domain
#   - Concise output (~100 words) for efficient retrieval expansion
#
# HyDE (Hypothetical Document Embeddings):
#   Query: "What activates T cells?"
#   HyDE output: "T cell activation requires TCR recognition of peptide-MHC..."
#   Combined: original query + HyDE text → better retrieval recall

from openai import OpenAI
from src import constant


LLM_HYDE_PROMPT = """You are an expert immunologist. Given the following question,
generate a concise hypothetical passage (max 100 words) that would be found in an
immunology textbook and would directly answer the question. Focus on relevant
technical details, molecular mechanisms, and immunological terminology.

Question: {query}

Hypothetical passage:"""


# Reuse the same vLLM endpoint as the main LLM client
llm_client = OpenAI(
    api_key="EMPTY",
    base_url=constant.vllm_base_url,
)


def request_hyde(query: str) -> str:
    """
    Generate a hypothetical document for HyDE query expansion.

    Args:
        query: Original user query

    Returns:
        A short hypothetical passage (~100 words) for augmenting the query
        before retrieval. Combined with the original query, this improves
        recall for academic questions where the query and answer use different
        vocabulary.
    """
    prompt = LLM_HYDE_PROMPT.format(query=query)

    completion = llm_client.chat.completions.create(
        model=constant.llm_model_name,
        messages=[
            {"role": "system", "content": "You are an expert immunology assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.3,   # Slightly higher than QA for more diverse expansion
        top_p=0.95,
    )
    return completion.choices[0].message.content.strip()


if __name__ == "__main__":
    query = "How do cytotoxic T lymphocytes kill target cells?"
    hyde_text = request_hyde(query)
    print(f"Original query: {query}")
    print(f"HyDE expansion: {hyde_text}")
    print(f"\nCombined query for retrieval:\n{query}\n{hyde_text}")
