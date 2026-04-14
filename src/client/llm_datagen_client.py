# =============================================================================
# ImmunoBiology RAG — LLM Client for Training Data Generation
# =============================================================================
# Used by train/build_train_data.py for generating QA pairs.
# Uses the same local vLLM endpoint as the main LLM client.
# Higher temperature for diverse QA generation.

import time
import random
from openai import OpenAI
from src import constant


llm_client = OpenAI(
    api_key="EMPTY",
    base_url=constant.vllm_base_url,
)


def chat(
    prompt: str,
    max_retry: int = 3,
    temperature: float = 0.85,
    top_p: float = 0.95,
    debug: bool = False,
) -> str:
    """
    Generate text from a prompt using the local LLM.
    Retries up to max_retry times on failure.

    Args:
        prompt:      Full prompt string
        max_retry:   Number of retries on API error
        temperature: Sampling temperature (higher = more diverse)
        top_p:       Top-p nucleus sampling parameter
        debug:       Print retry info if True

    Returns:
        Generated text string, or None if all retries failed.
    """
    def _do_chat():
        completion = llm_client.chat.completions.create(
            model=constant.llm_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful immunology AI assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message.content

    for attempt in range(max_retry):
        try:
            return _do_chat()
        except Exception as e:
            sleep_secs = random.randint(1, 4)
            if debug:
                print(f"[DatagenLLM] Attempt {attempt+1}/{max_retry} failed: {e}. "
                      f"Retrying in {sleep_secs}s...")
            time.sleep(sleep_secs)
    return None
