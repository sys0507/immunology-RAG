# =============================================================================
# ImmunoBiology RAG — Semantic Chunking Service Client
# =============================================================================
# Adapted from Tesla RAG: src/client/semantic_chunk_client.py
# Changes: English comments; reads service URL from constant.py

import json
import requests
from src import constant


def request_semantic_chunk(sentences: str, group_size: int = None) -> list[str]:
    """
    Call the local semantic chunking FastAPI service.

    Args:
        sentences:  Raw text to split (a full page or section of text)
        group_size: Target maximum number of paragraphs per group.
                    Defaults to config value (15 for English academic text).

    Returns:
        List of semantically grouped text chunks.
        Falls back to [sentences] (unsplit) if the service is unavailable.
    """
    if group_size is None:
        group_size = constant.semantic_group_size

    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"sentences": sentences, "group_size": group_size})

    try:
        response = requests.post(
            constant.semantic_service_url,
            headers=headers,
            data=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["chunks"]
    except requests.exceptions.ConnectionError:
        print(
            f"[SemanticChunkClient] WARNING: Service unavailable at "
            f"{constant.semantic_service_url}. "
            "Returning unsplit text. Start src/server/semantic_chunk.py first."
        )
        return [sentences]
    except Exception as e:
        print(f"[SemanticChunkClient] Request failed: {e}. Returning unsplit text.")
        return [sentences]
