# =============================================================================
# ImmunoBiology RAG — Pydantic model for MongoDB chunk document
# =============================================================================
# Adapted from Tesla RAG: src/fields/manual_info_mongo.py
# Changes: renamed ManualInfo → ChunkInfo; English descriptions;
#          added immunology-specific metadata fields.

from typing import Optional
from pydantic import BaseModel, Field


class ChunkInfo(BaseModel):
    """
    Represents a single chunk (parent or child) stored in MongoDB.

    Design: mirrors Tesla's ManualInfo but adds multi-document metadata:
    source_file, doc_type, chapter, chunk_id.
    """

    unique_id: str = Field(
        description="MD5 hash of page_content — used as MongoDB primary key"
    )
    metadata: dict = Field(
        description=(
            "Document metadata dict containing: unique_id, parent_id, page, "
            "source_file, doc_type, chapter, chunk_id, images_info, has_figure_caption"
        )
    )
    page_content: Optional[str] = Field(
        default=None,
        description="Text content of this chunk"
    )
