# =============================================================================
# ImmunoBiology RAG — Pydantic model for extracted figure metadata
# =============================================================================
# Adapted from Tesla RAG: src/fields/manual_images.py
# Changes: renamed ManualImages → ImmunoImages; English field descriptions;
#          added has_caption flag for figure caption detection.

from typing import Optional
from pydantic import BaseModel, Field


class ImmunoImages(BaseModel):
    """Metadata for a figure extracted from an immunology textbook or paper."""

    page: Optional[int] = Field(
        ge=1,
        description="1-indexed page number where the figure appears"
    )
    image_path: Optional[str] = Field(
        min_length=1,
        description="Local file path to the saved figure PNG"
    )
    title: Optional[str] = Field(
        default=None,
        description="Figure caption text; multiple blocks joined with newline"
    )
    has_caption: Optional[bool] = Field(
        default=False,
        description="True if a 'Figure X.X' caption was detected near this image"
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Source PDF filename this figure was extracted from"
    )
