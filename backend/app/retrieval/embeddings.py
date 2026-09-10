"""Gemini query embedding for live retrieval."""

from __future__ import annotations

import math

from google import genai
from google.genai import types

from app.config import settings


def _client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def _normalize(embedding: list[float]) -> list[float]:
    """L2-normalize the embedding.

    gemini-embedding-001 only returns pre-normalized vectors at its native
    3072 dimensions. Any smaller `output_dimensionality` (we request 1536,
    to match the existing pgvector(1536) schema) comes back un-normalized,
    so cosine/inner-product search would be skewed without this step.
    """
    norm = math.sqrt(sum(value * value for value in embedding))
    if norm == 0:
        return embedding
    return [value / norm for value in embedding]


def embed_query(text: str) -> list[float]:
    client = _client()
    response = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=settings.gemini_embedding_dimensions,
        ),
    )
    embedding = response.embeddings[0].values
    expected_dims = settings.gemini_embedding_dimensions
    if len(embedding) != expected_dims:
        raise ValueError(
            f"Expected embedding dimension {expected_dims}, got {len(embedding)}"
        )
    return _normalize(embedding)
