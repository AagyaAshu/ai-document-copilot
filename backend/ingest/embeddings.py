"""Gemini embedding generation for document chunks."""

from __future__ import annotations

import math
import time

from google import genai
from google.genai import errors, types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings


def _log_retry(retry_state) -> None:
    wait_time = getattr(retry_state.next_action, "sleep", 0)
    print(
        f"  Rate limited, waiting {wait_time:.0f}s before retry "
        f"(attempt {retry_state.attempt_number})...",
        flush=True,
    )

# Free-tier embed_content has THREE separate caps: requests/minute,
# requests/day, and tokens/minute (TPM = 30,000 on the free tier as of
# Aug 2026). Chunks run up to CHUNK_MAX_TOKENS (512) each, so a
# batch_size=100 request can be ~51K tokens in one call — well over the
# 30K/minute cap, and gets rejected before even touching the per-minute
# request count. Keep batches small enough that one batch's worst-case
# token count (batch_size * 512) stays under budget, and pace them so
# consecutive batches don't stack up past 30K within the same 60s window.
EMBED_BATCH_SIZE = 25
_BATCH_PAUSE_SECONDS = 20.0


def _client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def _normalize(embedding: list[float]) -> list[float]:
    """L2-normalize the embedding.

    gemini-embedding-001 only returns pre-normalized vectors at its native
    3072 dimensions. Any smaller `output_dimensionality` (we request 1536,
    to match the existing pgvector(1536) schema) comes back un-normalized,
    so cosine/inner-product search would be skewed without this step. Must
    match app/retrieval/embeddings.py exactly, since query and document
    vectors have to live in the same normalized space.
    """
    norm = math.sqrt(sum(value * value for value in embedding))
    if norm == 0:
        return embedding
    return [value / norm for value in embedding]


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, errors.ClientError) and exc.code == 429


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=10, min=10, max=120),
    stop=stop_after_attempt(8),
    before_sleep=_log_retry,
    reraise=True,
)
def _embed_batch(client: genai.Client, batch: list[str], expected_dims: int) -> list[list[float]]:
    response = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=batch,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=expected_dims,
        ),
    )
    vectors: list[list[float]] = []
    for item in response.embeddings:
        embedding = item.values
        if len(embedding) != expected_dims:
            raise ValueError(
                f"Expected embedding dimension {expected_dims}, got {len(embedding)}"
            )
        vectors.append(_normalize(embedding))
    return vectors


def embed_texts(texts: list[str], *, batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    if not texts:
        return []

    expected_dims = settings.gemini_embedding_dimensions
    vectors: list[list[float]] = []
    client = _client()

    batch_starts = list(range(0, len(texts), batch_size))
    for i, start in enumerate(batch_starts):
        batch = texts[start : start + batch_size]
        vectors.extend(_embed_batch(client, batch, expected_dims))
        if i < len(batch_starts) - 1:
            time.sleep(_BATCH_PAUSE_SECONDS)

    return vectors
