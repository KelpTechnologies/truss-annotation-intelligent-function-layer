"""Minimal Pinecone utility helpers.

These utilities mirror the behaviour of the JavaScript helpers used by other
services. They provide:

1. Lazy initialisation of the Pinecone client using environment variables
2. Cached index lookups (default and custom)
3. Helpers for querying similar vectors and fetching vectors by id
4. Automatic retry logic for transient Pinecone errors

Environment variables used:
- ``PINECONE_API_KEY`` (required)
- ``PINECONE_ENVIRONMENT`` (optional)
- ``PINECONE_INDEX_NAME`` (optional, default: ``mfc-classifier-bags-models``)
- ``PINECONE_DEFAULT_NAMESPACE`` (optional, default: ``jacquemus``)
- ``PINECONE_DEFAULT_TOP_K`` (optional, default: ``200``)
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional


DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mfc-classifier-bags-models")
DEFAULT_NAMESPACE = os.getenv("PINECONE_DEFAULT_NAMESPACE", "jacquemus")
DEFAULT_TOP_K = int(os.getenv("PINECONE_DEFAULT_TOP_K", "200"))
MAX_RETRIES = int(os.getenv("PINECONE_MAX_RETRIES", "3"))


_pinecone_client = None
_default_index = None


def _import_pinecone_client_class():
    """Import the Pinecone client class, raising an informative error if missing."""

    try:
        from pinecone import Pinecone
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "The 'pinecone' Python package is required. Install 'pinecone-client' or "
            "add a Lambda layer that provides the Pinecone SDK."
        ) from exc

    return Pinecone


def get_pinecone_client():
    """Return a singleton Pinecone client instance."""

    global _pinecone_client

    if _pinecone_client:
        return _pinecone_client

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    Pinecone = _import_pinecone_client_class()

    environment = os.getenv("PINECONE_ENVIRONMENT")
    if environment:
        _pinecone_client = Pinecone(api_key=api_key, environment=environment)
    else:
        _pinecone_client = Pinecone(api_key=api_key)

    return _pinecone_client


def get_index(index_name: Optional[str] = None):
    """Return a Pinecone index instance, caching the default index."""

    global _default_index

    resolved_name = index_name or DEFAULT_INDEX_NAME

    if resolved_name == DEFAULT_INDEX_NAME and _default_index:
        return _default_index

    client = get_pinecone_client()
    index = client.Index(resolved_name)

    if resolved_name == DEFAULT_INDEX_NAME:
        _default_index = index

    return index


def fetch_vectors(
    ids: Iterable[str],
    *,
    namespace: Optional[str] = None,
    index_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch vectors by id from Pinecone.

    Args:
        ids: Iterable of vector ids
        namespace: Pinecone namespace to use (defaults to ``DEFAULT_NAMESPACE``)
        index_name: Pinecone index name (defaults to ``DEFAULT_INDEX_NAME``)

    Returns:
        Mapping of id -> vector record.
    """

    id_list = list(ids)
    if not id_list:
        return {}

    index = get_index(index_name)
    ns = namespace or DEFAULT_NAMESPACE or None

    response = execute_with_retry(
        lambda: index.fetch(ids=id_list, namespace=ns)
    )
    return response.get("records", {}) if isinstance(response, dict) else response.records


def query_similar_vectors(
    *,
    vector: Optional[List[float]] = None,
    vector_id: Optional[str] = None,
    top_k: Optional[int] = None,
    namespace: Optional[str] = None,
    index_name: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True,
    include_values: bool = False,
) -> List[Dict[str, Any]]:
    """Query Pinecone for similar vectors."""

    if vector_id is None and (not vector or not isinstance(vector, (list, tuple))):
        raise ValueError("Either 'vector' or 'vector_id' must be provided for a Pinecone query")

    index = get_index(index_name)
    ns = namespace or DEFAULT_NAMESPACE or None

    payload: Dict[str, Any] = {
        "top_k": top_k or DEFAULT_TOP_K,
        "include_metadata": include_metadata,
        "include_values": include_values,
    }

    if filter:
        payload["filter"] = filter

    if vector_id:
        payload["id"] = vector_id
    else:
        payload["vector"] = vector

    if ns:
        payload["namespace"] = ns

    response = execute_with_retry(lambda: index.query(**payload))

    # The Python client returns an object with a ``matches`` attribute. Support both dict/object.
    if isinstance(response, dict):
        return response.get("matches", [])

    return getattr(response, "matches", [])


def execute_with_retry(operation: Callable[[], Any], attempt: int = 1) -> Any:
    """Execute a Pinecone operation with simple exponential backoff."""

    try:
        return operation()
    except Exception as error:  # noqa: broad-except - Pinecone raises many different error types
        if should_retry(error) and attempt < MAX_RETRIES:
            delay = (2 ** attempt) * 0.1
            time.sleep(delay)
            return execute_with_retry(operation, attempt + 1)
        raise


def should_retry(error: Exception) -> bool:
    """Determine whether an error warrants a retry."""

    retryable_codes = {"RateLimitExceeded", "ServiceUnavailable", "InternalServerError"}

    code = getattr(error, "code", None)
    if code and code in retryable_codes:
        return True

    status_code = getattr(error, "status_code", None)
    if status_code and int(status_code) in {408, 429, 500, 502, 503, 504}:
        return True

    message = str(getattr(error, "message", "") or str(error)).lower()
    if "rate limit" in message or "timeout" in message or "temporarily" in message:
        return True

    return False


__all__ = [
    "DEFAULT_INDEX_NAME",
    "DEFAULT_NAMESPACE",
    "DEFAULT_TOP_K",
    "MAX_RETRIES",
    "get_pinecone_client",
    "get_index",
    "fetch_vectors",
    "query_similar_vectors",
]


