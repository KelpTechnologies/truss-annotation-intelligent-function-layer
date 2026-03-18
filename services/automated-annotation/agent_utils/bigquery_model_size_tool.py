"""
Model Size Search Tool (Postgres-first, BigQuery fallback)
==========================================================

Searches model_size_knowledge_display table for size options.
Used by model size classification workflow.

Tries Cloud SQL Postgres first for speed, falls back to BigQuery on failure.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import pandas as pd

# Import centralized credentials management
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils.credentials import ensure_gcp_adc

logger = logging.getLogger(__name__)

# Configuration
BIGQUERY_DATASET = 'api'
BIGQUERY_TABLE = 'model_size_knowledge_display'

# Singleton for BigQuery client (reused across calls)
_bigquery_client_singleton = None


def get_bigquery_client():
    """Get a BigQuery client using ADC. Singleton."""
    global _bigquery_client_singleton
    if _bigquery_client_singleton is not None:
        return _bigquery_client_singleton
    try:
        from google.cloud import bigquery
        ensure_gcp_adc()
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEXAI_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT or VERTEXAI_PROJECT must be set")
        _bigquery_client_singleton = bigquery.Client(project=project_id)
        return _bigquery_client_singleton
    except Exception as e:
        raise Exception(f"Failed to create BigQuery client: {e}")


def _get_size_options_postgres(model_id: int, verbose: bool = False) -> Optional[List[Dict[str, Any]]]:
    """Try size options lookup via Postgres. Returns None if unavailable."""
    try:
        from core.utils.postgres_client import query as pg_query, _get_api_schema
    except ImportError:
        return None

    schema = _get_api_schema()
    # pg8000 uses %s placeholders
    # Note: BQ table has 'depth' column, Postgres schema has 'depth' too
    sql = f"""
    SELECT id, model_id, size, height, width, depth
    FROM {schema}.model_size_knowledge_display
    WHERE model_id = %s
    ORDER BY size
    """

    if verbose:
        print(f"[Postgres] Querying size options for model_id={model_id}...")

    rows = pg_query(sql, [model_id])
    if rows is None:
        return None

    options = []
    for r in rows:
        options.append({
            "id": int(r["id"]),
            "model_id": int(r["model_id"]) if r.get("model_id") else None,
            "size": r.get("size"),
            "height": float(r["height"]) if r.get("height") is not None else None,
            "width": float(r["width"]) if r.get("width") is not None else None,
            "length": float(r["depth"]) if r.get("depth") is not None else None,
        })

    if verbose:
        print(f"[Postgres] Found {len(options)} model size option(s)")

    return options


def _get_size_options_bigquery(model_id: int, verbose: bool = False) -> List[Dict[str, Any]]:
    """Size options via BigQuery (fallback)."""
    client = get_bigquery_client()

    query = f"""
    SELECT id, model_id, size, height, width, length
    FROM `{BIGQUERY_DATASET}.{BIGQUERY_TABLE}`
    WHERE model_id = {model_id}
    ORDER BY size
    """

    if verbose:
        print(f"[BigQuery] Querying size options for model_id={model_id}...")

    query_job = client.query(query)
    results = query_job.result()

    options = []
    for row in results:
        options.append({
            "id": int(row.id),
            "model_id": int(row.model_id) if row.model_id else None,
            "size": row.size,
            "height": float(row.height) if row.height else None,
            "width": float(row.width) if row.width else None,
            "length": float(row.length) if row.length else None,
        })

    if verbose:
        print(f"[BigQuery] Found {len(options)} model size option(s)")

    return options


def get_model_size_options(model_id: int, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Retrieve model size options for a specific model.

    Tries Postgres first, falls back to BigQuery on failure.

    Args:
        model_id: Required model_id to filter by
        verbose: Whether to print query details

    Returns:
        List of dicts with: id, model_id, size, height, width, length (cm)
    """
    # Try Postgres first
    try:
        pg_result = _get_size_options_postgres(model_id, verbose=verbose)
        if pg_result is not None:
            return pg_result
    except Exception as e:
        logger.warning(f"[ModelSize] Postgres failed, falling back to BigQuery: {e}")
        if verbose:
            print(f"[Postgres] Failed: {e}, falling back to BigQuery")

    # Fallback to BigQuery
    try:
        return _get_size_options_bigquery(model_id, verbose=verbose)
    except Exception as e:
        raise Exception(f"Failed to query model size options: {e}")
