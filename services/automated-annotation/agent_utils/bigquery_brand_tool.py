"""
Brand Search Tool (Postgres-first, BigQuery fallback)
=====================================================

Searches brand_knowledge_display table for matching brands.
Used by Agent 1 in brand classification workflow.

Tries Cloud SQL Postgres first for speed, falls back to BigQuery on failure.
"""

import logging
import os
import json
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
BIGQUERY_TABLE = 'brand_knowledge_display'

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


def reset_bigquery_client():
    """Reset the singleton (useful for testing or credential refresh)."""
    global _bigquery_client_singleton
    _bigquery_client_singleton = None


def _search_postgres(normalized_terms: List[str], verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Try brand search via Postgres. Returns None if unavailable/fails."""
    try:
        from core.utils.postgres_client import query as pg_query, _get_api_schema
    except ImportError:
        return None

    schema = _get_api_schema()

    # Build LIKE conditions with parameterized patterns
    like_conditions = " OR ".join([f"LOWER(brand) LIKE %s" for _ in normalized_terms])
    def _escape_like(t):
        return "%" + t.replace("%", "\\%").replace("_", "\\_") + "%"
    params = [_escape_like(term) for term in normalized_terms]

    sql = f"""
    SELECT DISTINCT id, brand
    FROM {schema}.brand_knowledge_display
    WHERE {like_conditions}
    ORDER BY brand
    """

    if verbose:
        print(f"[Postgres] Searching brands...")

    rows = pg_query(sql, params)
    if rows is None:
        return None

    matches = [{"id": int(r["id"]), "brand": r["brand"]} for r in rows]
    if verbose:
        print(f"[Postgres] Found {len(matches)} brand match(es)")

    return {
        "matches": matches,
        "search_terms_used": normalized_terms,
        "query_executed": None,
        "source": "postgres"
    }


def _search_bigquery(normalized_terms: List[str], verbose: bool = False) -> Dict[str, Any]:
    """Brand search via BigQuery (fallback)."""
    client = get_bigquery_client()

    escaped_terms = []
    for term in normalized_terms:
        escaped = term.replace('%', '\\%').replace('_', '\\_')
        escaped_terms.append(f"LOWER(brand) LIKE '%{escaped}%'")
    like_conditions = " OR ".join(escaped_terms)

    query = f"""
    SELECT DISTINCT id, brand
    FROM `{BIGQUERY_DATASET}.{BIGQUERY_TABLE}`
    WHERE {like_conditions}
    ORDER BY brand
    """

    if verbose:
        print(f"[BigQuery] Searching brands...")

    query_job = client.query(query)
    results = query_job.result()
    matches = [{"id": int(row.id), "brand": row.brand} for row in results]

    if verbose:
        print(f"[BigQuery] Found {len(matches)} brand match(es)")

    return {
        "matches": matches,
        "search_terms_used": normalized_terms,
        "query_executed": query if verbose else None,
        "source": "bigquery"
    }


def search_brand_database(search_terms: List[str], verbose: bool = False) -> Dict[str, Any]:
    """
    Search brand_knowledge_display for matching brands.

    Tries Postgres first, falls back to BigQuery on failure.
    Uses case-insensitive partial matching on the brand column.

    Args:
        search_terms: List of potential brand names/variations to search for
        verbose: Whether to print query details

    Returns:
        Dictionary with matches, search_terms_used, query_executed, source
    """
    normalized_terms = list(set([
        term.strip().lower()
        for term in search_terms
        if term.strip()
    ]))

    if not normalized_terms:
        return {"matches": [], "search_terms_used": [], "query_executed": None}

    if len(normalized_terms) > 30:
        normalized_terms = normalized_terms[:30]
        if verbose:
            print(f"WARNING: Limited search terms to 30 (had {len(search_terms)})")

    # Try Postgres first
    try:
        pg_result = _search_postgres(normalized_terms, verbose=verbose)
        if pg_result is not None:
            return pg_result
    except Exception as e:
        logger.warning(f"[Brand] Postgres failed, falling back to BigQuery: {e}")
        if verbose:
            print(f"[Postgres] Failed: {e}, falling back to BigQuery")

    # Fallback to BigQuery
    try:
        return _search_bigquery(normalized_terms, verbose=verbose)
    except Exception as e:
        raise Exception(f"Failed to search brand database: {e}")
