"""
BigQuery Brand Search Tool
==========================

Tool for searching brand_knowledge table in BigQuery.
Used by Agent 1 in brand classification workflow.
"""

import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from google.cloud import bigquery

# Import centralized credentials management
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils.credentials import ensure_gcp_adc


# Configuration
BIGQUERY_DATASET = 'api'
BIGQUERY_TABLE = 'brand_knowledge_display'

# Singleton for BigQuery client (reused across calls)
_bigquery_client_singleton = None


def get_bigquery_client():
    """
    Get a BigQuery client using Application Default Credentials.
    Uses centralized credentials from truss-platform-secrets via ensure_gcp_adc().
    Uses singleton pattern to reuse client across calls.
    """
    global _bigquery_client_singleton

    if _bigquery_client_singleton is not None:
        return _bigquery_client_singleton

    try:
        # Ensure GCP Application Default Credentials are set up
        ensure_gcp_adc()

        # Get project from environment (set by ensure_gcp_adc)
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEXAI_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT or VERTEXAI_PROJECT must be set")

        # Create BigQuery client using ADC
        client = bigquery.Client(project=project_id)

        # Cache the client
        _bigquery_client_singleton = client

        return client
    except Exception as e:
        raise Exception(f"Failed to create BigQuery client: {e}")


def reset_bigquery_client():
    """Reset the singleton (useful for testing or credential refresh)."""
    global _bigquery_client_singleton
    _bigquery_client_singleton = None


def search_brand_database(search_terms: List[str], verbose: bool = False) -> Dict[str, Any]:
    """
    Search the brand_knowledge table for matching brands.
    
    Uses case-insensitive partial matching on the brand column.
    
    Args:
        search_terms: List of potential brand names/variations to search for
        verbose: Whether to print query details
        
    Returns:
        Dictionary with:
            - matches: List of {id, brand} dicts
            - search_terms_used: List of normalized terms actually searched
            - query_executed: SQL query string (if verbose)
    """
    # Normalize and dedupe search terms
    normalized_terms = list(set([
        term.strip().lower() 
        for term in search_terms 
        if term.strip()
    ]))
    
    if not normalized_terms:
        return {
            "matches": [],
            "search_terms_used": [],
            "query_executed": None
        }
    
    # Limit to 30 terms to avoid huge queries
    if len(normalized_terms) > 30:
        normalized_terms = normalized_terms[:30]
        if verbose:
            print(f"WARNING: Limited search terms to 30 (had {len(search_terms)})")
    
    try:
        client = get_bigquery_client()
        
        # Build query with LIKE conditions
        # Escape special characters for LIKE
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
            print(f"Executing BigQuery query...")
            print(f"Search terms: {normalized_terms[:5]}{'...' if len(normalized_terms) > 5 else ''}")
        
        # Execute query
        query_job = client.query(query)
        results = query_job.result()
        
        matches = [{"id": int(row.id), "brand": row.brand} for row in results]
        
        if verbose:
            print(f"Found {len(matches)} brand match(es)")
        
        return {
            "matches": matches,
            "search_terms_used": normalized_terms,
            "query_executed": query if verbose else None
        }
        
    except Exception as e:
        raise Exception(f"Failed to search brand database: {e}")
