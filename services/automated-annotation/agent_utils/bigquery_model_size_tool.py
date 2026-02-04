"""
BigQuery Model Size Search Tool
================================

Tool for searching model_size_knowledge_display table in BigQuery.
Used by model size classification workflow.
"""

import os
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
BIGQUERY_TABLE = 'model_size_knowledge_display'

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


def get_model_size_options(model_id: int, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Retrieve model size options from BigQuery for a specific model.
    
    Args:
        model_id: Required model_id to filter by (size options are model-specific)
        verbose: Whether to print query details
        
    Returns:
        List of dicts with: id, model_id, size, height, width, length
        All measurements are in cm
    """
    try:
        client = get_bigquery_client()
        
        query = f"""
        SELECT id, model_id, size, height, width, length
        FROM `{BIGQUERY_DATASET}.{BIGQUERY_TABLE}`
        WHERE model_id = {model_id}
        ORDER BY size
        """
        
        if verbose:
            print(f"Executing BigQuery query...")
            print(f"Filtering by model_id: {model_id}")
        
        # Execute query
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
                "length": float(row.length) if row.length else None
            })
        
        if verbose:
            print(f"Found {len(options)} model size option(s)")
        
        return options
        
    except Exception as e:
        raise Exception(f"Failed to query model size options: {e}")
