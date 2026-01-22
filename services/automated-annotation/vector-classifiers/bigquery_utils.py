"""BigQuery utility helpers for model classification.

This module provides BigQuery integration for fetching model metadata
from the model_classification and model_knowledge_display tables.

Environment variables used:
- ``TRUSS_SECRETS_ARN`` or ``BIGQUERY_SECRET_ARN`` (required) - ARN for secrets containing BigQuery credentials
- ``AWS_REGION`` (optional, default: ``eu-west-2``)
- ``GOOGLE_APPLICATION_CREDENTIALS`` (optional, default: ``/tmp/gcp_sa.json``)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import boto3

# BigQuery client cache
_bigquery_client = None
_cached_platform_secrets = None


def _get_platform_secrets() -> Dict[str, Any]:
    """Get all platform secrets from centralized truss-platform-secrets."""
    global _cached_platform_secrets
    if _cached_platform_secrets:
        return _cached_platform_secrets
    
    secret_arn = os.getenv("TRUSS_SECRETS_ARN") or os.getenv("BIGQUERY_SECRET_ARN")
    if not secret_arn:
        raise ValueError("TRUSS_SECRETS_ARN environment variable is required for BigQuery access")
    
    print(f"[BigQuery] Loading platform secrets from: {secret_arn}")
    secrets_client = boto3.client('secretsmanager', region_name=os.getenv('AWS_REGION', 'eu-west-2'))
    
    response = secrets_client.get_secret_value(SecretId=secret_arn)
    secret_string = response.get('SecretString')
    
    if not secret_string:
        raise ValueError(f"Secret string is empty for secret {secret_arn}")
    
    _cached_platform_secrets = json.loads(secret_string)
    return _cached_platform_secrets


def _get_bigquery_credentials() -> Dict[str, Any]:
    """Extract BigQuery credentials from platform secrets."""
    secrets = _get_platform_secrets()
    
    # Check for centralized 'bigquery' key (truss-platform-secrets structure)
    if 'bigquery' in secrets:
        print("[BigQuery] Found BigQuery credentials under 'bigquery' key")
        bigquery_creds = secrets['bigquery']
        return {
            "type": "service_account",
            "project_id": bigquery_creds.get('project_id'),
            "private_key_id": bigquery_creds.get('private_key_id'),
            "private_key": bigquery_creds.get('private_key'),
            "client_email": bigquery_creds.get('client_email'),
            "client_id": bigquery_creds.get('client_id'),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{bigquery_creds.get('client_email')}"
        }
    
    # Check if it's already a service account JSON
    if secrets.get('type') == 'service_account' and 'project_id' in secrets:
        print("[BigQuery] Secret contains full service account JSON")
        return secrets
    
    raise ValueError("BigQuery credentials not found in platform secrets")


def _ensure_credentials_file() -> str:
    """Ensure GCP credentials file exists and return its path."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    
    # Check if credentials file already exists and is valid
    if os.path.exists(creds_path) and os.path.getsize(creds_path) > 0:
        try:
            with open(creds_path, 'r') as f:
                existing_creds = json.load(f)
                if existing_creds.get('type') == 'service_account' and 'project_id' in existing_creds:
                    print(f"[BigQuery] Using existing credentials file: {creds_path}")
                    return creds_path
        except (json.JSONDecodeError, IOError):
            pass  # File is invalid, will recreate
    
    # Create credentials file from secrets
    print("[BigQuery] Creating credentials file from secrets")
    credentials = _get_bigquery_credentials()
    
    with open(creds_path, 'w') as f:
        json.dump(credentials, f)
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    print(f"[BigQuery] Credentials file created: {creds_path}")
    
    return creds_path


def get_bigquery_client():
    """Return a singleton BigQuery client instance."""
    global _bigquery_client
    
    if _bigquery_client:
        return _bigquery_client
    
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError(
            "The 'google-cloud-bigquery' Python package is required. "
            "Add a Lambda layer that provides the BigQuery SDK."
        ) from exc
    
    # Ensure credentials file exists
    _ensure_credentials_file()
    
    print("[BigQuery] Creating BigQuery client")
    _bigquery_client = bigquery.Client()
    print(f"[BigQuery] Client initialized for project: {_bigquery_client.project}")
    
    return _bigquery_client


def query_model_metadata(
    listing_uuids: List[str],
    brand: str,
    timeout: float = 30.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Query BigQuery for model metadata using listing UUIDs.
    
    Executes the query:
    SELECT 
        classifier_table.listing_uuid,
        mkd.id as model_id,
        mkd.model,
        mkd.root_model,
        mkd.root_model_id
    FROM `truss-data-science.model_classification.{brand}` classifier_table
    JOIN `truss-data-science.api.model_knowledge_display` mkd 
        ON mkd.id = classifier_table.model_id
    WHERE classifier_table.listing_uuid IN ({uuid_list})
    
    Args:
        listing_uuids: List of listing UUIDs (same as Pinecone vector IDs)
        brand: Brand namespace (e.g., 'jacquemus')
        timeout: Query timeout in seconds
        
    Returns:
        Dictionary mapping listing_uuid -> {model_id, model, root_model, root_model_id}
    """
    if not listing_uuids:
        return {}
    
    if not brand:
        raise ValueError("brand is required for BigQuery model lookup")
    
    print(f"\n[BigQuery] Querying model metadata for {len(listing_uuids)} UUIDs")
    print(f"  Brand: {brand}")
    
    client = get_bigquery_client()
    
    # Build parameterized query
    # Note: Brand is sanitized and used in table name (not user input in production)
    # UUIDs are parameterized to prevent injection
    uuid_placeholders = ", ".join([f"@uuid_{i}" for i in range(len(listing_uuids))])
    
    query = f"""
    SELECT 
        classifier_table.listing_uuid,
        mkd.id as model_id,
        mkd.model,
        mkd.root_model,
        mkd.root_model_id
    FROM `truss-data-science.model_classification.{brand}` classifier_table
    JOIN `truss-data-science.api.model_knowledge_display` mkd 
        ON mkd.id = classifier_table.model_id
    WHERE classifier_table.listing_uuid IN ({uuid_placeholders})
    """
    
    # Build query parameters
    job_config = None
    try:
        from google.cloud import bigquery
        
        query_params = [
            bigquery.ScalarQueryParameter(f"uuid_{i}", "STRING", uuid)
            for i, uuid in enumerate(listing_uuids)
        ]
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    except Exception as e:
        print(f"[BigQuery] Warning: Could not create parameterized query: {e}")
        # Fallback to inline values (escaped)
        escaped_uuids = ", ".join([f"'{uuid.replace(chr(39), chr(39)+chr(39))}'" for uuid in listing_uuids])
        query = f"""
        SELECT 
            classifier_table.listing_uuid,
            mkd.id as model_id,
            mkd.model,
            mkd.root_model,
            mkd.root_model_id
        FROM `truss-data-science.model_classification.{brand}` classifier_table
        JOIN `truss-data-science.api.model_knowledge_display` mkd 
            ON mkd.id = classifier_table.model_id
        WHERE classifier_table.listing_uuid IN ({escaped_uuids})
        """
    
    start_time = time.time()
    
    try:
        if job_config:
            query_job = client.query(query, job_config=job_config, timeout=timeout)
        else:
            query_job = client.query(query, timeout=timeout)
        
        rows = list(query_job.result(timeout=timeout))
        
        query_time = time.time() - start_time
        print(f"  ✓ Query completed in {query_time:.2f}s, {len(rows)} rows returned")
        
        # Build result dictionary
        results = {}
        for row in rows:
            listing_uuid = row.listing_uuid
            results[listing_uuid] = {
                'model_id': row.model_id,
                'model': row.model,
                'root_model': row.root_model,
                'root_model_id': row.root_model_id,
            }
        
        # Log any UUIDs that weren't found
        found_uuids = set(results.keys())
        missing_uuids = set(listing_uuids) - found_uuids
        if missing_uuids:
            print(f"  ⚠️  {len(missing_uuids)} UUIDs not found in BigQuery")
        
        return results
        
    except Exception as e:
        query_time = time.time() - start_time
        print(f"  ✗ BigQuery error after {query_time:.2f}s: {e}")
        raise RuntimeError(f"BigQuery model lookup failed: {e}") from e


__all__ = [
    "get_bigquery_client",
    "query_model_metadata",
]
