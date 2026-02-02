"""
BigQuery Brand Search Tool
==========================

Tool for searching brand_knowledge table in BigQuery.
Used by Agent 1 in brand classification workflow.
"""

import json
from typing import List, Dict, Any, Optional
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import boto3
import botocore
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig


# Configuration
BIGQUERY_SECRET_NAME = 'gcp/bigquery-and-gcs-cloud-migration-service-account'
BIGQUERY_REGION = 'eu-west-2'
BIGQUERY_PROJECT_ID = None
BIGQUERY_DATASET = 'api'
BIGQUERY_TABLE = 'brand_knowledge_display'

# Singleton for BigQuery client (reused across calls)
_bigquery_client_singleton = None


def get_bigquery_secret() -> Dict[str, Any]:
    """
    Retrieve BigQuery/GCP service account credentials from AWS Secrets Manager.
    Returns a dictionary with GCP service account credentials.
    """
    try:
        client = botocore.session.get_session().create_client('secretsmanager', region_name=BIGQUERY_REGION)
        cache_config = SecretCacheConfig()
        cache = SecretCache(config=cache_config, client=client)
        secret_string = cache.get_secret_string(BIGQUERY_SECRET_NAME)
        secret = json.loads(secret_string)
        return secret
    except Exception as e:
        raise Exception(f"Failed to retrieve BigQuery secret '{BIGQUERY_SECRET_NAME}': {e}")


def get_bigquery_client():
    """
    Get a BigQuery client using service account credentials from AWS Secrets Manager.
    Uses singleton pattern to reuse client across calls.
    """
    global _bigquery_client_singleton

    if _bigquery_client_singleton is not None:
        return _bigquery_client_singleton

    try:
        # Get credentials from AWS Secrets Manager
        secret = get_bigquery_secret()

        # Create service account credentials from secret
        credentials_info = {
            "type": secret.get("type", "service_account"),
            "project_id": secret.get("project_id"),
            "private_key_id": secret.get("private_key_id"),
            "private_key": secret.get("private_key"),
            "client_email": secret.get("client_email"),
            "client_id": secret.get("client_id"),
            "auth_uri": secret.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": secret.get("token_uri", "https://oauth2.googleapis.com/token"),
            "client_x509_cert_url": secret.get("client_x509_cert_url"),
            "universe_domain": secret.get("universe_domain", "googleapis.com")
        }

        # Validate required fields
        required_fields = ["project_id", "private_key", "client_email"]
        missing_fields = [field for field in required_fields if not credentials_info.get(field)]
        if missing_fields:
            raise ValueError(f"Secret missing required fields: {missing_fields}")

        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(credentials_info)

        # Create BigQuery client
        client = bigquery.Client(credentials=credentials, project=credentials_info["project_id"])

        # Store project ID globally for convenience
        global BIGQUERY_PROJECT_ID
        BIGQUERY_PROJECT_ID = credentials_info["project_id"]

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
