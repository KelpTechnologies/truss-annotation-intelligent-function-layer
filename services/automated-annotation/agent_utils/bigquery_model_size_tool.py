"""
BigQuery Model Size Search Tool
================================

Tool for searching model_size_knowledge_display table in BigQuery.
Used by model size classification workflow.
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
BIGQUERY_TABLE = 'model_size_knowledge_display'


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
    """
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
