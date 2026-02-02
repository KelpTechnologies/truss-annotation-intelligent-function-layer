"""
BigQuery Taxonomy Root Lookup Tool

This module provides functions to look up root/parent taxonomy values from child IDs
using BigQuery as the data source.

Database Schema:
---------------------------
Queries the model_knowledge_display and material_knowledge_display tables in BigQuery.

For model:
    Table: `truss-data-science.{api_env}.model_knowledge_display`
    Query: SELECT root_model, root_model_id WHERE model_id = ?

For material:
    Table: `truss-data-science.{api_env}.material_knowledge_display`
    Query: SELECT root_material, root_material_id WHERE material_id = ?

API Environment Mapping:
    - dev -> api_Dev
    - staging -> api_staging
    - prod -> api_prod
"""

import json
import logging
import os
from typing import Dict, Any, Optional

import boto3

logger = logging.getLogger(__name__)

# BigQuery configuration constants
BIGQUERY_PROJECT_ID = "truss-data-science"

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

    logger.info(f"[BigQuery] Loading platform secrets from: {secret_arn}")
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
        logger.info("[BigQuery] Found BigQuery credentials under 'bigquery' key")
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
        logger.info("[BigQuery] Secret contains full service account JSON")
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
                    logger.info(f"[BigQuery] Using existing credentials file: {creds_path}")
                    return creds_path
        except (json.JSONDecodeError, IOError):
            pass  # File is invalid, will recreate

    # Create credentials file from secrets
    logger.info("[BigQuery] Creating credentials file from secrets")
    credentials = _get_bigquery_credentials()

    with open(creds_path, 'w') as f:
        json.dump(credentials, f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    logger.info(f"[BigQuery] Credentials file created: {creds_path}")

    return creds_path


def _get_bigquery_client():
    """
    Initialize and return a BigQuery client.

    Returns:
        google.cloud.bigquery.Client or None if initialization fails
    """
    global _bigquery_client

    if _bigquery_client:
        return _bigquery_client

    try:
        from google.cloud import bigquery
    except ImportError as exc:
        logger.error(
            "The 'google-cloud-bigquery' Python package is required. "
            "Add a Lambda layer that provides the BigQuery SDK."
        )
        return None

    try:
        # Ensure credentials file exists
        _ensure_credentials_file()

        logger.info("[BigQuery] Creating BigQuery client")
        _bigquery_client = bigquery.Client()
        logger.info(f"[BigQuery] Client initialized for project: {_bigquery_client.project}")

        return _bigquery_client
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {e}", exc_info=True)
        return None


def _get_api_env_name() -> str:
    """Map STAGE environment variable to BigQuery dataset name."""
    stage = os.getenv("STAGE", "dev").lower()
    mapping = {
        "dev": "api_Dev",
        "staging": "api_staging",
        "prod": "api_prod"
    }
    env_name = mapping.get(stage, "api_Dev")
    logger.info(f"[BigQuery] Mapped STAGE={stage} to API environment: {env_name}")
    return env_name


def _query_root_from_child_bigquery(
    child_id: int,
    property_type: str,
    category: str = "bags"
) -> Optional[Dict[str, Any]]:
    """
    Query BigQuery to find root taxonomy entry from child ID.

    Args:
        child_id: Child taxonomy ID to look up (e.g., model_id or material_id)
        property_type: Type of property ('model' or 'material')
        category: Category name (default: 'bags') - currently unused but kept for API compatibility

    Returns:
        Dictionary with root_id and root_name, or None if not found

    Database Tables:
        - model: truss-data-science.{api_env}.model_knowledge_display
        - material: truss-data-science.{api_env}.material_knowledge_display
    """
    client = _get_bigquery_client()
    if not client:
        logger.error("BigQuery client is not available")
        return None

    try:
        from google.cloud import bigquery
    except ImportError:
        logger.error("google-cloud-bigquery package not available")
        return None

    # Get API environment name based on STAGE
    api_env = _get_api_env_name()

    # Construct table name and column names based on property type
    if property_type == "model":
        table_name = f"{BIGQUERY_PROJECT_ID}.{api_env}.model_knowledge_display"
        id_column = "model_id"
        root_id_column = "root_model_id"
        root_name_column = "root_model"
    elif property_type == "material":
        table_name = f"{BIGQUERY_PROJECT_ID}.{api_env}.material_knowledge_display"
        id_column = "material_id"
        root_id_column = "root_material_id"
        root_name_column = "root_material"
    else:
        logger.error(f"Invalid property_type: {property_type}. Must be 'model' or 'material'")
        return None

    # Build parameterized query (prevents SQL injection)
    query = f"""
        SELECT {root_name_column}, {root_id_column}
        FROM `{table_name}`
        WHERE {id_column} = @child_id
        LIMIT 1
    """

    # Configure query parameters
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("child_id", "INT64", child_id)
        ]
    )

    try:
        logger.info(
            f"[BigQuery] Querying {property_type} root: "
            f"child_id={child_id}, table={table_name}"
        )

        # Execute query
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()  # Wait for query to complete

        # Fetch first row
        for row in results:
            root_id = row[root_id_column]
            root_name = row[root_name_column]
            logger.info(
                f"[BigQuery] Root lookup successful: {property_type}_id={child_id} -> "
                f"root_id={root_id}, root_name={root_name}"
            )
            return {
                'root_id': root_id,
                'root_name': root_name
            }

        # No results found
        logger.warning(
            f"[BigQuery] No root found for {property_type} child_id {child_id} in {table_name}"
        )
        return None

    except Exception as e:
        logger.error(
            f"[BigQuery] Query failed for {property_type} child_id {child_id}: {e}",
            exc_info=True
        )
        return None


def lookup_root_from_child_bigquery(
    child_id: int,
    property_type: str = None,
    category: str = "bags"
) -> Optional[Dict[str, Any]]:
    """
    Look up root taxonomy item from child ID using BigQuery.

    This is the main entry point for root taxonomy lookups. It wraps the
    internal query function with additional validation and error handling.

    Args:
        child_id: Child taxonomy ID to look up
        property_type: Type of property ('model' or 'material')
                      If None, will attempt to infer from context
        category: Category name (default: 'bags')

    Returns:
        Dictionary with:
        {
            'root_id': int or None,
            'root_name': str or None,
            'error': str or None
        }

    Example Usage:
        >>> result = lookup_root_from_child_bigquery(
        ...     child_id=1234,
        ...     property_type='model',
        ...     category='bags'
        ... )
        >>> if result and result.get('root_id'):
        ...     print(f"Root: {result['root_name']} (ID: {result['root_id']})")
        ... else:
        ...     print(f"Error: {result.get('error')}")
    """
    # Validate inputs
    if not child_id or child_id == 0:
        error_msg = f"Invalid child_id: {child_id}"
        logger.warning(error_msg)
        return {
            'root_id': None,
            'root_name': None,
            'error': error_msg
        }

    if property_type and property_type not in ['model', 'material']:
        error_msg = f"Invalid property_type: {property_type}. Must be 'model' or 'material'"
        logger.error(error_msg)
        return {
            'root_id': None,
            'root_name': None,
            'error': error_msg
        }

    try:
        logger.info(
            f"Looking up root for {property_type} child_id {child_id} in category {category}"
        )

        # Query BigQuery for root taxonomy entry
        result = _query_root_from_child_bigquery(child_id, property_type, category)

        if result and result.get('root_id'):
            logger.info(
                f"Root lookup successful: child_id={child_id} -> "
                f"root_id={result['root_id']}, root_name={result['root_name']}"
            )
            return {
                'root_id': result['root_id'],
                'root_name': result['root_name'],
                'error': None
            }
        else:
            error_msg = f"No root found for {property_type} child_id {child_id}"
            logger.warning(error_msg)
            return {
                'root_id': None,
                'root_name': None,
                'error': error_msg
            }

    except Exception as e:
        error_msg = f"Root lookup failed for {property_type} child_id {child_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'root_id': None,
            'root_name': None,
            'error': error_msg
        }


# Convenience functions for specific property types
def lookup_model_root_bigquery(model_id: int, category: str = "bags") -> Optional[Dict[str, Any]]:
    """
    Look up root model from model ID.

    Args:
        model_id: Model taxonomy ID
        category: Category name (default: 'bags')

    Returns:
        Dictionary with root_id, root_name, error
    """
    return lookup_root_from_child_bigquery(
        child_id=model_id,
        property_type='model',
        category=category
    )


def lookup_material_root_bigquery(material_id: int, category: str = "bags") -> Optional[Dict[str, Any]]:
    """
    Look up root material from material ID.

    Args:
        material_id: Material taxonomy ID
        category: Category name (default: 'bags')

    Returns:
        Dictionary with root_id, root_name, error
    """
    return lookup_root_from_child_bigquery(
        child_id=material_id,
        property_type='material',
        category=category
    )
