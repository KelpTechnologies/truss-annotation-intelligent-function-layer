"""
Credential management utilities for Lambda function.

Handles GCP, Pinecone, and other service credentials.
Loads credentials from AWS Secrets Manager for Lambda execution.
"""

import os
import json
import base64
import logging
from typing import Dict, Optional

import boto3

logger = logging.getLogger(__name__)

# Global storage for request auth headers (for pass-through)
_request_auth_headers: Dict[str, str] = {}

# Cache for centralized secrets
_cached_platform_secrets: Optional[Dict] = None

# Flag to track if Pinecone key has been loaded from secrets
_pinecone_key_loaded_from_secrets: bool = False


def set_request_auth_headers(headers: Dict[str, str]) -> None:
    """Set authentication headers from incoming request for pass-through to downstream services."""
    global _request_auth_headers
    _request_auth_headers = headers.copy()
    logger.debug(f"Set request auth headers: {list(headers.keys())}")


def get_request_auth_headers() -> Dict[str, str]:
    """Get authentication headers from incoming request for pass-through."""
    return _request_auth_headers.copy()


def _get_platform_secrets() -> Dict:
    """Get all platform secrets from centralized truss-platform-secrets."""
    global _cached_platform_secrets
    if _cached_platform_secrets:
        return _cached_platform_secrets
    
    secret_arn = os.getenv("TRUSS_SECRETS_ARN") or os.getenv("BIGQUERY_SECRET_ARN")
    if not secret_arn:
        raise ValueError("TRUSS_SECRETS_ARN environment variable is required")
    
    logger.info(f"Loading platform secrets from: {secret_arn}")
    secrets_client = boto3.client('secretsmanager', region_name=os.getenv('AWS_REGION', 'eu-west-2'))
    
    response = secrets_client.get_secret_value(SecretId=secret_arn)
    secret_string = response.get('SecretString')
    
    if not secret_string:
        raise ValueError(f"Secret string is empty for secret {secret_arn}")
    
    _cached_platform_secrets = json.loads(secret_string)
    return _cached_platform_secrets


def _load_gcp_credentials_from_secrets() -> str:
    """Load GCP service account JSON from AWS Secrets Manager (centralized truss-platform-secrets)."""
    secret_arn = os.getenv("TRUSS_SECRETS_ARN") or os.getenv("BIGQUERY_SECRET_ARN")
    if not secret_arn:
        raise ValueError("TRUSS_SECRETS_ARN environment variable is required")
    
    logger.info(f"Loading GCP credentials from Secrets Manager: {secret_arn}")
    secrets_client = boto3.client('secretsmanager', region_name=os.getenv('AWS_REGION', 'eu-west-2'))
    
    response = secrets_client.get_secret_value(SecretId=secret_arn)
    secret_string = response.get('SecretString')
    
    if not secret_string:
        raise ValueError(f"Secret string is empty for secret {secret_arn}")
    
    # Parse as JSON to validate and extract
    try:
        secret_data = json.loads(secret_string)
        logger.debug(f"Secret parsed as JSON, keys: {list(secret_data.keys()) if isinstance(secret_data, dict) else 'not a dict'}")
        
        # If it's a dict, try to extract the service account JSON
        if isinstance(secret_data, dict):
            # Check for centralized 'bigquery' key (truss-platform-secrets structure)
            if 'bigquery' in secret_data:
                logger.info("Found BigQuery credentials under 'bigquery' key (centralized secrets)")
                bigquery_creds = secret_data['bigquery']
                # Build complete service account JSON
                service_account = {
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
                return json.dumps(service_account)
            
            # Check if it's already a service account JSON (has 'type' and 'project_id')
            if secret_data.get('type') == 'service_account' and 'project_id' in secret_data:
                logger.info("Secret contains full service account JSON")
                return json.dumps(secret_data)
            
            # Otherwise, try to find nested service account JSON
            for key in ['service_account_json', 'credentials', 'value', 'gcp_service_account']:
                if key in secret_data:
                    logger.info(f"Found nested service account JSON under key: {key}")
                    nested_value = secret_data[key]
                    if isinstance(nested_value, str):
                        return nested_value
                    elif isinstance(nested_value, dict):
                        return json.dumps(nested_value)
                    break
            
            # If no nested key found, use the whole dict
            logger.info("Using entire secret as service account JSON")
            return json.dumps(secret_data)
        else:
            # Not a dict, return as-is (should be JSON string already)
            logger.info("Secret is not a dict, returning as-is")
            return secret_string
    except json.JSONDecodeError as e:
        raise ValueError(f"Secret is not valid JSON: {str(e)}")


def ensure_gcp_adc() -> None:
    """
    Ensure GCP Application Default Credentials (ADC) are set up.
    
    Loads credentials from AWS Secrets Manager and writes to /tmp/gcp_sa.json.
    Sets GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT, and VERTEXAI_PROJECT.
    """
    logger.info("Setting up GCP Application Default Credentials")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    
    # Check if credentials file already exists and is valid
    if os.path.exists(creds_path) and os.path.getsize(creds_path) > 0:
        try:
            with open(creds_path, 'r') as f:
                existing_creds = json.load(f)
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                if all(field in existing_creds for field in required_fields):
                    logger.info(f"GCP credentials file already exists and is valid at {creds_path}")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                    # Ensure project env vars are set
                    _set_project_env_vars(existing_creds.get('project_id'))
                    return
                else:
                    logger.warning(f"Existing credentials file at {creds_path} is missing required fields, will recreate")
        except Exception as e:
            logger.warning(f"Existing credentials file at {creds_path} is invalid, will recreate: {e}")
    
    # Try to get from environment variable first
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    
    # If not in env, load from Secrets Manager
    if not sa_json:
        logger.info("GCP_SERVICE_ACCOUNT_JSON not in environment, loading from Secrets Manager")
        sa_json = _load_gcp_credentials_from_secrets()
    
    if not sa_json:
        raise ValueError("GCP credentials not available - cannot proceed with LLM classification")

    logger.info(f"Setting up GCP credentials at {creds_path}")
    
    # Parse JSON content
    if sa_json.strip().startswith("{"):
        content = sa_json
        logger.debug("GCP credentials are JSON format")
    else:
        try:
            content = base64.b64decode(sa_json).decode("utf-8")
            logger.debug("GCP credentials decoded from base64")
        except Exception as e:
            raise ValueError(f"Failed to decode base64 credentials: {str(e)}")

    # Validate JSON content and ensure private key has proper newlines
    try:
        creds_dict = json.loads(content)
        
        # Ensure private key has actual newlines (not escaped \n)
        if 'private_key' in creds_dict:
            private_key = creds_dict['private_key']
            if '\\n' in private_key and '\n' not in private_key:
                logger.debug("Converting escaped newlines in private key to actual newlines")
                creds_dict['private_key'] = private_key.replace('\\n', '\n')
                content = json.dumps(creds_dict)
        
        # Re-validate after potential modification
        json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"GCP credentials are not valid JSON: {str(e)}")

    # Ensure directory exists
    creds_dir = os.path.dirname(creds_path)
    if creds_dir:
        os.makedirs(creds_dir, exist_ok=True)

    # Write credentials file
    with open(creds_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Set file permissions to 600 (read/write for owner only)
    os.chmod(creds_path, 0o600)
    
    # Verify file was written
    if not os.path.exists(creds_path):
        raise IOError(f"Failed to create credentials file at {creds_path}")
    
    file_size = os.path.getsize(creds_path)
    if file_size == 0:
        raise IOError(f"Credentials file is empty at {creds_path}")
    
    logger.info(f"GCP credentials written successfully to {creds_path} (size: {file_size} bytes)")

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS={creds_path}")

    # Set project environment variables
    creds_data = json.loads(content)
    _set_project_env_vars(creds_data.get('project_id'))
    
    service_account_email = creds_data.get("client_email", "unknown")
    logger.info(f"GCP Configuration - Project: {os.getenv('VERTEXAI_PROJECT')}, Service Account: {service_account_email}")


def _set_project_env_vars(project_id: Optional[str]) -> None:
    """Set project-related environment variables."""
    vertexai_project = os.getenv("VERTEXAI_PROJECT") or project_id
    
    if not vertexai_project:
        raise ValueError("VERTEXAI_PROJECT environment variable is required (or project_id must be in GCP service account JSON)")
    
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        os.environ["GOOGLE_CLOUD_PROJECT"] = vertexai_project
        logger.info(f"Set GOOGLE_CLOUD_PROJECT={vertexai_project}")
    
    if not os.getenv("VERTEXAI_PROJECT"):
        os.environ["VERTEXAI_PROJECT"] = vertexai_project
        logger.info(f"Set VERTEXAI_PROJECT={vertexai_project}")


def ensure_pinecone_api_key() -> None:
    """
    Ensure Pinecone API key is available from Secrets Manager.
    
    Always loads from Secrets Manager to ensure the latest key is used.
    Only skips if we've already loaded from secrets in this Lambda invocation.
    """
    global _pinecone_key_loaded_from_secrets
    
    # If we've already loaded from secrets in this invocation, skip
    if _pinecone_key_loaded_from_secrets:
        logger.info("PINECONE_API_KEY already loaded from secrets")
        return
    
    logger.info("Loading Pinecone API key from centralized secrets")
    try:
        secrets = _get_platform_secrets()
        pinecone_key = secrets.get('pinecone', {}).get('api_key')
        if pinecone_key:
            os.environ["PINECONE_API_KEY"] = pinecone_key
            _pinecone_key_loaded_from_secrets = True
            logger.info("PINECONE_API_KEY set from centralized secrets")
        else:
            logger.warning("Pinecone API key not found in centralized secrets")
            raise ValueError("PINECONE_API_KEY not found in secrets")
    except Exception as e:
        logger.error(f"Failed to load Pinecone API key: {e}")
        raise
