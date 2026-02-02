"""
Credential management utilities for Lambda function.

Handles GCP, Pinecone, and other service credentials.
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Global storage for request auth headers (for pass-through)
_request_auth_headers: Dict[str, str] = {}


def set_request_auth_headers(headers: Dict[str, str]) -> None:
    """Set authentication headers from incoming request for pass-through to downstream services."""
    global _request_auth_headers
    _request_auth_headers = headers.copy()
    logger.debug(f"Set request auth headers: {list(headers.keys())}")


def get_request_auth_headers() -> Dict[str, str]:
    """Get authentication headers from incoming request for pass-through."""
    return _request_auth_headers.copy()


def ensure_gcp_adc() -> None:
    """
    Ensure GCP Application Default Credentials (ADC) are set up.
    
    For local development, this checks if GOOGLE_APPLICATION_CREDENTIALS is set.
    For Lambda, credentials are typically provided via environment or IAM role.
    
    This is a no-op for local testing if credentials are already configured.
    """
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    secret_arn = os.getenv("TRUSS_SECRETS_ARN") or os.getenv("BIGQUERY_SECRET_ARN")
    
    if creds_path:
        if os.path.exists(creds_path):
            logger.debug(f"GCP credentials file found: {creds_path}")
        else:
            logger.warning(f"GCP credentials file specified but not found: {creds_path}")
    elif secret_arn:
        logger.debug(f"GCP credentials will be loaded from Secrets Manager: {secret_arn}")
    else:
        # For local testing, this might be okay if using gcloud auth
        logger.debug("No explicit GCP credentials configured - may use default credentials")
    
    # Note: In Lambda, credentials are typically set up via IAM role or environment
    # This function mainly serves as a checkpoint to ensure setup is considered


def ensure_pinecone_api_key() -> None:
    """
    Ensure Pinecone API key is available.
    
    Raises ValueError if PINECONE_API_KEY is not set.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    logger.debug("Pinecone API key is configured")
