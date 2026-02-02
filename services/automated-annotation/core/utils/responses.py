"""
HTTP response utilities for Lambda function.
"""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse request body from Lambda event.
    
    Args:
        event: Lambda event dictionary
        
    Returns:
        Parsed JSON body as dictionary, or empty dict if no body
    """
    body = event.get("body")
    if not body:
        return {}
    
    # Handle string body (API Gateway)
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON body: {str(e)}")
            raise ValueError(f"Invalid JSON in request body: {str(e)}")
    
    # Handle already-parsed body
    if isinstance(body, dict):
        return body
    
    logger.warning(f"Unexpected body type: {type(body)}")
    return {}


def create_response(
    status_code: int,
    body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Create Lambda API Gateway response.
    
    Args:
        status_code: HTTP status code
        body: Response body (will be JSON-serialized)
        headers: Optional response headers
        
    Returns:
        Lambda API Gateway response dictionary
    """
    default_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS"
    }
    
    if headers:
        default_headers.update(headers)
    
    return {
        "statusCode": status_code,
        "headers": default_headers,
        "body": json.dumps(body, default=str)
    }
