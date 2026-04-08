"""
Shared Lambda invocation utility for internal service-to-service calls.

Replaces HTTP API Gateway calls with direct Lambda invocation using internal auth,
bypassing API Gateway authorization. Used for calling DSL and ADSL services.
"""

import os
import json
import logging
from typing import Any, Dict, Optional

import boto3

from stage_urls import get_stage

logger = logging.getLogger(__name__)

# Lazy-initialized Lambda client
_lambda_client = None


def _get_lambda_client():
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client(
            "lambda",
            region_name=os.getenv("AWS_REGION", "eu-west-2"),
        )
    return _lambda_client


def invoke_dsl(service_name: str, http_method: str, path: str,
               query_params: Optional[Dict[str, str]] = None,
               body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Invoke a DSL Lambda function directly with internal auth.

    Args:
        service_name: DSL service name (e.g., "knowledge", "products")
        http_method: HTTP method (GET, POST, etc.)
        path: API path (e.g., "/bags/knowledge/lookup-root")
        query_params: Optional query string parameters
        body: Optional request body (for POST/PUT)

    Returns:
        Parsed response body

    Raises:
        ValueError: If the Lambda returns an error status
    """
    stage = get_stage()
    function_name = f"truss-data-service-{service_name}-{stage}"
    return _invoke_lambda(function_name, http_method, path, query_params, body)


def invoke_adsl(service_name: str, http_method: str, path: str,
                query_params: Optional[Dict[str, str]] = None,
                body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Invoke an ADSL Lambda function directly with internal auth.

    Args:
        service_name: ADSL service name (e.g., "image-service", "knowledge")
        http_method: HTTP method (GET, POST, etc.)
        path: API path (e.g., "/images/processed/{id}")
        query_params: Optional query string parameters
        body: Optional request body (for POST/PUT)

    Returns:
        Parsed response body

    Raises:
        ValueError: If the Lambda returns an error status
    """
    stage = get_stage()
    function_name = f"truss-annotation-data-service-{service_name}-{stage}"
    return _invoke_lambda(function_name, http_method, path, query_params, body)


def _invoke_lambda(function_name: str, http_method: str, path: str,
                   query_params: Optional[Dict[str, str]] = None,
                   body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Internal: invoke a Lambda with API Gateway-style event and internal auth.
    """
    stage = get_stage()

    event = {
        "httpMethod": http_method.upper(),
        "path": path,
        "headers": {},
        "queryStringParameters": query_params,
        "body": json.dumps(body) if body else None,
        "requestContext": {
            "stage": stage,
            "authorizer": {
                "authType": "internal",
            },
        },
    }

    logger.info(f"Invoking {function_name}: {http_method} {path}")
    client = _get_lambda_client()
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(event),
    )

    # Check for Lambda-level errors
    if response.get("FunctionError"):
        payload = json.loads(response["Payload"].read().decode())
        error_msg = payload.get("errorMessage", "Lambda function error")
        raise ValueError(f"Lambda error from {function_name}: {error_msg}")

    # Parse Lambda response
    lambda_result = json.loads(response["Payload"].read().decode())
    status_code = lambda_result.get("statusCode", 200)

    # Parse response body
    resp_body = lambda_result.get("body", "{}")
    if isinstance(resp_body, str):
        try:
            resp_body = json.loads(resp_body)
        except (json.JSONDecodeError, ValueError):
            pass

    if status_code >= 400:
        raise ValueError(
            f"{function_name} returned {status_code}: {json.dumps(resp_body, default=str)[:300]}"
        )

    return resp_body
