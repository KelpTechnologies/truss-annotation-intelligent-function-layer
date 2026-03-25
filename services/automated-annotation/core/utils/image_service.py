"""
Image service utilities for fetching signed image URLs.

Uses direct Lambda invocation (internal auth) to call the ADSL image-service,
bypassing API Gateway authorization.
"""

import os
import json
import logging
from typing import Optional

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


def _get_adsl_function_name(service_name: str) -> str:
    """Build ADSL Lambda function name for the current stage."""
    stage = get_stage()
    return f"truss-annotation-data-service-{service_name}-{stage}"


def get_signed_image_url(image_id: str) -> Optional[str]:
    """
    Fetch signed image URL from annotation-data-service-layer via direct Lambda invoke.

    Args:
        image_id: Image/processing ID from image-processing table

    Returns:
        Signed image URL or None if fetch fails

    Raises:
        ValueError: If image URL cannot be fetched
    """
    stage = get_stage()
    function_name = _get_adsl_function_name("image-service")

    # Build API Gateway-style event for the Lambda handler
    event = {
        "httpMethod": "GET",
        "path": f"/images/processed/{image_id}",
        "headers": {},
        "queryStringParameters": None,
        "body": None,
        "requestContext": {
            "stage": stage,
            "authorizer": {
                "authType": "internal",
            },
        },
    }

    try:
        logger.info(f"Invoking {function_name} for image: {image_id}")
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
            raise ValueError(f"Lambda error for image {image_id}: {error_msg}")

        # Parse Lambda response
        lambda_result = json.loads(response["Payload"].read().decode())
        status_code = lambda_result.get("statusCode", 200)

        if status_code >= 400:
            body = lambda_result.get("body", "")
            raise ValueError(
                f"ADSL image-service returned {status_code} for image {image_id}: {body}"
            )

        # Parse response body
        body = lambda_result.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)

        # Unwrap 'data' wrapper if present (API response structure)
        data = body.get("data", body)

        # Extract downloadUrl from processedImage object
        processed_image = data.get("processedImage", {})
        download_url = processed_image.get("downloadUrl")

        if download_url:
            logger.debug(f"Successfully fetched download URL for image {image_id}")
            return download_url
        else:
            logger.error(
                f"downloadUrl not found in processedImage response for image {image_id}. "
                f"Response: {body}"
            )
            raise ValueError(f"downloadUrl not found in response for image {image_id}")

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch signed URL for image {image_id}: {str(e)}")
        raise ValueError(f"Failed to fetch signed URL for image {image_id}: {str(e)}")
