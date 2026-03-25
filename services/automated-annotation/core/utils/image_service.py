"""
Image service utilities for fetching signed image URLs.

Uses direct Lambda invocation (internal auth) to call the ADSL image-service,
bypassing API Gateway authorization.
"""

import logging
from typing import Optional

from core.utils.lambda_invoke import invoke_adsl

logger = logging.getLogger(__name__)


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
    try:
        logger.info(f"Fetching signed URL for image: {image_id}")
        body = invoke_adsl(
            "image-service", "GET", f"/images/processed/{image_id}"
        )

        # Unwrap 'data' wrapper if present (API response structure)
        data = body.get("data", body) if isinstance(body, dict) else body

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
