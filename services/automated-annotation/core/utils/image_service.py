"""
Image service utilities for fetching signed image URLs.
"""

import os
import logging
from typing import Optional
import requests

from stage_urls import get_annotation_dsl_url
from core.utils.credentials import get_request_auth_headers

logger = logging.getLogger(__name__)


def get_signed_image_url(image_id: str) -> Optional[str]:
    """
    Fetch signed image URL from annotation-data-service-layer.
    
    Args:
        image_id: Image/processing ID from image-processing table
        
    Returns:
        Signed image URL or None if fetch fails
        
    Raises:
        ValueError: If image URL cannot be fetched
    """
    base_url = get_annotation_dsl_url()
    api_key = os.getenv("ANNOTATION_API_KEY")
    
    # Get auth headers from incoming request for pass-through
    auth_headers = get_request_auth_headers()
    
    # Build request headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Use pass-through auth if available, otherwise use API key
    if auth_headers:
        headers.update(auth_headers)
        logger.debug("Using pass-through auth headers for image service")
    elif api_key:
        headers["x-api-key"] = api_key
        logger.debug("Using API key for image service")
    else:
        logger.warning("No authentication available for image service")
    
    # Construct API endpoint for processed image lookup
    # Endpoint: GET /images/processed/{image_id}
    url = f"{base_url.rstrip('/')}/images/processed/{image_id}"
    
    try:
        logger.info(f"Fetching signed URL for image: {image_id}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        
        # Unwrap 'data' wrapper if present (API response structure)
        data = result.get('data', result)
        
        # Extract downloadUrl from processedImage object
        processed_image = data.get('processedImage', {})
        download_url = processed_image.get('downloadUrl')
        
        if download_url:
            logger.debug(f"Successfully fetched download URL for image {image_id}")
            return download_url
        else:
            logger.error(f"downloadUrl not found in processedImage response for image {image_id}. Response: {result}")
            raise ValueError(f"downloadUrl not found in response for image {image_id}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch signed URL for image {image_id}: {str(e)}")
        raise ValueError(f"Failed to fetch signed URL for image {image_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error fetching signed URL for image {image_id}: {str(e)}")
        raise ValueError(f"Failed to fetch signed URL for image {image_id}: {str(e)}")
