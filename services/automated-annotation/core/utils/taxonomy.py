"""
Taxonomy utilities for root/child lookups.
"""

import os
import logging
from typing import Dict, Any, Optional

from agent_utils.dsl_api_client import DSLAPIClient
from stage_urls import get_dsl_url, get_stage

logger = logging.getLogger(__name__)


def lookup_root_from_child(
    child_id: int,
    category: str = "bags"
) -> Optional[Dict[str, Any]]:
    """
    Look up root taxonomy item from child ID.
    
    Args:
        child_id: Child taxonomy ID
        category: Category name (e.g., "bags")
        
    Returns:
        Dictionary with root_id and root_name, or None if not found
    """
    try:
        api_base_url = get_dsl_url()
        api_key = os.getenv("DSL_API_KEY")
        
        if not api_key:
            logger.warning("DSL_API_KEY not set - cannot lookup root taxonomy")
            return None
        
        # Get auth headers for pass-through
        from core.utils.credentials import get_request_auth_headers
        auth_headers = get_request_auth_headers()
        
        client = DSLAPIClient(
            base_url=api_base_url,
            api_key=api_key,
            auth_headers=auth_headers if auth_headers else None
        )
        
        # Call taxonomy lookup endpoint
        # Use _make_request method (private but only way to make custom API calls)
        response = client._make_request(
            method='GET',
            endpoint=f"/taxonomy/{category}/root-from-child/{child_id}"
        )
        
        if response and response.get("root_id"):
            return {
                "root_id": response.get("root_id"),
                "root_name": response.get("root_name")
            }
        else:
            logger.warning(f"Root taxonomy not found for child_id {child_id}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to lookup root taxonomy for child_id {child_id}: {str(e)}")
        return None
