"""
Taxonomy utilities for root/child lookups.
"""

import json
import time
import logging
from typing import Dict, Any, Optional

from agent_utils.dsl_api_client import DSLAPIClient

logger = logging.getLogger(__name__)


def lookup_root_from_child(
    api_client: DSLAPIClient,
    property_type: str,
    value: str,
    brand: str = None,
    root_type: str = None,
    partition: str = "bags"
) -> Optional[Dict[str, Any]]:
    """
    Lookup root taxonomy value from child value using the knowledge service API.
    
    Args:
        api_client: DSL API client instance
        property_type: Type of property (e.g., "model", "material")
        value: Child value to look up (e.g., model name)
        brand: Brand name (optional, used for model lookups)
        root_type: Root type filter (optional)
        partition: Category partition (default: "bags")
        
    Returns:
        Dictionary with root, root_id, and child_id, or None if not found
    """
    start_time = time.time()
    logger.info(f"Looking up root for {property_type}='{value}' (brand={brand}, root_type={root_type}, partition={partition})")
    
    try:
        lookup_result = api_client.lookup_root(
            property_type=property_type,
            value=value,
            brand=brand,
            root_type=root_type,
            partition=partition
        )
        elapsed_time = time.time() - start_time
        logger.debug(f"Lookup API response received in {elapsed_time:.2f}s: {json.dumps(lookup_result, default=str)[:500]}")
        
        data = lookup_result.get('data', lookup_result)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
            logger.debug("Extracted first item from list response")
        elif not isinstance(data, dict):
            data = lookup_result
            logger.debug("Using lookup_result directly as data")
            
        if not data.get('found'):
            logger.info(f"No root found for {property_type}='{value}' (lookup completed in {elapsed_time:.2f}s)")
            return None
            
        result = {
            'root': data.get('root'),
            'root_id': data.get('root_id'),
            'child_id': data.get(f'{property_type}_id'),
        }
        
        if result['root'] and result['root_id']:
            logger.info(f"Found root: {result['root']} (ID: {result['root_id']}) for {property_type}='{value}' in {elapsed_time:.2f}s")
            return result
        else:
            logger.warning(f"Incomplete root lookup result for {property_type}='{value}': {result}")
            return None
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error during root lookup for {property_type}='{value}' after {elapsed_time:.2f}s: {str(e)}")
        raise
