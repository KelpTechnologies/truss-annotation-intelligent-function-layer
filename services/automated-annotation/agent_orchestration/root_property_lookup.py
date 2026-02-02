"""
Root property lookup utilities for hierarchical taxonomy resolution.

Provides lookup functions to find root/parent taxonomy items from child IDs.

DATA SOURCE: BigQuery taxonomy hierarchy tables
See BIGQUERY_TAXONOMY_LOOKUP_IMPLEMENTATION.md for implementation details.
"""

import logging
from typing import Dict, Any, Optional, List

# Use BigQuery-based taxonomy lookup
# NOTE: This requires BigQuery tables to be set up and functions to be implemented
# See core/utils/bigquery_taxonomy_lookup.py and BIGQUERY_TAXONOMY_LOOKUP_IMPLEMENTATION.md
from core.utils.bigquery_taxonomy_lookup import lookup_root_from_child_bigquery

logger = logging.getLogger(__name__)


def lookup_root_property(
    property_type: str,
    property_label_id: int,
    category: str = "bags"
) -> Dict[str, Any]:
    """
    Look up root property from child property ID.

    Args:
        property_type: Property type ('model' or 'material')
        property_label_id: Child taxonomy ID to lookup root for
        category: Category name (default: 'bags')

    Returns:
        Dictionary with:
        - root_property_name: Root taxonomy name (or None if not found)
        - root_property_id: Root taxonomy ID (or None if not found)
        - error_logs: List of error messages (empty if successful)
    """
    error_logs = []

    # Validate inputs
    if property_type not in ['model', 'material']:
        error = f"Invalid property_type '{property_type}'. Must be 'model' or 'material'."
        logger.error(error)
        error_logs.append(error)
        return {
            'root_property_name': None,
            'root_property_id': None,
            'error_logs': error_logs
        }

    if not property_label_id or property_label_id == 0:
        error = f"Invalid property_label_id '{property_label_id}' for {property_type}"
        logger.warning(error)
        error_logs.append(error)
        return {
            'root_property_name': None,
            'root_property_id': None,
            'error_logs': error_logs
        }

    # Call BigQuery taxonomy lookup
    try:
        logger.info(f"Looking up root for {property_type} with ID {property_label_id} via BigQuery")
        bq_result = lookup_root_from_child_bigquery(
            child_id=property_label_id,
            property_type=property_type,
            category=category
        )

        # Check if lookup was successful
        if bq_result and bq_result.get('root_id'):
            logger.info(
                f"Root lookup successful: {property_type} ID {property_label_id} -> "
                f"root_id={bq_result['root_id']}, root_name={bq_result['root_name']}"
            )
            return {
                'root_property_name': bq_result.get('root_name'),
                'root_property_id': bq_result.get('root_id'),
                'error_logs': []
            }
        else:
            # Lookup failed or returned None
            error_from_bq = bq_result.get('error') if bq_result else f"BigQuery lookup returned None for {property_type} ID {property_label_id}"
            error = error_from_bq or f"No root found for {property_type} ID {property_label_id}"
            logger.warning(error)
            error_logs.append(error)
            return {
                'root_property_name': None,
                'root_property_id': None,
                'error_logs': error_logs
            }

    except Exception as e:
        error = f"Root lookup failed for {property_type} ID {property_label_id}: {str(e)}"
        logger.error(error, exc_info=True)
        error_logs.append(error)
        return {
            'root_property_name': None,
            'root_property_id': None,
            'error_logs': error_logs
        }


def lookup_model_root(model_id: int, category: str = "bags") -> Dict[str, Any]:
    """
    Convenience function for model root lookup.

    Args:
        model_id: Model taxonomy ID
        category: Category name

    Returns:
        Dict with root_model_name, root_model_id, error_logs
    """
    result = lookup_root_property(
        property_type='model',
        property_label_id=model_id,
        category=category
    )

    # Rename keys for model-specific response
    return {
        'root_model_name': result['root_property_name'],
        'root_model_id': result['root_property_id'],
        'error_logs': result['error_logs']
    }


def lookup_material_root(material_id: int, category: str = "bags") -> Dict[str, Any]:
    """
    Convenience function for material root lookup.

    Args:
        material_id: Material taxonomy ID
        category: Category name

    Returns:
        Dict with root_material_name, root_material_id, error_logs
    """
    result = lookup_root_property(
        property_type='material',
        property_label_id=material_id,
        category=category
    )

    # Rename keys for material-specific response
    return {
        'root_material_name': result['root_property_name'],
        'root_material_id': result['root_property_id'],
        'error_logs': result['error_logs']
    }
