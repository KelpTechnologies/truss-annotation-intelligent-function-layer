"""
API-Compatible Classification Orchestration

Orchestration function designed for Lambda API requests.
Handles input transformation (image IDs, metadata fields) and returns
standardized classification output with both IDs and names.

This function should be called by agent_orchestration_api_handler.py
and handles all Lambda-specific concerns internally.
"""

import json
import logging
import time
from typing import Dict, Any, Optional

from agent_architecture import LLMAnnotationAgent
from agent_architecture.validation import AgentStatus
from agent_orchestration.csv_config_loader import ConfigLoader
from agent_orchestration.classifier_orchestration import (
    extract_classification_result,
    get_name_from_schema
)
from agent_orchestration.root_property_lookup import lookup_material_root, lookup_model_root

logger = logging.getLogger(__name__)


def classify_for_api(
    config_id: str,
    image_id: Optional[str] = None,
    image_url: Optional[str] = None,
    text_input: Optional[str] = None,
    text_metadata: Optional[Dict[str, Any]] = None,
    input_mode: str = "auto",
    env: Optional[str] = None,
    get_image_url_fn: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Classify using agent architecture - API-compatible orchestration function.
    
    This function handles:
    - Input transformation (image IDs → URLs, metadata → stringified JSON)
    - Config loading
    - Agent initialization
    - Agent execution
    - Output formatting (IDs + names)
    
    Args:
        config_id: Agent configuration ID (e.g., 'material-30', 'classifier-model-bags')
        image_id: Image ID (processing_id) - will fetch signed URL if get_image_url_fn provided
        image_url: Direct image URL (if already signed, takes precedence over image_id)
        text_input: Pre-formatted text input (string) - takes precedence if provided
        text_metadata: Text metadata as JSON dict (e.g., {"brand": "...", "title": "...", "description": "..."})
            - Will be converted to stringified JSON when passed to agent
            - Allows filtering of columns where required
        input_mode: Input mode ('auto', 'image-only', 'text-only', 'multimodal')
        env: Environment for config loading ('dev', 'staging', 'prod') - defaults to 'staging'
        get_image_url_fn: Optional function to fetch signed URL from image_id (Lambda-specific)
    
    Returns:
        Standardized classification result dictionary:
        {
            'config_id': str,
            'primary': str,              # Legacy format: "ID X"
            'primary_id': int,           # Integer ID
            'primary_name': str,         # Name from schema
            'alternatives': List[str],   # Legacy format: ["ID X", ...]
            'alternative_ids': List[int],
            'alternative_names': List[str],
            'confidence': float,
            'reasoning': str,
            'success': bool,
            'status': str,
            'processing_time_seconds': float,
            'validation_passed': bool,
            'error': Optional[str],
            'error_type': Optional[str]
        }
    """
    start_time = time.time()
    
    if env is None:
        env = 'staging'  # Default environment
    
    logger.info(f"Starting API classification - config_id: {config_id}, env: {env}")
    
    # Initialize config loader
    config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
    
    # Load full agent config
    logger.info(f"Loading agent config: {config_id}")
    try:
        full_config = config_loader.load_full_agent_config(config_id)
    except Exception as e:
        logger.error(f"Failed to load agent config {config_id}: {str(e)}")
        raise ValueError(f"Agent config '{config_id}' not found: {str(e)}")
    
    # Get image URL if image_id provided and function available
    if image_id and not image_url and get_image_url_fn:
        logger.info(f"Fetching signed URL for image: {image_id}")
        try:
            image_url = get_image_url_fn(image_id)
        except Exception as e:
            logger.error(f"Failed to fetch image URL for {image_id}: {str(e)}")
            raise ValueError(f"Failed to fetch image URL: {str(e)}")
    
    # Build input_data for agent
    input_data = {
        "item_id": str(image_id) if image_id else "unknown",
        "input_mode": input_mode,
    }
    
    if image_url:
        input_data["image_url"] = image_url
    
    # Handle text input: prefer text_input (string), otherwise convert text_metadata (dict) to stringified JSON
    if text_input:
        # Pre-formatted text string takes precedence
        input_data["text_input"] = text_input
    elif text_metadata:
        # Convert JSON dict to stringified JSON for LLM (preserves structure)
        # Filter out None/empty values
        filtered_metadata = {k: v for k, v in text_metadata.items() if v is not None and str(v).strip()}
        if filtered_metadata:
            input_data["text_input"] = json.dumps(filtered_metadata, indent=2)
            logger.debug(f"Converted text_metadata to stringified JSON: {len(filtered_metadata)} fields")
    
    logger.info(f"Agent input - image_url: {bool(image_url)}, text_input: {bool(text_input)}, input_mode: {input_mode}")
    
    # Initialize agent
    logger.info(f"Initializing agent with model: {full_config['model_config'].get('model')}")
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    
    # Execute agent
    logger.info("Executing agent...")
    agent_start = time.time()
    agent_result = agent.execute(input_data=input_data)
    agent_elapsed = time.time() - agent_start
    logger.info(f"Agent execution completed in {agent_elapsed:.2f}s - Status: {agent_result.status.value}")
    
    # Process result
    total_elapsed = time.time() - start_time
    
    if agent_result.status == AgentStatus.SUCCESS:
        # Extract classification result
        classification_data = extract_classification_result(
            agent_result.result,
            schema=agent_result.schema
        )
        
        # Handle "Unknown" results (ID 0) - return success but with null extraction
        # This indicates the extraction worked but no valid classification was found
        primary_name = classification_data.get("primary_name")
        primary_id = classification_data.get("primary_id")
        is_unknown_result = primary_id == 0 or (primary_name and primary_name.lower() == "unknown")
        
        if is_unknown_result:
            logger.info(f"Classification returned 'Unknown' result - returning success with null extraction")
            result = {
                "config_id": config_id,
                "primary": None,
                "primary_id": None,
                "primary_name": None,
                "alternatives": [],
                "alternative_ids": [],
                "alternative_names": [],
                "confidence": 0.0,
                "reasoning": classification_data.get("reasoning") or "No valid classification found",
                "success": True,  # Still success - extraction worked, just no valid result
                "status": "no_match",
                "processing_time_seconds": round(total_elapsed, 3),
                "validation_passed": True,
                "no_extraction": True,  # Flag to indicate intentionally null extraction
            }
            
            # Set root fields to None for material/model
            if 'material' in config_id.lower():
                result['root_material_name'] = None
                result['root_material_id'] = None
            elif 'model' in config_id.lower():
                result['root_model_name'] = None
                result['root_model_id'] = None
            
            logger.info(f"Returning null extraction result for 'Unknown' classification")
            return result
        
        result = {
            "config_id": config_id,
            "primary": classification_data.get("primary"),
            "primary_id": classification_data.get("primary_id"),
            "primary_name": classification_data.get("primary_name"),
            "alternatives": classification_data.get("alternatives", []),
            "alternative_ids": classification_data.get("alternative_ids", []),
            "alternative_names": classification_data.get("alternative_names", []),
            "confidence": classification_data.get("confidence", 0.0),
            "reasoning": classification_data.get("reasoning"),
            "success": True,
            "status": agent_result.status.value,
            "processing_time_seconds": round(total_elapsed, 3),
            "validation_passed": agent_result.validation_info.is_valid if agent_result.validation_info else True,
        }

        # Perform root lookup for model and material classifications
        primary_id = classification_data.get("primary_id")
        if primary_id:
            # Detect property type from config_id
            if 'material' in config_id.lower():
                logger.info(f"Performing root lookup for material ID: {primary_id}")
                root_lookup = lookup_material_root(
                    material_id=primary_id,
                    category='bags'  # Could extract from config_id if needed
                )
                result['root_material_name'] = root_lookup['root_material_name']
                result['root_material_id'] = root_lookup['root_material_id']
                if root_lookup['error_logs']:
                    result['root_lookup_errors'] = root_lookup['error_logs']
                    logger.warning(f"Root material lookup errors: {root_lookup['error_logs']}")
            elif 'model' in config_id.lower():
                logger.info(f"Performing root lookup for model ID: {primary_id}")
                root_lookup = lookup_model_root(
                    model_id=primary_id,
                    category='bags'  # Could extract from config_id if needed
                )
                result['root_model_name'] = root_lookup['root_model_name']
                result['root_model_id'] = root_lookup['root_model_id']
                if root_lookup['error_logs']:
                    result['root_lookup_errors'] = root_lookup['error_logs']
                    logger.warning(f"Root model lookup errors: {root_lookup['error_logs']}")
        else:
            # No primary_id - set root fields to None if material or model
            if 'material' in config_id.lower():
                result['root_material_name'] = None
                result['root_material_id'] = None
            elif 'model' in config_id.lower():
                result['root_model_name'] = None
                result['root_model_id'] = None

        logger.info(f"Classification successful - primary: {classification_data.get('primary_name')} (ID: {classification_data.get('primary_id')})")
        return result
    else:
        # Handle failure
        error_msg = agent_result.error_report.message if agent_result.error_report else "Unknown error"
        
        # Try to extract partial result if available
        classification_data = None
        if agent_result.result:
            classification_data = extract_classification_result(
                agent_result.result,
                schema=agent_result.schema
            )
        
        result = {
            "config_id": config_id,
            "primary": classification_data.get("primary") if classification_data else None,
            "primary_id": classification_data.get("primary_id") if classification_data else None,
            "primary_name": classification_data.get("primary_name") if classification_data else None,
            "alternatives": classification_data.get("alternatives", []) if classification_data else [],
            "alternative_ids": classification_data.get("alternative_ids", []) if classification_data else [],
            "alternative_names": classification_data.get("alternative_names", []) if classification_data else [],
            "confidence": classification_data.get("confidence", 0.0) if classification_data else 0.0,
            "reasoning": classification_data.get("reasoning") if classification_data else None,
            "success": False,
            "status": agent_result.status.value,
            "error": error_msg,
            "error_type": agent_result.error_report.error_type if agent_result.error_report else "unknown",
            "processing_time_seconds": round(total_elapsed, 3),
            "validation_passed": agent_result.validation_info.is_valid if agent_result.validation_info else False,
        }

        # Set root fields to None for failed classifications
        if 'material' in config_id.lower():
            result['root_material_name'] = None
            result['root_material_id'] = None
        elif 'model' in config_id.lower():
            result['root_model_name'] = None
            result['root_model_id'] = None
        
        logger.error(f"Classification failed: {error_msg}")
        return result
