"""
Lightweight API Handler for Agent Orchestration Functions

This module provides a minimal interface between Lambda API requests
and agent orchestration functions. It should:
- Call orchestration functions from agent_orchestration/
- Apply minimal input validation
- Apply minimal output transformation (if needed for API contract)
- NOT directly call agent_architecture
- NOT load configs or initialize agents

All transformation, config loading, and agent initialization should
be handled by the orchestration functions.
"""

import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import Dict, Any, Optional, Callable, List, Tuple

from stage_urls import get_stage
from core.utils.credentials import ensure_gcp_adc
from core.utils.image_service import get_signed_image_url

logger = logging.getLogger(__name__)

# Default batch processing settings
DEFAULT_MAX_WORKERS = 200


def detect_input_mode(api_input: Dict[str, Any]) -> str:
    """
    Detect input format from payload: image-only, text-only, or multimodal.

    - Image included, text not (or empty) -> image-only
    - Image not included, text present and non-empty -> text-only
    - Both present and non-empty -> multimodal

    Args:
        api_input: Request payload with image, image_url, text_input, text_metadata, etc.

    Returns:
        One of: "image-only", "text-only", "multimodal"
    """
    has_image = bool(
        api_input.get("image") or
        (api_input.get("image_url") and str(api_input.get("image_url", "")).strip())
    )
    text_input = api_input.get("text_input") or api_input.get("text_dump")
    has_text = bool(text_input and str(text_input).strip()) if isinstance(text_input, str) else False
    if not has_text and api_input.get("text_metadata"):
        has_text = bool(api_input.get("text_metadata"))
    if not has_text:
        has_text = any(
            api_input.get(f) and str(api_input.get(f)).strip()
            for f in ("brand", "title", "description")
        )

    if has_image and has_text:
        return "multimodal"
    if has_image:
        return "image-only"
    if has_text:
        return "text-only"
    return "text-only"  # Default when neither present


def get_config_id_for_input_mode(base_config_id: str, input_mode: str) -> str:
    """
    Resolve config_id for a given input mode.

    - text-only: classifier-{property}-{category}-text (e.g. classifier-colour-bags-text)
    - image-only or multimodal: classifier-{property}-{category} (e.g. classifier-colour-bags)

    Args:
        base_config_id: Base config ID (e.g. classifier-colour-bags), optionally with -text suffix.
        input_mode: One of "text-only", "image-only", "multimodal".

    Returns:
        Config ID to use for loading the agent.
    """
    base = base_config_id.rstrip("-text") if base_config_id.endswith("-text") else base_config_id
    if input_mode == "text-only":
        return f"{base}-text"
    return base


def get_config_id_for_property(category: str, property_name: str) -> str:
    """
    Map category and property to base agent config_id (visual/multimodal variant).

    Use get_config_id_for_input_mode(base_config_id, input_mode) to get the
    text-only variant (classifier-{property}-{category}-text) when input_mode is text-only.

    Args:
        category: Category (e.g., 'bags')
        property_name: Property name (e.g., 'model', 'material', 'colour', 'type')

    Returns:
        Base config ID string (e.g., 'classifier-material-bags')
    """
    # Mapping from old property names to new config_ids (base = visual/multimodal)
    config_mapping = {
        "bags": {
            "model": "classifier-model-bags",
            "material": "classifier-material-bags",
            "colour": "classifier-colour-bags",
            "type": "classifier-type-bags",
            "condition": "classifier-condition-bags",
        }
    }

    category_mapping = config_mapping.get(category, {})
    config_id = category_mapping.get(property_name)

    if not config_id:
        config_id = f"{property_name}-30"
        logger.warning(f"No explicit mapping for {category}/{property_name}, using fallback: {config_id}")

    return config_id


def execute_classification_for_api(
    config_id: str,
    api_input: Dict[str, Any],
    env: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: Optional[int] = None
) -> Any:
    """
    Execute classification orchestration for API request.
    
    Supports both single-item and batch processing modes.
    - Single mode: api_input is a dict, returns a single result dict
    - Batch mode: api_input contains 'items' list, returns a list of result dicts
    
    This is a lightweight wrapper that:
    1. Ensures Lambda-specific setup (credentials)
    2. Calls orchestration function with proper parameters
    3. Returns orchestration result (minimal transformation)
    
    Args:
        config_id: Agent configuration ID
        api_input: API request input (flexible format)
            Single mode:
            - image: Image ID (optional)
            - image_url: Direct image URL (optional)
            - text_input: Pre-formatted text string (optional, takes precedence)
            - text_metadata: Text metadata as JSON dict (optional)
            - text_dump: Legacy field - treated as text_input if provided (optional)
            - brand: Brand name (optional, used to build text_metadata dict)
            - title: Title (optional, used to build text_metadata dict)
            - description: Description (optional, used to build text_metadata dict)
            - input_mode: Input mode (optional, default: "auto")
            Batch mode:
            - items: List of items (each item has same structure as single mode)
            - max_workers: Optional, maximum concurrent workers (default: 200)
            - batch_size: Optional, process in batches of this size
        env: Environment (defaults to STAGE env var)
        max_workers: Maximum concurrent workers for batch mode (default: 200)
        batch_size: Process in batches of this size (optional)
    
    Returns:
        Single mode: Classification result dict
        Batch mode: List of classification result dicts
    """
    # Check if batch mode (api_input contains 'items' list)
    if isinstance(api_input, dict) and 'items' in api_input and isinstance(api_input['items'], list):
        items = api_input['items']
        batch_max_workers = api_input.get('max_workers', max_workers)
        batch_batch_size = api_input.get('batch_size', batch_size)
        return execute_classification_batch_for_api(
            config_id=config_id,
            items=items,
            max_workers=batch_max_workers,
            batch_size=batch_batch_size,
            env=env
        )
    
    # Single item mode
    # Ensure GCP credentials are set up (Lambda-specific)
    ensure_gcp_adc()
    
    if env is None:
        env = get_stage()
    
    # Import orchestration function (lazy import to avoid circular dependencies)
    from agent_orchestration.classifier_api_orchestration import classify_for_api
    
    # Extract API input fields and detect input_mode if not provided
    image_id = api_input.get("image")
    image_url = api_input.get("image_url")
    text_input = api_input.get("text_input")
    input_mode = api_input.get("input_mode") or detect_input_mode(api_input)

    # Handle text_metadata: prefer explicit text_metadata dict, otherwise build from individual fields
    # text_dump is treated as text_input for backward compatibility
    if api_input.get("text_dump") and not text_input:
        text_input = api_input.get("text_dump")
    
    text_metadata = api_input.get("text_metadata")
    if not text_metadata and not text_input:
        # Build text_metadata dict from individual fields if not provided
        text_metadata = {}
        for field in ["brand", "title", "description"]:
            value = api_input.get(field)
            if value and str(value).strip():
                text_metadata[field] = value
    
    # Call orchestration function
    # Pass get_signed_image_url as a function for Lambda-specific image URL fetching
    result = classify_for_api(
        config_id=config_id,
        image_id=image_id,
        image_url=image_url,
        text_input=text_input,
        text_metadata=text_metadata if text_metadata else None,
        input_mode=input_mode,
        env=env,
        get_image_url_fn=get_signed_image_url  # Lambda-specific utility
    )
    
    return result


def format_classification_for_legacy_api(
    result: Dict[str, Any],
    target: str
) -> Dict[str, Any]:
    """
    Format classification result for legacy API contract.
    
    This applies minimal transformation to match the old API response format.
    The orchestration function already provides both IDs and names, so this
    just selects the appropriate fields for the legacy format.
    
    Args:
        result: Classification result from orchestration function
        target: Property name (e.g., 'model', 'material', 'colour')
    
    Returns:
        Formatted result matching legacy API contract
    """
    # Preserve critical fields for frontend error detection
    base_fields = {
        "success": result.get("success", True),
        "validation_passed": result.get("validation_passed", True),
        "status": result.get("status", ""),
        "error_type": result.get("error_type", ""),
        "error": result.get("error", ""),
        "primary_name": result.get("primary_name", ""),
        "primary_id": result.get("primary_id"),
        "reasoning": result.get("reasoning", ""),
    }

    if target == "model":
        return {
            **base_fields,
            "model": result.get("primary_name") or result.get("primary"),
            "model_id": result.get("primary_id"),
            "root_model": result.get("root_model_name"),
            "root_model_id": result.get("root_model_id"),
            "confidence": result.get("confidence", 0.0),
            "root_lookup_errors": result.get("root_lookup_errors", [])
        }
    elif target == "material":
        return {
            **base_fields,
            "material": result.get("primary_name") or result.get("primary"),
            "material_id": result.get("primary_id"),
            "root_material": result.get("root_material_name"),
            "root_material_id": result.get("root_material_id"),
            "confidence": result.get("confidence", 0.0),
            "root_lookup_errors": result.get("root_lookup_errors", [])
        }
    elif target == "hardware":
        return {
            **base_fields,
            "hardware": result.get("primary_name") or result.get("primary"),
            "hardware_id": result.get("primary_id"),
            "confidence": result.get("confidence", 0.0),
        }
    elif target == "brand":
        return {
            **base_fields,
            "brand": result.get("primary_name") or result.get("brand") or result.get("final_brand"),
            "brand_id": result.get("primary_id") or result.get("brand_id") or result.get("final_brand_id"),
            "confidence": result.get("confidence", 0.0),
        }
    else:
        return {
            **base_fields,
            target: result.get("primary_name") or result.get("primary"),
            "confidence": result.get("confidence", 0.0),
        }


def execute_brand_classification_for_api(
    api_input: Dict[str, Any],
    env: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute brand classification via brand_classification_orchestration (Agent1 -> BigQuery -> Agent2).
    Single-item only; batch mode can be added later.
    """
    ensure_gcp_adc()
    if env is None:
        env = get_stage()

    from agent_orchestration.brand_classification_orchestration import run_brand_classification_workflow

    text_input = api_input.get("text_input") or api_input.get("text_dump")
    if not text_input and (api_input.get("title") or api_input.get("description")):
        text_input = " ".join(
            str(p) for p in (api_input.get("title", ""), api_input.get("description", "")) if p
        ).strip()
    if not text_input:
        brand_hint = api_input.get("brand")
        if brand_hint and str(brand_hint).strip():
            text_input = str(brand_hint).strip()
        else:
            other = " ".join(
                str(v) for k, v in api_input.items()
                if v and k not in ("image", "image_url", "input_mode") and isinstance(v, (str, int, float))
            ).strip()
            if other:
                text_input = other
    if not text_input:
        raise ValueError("'text_input' or 'text_dump' or (title and/or description) or 'brand' is required for brand classification")

    name = api_input.get("name") or api_input.get("title")

    result = run_brand_classification_workflow(
        raw_text=text_input,
        name=name,
        env=env,
        verbose=False,
    )

    if result.get("workflow_status") != "success":
        return {
            "primary_name": result.get("final_brand") or "",
            "primary_id": result.get("final_brand_id"),
            "confidence": 0.0,
            "reasoning": result.get("error", "Brand workflow failed"),
        }

    agent2 = result.get("agent2_result") or {}
    return {
        "primary_name": result.get("final_brand") or agent2.get("brand_name"),
        "primary_id": result.get("final_brand_id") or agent2.get("prediction_id"),
        "confidence": result.get("confidence") or agent2.get("confidence", 0.0),
        "reasoning": agent2.get("reasoning", ""),
        "final_brand": result.get("final_brand"),
        "final_brand_id": result.get("final_brand_id"),
    }


def execute_model_classification_for_api(
    api_input: Dict[str, Any],
    env: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute model classification via classifier_model_orchestration (brand-specific LLM config).
    Used when the request has text only (no image). When image is present, use the vector
    classifier (classify_model) instead.
    """
    ensure_gcp_adc()
    if env is None:
        env = get_stage()

    from agent_orchestration.classifier_model_orchestration import get_model_config_id

    brand = api_input.get("brand")
    if not brand or not str(brand).strip():
        raise ValueError("'brand' is required for model classification (text-only path)")

    text_input = api_input.get("text_input") or api_input.get("text_dump")
    if not text_input and (api_input.get("title") or api_input.get("description")):
        text_input = " ".join(
            str(p) for p in (api_input.get("title", ""), api_input.get("description", "")) if p
        ).strip()
    if not text_input:
        raise ValueError("'text_input' or 'text_dump' or (title and/or description) is required for text-only model classification")

    config_id = get_model_config_id(str(brand).strip())
    api_input_text_only = {
        **api_input,
        "input_mode": "text-only",
        "text_input": text_input,
        "text_dump": text_input,
    }
    return execute_classification_for_api(
        config_id=config_id,
        api_input=api_input_text_only,
        env=env,
    )


def execute_size_classification_for_api(
    api_input: Dict[str, Any],
    env: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: Optional[int] = None
) -> Any:
    """
    Execute size classification orchestration for API request.
    
    Supports both single-item and batch processing modes.
    - Single mode: api_input is a dict, returns a single result dict
    - Batch mode: api_input contains 'items' list, returns a list of result dicts
    
    This wrapper calls the size classification workflow which requires:
    - raw_text: Text input for size classification
    - model_id: Model ID to filter size options (required)
    
    Args:
        api_input: API request input
            Single mode:
            - text_input: Text input (required)
            - text_dump: Legacy field - treated as text_input if provided (optional)
            - model_id: Model ID (required) - used to filter size options from BigQuery
            - text_metadata: Text metadata as JSON dict (optional, will be formatted to text)
            Batch mode:
            - items: List of items (each item has same structure as single mode)
            - max_workers: Optional, maximum concurrent workers (default: 200)
            - batch_size: Optional, process in batches of this size
        env: Environment (defaults to STAGE env var)
        max_workers: Maximum concurrent workers for batch mode (default: 200)
        batch_size: Process in batches of this size (optional)
    
    Returns:
        Single mode: Size classification result dict
        Batch mode: List of size classification result dicts
    """
    # Check if batch mode (api_input contains 'items' list)
    if isinstance(api_input, dict) and 'items' in api_input and isinstance(api_input['items'], list):
        items = api_input['items']
        batch_max_workers = api_input.get('max_workers', max_workers)
        batch_batch_size = api_input.get('batch_size', batch_size)
        return execute_size_classification_batch_for_api(
            items=items,
            max_workers=batch_max_workers,
            batch_size=batch_batch_size,
            env=env
        )
    
    # Single item mode
    # Ensure GCP credentials are set up (Lambda-specific)
    ensure_gcp_adc()
    
    if env is None:
        env = get_stage()
    
    # Import orchestration function (lazy import to avoid circular dependencies)
    from agent_orchestration.model_size_classification_orchestration import run_model_size_classification_workflow
    
    # Extract text input
    text_input = api_input.get("text_input")
    if api_input.get("text_dump") and not text_input:
        text_input = api_input.get("text_dump")
    
    # Build text from text_metadata if provided
    text_metadata = api_input.get("text_metadata")
    if text_metadata and not text_input:
        # Format text_metadata into a text string
        parts = []
        for key, value in text_metadata.items():
            if value and str(value).strip():
                parts.append(f"{key.replace('_', ' ').title()}: {value}")
        if parts:
            text_input = "\n".join(parts)
    
    if not text_input:
        raise ValueError("'text_input' or 'text_dump' or 'text_metadata' is required for size classification")
    
    # Extract model_id (required)
    model_id = api_input.get("model_id")
    if model_id is None:
        raise ValueError("'model_id' is required for size classification")
    
    try:
        model_id_int = int(model_id)
    except (ValueError, TypeError):
        raise ValueError(f"'model_id' must be an integer, got: {model_id}")
    
    # Call orchestration function
    result = run_model_size_classification_workflow(
        raw_text=text_input,
        model_id=model_id_int,
        env=env
    )
    
    # Transform result to match API format
    if result.get("workflow_status") == "success":
        final_result = result.get("final_result", {})
        if final_result.get("success"):
            return {
                "size": final_result.get("size", ""),
                "size_id": final_result.get("prediction_id"),
                "confidence": final_result.get("confidence", 0.0),
                "reasoning": final_result.get("reasoning", ""),
                "success": True,
                "workflow_status": "success"
            }
        else:
            return {
                "size": "",
                "size_id": None,
                "confidence": 0.0,
                "reasoning": final_result.get("reasoning", ""),
                "success": False,
                "workflow_status": "failed",
                "error": final_result.get("error", "Size classification failed")
            }
    else:
        return {
            "size": "",
            "size_id": None,
            "confidence": 0.0,
            "reasoning": "",
            "success": False,
            "workflow_status": "failed",
            "error": result.get("error", "Size workflow failed")
        }


def execute_keyword_classification_for_api(
    api_input: Dict[str, Any],
    env: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: Optional[int] = None
) -> Any:
    """
    Execute keyword classification orchestration for API request.
    
    Supports both single-item and batch processing modes.
    - Single mode: api_input is a dict, returns a single result dict
    - Batch mode: api_input contains 'items' list, returns a list of result dicts
    
    This wrapper calls the keyword classification workflow which requires:
    - general_input_text: Dict with text fields to process (e.g., {"title": "...", "description": "..."})
    - text_to_avoid: Dict with classifications to avoid (e.g., {"material": "Leather", "condition": "Excellent"})
    
    Args:
        api_input: API request input
            Single mode:
            - general_input_text: Dict with text fields (required)
            - text_to_avoid: Dict with classifications to avoid (required)
            - item_id: Item identifier (optional, default: "1")
            Batch mode:
            - items: List of items (each item has same structure as single mode)
            - max_workers: Optional, maximum concurrent workers (default: 200)
            - batch_size: Optional, process in batches of this size
        env: Environment (defaults to STAGE env var)
        max_workers: Maximum concurrent workers for batch mode (default: 200)
        batch_size: Process in batches of this size (optional)
    
    Returns:
        Single mode: Keyword classification result dict
        Batch mode: List of keyword classification result dicts
    """
    # Check if batch mode (api_input contains 'items' list)
    if isinstance(api_input, dict) and 'items' in api_input and isinstance(api_input['items'], list):
        items = api_input['items']
        batch_max_workers = api_input.get('max_workers', max_workers)
        batch_batch_size = api_input.get('batch_size', batch_size)
        return execute_keyword_classification_batch_for_api(
            items=items,
            max_workers=batch_max_workers,
            batch_size=batch_batch_size,
            env=env
        )
    
    # Single item mode
    # Ensure GCP credentials are set up (Lambda-specific)
    ensure_gcp_adc()
    
    if env is None:
        env = get_stage()
    
    # Import orchestration function (lazy import to avoid circular dependencies)
    from agent_orchestration.keyword_classifier_orchestration import run_keyword_classification
    from agent_orchestration.csv_config_loader import ConfigLoader
    
    # Extract required inputs
    general_input_text = api_input.get("general_input_text")
    if not general_input_text:
        raise ValueError("'general_input_text' is required for keyword classification")
    
    text_to_avoid = api_input.get("text_to_avoid")
    if text_to_avoid is None:
        # Allow empty dict but require the field
        text_to_avoid = {}
    
    item_id = api_input.get("item_id", "1")
    
    # Load config
    config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
    full_config = config_loader.load_full_agent_config('classifier-keywords-bags')
    
    # Call orchestration function
    result = run_keyword_classification(
        general_input_text=general_input_text,
        text_to_avoid=text_to_avoid,
        full_config=full_config,
        item_id=item_id
    )
    
    # Transform result to match API format
    if result.get("success"):
        keywords = result.get("keywords", [])
        # Keywords are list of dicts with 'keyword' and 'confidence' keys
        keyword_1 = keywords[0].get("keyword", "") if len(keywords) > 0 else ""
        keyword_2 = keywords[1].get("keyword", "") if len(keywords) > 1 else ""
        keyword_3 = keywords[2].get("keyword", "") if len(keywords) > 2 else ""
        keyword_1_confidence = keywords[0].get("confidence", 0.0) if len(keywords) > 0 else 0.0
        keyword_2_confidence = keywords[1].get("confidence", 0.0) if len(keywords) > 1 else 0.0
        keyword_3_confidence = keywords[2].get("confidence", 0.0) if len(keywords) > 2 else 0.0
        
        return {
            "keywords": keywords,
            "keyword_count": len(keywords),
            "keyword_1": keyword_1,
            "keyword_2": keyword_2,
            "keyword_3": keyword_3,
            "keyword_1_confidence": keyword_1_confidence,
            "keyword_2_confidence": keyword_2_confidence,
            "keyword_3_confidence": keyword_3_confidence,
            "reasoning": result.get("reasoning", ""),
            "success": True,
            "processing_time_seconds": result.get("processing_time_seconds", 0.0)
        }
    else:
        return {
            "keywords": [],
            "keyword_count": 0,
            "keyword_1": "",
            "keyword_2": "",
            "keyword_3": "",
            "keyword_1_confidence": 0.0,
            "keyword_2_confidence": 0.0,
            "keyword_3_confidence": 0.0,
            "reasoning": result.get("reasoning", ""),
            "success": False,
            "error": result.get("error", "Keyword classification failed"),
            "processing_time_seconds": result.get("processing_time_seconds", 0.0)
        }


def execute_hardware_classification_for_api(
    api_input: Dict[str, Any],
    env: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: Optional[int] = None
) -> Any:
    """
    Execute hardware classification orchestration for API request.
    
    Supports both single-item and batch processing modes.
    - Single mode: api_input is a dict, returns a single result dict
    - Batch mode: api_input contains 'items' list, returns a list of result dicts
    
    This wrapper calls the standard classification API with hardware config_id.
    Hardware classification uses the same pattern as other property classifiers.
    
    Args:
        api_input: API request input (same format as other property classifiers)
            Single mode:
            - image: Image ID (optional)
            - image_url: Direct image URL (optional)
            - text_input: Pre-formatted text string (optional, takes precedence)
            - text_metadata: Text metadata as JSON dict (optional)
            - text_dump: Legacy field - treated as text_input if provided (optional)
            - brand: Brand name (optional, used to build text_metadata dict)
            - title: Title (optional, used to build text_metadata dict)
            - description: Description (optional, used to build text_metadata dict)
            - input_mode: Input mode (optional, default: "auto")
            Batch mode:
            - items: List of items (each item has same structure as single mode)
            - max_workers: Optional, maximum concurrent workers (default: 200)
            - batch_size: Optional, process in batches of this size
        env: Environment (defaults to STAGE env var)
        max_workers: Maximum concurrent workers for batch mode (default: 200)
        batch_size: Process in batches of this size (optional)
    
    Returns:
        Single mode: Hardware classification result dict
        Batch mode: List of hardware classification result dicts
    """
    # Check if batch mode (api_input contains 'items' list)
    if isinstance(api_input, dict) and 'items' in api_input and isinstance(api_input['items'], list):
        items = api_input['items']
        batch_max_workers = api_input.get('max_workers', max_workers)
        batch_batch_size = api_input.get('batch_size', batch_size)
        return execute_hardware_classification_batch_for_api(
            items=items,
            max_workers=batch_max_workers,
            batch_size=batch_batch_size,
            env=env
        )
    
    # Single item mode
    # Use the standard classification API handler with hardware config_id
    return execute_classification_for_api(
        config_id="classifier-hardware-bags",
        api_input=api_input,
        env=env
    )


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================

def process_items_parallel(
    items: List[Dict[str, Any]],
    process_func: Callable[[Dict[str, Any], int], Dict[str, Any]],
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Process a list of items in parallel using ThreadPoolExecutor.
    
    Args:
        items: List of input items (each item is a dict)
        process_func: Function that takes (item, idx) and returns result dict
        max_workers: Maximum number of concurrent workers
        batch_size: If provided, process in batches (useful for large datasets)
        
    Returns:
        List of result dictionaries in same order as input items
    """
    total = len(items)
    results = [None] * total  # Pre-allocate to maintain order
    
    # Semaphore to limit concurrent API calls
    semaphore = Semaphore(max_workers)
    
    def process_with_semaphore(idx: int, item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Wrapper to limit concurrent executions"""
        with semaphore:
            try:
                result = process_func(item, idx)
                return (idx, result)
            except Exception as e:
                # Return error result
                logger.error(f"Error processing item {idx}: {str(e)}")
                return (idx, {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'success': False
                })
    
    # Process in batches if batch_size is specified
    if batch_size:
        logger.info(f"Processing {total} items in batches of {batch_size}...")
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_items = items[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1} (items {batch_start+1}-{batch_end})...")
            batch_start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_items))) as executor:
                futures = {
                    executor.submit(process_with_semaphore, batch_start + idx, item): (batch_start + idx, item)
                    for idx, item in enumerate(batch_items)
                }
                
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch completed in {batch_time:.2f}s ({len(batch_items)/batch_time:.2f} items/sec)")
    else:
        # Process all items in parallel
        logger.info(f"Processing {total} items in parallel (max {max_workers} concurrent)...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
            futures = {
                executor.submit(process_with_semaphore, idx, item): (idx, item)
                for idx, item in enumerate(items)
            }
            
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
        
        elapsed = time.time() - start_time
        logger.info(f"All items processed in {elapsed:.2f}s ({total/elapsed:.2f} items/sec)")
    
    return results


# ============================================================================
# BATCH PROCESSING FUNCTIONS FOR ORCHESTRATORS
# ============================================================================

def execute_size_classification_batch_for_api(
    items: List[Dict[str, Any]],
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: Optional[int] = None,
    env: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute size classification in batch mode (parallel processing).
    
    Args:
        items: List of API input items (each item has same structure as single-item input)
        max_workers: Maximum number of concurrent workers
        batch_size: If provided, process in batches
        env: Environment (defaults to STAGE env var)
    
    Returns:
        List of size classification results in same order as input items
    """
    # Ensure GCP credentials are set up (Lambda-specific)
    ensure_gcp_adc()
    
    if env is None:
        env = get_stage()
    
    # Import orchestration function (lazy import to avoid circular dependencies)
    from agent_orchestration.model_size_classification_orchestration import run_model_size_classification_workflow
    
    def process_single_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Process a single item for size classification"""
        # Extract text input
        text_input = item.get("text_input")
        if item.get("text_dump") and not text_input:
            text_input = item.get("text_dump")
        
        # Build text from text_metadata if provided
        text_metadata = item.get("text_metadata")
        if text_metadata and not text_input:
            parts = []
            for key, value in text_metadata.items():
                if value and str(value).strip():
                    parts.append(f"{key.replace('_', ' ').title()}: {value}")
            if parts:
                text_input = "\n".join(parts)
        
        if not text_input:
            return {
                "size": "",
                "size_id": None,
                "confidence": 0.0,
                "reasoning": "No text input provided",
                "success": False,
                "workflow_status": "failed",
                "error": "'text_input' or 'text_dump' or 'text_metadata' is required"
            }
        
        # Extract model_id (required)
        model_id = item.get("model_id")
        if model_id is None:
            return {
                "size": "",
                "size_id": None,
                "confidence": 0.0,
                "reasoning": "model_id is required",
                "success": False,
                "workflow_status": "failed",
                "error": "'model_id' is required for size classification"
            }
        
        try:
            model_id_int = int(model_id)
        except (ValueError, TypeError):
            return {
                "size": "",
                "size_id": None,
                "confidence": 0.0,
                "reasoning": f"Invalid model_id: {model_id}",
                "success": False,
                "workflow_status": "failed",
                "error": f"'model_id' must be an integer, got: {model_id}"
            }
        
        # Call orchestration function
        try:
            result = run_model_size_classification_workflow(
                raw_text=text_input,
                model_id=model_id_int,
                env=env
            )
            
            # Transform result to match API format
            if result.get("workflow_status") == "success":
                final_result = result.get("final_result", {})
                if final_result.get("success"):
                    return {
                        "size": final_result.get("size", ""),
                        "size_id": final_result.get("prediction_id"),
                        "confidence": final_result.get("confidence", 0.0),
                        "reasoning": final_result.get("reasoning", ""),
                        "success": True,
                        "workflow_status": "success"
                    }
                else:
                    return {
                        "size": "",
                        "size_id": None,
                        "confidence": 0.0,
                        "reasoning": final_result.get("reasoning", ""),
                        "success": False,
                        "workflow_status": "failed",
                        "error": final_result.get("error", "Size classification failed")
                    }
            else:
                return {
                    "size": "",
                    "size_id": None,
                    "confidence": 0.0,
                    "reasoning": "",
                    "success": False,
                    "workflow_status": "failed",
                    "error": result.get("error", "Size workflow failed")
                }
        except Exception as e:
            logger.error(f"Error in size classification for item {idx}: {str(e)}")
            return {
                "size": "",
                "size_id": None,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "success": False,
                "workflow_status": "failed",
                "error": str(e)
            }
    
    return process_items_parallel(
        items=items,
        process_func=process_single_item,
        max_workers=max_workers,
        batch_size=batch_size
    )


def execute_keyword_classification_batch_for_api(
    items: List[Dict[str, Any]],
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: Optional[int] = None,
    env: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute keyword classification in batch mode (parallel processing).
    
    Args:
        items: List of API input items (each item has same structure as single-item input)
        max_workers: Maximum number of concurrent workers
        batch_size: If provided, process in batches
        env: Environment (defaults to STAGE env var)
    
    Returns:
        List of keyword classification results in same order as input items
    """
    # Ensure GCP credentials are set up (Lambda-specific)
    ensure_gcp_adc()
    
    if env is None:
        env = get_stage()
    
    # Import orchestration function (lazy import to avoid circular dependencies)
    from agent_orchestration.keyword_classifier_orchestration import run_keyword_classification
    from agent_orchestration.csv_config_loader import ConfigLoader
    
    # Load config once for all items
    config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
    full_config = config_loader.load_full_agent_config('classifier-keywords-bags')
    
    def process_single_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Process a single item for keyword classification"""
        # Extract required inputs
        general_input_text = item.get("general_input_text")
        if not general_input_text:
            return {
                "keywords": [],
                "keyword_count": 0,
                "keyword_1": "",
                "keyword_2": "",
                "keyword_3": "",
                "keyword_1_confidence": 0.0,
                "keyword_2_confidence": 0.0,
                "keyword_3_confidence": 0.0,
                "reasoning": "general_input_text is required",
                "success": False,
                "error": "'general_input_text' is required for keyword classification"
            }
        
        text_to_avoid = item.get("text_to_avoid")
        if text_to_avoid is None:
            text_to_avoid = {}
        
        item_id = item.get("item_id", str(idx + 1))
        
        # Call orchestration function
        try:
            result = run_keyword_classification(
                general_input_text=general_input_text,
                text_to_avoid=text_to_avoid,
                full_config=full_config,
                item_id=item_id
            )
            
            # Transform result to match API format
            if result.get("success"):
                keywords = result.get("keywords", [])
                keyword_1 = keywords[0].get("keyword", "") if len(keywords) > 0 else ""
                keyword_2 = keywords[1].get("keyword", "") if len(keywords) > 1 else ""
                keyword_3 = keywords[2].get("keyword", "") if len(keywords) > 2 else ""
                keyword_1_confidence = keywords[0].get("confidence", 0.0) if len(keywords) > 0 else 0.0
                keyword_2_confidence = keywords[1].get("confidence", 0.0) if len(keywords) > 1 else 0.0
                keyword_3_confidence = keywords[2].get("confidence", 0.0) if len(keywords) > 2 else 0.0
                
                return {
                    "keywords": keywords,
                    "keyword_count": len(keywords),
                    "keyword_1": keyword_1,
                    "keyword_2": keyword_2,
                    "keyword_3": keyword_3,
                    "keyword_1_confidence": keyword_1_confidence,
                    "keyword_2_confidence": keyword_2_confidence,
                    "keyword_3_confidence": keyword_3_confidence,
                    "reasoning": result.get("reasoning", ""),
                    "success": True,
                    "processing_time_seconds": result.get("processing_time_seconds", 0.0)
                }
            else:
                return {
                    "keywords": [],
                    "keyword_count": 0,
                    "keyword_1": "",
                    "keyword_2": "",
                    "keyword_3": "",
                    "keyword_1_confidence": 0.0,
                    "keyword_2_confidence": 0.0,
                    "keyword_3_confidence": 0.0,
                    "reasoning": result.get("reasoning", ""),
                    "success": False,
                    "error": result.get("error", "Keyword classification failed"),
                    "processing_time_seconds": result.get("processing_time_seconds", 0.0)
                }
        except Exception as e:
            logger.error(f"Error in keyword classification for item {idx}: {str(e)}")
            return {
                "keywords": [],
                "keyword_count": 0,
                "keyword_1": "",
                "keyword_2": "",
                "keyword_3": "",
                "keyword_1_confidence": 0.0,
                "keyword_2_confidence": 0.0,
                "keyword_3_confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
                "processing_time_seconds": 0.0
            }
    
    return process_items_parallel(
        items=items,
        process_func=process_single_item,
        max_workers=max_workers,
        batch_size=batch_size
    )


def execute_hardware_classification_batch_for_api(
    items: List[Dict[str, Any]],
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: Optional[int] = None,
    env: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute hardware classification in batch mode (parallel processing).
    
    Args:
        items: List of API input items (each item has same structure as single-item input)
        max_workers: Maximum number of concurrent workers
        batch_size: If provided, process in batches
        env: Environment (defaults to STAGE env var)
    
    Returns:
        List of hardware classification results in same order as input items
    """
    # Use the standard classification API handler with hardware config_id
    return execute_classification_batch_for_api(
        config_id="classifier-hardware-bags",
        items=items,
        max_workers=max_workers,
        batch_size=batch_size,
        env=env
    )


def execute_classification_batch_for_api(
    config_id: str,
    items: List[Dict[str, Any]],
    max_workers: int = DEFAULT_MAX_WORKERS,
    batch_size: Optional[int] = None,
    env: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute classification in batch mode (parallel processing) for standard property classifiers.
    
    Args:
        config_id: Agent configuration ID
        items: List of API input items (each item has same structure as single-item input)
        max_workers: Maximum number of concurrent workers
        batch_size: If provided, process in batches
        env: Environment (defaults to STAGE env var)
    
    Returns:
        List of classification results in same order as input items
    """
    # Ensure GCP credentials are set up (Lambda-specific)
    ensure_gcp_adc()
    
    if env is None:
        env = get_stage()
    
    # Import orchestration function (lazy import to avoid circular dependencies)
    from agent_orchestration.classifier_api_orchestration import classify_for_api
    
    def process_single_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Process a single item for classification"""
        # Extract API input fields and detect input_mode (config_id is base; resolve per item)
        image_id = item.get("image")
        image_url = item.get("image_url")
        text_input = item.get("text_input")
        input_mode = item.get("input_mode") or detect_input_mode(item)
        config_id_effective = get_config_id_for_input_mode(config_id, input_mode)

        # Handle text_metadata: prefer explicit text_metadata dict, otherwise build from individual fields
        if item.get("text_dump") and not text_input:
            text_input = item.get("text_dump")

        text_metadata = item.get("text_metadata")
        if not text_metadata and not text_input:
            text_metadata = {}
            for field in ["brand", "title", "description"]:
                value = item.get(field)
                if value and str(value).strip():
                    text_metadata[field] = value

        # Call orchestration function with per-item config and input_mode
        try:
            result = classify_for_api(
                config_id=config_id_effective,
                image_id=image_id,
                image_url=image_url,
                text_input=text_input,
                text_metadata=text_metadata if text_metadata else None,
                input_mode=input_mode,
                env=env,
                get_image_url_fn=get_signed_image_url
            )
            return result
        except Exception as e:
            logger.error(f"Error in classification for item {idx}: {str(e)}")
            return {
                'config_id': config_id_effective,
                'primary': "",
                'primary_id': None,
                'primary_name': "",
                'alternatives': [],
                'alternative_ids': [],
                'alternative_names': [],
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}",
                'success': False,
                'status': 'error',
                'processing_time_seconds': 0.0,
                'validation_passed': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    return process_items_parallel(
        items=items,
        process_func=process_single_item,
        max_workers=max_workers,
        batch_size=batch_size
    )


# ============================================================================
# CSV CONFIG GENERATION API HANDLER
# ============================================================================

def execute_csv_config_generation_for_api(
    api_input: Dict[str, Any],
    env: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute CSV config generation orchestration for API request.
    
    This function generates a CSV column mapping configuration by analyzing
    sample rows from a CSV file. The config is saved to DynamoDB for future use.
    
    Args:
        api_input: API request input
            - CSV_uuid: Unique identifier for the CSV (required)
            - sample_rows: List of sample row dicts (required, 5-10 rows recommended)
            - organisation_uuid: Optional organisation UUID for filtering
            - max_chars: Optional max characters for formatting (default: 1500)
        env: Environment (defaults to STAGE env var)
    
    Returns:
        Dict with:
            - CSV_uuid: The input CSV UUID
            - CSV_config_id: The DynamoDB identifier for the saved config
            - csv_config: The generated column mapping configuration
    """
    # Ensure GCP credentials are set up (Lambda-specific)
    ensure_gcp_adc()
    
    if env is None:
        env = get_stage()
    
    # Import orchestration functions (lazy import to avoid circular dependencies)
    from agent_orchestration.csv_config_orchestration import format_csv_sample_for_prompt
    from agent_orchestration.csv_config_loader import ConfigLoader
    from agent_architecture.base_agent import LLMAnnotationAgent
    from agent_architecture.validation import AgentStatus
    from datetime import datetime
    import pandas as pd
    
    # Extract required inputs
    csv_uuid = api_input.get("CSV_uuid")
    if not csv_uuid:
        raise ValueError("'CSV_uuid' is required for CSV config generation")
    
    sample_rows = api_input.get("sample_rows")
    if not sample_rows or not isinstance(sample_rows, list):
        raise ValueError("'sample_rows' is required and must be a list of row dicts")
    
    if len(sample_rows) == 0:
        raise ValueError("'sample_rows' must contain at least one row")
    
    organisation_uuid = api_input.get("organisation_uuid")
    max_chars = api_input.get("max_chars", 1500)
    
    logger.info(f"Generating CSV config for CSV_uuid: {csv_uuid} with {len(sample_rows)} sample rows")
    
    # Initialize config loader
    config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env=env)
    
    # Extract columns from sample_rows
    # Get all unique keys from all rows
    all_keys = set()
    for row in sample_rows:
        if isinstance(row, dict):
            all_keys.update(row.keys())
    columns = sorted(list(all_keys))
    
    logger.info(f"Extracted {len(columns)} columns from sample rows")
    
    # Calculate total chars for formatting
    total_chars = sum(len(json.dumps(row, default=str)) for row in sample_rows)
    
    # Check for existing matching config in DynamoDB
    logger.info("Checking for existing matching configs in DynamoDB...")
    matching_config = config_loader.find_matching_csv_config(
        csv_columns=columns,
        organisation_uuid=organisation_uuid
    )
    
    if matching_config:
        csv_config_id = matching_config.get('csv_config_identifier')
        csv_config = matching_config.get('csv_column_metadata_mappings', {})
        logger.info(f"Found existing matching config: {csv_config_id}")
        
        return {
            "CSV_uuid": csv_uuid,
            "CSV_config_id": csv_config_id,
            "csv_config": csv_config
        }
    
    # Load agent config
    logger.info("Loading agent config: inventory_csv_config_generator")
    full_config = config_loader.load_full_agent_config('inventory_csv_config_generator')
    
    # Format prompt input
    csv_sample_text = format_csv_sample_for_prompt(columns, sample_rows, total_chars)
    
    # Initialize agent with full config
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    
    # Call agent with execute()
    logger.info("Analyzing CSV structure with agent...")
    input_data = {
        "input_text": csv_sample_text
    }
    context = {"csv_columns": columns}
    agent_result = agent.execute(input_data=input_data, context=context)
    
    # Handle result
    if agent_result.status == AgentStatus.SUCCESS:
        config_output = agent_result.result
        logger.info("Config generated successfully")
        
        # Generate CSV config identifier using CSV_uuid and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_config_id = f"{csv_uuid}_config_{timestamp}"
        
        # Save to DynamoDB
        try:
            saved_config = config_loader.save_csv_config(
                csv_config_identifier=csv_config_id,
                csv_columns=columns,
                csv_column_metadata_mappings=config_output,
                organisation_uuid=organisation_uuid
            )
            logger.info(f"Config saved to DynamoDB: {csv_config_id}")
            
            return {
                "CSV_uuid": csv_uuid,
                "CSV_config_id": csv_config_id,
                "csv_config": config_output
            }
        except Exception as e:
            logger.error(f"Failed to save config to DynamoDB: {e}")
            # Still return the config even if save failed
            return {
                "CSV_uuid": csv_uuid,
                "CSV_config_id": None,  # Indicates save failed
                "csv_config": config_output,
                "error": f"Config generated but failed to save: {str(e)}"
            }
    else:
        error_msg = agent_result.error_report.message if agent_result.error_report else "Unknown error"
        logger.error(f"CSV config generation failed: {error_msg}")
        
        if agent_result.validation_info and agent_result.validation_info.errors:
            error_details = [err.message for err in agent_result.validation_info.errors]
            error_msg = f"{error_msg}. Validation errors: {', '.join(error_details)}"
        
        raise RuntimeError(f"CSV config generation failed: {error_msg}")
