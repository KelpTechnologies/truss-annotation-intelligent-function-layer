"""
Model classification API handler using vector similarity search.
"""

import json
import os
import logging
import time

from agent_utils.dsl_api_client import DSLAPIClient
from stage_urls import get_dsl_url, get_stage
from core.utils.config import MIN_MODEL_CONFIDENCE_THRESHOLD, CATEGORY_CONFIG
from core.utils.credentials import ensure_pinecone_api_key, get_request_auth_headers
from core.utils.taxonomy import lookup_root_from_child

logger = logging.getLogger(__name__)

# Import vector classifier pipeline
_classify_image = None

def _load_vector_classifier():
    """Lazy load the vector classifier pipeline."""
    global _classify_image
    if _classify_image is not None:
        return _classify_image
    
    try:
        import importlib.util
        import sys
        
        # Updated path: from core/orchestration_api_handlers/ to vector-classifiers/
        pipeline_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "vector-classifiers",
            "model_classifier_pipeline.py"
        )
        
        if os.path.exists(pipeline_path):
            package_dir = os.path.dirname(pipeline_path)
            if package_dir not in sys.path:
                sys.path.insert(0, package_dir)

            spec = importlib.util.spec_from_file_location("model_classifier_pipeline", pipeline_path)
            pipeline_module = importlib.util.module_from_spec(spec)
            sys.modules["model_classifier_pipeline"] = pipeline_module
            spec.loader.exec_module(pipeline_module)
            _classify_image = pipeline_module.classify_image
            logger.info("Vector classifier pipeline loaded successfully")
        else:
            logger.warning(f"Vector classifier pipeline not found at {pipeline_path}")
            _classify_image = False
    except Exception as e:
        logger.error(f"Could not load vector classifier pipeline: {e}")
        _classify_image = False
    
    return _classify_image


def classify_model(payload: dict):
    """Classify model using a pre-computed image vector from the image-processing table."""
    start_time = time.time()
    logger.info(f"Starting model classification - payload: {json.dumps(payload, default=str)}")
    
    # Ensure Pinecone API key is available before classification
    ensure_pinecone_api_key()
    
    classify_image = _load_vector_classifier()
    if not classify_image:
        raise ValueError("Vector classifier pipeline not available")

    processing_id = payload.get("processing_id") or payload.get("processingId")
    if not processing_id:
        raise ValueError("'processing_id' is required for model classification")

    brand = payload.get("brand")
    if not brand:
        raise ValueError("'brand' is required for model classification")

    logger.info(f"Calling vector classifier for processing_id={processing_id}, brand={brand}")
    vector_start = time.time()
    result = classify_image(
        processing_id=processing_id,
        brand=brand,
        k=7,
    )
    vector_elapsed = time.time() - vector_start
    logger.info(f"Vector classification completed in {vector_elapsed:.2f}s")

    predicted_model = result.get("predicted_model")
    predicted_root_model = result.get("predicted_root_model")
    confidence = result.get("confidence", 0.0)
    
    # Check confidence threshold - require at least 4/7 neighbors to agree (57.14%)
    if confidence < MIN_MODEL_CONFIDENCE_THRESHOLD:
        logger.warning(f"Model classification confidence {confidence:.1f}% is below threshold {MIN_MODEL_CONFIDENCE_THRESHOLD}% - returning null result")
        total_elapsed = time.time() - start_time
        below_threshold_result = {
            "model": None,
            "model_id": None,
            "root_model": None,
            "root_model_id": None,
            "confidence": confidence,
            "below_threshold": True,
            "threshold": MIN_MODEL_CONFIDENCE_THRESHOLD,
        }
        # Include metadata from classification result (match_details, voting_details, etc.)
        if result.get("metadata"):
            below_threshold_result["metadata"] = result["metadata"]
        return below_threshold_result
    
    logger.info(f"Vector classifier result - model: {predicted_model}, root_model: {predicted_root_model}, confidence: {confidence}")

    # Always lookup root from knowledge service
    root_lookup_result = None
    if predicted_model:
        logger.info(f"Looking up root for predicted model: {predicted_model}")
        api_base_url = get_dsl_url()
        api_key = os.getenv("DSL_API_KEY")
        logger.info(f"Using DSL URL for stage '{get_stage()}': {api_base_url}")
        
        api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key, auth_headers=get_request_auth_headers())
        category = payload.get("category", "bags")
        partition = category
        root_type = "Bags" if category == "bags" else None
        
        root_lookup_result = lookup_root_from_child(
            api_client=api_client,
            property_type="model",
            value=predicted_model,
            brand=brand,
            root_type=root_type,
            partition=partition
        )
        
        # Use lookup result if available, otherwise use predicted_root_model
        root_model = root_lookup_result.get('root') if root_lookup_result else predicted_root_model
        root_model_id = root_lookup_result.get('root_id') if root_lookup_result else None
        model_id = root_lookup_result.get('child_id') if root_lookup_result else None
        
        if root_lookup_result:
            logger.info(f"Root lookup successful - root_model: {root_model}, root_model_id: {root_model_id}, model_id: {model_id}")
        else:
            logger.warning(f"Root lookup failed, using predicted_root_model: {predicted_root_model}")
    else:
        logger.warning("No predicted model from vector classifier, skipping root lookup")
        root_model = None
        root_model_id = None
        model_id = None

    total_elapsed = time.time() - start_time
    final_result = {
        "model": predicted_model,
        "model_id": model_id,
        "root_model": root_model,
        "root_model_id": root_model_id,
        "confidence": confidence,
    }
    
    # Include metadata from classification result (match_details, voting_details, etc.)
    if result.get("metadata"):
        final_result["metadata"] = result["metadata"]
    
    logger.info(f"Model classification completed in {total_elapsed:.2f}s - Result: {json.dumps(final_result, default=str)}")
    return final_result
