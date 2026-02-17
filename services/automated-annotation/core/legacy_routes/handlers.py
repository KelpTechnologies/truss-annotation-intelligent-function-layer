"""
Route handler functions for API endpoints.

This module handles both legacy endpoints (using old classifier system)
and new agent-based endpoints (using orchestrator system).
"""

import json
import logging
import time

from agent_utils.dsl_api_client import DSLAPIClient
from stage_urls import get_dsl_url, get_stage
from structured_logger import StructuredLogger
from core.utils.config import CATEGORY_CONFIG
from core.utils.responses import create_response, parse_body
from core.utils.credentials import get_request_auth_headers, set_request_auth_headers
from core.orchestration_api_handlers.model_vector_classifier_api_handler import classify_model
from core.orchestration_api_handlers.agent_orchestration_api_handler import (
    execute_classification_for_api,
    execute_brand_classification_for_api,
    execute_model_classification_for_api,
    get_config_id_for_property,
    get_config_id_for_input_mode,
    detect_input_mode,
    format_classification_for_legacy_api,
    execute_size_classification_for_api,
    execute_keyword_classification_for_api,
    execute_hardware_classification_for_api,
    execute_csv_config_generation_for_api
)

logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(layer="aifl", service_name="automated-annotation")


def handle_options(req_ctx):
    """Handle OPTIONS (CORS preflight) requests."""
    logger.info("Handling OPTIONS request")
    response = create_response(200, {"ok": True})
    structured_logger.log_response(req_ctx, status_code=200)
    return response


def handle_health(req_ctx):
    """Handle health check requests."""
    logger.info("Processing health check request")
    api_base_url = get_dsl_url()
    api_key = __import__('os').getenv("DSL_API_KEY")
    status = {"status": "unhealthy", "stage": get_stage()}
    try:
        logger.info(f"Performing DSL API health check - stage: {get_stage()}, url: {api_base_url}")
        client = DSLAPIClient(base_url=api_base_url, api_key=api_key, auth_headers=get_request_auth_headers())
        status = client.health_check()
        status["stage"] = get_stage()
        logger.info(f"Health check result: {json.dumps(status, default=str)}")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        status = {"status": "unhealthy", "error": str(e), "stage": get_stage()}

    response = create_response(200, {"component_type": "health_check", "data": [status], "metadata": {}}, methods="GET,OPTIONS")
    structured_logger.log_response(req_ctx, status_code=200)
    return response


def handle_classification(req_ctx, category: str, target: str, payload: dict):
    """Handle classification requests (model or property)."""
    request_start_time = time.time()
    
    if category not in CATEGORY_CONFIG:
        error = ValueError(f"Unsupported category '{category}'")
        structured_logger.log_error(req_ctx, error, status_code=404)
        response = create_response(404, {"error": f"Unsupported category '{category}'"})
        structured_logger.log_response(req_ctx, status_code=404)
        return response

    try:
        if target == "model":
            logger.info("Processing model classification request")
            brand = payload.get("brand") or CATEGORY_CONFIG[category].get("default_brand")
            if not brand:
                raise ValueError("'brand' is required for model classification")

            has_image = bool(
                (payload.get("image") or payload.get("processing_id") or payload.get("processingId") or payload.get("image_url"))
                and str(payload.get("image") or payload.get("processing_id") or payload.get("processingId") or payload.get("image_url") or "").strip()
            )

            if has_image:
                image = payload.get("image") or payload.get("processing_id") or payload.get("processingId") or payload.get("image_url")
                logger.info(f"Model classification (vector) - image present, brand: {brand}, category: {category}")
                result = classify_model({
                    "processing_id": image,
                    "processingId": image,
                    "brand": brand,
                    "category": category,
                })
            else:
                logger.info(f"Model classification (orchestration text-only) - brand: {brand}, category: {category}")
                result = execute_model_classification_for_api(api_input=payload)
                result = format_classification_for_legacy_api(result, "model")

            elapsed = time.time() - request_start_time
            logger.info(f"Model classification request completed in {elapsed:.2f}s")
            response = create_response(
                200,
                {
                    "component_type": "model_classification_result",
                    "data": [result],
                    "metadata": {"category": category, "target": target},
                },
            )
            structured_logger.log_response(req_ctx, status_code=200)
            return response

        logger.info(f"Processing property classification - target: {target}")
        
        # Check if batch mode (payload contains 'items' list)
        is_batch_mode = isinstance(payload, dict) and 'items' in payload and isinstance(payload.get('items'), list)
        
        if is_batch_mode:
            logger.info(f"Processing batch {target} classification request ({len(payload['items'])} items)")
        else:
            logger.info(f"Processing single {target} classification request")
        
        # Handle special orchestrators (size, keywords, hardware)
        if target == "size":
            logger.info("Processing size classification request")
            result = execute_size_classification_for_api(api_input=payload)

            if is_batch_mode:
                # Batch mode: result is a list
                formatted_results = [
                    {
                        "size": r.get("size", ""),
                        "size_id": r.get("size_id"),
                        "confidence": r.get("confidence", 0.0),
                        "reasoning": r.get("reasoning", ""),
                        # Preserve critical fields for frontend error detection
                        "success": r.get("success", True),
                        "validation_passed": r.get("validation_passed", True),
                        "status": r.get("status", ""),
                        "error_type": r.get("error_type", ""),
                        "error": r.get("error", ""),
                        "primary_name": r.get("size", ""),
                        "primary_id": r.get("size_id"),
                    }
                    for r in result
                ]
                component_type = "size_classification_result"
                data = formatted_results
            else:
                # Single mode: result is a dict
                formatted_result = {
                    "size": result.get("size", ""),
                    "size_id": result.get("size_id"),
                    "confidence": result.get("confidence", 0.0),
                    "reasoning": result.get("reasoning", ""),
                    # Preserve critical fields for frontend error detection
                    "success": result.get("success", True),
                    "validation_passed": result.get("validation_passed", True),
                    "status": result.get("status", ""),
                    "error_type": result.get("error_type", ""),
                    "error": result.get("error", ""),
                    "primary_name": result.get("size", ""),
                    "primary_id": result.get("size_id"),
                }
                component_type = "size_classification_result"
                data = [formatted_result]
                
        elif target in ("keyword", "keywords", "key-words"):
            logger.info("Processing keyword classification request (keyword_classifier_orchestration)")
            result = execute_keyword_classification_for_api(api_input=payload)

            if is_batch_mode:
                # Batch mode: result is a list
                formatted_results = [
                    {
                        "keywords": r.get("keywords", []),
                        "keyword_count": r.get("keyword_count", 0),
                        "keyword_1": r.get("keyword_1", ""),
                        "keyword_2": r.get("keyword_2", ""),
                        "keyword_3": r.get("keyword_3", ""),
                        "keyword_1_confidence": r.get("keyword_1_confidence", 0.0),
                        "keyword_2_confidence": r.get("keyword_2_confidence", 0.0),
                        "keyword_3_confidence": r.get("keyword_3_confidence", 0.0),
                        "reasoning": r.get("reasoning", ""),
                        # Preserve critical fields for frontend error detection
                        "success": r.get("success", True),
                        "validation_passed": r.get("validation_passed", True),
                        "status": r.get("status", ""),
                        "error_type": r.get("error_type", ""),
                        "error": r.get("error", ""),
                        "primary_name": r.get("keyword_1", ""),  # Use first keyword as primary
                        "primary_id": None,  # Keywords don't have IDs
                    }
                    for r in result
                ]
                component_type = "keyword_classification_result"
                data = formatted_results
            else:
                # Single mode: result is a dict
                formatted_result = {
                    "keywords": result.get("keywords", []),
                    "keyword_count": result.get("keyword_count", 0),
                    "keyword_1": result.get("keyword_1", ""),
                    "keyword_2": result.get("keyword_2", ""),
                    "keyword_3": result.get("keyword_3", ""),
                    "keyword_1_confidence": result.get("keyword_1_confidence", 0.0),
                    "keyword_2_confidence": result.get("keyword_2_confidence", 0.0),
                    "keyword_3_confidence": result.get("keyword_3_confidence", 0.0),
                    "reasoning": result.get("reasoning", ""),
                    # Preserve critical fields for frontend error detection
                    "success": result.get("success", True),
                    "validation_passed": result.get("validation_passed", True),
                    "status": result.get("status", ""),
                    "error_type": result.get("error_type", ""),
                    "error": result.get("error", ""),
                    "primary_name": result.get("keyword_1", ""),  # Use first keyword as primary
                    "primary_id": None,  # Keywords don't have IDs
                }
                component_type = "keyword_classification_result"
                data = [formatted_result]
                
        elif target == "hardware":
            logger.info("Processing hardware classification request")
            input_mode = payload.get("input_mode") or detect_input_mode(payload)
            base_config_id = "classifier-hardware-bags"
            config_id = get_config_id_for_input_mode(base_config_id, input_mode)
            logger.info(
                f"Mapped {category}/hardware to config_id: {config_id} (input_mode={input_mode})"
            )
            if "input_mode" not in payload:
                payload = {**payload, "input_mode": input_mode}

            result = execute_classification_for_api(
                config_id=config_id,
                api_input=payload
            )
            
            if is_batch_mode:
                # Batch mode: result is a list
                formatted_results = [format_classification_for_legacy_api(r, target) for r in result]
                component_type = "hardware_classification_result"
                data = formatted_results
            else:
                # Single mode: result is a dict
                formatted_result = format_classification_for_legacy_api(result, target)
                component_type = "hardware_classification_result"
                data = [formatted_result]

        elif target == "brand":
            logger.info("Processing brand classification request (brand_classification_orchestration)")
            result = execute_brand_classification_for_api(api_input=payload)
            formatted_result = format_classification_for_legacy_api(result, target)
            component_type = "brand_classification_result"
            data = [formatted_result]

        else:
            # Use new agent-based orchestrator system for other properties
            # Detect input format and choose config (text-only -> -text config, else base config)
            input_mode = payload.get("input_mode") or detect_input_mode(payload)
            base_config_id = get_config_id_for_property(category, target)
            config_id = get_config_id_for_input_mode(base_config_id, input_mode)
            logger.info(f"Mapped {category}/{target} to config_id: {config_id} (input_mode={input_mode})")
            if "input_mode" not in payload:
                payload = {**payload, "input_mode": input_mode}

            # Execute classification via lightweight API handler
            result = execute_classification_for_api(
                config_id=config_id,
                api_input=payload
            )
            
            if is_batch_mode:
                # Batch mode: result is a list
                formatted_results = [format_classification_for_legacy_api(r, target) for r in result]
                component_type = "classification_result"
                data = formatted_results
            else:
                # Single mode: result is a dict
                formatted_result = format_classification_for_legacy_api(result, target)
                component_type = "classification_result"
                data = [formatted_result]
                
                # For single-item mode, check if extraction failed (not just "Unknown"/null)
                # Return 400 for actual errors (success=False), but 200 for successful null extractions
                if result.get("success") is False and not result.get("no_extraction"):
                    elapsed = time.time() - request_start_time
                    logger.error(f"Classification failed after {elapsed:.2f}s: {result.get('error', 'Unknown error')}")
                    response = create_response(
                        400,
                        {
                            "component_type": component_type,
                            "data": data,
                            "metadata": {
                                "category": category,
                                "target": target,
                                "error": result.get("error", "Classification failed"),
                                "error_type": result.get("error_type", "unknown")
                            },
                        },
                    )
                    structured_logger.log_error(req_ctx, ValueError(result.get("error", "Classification failed")), status_code=400)
                    return response
        
        elapsed = time.time() - request_start_time
        logger.info(f"Property classification request completed in {elapsed:.2f}s")
        response = create_response(
            200,
            {
                "component_type": component_type,
                "data": data,
                "metadata": {
                    "category": category,
                    "target": target,
                    "batch_mode": is_batch_mode,
                    "item_count": len(data) if is_batch_mode else 1
                },
            },
        )
        structured_logger.log_response(req_ctx, status_code=200)
        return response
    except ValueError as exc:
        elapsed = time.time() - request_start_time
        logger.error(f"Validation error after {elapsed:.2f}s: {str(exc)}")
        response = create_response(400, {"error": str(exc)})
        structured_logger.log_error(req_ctx, exc, status_code=400)
        return response
    except PermissionError as exc:
        elapsed = time.time() - request_start_time
        logger.error(f"Permission error after {elapsed:.2f}s: {str(exc)}")
        response = create_response(403, {"error": str(exc)})
        structured_logger.log_error(req_ctx, exc, status_code=403)
        return response
    except RuntimeError as exc:
        elapsed = time.time() - request_start_time
        logger.error(f"Runtime error after {elapsed:.2f}s: {str(exc)}")
        response = create_response(500, {"error": str(exc)})
        structured_logger.log_error(req_ctx, exc, status_code=500)
        return response


def handle_csv_config_generation(req_ctx, payload: dict):
    """Handle CSV config generation requests."""
    request_start_time = time.time()
    
    try:
        logger.info("Processing CSV config generation request")
        
        # Execute CSV config generation via API handler
        result = execute_csv_config_generation_for_api(api_input=payload)
        
        elapsed = time.time() - request_start_time
        logger.info(f"CSV config generation request completed in {elapsed:.2f}s")
        
        response = create_response(
            200,
            {
                "component_type": "csv_config_generation_result",
                "data": [result],
                "metadata": {
                    "CSV_uuid": result.get("CSV_uuid"),
                    "CSV_config_id": result.get("CSV_config_id")
                },
            },
        )
        structured_logger.log_response(req_ctx, status_code=200)
        return response
    except ValueError as exc:
        elapsed = time.time() - request_start_time
        logger.error(f"Validation error after {elapsed:.2f}s: {str(exc)}")
        response = create_response(400, {"error": str(exc)})
        structured_logger.log_error(req_ctx, exc, status_code=400)
        return response
    except RuntimeError as exc:
        elapsed = time.time() - request_start_time
        logger.error(f"Runtime error after {elapsed:.2f}s: {str(exc)}")
        response = create_response(500, {"error": str(exc)})
        structured_logger.log_error(req_ctx, exc, status_code=500)
        return response
    except Exception as exc:
        elapsed = time.time() - request_start_time
        logger.error(f"Unexpected error after {elapsed:.2f}s: {str(exc)}")
        response = create_response(500, {"error": str(exc)})
        structured_logger.log_error(req_ctx, exc, status_code=500)
        return response
