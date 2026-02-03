"""
Main entry point for the automated annotation Lambda function.

This module serves as the central routing point for all API endpoints.
All endpoints are clearly defined and mapped to their respective handlers.
"""

import json
import logging
import time

from structured_logger import StructuredLogger
from stage_urls import get_stage, get_dsl_url, get_annotation_dsl_url, get_ifl_url, get_annotation_ifl_url
from core.utils.responses import parse_body, create_response
from core.utils.credentials import set_request_auth_headers
from core.legacy_routes.handlers import handle_classification, handle_health, handle_options
from core.legacy_routes.agent_handlers import handle_agent_execution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize structured logger for metrics (REQUEST/RESPONSE/ERROR lifecycle events)
structured_logger = StructuredLogger(layer="aifl", service_name="automated-annotation")

# Log stage-aware URL configuration at module load
logger.info(f"Stage URL Configuration - Stage: {get_stage()}")
logger.info(f"   DSL_API_BASE_URL: {get_dsl_url()}")
logger.info(f"   ANNOTATION_DSL_API_BASE_URL: {get_annotation_dsl_url()}")
logger.info(f"   IFL_API_BASE_URL: {get_ifl_url()}")
logger.info(f"   ANNOTATION_IFL_API_BASE_URL: {get_annotation_ifl_url()}")


# ============================================================================
# API ENDPOINT DEFINITIONS
# ============================================================================
#
# All available endpoints are documented here for easy reference:
#
# LEGACY ENDPOINTS (mapped to new agent system):
# ==============================================
#
# 1. POST /automations/annotation/{category}/classify/model
#    - Classify bag model using vector-based similarity search (legacy)
#    - Mapped to: New agent system (maintains backward compatibility)
#    - Requires: image (processing_id), brand
#    - Returns: model, model_id, root_model, root_model_id, confidence
#
# 2. POST /automations/annotation/{category}/classify/{property}
#    - Classify a property (type, material, colour, etc.) using LLM
#    - Mapped to: New agent system via orchestrator
#    - Supports both single-item and batch processing modes
#    - Single mode: Standard payload with property fields
#    - Batch mode: Payload with 'items' list (each item has same structure as single mode)
#    - Optional batch params: max_workers (default: 200), batch_size (for chunked processing)
#    - Requires: image, text_dump
#    - Optional: brand, input_mode, resolve_names
#    - Returns: property value, confidence, and root taxonomy IDs (for model/material)
#    - Batch mode returns: List of results in same order as input items
#
# 2a. POST /automations/annotation/{category}/classify/size
#     - Classify size using size orchestrator (two-pipeline workflow)
#     - Supports both single-item and batch processing modes
#     - Single mode: Standard payload
#     - Batch mode: Payload with 'items' list
#     - Requires: text_input (or text_dump or text_metadata), model_id
#     - Optional batch params: max_workers (default: 200), batch_size
#     - Returns: size, size_id, confidence, reasoning
#     - Batch mode returns: List of results in same order as input items
#
# 2b. POST /automations/annotation/{category}/classify/keywords
#     - Extract keywords using keyword classifier orchestrator
#     - Supports both single-item and batch processing modes
#     - Single mode: Standard payload
#     - Batch mode: Payload with 'items' list
#     - Requires: general_input_text (dict), text_to_avoid (dict)
#     - Optional: item_id
#     - Optional batch params: max_workers (default: 200), batch_size
#     - Returns: keywords (list), keyword_count, keyword_1-3, keyword_1-3_confidence, reasoning
#     - Batch mode returns: List of results in same order as input items
#
# 2c. POST /automations/annotation/{category}/classify/hardware
#     - Classify hardware using hardware classifier orchestrator
#     - Supports both single-item and batch processing modes
#     - Single mode: Standard payload
#     - Batch mode: Payload with 'items' list
#     - Requires: image, text_dump (or text_input or text_metadata)
#     - Optional: brand, input_mode
#     - Optional batch params: max_workers (default: 200), batch_size
#     - Returns: hardware, hardware_id, confidence, reasoning
#     - Batch mode returns: List of results in same order as input items
#
# NEW AGENT ARCHITECTURE ENDPOINTS:
# ==================================
#
# 3. POST /automations/agents
#    - Direct agent execution endpoint (internal use - called by external orchestration scripts)
#    - NOT publicly accessible - used by external batch processing/legacy systems
#    - ARCHITECTURE EXCEPTION: Directly calls agent_architecture (violates normal pattern)
#    - DEPRECATED: Will be removed once legacy systems are migrated
#    - Requires: config_id, input_data
#    - Returns: AgentResult with status, result, validation_info, error_report
#    - NOTE: New code should use orchestration functions via agent_orchestration_api_handler.py
#
# UTILITY ENDPOINTS:
# ==================
#
# 4. GET /health
#    - Health check endpoint
#    - Returns: service status and stage information
#
# 5. POST /automations/annotation/csv-config
#    - Generate CSV column mapping configuration
#    - Mapped to: CSV config generation orchestration
#    - Requires: CSV_uuid, sample_rows (list of 5-10 sample row dicts)
#    - Optional: organisation_uuid, max_chars
#    - Returns: CSV_uuid, CSV_config_id, csv_config
#    - Config is saved to DynamoDB for future use
#
# 6. OPTIONS *
#    - CORS preflight handler
#    - Returns: CORS headers
#
# ============================================================================


def lambda_handler(event, context):
    """
    Main Lambda handler function.
    
    Routes incoming API Gateway events to the appropriate endpoint handlers.
    """
    request_start_time = time.time()
    request_id = context.aws_request_id if context else "unknown"
    
    # Start structured logging for metrics (captures request timing)
    req_ctx = structured_logger.start_request(event)
    
    logger.info("=" * 80)
    logger.info(f"Lambda invocation started - Request ID: {request_id}")
    logger.info(f"Event method: {event.get('httpMethod', 'UNKNOWN')}, path: {event.get('path', 'UNKNOWN')}")
    logger.debug(f"Full event: {json.dumps(event, default=str)[:1000]}")
    
    # Extract auth headers from original request for pass-through to downstream services
    incoming_headers = event.get("headers") or {}
    auth_headers = {}
    # Check both lowercase and mixed-case header names (API Gateway normalizes differently)
    for key in incoming_headers:
        key_lower = key.lower()
        if key_lower == 'authorization':
            auth_headers['Authorization'] = incoming_headers[key]
            logger.debug("Captured Authorization header for pass-through")
        elif key_lower == 'x-api-key':
            auth_headers['x-api-key'] = incoming_headers[key]
            logger.debug("Captured x-api-key header for pass-through")
    
    # Set auth headers for use in downstream services
    set_request_auth_headers(auth_headers)
    
    try:
        method = event.get("httpMethod", "GET")
        path = (event.get("path") or "").rstrip("/")
        
        # Strip custom domain base path prefix if present (e.g., /agents from api.trussarchive.io/agents/...)
        if path.startswith("/agents/"):
            path = path[7:]  # Remove "/agents" prefix, keep leading "/"
            logger.info(f"Stripped /agents prefix from path")
        
        logger.info(f"Processing {method} request to {path}")

        # Handle OPTIONS (CORS preflight)
        if method == "OPTIONS":
            return handle_options(req_ctx)

        # Parse path segments
        segments = [segment for segment in path.strip("/").split("/") if segment]
        logger.debug(f"Path segments: {segments}")

        # Route: POST /automations/annotation/{category}/classify/{target}
        # Handles both model and property classification
        if (
            len(segments) >= 5
            and segments[0] == "automations"
            and segments[1] == "annotation"
            and segments[3] == "classify"
            and method == "POST"
        ):
            category = segments[2]
            target = segments[4]
            
            logger.info(f"Classification request - category: {category}, target: {target}")

            payload = parse_body(event)
            logger.info(f"Parsed payload keys: {list(payload.keys())}")

            return handle_classification(req_ctx, category, target, payload)

        # Route: POST /automations/agents
        # Agent execution endpoint (internal - called by orchestrators)
        if (
            len(segments) >= 2
            and segments[0] == "automations"
            and segments[1] == "agents"
            and method == "POST"
        ):
            logger.info("Agent execution request")
            payload = parse_body(event)
            return handle_agent_execution(req_ctx, payload)

        # Route: POST /automations/annotation/csv-config
        # CSV config generation endpoint
        if (
            len(segments) >= 3
            and segments[0] == "automations"
            and segments[1] == "annotation"
            and segments[2] == "csv-config"
            and method == "POST"
        ):
            logger.info("CSV config generation request")
            payload = parse_body(event)
            from core.legacy_routes.handlers import handle_csv_config_generation
            return handle_csv_config_generation(req_ctx, payload)

        # Route: POST /automations/workflows/csv-config
        # CSV config generation endpoint (workflow alias)
        if (
            len(segments) >= 3
            and segments[0] == "automations"
            and segments[1] == "workflows"
            and segments[2] == "csv-config"
            and method == "POST"
        ):
            logger.info("CSV config generation request (workflow endpoint)")
            payload = parse_body(event)
            from core.legacy_routes.handlers import handle_csv_config_generation
            return handle_csv_config_generation(req_ctx, payload)

        # Route: GET /health
        if path.endswith("/health"):
            return handle_health(req_ctx)

        # Route not found
        structured_logger.log_warning(req_ctx, f"Endpoint not found - path: {path}, method: {method}", {
            "path": path,
            "method": method,
        })
        response = create_response(404, {"error": "Endpoint not found"})
        structured_logger.log_response(req_ctx, status_code=404)
        return response

    except Exception as e:
        elapsed = time.time() - request_start_time
        # Log structured error for metrics (captures duration, error details)
        structured_logger.log_error(req_ctx, e, status_code=500)
        return create_response(500, {"error": str(e)})
    finally:
        total_elapsed = time.time() - request_start_time
        logger.info(f"Lambda invocation completed in {total_elapsed:.2f}s - Request ID: {request_id}")
        logger.info("=" * 80)
