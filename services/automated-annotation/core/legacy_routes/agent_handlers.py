"""
Agent endpoint handlers for the new agent architecture system.

ARCHITECTURE EXCEPTION:
======================

This module is an EXCEPTION to the established architecture pattern.

NORMAL ARCHITECTURE:
- agent_orchestration_api_handler.py should ONLY call agent_orchestration functions
- agent_architecture should ONLY be called by agent_orchestration functions
- Config loading and agent initialization should ONLY happen in orchestration functions

THIS MODULE (EXCEPTION):
- Directly calls agent_architecture.LLMAnnotationAgent
- Directly loads configs using ConfigLoader
- Directly initializes and executes agents

WHY THIS EXCEPTION EXISTS:
- This endpoint (POST /automations/agents) is intended for external orchestration scripts
  (batch processing, CI/CD pipelines, etc.) that run outside the Lambda and need to call
  agents via HTTP API
- It provides a direct agent execution interface without requiring orchestration logic
- It's used by legacy systems that haven't been migrated to the new orchestration pattern

DEPRECATION PLAN:
- This module will be REMOVED once the legacy agent architecture is decommissioned
- External scripts should be migrated to use orchestration functions instead
- All agent calls should go through agent_orchestration functions, not directly

DO NOT USE THIS PATTERN FOR NEW CODE:
- New endpoints should use agent_orchestration_api_handler.py
- New orchestration functions should be added to agent_orchestration/
- This is a legacy compatibility layer, not a template to follow
"""

import json
import logging
import time
from typing import Dict, Any, Optional

from agent_architecture import LLMAnnotationAgent
from agent_architecture.validation import AgentStatus
from agent_orchestration.csv_config_loader import ConfigLoader
from stage_urls import get_stage
from structured_logger import StructuredLogger
from core.utils.responses import create_response, parse_body
from core.utils.credentials import ensure_gcp_adc, get_request_auth_headers

logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(layer="aifl", service_name="automated-annotation")


def handle_agent_execution(req_ctx, payload: dict):
    """
    Handle agent execution requests (POST /automations/agents).
    
    ARCHITECTURE EXCEPTION: This function directly calls agent_architecture,
    violating the normal architecture pattern. See module docstring for details.
    
    This endpoint is not publicly accessible - it's called by external orchestration
    scripts (batch processing, legacy systems, etc.).
    
    DEPRECATED: This will be removed once legacy systems are migrated.
    New code should use orchestration functions via agent_orchestration_api_handler.py.
    
    Request body:
    {
        "config_id": "material-30",  # Required
        "input_data": {              # Required
            "item_id": "12345",
            "image_url": "...",
            "text_input": "...",
            "input_mode": "auto"
        }
    }
    """
    request_start_time = time.time()
    
    config_id = payload.get("config_id")
    input_data = payload.get("input_data")
    
    if not config_id:
        error = ValueError("'config_id' is required")
        structured_logger.log_error(req_ctx, error, status_code=400)
        return create_response(400, {"error": "'config_id' is required"})
    
    if not input_data:
        error = ValueError("'input_data' is required")
        structured_logger.log_error(req_ctx, error, status_code=400)
        return create_response(400, {"error": "'input_data' is required"})
    
    try:
        logger.info(f"Agent execution request - config_id: {config_id}")
        
        # Ensure GCP credentials are set up
        ensure_gcp_adc()
        
        # Initialize config loader
        env = get_stage()
        config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
        
        # Load full agent config
        logger.info(f"Loading agent config: {config_id}")
        full_config = config_loader.load_full_agent_config(config_id)
        
        # Initialize agent
        logger.info(f"Initializing agent with model: {full_config['model_config'].get('model')}")
        agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
        
        # Execute agent
        logger.info("Executing agent...")
        agent_start = time.time()
        agent_result = agent.execute(input_data=input_data)
        agent_elapsed = time.time() - agent_start
        logger.info(f"Agent execution completed in {agent_elapsed:.2f}s - Status: {agent_result.status.value}")
        
        # Build response
        elapsed = time.time() - request_start_time
        
        response_data = {
            "config_id": config_id,
            "status": agent_result.status.value,
            "result": agent_result.result,
            "schema": agent_result.schema,
            "metadata": agent_result.metadata,
            "processing_time_seconds": round(elapsed, 3),
        }
        
        # Add validation info if available
        if agent_result.validation_info:
            response_data["validation_info"] = {
                "is_valid": agent_result.validation_info.is_valid,
                "category": agent_result.validation_info.category,
                "warnings": [{"rule_id": w.rule_id, "message": w.message} for w in agent_result.validation_info.warnings],
                "errors": [{"rule_id": e.rule_id, "message": e.message} for e in agent_result.validation_info.errors],
            }
        
        # Add error report if failed
        if agent_result.error_report:
            response_data["error_report"] = {
                "error_type": agent_result.error_report.error_type,
                "message": agent_result.error_report.message,
                "recoverable": agent_result.error_report.recoverable,
                "details": agent_result.error_report.details,
            }
        
        status_code = 200 if agent_result.status == AgentStatus.SUCCESS else 400
        structured_logger.log_response(req_ctx, status_code=status_code)
        
        return create_response(status_code, {
            "component_type": "agent_execution_result",
            "data": [response_data],
            "metadata": {"config_id": config_id},
        })
        
    except ValueError as exc:
        elapsed = time.time() - request_start_time
        logger.error(f"Validation error after {elapsed:.2f}s: {str(exc)}")
        response = create_response(400, {"error": str(exc)})
        structured_logger.log_error(req_ctx, exc, status_code=400)
        return response
    except Exception as exc:
        elapsed = time.time() - request_start_time
        logger.error(f"Error after {elapsed:.2f}s: {str(exc)}")
        response = create_response(500, {"error": str(exc)})
        structured_logger.log_error(req_ctx, exc, status_code=500)
        return response
