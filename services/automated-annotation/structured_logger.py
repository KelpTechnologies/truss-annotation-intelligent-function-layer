"""
Structured Logger - Python implementation for schema v2 log monitoring compliance.

This module provides structured logging that emits JSON logs compatible with
the centralized log monitor in truss-api-platform.

Usage:
    from structured_logger import StructuredLogger
    
    logger = StructuredLogger(layer="aifl", service_name="automated-annotation")
    
    def lambda_handler(event, context):
        req_ctx = logger.start_request(event)
        
        try:
            # ... your code ...
            logger.log_response(req_ctx, status_code=200)
            return response
        except Exception as e:
            logger.log_error(req_ctx, e, status_code=500)
            raise
"""

import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
import re

LOG_SCHEMA_VERSION = 2.2

# Log levels
LOG_LEVELS = {
    "ERROR": 0,
    "WARN": 1,
    "INFO": 2,
    "DEBUG": 3,
}

# Map log types to minimum required log level
LOG_TYPE_LEVELS = {
    "ERROR": LOG_LEVELS["ERROR"],     # 0 — always emits
    "WARNING": LOG_LEVELS["WARN"],    # 1
    "REQUEST": LOG_LEVELS["INFO"],    # 2
    "RESPONSE": LOG_LEVELS["INFO"],   # 2
    "METRIC": LOG_LEVELS["INFO"],     # 2
    "DEBUG": LOG_LEVELS["DEBUG"],     # 3
}

import os
_current_log_level = LOG_LEVELS.get(os.environ.get("LOG_LEVEL", "INFO").upper(), LOG_LEVELS["INFO"])

# Layer detection mapping (matches JavaScript implementation)
LAYER_MAP = {
    "/automations/annotation": "aifl",
    "/automations/pricing": "pricing-ifl",
    "/images": "aifl",
    "/knowledge": "annotation-dsl",
    "/visual-classifier": "annotation-dsl",
    "/footwear": "dsl",
    "/bags": "dsl",
    "/apparel": "dsl",
    "/analytics": "dsl",
    "/prices": "dsl",
    "/products": "dsl",
    "/brands": "dsl",
    "/beta": "dsl",
    "/discounts": "dsl",
    "/pricing": "pricing-ifl",
}

# Sort prefixes by length (longest first) for proper matching
SORTED_PREFIXES = sorted(LAYER_MAP.keys(), key=len, reverse=True)


def detect_layer(resource_path: str) -> str:
    """Detect layer from resource path."""
    if not resource_path or not isinstance(resource_path, str):
        return "unknown"
    
    normalized_path = resource_path if resource_path.startswith("/") else f"/{resource_path}"
    
    for prefix in SORTED_PREFIXES:
        if normalized_path.startswith(prefix):
            if len(normalized_path) == len(prefix) or normalized_path[len(prefix)] == "/":
                return LAYER_MAP[prefix]
    
    return "unknown"


def normalize_route(method: str, path: str, path_parameters: Optional[Dict[str, str]] = None) -> str:
    """Normalize route by replacing dynamic segments with placeholders."""
    if not path:
        return f"{method} /" if method else "/"
    
    normalized = path
    path_parameters = path_parameters or {}
    
    # Replace known path parameter values
    for key, value in path_parameters.items():
        if value:
            escaped = re.escape(value)
            normalized = re.sub(f"/{escaped}(?=/|$)", f"/{{{key}}}", normalized)
    
    # Common patterns to normalize
    patterns = [
        # UUIDs
        (r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "/{id}"),
        # Numeric IDs
        (r"/\d+(?=/|$)", "/{id}"),
        # Processing IDs
        (r"/proc_[a-zA-Z0-9]+", "/{processingId}"),
    ]
    
    for pattern, replacement in patterns:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    
    return f"{method} {normalized}" if method else normalized


RESOURCE_TYPE_MAP = {
    "/products": "product",
    "/prices": "product",
    "/analytics": "product",
    "/listings": "listing",
    "/discounts": "product",
    "/forecasts": "product",
    "/brands": "brand",
    "/accounts": "account",
    "/knowledge": "knowledge",
    "/images": "image",
    "/automations/annotation": "annotation",
    "/automations/pricing": "pricing",
}

SORTED_RESOURCE_PREFIXES = sorted(RESOURCE_TYPE_MAP.keys(), key=len, reverse=True)


def extract_id_from_path(path: str) -> Optional[str]:
    """Extract resource ID (UUID or numeric) from URL path segments."""
    if not path:
        return None
    segments = [s for s in path.split("/") if s]
    for seg in reversed(segments[1:]):
        if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", seg, re.IGNORECASE):
            return seg
        if re.match(r"^\d{5,}$", seg):
            return seg
        if re.match(r"^proc_[a-zA-Z0-9]+$", seg):
            return seg
    return None


def detect_resource_type_from_path(path: str) -> Optional[str]:
    """Detect resource type from URL path prefix."""
    if not path:
        return None
    normalized = path if path.startswith("/") else f"/{path}"
    for prefix in SORTED_RESOURCE_PREFIXES:
        if normalized.startswith(prefix) and (len(normalized) == len(prefix) or normalized[len(prefix)] == "/"):
            return RESOURCE_TYPE_MAP[prefix]
    return None


def extract_resource_context(event: dict) -> dict:
    """Extract resource context from Lambda event. Priority: headers > path > query > body."""
    headers = event.get("headers") or {}
    query_params = event.get("queryStringParameters") or {}
    path = event.get("path", "") or event.get("requestContext", {}).get("http", {}).get("path", "")

    resource_id = (
        headers.get("x-entity-id") or
        extract_id_from_path(path) or
        query_params.get("product_id") or
        query_params.get("entity_id") or
        None
    )
    if not resource_id:
        body_str = event.get("body")
        if body_str and isinstance(body_str, str):
            try:
                body = json.loads(body_str)
                resource_id = body.get("entity_id") or body.get("product_id") or body.get("processing_id")
                if resource_id is not None:
                    resource_id = str(resource_id)
            except (json.JSONDecodeError, AttributeError):
                pass

    resource_type = headers.get("x-entity-type") or detect_resource_type_from_path(path)
    operation_run_id = headers.get("x-operation-run-id")

    return {
        "resource_id": resource_id,
        "resource_type": resource_type,
        "operation_run_id": operation_run_id,
    }


class StructuredLogger:
    """
    Structured logger for emitting schema v2 compatible logs.
    
    Emits JSON logs that can be parsed by the centralized log monitor
    in truss-api-platform.
    """
    
    def __init__(self, layer: str = "aifl", service_name: Optional[str] = None):
        """
        Initialize the structured logger.
        
        Args:
            layer: Service layer identifier (e.g., "aifl", "dsl", "pricing-ifl")
            service_name: Name of the service (e.g., "automated-annotation")
        """
        self.layer = layer
        self.service_name = service_name
    
    def start_request(self, event: dict) -> dict:
        """
        Start tracking a request. Call this at the beginning of your handler.
        
        Args:
            event: Lambda event object
            
        Returns:
            Request context dict to pass to log_response or log_error
        """
        request_context = event.get("requestContext", {})
        authorizer = request_context.get("authorizer", {})
        headers = event.get("headers", {}) or {}
        
        # Get request ID from various sources
        request_id = (
            request_context.get("requestId") or
            headers.get("x-request-id") or
            headers.get("X-Request-Id") or
            self._generate_request_id()
        )

        # Extract or generate correlation ID for cross-service tracing
        correlation_id = (
            headers.get("x-correlation-id") or
            headers.get("X-Correlation-Id") or
            self._generate_correlation_id()
        )

        method = event.get("httpMethod", "") or request_context.get("http", {}).get("method", "")
        path = event.get("path", "") or request_context.get("http", {}).get("path", "")
        path_params = event.get("pathParameters") or {}

        # Extract user context from authorizer
        user_id = (
            authorizer.get("keyId") or
            authorizer.get("userId") or
            authorizer.get("principalId")
        )
        tenant_id = authorizer.get("tenantId")
        auth_type = authorizer.get("authType", "unknown")

        # Detect layer from path if not explicitly set
        layer = self.layer or detect_layer(path)

        # Extract resource context for product/entity tracking
        resource_ctx = extract_resource_context(event)

        context = {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "method": method,
            "path": path,
            "route": f"{method} {path}",
            "route_normalized": normalize_route(method, path, path_params),
            "layer": layer,
            "service_name": self.service_name,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "auth_type": auth_type,
            "resource_id": resource_ctx["resource_id"],
            "resource_type": resource_ctx["resource_type"],
            "operation_run_id": resource_ctx["operation_run_id"],
            "start_time": time.time(),
            "query_params": event.get("queryStringParameters") or {},
            "path_params": path_params,
        }
        
        # Emit REQUEST log
        self._emit("REQUEST", context, {
            "queryParams": context["query_params"],
            "pathParams": context["path_params"],
        })
        
        return context
    
    def log_response(
        self,
        ctx: dict,
        status_code: int = 200,
        count: Optional[int] = None,
        query_complexity: Optional[str] = None,
    ):
        """
        Log a successful response.
        
        Args:
            ctx: Request context from start_request
            status_code: HTTP status code
            count: Optional record count (useful for DSL queries)
            query_complexity: Optional query complexity level
        """
        duration_ms = int((time.time() - ctx.get("start_time", time.time())) * 1000)
        
        extra = {
            "statusCode": status_code,
            "durationMs": duration_ms,
        }
        
        if count is not None:
            extra["count"] = count
        if query_complexity is not None:
            extra["queryComplexity"] = query_complexity
        
        self._emit("RESPONSE", ctx, extra)
    
    def log_error(
        self,
        ctx: dict,
        error: Exception,
        status_code: int = 500,
    ):
        """
        Log an error.
        
        Args:
            ctx: Request context from start_request
            error: The exception that occurred
            status_code: HTTP status code
        """
        duration_ms = int((time.time() - ctx.get("start_time", time.time())) * 1000)
        
        import traceback
        stack_trace = traceback.format_exc()
        # Truncate stack trace to 1000 chars like JS version
        if len(stack_trace) > 1000:
            stack_trace = stack_trace[:1000]
        
        self._emit("ERROR", ctx, {
            "statusCode": status_code,
            "durationMs": duration_ms,
            "error": {
                "message": str(error),
                "name": type(error).__name__,
                "stack": stack_trace,
                "code": getattr(error, "code", None),
            },
        })
    
    def log_warning(
        self,
        ctx: dict,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a warning.
        Unlike errors, warnings don't terminate the request - they're informational alerts.
        Can be called multiple times per request to emit multiple warnings.
        
        Args:
            ctx: Request context from start_request
            message: Warning message
            details: Additional details about the warning
        """
        warning_data = {
            "message": message,
            "messageOriginal": message,
        }
        if details:
            warning_data.update(details)
        
        self._emit("WARNING", ctx, {
            "warning": warning_data,
        })
    
    def log_metric(
        self,
        ctx: dict,
        metric_name: str,
        value: float,
        unit: str = "Count",
    ):
        """
        Log a custom metric.
        
        Args:
            ctx: Request context from start_request
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
        """
        self._emit("METRIC", ctx, {
            "metricName": metric_name,
            "value": value,
            "unit": unit,
        })
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None, ctx: Optional[dict] = None):
        """
        Log debug information.
        
        Args:
            message: Debug message
            data: Additional data to include
            ctx: Optional request context
        """
        import os
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        if log_level != "DEBUG":
            return
        
        extra = {"message": message}
        if data:
            extra.update(data)
        
        self._emit("DEBUG", ctx or {"request_id": None}, extra)
    
    def _emit(self, log_type: str, ctx: dict, extra: Optional[Dict[str, Any]] = None):
        """
        Internal method to emit a structured log.

        Args:
            log_type: Type of log (REQUEST, RESPONSE, ERROR, METRIC, DEBUG)
            ctx: Request context
            extra: Additional fields to include
        """
        # Gate by LOG_LEVEL
        required_level = LOG_TYPE_LEVELS.get(log_type, LOG_LEVELS["INFO"])
        if _current_log_level < required_level:
            return

        payload = {
            "schemaVersion": LOG_SCHEMA_VERSION,
            "logType": log_type,
            "ts": datetime.utcnow().isoformat() + "Z",
            "requestId": ctx.get("request_id"),
            "correlationId": ctx.get("correlation_id"),
            "layer": ctx.get("layer", self.layer),
            "serviceName": ctx.get("service_name", self.service_name),
            "route": ctx.get("route"),
            "routeNormalized": ctx.get("route_normalized"),
            "userId": ctx.get("user_id"),
            "tenantId": ctx.get("tenant_id"),
            "authType": ctx.get("auth_type", "unknown"),
            "resourceId": ctx.get("resource_id"),
            "resourceType": ctx.get("resource_type"),
            "operationRunId": ctx.get("operation_run_id"),
        }
        
        if extra:
            payload.update(extra)
        
        # Print as JSON (CloudWatch will capture this)
        print(json.dumps(payload))
    
    @staticmethod
    def _generate_request_id() -> str:
        """Generate a unique request ID."""
        import random
        import string
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
        return f"req_{int(time.time() * 1000)}_{random_suffix}"

    @staticmethod
    def _generate_correlation_id() -> str:
        """Generate a unique correlation ID for cross-service tracing."""
        import random
        import string
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
        return f"corr_{int(time.time() * 1000)}_{random_suffix}"


# Convenience function to create a logger instance
def create_logger(layer: str = "aifl", service_name: Optional[str] = None) -> StructuredLogger:
    """
    Create a structured logger instance.
    
    Args:
        layer: Service layer identifier
        service_name: Name of the service
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(layer=layer, service_name=service_name)

