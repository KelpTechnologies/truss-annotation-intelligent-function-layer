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

LOG_SCHEMA_VERSION = 2.1

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

