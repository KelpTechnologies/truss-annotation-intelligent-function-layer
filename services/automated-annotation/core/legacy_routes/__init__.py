"""
Legacy route handlers for the automated annotation service.

DEPRECATED: This folder contains legacy route handlers that will be removed
once the legacy endpoints are decommissioned. All new endpoints should use
the orchestration_api_handlers pattern instead.
"""

from .handlers import handle_classification, handle_health, handle_options
from .agent_handlers import handle_agent_execution

__all__ = [
    "handle_classification",
    "handle_health",
    "handle_options",
    "handle_agent_execution",
]
