"""
Validation Module
=================

Validation system for classification responses.
"""

from .models import (
    AgentResult,
    AgentStatus,
    ValidationInfo,
    ValidationError,
    ErrorReport
)
from .validator import Validator
from .rules import create_rule, RULE_REGISTRY, BaseRule
from .pydantic_registry import register_pydantic_model, get_pydantic_model, list_registered_models, PYDANTIC_REGISTRY

__all__ = [
    'AgentResult',
    'AgentStatus',
    'ValidationInfo',
    'ValidationError',
    'ErrorReport',
    'Validator',
    'create_rule',
    'RULE_REGISTRY',
    'BaseRule',
    'register_pydantic_model',
    'get_pydantic_model',
    'list_registered_models',
    'PYDANTIC_REGISTRY',
]
