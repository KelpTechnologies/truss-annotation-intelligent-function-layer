"""
LLM Annotation System Package
=============================

Generalized LangChain-powered agent for fashion product metadata annotation.
"""

from .base_agent import LLMAnnotationAgent, ClassificationResponse, PredictionScore
from .validation import (
    AgentResult,
    AgentStatus,
    ValidationInfo,
    ValidationError,
    ErrorReport,
    Validator
)

__version__ = "1.0.0"
__all__ = [
    "LLMAnnotationAgent",
    "ClassificationResponse",
    "PredictionScore",
    "AgentResult",
    "AgentStatus",
    "ValidationInfo",
    "ValidationError",
    "ErrorReport",
    "Validator",
]
