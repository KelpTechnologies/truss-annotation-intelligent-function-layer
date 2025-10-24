"""
LLM Annotation System Package
=============================

Generalized LangChain-powered classifier for fashion product metadata annotation.
"""

from .base_classifier import LLMAnnotationAgent, ClassificationResponse, PredictionScore

__version__ = "1.0.0"
__all__ = ["LLMAnnotationAgent", "ClassificationResponse", "PredictionScore"]
