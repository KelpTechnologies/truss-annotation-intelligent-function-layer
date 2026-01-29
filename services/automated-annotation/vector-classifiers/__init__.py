"""
Vector-based model classification pipeline.

This module provides visual classification using:
1. Pre-computed image vectors stored by the image processing pipeline
2. Pinecone similarity search
3. BigQuery metadata lookup (model_knowledge_display)
4. Majority voting algorithm
"""

from .model_classifier_pipeline import classify_image, query_pinecone, query_bigquery, perform_voting, normalize_brand_to_namespace
from .bigquery_utils import get_bigquery_client, query_model_metadata

__all__ = [
    'classify_image',
    'query_pinecone',
    'query_bigquery',
    'perform_voting',
    'normalize_brand_to_namespace',
    'get_bigquery_client',
    'query_model_metadata',
]

