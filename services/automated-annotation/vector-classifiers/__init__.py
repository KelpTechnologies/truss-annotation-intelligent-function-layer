"""
Vector-based model classification pipeline.

This module provides visual classification using:
1. Pre-computed image vectors stored by the image processing pipeline
2. Pinecone similarity search
3. DynamoDB metadata lookup
4. Majority voting algorithm
"""

from .model_classifier_pipeline import classify_image, vectorize_image, query_pinecone, query_dynamodb, perform_voting, normalize_brand_to_namespace

__all__ = ['classify_image', 'vectorize_image', 'query_pinecone', 'query_dynamodb', 'perform_voting', 'normalize_brand_to_namespace']

