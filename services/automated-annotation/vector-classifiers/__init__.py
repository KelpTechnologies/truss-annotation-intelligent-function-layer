"""
Vector-based model classification pipeline.

This module provides visual classification using:
1. Image vectorization (FashionCLIP)
2. Pinecone similarity search
3. DynamoDB metadata lookup
4. Majority voting algorithm
"""

from .model_classifier_pipeline import classify_image, vectorize_image, query_pinecone, query_dynamodb, perform_voting

__all__ = ['classify_image', 'vectorize_image', 'query_pinecone', 'query_dynamodb', 'perform_voting']

