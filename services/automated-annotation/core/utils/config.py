"""
Configuration constants for the Lambda function.
"""

# Minimum confidence threshold for model classification
MIN_MODEL_CONFIDENCE_THRESHOLD = 50.0

# Category-specific configuration
CATEGORY_CONFIG = {
    "bags": {
        "default_brand": "jacquemus",
        "vector_index": "mfc-classifier-bags-models-userdata",
        "default_namespace": "jacquemus",
        "model_classification_enabled": True,
    },
    "footwear": {
        "default_brand": None,
        "vector_index": None,
        "default_namespace": None,
        "model_classification_enabled": False,
    },
    "apparel": {
        "default_brand": None,
        "vector_index": None,
        "default_namespace": None,
        "model_classification_enabled": False,
    },
}
