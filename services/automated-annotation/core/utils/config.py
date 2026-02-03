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
        "default_namespace": "jacquemus"
    }
    # Add other categories as needed
}
