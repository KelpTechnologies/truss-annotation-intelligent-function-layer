"""
Stage-aware API URL configuration.
URLs are selected based on the STAGE environment variable (dev, staging, prod).
"""

import os

# URL configurations per stage
STAGE_URLS = {
    "dev": {
        "DSL_API_BASE_URL": "https://1hc7v2pi9d.execute-api.eu-west-2.amazonaws.com/staging",
        "ANNOTATION_DSL_API_BASE_URL": "https://xu89om0525.execute-api.eu-west-2.amazonaws.com/staging",
        "IFL_API_BASE_URL": "https://ituo0x1fj5.execute-api.eu-west-2.amazonaws.com/staging",
        "ANNOTATION_IFL_API_BASE_URL": "https://mabi7l7lvl.execute-api.eu-west-2.amazonaws.com/staging",
    },
    "staging": {
        "DSL_API_BASE_URL": "https://1hc7v2pi9d.execute-api.eu-west-2.amazonaws.com/staging",
        "ANNOTATION_DSL_API_BASE_URL": "https://xu89om0525.execute-api.eu-west-2.amazonaws.com/staging",
        "IFL_API_BASE_URL": "https://ituo0x1fj5.execute-api.eu-west-2.amazonaws.com/staging",
        "ANNOTATION_IFL_API_BASE_URL": "https://mabi7l7lvl.execute-api.eu-west-2.amazonaws.com/staging",
    },
    "prod": {
        "DSL_API_BASE_URL": "https://k7sb288446.execute-api.eu-west-2.amazonaws.com/prod",
        "ANNOTATION_DSL_API_BASE_URL": "https://quc8rco4za.execute-api.eu-west-2.amazonaws.com/prod",
        "IFL_API_BASE_URL": "https://74e1x6rfk6.execute-api.eu-west-2.amazonaws.com/prod",
        "ANNOTATION_IFL_API_BASE_URL": "https://ecmeogytm1.execute-api.eu-west-2.amazonaws.com/prod",
    },
}


def get_stage():
    """Get the current deployment stage from environment."""
    return os.getenv("STAGE", "dev")


def get_url(url_key):
    """
    Get a URL for the current stage.
    Falls back to staging URLs if stage is not found.
    
    Args:
        url_key: One of DSL_API_BASE_URL, ANNOTATION_DSL_API_BASE_URL, 
                 IFL_API_BASE_URL, ANNOTATION_IFL_API_BASE_URL
    
    Returns:
        The URL string for the current stage
    """
    stage = get_stage()
    stage_config = STAGE_URLS.get(stage, STAGE_URLS["staging"])
    return stage_config.get(url_key, STAGE_URLS["staging"].get(url_key))


def get_all_urls():
    """Get all URLs for the current stage as a dictionary."""
    stage = get_stage()
    return STAGE_URLS.get(stage, STAGE_URLS["staging"]).copy()


# Convenience functions for each URL
def get_dsl_url():
    return get_url("DSL_API_BASE_URL")


def get_annotation_dsl_url():
    return get_url("ANNOTATION_DSL_API_BASE_URL")


def get_ifl_url():
    return get_url("IFL_API_BASE_URL")


def get_annotation_ifl_url():
    return get_url("ANNOTATION_IFL_API_BASE_URL")
