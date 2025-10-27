"""
Config loader for classifier configs, prompt templates, and context data (API-first)
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    def __init__(self, mode: str = 'api', api_client: Optional[Any] = None):
        self.mode = mode.lower()
        self.api_client = api_client
        if self.mode == 'api' and not api_client:
            raise ValueError("api_client is required when mode='api'")
        if self.mode not in ['api']:
            raise ValueError("Only API mode is supported in Lambda")
        logger.info(f"Initialized ConfigLoader in {mode} mode")

    def load_classifier_config(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        return self.api_client.get_classifier_config(property_type, root_type_id)

    def load_prompt_template(self, template_id: str) -> Dict[str, Any]:
        return self.api_client.get_prompt_template(template_id)

    def load_context_data(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        try:
            return self.api_client.get_context_data(property_type, root_type_id)
        except Exception:
            # Fallback to universal context (root_type_id=0)
            return self.api_client.get_context_data(property_type, 0)
