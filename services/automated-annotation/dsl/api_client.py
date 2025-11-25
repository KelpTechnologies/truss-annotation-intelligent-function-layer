"""
Simplified DSL API Client for Truss Annotation Data Service
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class DSLAPIError(Exception):
    """Custom exception for DSL API errors"""
    pass


class DSLAPIClient:
    """HTTP client for the Truss Annotation Data Service API."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if api_key:
            headers['x-api-key'] = api_key
        self.session.headers.update(headers)

        logger.info(f"Initialized DSL API Client with base URL: {base_url} (Auth: {'API Key' if api_key else 'None'})")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            raise DSLAPIError(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from API: {e}")
            raise DSLAPIError(f"Invalid JSON response: {e}")

    # Classifier Config
    def get_classifier_config(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        endpoint = f'/knowledge/classifier-configs/{property_type}/{root_type_id}'
        return self._make_request('GET', endpoint)

    # Context Data
    def get_context_data(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        endpoint = f'/knowledge/context-data/{property_type}/{root_type_id}'
        return self._make_request('GET', endpoint)

    # Prompt Templates
    def get_prompt_template(self, template_id: str) -> Dict[str, Any]:
        endpoint = f'/knowledge/prompt-templates/{template_id}'
        return self._make_request('GET', endpoint)

    # Root Lookup
    def lookup_root(self, property_type: str, value: str, brand: Optional[str] = None, root_type: Optional[str] = None, partition: Optional[str] = None) -> Dict[str, Any]:
        """
        Lookup root taxonomy value from child value for models or materials.
        
        Args:
            property_type: "model" or "material"
            value: The child model/material name to lookup
            brand: Optional brand filter for models
            root_type: Optional root_type filter
            partition: Optional partition path (e.g., "bags", "api", "apparel", "footwear", "all")
                      If not provided, defaults to "bags"
            
        Returns:
            Dictionary with root information
        """
        params = {
            'property_type': property_type,
            'value': value
        }
        if brand:
            params['brand'] = brand
        if root_type:
            params['root_type'] = root_type
        
        # Build endpoint with partition path
        # Default to "bags" if partition not provided
        partition_path = partition or "bags"
        endpoint = f'/{partition_path}/knowledge/lookup-root'
        return self._make_request('GET', endpoint, params=params)

    # Health
    def health_check(self) -> Dict[str, Any]:
        try:
            response = self._make_request('GET', '/health')
            return {"status": "healthy", "response": response}
        except DSLAPIError:
            return {"status": "unhealthy", "error": "API unreachable"}
