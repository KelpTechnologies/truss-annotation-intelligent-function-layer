"""
DSL API Client for Truss Annotation Data Service

Provides a unified interface for interacting with the Truss Annotation Data Service API
for classifier configurations, context data, and prompt templates.
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class DSLAPIError(Exception):
    """Custom exception for DSL API errors"""
    pass


class DSLAPIClient:
    """
    Client for interacting with the Truss Annotation Data Service API

    Supports all CRUD operations for:
    - Classifier configurations
    - Context data
    - Prompt templates
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, auth_headers: Optional[Dict[str, str]] = None, timeout: int = 30):
        """
        Initialize API client with authentication and environment selection

        Args:
            base_url: Base URL for the DSL API
            api_key: Optional API key for external endpoint authentication
            auth_headers: Optional dict of auth headers to pass through (e.g., Authorization, x-api-key)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        # Determine environment (default to 'prod' if not set)
        import os
        self.env = os.environ.get('DSL_ENV', 'prod').lower()
        if self.env not in ['dev', 'staging', 'prod']:
            logger.warning(f"Invalid DSL_ENV '{self.env}', defaulting to 'prod'")
            self.env = 'prod'
            
        logger.info(f"DSL API Client initialized for environment: {self.env.upper()}")

        # Set default headers
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-dsl-env': self.env  # Pass env header to API (if API supports it)
        }
        
        # Pass through auth headers from original request (Authorization, x-api-key)
        if auth_headers:
            headers.update(auth_headers)
            logger.info(f"Using pass-through auth headers: {list(auth_headers.keys())}")
        elif api_key:
            headers['x-api-key'] = api_key
        
        self.session.headers.update(headers)
        
        auth_type = 'Pass-through' if auth_headers else ('API Key' if api_key else 'None')
        logger.info(f"DSL API Client authentication: {auth_type}")

        # Add API key header if provided (for external endpoint)
        if api_key:
            headers['x-api-key'] = api_key

        self.session.headers.update(headers)

        auth_type = "API Key" if api_key else "No Authentication"
        logger.info(f"Initialized DSL API Client with base URL: {base_url} (Auth: {auth_type})")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with error handling

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional requests parameters

        Returns:
            Parsed JSON response

        Raises:
            DSLAPIError: For API errors
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()

            if response.content:
                return response.json()
            else:
                return {}

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            raise DSLAPIError(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from API: {e}")
            raise DSLAPIError(f"Invalid JSON response: {e}")

    # ============================================================================
    # Classifier Config Methods
    # ============================================================================

    def get_all_classifier_configs(self, property_type: Optional[str] = None,
                                 root_type_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all classifier configurations, optionally filtered

        Args:
            property_type: Filter by property type (optional)
            root_type_id: Filter by root type ID (optional)

        Returns:
            List of classifier configurations
        """
        params = {}
        if property_type:
            params['property_type'] = property_type
        if root_type_id:
            params['root_type_id'] = root_type_id

        response = self._make_request('GET', '/knowledge/classifier-configs', params=params)
        return response.get('data', [])

    def get_classifier_config(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Get a specific classifier configuration

        Args:
            property_type: Property type (e.g., 'material', 'condition')
            root_type_id: Root type ID

        Returns:
            Classifier configuration dictionary
        """
        endpoint = f'/knowledge/classifier-configs/{property_type}/{root_type_id}'
        return self._make_request('GET', endpoint)

    def create_classifier_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new classifier configuration

        Args:
            config_data: Configuration data dictionary

        Returns:
            Created configuration
        """
        return self._make_request('POST', '/knowledge/classifier-configs', json=config_data)

    def update_classifier_config(self, property_type: str, root_type_id: int,
                               config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing classifier configuration

        Args:
            property_type: Property type
            root_type_id: Root type ID
            config_data: Updated configuration data

        Returns:
            Updated configuration
        """
        endpoint = f'/knowledge/classifier-configs/{property_type}/{root_type_id}'
        return self._make_request('PUT', endpoint, json=config_data)

    def delete_classifier_config(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Delete a classifier configuration

        Args:
            property_type: Property type
            root_type_id: Root type ID

        Returns:
            Deletion confirmation
        """
        endpoint = f'/knowledge/classifier-configs/{property_type}/{root_type_id}'
        return self._make_request('DELETE', endpoint)

    # ============================================================================
    # Context Data Methods
    # ============================================================================

    def get_all_context_data(self, property_type: Optional[str] = None,
                           root_type_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all context data, optionally filtered

        Args:
            property_type: Filter by property type (optional)
            root_type_id: Filter by root type ID (optional)

        Returns:
            List of context data entries
        """
        params = {}
        if property_type:
            params['property_type'] = property_type
        if root_type_id:
            params['root_type_id'] = root_type_id

        response = self._make_request('GET', '/knowledge/context-data', params=params)
        return response.get('data', [])

    def get_context_data(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Get specific context data

        Args:
            property_type: Property type
            root_type_id: Root type ID

        Returns:
            Context data dictionary
        """
        endpoint = f'/knowledge/context-data/{property_type}/{root_type_id}'
        return self._make_request('GET', endpoint)

    def create_context_data(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new context data

        Args:
            context_data: Context data dictionary

        Returns:
            Created context data
        """
        return self._make_request('POST', '/knowledge/context-data', json=context_data)

    def update_context_data(self, property_type: str, root_type_id: int,
                          context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing context data

        Args:
            property_type: Property type
            root_type_id: Root type ID
            context_data: Updated context data

        Returns:
            Updated context data
        """
        endpoint = f'/knowledge/context-data/{property_type}/{root_type_id}'
        return self._make_request('PUT', endpoint, json=context_data)

    def delete_context_data(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Delete context data

        Args:
            property_type: Property type
            root_type_id: Root type ID

        Returns:
            Deletion confirmation
        """
        endpoint = f'/knowledge/context-data/{property_type}/{root_type_id}'
        return self._make_request('DELETE', endpoint)

    # ============================================================================
    # Prompt Template Methods
    # ============================================================================

    def get_all_prompt_templates(self, template_type: Optional[str] = None,
                               model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all prompt templates, optionally filtered

        Args:
            template_type: Filter by template type (optional)
            model_name: Filter by model name (optional)

        Returns:
            List of prompt templates
        """
        params = {}
        if template_type:
            params['template_type'] = template_type
        if model_name:
            params['model_name'] = model_name

        response = self._make_request('GET', '/knowledge/prompt-templates', params=params)
        return response.get('data', [])

    def get_prompt_template(self, template_id: str) -> Dict[str, Any]:
        """
        Get a specific prompt template

        Args:
            template_id: Template ID

        Returns:
            Prompt template dictionary
        """
        endpoint = f'/knowledge/prompt-templates/{template_id}'
        return self._make_request('GET', endpoint)

    def create_prompt_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new prompt template

        Args:
            template_data: Template data dictionary

        Returns:
            Created template
        """
        return self._make_request('POST', '/knowledge/prompt-templates', json=template_data)

    def update_prompt_template(self, template_id: str, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing prompt template

        Args:
            template_id: Template ID
            template_data: Updated template data

        Returns:
            Updated template
        """
        endpoint = f'/knowledge/prompt-templates/{template_id}'
        return self._make_request('PUT', endpoint, json=template_data)

    def delete_prompt_template(self, template_id: str) -> Dict[str, Any]:
        """
        Delete a prompt template

        Args:
            template_id: Template ID

        Returns:
            Deletion confirmation
        """
        endpoint = f'/knowledge/prompt-templates/{template_id}'
        return self._make_request('DELETE', endpoint)

    # ============================================================================
    # Health Check Methods
    # ============================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the API

        Returns:
            Health status information
        """
        try:
            response = self._make_request('GET', '/health')
            return {"status": "healthy", "response": response}
        except DSLAPIError:
            return {"status": "unhealthy", "error": "API unreachable"}

    def test_authentication(self) -> bool:
        """
        Test if API connection is working

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Try to get classifier configs (tests basic connectivity)
            self.get_all_classifier_configs()
            return True
        except DSLAPIError:
            return False

    # ============================================================================
    # Taxonomy Lookup Methods
    # ============================================================================

    def lookup_root(self, property_type: str, value: str, brand: Optional[str] = None,
                    root_type: Optional[str] = None, partition: Optional[str] = None) -> Dict[str, Any]:
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

        partition_path = partition or "bags"
        endpoint = f'/{partition_path}/knowledge/lookup-root'
        return self._make_request('GET', endpoint, params=params)








