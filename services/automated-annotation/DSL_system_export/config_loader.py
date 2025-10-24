"""
Configuration Loader Abstraction Layer

Provides a unified interface for loading classifier configurations, prompt templates,
and context data from either local files or the DSL API.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Unified configuration loader supporting both local and API modes

    Handles loading of:
    - Classifier configurations
    - Prompt templates
    - Context data
    """

    def __init__(self, mode: str = 'api', api_client: Optional[Any] = None,
                 local_config_dir: str = 'config'):
        """
        Initialize configuration loader

        Args:
            mode: 'local' or 'api' (default: 'api' for cloud deployment)
            api_client: DSLAPIClient instance (required for API mode)
            local_config_dir: Directory containing local config files
        """
        self.mode = mode.lower()
        self.api_client = api_client
        self.local_config_dir = Path(local_config_dir)

        if self.mode == 'api' and not api_client:
            raise ValueError("api_client is required when mode='api'")
        if self.mode not in ['local', 'api']:
            raise ValueError("mode must be 'local' or 'api'")

        logger.info(f"Initialized ConfigLoader in {mode} mode")

    def load_classifier_config(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Load classifier configuration

        Args:
            property_type: Property type (e.g., 'material', 'condition')
            root_type_id: Root type ID

        Returns:
            Classifier configuration dictionary
        """
        if self.mode == 'api':
            return self._load_classifier_config_api(property_type, root_type_id)
        else:
            return self._load_classifier_config_local(property_type, root_type_id)

    def load_prompt_template(self, template_id: str) -> Dict[str, Any]:
        """
        Load prompt template

        Args:
            template_id: Template identifier

        Returns:
            Prompt template dictionary
        """
        if self.mode == 'api':
            return self._load_prompt_template_api(template_id)
        else:
            return self._load_prompt_template_local(template_id)

    def load_context_data(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Load context data

        Args:
            property_type: Property type
            root_type_id: Root type ID

        Returns:
            Context data dictionary
        """
        if self.mode == 'api':
            return self._load_context_data_api(property_type, root_type_id)
        else:
            return self._load_context_data_local(property_type, root_type_id)

    # ============================================================================
    # API Loading Methods
    # ============================================================================

    def _load_classifier_config_api(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """Load classifier config from API"""
        logger.debug(f"Loading classifier config from API: {property_type}/{root_type_id}")
        return self.api_client.get_classifier_config(property_type, root_type_id)

    def _load_prompt_template_api(self, template_id: str) -> Dict[str, Any]:
        """Load prompt template from API"""
        logger.debug(f"Loading prompt template from API: {template_id}")
        return self.api_client.get_prompt_template(template_id)

    def _load_context_data_api(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """Load context data from API with specific fallback to root_type_id=0"""
        logger.debug(f"Loading context data from API: {property_type}/{root_type_id}")

        try:
            # First try the specific root_type_id
            return self.api_client.get_context_data(property_type, root_type_id)
        except Exception as e:
            print(f"[WARNING] Context data not found for {property_type}/{root_type_id}, trying universal context ({property_type}/0)")

            # If that fails, try universal context data (root_type_id=0) for the same property
            try:
                result = self.api_client.get_context_data(property_type, 0)
                print(f"[INFO] Successfully loaded universal context data for {property_type}/0")
                return result
            except Exception as e2:
                print(f"[ERROR] No context data found for {property_type} (tried root_type_id {root_type_id} and 0)")
                raise e2

    # ============================================================================
    # Local File Loading Methods
    # ============================================================================

    def _load_classifier_config_local(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Load classifier configuration from local JSON file

        Expected structure: config/classifier_config.json
        """
        config_file = self.local_config_dir / 'classifier_config.json'

        if not config_file.exists():
            raise FileNotFoundError(f"Local config file not found: {config_file}")

        logger.debug(f"Loading classifier config from local file: {config_file}")

        with open(config_file, 'r') as f:
            all_configs = json.load(f)

        # Navigate to specific config
        if property_type not in all_configs:
            raise KeyError(f"Property type '{property_type}' not found in local config")

        property_configs = all_configs[property_type]

        if str(root_type_id) not in property_configs:
            raise KeyError(f"Root type ID '{root_type_id}' not found for property '{property_type}'")

        return property_configs[str(root_type_id)]

    def _load_prompt_template_local(self, template_id: str) -> Dict[str, Any]:
        """
        Load prompt template from local JSON file

        Expected structure: config/prompts/{template_id}.json
        """
        template_file = self.local_config_dir / 'prompts' / f'{template_id}.json'

        if not template_file.exists():
            raise FileNotFoundError(f"Local template file not found: {template_file}")

        logger.debug(f"Loading prompt template from local file: {template_file}")

        with open(template_file, 'r') as f:
            return json.load(f)

    def _load_context_data_local(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Load context data from local CSV file

        Expected structure: data/{property_type}_llm_context.csv
        """
        # Import pandas locally to avoid hard dependency
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for local context data loading")

        context_file = Path('data') / f'{property_type}_llm_context.csv'

        if not context_file.exists():
            raise FileNotFoundError(f"Local context file not found: {context_file}")

        logger.debug(f"Loading context data from local file: {context_file}")

        # Read CSV and filter by root_type_id
        df = pd.read_csv(context_file)

        # Filter by root_type_id with smart fallback
        filtered_df = df[df['root_type_id'] == root_type_id]

        # If no specific matches, try universal (root_type_id=0)
        if filtered_df.empty:
            filtered_df = df[df['root_type_id'] == 0]

        if filtered_df.empty:
            raise ValueError(f"No context data found for {property_type} with root_type_id {root_type_id}")

        # Determine context mode (use most common or first)
        context_mode = filtered_df['context_mode'].iloc[0] if 'context_mode' in filtered_df.columns else 'full-context'

        # Apply context mode filtering
        if context_mode == 'reduced-taxonomy':
            filtered_df = filtered_df[filtered_df['context_mode'] == 'REDUCED']
        elif context_mode == 'full-context':
            filtered_df = filtered_df[filtered_df['context_mode'] == 'FULL']

        # Build structured context data
        items = []
        descriptions = {}

        name_col = f'{property_type}_name'
        id_col = f'{property_type}_id'
        desc_col = 'llm_description'

        for _, row in filtered_df.iterrows():
            item_id = int(row[id_col])
            item_name = row[name_col]
            item_key = f"ID {item_id}: {item_name}"
            items.append(item_key)

            if desc_col in row and pd.notna(row[desc_col]):
                descriptions[item_key] = str(row[desc_col])

        return {
            "materials": items,  # Keep 'materials' key for backward compatibility
            "descriptions": descriptions,
            "mode": context_mode,
            "root_type_id": root_type_id,
            "count": len(items),
            "property_type": property_type
        }

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def list_available_configs(self) -> Dict[str, Any]:
        """
        List all available configurations (for debugging)

        Returns:
            Dictionary with available configs, templates, and context data
        """
        if self.mode == 'api':
            return {
                'classifier_configs': self.api_client.get_all_classifier_configs(),
                'prompt_templates': self.api_client.get_all_prompt_templates(),
                'context_data': self.api_client.get_all_context_data()
            }
        else:
            # For local mode, we'd need to scan directories
            # This is a simplified version
            return {
                'classifier_configs': 'Scan config/classifier_config.json',
                'prompt_templates': 'Scan config/prompts/',
                'context_data': 'Scan data/'
            }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a classifier configuration

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'model', 'default_context_mode', 'prompt_template',
            'context_data', 'property_column', 'root_type_id'
        ]

        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False

        return True
