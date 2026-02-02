"""
Configuration Loader Abstraction Layer

Provides a unified interface for loading classifier configurations, prompt templates,
and context data from:
1. DSL API ('api' mode)
2. Local Files ('local' mode)
3. Direct DynamoDB ('dynamo' mode)
"""

import json
import logging
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

# New table name templates
NEW_TABLE_NAMES = {
    "configs": "truss-agent-configs-{env}",
    "schemas": "truss-agent-schemas-{env}",
    "prompt_templates": "truss-agent-prompt-templates-{env}"
}

# Legacy table name templates
LEGACY_TABLE_NAMES = {
    "configs": "truss-annotation-classifier-configs-{env}",
    "schemas": "truss-annotation-context-data-{env}",
    "prompt_templates": "truss-annotation-prompt-templates-{env}"
}

# CSV config table name template
CSV_CONFIG_TABLE_NAME = "linesheet-upload-csv-configs-{env}"


class ConfigLoader:
    """
    Unified configuration loader supporting local, API, and DynamoDB modes.
    """

    def __init__(self, mode: str = 'api', api_client: Optional[Any] = None,
                 local_config_dir: str = 'config', env: str = 'dev', fallback_env: str = 'prod'):
        """
        Initialize configuration loader

        Args:
            mode: 'local', 'api', or 'dynamo'
            api_client: DSLAPIClient instance (required for API mode)
            local_config_dir: Directory containing local config files (for local mode)
            env: Environment for DynamoDB tables ('dev', 'staging', 'prod') - used in 'dynamo' mode
            fallback_env: Environment to use for DynamoDB fallback when API fails (default: 'staging')
        """
        self.mode = mode.lower()
        self.api_client = api_client
        self.local_config_dir = Path(local_config_dir)
        self.env = env
        self.fallback_env = fallback_env

        if self.mode == 'api' and not api_client:
            raise ValueError("api_client is required when mode='api'")
        if self.mode not in ['local', 'api', 'dynamo']:
            raise ValueError("mode must be 'local', 'api', or 'dynamo'")

        if self.mode == 'dynamo':
            self.dynamodb = boto3.resource('dynamodb')
            # New tables
            self.new_config_table = self.dynamodb.Table(NEW_TABLE_NAMES["configs"].format(env=self.env))
            self.new_schema_table = self.dynamodb.Table(NEW_TABLE_NAMES["schemas"].format(env=self.env))
            self.new_template_table = self.dynamodb.Table(NEW_TABLE_NAMES["prompt_templates"].format(env=self.env))
            # Legacy tables (for backward compatibility)
            self.config_table = self.dynamodb.Table(LEGACY_TABLE_NAMES["configs"].format(env=self.env))
            self.template_table = self.dynamodb.Table(LEGACY_TABLE_NAMES["prompt_templates"].format(env=self.env))
            self.context_table = self.dynamodb.Table(LEGACY_TABLE_NAMES["schemas"].format(env=self.env))
            # CSV config table
            self.csv_config_table = self.dynamodb.Table(CSV_CONFIG_TABLE_NAME.format(env=self.env))
            logger.info(f"Initialized ConfigLoader in DynamoDB mode (env={self.env})")
        else:
            logger.info(f"Initialized ConfigLoader in {mode} mode")

    def load_classifier_config(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """Load classifier configuration"""
        if self.mode == 'api':
            config = self._load_classifier_config_api(property_type, root_type_id)
        elif self.mode == 'dynamo':
            config = self._load_classifier_config_dynamo(property_type, root_type_id)
        else:
            config = self._load_classifier_config_local(property_type, root_type_id)

        # Standardize config structure
        if isinstance(config, dict) and 'model_config' in config:
            model_config = config.pop('model_config', {})
            config.update(model_config)

        if 'template_id' in config and 'prompt_template' not in config:
            config['prompt_template'] = config['template_id']

        return config

    def load_prompt_template(self, template_id: str) -> Dict[str, Any]:
        """Load prompt template"""
        if self.mode == 'api':
            return self._load_prompt_template_api(template_id)
        elif self.mode == 'dynamo':
            return self._load_prompt_template_dynamo(template_id)
        else:
            return self._load_prompt_template_local(template_id)

    def load_context_data(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """Load context data"""
        if self.mode == 'api':
            return self._load_context_data_api(property_type, root_type_id)
        elif self.mode == 'dynamo':
            return self._load_context_data_dynamo(property_type, root_type_id)
        else:
            return self._load_context_data_local(property_type, root_type_id)

    def load_validation_config(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Load validation config for a property/root_type.
        
        Sources:
        1. Embedded in classifier config under 'validation_config' key
        
        Args:
            property_type: Property type (e.g., 'material', 'condition')
            root_type_id: Root type ID
            
        Returns:
            Validation configuration dictionary
            
        Raises:
            ValueError: If validation config is not found in classifier config
        """
        classifier_config = self.load_classifier_config(property_type, root_type_id)
        validation_config = classifier_config.get('validation_config')
        
        if not validation_config:
            raise ValueError(
                f"validation_config not found in classifier config for {property_type}/{root_type_id}. "
                "Validation config is required and must be embedded in the classifier config."
            )
        
        logger.debug(f"Loaded validation config from classifier config for {property_type}/{root_type_id}")
        return validation_config

    def load_stopping_conditions(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """
        Load stopping conditions for a property/root_type.
        
        Sources:
        1. Embedded in classifier config under 'stopping_conditions' key
        
        Args:
            property_type: Property type (e.g., 'material', 'condition')
            root_type_id: Root type ID
            
        Returns:
            Stopping conditions configuration dictionary
            
        Raises:
            ValueError: If stopping conditions are not found in classifier config
        """
        classifier_config = self.load_classifier_config(property_type, root_type_id)
        stopping_conditions = classifier_config.get('stopping_conditions')
        
        if not stopping_conditions:
            raise ValueError(
                f"stopping_conditions not found in classifier config for {property_type}/{root_type_id}. "
                "Stopping conditions are required and must be embedded in the classifier config."
            )
        
        logger.debug(f"Loaded stopping conditions from classifier config for {property_type}/{root_type_id}")
        return stopping_conditions

    # ============================================================================
    # API Loading Methods
    # ============================================================================

    def _load_classifier_config_api(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        logger.debug(f"Loading classifier config from API: {property_type}/{root_type_id}")
        return self.api_client.get_classifier_config(property_type, root_type_id)

    def _load_prompt_template_api(self, template_id: str) -> Dict[str, Any]:
        logger.debug(f"Loading prompt template from API: {template_id}")
        return self.api_client.get_prompt_template(template_id)

    def _load_context_data_api(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        logger.debug(f"Loading context data from API: {property_type}/{root_type_id}")
        try:
            return self.api_client.get_context_data(property_type, root_type_id)
        except Exception:
            print(f"[WARNING] Context data not found for {property_type}/{root_type_id}, trying universal context ({property_type}/0)")
            try:
                return self.api_client.get_context_data(property_type, 0)
            except Exception as e2:
                print(f"[ERROR] No context data found for {property_type}")
                raise e2

    # ============================================================================
    # DynamoDB Loading Methods
    # ============================================================================

    def _load_classifier_config_dynamo(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        logger.debug(f"Loading classifier config from DynamoDB: {property_type}/{root_type_id}")
        try:
            resp = self.config_table.get_item(
                Key={'property_type': property_type, 'root_type_id': int(root_type_id)}
            )
            item = resp.get('Item')
            if not item:
                raise ValueError(f"Config not found in DynamoDB for {property_type}/{root_type_id}")
            return json.loads(json.dumps(item, default=self._decimal_default))
        except Exception as e:
            logger.error(f"Error loading config from DynamoDB: {e}")
            raise

    def _load_prompt_template_dynamo(self, template_id: str) -> Dict[str, Any]:
        logger.debug(f"Loading prompt template from DynamoDB: {template_id}")
        try:
            resp = self.template_table.get_item(Key={'template_id': template_id})
            item = resp.get('Item')
            if not item:
                raise ValueError(f"Template {template_id} not found in DynamoDB")
            
            # Convert Decimals
            clean_item = json.loads(json.dumps(item, default=self._decimal_default))
            
            # Return 'template_content' if it exists (standard structure), else the whole item
            if 'template_content' in clean_item:
                return clean_item['template_content']
            
            return clean_item
        except Exception as e:
            logger.error(f"Error loading template from DynamoDB: {e}")
            raise

    def _load_context_data_dynamo(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        logger.debug(f"Loading context data from DynamoDB: {property_type}/{root_type_id}")
        try:
            # 1. NEW SCHEMA (v2): Direct Get Item
            # PK: property_type, SK: root_type_id (Number)
            try:
                resp = self.context_table.get_item(
                    Key={'property_type': property_type, 'root_type_id': int(root_type_id)}
                )
                if 'Item' in resp:
                    item = json.loads(json.dumps(resp['Item'], default=self._decimal_default))
                    # If it's the new aggregated format, return it directly
                    if 'context_content' in item:
                        return item
                    # If it's a single item (legacy row in new table?), wrap it
                    return self._wrap_single_item(item, property_type, root_type_id)
            except Exception as e:
                logger.warning(f"Direct get_item failed (might be legacy schema): {e}")

            # 2. OLD SCHEMA / GSI Fallback
            # If direct get failed or didn't return what we wanted, try the GSI approach
            # useful for finding multiple individual items (Legacy Prod Schema)
            
            items = []
            try:
                resp = self.context_table.query(
                    IndexName='RootTypeIndex',
                    KeyConditionExpression=Key('root_type_id').eq(int(root_type_id))
                )
                items = resp.get('Items', [])
            except Exception as gsi_error:
                logger.warning(f"GSI query failed: {gsi_error}")
                # If GSI fails, we might be in Dev environment without GSI. 
                # In that case, we rely on the direct get_item above. 
                # If that also returned nothing, then truly nothing exists.
            
            # Filter by property_type from GSI results
            filtered_items = [item for item in items if item.get('property_type') == property_type]
            
            if not filtered_items:
                # 3. Universal Fallback (root_type_id=0)
                # Try direct get for 0
                try:
                    resp_univ = self.context_table.get_item(
                         Key={'property_type': property_type, 'root_type_id': 0}
                    )
                    if 'Item' in resp_univ:
                         item = json.loads(json.dumps(resp_univ['Item'], default=self._decimal_default))
                         if 'context_content' in item:
                             print(f"[INFO] Using universal context for {property_type} (root_type_id=0)")
                             return item
                except Exception:
                    pass

                raise ValueError(f"Context not found in DynamoDB for {property_type}/{root_type_id}")

            # Aggregate items into standard structure (Legacy aggregation)
            aggregated_content = {}
            context_mode = "REDUCED" # Default
            
            for item in filtered_items:
                item = json.loads(json.dumps(item, default=self._decimal_default))
                item_id = item.get('item_id')
                
                # Check if it's an individual item (has name/model_name)
                # Model items have 'model_name' and 'brand'
                # Other items have '{property}_name'
                name = item.get('model_name') or item.get(f'{property_type}_name') or item.get('name')
                
                if item_id and name:
                    # Structure: {id: {name: ..., description: ..., brand: ...}}
                    aggregated_content[str(item_id)] = {
                        "name": name,
                        "description": item.get('llm_description', ''),
                        "brand": item.get('brand', '')
                    }
                    if 'context_mode' in item:
                        context_mode = item['context_mode']

            return {
                "property_type": property_type,
                "root_type_id": root_type_id,
                "context_mode": context_mode,
                "context_content": aggregated_content,
                "source": "dynamo_aggregation",
                "count": len(aggregated_content)
            }

        except Exception as e:
            logger.error(f"Error loading context from DynamoDB: {e}")
            raise

    def _decimal_default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return str(obj)
    
    def _get_dynamodb_tables_for_env(self, env: str):
        """Get DynamoDB table references for a specific environment."""
        dynamodb = boto3.resource('dynamodb')
        return {
            'new_config_table': dynamodb.Table(NEW_TABLE_NAMES["configs"].format(env=env)),
            'new_schema_table': dynamodb.Table(NEW_TABLE_NAMES["schemas"].format(env=env)),
            'new_template_table': dynamodb.Table(NEW_TABLE_NAMES["prompt_templates"].format(env=env))
        }
    
    def load_agent_config(self, config_id: str) -> Dict[str, Any]:
        """
        Load agent config from new truss-agent-configs table by config_id.
        
        Args:
            config_id: Configuration ID (e.g., 'material-30')
            
        Returns:
            Config dictionary
        """
        if self.mode == 'dynamo':
            try:
                resp = self.new_config_table.get_item(Key={'config_id': config_id})
                item = resp.get('Item')
                if not item:
                    raise ValueError(f"Config {config_id} not found in DynamoDB")
                return json.loads(json.dumps(item, default=self._decimal_default))
            except Exception as e:
                logger.error(f"Error loading config from DynamoDB: {e}")
                raise
        else:
            raise ValueError(f"load_agent_config only supports 'dynamo' mode, current mode: {self.mode}")
    
    def load_agent_schema(self, schema_id: str) -> Dict[str, Any]:
        """
        Load schema from new truss-agent-schemas table by schema_id.
        
        Args:
            schema_id: Schema ID (e.g., 'material-30')
            
        Returns:
            Schema dictionary with schema_content and schema_metadata
        """
        if self.mode == 'dynamo':
            try:
                resp = self.new_schema_table.get_item(Key={'schema_id': schema_id})
                item = resp.get('Item')
                if not item:
                    raise ValueError(f"Schema {schema_id} not found in DynamoDB")
                return json.loads(json.dumps(item, default=self._decimal_default))
            except Exception as e:
                logger.error(f"Error loading schema from DynamoDB: {e}")
                raise
        else:
            raise ValueError(f"load_agent_schema only supports 'dynamo' mode, current mode: {self.mode}")
    
    def load_agent_prompt_template(self, prompt_template_id: str) -> Dict[str, Any]:
        """
        Load prompt template from new truss-agent-prompt-templates table.
        
        Args:
            prompt_template_id: Template ID (e.g., 'material-bag-v1')
            
        Returns:
            Template content dictionary
        """
        if self.mode == 'dynamo':
            try:
                resp = self.new_template_table.get_item(Key={'prompt_template_id': prompt_template_id})
                item = resp.get('Item')
                if not item:
                    raise ValueError(f"Template {prompt_template_id} not found in DynamoDB")
                clean_item = json.loads(json.dumps(item, default=self._decimal_default))
                # Return template_content if it exists, else the whole item
                return clean_item.get('template_content', clean_item)
            except Exception as e:
                logger.error(f"Error loading template from DynamoDB: {e}")
                raise
        else:
            raise ValueError(f"load_agent_prompt_template only supports 'dynamo' mode, current mode: {self.mode}")
    
    def _load_full_config_via_api(self, config_id: str) -> Dict[str, Any]:
        """Load full config via API (if API client has these methods)."""
        # Try to use API client methods if they exist
        if hasattr(self.api_client, 'get_agent_config'):
            config = self.api_client.get_agent_config(config_id)
            schema_id = config.get('schema_id')
            prompt_template_id = config.get('prompt_template_id')
            
            # Load schema only if schema_id is provided
            schema = None
            if schema_id:
                if hasattr(self.api_client, 'get_agent_schema'):
                    schema = self.api_client.get_agent_schema(schema_id)
                else:
                    raise ValueError("API client missing get_agent_schema method")
            
            if hasattr(self.api_client, 'get_agent_prompt_template') and prompt_template_id:
                template = self.api_client.get_agent_prompt_template(prompt_template_id)
            else:
                raise ValueError("API client missing get_agent_prompt_template method")
            
            return self._build_full_config_bundle(config_id, config, schema, template, prompt_template_id)
        else:
            raise ValueError("API client does not support new agent config endpoints")
    
    def _load_full_config_via_dynamo(self, config_id: str, env: str = None) -> Dict[str, Any]:
        """Load full config via DynamoDB."""
        if env is None:
            env = self.env
        
        tables = self._get_dynamodb_tables_for_env(env)
        
        # Load config
        resp = tables['new_config_table'].get_item(Key={'config_id': config_id})
        config = resp.get('Item')
        if not config:
            raise ValueError(f"Config {config_id} not found in DynamoDB ({env})")
        config = json.loads(json.dumps(config, default=self._decimal_default))
        
        # Extract references
        schema_id = config.get('schema_id')
        prompt_template_id = config.get('prompt_template_id')
        
        # Validate prompt_template_id is required
        if not prompt_template_id:
            raise ValueError(f"Config {config_id} missing prompt_template_id")
        
        # Load schema only if schema_id is provided
        schema = None
        if schema_id:
            resp = tables['new_schema_table'].get_item(Key={'schema_id': schema_id})
            schema = resp.get('Item')
            if not schema:
                raise ValueError(f"Schema {schema_id} not found in DynamoDB ({env})")
            schema = json.loads(json.dumps(schema, default=self._decimal_default))
            logger.info(f"Loaded schema: {schema_id}")
        else:
            logger.info(f"No schema_id specified for config {config_id}, skipping schema load")
        
        # Load template
        resp = tables['new_template_table'].get_item(Key={'prompt_template_id': prompt_template_id})
        template_item = resp.get('Item')
        if not template_item:
            raise ValueError(f"Template {prompt_template_id} not found in DynamoDB ({env})")
        template_item = json.loads(json.dumps(template_item, default=self._decimal_default))
        template_content = template_item.get('template_content', template_item)
        
        return self._build_full_config_bundle(config_id, config, schema, template_content, prompt_template_id)
    
    def _build_full_config_bundle(self, config_id: str, config: Dict, schema: Optional[Dict], 
                                  template_content: Dict, prompt_template_id: str) -> Dict[str, Any]:
        """Build unified full config bundle."""
        # Build schema dict only if schema is provided
        schema_dict = None
        if schema:
            schema_dict = {
                "schema_id": schema.get('schema_id'),
                "schema_content": schema.get('schema_content', {}),
                "schema_metadata": schema.get('schema_metadata', {})
            }
        
        return {
            "config_id": config_id,
            "model_config": config.get('model_config', {}),
            "validation_config": config.get('validation_config', {}),
            "stopping_conditions": config.get('stopping_conditions', {}),
            "schema": schema_dict,
            "prompt_template": template_content,
            "prompt_template_id": prompt_template_id,
            "schema_id": config.get('schema_id'),
            "is_active": config.get('is_active', True),
            "created_at": config.get('created_at')
        }
    
    def load_full_agent_config(self, config_id: str) -> Dict[str, Any]:
        """
        Load full agent config with all references resolved.
        
        Main entry point that tries API first, then falls back to DynamoDB.
        When mode is 'dynamo', uses self.env directly.
        
        Args:
            config_id: Configuration ID (e.g., 'material-30')
            
        Returns:
            Complete config bundle with config, schema, and template resolved
            
        Raises:
            ValueError: If config cannot be loaded from either API or DynamoDB
        """
        # If mode is 'dynamo', use self.env directly
        if self.mode == 'dynamo':
            try:
                logger.info(f"Loading config {config_id} from DynamoDB ({self.env})...")
                result = self._load_full_config_via_dynamo(config_id, env=self.env)
                logger.info(f"Successfully loaded config {config_id} from DynamoDB")
                return result
            except Exception as dynamo_error:
                error_msg = f"Failed to load config {config_id} from DynamoDB ({self.env}): {dynamo_error}"
                logger.error(error_msg)
                raise ValueError(error_msg) from dynamo_error
        
        # For API mode: Try API first, then fallback to DynamoDB
        api_error = None
        
        # Try API first if available
        if self.api_client:
            try:
                logger.info(f"Attempting to load config {config_id} via API...")
                result = self._load_full_config_via_api(config_id)
                logger.info(f"Successfully loaded config {config_id} via API")
                return result
            except Exception as e:
                api_error = e
                logger.warning(f"API load failed for {config_id}: {e}")
                logger.info(f"Falling back to DynamoDB ({self.fallback_env})")
        
        # Fallback to DynamoDB
        try:
            logger.info(f"Loading config {config_id} from DynamoDB ({self.fallback_env})...")
            result = self._load_full_config_via_dynamo(config_id, env=self.fallback_env)
            logger.info(f"Successfully loaded config {config_id} from DynamoDB")
            return result
        except Exception as dynamo_error:
            error_msg = (
                f"Failed to load config {config_id} from both API and DynamoDB. "
                f"API error: {api_error if api_error else 'N/A'}. "
                f"DynamoDB error: {dynamo_error}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from dynamo_error

    def _wrap_single_item(self, item, property_type, root_type_id):
        """Helper to wrap a single item into the context structure if needed"""
        return {
            "property_type": property_type,
            "root_type_id": root_type_id,
            "context_mode": item.get('context_mode', 'REDUCED'),
            "context_content": item.get('context_content', {}), # It might be here
            "source": "dynamo_direct",
            "count": 1
        }

    # ============================================================================
    # Local File Loading Methods
    # ============================================================================

    def _load_classifier_config_local(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """Load classifier configuration from local JSON file"""
        config_file = self.local_config_dir / 'classifier_config.json'
        if not config_file.exists():
            raise FileNotFoundError(f"Local config file not found: {config_file}")

        with open(config_file, 'r') as f:
            all_configs = json.load(f)

        if property_type not in all_configs:
            raise KeyError(f"Property type '{property_type}' not found in local config")

        property_configs = all_configs[property_type]
        if str(root_type_id) not in property_configs:
            raise KeyError(f"Root type ID '{root_type_id}' not found for property '{property_type}'")

        return property_configs[str(root_type_id)]

    def _load_prompt_template_local(self, template_id: str) -> Dict[str, Any]:
        """Load prompt template from local JSON file"""
        # Try both direct path (if it looks like a path) and simplified ID
        possible_paths = [
            self.local_config_dir / 'prompts' / f'{template_id}.json',
            Path(template_id) if template_id.endswith('.json') else None
        ]

        for template_file in possible_paths:
            if template_file and template_file.exists():
                 with open(template_file, 'r') as f:
                    return json.load(f)

        raise FileNotFoundError(f"Local template file not found for ID: {template_id}")

    def _load_context_data_local(self, property_type: str, root_type_id: int) -> Dict[str, Any]:
        """Load context data from local CSV file"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for local context data loading")

        context_file = Path('data') / f'{property_type}_llm_context.csv'
        if not context_file.exists():
            raise FileNotFoundError(f"Local context file not found: {context_file}")

        df = pd.read_csv(context_file)
        filtered_df = df[df['root_type_id'] == root_type_id]
        if filtered_df.empty:
            filtered_df = df[df['root_type_id'] == 0]
        
        if filtered_df.empty:
             # Just return empty structure rather than crashing, or handle gracefully
             # For now, consistent with other methods:
            raise ValueError(f"No context data found for {property_type}")

        context_mode = filtered_df['context_mode'].iloc[0] if 'context_mode' in filtered_df.columns else 'full-context'
        
        items = []
        descriptions = {}
        
        name_col = f'{property_type}_name'
        id_col = f'{property_type}_id'
        
        # Special case for model which has different column names usually?
        if property_type == 'model':
             name_col = 'model_name'
             id_col = 'model_id'

        desc_col = 'llm_description'

        for _, row in filtered_df.iterrows():
            if id_col not in row or name_col not in row:
                continue 
            
            item_id = int(row[id_col])
            item_name = row[name_col]
            item_key = f"ID {item_id}: {item_name}"
            items.append(item_key)

            if desc_col in row and pd.notna(row[desc_col]):
                descriptions[item_key] = str(row[desc_col])

        return {
            "materials": items,
            "descriptions": descriptions,
            "mode": context_mode,
            "root_type_id": root_type_id,
            "count": len(items),
            "property_type": property_type,
             # Return as dict matching API format somewhat
             "context_content": {
                 str(k): {"name": v} for k, v in zip([x.split(':')[0].replace('ID ', '') for x in items], [x.split(': ')[1] for x in items])
             }
        }
    
    # ============================================================================
    # CSV Config Management Methods
    # ============================================================================
    
    def load_csv_configs(self, organisation_uuid: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load all CSV configs from DynamoDB, optionally filtered by organisation_uuid.
        
        Args:
            organisation_uuid: Optional organisation UUID to filter by. If None, loads all configs.
            
        Returns:
            List of CSV config dictionaries
        """
        if self.mode != 'dynamo':
            raise ValueError("load_csv_configs only supports 'dynamo' mode")
        
        try:
            if organisation_uuid:
                # Use GSI to query by organisation_uuid
                response = self.csv_config_table.query(
                    IndexName='organisation-uuid-index',
                    KeyConditionExpression=Key('organisation_uuid').eq(organisation_uuid)
                )
                items = response.get('Items', [])
            else:
                # Scan all items (load all rows regardless of organisation_uuid)
                items = []
                response = self.csv_config_table.scan()
                items.extend(response.get('Items', []))
                
                # Handle pagination
                while 'LastEvaluatedKey' in response:
                    response = self.csv_config_table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                    items.extend(response.get('Items', []))
            
            # Convert Decimal to native types and parse JSON fields
            configs = []
            for item in items:
                clean_item = json.loads(json.dumps(item, default=self._decimal_default))
                # Parse JSON fields if they're strings
                if isinstance(clean_item.get('csv_columns'), str):
                    clean_item['csv_columns'] = json.loads(clean_item['csv_columns'])
                if isinstance(clean_item.get('csv_column_metadata_mappings'), str):
                    clean_item['csv_column_metadata_mappings'] = json.loads(clean_item['csv_column_metadata_mappings'])
                configs.append(clean_item)
            
            logger.info(f"Loaded {len(configs)} CSV config(s) from DynamoDB" + 
                       (f" for organisation {organisation_uuid}" if organisation_uuid else ""))
            return configs
            
        except Exception as e:
            logger.error(f"Error loading CSV configs from DynamoDB: {e}")
            raise
    
    def find_matching_csv_config(self, csv_columns: List[str], organisation_uuid: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find a CSV config that matches the given CSV columns.
        
        A config matches if ALL columns referenced in csv_column_metadata_mappings exist in the CSV.
        
        Args:
            csv_columns: List of column names from the CSV
            organisation_uuid: Optional organisation UUID to filter by
            
        Returns:
            Matching config dictionary, or None if no match
        """
        if self.mode != 'dynamo':
            raise ValueError("find_matching_csv_config only supports 'dynamo' mode")
        
        csv_columns_set = set(csv_columns)
        configs = self.load_csv_configs(organisation_uuid=organisation_uuid)
        
        for config in configs:
            mappings = config.get('csv_column_metadata_mappings', {})
            if not mappings:
                continue
            
            # Get all columns referenced in the mappings
            referenced_columns = set()
            for key, columns in mappings.items():
                if isinstance(columns, list):
                    referenced_columns.update(columns)
            
            # Check if all referenced columns exist in CSV
            if referenced_columns and referenced_columns.issubset(csv_columns_set):
                logger.info(f"Found matching CSV config: {config.get('csv_config_identifier')}")
                return config
        
        logger.info("No matching CSV config found")
        return None
    
    def save_csv_config(
        self,
        csv_config_identifier: str,
        csv_columns: List[str],
        csv_column_metadata_mappings: Dict[str, List[str]],
        organisation_uuid: Optional[str] = None,
        created_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save a CSV config to DynamoDB.
        
        Args:
            csv_config_identifier: Unique identifier for the config (PK)
            csv_columns: List of column names in the CSV
            csv_column_metadata_mappings: Mapping of metadata categories to columns
            organisation_uuid: Optional organisation UUID
            created_timestamp: Optional timestamp (defaults to current time)
            
        Returns:
            Saved config dictionary
        """
        if self.mode != 'dynamo':
            raise ValueError("save_csv_config only supports 'dynamo' mode")
        
        from datetime import datetime
        
        if created_timestamp is None:
            created_timestamp = datetime.utcnow().isoformat() + 'Z'
        
        item = {
            'csv_config_identifier': csv_config_identifier,
            'csv_columns': csv_columns,
            'csv_column_metadata_mappings': csv_column_metadata_mappings,
            'created_timestamp': created_timestamp
        }
        
        if organisation_uuid:
            item['organisation_uuid'] = organisation_uuid
        
        self.csv_config_table.put_item(Item=item)
        logger.info(f"Saved CSV config: {csv_config_identifier}")
        return item