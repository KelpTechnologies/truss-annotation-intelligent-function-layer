#!/usr/bin/env python3
"""
Lean Classification Orchestrator - Cloud Export Package
=======================================================

Simplified orchestrator that runs a single agent based on API or DynamoDB config.
Designed for cloud deployment (API or DynamoDB modes).

Usage:
    python lean_orchestration.py --dataset FILENAME --config-id CONFIG_ID [--input-mode MODE] [--mode MODE] [--env ENV]

Examples:
    python lean_orchestration.py --dataset bag_test.csv --config-id material-30 --input-mode image-only --mode api
    python lean_orchestration.py --dataset bag_test.csv --config-id condition-30 --mode dynamo --env dev
    python lean_orchestration.py --dataset bag_test.csv --config-id classifier-hardware-bags --mode dynamo --env staging

ENVIRONMENT VARIABLES:
=====================
Required for API access:
- DSL_API_BASE_URL: API endpoint URL
- DSL_API_KEY: API authentication key

FOR LAMBDA DEPLOYMENT:
=====================
- Remove the python-dotenv dependency from requirements.txt
- Remove the load_dotenv() calls below
- Set environment variables directly in Lambda configuration
- Ensure Lambda has appropriate IAM permissions for VertexAI and external API access (and DynamoDB if using dynamo mode)
"""

import pandas as pd
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime

# ENVIRONMENT VARIABLE LOADING
# ============================
# For LOCAL DEVELOPMENT: Load from .env file using python-dotenv
# For LAMBDA DEPLOYMENT: Comment out the try/except block below and set env vars in Lambda config
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file (local development)")
except (ImportError, UnicodeDecodeError, FileNotFoundError) as e:
    if isinstance(e, ImportError):
        print("python-dotenv not available - using system environment variables (Lambda/production)")
    elif isinstance(e, UnicodeDecodeError):
        print(f"Warning: Could not load .env file due to encoding issue: {e}")
        print("Please ensure your .env file is saved as UTF-8 without BOM")
    else:
        print(f"Warning: .env file not found: {e}")
    print("Using system environment variables")

# Add parent directory to path for imports (for agent_architecture)
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_architecture import LLMAnnotationAgent
from agent_architecture.validation import AgentResult, AgentStatus
from agent_utils.dsl_api_client import DSLAPIClient
from agent_orchestration.csv_config_loader import ConfigLoader


def extract_text_metadata(row: pd.Series, metadata_columns: list = None) -> str:
    """
    Extract and format text metadata from a DataFrame row.

    Args:
        row: DataFrame row
        metadata_columns: List of column names to include (default: brand, title, description)

    Returns:
        Formatted text metadata string
    """
    if metadata_columns is None:
        metadata_columns = ['brand', 'title', 'description']

    metadata_parts = []
    for col in metadata_columns:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            # Capitalize column name for display
            col_display = col.replace('_', ' ').title()
            metadata_parts.append(f"{col_display}: {row[col]}")

    return "\n".join(metadata_parts) if metadata_parts else None


def load_test_data(csv_path: str, limit: int = None) -> pd.DataFrame:
    """Load test data from CSV file."""
    df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)

    print(f"Loaded {len(df)} test samples from {csv_path}")
    return df


def extract_property_from_config_id(config_id: str) -> str:
    """
    Extract property name from config_id.
    
    Examples:
        'material-30' -> 'material'
        'classifier-condition-bags' -> 'condition'
        'classifier-model-bags-gucci-full-taxo' -> 'model'
        'classifier-hardware-bags' -> 'hardware'
    """
    # Handle new classifier-X-bags format
    if config_id.startswith('classifier-'):
        parts = config_id.replace('classifier-', '').split('-')
        if parts:
            return parts[0]  # e.g., 'condition', 'model', 'material'
    
    # Handle old format (property-version)
    return config_id.rsplit('-', 1)[0]


def get_name_from_schema(schema: Optional[Dict], id_value: int) -> str:
    """
    Look up name from schema using ID.

    Args:
        schema: Schema dict with 'schema_content' containing ID->name mapping
        id_value: The ID to look up

    Returns:
        Name string if found, "Unknown" for ID 0, empty string for invalid IDs
    """
    # Special case: ID 0 always means "Unknown" (no match found)
    if id_value == 0:
        return 'Unknown'

    if not schema:
        return ''

    schema_content = schema.get('schema_content', {})
    id_str = str(id_value)

    if id_str in schema_content:
        item = schema_content[id_str]
        if isinstance(item, dict):
            return item.get('name', '')
        return str(item)

    # ID not in schema - return empty string (validation should have caught this)
    return ''


def extract_classification_result(result_dict: dict, schema: Optional[Dict] = None) -> dict:
    """
    Extract classification-style properties from a validated result dict.
    
    Args:
        result_dict: The parsed LLM response
        schema: Optional schema for ID->name mapping
    
    Returns dict with: 
        - primary, alternatives (legacy format: "ID X")
        - primary_id, primary_name, alternative_ids, alternative_names (new format)
        - confidence, reasoning
    """
    prediction_id = result_dict.get('prediction_id')
    scores = result_dict.get('scores', [])
    
    # Extract primary ID and name
    primary_id = prediction_id if prediction_id is not None else None
    primary_name = get_name_from_schema(schema, prediction_id) if prediction_id is not None else ''
    primary = f"ID {prediction_id}" if prediction_id is not None else None
    
    # Extract alternatives (skip first score if it matches prediction_id)
    alternatives = []
    alternative_ids = []
    alternative_names = []
    
    if len(scores) > 1:
        for score in scores[1:]:
            if isinstance(score, dict):
                alt_id = score.get('id')
                if alt_id is not None and alt_id != prediction_id:
                    alternatives.append(f"ID {alt_id}")
                    alternative_ids.append(alt_id)
                    alt_name = get_name_from_schema(schema, alt_id)
                    alternative_names.append(alt_name)
    
    # Extract confidence (first score's score value)
    confidence = 0.0
    if scores:
        first_score = scores[0]
        if isinstance(first_score, dict):
            confidence = first_score.get('score', 0.0)
    
    # Extract reasoning
    reasoning = result_dict.get('reasoning', '')
    
    return {
        'primary': primary,
        'primary_id': primary_id,
        'primary_name': primary_name,
        'alternatives': alternatives,
        'alternative_ids': alternative_ids,
        'alternative_names': alternative_names,
        'confidence': confidence,
        'reasoning': reasoning
    }


def classify_single_item(
    row: pd.Series,
    idx: int,
    total: int,
    config_id: str,
    agent: LLMAnnotationAgent,
    input_mode: str = "auto"
) -> Dict[str, Any]:
    """Single item classification."""
    import time

    property_name = extract_property_from_config_id(config_id)
    item_id = str(row.get('garment_id', row.get('item_id', row.get('id', idx))))
    image_url = row.get('image_link') or row.get('image_url')
    text_input = extract_text_metadata(row)
    brand = row.get('brand') if 'brand' in row else None

    mode_display = f"({input_mode})"
    if text_input and image_url:
        mode_display = "(multimodal)" if input_mode == "auto" else f"({input_mode})"
    elif text_input:
        mode_display = "(text-only)"
    elif image_url:
        mode_display = "(image-only)"

    print(f"\n[{idx}/{total}] Processing item {item_id} - {property_name} {mode_display}")

    try:
        start_time = time.time()
        
        # Build input data dict for execute()
        input_data = {
            'item_id': item_id,
            'image_url': image_url,
            'text_input': text_input,
            'input_mode': input_mode
        }
        
        # Execute agent (no context override needed - schema is in full_config)
        agent_result: AgentResult = agent.execute(input_data=input_data)
        end_time = time.time()

        # Handle AgentResult
        if agent_result.status == AgentStatus.SUCCESS:
            result = agent_result.result  # This is now a dict
            classification_data = extract_classification_result(result, schema=agent_result.schema)
            
            result_dict = {
                'item_id': item_id,
                'property': property_name,
                'config_id': config_id,
                'primary': classification_data['primary'],
                'primary_id': classification_data['primary_id'],
                'primary_name': classification_data['primary_name'],
                'alternatives': classification_data['alternatives'],
                'alternative_ids': classification_data['alternative_ids'],
                'alternative_names': classification_data['alternative_names'],
                'confidence': classification_data['confidence'],
                'reasoning': classification_data['reasoning'],
                'processing_time_seconds': round(end_time - start_time, 3),
                'success': True,
                'status': agent_result.status.value,
                'validation_passed': agent_result.validation_info.is_valid if agent_result.validation_info else True,
                'validation_category': agent_result.validation_info.category if agent_result.validation_info else 'success',
                'warnings': len(agent_result.validation_info.warnings) if agent_result.validation_info else 0,
                'image_url': image_url,
                'has_text_input': bool(text_input),
                'input_mode_used': input_mode,
                'attempt': agent_result.metadata.get('attempt', 1)
            }

            # Print results
            primary_display = f"{classification_data['primary_name']} (ID {classification_data['primary_id']})" if classification_data['primary_name'] else classification_data['primary']
            print(f"  Result: {primary_display} (confidence: {classification_data['confidence']:.3f})")

            if classification_data['alternatives']:
                alt_str = ', '.join(classification_data['alternatives'][:3])
                print(f"  Alternatives: {alt_str}")
            if classification_data['reasoning']:
                print(f"  Reasoning: {classification_data['reasoning']}")
            
            # Print validation info
            if agent_result.validation_info:
                if agent_result.validation_info.warnings:
                    print(f"  Validation warnings: {len(agent_result.validation_info.warnings)}")
                if not agent_result.validation_info.is_valid:
                    print(f"  Validation failed: {agent_result.validation_info.category}")
            
            print(f"  Processing time: {result_dict['processing_time_seconds']}s")
            print(f"  Attempts: {agent_result.metadata.get('attempt', 1)}")

            return result_dict

        else:
            # Handle failure cases
            error_msg = agent_result.error_report.message if agent_result.error_report else "Unknown error"
            
            # Extract classification data if result exists (even if validation failed)
            classification_data = None
            if agent_result.result:
                classification_data = extract_classification_result(agent_result.result, schema=agent_result.schema)
            
            result_dict = {
                'item_id': item_id,
                'property': property_name,
                'config_id': config_id,
                'primary': classification_data['primary'] if classification_data else 'unknown',
                'primary_id': classification_data['primary_id'] if classification_data else None,
                'primary_name': classification_data['primary_name'] if classification_data else '',
                'alternatives': classification_data['alternatives'] if classification_data else [],
                'alternative_ids': classification_data['alternative_ids'] if classification_data else [],
                'alternative_names': classification_data['alternative_names'] if classification_data else [],
                'confidence': classification_data['confidence'] if classification_data else 0.0,
                'reasoning': classification_data['reasoning'] if classification_data else '',
                'processing_time_seconds': round(end_time - start_time, 3),
                'success': False,
                'status': agent_result.status.value,
                'error': error_msg,
                'error_type': agent_result.error_report.error_type if agent_result.error_report else 'unknown',
                'recoverable': agent_result.error_report.recoverable if agent_result.error_report else False,
                'validation_passed': agent_result.validation_info.is_valid if agent_result.validation_info else False,
                'validation_category': agent_result.validation_info.category if agent_result.validation_info else 'unknown',
                'warnings': len(agent_result.validation_info.warnings) if agent_result.validation_info else 0,
                'image_url': image_url,
                'has_text_input': bool(text_input),
                'input_mode_used': input_mode,
                'attempt': agent_result.metadata.get('attempt', 1)
            }

            print(f"  ERROR: {error_msg} (Status: {agent_result.status.value})")
            if agent_result.validation_info and agent_result.validation_info.errors:
                print(f"  Validation errors: {len(agent_result.validation_info.errors)}")
                for err in agent_result.validation_info.errors[:3]:
                    print(f"    - {err.rule_id}: {err.message}")

            return result_dict

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        # import traceback
        # traceback.print_exc()
        return {
            'item_id': item_id,
            'property': property_name,
            'config_id': config_id,
            'primary': 'unknown',
            'primary_id': None,
            'primary_name': '',
            'alternatives': [],
            'alternative_ids': [],
            'alternative_names': [],
            'confidence': 0.0,
            'reasoning': '',
            'processing_time_seconds': 0.0,
            'success': False,
            'status': 'exception',
            'error': str(e),
            'error_type': 'exception',
            'recoverable': False,
            'validation_passed': False,
            'validation_category': 'exception',
            'warnings': 0,
            'image_url': image_url,
            'has_text_input': bool(text_input),
            'input_mode_used': input_mode,
            'attempt': 0
        }


def run_single_classification(
    df: pd.DataFrame,
    config_id: str,
    full_config: Dict[str, Any],
    input_mode: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Run classification for a single config on the dataset.

    Args:
        df: DataFrame with test data
        config_id: Configuration ID (e.g., 'material-30', 'classifier-hardware-bags')
        full_config: Full agent configuration bundle (from load_full_agent_config)
        input_mode: Input mode (auto, image-only, text-only, multimodal)

    Returns:
        List of classification results
    """
    results = []
    property_name = extract_property_from_config_id(config_id)
    model_config = full_config['model_config']

    print(f"\n{'='*70}")
    print(f"CLASSIFIER ORCHESTRATION: {property_name.upper()} for {len(df)} items (config_id={config_id})")
    print(f"Model: {model_config.get('model')}")
    print(f"Input mode: {input_mode}")
    print(f"{'='*70}")

    # Initialize agent with full config (single line!)
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    
    print(f"Loaded prompt template: {full_config.get('prompt_template_id')}")
    print(f"Loaded schema: {full_config.get('schema_id')} ({len(agent._valid_ids)} items)")

    total = len(df)

    # Sequential processing
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        result = classify_single_item(
            row, idx, total, config_id, agent, input_mode
        )
        results.append(result)

    return results


def run_bulk_classification_parallel(
    df: pd.DataFrame,
    config_id: str,
    config_loader: ConfigLoader,
    text_extraction_fn: Callable[[pd.Series, int], str],
    input_mode: str = 'text-only',
    max_workers: int = 200,
    batch_size: Optional[int] = None,
    property_name: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    image_url_fn: Optional[Callable[[pd.Series, int], Optional[str]]] = None
) -> pd.DataFrame:
    """
    Run classification on DataFrame rows in parallel for bulk processing.

    This is the single bulk/parallel entry point for generic property classifiers.
    All parallelism is internal (ThreadPoolExecutor + Semaphore); callers pass
    df, config_id, and extraction functions and receive a DataFrame. Do not
    implement a parallel loop in the caller when using this function.

    Supports text-only, image-only, or multimodal input when image_url_fn is provided.

    Args:
        df: DataFrame to process
        config_id: Configuration ID (e.g., 'classifier-material-bags', 'classifier-colour-bags')
        config_loader: ConfigLoader instance
        text_extraction_fn: Function that takes (row, idx) and returns text string
        input_mode: Input mode for classifiers ('text-only', 'image-only', 'multimodal')
        max_workers: Maximum concurrent API calls
        batch_size: Process in batches of this size (None = process all at once)
        property_name: Property name for output columns (e.g., 'material', 'condition', 'hardware')
                       If None, extracted from config_id
        progress_callback: Optional callback function(idx, total) for progress updates
        image_url_fn: Optional function (row, idx) -> image URL for multimodal/image-only

    Returns:
        DataFrame with classification results (TRUSS_{property}_* columns)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Semaphore
    
    # Extract property name if not provided
    if property_name is None:
        property_name = extract_property_from_config_id(config_id)
    
    print(f"\n{'='*70}")
    print(f"{property_name.upper()} CLASSIFICATION (PARALLEL)")
    print(f"{'='*70}")
    
    # Load classifier config
    try:
        full_config = config_loader.load_full_agent_config(config_id)
    except Exception as e:
        print(f"ERROR: Could not load {property_name} classifier config: {e}")
        # Return same shape as success so callers (e.g. material two-pass) do not KeyError
        err_msg = str(e)[:200] if str(e) else "Config load failed"
        return pd.DataFrame({
            f'TRUSS_{property_name}_id': [None] * len(df),
            f'TRUSS_{property_name}_name': [''] * len(df),
            f'TRUSS_{property_name}_confidence': [0.0] * len(df),
            f'TRUSS_{property_name}_reasoning': [err_msg] * len(df),
        })
    
    # Initialize agent once (reused for all rows)
    from agent_architecture import LLMAnnotationAgent
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    print(f"Initialized agent: {full_config['model_config'].get('model')}")
    
    # Semaphore to limit concurrent API calls
    semaphore = Semaphore(max_workers)
    total = len(df)
    results = [None] * total  # Pre-allocate to maintain order
    
    def process_with_semaphore(idx: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
        """Wrapper to limit concurrent executions"""
        with semaphore:
            try:
                text_input = text_extraction_fn(row, idx)
                image_url = image_url_fn(row, idx) if image_url_fn else None
                if isinstance(text_input, str):
                    text_input = text_input.strip() or None
                if not text_input and not image_url:
                    return (idx, {
                        f'TRUSS_{property_name}_id': None,
                        f'TRUSS_{property_name}_name': '',
                        f'TRUSS_{property_name}_confidence': 0.0,
                        f'TRUSS_{property_name}_reasoning': 'No text or image available'
                    })
                input_data = {
                    'item_id': str(idx),
                    'input_mode': input_mode
                }
                if text_input:
                    input_data['text_input'] = text_input
                if image_url:
                    input_data['image_url'] = image_url

                agent_result = agent.execute(input_data=input_data)
                
                if agent_result.status == AgentStatus.SUCCESS:
                    classification = extract_classification_result(agent_result.result, schema=agent_result.schema)
                    return (idx, {
                        f'TRUSS_{property_name}_id': classification['primary_id'],
                        f'TRUSS_{property_name}_name': classification['primary_name'],
                        f'TRUSS_{property_name}_confidence': classification['confidence'],
                        f'TRUSS_{property_name}_reasoning': classification['reasoning']
                    })
                else:
                    error_msg = agent_result.error_report.message if agent_result.error_report else 'Unknown error'
                    return (idx, {
                        f'TRUSS_{property_name}_id': None,
                        f'TRUSS_{property_name}_name': '',
                        f'TRUSS_{property_name}_confidence': 0.0,
                        f'TRUSS_{property_name}_reasoning': f'Error: {error_msg[:100]}'
                    })
            except Exception as e:
                return (idx, {
                    f'TRUSS_{property_name}_id': None,
                    f'TRUSS_{property_name}_name': '',
                    f'TRUSS_{property_name}_confidence': 0.0,
                    f'TRUSS_{property_name}_reasoning': f'Error: {str(e)[:100]}'
                })
    
    # Process in batches if batch_size is specified
    if batch_size:
        print(f"Processing {total} rows in batches of {batch_size}...")
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_df = df.iloc[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1} (rows {batch_start+1}-{batch_end})...")
            
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_df))) as executor:
                futures = {
                    executor.submit(process_with_semaphore, batch_start + idx, row): (batch_start + idx, row)
                    for idx, (_, row) in enumerate(batch_df.iterrows())
                }
                
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    if progress_callback:
                        progress_callback(idx + 1, total)
    else:
        # Process all rows in parallel
        print(f"Processing {total} rows in parallel (max {max_workers} concurrent)...")
        
        with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
            futures = {
                executor.submit(process_with_semaphore, idx, row): (idx, row)
                for idx, (_, row) in enumerate(df.iterrows())
            }
            
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                if progress_callback:
                    progress_callback(idx + 1, total)
    
    return pd.DataFrame(results)


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Save classification results to CSV."""
    if not results:
        print("No results to save")
        return

    df_results = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = [
        'item_id', 'property', 'config_id', 
        'primary_id', 'primary_name', 'alternative_ids', 'alternative_names',
        'primary', 'alternatives',  # Legacy columns
        'confidence', 'reasoning', 'processing_time_seconds', 'success',
        'status', 'validation_passed', 'validation_category', 'warnings',
        'error', 'error_type', 'recoverable', 'attempt',
        'image_url', 'has_text_input', 'input_mode_used', 'brand'
    ]

    # Only include columns that exist
    available_columns = [col for col in column_order if col in df_results.columns]
    df_results = df_results[available_columns]

    df_results.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nResults saved to: {output_file}")
    print(f"Total classifications: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('success', False))}")
    print(f"Failed: {sum(1 for r in results if not r.get('success', False))}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Lean Agent Classification')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset filename (CSV file)')
    parser.add_argument('--config-id', type=str, required=True,
                       help='Agent configuration ID (e.g., material-30, condition-30, classifier-hardware-bags)')
    parser.add_argument('--input-mode', type=str, default='auto',
                       choices=['auto', 'image-only', 'text-only', 'multimodal'],
                       help='Input mode (default: auto)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of items to process')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: auto-generated)')
    
    # Arguments for flexible loading
    parser.add_argument('--mode', type=str, default='api', choices=['api', 'local', 'dynamo'],
                       help='Configuration loading mode (default: api)')
    parser.add_argument('--env', type=str, default='dev', choices=['dev', 'staging', 'prod'],
                       help='Environment for DynamoDB tables (used in dynamo mode)')

    args = parser.parse_args()

    print(f"Lean Orchestration - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode.upper()} ({args.env})")
    print(f"Dataset: {args.dataset}")
    print(f"Config ID: {args.config_id}")
    print(f"Input Mode: {args.input_mode}")
    print(f"Limit: {args.limit or 'None'}")

    try:
        # Initialize API client if available
        api_client = None
        try:
            api_base_url = os.getenv('DSL_API_BASE_URL')
            api_key = os.getenv('DSL_API_KEY')
            if api_base_url and api_key:
                api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key)
        except Exception:
            pass  # Will use DynamoDB fallback
            
        # Initialize Config Loader with fallback
        config_loader = ConfigLoader(
            mode=args.mode,
            api_client=api_client,
            env=args.env,
            fallback_env='staging'
        )

        # Load full config (API with DynamoDB fallback)
        print(f"Loading configuration for {args.config_id}...")
        full_config = config_loader.load_full_agent_config(args.config_id)
        print(f"Configuration loaded: Model={full_config['model_config'].get('model')}")

        # Load test data
        df = load_test_data(args.dataset, args.limit)

        # Run classification
        results = run_single_classification(
            df=df,
            config_id=args.config_id,
            full_config=full_config,
            input_mode=args.input_mode
        )

        # Save results
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            property_name = extract_property_from_config_id(args.config_id)
            output_file = f"lean_results_{property_name}_{args.config_id}_{timestamp}.csv"

        save_results_to_csv(results, output_file)

        print(f"\n{'='*70}")
        print("CLASSIFIER ORCHESTRATION COMPLETE")
        print(f"{'='*70}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
