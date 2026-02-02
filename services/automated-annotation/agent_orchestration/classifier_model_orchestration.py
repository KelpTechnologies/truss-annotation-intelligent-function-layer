#!/usr/bin/env python3
"""
Model Classifier Orchestrator
=============================

Orchestrates model classification with brand-specific agent configs.
Reads brand from each row and dynamically selects the appropriate config:
    classifier-model-bags-{safe_brand}-full-taxo

Usage:
    python classifier_model_orchestration.py --dataset FILENAME [--input-mode MODE] [--mode MODE] [--env ENV]

Examples:
    python classifier_model_orchestration.py --dataset bags_test.csv
    python classifier_model_orchestration.py --dataset bags_test.csv --input-mode text-only
    python classifier_model_orchestration.py --dataset bags_test.csv --mode dynamo --env staging
    python classifier_model_orchestration.py --dataset bags_test.csv --limit 10 --brand-column Brand

ENVIRONMENT VARIABLES:
=====================
Required for API access:
- DSL_API_BASE_URL: API endpoint URL
- DSL_API_KEY: API authentication key
"""

import argparse
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import Optional, Dict, Any, List, Set, Callable, Tuple

import pandas as pd

# ENVIRONMENT VARIABLE LOADING
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file (local development)")
except (ImportError, UnicodeDecodeError, FileNotFoundError) as e:
    if isinstance(e, ImportError):
        print("python-dotenv not available - using system environment variables (Lambda/production)")
    elif isinstance(e, UnicodeDecodeError):
        print(f"Warning: Could not load .env file due to encoding issue: {e}")
    else:
        print(f"Warning: .env file not found: {e}")
    print("Using system environment variables")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_architecture import LLMAnnotationAgent
from agent_architecture.validation import AgentResult, AgentStatus
from agent_utils.dsl_api_client import DSLAPIClient
from agent_orchestration.csv_config_loader import ConfigLoader
from agent_orchestration.classifier_orchestration import (
    load_test_data,
    save_results_to_csv,
    extract_property_from_config_id,
    extract_text_metadata,
    extract_classification_result,
    get_name_from_schema
)


# =============================================================================
# Brand Name Normalization
# =============================================================================

def safe_mapping(brand: str) -> str:
    """
    Normalize brand name to config-compatible format.
    
    Rules:
    - Lowercase
    - Only UTF-8 lowercase alphanumeric Latin characters and dashes
    - Map accented chars to non-accented ASCII (e -> e)
    - Replace spaces with "-"
    - Remove apostrophes
    
    Examples:
    - "Celine" -> "celine"
    - "Louis Vuitton" -> "louis-vuitton"
    - "Saint Laurent" -> "saint-laurent"
    """
    if pd.isna(brand) or not brand:
        raise ValueError("Brand name cannot be empty")
    
    # Convert to string and strip whitespace
    brand_str = str(brand).strip()
    
    # Normalize Unicode characters (decompose accented chars)
    normalized = unicodedata.normalize('NFKD', brand_str)
    
    # Remove diacritics (accents) and convert to ASCII
    ascii_brand = normalized.encode('ascii', 'ignore').decode('ascii')
    
    # Convert to lowercase
    ascii_brand = ascii_brand.lower()
    
    # Replace spaces with dashes
    ascii_brand = ascii_brand.replace(' ', '-')
    
    # Remove apostrophes (all types)
    apostrophes = ["'", "'", "'", '"', '"', '"']
    for apostrophe in apostrophes:
        ascii_brand = ascii_brand.replace(apostrophe, '')
    
    # Remove any characters that aren't lowercase alphanumeric or dashes
    ascii_brand = re.sub(r'[^a-z0-9-]', '', ascii_brand)
    
    # Remove multiple consecutive dashes
    ascii_brand = re.sub(r'-+', '-', ascii_brand)
    
    # Remove leading/trailing dashes
    ascii_brand = ascii_brand.strip('-')
    
    if not ascii_brand:
        raise ValueError(f"Brand name '{brand}' resulted in empty string after normalization")
    
    return ascii_brand


def get_model_config_id(brand: str) -> str:
    """
    Generate the config_id for a given brand.
    
    Format: classifier-model-bags-{safe_brand}-full-taxo
    """
    safe_brand = safe_mapping(brand)
    return f"classifier-model-bags-{safe_brand}-full-taxo"


# =============================================================================
# Validation Functions
# =============================================================================

def get_unique_brands(df: pd.DataFrame, brand_column: str) -> Set[str]:
    """Extract unique non-null brands from DataFrame."""
    if brand_column not in df.columns:
        raise ValueError(f"Brand column '{brand_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    brands = df[brand_column].dropna().unique()
    return set(str(b).strip() for b in brands if str(b).strip())


def validate_configs_exist(
    brands: Set[str],
    config_loader: ConfigLoader
) -> Dict[str, Dict[str, Any]]:
    """
    Validate that configs exist for all brands and pre-load them.
    
    Returns:
        Dict mapping brand -> full_config
    
    Raises:
        ValueError if any config is missing
    """
    print(f"\n[VALIDATION] Checking configs for {len(brands)} unique brand(s)...")
    
    configs = {}
    missing = []
    
    for brand in sorted(brands):
        config_id = get_model_config_id(brand)
        safe_brand = safe_mapping(brand)
        
        try:
            full_config = config_loader.load_full_agent_config(config_id)
            configs[brand] = full_config
            print(f"  [OK] {brand} -> {config_id}")
        except Exception as e:
            missing.append((brand, config_id, str(e)))
            print(f"  [MISSING] {brand} -> {config_id}")
    
    if missing:
        print(f"\n[ERROR] Missing configs for {len(missing)} brand(s):")
        for brand, config_id, error in missing:
            print(f"  - {brand}: {config_id}")
            print(f"    Error: {error}")
        raise ValueError(f"Missing configs for brands: {[b for b, _, _ in missing]}")
    
    print(f"\n[VALIDATION PASSED] All {len(brands)} brand configs found and loaded.")
    return configs


# =============================================================================
# Classification Functions
# =============================================================================

def classify_single_item_with_brand(
    row: pd.Series,
    idx: int,
    total: int,
    brand_column: str,
    brand_configs: Dict[str, Dict[str, Any]],
    brand_agents: Dict[str, LLMAnnotationAgent],
    input_mode: str = "auto"
) -> Dict[str, Any]:
    """Classify a single item using the brand-specific agent."""
    
    brand = str(row.get(brand_column, '')).strip()
    item_id = str(row.get('garment_id', row.get('item_id', row.get('id', idx))))
    image_url = row.get('image_link') or row.get('image_url')
    text_input = extract_text_metadata(row)
    
    # Get brand-specific agent
    if brand not in brand_agents:
        return {
            'item_id': item_id,
            'brand': brand,
            'property': 'model',
            'config_id': None,
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
            'status': 'invalid_input',
            'error': f"No config for brand: {brand}",
            'image_url': image_url,
            'has_text_input': bool(text_input),
            'input_mode_used': input_mode
        }
    
    agent = brand_agents[brand]
    config_id = get_model_config_id(brand)
    
    mode_display = f"({input_mode})"
    if text_input and image_url:
        mode_display = "(multimodal)" if input_mode == "auto" else f"({input_mode})"
    elif text_input:
        mode_display = "(text-only)"
    elif image_url:
        mode_display = "(image-only)"
    
    print(f"\n[{idx}/{total}] Processing item {item_id} - {brand} {mode_display}")
    
    try:
        start_time = time.time()
        
        # Build input data dict for execute()
        input_data = {
            'item_id': item_id,
            'image_url': image_url,
            'text_input': text_input,
            'input_mode': input_mode
        }
        
        # Execute agent
        agent_result: AgentResult = agent.execute(input_data=input_data)
        end_time = time.time()
        
        # Handle AgentResult
        if agent_result.status == AgentStatus.SUCCESS:
            result = agent_result.result
            classification_data = extract_classification_result(result, schema=agent_result.schema)
            
            result_dict = {
                'item_id': item_id,
                'brand': brand,
                'property': 'model',
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
            
            primary_display = f"{classification_data['primary_name']} (ID {classification_data['primary_id']})" if classification_data['primary_name'] else classification_data['primary']
            print(f"  Result: {primary_display} (confidence: {classification_data['confidence']:.3f})")
            if classification_data['reasoning']:
                print(f"  Reasoning: {classification_data['reasoning']}")
            print(f"  Processing time: {result_dict['processing_time_seconds']}s")
            
            return result_dict
        
        else:
            # Handle failure cases: log error but do not return classification data as if successful
            error_msg = agent_result.error_report.message if agent_result.error_report else "Unknown error"
            print(f"  ERROR: {error_msg}")

            # Return empty classification fields so callers do not treat this as a valid result
            result_dict = {
                'item_id': item_id,
                'brand': brand,
                'property': 'model',
                'config_id': config_id,
                'primary': 'unknown',
                'primary_id': None,
                'primary_name': '',
                'alternatives': [],
                'alternative_ids': [],
                'alternative_names': [],
                'confidence': 0.0,
                'reasoning': '',
                'processing_time_seconds': round(end_time - start_time, 3),
                'success': False,
                'status': agent_result.status.value,
                'error': error_msg,
                'error_type': agent_result.error_report.error_type if agent_result.error_report else 'unknown',
                'validation_passed': agent_result.validation_info.is_valid if agent_result.validation_info else False,
                'validation_category': agent_result.validation_info.category if agent_result.validation_info else 'unknown',
                'image_url': image_url,
                'has_text_input': bool(text_input),
                'input_mode_used': input_mode,
                'attempt': agent_result.metadata.get('attempt', 1)
            }
            return result_dict
    
    except Exception as e:
        print(f"  EXCEPTION: {str(e)}")
        return {
            'item_id': item_id,
            'brand': brand,
            'property': 'model',
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
            'image_url': image_url,
            'has_text_input': bool(text_input),
            'input_mode_used': input_mode
        }


def run_brand_based_classification(
    df: pd.DataFrame,
    brand_column: str,
    brand_configs: Dict[str, Dict[str, Any]],
    input_mode: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Run classification for all items, using brand-specific configs.
    
    Args:
        df: DataFrame with test data
        brand_column: Column name containing brand
        brand_configs: Dict mapping brand -> full_config
        input_mode: Input mode (auto, image-only, text-only, multimodal)
    
    Returns:
        List of classification results
    """
    print(f"\n{'='*70}")
    print(f"MODEL CLASSIFIER: Processing {len(df)} items with {len(brand_configs)} brand config(s)")
    print(f"Input mode: {input_mode}")
    print(f"{'='*70}")
    
    # Initialize agents for each brand (reuse across items)
    brand_agents: Dict[str, LLMAnnotationAgent] = {}
    for brand, full_config in brand_configs.items():
        brand_agents[brand] = LLMAnnotationAgent(full_config=full_config, log_IO=False)
        print(f"Initialized agent for {brand}: {full_config['model_config'].get('model')}")
    
    results = []
    total = len(df)
    
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        result = classify_single_item_with_brand(
            row=row,
            idx=idx,
            total=total,
            brand_column=brand_column,
            brand_configs=brand_configs,
            brand_agents=brand_agents,
            input_mode=input_mode
        )
        results.append(result)
    
    return results


def run_bulk_model_classification(
    df: pd.DataFrame,
    text_extraction_fn: Callable[[pd.Series, int], str],
    config_loader: ConfigLoader,
    max_workers: int = 200,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> pd.DataFrame:
    """
    Run model classification on all rows in parallel (bulk mode).

    Parallelism is internal. Expects df to have TRUSS_brand_name (from brand
    classification). Uses brand-specific configs; one agent per brand, reused.

    Args:
        df: DataFrame with TRUSS_brand_name column
        text_extraction_fn: (row, idx) -> text string for model classification
        config_loader: ConfigLoader instance
        max_workers: Max concurrent agent calls
        batch_size: Process in batches (None = all at once)
        progress_callback: Optional (idx, total)

    Returns:
        DataFrame with TRUSS_model_id, TRUSS_model_name, TRUSS_model_confidence, TRUSS_model_reasoning
    """
    if 'TRUSS_brand_name' not in df.columns:
        return pd.DataFrame([{
            'TRUSS_model_id': None,
            'TRUSS_model_name': '',
            'TRUSS_model_confidence': 0.0,
            'TRUSS_model_reasoning': 'TRUSS_brand not available - brand classification required first'
        }] * len(df))

    brands = set()
    for _, row in df.iterrows():
        b = row.get('TRUSS_brand_name')
        if b and pd.notna(b) and str(b).strip():
            brands.add(str(b).strip())
    if not brands:
        return pd.DataFrame([{
            'TRUSS_model_id': None,
            'TRUSS_model_name': '',
            'TRUSS_model_confidence': 0.0,
            'TRUSS_model_reasoning': 'No TRUSS_brand available'
        }] * len(df))

    brand_agents = {}
    for brand in brands:
        config_id = get_model_config_id(brand)
        try:
            full_config = config_loader.load_full_agent_config(config_id)
            brand_agents[brand] = LLMAnnotationAgent(full_config=full_config, log_IO=False)
        except Exception:
            pass
    if not brand_agents:
        return pd.DataFrame([{
            'TRUSS_model_id': None,
            'TRUSS_model_name': '',
            'TRUSS_model_confidence': 0.0,
            'TRUSS_model_reasoning': 'No config available for brand'
        }] * len(df))

    total = len(df)
    results = [None] * total
    semaphore = Semaphore(max_workers)

    def process_row(idx: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
        with semaphore:
            try:
                truss_brand = row.get('TRUSS_brand_name')
                if not truss_brand or pd.isna(truss_brand) or not str(truss_brand).strip():
                    return (idx, {
                        'TRUSS_model_id': None,
                        'TRUSS_model_name': '',
                        'TRUSS_model_confidence': 0.0,
                        'TRUSS_model_reasoning': 'No TRUSS_brand available - skipping model classification'
                    })
                truss_brand = str(truss_brand).strip()
                if truss_brand not in brand_agents:
                    return (idx, {
                        'TRUSS_model_id': None,
                        'TRUSS_model_name': '',
                        'TRUSS_model_confidence': 0.0,
                        'TRUSS_model_reasoning': f'No config for TRUSS_brand: {truss_brand}'
                    })
                text_input = text_extraction_fn(row, idx + 1)
                if not text_input or not str(text_input).strip():
                    return (idx, {
                        'TRUSS_model_id': None,
                        'TRUSS_model_name': '',
                        'TRUSS_model_confidence': 0.0,
                        'TRUSS_model_reasoning': 'No text data available - text-only classifier skipped'
                    })
                agent = brand_agents[truss_brand]
                input_data = {'item_id': str(idx + 1), 'text_input': text_input, 'input_mode': 'text-only'}
                agent_result = agent.execute(input_data=input_data)
                if agent_result.status == AgentStatus.SUCCESS:
                    classification = extract_classification_result(agent_result.result, schema=agent_result.schema)
                    return (idx, {
                        'TRUSS_model_id': classification['primary_id'],
                        'TRUSS_model_name': classification['primary_name'],
                        'TRUSS_model_confidence': classification['confidence'],
                        'TRUSS_model_reasoning': classification['reasoning']
                    })
                return (idx, {
                    'TRUSS_model_id': None,
                    'TRUSS_model_name': '',
                    'TRUSS_model_confidence': 0.0,
                    'TRUSS_model_reasoning': 'Model classification failed'
                })
            except Exception as e:
                return (idx, {
                    'TRUSS_model_id': None,
                    'TRUSS_model_name': '',
                    'TRUSS_model_confidence': 0.0,
                    'TRUSS_model_reasoning': f'Error: {str(e)[:100]}'
                })

    if batch_size:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_df = df.iloc[batch_start:batch_end]
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_df))) as executor:
                futures = {executor.submit(process_row, batch_start + i, row): batch_start + i
                          for i, (_, row) in enumerate(batch_df.iterrows())}
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    if progress_callback:
                        progress_callback(idx + 1, total)
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
            futures = {executor.submit(process_row, idx, row): idx
                      for idx, (_, row) in enumerate(df.iterrows())}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                if progress_callback:
                    progress_callback(idx + 1, total)
    return pd.DataFrame(results)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution for model classification."""
    parser = argparse.ArgumentParser(
        description='Model Classifier with Brand-specific Configs (reads brand from CSV)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python classifier_model_orchestration.py --dataset bags_test.csv
    python classifier_model_orchestration.py --dataset bags_test.csv --input-mode text-only
    python classifier_model_orchestration.py --dataset bags_test.csv --mode dynamo --env staging
    python classifier_model_orchestration.py --dataset bags_test.csv --limit 10 --brand-column Brand
    python classifier_model_orchestration.py --dataset bags_test.csv --manual-brand Gucci
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset filename (CSV file)')
    
    # Brand options (mutually informative, not exclusive - manual-brand overrides brand-column)
    parser.add_argument('--brand-column', type=str, default='brand',
                       help='Column name containing brand (default: brand)')
    parser.add_argument('--manual-brand', type=str, default=None,
                       help='Override: use this brand for ALL rows (ignores brand-column)')
    
    # Optional arguments
    parser.add_argument('--input-mode', type=str, default='auto',
                       choices=['auto', 'image-only', 'text-only', 'multimodal'],
                       help='Input mode (default: auto)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of items to process')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: auto-generated)')
    
    # Config loading arguments
    parser.add_argument('--mode', type=str, default='dynamo', choices=['api', 'local', 'dynamo'],
                       help='Configuration loading mode (default: dynamo)')
    parser.add_argument('--env', type=str, default='staging', choices=['dev', 'staging', 'prod'],
                       help='Environment for DynamoDB tables (default: staging)')

    args = parser.parse_args()

    # Determine brand mode
    use_manual_brand = args.manual_brand is not None
    
    print("=" * 70)
    if use_manual_brand:
        print(f"MODEL CLASSIFIER ORCHESTRATION (Manual Brand: {args.manual_brand})")
    else:
        print("MODEL CLASSIFIER ORCHESTRATION (Brand from CSV)")
    print("=" * 70)
    print(f"Dataset:      {args.dataset}")
    if use_manual_brand:
        print(f"Manual Brand: {args.manual_brand} (overrides CSV column)")
    else:
        print(f"Brand Column: {args.brand_column}")
    print(f"Input Mode:   {args.input_mode}")
    print(f"Mode:         {args.mode.upper()} ({args.env})")
    print(f"Limit:        {args.limit or 'None'}")
    print("=" * 70)

    try:
        # Load test data first to get brands
        print(f"\n[STEP 1] Loading dataset...")
        df = load_test_data(args.dataset, args.limit)
        
        # Get unique brands (either from manual override or CSV column)
        if use_manual_brand:
            print(f"\n[STEP 2] Using manual brand override: {args.manual_brand}")
            unique_brands = {args.manual_brand}
        else:
            print(f"\n[STEP 2] Extracting unique brands from '{args.brand_column}' column...")
            unique_brands = get_unique_brands(df, args.brand_column)
        print(f"Found {len(unique_brands)} unique brand(s): {sorted(unique_brands)}")
        
        # Initialize API client if available
        api_client = None
        try:
            api_base_url = os.getenv('DSL_API_BASE_URL')
            api_key = os.getenv('DSL_API_KEY')
            if api_base_url and api_key:
                api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key)
        except Exception:
            pass
        
        # Initialize Config Loader
        config_loader = ConfigLoader(
            mode=args.mode,
            api_client=api_client,
            env=args.env,
            fallback_env='staging'
        )
        
        # Validate all brand configs exist (this will raise if any missing)
        print(f"\n[STEP 3] Validating configs exist for all brands...")
        brand_configs = validate_configs_exist(unique_brands, config_loader)
        
        # If using manual brand, set all rows to that brand
        brand_column_to_use = args.brand_column
        if use_manual_brand:
            # Create a temporary column with the manual brand for all rows
            brand_column_to_use = '_manual_brand_override'
            df[brand_column_to_use] = args.manual_brand
            print(f"\n[STEP 3b] Applied manual brand '{args.manual_brand}' to all {len(df)} rows")
        
        # Run classification
        print(f"\n[STEP 4] Running classification...")
        results = run_brand_based_classification(
            df=df,
            brand_column=brand_column_to_use,
            brand_configs=brand_configs,
            input_mode=args.input_mode
        )
        
        # Save results
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"model_results_multi_brand_{timestamp}.csv"
        
        save_results_to_csv(results, output_file)
        
        # Summary
        print(f"\n{'='*70}")
        print("MODEL CLASSIFIER ORCHESTRATION COMPLETE")
        print(f"{'='*70}")
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"Total:     {len(results)}")
        print(f"Success:   {success_count}")
        print(f"Failed:    {len(results) - success_count}")
        print(f"Output:    {output_file}")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
