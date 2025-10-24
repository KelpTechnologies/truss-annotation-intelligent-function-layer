#!/usr/bin/env python3
"""
Lean Classification Orchestrator - Cloud Export Package
=======================================================

Simplified orchestrator that runs a single classifier based on API config.
Designed for cloud deployment (API-only mode).

Usage:
    python lean_orchestration.py --dataset FILENAME --property PROPERTY --root-type-id ID [--input-mode MODE]

Examples:
    python lean_orchestration.py --dataset bag_test.csv --property material --root-type-id 30 --input-mode image-only
    python lean_orchestration.py --dataset bag_test.csv --property condition --root-type-id 30 --input-mode multimodal

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
- Ensure Lambda has appropriate IAM permissions for VertexAI and external API access
"""

import pandas as pd
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
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

from llm_annotation_system import LLMAnnotationAgent
from output_parser import load_property_id_mapping_api, validate_and_map_prediction
from dsl_api_client import DSLAPIClient
from config_loader import ConfigLoader


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


def classify_single_item_for_property(
    row: pd.Series,
    idx: int,
    total: int,
    property_name: str,
    root_type_id: int,
    classifier: LLMAnnotationAgent,
    property_column: str,
    input_mode: str = "auto",
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Single item classification."""
    import time

    garment_id = str(row['garment_id'])
    image_url = row.get('image_link')
    text_metadata = extract_text_metadata(row)
    brand = row.get('brand') if 'brand' in row else None
    existing_value = row.get(property_column, '')

    mode_display = f"({input_mode})"
    if text_metadata and image_url:
        mode_display = "(multimodal)" if input_mode == "auto" else f"({input_mode})"
    elif text_metadata:
        mode_display = "(text-only)"
    elif image_url:
        mode_display = "(image-only)"

    # Add brand info for model classification
    brand_info = f" [Brand: {brand}]" if property_name == "model" and brand else ""
    print(f"\n[{idx}/{total}] Processing garment {garment_id} - {property_name} {mode_display}{brand_info}")

    try:
        start_time = time.time()
        result = classifier.classify(
            image_url=image_url,
            text_metadata=text_metadata,
            property_type=property_name,
            garment_id=garment_id,
            root_type_id=root_type_id,
            brand=brand,
            input_mode=input_mode,
            context_data=context_data
        )
        end_time = time.time()

        result_dict = {
            'garment_id': garment_id,
            'property': property_name,
            'root_type_id': root_type_id,
            'primary': result.primary,
            'alternatives': result.alternatives,
            'confidence': result.confidence,
            'reasoning': getattr(result, 'reasoning', ''),
            'processing_time_seconds': round(end_time - start_time, 3),
            'success': True,
            'existing_value': existing_value,
            'image_url': image_url,
            'has_text_metadata': bool(text_metadata),
            'input_mode_used': input_mode
        }

        # Print results
        print(f"  Result: {result.primary} (confidence: {result.confidence:.3f})")

        if result.alternatives and isinstance(result.alternatives, list):
            # Parse alternatives - they come as "ID X" strings
            alt_list = []
            for alt in result.alternatives[:3]:
                if isinstance(alt, str) and alt.startswith('ID '):
                    try:
                        alt_id = int(alt.split('ID ')[1])
                        alt_list.append(f"ID {alt_id}")
                    except (ValueError, IndexError):
                        alt_list.append(alt)
                else:
                    alt_list.append(str(alt))
            alt_str = ', '.join(alt_list)
            print(f"  Alternatives: {alt_str}")
        if hasattr(result, 'reasoning') and result.reasoning:
            print(f"  Reasoning: {result.reasoning}")
        print(f"  Processing time: {result_dict['processing_time_seconds']}s")

        return result_dict

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {
            'garment_id': garment_id,
            'property': property_name,
            'root_type_id': root_type_id,
            'primary': 'unknown',
            'alternatives': [],
            'confidence': 0.0,
            'reasoning': '',
            'processing_time_seconds': 0.0,
            'success': False,
            'error': str(e),
            'existing_value': existing_value,
            'image_url': image_url,
            'has_text_metadata': bool(text_metadata),
            'input_mode_used': input_mode
        }


def run_single_classification(
    df: pd.DataFrame,
    property_name: str,
    root_type_id: int,
    config: Dict,
    config_loader: ConfigLoader,
    input_mode: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Run classification for a single property on the dataset.

    Args:
        df: DataFrame with test data
        property_name: Property to classify
        root_type_id: Root type ID
        config: Property configuration
        input_mode: Input mode (auto, image-only, text-only, multimodal)

    Returns:
        List of classification results
    """
    results = []

    print(f"\n{'='*70}")
    print(f"LEAN CLASSIFICATION: {property_name.upper()} for {len(df)} items (root_type_id={root_type_id})")
    print(f"Model: {config['model']}, Context: {config['default_context_mode']}")
    print(f"Input mode: {input_mode}")
    print(f"{'='*70}")

    # Load prompt template from API - required for operation
    template_id = config.get('prompt_template')
    template = None

    if not template_id:
        raise ValueError("No prompt_template specified in configuration. Template is required for operation.")

    try:
        # Check if it's an API template ID (not a file path)
        if not template_id.startswith('llm_annotation_system/') and not template_id.endswith('.json'):
            template = config_loader.load_prompt_template(template_id)
            print(f"Loaded prompt template from API: {template_id}")
        else:
            raise ValueError(f"Template path detected ({template_id}), but API templates are required. File-based templates not supported in API-only mode.")
    except Exception as e:
        raise ValueError(f"Failed to load required prompt template {template_id}: {e}")

    # Load context data from API
    context_data = config_loader.load_context_data(property_name, root_type_id)
    print(f"Loaded context data for {property_name}/{root_type_id}")

    # Initialize classifier
    classifier = LLMAnnotationAgent(
        model_name=config['model'],
        project_id=config.get('project_id', 'truss-data-science'),
        location=config.get('location', 'us-central1'),
        prompt_template_path=None,  # Use built-in template logic
        context_mode=config['default_context_mode'],
        log_IO=False,  # Disable file logging, print to terminal instead
        config={
            'temperature': config.get('temperature', 0.1),
            'max_output_tokens': config.get('max_output_tokens', 1024)
        }
    )

    # Set template if loaded from API
    if template:
        classifier.prompt_template = template

    property_column = config.get('property_column', property_name)
    total = len(df)

    # Sequential processing only (no parallelism)
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        result = classify_single_item_for_property(
            row, idx, total, property_name, root_type_id, classifier, property_column, input_mode, context_data
        )
        results.append(result)

    return results


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Save classification results to CSV."""
    if not results:
        print("No results to save")
        return

    df_results = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = [
        'garment_id', 'property', 'root_type_id', 'primary', 'alternatives',
        'confidence', 'reasoning', 'processing_time_seconds', 'success',
        'existing_value', 'image_url', 'has_text_metadata', 'input_mode_used'
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
    parser = argparse.ArgumentParser(description='Lean Single Property Classification')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset filename (CSV file)')
    parser.add_argument('--property', type=str, required=True,
                       help='Property to classify (e.g., material, condition, color)')
    parser.add_argument('--root-type-id', type=int, required=True,
                       help='Root type ID for the classifier configuration')
    parser.add_argument('--input-mode', type=str, default='auto',
                       choices=['auto', 'image-only', 'text-only', 'multimodal'],
                       help='Input mode (default: auto)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of items to process')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: auto-generated)')
    # Note: Configuration is loaded from DSL API, no local config file needed

    args = parser.parse_args()

    print(f"Lean Orchestration - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}")
    print(f"Property: {args.property}")
    print(f"Root Type ID: {args.root_type_id}")
    print(f"Input Mode: {args.input_mode}")
    print(f"Limit: {args.limit or 'None'}")

    try:
        # Initialize API client and config loader
        api_base_url = os.getenv('DSL_API_BASE_URL')
        api_key = os.getenv('DSL_API_KEY')

        if not api_base_url or not api_key:
            raise ValueError("DSL_API_BASE_URL and DSL_API_KEY environment variables are required")

        api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key)
        config_loader = ConfigLoader(mode='api', api_client=api_client)

        # Load configuration from API
        api_response = config_loader.load_classifier_config(args.property, args.root_type_id)
        specific_config = api_response.get('data', api_response)  # Extract data from API response
        print(f"Configuration loaded from API: {specific_config}")

        # Load test data
        df = load_test_data(args.dataset, args.limit)

        # Filter by root type if needed
        if 'root_type_id' in df.columns:
            df_filtered = df[df['root_type_id'] == args.root_type_id]
            if len(df_filtered) == 0:
                print(f"Warning: No items found with root_type_id={args.root_type_id}")
                if len(df) > 0:
                    print(f"Available root_type_ids: {sorted(df['root_type_id'].unique())}")
                return
            df = df_filtered
            print(f"Filtered to {len(df)} items with root_type_id={args.root_type_id}")

        # Run classification
        results = run_single_classification(
            df=df,
            property_name=args.property,
            root_type_id=args.root_type_id,
            config=specific_config,
            config_loader=config_loader,
            input_mode=args.input_mode
        )

        # Save results
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"lean_results_{args.property}_{args.root_type_id}_{timestamp}.csv"

        save_results_to_csv(results, output_file)

        print(f"\n{'='*70}")
        print("LEAN CLASSIFICATION COMPLETE")
        print(f"{'='*70}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
