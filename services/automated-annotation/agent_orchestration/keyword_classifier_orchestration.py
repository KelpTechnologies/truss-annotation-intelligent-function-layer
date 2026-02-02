#!/usr/bin/env python3
"""
Keyword Classifier Orchestration
================================

Orchestrator for keyword extraction that:
- Extracts distinctive keywords NOT covered by other TRUSS properties
- Takes two JSON inputs: general_input_text and text_to_avoid
- Produces JSON with up to 3 keywords prioritized by importance
- No schema validation (keywords are free-form)

Usage:
    python keyword_classifier_orchestration.py --general-input-text JSON --text-to-avoid JSON --config-id classifier-keywords-bags [--mode MODE] [--env ENV]

Examples:
    python keyword_classifier_orchestration.py \
      --general-input-text '{"title": "Chanel Classic Flap", "description": "Quilted leather bag"}' \
      --text-to-avoid '{"material": "Leather", "condition": "Excellent"}' \
      --mode dynamo --env staging
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime

# ENVIRONMENT VARIABLE LOADING
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file (local development)")
except (ImportError, UnicodeDecodeError, FileNotFoundError) as e:
    if isinstance(e, ImportError):
        print("python-dotenv not available - using system environment variables (Lambda/production)")
    else:
        print(f"Warning: Could not load .env file: {e}")
    print("Using system environment variables")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from agent_architecture import LLMAnnotationAgent
from agent_architecture.validation import AgentResult, AgentStatus
from agent_utils.dsl_api_client import DSLAPIClient
from agent_orchestration.csv_config_loader import ConfigLoader


def format_input_text(general_input_text: Dict[str, Any]) -> str:
    """
    Format general input text JSON for the prompt.
    
    Args:
        general_input_text: Dict with text fields to process
        
    Returns:
        Formatted string for processing
    """
    if not general_input_text:
        return "No input text provided."
    
    parts = ["TEXT TO PROCESS:"]
    for key, value in general_input_text.items():
        if value and str(value).strip():
            key_display = key.replace('_', ' ').title()
            parts.append(f"  {key_display}: {value}")
    
    return "\n".join(parts)


def format_text_to_avoid(text_to_avoid: Dict[str, Any]) -> str:
    """
    Format text_to_avoid JSON for the prompt as EXISTING CLASSIFICATIONS.
    
    Args:
        text_to_avoid: Dict with classifications/properties to avoid
        
    Returns:
        Formatted string describing existing classifications
    """
    if not text_to_avoid:
        return "No existing classifications specified."
    
    parts = ["EXISTING CLASSIFICATIONS:"]
    for key, value in text_to_avoid.items():
        if value and str(value).strip():
            key_display = key.replace('_', ' ').title()
            parts.append(f"  {key_display}: {value}")
    
    return "\n".join(parts)


def parse_json_input(json_str: str) -> Dict[str, Any]:
    """
    Parse JSON input string.
    
    Args:
        json_str: JSON string to parse (may be a file path or JSON string)
        
    Returns:
        Parsed dict
    """
    # Check if it's a file path
    if os.path.exists(json_str):
        try:
            with open(json_str, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error reading JSON file '{json_str}': {e}")
    
    # Try to parse as JSON string
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}\n\nReceived: {json_str[:200]}...")


def extract_keyword_result(result_dict: dict) -> dict:
    """
    Extract keyword-style properties from a validated result dict.
    
    Args:
        result_dict: The parsed LLM response with keywords structure
    
    Returns dict with:
        - keywords: List of keyword dicts with keyword and confidence
        - reasoning: Explanation string
    """
    keywords_obj = result_dict.get('keywords', {})
    reasoning = result_dict.get('reasoning', '')
    
    # Convert keywords object to list
    keywords_list = []
    for key in sorted(keywords_obj.keys()):  # keyword_1, keyword_2, keyword_3
        keyword_data = keywords_obj[key]
        if isinstance(keyword_data, dict):
            keywords_list.append({
                'keyword': keyword_data.get('keyword', ''),
                'confidence': keyword_data.get('confidence', 0.0)
            })
    
    return {
        'keywords': keywords_list,
        'keyword_count': len(keywords_list),
        'reasoning': reasoning
    }


def classify_keywords(
    general_input_text: Dict[str, Any],
    text_to_avoid: Dict[str, Any],
    agent: LLMAnnotationAgent,
    item_id: str = "1"
) -> Dict[str, Any]:
    """Single keyword classification."""
    import time
    
    config_id = 'classifier-keywords-bags'

    # Format the inputs for the prompt
    input_text_formatted = format_input_text(general_input_text)
    existing_classifications_formatted = format_text_to_avoid(text_to_avoid)
    
    # Combine with clear separation
    full_text_input = f"{input_text_formatted}\n\n{existing_classifications_formatted}"

    print(f"\nProcessing keywords for item {item_id} (text-only)")

    try:
        start_time = time.time()
        
        # Build input data dict for execute()
        input_data = {
            'item_id': item_id,
            'text_input': full_text_input,
            'input_mode': 'text-only'  # Keywords classifier is text-only
        }
        
        # Execute agent
        agent_result: AgentResult = agent.execute(input_data=input_data)
        end_time = time.time()

        # Handle AgentResult
        if agent_result.status == AgentStatus.SUCCESS:
            result = agent_result.result
            keyword_data = extract_keyword_result(result)
            
            result_dict = {
                'item_id': item_id,
                'property': 'keywords',
                'config_id': config_id,
                'keywords': keyword_data['keywords'],
                'keyword_count': keyword_data['keyword_count'],
                'reasoning': keyword_data['reasoning'],
                'processing_time_seconds': round(end_time - start_time, 3),
                'success': True,
                'status': agent_result.status.value,
                'validation_passed': agent_result.validation_info.is_valid if agent_result.validation_info else True,
                'validation_category': agent_result.validation_info.category if agent_result.validation_info else 'success',
                'warnings': len(agent_result.validation_info.warnings) if agent_result.validation_info else 0,
                'input_mode_used': 'text-only',
                'attempt': agent_result.metadata.get('attempt', 1)
            }

            # Print results
            keywords_str = ', '.join([f"{k['keyword']} ({k['confidence']:.2f})" for k in keyword_data['keywords']])
            print(f"  Result: {keywords_str}")
            if keyword_data['reasoning']:
                print(f"  Reasoning: {keyword_data['reasoning'][:100]}...")
            
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
            
            # Extract keyword data if result exists (even if validation failed)
            keyword_data = None
            if agent_result.result:
                keyword_data = extract_keyword_result(agent_result.result)
            
            result_dict = {
                'item_id': item_id,
                'property': 'keywords',
                'config_id': 'classifier-keywords-bags',
                'keywords': keyword_data['keywords'] if keyword_data else [],
                'keyword_count': keyword_data['keyword_count'] if keyword_data else 0,
                'reasoning': keyword_data['reasoning'] if keyword_data else '',
                'processing_time_seconds': round(end_time - start_time, 3),
                'success': False,
                'status': agent_result.status.value,
                'error': error_msg,
                'error_type': agent_result.error_report.error_type if agent_result.error_report else 'unknown',
                'recoverable': agent_result.error_report.recoverable if agent_result.error_report else False,
                'validation_passed': agent_result.validation_info.is_valid if agent_result.validation_info else False,
                'validation_category': agent_result.validation_info.category if agent_result.validation_info else 'unknown',
                'warnings': len(agent_result.validation_info.warnings) if agent_result.validation_info else 0,
                'input_mode_used': 'text-only',
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
        import traceback
        traceback.print_exc()
        return {
            'item_id': item_id,
            'property': 'keywords',
            'config_id': 'classifier-keywords-bags',
            'keywords': [],
            'keyword_count': 0,
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
            'input_mode_used': 'text-only',
            'attempt': 0
        }


def run_keyword_classification(
    general_input_text: Dict[str, Any],
    text_to_avoid: Dict[str, Any],
    full_config: Dict[str, Any],
    item_id: str = "1"
) -> Dict[str, Any]:
    """
    Run keyword classification.

    Args:
        general_input_text: Dict with text fields to process (e.g., {"title": "...", "description": "..."})
        text_to_avoid: Dict with classifications to avoid (e.g., {"material": "Leather", "condition": "Excellent"})
        full_config: Full agent configuration bundle (from load_full_agent_config)
        item_id: Item identifier (default: "1")

    Returns:
        Classification result dict
    """
    config_id = 'classifier-keywords-bags'
    model_config = full_config['model_config']

    print(f"\n{'='*70}")
    print(f"KEYWORD CLASSIFIER ORCHESTRATION (config_id={config_id})")
    print(f"Model: {model_config.get('model')}")
    print(f"Input mode: text-only")
    print(f"Schema: None (free-form keywords)")
    print(f"{'='*70}")

    # Initialize agent with full config
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    
    print(f"Loaded prompt template: {full_config.get('prompt_template_id')}")
    print(f"No schema (keywords are free-form)")

    # Run classification
    result = classify_keywords(
        general_input_text=general_input_text,
        text_to_avoid=text_to_avoid,
        agent=agent,
        item_id=item_id
    )

    return result


def _build_text_to_avoid_from_row(row: pd.Series) -> Dict[str, Any]:
    """Build text_to_avoid dict from TRUSS_*_name columns in a row."""
    text_to_avoid = {}
    truss_name_cols = [c for c in row.index if isinstance(c, str) and c.startswith('TRUSS_') and c.endswith('_name')]
    for col in truss_name_cols:
        value = row.get(col)
        if value is not None and pd.notna(value):
            value_str = str(value).strip()
            if value_str:
                property_name = col.replace('TRUSS_', '').replace('_name', '')
                text_to_avoid[property_name] = value_str
    return text_to_avoid


def run_bulk_keyword_classification(
    df: pd.DataFrame,
    general_input_fn: Callable[[pd.Series, int], Dict[str, Any]],
    config_loader: ConfigLoader,
    max_workers: int = 200,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> pd.DataFrame:
    """
    Run keyword classification on all rows in parallel (bulk mode).

    Parallelism is internal. Builds text_to_avoid from TRUSS_*_name columns in each row.
    Expects df to already have all TRUSS classification columns (run last in pipeline).

    Args:
        df: DataFrame with TRUSS_*_name columns for text_to_avoid
        general_input_fn: (row, idx) -> dict of text fields for keyword extraction
        config_loader: ConfigLoader instance (loads classifier-keywords-bags)
        max_workers: Max concurrent agent calls
        batch_size: Process in batches (None = all at once)
        progress_callback: Optional (idx, total)

    Returns:
        DataFrame with TRUSS_keyword_1..3, TRUSS_keyword_*_confidence, TRUSS_keyword_reasoning
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Semaphore

    config_id = 'classifier-keywords-bags'
    try:
        full_config = config_loader.load_full_agent_config(config_id)
    except Exception as e:
        return pd.DataFrame({
            'TRUSS_keyword_1': [''] * len(df),
            'TRUSS_keyword_2': [''] * len(df),
            'TRUSS_keyword_3': [''] * len(df),
            'TRUSS_keyword_1_confidence': [0.0] * len(df),
            'TRUSS_keyword_2_confidence': [0.0] * len(df),
            'TRUSS_keyword_3_confidence': [0.0] * len(df),
            'TRUSS_keyword_reasoning': [f'Config error: {e}'] * len(df)
        })

    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    total = len(df)
    results = [None] * total
    semaphore = Semaphore(max_workers)

    def _default_row():
        return {
            'TRUSS_keyword_1': '',
            'TRUSS_keyword_2': '',
            'TRUSS_keyword_3': '',
            'TRUSS_keyword_1_confidence': 0.0,
            'TRUSS_keyword_2_confidence': 0.0,
            'TRUSS_keyword_3_confidence': 0.0,
            'TRUSS_keyword_reasoning': ''
        }

    def process_row(idx: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
        with semaphore:
            try:
                general_input = general_input_fn(row, idx + 1)
                if not general_input:
                    r = _default_row()
                    r['TRUSS_keyword_reasoning'] = 'No text data available for keyword classification'
                    return (idx, r)
                text_to_avoid = _build_text_to_avoid_from_row(row)
                result = classify_keywords(
                    general_input_text=general_input,
                    text_to_avoid=text_to_avoid,
                    agent=agent,
                    item_id=str(idx + 1)
                )
                keywords = result.get('keywords', [])
                r = _default_row()
                r['TRUSS_keyword_reasoning'] = result.get('reasoning', '')
                for i, kw in enumerate(keywords[:3], 1):
                    if isinstance(kw, dict):
                        r[f'TRUSS_keyword_{i}'] = kw.get('keyword', '')
                        r[f'TRUSS_keyword_{i}_confidence'] = kw.get('confidence', 0.0)
                return (idx, r)
            except Exception as e:
                r = _default_row()
                r['TRUSS_keyword_reasoning'] = f'Error: {str(e)[:100]}'
                return (idx, r)

    if batch_size:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_df = df.iloc[batch_start:batch_end]
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_df))) as executor:
                futures = {
                    executor.submit(process_row, batch_start + i, row): batch_start + i
                    for i, (_, row) in enumerate(batch_df.iterrows())
                }
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    if progress_callback:
                        progress_callback(idx + 1, total)
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
            futures = {
                executor.submit(process_row, idx, row): idx
                for idx, (_, row) in enumerate(df.iterrows())
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                if progress_callback:
                    progress_callback(idx + 1, total)

    return pd.DataFrame(results)


def save_result_to_json(result: Dict[str, Any], output_file: str):
    """Save keyword classification result to JSON."""
    if not result:
        print("No result to save")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResult saved to: {output_file}")
    print(f"Success: {result.get('success', False)}")
    if result.get('keywords'):
        print(f"Keywords extracted: {len(result['keywords'])}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Keyword Classifier Orchestration')
    parser.add_argument('--general-input-text', type=str, required=True,
                       help='JSON string or file path with text to process. For PowerShell, use: --general-input-text \'{"title":"..."}\' or --general-input-text input.json')
    parser.add_argument('--text-to-avoid', type=str, required=True,
                       help='JSON string or file path with classifications to avoid. For PowerShell, use: --text-to-avoid \'{"material":"Leather"}\' or --text-to-avoid avoid.json')
    parser.add_argument('--item-id', type=str, default='1',
                       help='Item identifier (default: 1)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: auto-generated)')
    
    # Arguments for flexible loading
    parser.add_argument('--mode', type=str, default='api', choices=['api', 'local', 'dynamo'],
                       help='Configuration loading mode (default: api)')
    parser.add_argument('--env', type=str, default='dev', choices=['dev', 'staging', 'prod'],
                       help='Environment for DynamoDB tables (used in dynamo mode)')

    args = parser.parse_args()

    config_id = 'classifier-keywords-bags'
    
    print(f"Keyword Classifier Orchestration - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode.upper()} ({args.env})")
    print(f"Config ID: {config_id} (hardcoded)")
    print(f"Item ID: {args.item_id}")

    try:
        # Parse JSON inputs
        # Handle PowerShell argument splitting by trying to reconstruct JSON
        general_input_text_str = args.general_input_text
        text_to_avoid_str = args.text_to_avoid
        
        # Check if argparse failed to capture the full JSON (common in PowerShell)
        # Look for the arguments in sys.argv and reconstruct
        try:
            general_idx = sys.argv.index('--general-input-text')
            avoid_idx = sys.argv.index('--text-to-avoid')
            
            # If there are arguments between the flag and the next flag, reconstruct
            if general_idx + 1 < avoid_idx:
                # Reconstruct JSON from all arguments between the flags
                potential_json_parts = sys.argv[general_idx + 1:avoid_idx]
                if len(potential_json_parts) > 1:
                    # Join all parts and try to parse
                    general_input_text_str = ' '.join(potential_json_parts)
            
            # Same for text-to-avoid
            if avoid_idx + 1 < len(sys.argv):
                # Find where the next argument starts (look for --)
                next_flag_idx = len(sys.argv)
                for i in range(avoid_idx + 2, len(sys.argv)):
                    if sys.argv[i].startswith('--'):
                        next_flag_idx = i
                        break
                
                if avoid_idx + 1 < next_flag_idx:
                    potential_json_parts = sys.argv[avoid_idx + 1:next_flag_idx]
                    if len(potential_json_parts) > 1:
                        text_to_avoid_str = ' '.join(potential_json_parts)
        except (ValueError, IndexError):
            # If reconstruction fails, use what argparse captured
            pass
        
        general_input_text = parse_json_input(general_input_text_str)
        text_to_avoid = parse_json_input(text_to_avoid_str)
        
        print(f"General input text fields: {list(general_input_text.keys())}")
        print(f"Text to avoid fields: {list(text_to_avoid.keys())}")

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
        print(f"Loading configuration for {config_id}...")
        full_config = config_loader.load_full_agent_config(config_id)
        print(f"Configuration loaded: Model={full_config['model_config'].get('model')}")
        print(f"Schema: {'None (free-form keywords)' if not full_config.get('schema_id') else full_config.get('schema_id')}")

        # Run classification
        result = run_keyword_classification(
            general_input_text=general_input_text,
            text_to_avoid=text_to_avoid,
            full_config=full_config,
            item_id=args.item_id
        )

        # Save result
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"keyword_result_{config_id}_{timestamp}.json"

        save_result_to_json(result, output_file)

        print(f"\n{'='*70}")
        print("KEYWORD CLASSIFIER ORCHESTRATION COMPLETE")
        print(f"{'='*70}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
