"""
Model Size Classification Orchestration
======================================

Two-pipeline workflow for model size classification:
1. Pipeline 1 (Textual): Classifier agent picks size from options
2. Pipeline 2 (Numerical): Extracts measurements, converts to cm, finds closest match

Combines results with majority vote.

Usage:
    python model_size_classification_orchestration.py --text "Hermes Birkin 30cm bag" --model-id 123
    python model_size_classification_orchestration.py --text "12 inches by 10 inches"
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore
from typing import Dict, Any, List, Optional, Tuple, Callable
import json

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_architecture import LLMAnnotationAgent
from agent_architecture.validation import AgentResult, AgentStatus
from agent_orchestration.csv_config_loader import ConfigLoader
from agent_utils.bigquery_model_size_tool import get_model_size_options
from agent_utils.measurement_utils import convert_to_cm, find_closest_size_match


def run_pipeline1_textual_classification(
    raw_text: str,
    size_options: List[Dict[str, Any]],
    config_loader: Optional[ConfigLoader] = None,
    env: str = 'staging'
) -> Dict[str, Any]:
    """
    Pipeline 1: Textual size classification.
    
    Args:
        raw_text: Input text
        size_options: List of size options from BigQuery
        config_loader: ConfigLoader instance
        env: Environment for config loading
        
    Returns:
        Classification result dict
    """
    print("\n" + "="*70)
    print("PIPELINE 1: TEXTUAL SIZE CLASSIFICATION")
    print("="*70)
    
    if not size_options:
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "confidence": 0.0,
            "reasoning": "No size options available",
            "error": "No size options"
        }
    
    # Initialize config loader if needed
    if config_loader is None:
        config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
    
    # Load textual classifier config
    try:
        full_config = config_loader.load_full_agent_config('model-size-textual-classifier-v1')
    except Exception as e:
        print(f"ERROR: Could not load textual classifier config: {e}")
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "confidence": 0.0,
            "reasoning": f"Config error: {e}",
            "error": str(e)
        }
    
    # Build dynamic schema from size options
    schema_content = {}
    for option in size_options:
        size_id = str(option['id'])
        size_name = option.get('size', f"Size {option['id']}")
        schema_content[size_id] = {
            "name": size_name,
            "description": f"Size ID {option['id']}"
        }
    
    # Inject schema into config
    full_config['schema'] = {
        "schema_id": "model-size-dynamic",
        "schema_content": schema_content,
        "schema_metadata": {}
    }
    
    # Initialize agent
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    
    # Build input text with size options
    size_options_text = "\n".join([
        f"ID {opt['id']}: {opt.get('size', 'N/A')}"
        for opt in size_options
    ])
    
    input_text = f"""Product Information:
{raw_text}

Available Sizes:
{size_options_text}

Select the most appropriate size ID based on the product information."""
    
    print(f"Classifying from {len(size_options)} size option(s)...")
    
    input_data = {
        "item_id": "size_classification",
        "text_input": input_text,
        "input_mode": "text-only",
        "format_as_product_info": False
    }
    
    agent_result = agent.execute(input_data=input_data)
    
    if agent_result.status == AgentStatus.SUCCESS:
        result = agent_result.result
        prediction_id = result.get('prediction_id')
        
        # Map ID to size name
        size_name = ""
        for option in size_options:
            if option['id'] == prediction_id:
                size_name = option.get('size', '')
                break
        
        scores = result.get('scores', [])
        confidence = scores[0]['score'] if scores else 0.0
        reasoning = result.get('reasoning', '')
        
        print(f"Textual classification: {size_name} (ID: {prediction_id}, confidence: {confidence:.2f})")
        
        return {
            "success": True,
            "prediction_id": prediction_id,
            "size": size_name,
            "confidence": confidence,
            "reasoning": reasoning,
            "error": None
        }
    else:
        error_msg = agent_result.error_report.message if agent_result.error_report else "Unknown error"
        print(f"ERROR: {error_msg}")
        
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "confidence": 0.0,
            "reasoning": f"Classification error: {error_msg}",
            "error": error_msg
        }


def run_pipeline2_numerical_extraction(
    raw_text: str,
    size_options: List[Dict[str, Any]],
    config_loader: Optional[ConfigLoader] = None,
    env: str = 'staging'
) -> Dict[str, Any]:
    """
    Pipeline 2: Numerical measurement extraction and matching.
    
    Args:
        raw_text: Input text
        size_options: List of size options from BigQuery
        config_loader: ConfigLoader instance
        env: Environment for config loading
        
    Returns:
        Classification result dict
    """
    print("\n" + "="*70)
    print("PIPELINE 2: NUMERICAL MEASUREMENT EXTRACTION")
    print("="*70)
    
    if not size_options:
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "distance": None,
            "reasoning": "No size options available",
            "error": "No size options"
        }
    
    # Initialize config loader if needed
    if config_loader is None:
        config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
    
    # Load measurement extraction config
    try:
        full_config = config_loader.load_full_agent_config('model-size-measurement-extraction-v1')
    except Exception as e:
        print(f"ERROR: Could not load measurement extraction config: {e}")
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "distance": None,
            "reasoning": f"Config error: {e}",
            "error": str(e)
        }
    
    # Initialize agent
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    
    input_data = {
        "item_id": "measurement_extraction",
        "text_input": raw_text,
        "input_mode": "text-only",
        "format_as_product_info": False
    }
    
    print("Extracting measurements from text...")
    
    agent_result = agent.execute(input_data=input_data)
    
    if agent_result.status != AgentStatus.SUCCESS:
        error_msg = agent_result.error_report.message if agent_result.error_report else "Unknown error"
        print(f"ERROR: {error_msg}")
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "distance": None,
            "reasoning": f"Extraction error: {error_msg}",
            "error": error_msg
        }
    
    # Parse extraction result
    result = agent_result.result
    measurements = result.get('measurements', {})
    unit = result.get('unit', 'cm')
    
    # Check if measurements are actually present (not empty dict and not all None/null)
    has_valid_measurements = False
    if measurements:
        # Check if at least one dimension has a non-None value
        for dim in ['height', 'width', 'length']:
            if measurements.get(dim) is not None:
                has_valid_measurements = True
                break
    
    if not has_valid_measurements:
        print("No measurements extracted - skipping size matching")
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "distance": None,
            "reasoning": "No measurements found in text - cannot perform numerical matching",
            "error": "No measurements extracted",
            "no_proposal": True  # Explicit flag for no proposal
        }
    
    # Validate unit
    valid_units = ['inches', 'cm', 'mm', 'in']
    if unit.lower() not in [u.lower() for u in valid_units]:
        print(f"WARNING: Invalid unit '{unit}', defaulting to 'cm'")
        unit = 'cm'
    else:
        # Normalize unit
        unit_lower = unit.lower()
        if unit_lower in ['in', 'inches']:
            unit = 'inches'
        elif unit_lower in ['cm', 'centimeter', 'centimetre']:
            unit = 'cm'
        elif unit_lower in ['mm', 'millimeter', 'millimetre']:
            unit = 'mm'
    
    print(f"Extracted measurements: {measurements} ({unit})")
    
    # Find closest match
    print("Finding closest size match...")
    closest_match = find_closest_size_match(measurements, size_options, unit=unit)
    
    if not closest_match:
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "distance": None,
            "reasoning": "Could not find matching size",
            "error": "No match found"
        }
    
    print(f"Closest match: {closest_match['size']} (ID: {closest_match['id']}, distance: {closest_match['distance']:.2f} cm)")
    
    return {
        "success": True,
        "prediction_id": closest_match['id'],
        "size": closest_match['size'],
        "distance": closest_match['distance'],
        "extracted_measurements": measurements,
        "unit": unit,
        "reasoning": f"Euclidean distance: {closest_match['distance']:.2f} cm",
        "error": None
    }


def combine_with_majority_vote(
    pipeline1_result: Dict[str, Any],
    pipeline2_result: Dict[str, Any],
    size_options: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combine pipeline results with majority vote logic.
    
    Args:
        pipeline1_result: Result from textual pipeline
        pipeline2_result: Result from numerical pipeline
        size_options: List of size options for validation
        
    Returns:
        Combined result dict
    """
    print("\n" + "="*70)
    print("COMBINING RESULTS (MAJORITY VOTE)")
    print("="*70)
    
    # Build validation mapping
    valid_size_ids = {opt['id'] for opt in size_options}
    id_to_size_map = {opt['id']: opt.get('size', '') for opt in size_options}
    
    # Get proposals
    proposal1 = pipeline1_result.get('prediction_id') if pipeline1_result.get('success') else None
    proposal2 = pipeline2_result.get('prediction_id') if pipeline2_result.get('success') else None
    
    # Check if Pipeline 2 has no proposal due to missing measurements
    pipeline2_no_proposal = pipeline2_result.get('no_proposal', False)
    
    print(f"Pipeline 1 proposal: {proposal1} ({pipeline1_result.get('size', 'N/A')})")
    if pipeline2_no_proposal:
        print(f"Pipeline 2 proposal: None (no measurements extracted - skipping numerical matching)")
    else:
        print(f"Pipeline 2 proposal: {proposal2} ({pipeline2_result.get('size', 'N/A')})")
    
    # Validation: Check proposals are in valid IDs
    validation_errors = []
    
    if proposal1 is not None and proposal1 not in valid_size_ids:
        validation_errors.append(f"Pipeline 1 proposal ID {proposal1} not in valid options")
        proposal1 = None
    
    if proposal2 is not None and proposal2 not in valid_size_ids:
        validation_errors.append(f"Pipeline 2 proposal ID {proposal2} not in valid options")
        proposal2 = None
    
    # Majority vote logic
    if proposal1 is None and proposal2 is None:
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "confidence": 0.0,
            "reasoning": "Both pipelines failed or returned invalid results",
            "pipeline1_result": pipeline1_result,
            "pipeline2_result": pipeline2_result,
            "validation_errors": validation_errors,
            "error": "No valid proposals"
        }
    
    if proposal1 is not None and proposal2 is None:
        # Only pipeline 1 succeeded
        size_name = id_to_size_map.get(proposal1, '')
        
        # Final validation
        final_validation_errors = []
        if proposal1 not in valid_size_ids:
            final_validation_errors.append(f"Proposal ID {proposal1} not in BigQuery results")
        elif size_name != id_to_size_map.get(proposal1, ''):
            final_validation_errors.append(f"Size name mismatch")
        
        if final_validation_errors:
            print(f"FINAL VALIDATION FAILED:")
            for error in final_validation_errors:
                print(f"  - {error}")
            validation_errors.extend(final_validation_errors)
            return {
                "success": False,
                "prediction_id": None,
                "size": "",
                "confidence": 0.0,
                "reasoning": f"Final validation failed: {'; '.join(final_validation_errors)}",
                "pipeline1_result": pipeline1_result,
                "pipeline2_result": pipeline2_result,
                "validation_errors": validation_errors,
                "error": "Final validation failed"
            }
        
        print(f"Using Pipeline 1 result (only proposal): {size_name} (ID: {proposal1})")
        print(f"Final validation passed: ID {proposal1} and size '{size_name}' match BigQuery results")
        
        return {
            "success": True,
            "prediction_id": proposal1,
            "size": size_name,
            "confidence": pipeline1_result.get('confidence', 0.0),
            "reasoning": f"Pipeline 1 only: {pipeline1_result.get('reasoning', '')}",
            "pipeline1_result": pipeline1_result,
            "pipeline2_result": pipeline2_result,
            "validation_errors": validation_errors,
            "validation_passed": True,
            "error": None
        }
    
    if proposal1 is None and proposal2 is not None:
        # Only pipeline 2 succeeded
        size_name = id_to_size_map.get(proposal2, '')
        
        # Final validation
        final_validation_errors = []
        if proposal2 not in valid_size_ids:
            final_validation_errors.append(f"Proposal ID {proposal2} not in BigQuery results")
        elif size_name != id_to_size_map.get(proposal2, ''):
            final_validation_errors.append(f"Size name mismatch")
        
        if final_validation_errors:
            print(f"FINAL VALIDATION FAILED:")
            for error in final_validation_errors:
                print(f"  - {error}")
            validation_errors.extend(final_validation_errors)
            return {
                "success": False,
                "prediction_id": None,
                "size": "",
                "confidence": 0.0,
                "reasoning": f"Final validation failed: {'; '.join(final_validation_errors)}",
                "pipeline1_result": pipeline1_result,
                "pipeline2_result": pipeline2_result,
                "validation_errors": validation_errors,
                "error": "Final validation failed"
            }
        
        print(f"Using Pipeline 2 result (only proposal): {size_name} (ID: {proposal2})")
        print(f"Final validation passed: ID {proposal2} and size '{size_name}' match BigQuery results")
        
        return {
            "success": True,
            "prediction_id": proposal2,
            "size": size_name,
            "confidence": 1.0 - (pipeline2_result.get('distance', 0.0) / 100.0),  # Convert distance to confidence
            "reasoning": f"Pipeline 2 only: {pipeline2_result.get('reasoning', '')}",
            "pipeline1_result": pipeline1_result,
            "pipeline2_result": pipeline2_result,
            "validation_errors": validation_errors,
            "validation_passed": True,
            "error": None
        }
    
    # Both proposals exist - check if they agree
    if proposal1 == proposal2:
        size_name = id_to_size_map.get(proposal1, '')
        
        # Final validation: Ensure ID and size match BigQuery
        final_validation_errors = []
        if proposal1 not in valid_size_ids:
            final_validation_errors.append(f"Final proposal ID {proposal1} not in BigQuery results")
        elif size_name != id_to_size_map.get(proposal1, ''):
            final_validation_errors.append(f"Size name mismatch: expected '{id_to_size_map.get(proposal1)}', got '{size_name}'")
        
        if final_validation_errors:
            print(f"FINAL VALIDATION FAILED:")
            for error in final_validation_errors:
                print(f"  - {error}")
            validation_errors.extend(final_validation_errors)
            return {
                "success": False,
                "prediction_id": None,
                "size": "",
                "confidence": 0.0,
                "reasoning": f"Final validation failed: {'; '.join(final_validation_errors)}",
                "pipeline1_result": pipeline1_result,
                "pipeline2_result": pipeline2_result,
                "validation_errors": validation_errors,
                "error": "Final validation failed"
            }
        
        print(f"Pipelines agree: {size_name} (ID: {proposal1})")
        print(f"Final validation passed: ID {proposal1} and size '{size_name}' match BigQuery results")
        
        return {
            "success": True,
            "prediction_id": proposal1,
            "size": size_name,
            "confidence": (pipeline1_result.get('confidence', 0.0) + (1.0 - pipeline2_result.get('distance', 0.0) / 100.0)) / 2.0,
            "reasoning": f"Both pipelines agree: {pipeline1_result.get('reasoning', '')}",
            "pipeline1_result": pipeline1_result,
            "pipeline2_result": pipeline2_result,
            "validation_errors": validation_errors,
            "validation_passed": True,
            "error": None
        }
    else:
        # Pipelines disagree
        size1 = id_to_size_map.get(proposal1, '')
        size2 = id_to_size_map.get(proposal2, '')
        print(f"Pipelines disagree: Pipeline 1={size1} (ID: {proposal1}), Pipeline 2={size2} (ID: {proposal2})")
        return {
            "success": False,
            "prediction_id": None,
            "size": "",
            "confidence": 0.0,
            "reasoning": f"Pipelines disagree: Pipeline 1={size1} (ID: {proposal1}), Pipeline 2={size2} (ID: {proposal2})",
            "pipeline1_result": pipeline1_result,
            "pipeline2_result": pipeline2_result,
            "validation_errors": validation_errors,
            "error": "Pipelines disagree"
        }


def run_model_size_classification_workflow(
    raw_text: str,
    model_id: int,
    env: str = 'staging'
) -> Dict[str, Any]:
    """
    Run the complete model size classification workflow.
    
    Args:
        raw_text: Raw text input
        model_id: Required model_id to filter size options (size options are model-specific)
        env: Environment for config loading
        
    Returns:
        Complete workflow result
    """
    print("="*70)
    print("MODEL SIZE CLASSIFICATION WORKFLOW")
    print("="*70)
    print(f"Input text: {raw_text[:100]}{'...' if len(raw_text) > 100 else ''}")
    print(f"Model ID: {model_id}")
    print("="*70)
    
    config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
    
    # Step 1: Get size options from BigQuery
    print("\n[STEP 1] Retrieving model size options from BigQuery...")
    try:
        size_options = get_model_size_options(model_id=model_id, verbose=True)
        
        if not size_options:
            return {
                "workflow_status": "failed",
                "error": f"No size options found in BigQuery for model_id={model_id}. Please verify the model_id is correct.",
                "size_options": [],
                "model_id": model_id
            }
        
        print(f"Retrieved {len(size_options)} size option(s)")
        for opt in size_options[:5]:
            print(f"  - ID {opt['id']}: {opt.get('size', 'N/A')}")
        
    except Exception as e:
        print(f"ERROR: Failed to retrieve size options: {e}")
        return {
            "workflow_status": "failed",
            "error": f"BigQuery error: {e}",
            "size_options": []
        }
    
    # Step 2: Run both pipelines
    print("\n[STEP 2] Running parallel pipelines...")
    
    pipeline1_result = run_pipeline1_textual_classification(
        raw_text=raw_text,
        size_options=size_options,
        config_loader=config_loader,
        env=env
    )
    
    pipeline2_result = run_pipeline2_numerical_extraction(
        raw_text=raw_text,
        size_options=size_options,
        config_loader=config_loader,
        env=env
    )
    
    # Step 3: Combine with majority vote
    print("\n[STEP 3] Combining results...")
    final_result = combine_with_majority_vote(
        pipeline1_result=pipeline1_result,
        pipeline2_result=pipeline2_result,
        size_options=size_options
    )
    
    # Final summary
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    if final_result['success']:
        print(f"Final Size: {final_result.get('size', 'N/A')} (ID: {final_result.get('prediction_id')})")
        print(f"Confidence: {final_result.get('confidence', 0.0):.2f}")
    else:
        print(f"Classification failed: {final_result.get('error', 'Unknown error')}")
        print(f"Pipeline 1: {pipeline1_result.get('size', 'N/A')} (ID: {pipeline1_result.get('prediction_id')})")
        print(f"Pipeline 2: {pipeline2_result.get('size', 'N/A')} (ID: {pipeline2_result.get('prediction_id')})")
    print("="*70)
    
    return {
        "workflow_status": "success" if final_result['success'] else "failed",
        "final_result": final_result,
        "size_options": size_options,
        "error": final_result.get('error')
    }


def run_bulk_size_classification(
    df: pd.DataFrame,
    text_extraction_fn: Callable[[pd.Series, int], str],
    env: str = 'staging',
    max_workers: int = 200,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> pd.DataFrame:
    """
    Run size classification on all rows in parallel (bulk mode).

    Parallelism is internal. Expects df to have TRUSS_model_id (from model
    classification). Calls run_model_size_classification_workflow per row.

    Args:
        df: DataFrame with TRUSS_model_id column
        text_extraction_fn: (row, idx) -> text string for size classification
        env: Environment for config loading
        max_workers: Max concurrent workflow calls
        batch_size: Process in batches (None = all at once)
        progress_callback: Optional (idx, total)

    Returns:
        DataFrame with TRUSS_size_*, TRUSS_height_cm, TRUSS_width_cm, TRUSS_depth_cm
    """
    if 'TRUSS_model_id' not in df.columns:
        return pd.DataFrame({
            'TRUSS_size_id': [None] * len(df),
            'TRUSS_size_name': [''] * len(df),
            'TRUSS_size_confidence': [0.0] * len(df),
            'TRUSS_size_reasoning': ['TRUSS_model_id not available - model classification required first'] * len(df),
            'TRUSS_height_cm': [None] * len(df),
            'TRUSS_width_cm': [None] * len(df),
            'TRUSS_depth_cm': [None] * len(df)
        })

    total = len(df)
    results = [None] * total
    semaphore = Semaphore(max_workers)

    def _default_row():
        return {
            'TRUSS_size_id': None,
            'TRUSS_size_name': '',
            'TRUSS_size_confidence': 0.0,
            'TRUSS_size_reasoning': '',
            'TRUSS_height_cm': None,
            'TRUSS_width_cm': None,
            'TRUSS_depth_cm': None
        }

    def process_row(idx: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
        with semaphore:
            try:
                model_id = row.get('TRUSS_model_id')
                if not model_id or pd.isna(model_id):
                    r = _default_row()
                    r['TRUSS_size_reasoning'] = 'No TRUSS_model_id available - skipping size classification'
                    return (idx, r)
                try:
                    model_id_int = int(model_id)
                except (ValueError, TypeError):
                    r = _default_row()
                    r['TRUSS_size_reasoning'] = f'Invalid TRUSS_model_id: {model_id}'
                    return (idx, r)
                text = text_extraction_fn(row, idx + 1)
                if not text or not str(text).strip():
                    r = _default_row()
                    r['TRUSS_size_reasoning'] = 'No text data available for size classification'
                    return (idx, r)
                workflow_result = run_model_size_classification_workflow(
                    raw_text=text,
                    model_id=model_id_int,
                    env=env
                )
                measurements = {
                    'TRUSS_height_cm': None,
                    'TRUSS_width_cm': None,
                    'TRUSS_depth_cm': None
                }
                pipeline2_result = workflow_result.get('final_result', {}).get('pipeline2_result', {})
                if pipeline2_result:
                    ext = pipeline2_result.get('extracted_measurements', {})
                    if ext:
                        measurements = {
                            'TRUSS_height_cm': ext.get('height'),
                            'TRUSS_width_cm': ext.get('width'),
                            'TRUSS_depth_cm': ext.get('length')
                        }
                if workflow_result.get('workflow_status') == 'success':
                    final_result = workflow_result.get('final_result', {})
                    if final_result.get('success'):
                        r = {
                            'TRUSS_size_id': final_result.get('prediction_id'),
                            'TRUSS_size_name': final_result.get('size', ''),
                            'TRUSS_size_confidence': final_result.get('confidence', 0.0),
                            'TRUSS_size_reasoning': final_result.get('reasoning', '')
                        }
                        r.update(measurements)
                        return (idx, r)
                    r = _default_row()
                    r['TRUSS_size_reasoning'] = f"Size classification failed: {final_result.get('error', 'Unknown error')}"
                    r.update(measurements)
                    return (idx, r)
                r = _default_row()
                r['TRUSS_size_reasoning'] = f"Size workflow failed: {workflow_result.get('error', 'Unknown error')}"
                r.update(measurements)
                return (idx, r)
            except Exception as e:
                r = _default_row()
                r['TRUSS_size_reasoning'] = f'Error: {str(e)[:100]}'
                return (idx, r)

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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Model Size Classification Workflow - Two-pipeline size classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python model_size_classification_orchestration.py --text "Hermes Birkin 30cm bag" --model-id 123
    python model_size_classification_orchestration.py --text "12 inches by 10 inches by 8 inches" --model-id 456
    python model_size_classification_orchestration.py --text "Birkin 30" --model-id 123 --env staging
        """
    )
    
    parser.add_argument('--text', type=str, required=True,
                       help='Raw text input to analyze')
    parser.add_argument('--model-id', type=int, required=True,
                       help='Model ID to filter size options (required - size options are model-specific)')
    parser.add_argument('--env', type=str, default='staging',
                       choices=['dev', 'staging', 'prod'],
                       help='Environment for config loading (default: staging)')
    
    args = parser.parse_args()
    
    try:
        result = run_model_size_classification_workflow(
            raw_text=args.text,
            model_id=args.model_id,
            env=args.env
        )
        
        # Print JSON result
        print("\n" + "="*70)
        print("RESULT (JSON)")
        print("="*70)
        print(json.dumps(result, indent=2))
        
        sys.exit(0 if result['workflow_status'] == 'success' else 1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
