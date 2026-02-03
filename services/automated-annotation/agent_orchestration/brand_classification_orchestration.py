"""
Brand Classification Orchestration
===================================

Two-agent workflow for brand classification:
1. Agent 1: Extracts brand candidates from text and searches BigQuery
2. Agent 2: Classifies which brand(s) apply using retrieved brand data

Usage:
    python brand_classification_orchestration.py --text "Nike Air Max sneakers" --name "Product Name"
    python brand_classification_orchestration.py --text "LV handbag" --name "Louis Vuitton Bag"
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore
from typing import Dict, Any, List, Optional, Callable, Tuple
import copy
import json

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_architecture import LLMAnnotationAgent
from agent_architecture.validation import AgentResult, AgentStatus
from agent_orchestration.csv_config_loader import ConfigLoader
from agent_utils.bigquery_brand_tool import search_brand_database


def extract_brand_candidates_deterministic(raw_text: str, known_synonyms: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Deterministic extraction of brand candidates from text.
    Used as fallback or to seed Agent 1.
    
    Args:
        raw_text: Input text to analyze
        known_synonyms: Optional dict mapping canonical brands to synonym lists
        
    Returns:
        List of potential brand terms to search
    """
    import re
    candidates = set()
    text_lower = raw_text.lower()
    
    # 1. Direct match against known synonyms
    if known_synonyms:
        all_known_terms = {}
        for brand, syns in known_synonyms.items():
            all_known_terms[brand.lower()] = brand
            for syn in syns:
                all_known_terms[syn.lower()] = brand
        
        for term, canonical in all_known_terms.items():
            if term in text_lower:
                candidates.add(canonical)
                candidates.add(term)
    
    # 2. Extract capitalized sequences (potential brands)
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', raw_text)
    candidates.update(capitalized)
    
    # 3. Extract words before common product terms
    product_indicators = ['shirt', 'dress', 'jacket', 'bag', 'handbag', 'shoes', 'sneakers', 
                         'jeans', 'pants', 'coat', 'boots', 'wallet', 'belt', 'watch']
    for indicator in product_indicators:
        pattern = rf'(\b\w+(?:\s+\w+)?)\s+{indicator}'
        matches = re.findall(pattern, text_lower)
        candidates.update(matches)
    
    # 4. Common brand abbreviations
    abbreviations = {
        'lv': 'Louis Vuitton',
        'ysl': 'Yves Saint Laurent',
        'ck': 'Calvin Klein',
        'dkny': 'DKNY',
        'rl': 'Ralph Lauren',
        'gucci': 'Gucci',
        'prada': 'Prada',
        'chanel': 'Chanel',
        'hermes': 'Hermes',
        'dior': 'Dior'
    }
    
    for abbr, full in abbreviations.items():
        if abbr in text_lower:
            candidates.add(full)
            candidates.add(abbr)
    
    return list(candidates)


def run_agent1_brand_extraction(
    raw_text: str,
    name: Optional[str] = None,
    known_synonyms: Optional[Dict[str, List[str]]] = None,
    config_loader: Optional[ConfigLoader] = None,
    env: str = 'staging',
    verbose: bool = False,
    agent: Optional[LLMAnnotationAgent] = None,
    full_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Agent 1: Extract brand candidates and search BigQuery.

    Args:
        raw_text: Raw text input
        name: Optional product name
        known_synonyms: Optional brand synonyms dict
        config_loader: ConfigLoader instance (not needed if agent or full_config provided)
        env: Environment for config loading
        verbose: If True, print progress and diagnostics
        agent: Pre-built LLMAnnotationAgent to reuse (thread-safe for Agent1)
        full_config: Pre-loaded config (skips DynamoDB call)

    Returns:
        Dict with extraction results
    """
    if verbose:
        print("\n" + "="*70)
        print("AGENT 1: BRAND EXTRACTION")
        print("="*70)
    
    if agent is None:
        if full_config is None:
            if config_loader is None:
                config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
            try:
                full_config = config_loader.load_full_agent_config('brand-extraction-v1')
            except Exception as e:
                if verbose:
                    print(f"ERROR: Could not load brand extraction config: {e}")
                return {
                    "success": False,
                    "extracted_candidates": [],
                    "search_terms_used": [],
                    "matched_brands": [],
                    "error": str(e)
                }
        agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    
    input_text = raw_text
    if name:
        input_text = f"Product Name: {name}\n\n{raw_text}"
    
    synonyms_section = ""
    if known_synonyms:
        synonyms_section = "\n\nKNOWN BRAND SYNONYMS:\n"
        for brand, syns in known_synonyms.items():
            synonyms_section += f"- {brand}: {', '.join(syns)}\n"
        input_text += synonyms_section
    
    if verbose:
        print(f"Input text length: {len(input_text)} chars")
        print("Extracting brand candidates...")

    input_data = {
        "item_id": "brand_extraction",
        "text_input": input_text,
        "input_mode": "text-only",
        "format_as_product_info": False
    }
    
    agent_result = agent.execute(input_data=input_data)
    
    if agent_result.status != AgentStatus.SUCCESS:
        error_msg = agent_result.error_report.message if agent_result.error_report else "Unknown error"
        if verbose:
            print(f"ERROR: {error_msg}")
            print("Falling back to deterministic extraction...")
        candidates = extract_brand_candidates_deterministic(raw_text, known_synonyms)
        search_terms = candidates
    else:
        result = agent_result.result
        candidates = result.get('extracted_candidates', [])
        search_terms = result.get('search_terms', candidates)
        if verbose:
            print(f"Agent 1 extracted {len(candidates)} candidate(s)")
            print(f"Search terms: {search_terms[:5]}{'...' if len(search_terms) > 5 else ''}")

    if verbose:
        print("\nSearching brand database...")
    try:
        search_results = search_brand_database(search_terms, verbose=verbose)
        
        if search_results is None:
            if verbose:
                print("WARNING: BigQuery returned None. No brands found.")
            return {
                "success": False,
                "extracted_candidates": candidates,
                "search_terms_used": search_terms,
                "matched_brands": [],
                "error": "BigQuery returned None"
            }
        
        matched_brands = search_results.get('matches', [])
        search_terms_used = search_results.get('search_terms_used', search_terms)
        
        if not matched_brands:
            if verbose:
                print("No brand matches found in BigQuery database.")
            return {
                "success": False,
                "extracted_candidates": candidates,
                "search_terms_used": search_terms_used,
                "matched_brands": [],
                "error": "No brands found in database"
            }
        
        if not isinstance(matched_brands, list):
            if verbose:
                print(f"WARNING: matched_brands is not a list: {type(matched_brands)}")
            return {
                "success": False,
                "extracted_candidates": candidates,
                "search_terms_used": search_terms_used,
                "matched_brands": [],
                "error": f"Invalid matches format: expected list, got {type(matched_brands)}"
            }
        
        if verbose:
            print(f"Found {len(matched_brands)} brand match(es)")
            for match in matched_brands[:5]:
                if isinstance(match, dict):
                    brand_name = match.get('brand', 'Unknown')
                    brand_id = match.get('id', 'N/A')
                    print(f"  - {brand_name} (ID: {brand_id})")

        return {
            "success": True,
            "extracted_candidates": candidates,
            "search_terms_used": search_terms_used,
            "matched_brands": matched_brands,
            "error": None
        }
        
    except Exception as e:
        if verbose:
            print(f"ERROR searching brand database: {e}")
        return {
            "success": False,
            "extracted_candidates": candidates,
            "search_terms_used": search_terms,
            "matched_brands": [],
            "error": str(e)
        }


def run_agent2_brand_classification(
    raw_text: str,
    name: Optional[str],
    matched_brands: List[Dict[str, Any]],
    config_loader: Optional[ConfigLoader] = None,
    env: str = 'staging',
    verbose: bool = False,
    base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Agent 2: Classify which brand(s) apply from matched brands.

    Args:
        raw_text: Original input text
        name: Optional product name
        matched_brands: List of {id, brand} from Agent 1
        config_loader: ConfigLoader instance (not needed if base_config provided)
        env: Environment for config loading
        verbose: If True, print progress and diagnostics
        base_config: Pre-loaded base config (schema will be overridden per call)

    Returns:
        Classification result dict
    """
    if verbose:
        print("\n" + "="*70)
        print("AGENT 2: BRAND CLASSIFICATION")
        print("="*70)

    if not matched_brands:
        if verbose:
            print("No matched brands to classify. Returning empty result.")
        return {
            "prediction_id": None,
            "brand_name": "",
            "confidence": 0.0,
            "reasoning": "No brands found in database"
        }
    
    if base_config is None:
        if config_loader is None:
            config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
        try:
            base_config = config_loader.load_full_agent_config('brand-classification-v1')
        except Exception as e:
            if verbose:
                print(f"ERROR: Could not load brand classification config: {e}")
            return {
                "prediction_id": None,
                "brand_name": "",
                "confidence": 0.0,
                "reasoning": f"Config error: {e}"
            }

    full_config = copy.deepcopy(base_config)

    if not isinstance(matched_brands, list):
        if verbose:
            print(f"ERROR: matched_brands is not a list: {type(matched_brands)}")
        return {
            "prediction_id": None,
            "brand_name": "",
            "confidence": 0.0,
            "reasoning": f"Invalid matched_brands format: expected list, got {type(matched_brands)}"
        }
    
    schema_content = {}
    for match in matched_brands:
        if not isinstance(match, dict):
            if verbose:
                print(f"WARNING: Skipping invalid match entry: {match} (not a dict)")
            continue

        brand_id = match.get('id')
        brand_name = match.get('brand')

        if brand_id is None or brand_name is None:
            if verbose:
                print(f"WARNING: Skipping match entry with missing fields: {match}")
            continue
        
        brand_id = str(brand_id)
        schema_content[brand_id] = {
            "name": brand_name,
            "description": f"Brand ID {brand_id}"
        }
    
    if not schema_content:
        if verbose:
            print("ERROR: No valid brand matches to build schema from")
        return {
            "prediction_id": None,
            "brand_name": "",
            "confidence": 0.0,
            "reasoning": "No valid brand matches found in matched_brands"
        }
    
    full_config['schema'] = {
        "schema_id": "brand-dynamic",
        "schema_content": schema_content,
        "schema_metadata": {}
    }
    
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    
    brand_options_list = []
    for m in matched_brands:
        if isinstance(m, dict):
            brand_id = m.get('id', 'N/A')
            brand_name = m.get('brand', 'Unknown')
            brand_options_list.append(f"ID {brand_id}: {brand_name}")
        else:
            brand_options_list.append(f"Invalid entry: {m}")
    
    brand_options = "\n".join(brand_options_list)
    
    input_text = f"""Product Information:
{name if name else 'N/A'}

Description:
{raw_text}

Available Brands:
{brand_options}

Select the most appropriate brand ID based on the product information."""

    if verbose:
        print(f"Classifying from {len(matched_brands)} brand option(s)...")

    input_data = {
        "item_id": "brand_classification",
        "text_input": input_text,
        "input_mode": "text-only",
        "format_as_product_info": False
    }
    
    agent_result = agent.execute(input_data=input_data)
    
    valid_brand_ids = set()
    id_to_brand_map = {}
    
    for match in matched_brands:
        if isinstance(match, dict):
            brand_id = match.get('id')
            brand_name = match.get('brand')
            if brand_id is not None and brand_name is not None:
                valid_brand_ids.add(brand_id)
                id_to_brand_map[brand_id] = brand_name
    
    if agent_result.status == AgentStatus.SUCCESS:
        result = agent_result.result
        prediction_id = result.get('prediction_id')
        
        validation_errors = []
        if prediction_id is None:
            validation_errors.append("prediction_id is missing")
        elif prediction_id not in valid_brand_ids:
            validation_errors.append(f"prediction_id {prediction_id} is not in valid BigQuery results")
        
        brand_name = ""
        if prediction_id in id_to_brand_map:
            brand_name = id_to_brand_map[prediction_id]
        elif prediction_id is not None:
            validation_errors.append(f"Brand name not found for ID {prediction_id}")
        
        scores = result.get('scores', [])
        confidence = scores[0]['score'] if scores else 0.0
        reasoning = result.get('reasoning', '')
        
        if validation_errors:
            if verbose:
                print(f"VALIDATION FAILED: {validation_errors}")
            return {
                "prediction_id": None,
                "brand_name": "",
                "confidence": 0.0,
                "reasoning": f"Validation failed: {'; '.join(validation_errors)}",
                "all_matched_brands": matched_brands,
                "validation_failed": True,
                "validation_errors": validation_errors
            }
        
        if verbose:
            print(f"Classified as: {brand_name} (ID: {prediction_id}, confidence: {confidence:.2f})")
        return {
            "prediction_id": prediction_id,
            "brand_name": brand_name,
            "confidence": confidence,
            "reasoning": reasoning,
            "all_matched_brands": matched_brands,
            "validation_passed": True
        }
    else:
        error_msg = agent_result.error_report.message if agent_result.error_report else "Unknown error"
        if verbose:
            print(f"ERROR: {error_msg}")
        return {
            "prediction_id": None,
            "brand_name": "",
            "confidence": 0.0,
            "reasoning": f"Classification error: {error_msg}",
            "all_matched_brands": matched_brands
        }


def run_brand_classification_workflow(
    raw_text: str,
    name: Optional[str] = None,
    known_synonyms: Optional[Dict[str, List[str]]] = None,
    env: str = 'staging',
    verbose: bool = False,
    config_loader: Optional[ConfigLoader] = None,
    agent1: Optional[LLMAnnotationAgent] = None,
    agent1_config: Optional[Dict[str, Any]] = None,
    agent2_base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run the complete two-agent brand classification workflow.

    Args:
        raw_text: Raw text input
        name: Optional product name
        known_synonyms: Optional brand synonyms dict
        env: Environment for config loading
        verbose: If True, print progress and diagnostics
        config_loader: Optional ConfigLoader (created if not provided)
        agent1: Pre-built Agent1 instance (thread-safe, can be shared)
        agent1_config: Pre-loaded Agent1 config (used if agent1 not provided)
        agent2_base_config: Pre-loaded Agent2 base config (schema injected per call)

    Returns:
        Complete workflow result
    """
    if verbose:
        print("="*70)
        print("BRAND CLASSIFICATION WORKFLOW")
        print("="*70)
        print(f"Input text: {raw_text[:100]}{'...' if len(raw_text) > 100 else ''}")
        if name:
            print(f"Product name: {name}")
        print("="*70)

    if config_loader is None and (agent1 is None and agent1_config is None):
        config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')

    agent1_result = run_agent1_brand_extraction(
        raw_text=raw_text,
        name=name,
        known_synonyms=known_synonyms,
        config_loader=config_loader,
        env=env,
        verbose=verbose,
        agent=agent1,
        full_config=agent1_config
    )

    if not agent1_result.get('success', False):
        error_msg = agent1_result.get('error', 'No brands found')
        if verbose:
            print(f"\nAgent 1 failed or found no brands: {error_msg}")
        return {
            "workflow_status": "failed",
            "agent1_result": agent1_result,
            "agent2_result": None,
            "final_brand": None,
            "final_brand_id": None,
            "confidence": 0.0,
            "error": error_msg
        }
    
    matched_brands = agent1_result.get('matched_brands', [])
    if not matched_brands:
        error_msg = agent1_result.get('error', 'No brands found in database')
        if verbose:
            print(f"\nNo matched brands from BigQuery: {error_msg}")
        return {
            "workflow_status": "failed",
            "agent1_result": agent1_result,
            "agent2_result": None,
            "final_brand": None,
            "final_brand_id": None,
            "confidence": 0.0,
            "error": error_msg
        }
    
    if verbose:
        print("\nProceeding to Agent 2 (brand classifier)...")
    agent2_result = run_agent2_brand_classification(
        raw_text=raw_text,
        name=name,
        matched_brands=matched_brands,
        config_loader=config_loader,
        env=env,
        verbose=verbose,
        base_config=agent2_base_config
    )

    if verbose:
        print("\n" + "="*70)
        print("WORKFLOW COMPLETE")
        print("="*70)
        print(f"Final Brand: {agent2_result.get('brand_name', 'N/A')}")
        print(f"Confidence: {agent2_result.get('confidence', 0.0):.2f}")
        print("="*70)
    
    return {
        "workflow_status": "success",
        "agent1_result": agent1_result,
        "agent2_result": agent2_result,
        "final_brand": agent2_result.get('brand_name'),
        "final_brand_id": agent2_result.get('prediction_id'),
        "confidence": agent2_result.get('confidence'),
        "error": None
    }


def run_brand_classification_bulk(
    df: pd.DataFrame,
    text_extraction_fn: Callable[[pd.Series, int], str],
    env: str = 'staging',
    max_workers: int = 200,
    batch_size: Optional[int] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> pd.DataFrame:
    """
    Run brand classification on all rows in parallel (bulk mode).

    OPTIMIZED: Loads configs once, reuses Agent1 across all rows.

    Args:
        df: DataFrame to process
        text_extraction_fn: (row, idx) -> text string for brand classification
        env: Environment for config loading
        max_workers: Max concurrent workflow calls
        batch_size: Process in batches of this size (None = all at once)
        verbose: If True, print progress from run_brand_classification_workflow
        progress_callback: Optional (idx, total) for progress updates

    Returns:
        DataFrame with TRUSS_brand_id, TRUSS_brand_name, TRUSS_brand_confidence, TRUSS_brand_reasoning
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Semaphore

    total = len(df)
    results = [None] * total
    semaphore = Semaphore(max_workers)

    print(f"[Brand Bulk] Initializing configs and Agent1 (one-time setup)...")
    config_loader = ConfigLoader(mode='dynamo', env=env, fallback_env='staging')
    
    try:
        agent1_config = config_loader.load_full_agent_config('brand-extraction-v1')
        agent1 = LLMAnnotationAgent(full_config=agent1_config, log_IO=False)
        print(f"[Brand Bulk] Agent1 initialized successfully")
    except Exception as e:
        print(f"[Brand Bulk] ERROR: Failed to load Agent1 config: {e}")
        return pd.DataFrame([{
            'TRUSS_brand_id': None,
            'TRUSS_brand_name': '',
            'TRUSS_brand_confidence': 0.0,
            'TRUSS_brand_reasoning': f'Config load error: {str(e)[:100]}'
        }] * total)
    
    try:
        agent2_base_config = config_loader.load_full_agent_config('brand-classification-v1')
        print(f"[Brand Bulk] Agent2 base config loaded successfully")
    except Exception as e:
        print(f"[Brand Bulk] ERROR: Failed to load Agent2 config: {e}")
        return pd.DataFrame([{
            'TRUSS_brand_id': None,
            'TRUSS_brand_name': '',
            'TRUSS_brand_confidence': 0.0,
            'TRUSS_brand_reasoning': f'Config load error: {str(e)[:100]}'
        }] * total)
    
    print(f"[Brand Bulk] Starting parallel processing of {total} rows with {max_workers} workers...")

    def process_row(idx: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
        with semaphore:
            try:
                text = text_extraction_fn(row, idx + 1)
                if not text or not str(text).strip():
                    return (idx, {
                        'TRUSS_brand_id': None,
                        'TRUSS_brand_name': '',
                        'TRUSS_brand_confidence': 0.0,
                        'TRUSS_brand_reasoning': 'No text data available - text-only classifier skipped'
                    })
                
                workflow_result = run_brand_classification_workflow(
                    raw_text=text,
                    name=None,
                    known_synonyms=None,
                    env=env,
                    verbose=verbose,
                    agent1=agent1,
                    agent2_base_config=agent2_base_config
                )
                
                if workflow_result.get('workflow_status') == 'success':
                    return (idx, {
                        'TRUSS_brand_id': workflow_result.get('final_brand_id'),
                        'TRUSS_brand_name': workflow_result.get('final_brand'),
                        'TRUSS_brand_confidence': workflow_result.get('confidence', 0.0),
                        'TRUSS_brand_reasoning': workflow_result.get('agent2_result', {}).get('reasoning', '')
                    })
                return (idx, {
                    'TRUSS_brand_id': None,
                    'TRUSS_brand_name': '',
                    'TRUSS_brand_confidence': 0.0,
                    'TRUSS_brand_reasoning': workflow_result.get('error', 'Brand classification failed')
                })
            except Exception as e:
                return (idx, {
                    'TRUSS_brand_id': None,
                    'TRUSS_brand_name': '',
                    'TRUSS_brand_confidence': 0.0,
                    'TRUSS_brand_reasoning': f'Error: {str(e)[:100]}'
                })

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

    print(f"[Brand Bulk] Completed processing {total} rows")
    return pd.DataFrame(results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Brand Classification Workflow - Two-agent brand extraction and classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python brand_classification_orchestration.py --text "Nike Air Max sneakers" --name "Nike Air Max 90"
    python brand_classification_orchestration.py --text "LV handbag" --name "Louis Vuitton Neverfull"
    python brand_classification_orchestration.py --text "Gucci leather bag" --env staging
        """
    )
    
    parser.add_argument('--text', type=str, required=True,
                       help='Raw text input to analyze')
    parser.add_argument('--name', type=str, default=None,
                       help='Optional product name')
    parser.add_argument('--env', type=str, default='staging',
                       choices=['dev', 'staging', 'prod'],
                       help='Environment for config loading (default: staging)')
    parser.add_argument('--synonyms', type=str, default=None,
                       help='Path to JSON file with brand synonyms dict')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print progress and diagnostics')

    args = parser.parse_args()

    # Load synonyms if provided
    known_synonyms = None
    if args.synonyms:
        with open(args.synonyms, 'r') as f:
            known_synonyms = json.load(f)

    try:
        result = run_brand_classification_workflow(
            raw_text=args.text,
            name=args.name,
            known_synonyms=known_synonyms,
            env=args.env,
            verbose=args.verbose
        )

        # Print JSON result (always when run from CLI)
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
