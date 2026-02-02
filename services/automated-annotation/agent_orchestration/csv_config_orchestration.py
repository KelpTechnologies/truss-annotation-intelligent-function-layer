"""
CSV Config Generation Orchestrator
===================================

Orchestrates the CSV config generator agent to analyze a CSV file
and produce a column mapping configuration for metadata extraction.

Usage:
    from csv_config_orchestration import orchestrate_csv_config_generation

    config = orchestrate_csv_config_generation(
        csv_path="vendor_linesheet.csv",
        output_path="configs/vendor_linesheet_config.json",  # Optional - auto-saves by default
        max_rows=10,
        max_chars=25000
    )
"""

import json
import pandas as pd
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Default output directory for generated configs
DEFAULT_CONFIG_OUTPUT_DIR = Path(__file__).parent / "bulk_editor_csv_configs"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_architecture.base_agent import LLMAnnotationAgent
from agent_architecture.validation import AgentResult, AgentStatus, ErrorReport
from agent_orchestration.csv_config_loader import ConfigLoader


def get_default_output_path(csv_path: str) -> Path:
    """
    Generate default output path based on input CSV filename.
    
    Args:
        csv_path: Path to input CSV file
        
    Returns:
        Path object for default output location
    """
    csv_name = Path(csv_path).stem  # e.g., "vendor_linesheet" from "vendor_linesheet.csv"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{csv_name}_config_{timestamp}.json"
    
    # Ensure output directory exists
    DEFAULT_CONFIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    return DEFAULT_CONFIG_OUTPUT_DIR / output_filename


def sample_csv_for_analysis(
    csv_path: str,
    max_rows: int = 10,
    max_chars: int = 25000
) -> Tuple[List[str], List[dict], int]:
    """
    Sample diverse rows from CSV within constraints.
    
    Sampling strategy:
    1. Load CSV with pandas
    2. Calculate null count per row
    3. Sort by null count (ascending - prefer complete rows)
    4. Take top max_rows * 2 as candidate pool
    5. Random sample max_rows from pool
    6. Convert to list of dicts
    7. Iterate rows, accumulating until max_chars exceeded
    
    Args:
        csv_path: Path to CSV file
        max_rows: Maximum rows to sample (default 10)
        max_chars: Maximum total characters in JSON payload (default 25000)
        
    Returns:
        Tuple of (columns: List[str], sample_rows: List[dict], total_chars: int)
    """
    # 1. Load CSV with pandas
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()
    
    # 2. Calculate null count per row
    df['_null_count'] = df.isnull().sum(axis=1)
    
    # 3. Sort by null count (ascending - prefer complete rows)
    df_sorted = df.sort_values('_null_count')
    
    # 4. Take top max_rows * 2 as candidate pool
    pool_size = min(len(df), max_rows * 2)
    candidate_pool = df_sorted.head(pool_size)
    
    # 5. Random sample max_rows from pool
    if len(candidate_pool) > max_rows:
        sampled = candidate_pool.sample(n=max_rows, random_state=42)
    else:
        sampled = candidate_pool
    
    # Remove helper column
    sampled = sampled.drop('_null_count', axis=1)
    
    # 6. Convert to list of dicts
    rows_as_dicts = sampled.to_dict('records')
    
    # 7. Iterate rows, accumulating until max_chars exceeded
    final_rows = []
    total_chars = 0
    for row in rows_as_dicts:
        row_json = json.dumps(row, default=str)
        if total_chars + len(row_json) > max_chars:
            break
        final_rows.append(row)
        total_chars += len(row_json)
    
    return columns, final_rows, total_chars


def format_csv_sample_for_prompt(
    columns: List[str],
    sample_rows: List[dict],
    total_chars: int
) -> str:
    """
    Format sampled data as text for LLM prompt.
    
    Format:
    **CSV Columns:**
    column1, column2, column3, ...
    
    **Sample Rows (N rows, M characters):**
    Row 1: {"column1": "value1", "column2": "value2", ...}
    Row 2: {"column1": "value1", "column2": "value2", ...}
    ...
    
    Args:
        columns: List[str] - Column names
        sample_rows: List[dict] - Row dicts
        total_chars: int - Char count for display
        
    Returns:
        Formatted string
    """
    # Format columns as comma-separated list
    columns_str = ", ".join(columns)
    
    # Format rows as readable JSON-like structure
    rows_formatted = []
    for i, row in enumerate(sample_rows, 1):
        # Clean nulls for readability
        clean_row = {k: v for k, v in row.items() if pd.notna(v)}
        rows_formatted.append(f"Row {i}: {json.dumps(clean_row, default=str)}")
    
    rows_str = "\n".join(rows_formatted)
    
    # Format according to specification
    return f"**CSV Columns:**\n{columns_str}\n\n**Sample Rows ({len(sample_rows)} rows, {total_chars} characters):**\n{rows_str}"


def orchestrate_csv_config_generation(
    csv_path: str,
    output_path: Optional[str] = None,
    max_rows: int = 10,
    max_chars: int = 25000,
    config_loader: Optional[ConfigLoader] = None,
    env: str = 'prod',
    mode: str = 'dynamo',
    auto_save: bool = True,
    organisation_uuid: Optional[str] = None
) -> dict:
    """
    Orchestrate CSV config generation.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save output config JSON (overrides auto_save location)
        max_rows: Maximum sample rows (default 10)
        max_chars: Maximum sample characters (default 25000)
        config_loader: Optional ConfigLoader instance
        env: Environment for config loading (default 'prod')
        mode: Config loader mode - 'api', 'dynamo', or 'local' (default 'dynamo')
        auto_save: If True and no output_path provided, auto-save to default directory (default True)
        
    Returns:
        Generated config dict mapping columns to metadata categories
        
    Raises:
        RuntimeError: If config generation fails
    """
    # Determine output path
    if output_path is None and auto_save:
        output_path = str(get_default_output_path(csv_path))
    
    print(f"{'='*70}")
    print(f"CSV CONFIG GENERATION")
    print(f"{'='*70}")
    print(f"Input: {csv_path}")
    if output_path:
        print(f"Output: {output_path}")
    print(f"Max rows: {max_rows}, Max chars: {max_chars}")
    
    # 1. Initialize config loader if not provided
    if config_loader is None:
        # Use the same env as fallback_env to ensure consistency
        config_loader = ConfigLoader(mode=mode, env=env, fallback_env=env)
    
    # 2. Sample CSV to get columns
    print(f"\nSampling CSV...")
    columns, sample_rows, total_chars = sample_csv_for_analysis(
        csv_path, max_rows, max_chars
    )
    print(f"  Columns: {len(columns)}")
    print(f"  Sample rows: {len(sample_rows)}")
    print(f"  Total chars: {total_chars}")
    
    # 3. Check for existing matching config in DynamoDB (if mode is dynamo)
    if mode == 'dynamo':
        print(f"\nChecking for existing configs in DynamoDB...")
        matching_config = config_loader.find_matching_csv_config(
            csv_columns=columns,
            organisation_uuid=organisation_uuid
        )
        if matching_config:
            print(f"Found matching config: {matching_config.get('csv_config_identifier')}")
            # Return the metadata mappings from the config
            return matching_config.get('csv_column_metadata_mappings', {})
    
    # 4. Load agent config
    print(f"\nLoading agent config: inventory_csv_config_generator")
    full_config = config_loader.load_full_agent_config('inventory_csv_config_generator')
    
    # 5. Format prompt input
    csv_sample_text = format_csv_sample_for_prompt(columns, sample_rows, total_chars)
    
    # 6. Initialize agent with full config
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)
    
    # 7. Call agent with execute()
    print(f"\nAnalyzing CSV structure...")
    input_data = {
        "input_text": csv_sample_text
    }
    context = {"csv_columns": columns}
    agent_result = agent.execute(input_data=input_data, context=context)
    
    # 8. Handle result
    if agent_result.status == AgentStatus.SUCCESS:
        config_output = agent_result.result
        print(f"\n✓ Config generated successfully")
        print(f"\nGenerated mapping:")
        for key, values in config_output.items():
            print(f"  {key}: {values}")
        
        # 9. Save to DynamoDB if mode is dynamo
        if mode == 'dynamo':
            csv_name = Path(csv_path).stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            config_identifier = f"{csv_name}_config_{timestamp}"
            
            try:
                config_loader.save_csv_config(
                    csv_config_identifier=config_identifier,
                    csv_columns=columns,
                    csv_column_metadata_mappings=config_output,
                    organisation_uuid=organisation_uuid
                )
                print(f"\nConfig saved to DynamoDB: {config_identifier}")
            except Exception as e:
                print(f"\nWarning: Failed to save config to DynamoDB: {e}")
        
        # 10. Save to local file if output path provided (for backward compatibility)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(config_output, f, indent=2)
            print(f"\nConfig saved to: {output_path}")
        
        return config_output
    
    else:
        error_msg = agent_result.error_report.message if agent_result.error_report else "Unknown error"
        print(f"\n✗ Config generation failed: {error_msg}")
        
        if agent_result.validation_info and agent_result.validation_info.errors:
            print(f"Validation errors:")
            for err in agent_result.validation_info.errors:
                print(f"  - {err.message}")
        
        raise RuntimeError(f"CSV config generation failed: {error_msg}")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Generate CSV column mapping configuration using LLM agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python csv_config_orchestration.py --csv-path data.csv
  python csv_config_orchestration.py --csv-path data.csv --output-path custom_config.json
  python csv_config_orchestration.py --csv-path data.csv --no-save
  python csv_config_orchestration.py --csv-path data.csv --env staging --max-chars 2000

Default output directory: {DEFAULT_CONFIG_OUTPUT_DIR}
        """
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Custom path to save output config JSON (overrides default directory)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save config to file (print to console only)'
    )
    
    parser.add_argument(
        '--max-rows',
        type=int,
        default=10,
        help='Maximum number of rows to sample (default: 10)'
    )
    
    parser.add_argument(
        '--max-chars',
        type=int,
        default=25000,
        help='Maximum total characters in sample JSON (default: 25000)'
    )
    
    parser.add_argument(
        '--env',
        type=str,
        default='prod',
        choices=['dev', 'staging', 'prod'],
        help='Environment for config loading (default: prod)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='dynamo',
        choices=['api', 'dynamo', 'local'],
        help='Config loader mode (default: dynamo)'
    )
    
    parser.add_argument(
        '--organisation-uuid',
        type=str,
        default=None,
        help='Optional organisation UUID to filter CSV configs by'
    )
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        # Determine auto_save based on --no-save flag
        auto_save = not args.no_save
        
        # Run orchestration
        config = orchestrate_csv_config_generation(
            csv_path=str(csv_path),
            output_path=args.output_path,
            max_rows=args.max_rows,
            max_chars=args.max_chars,
            config_loader=None,  # Will be created with specified mode/env
            env=args.env,
            mode=args.mode,
            auto_save=auto_save,
            organisation_uuid=args.organisation_uuid
        )
        
        print(f"\n{'='*70}")
        print("SUCCESS")
        print(f"{'='*70}")
        print(f"Generated configuration with {len(config)} metadata categories")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print(f"{'='*70}")
        print(f"Failed to generate CSV config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
