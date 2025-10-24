#!/usr/bin/env python3
"""
Output Parser for LLM Classification Results (API-only mode)
==========================================================

Parses JSON classification results and creates formatted CSV outputs with
original data merged for easy review. Modified for API-only operation.
"""

import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import shutil
from typing import Dict, Tuple, Optional, Any, List
import re


def load_property_id_mapping_api(context_data: Dict[str, Any]) -> Dict[int, str]:
    """
    Load property ID to name mapping from API-loaded context data.

    Args:
        context_data: Context data dictionary from API

    Returns:
        Dictionary mapping property_id to property_name
    """
    id_mapping = {}

    if isinstance(context_data, dict) and 'context_content' in context_data:
        # API-loaded context data format
        content = context_data['context_content']
        for item_id, item_data in content.items():
            try:
                if isinstance(item_data, dict):
                    name = item_data.get('name', f'ID {item_id}')
                    id_mapping[int(item_id)] = name
                else:
                    # Legacy format
                    id_mapping[int(item_id)] = str(item_data)
            except (ValueError, TypeError):
                continue
    else:
        # Legacy format - try to extract from items list
        items = context_data.get("materials", context_data.get("items", []))
        for item in items:
            if isinstance(item, str) and item.startswith("ID "):
                try:
                    id_part = item.split(":")[0].strip()
                    id_num = int(id_part.split()[1])
                    name = item.split(":", 1)[1].strip() if ":" in item else f"ID {id_num}"
                    id_mapping[id_num] = name
                except (ValueError, IndexError):
                    continue

    return id_mapping


def validate_and_map_prediction(predicted_value: str, id_mapping: Dict[int, str], property_name: str = "material") -> Tuple[str, Optional[str]]:
    """
    Validate prediction against taxonomy and map ID to correct name.
    Returns (corrected_value, error_message)

    Args:
        predicted_value: Prediction string (e.g., "ID 2: Very Good")
        id_mapping: Dictionary mapping IDs to valid names
        property_name: Property being classified

    Returns:
        Tuple of (corrected_prediction, error_message)
        - If valid: (corrected_prediction, None)
        - If invalid: ("unknown", error_message)
    """
    if not predicted_value or pd.isna(predicted_value) or predicted_value == "unknown":
        return predicted_value, None

    predicted_value = str(predicted_value).strip()

    # Check if format is "ID X" or "ID X: ..."
    id_match = re.match(r'^ID\s+(\d+)', predicted_value, re.IGNORECASE)
    if id_match:
        pred_id = int(id_match.group(1))
        correct_name = id_mapping.get(pred_id)

        # Validate ID exists in taxonomy
        if not correct_name:
            error = f"INVALID_ID: Predicted ID {pred_id} not in {property_name} taxonomy (valid IDs: {list(id_mapping.keys())})"
            print(f"  [WARNING] {error}")
            return "unknown", error

        # Extract predicted name if present
        if ':' in predicted_value:
            pred_name = predicted_value.split(':', 1)[1].strip().lower()
            correct_name_lower = correct_name.lower()

            # Validate name matches taxonomy
            if pred_name != correct_name_lower:
                error = f"INVALID_NAME: Predicted 'ID {pred_id}: {pred_name}' but taxonomy has 'ID {pred_id}: {correct_name}'"
                print(f"  [WARNING] {error}")
                # Return corrected version
                return f"ID {pred_id}: {correct_name}", error

        # Return properly formatted prediction
        return f"ID {pred_id}: {correct_name}", None

    # No ID match - invalid format
    error = f"INVALID_FORMAT: Prediction '{predicted_value}' doesn't match 'ID X' format"
    print(f"  [WARNING] {error}")
    return "unknown", error


def map_id_to_name(predicted_value: str, id_mapping: Dict[int, str]) -> str:
    """
    Map material ID to name.
    """
    corrected, _ = validate_and_map_prediction(predicted_value, id_mapping, "material")
    return corrected


def extract_material_name(predicted_value: str) -> str:
    """
    Extract material name from prediction string.

    Examples:
        "ID 1: leather" -> "leather"
        "leather" -> "leather"
        "ID 10: patent leather" -> "patent leather"
    """
    if not predicted_value or pd.isna(predicted_value):
        return ""

    predicted_value = str(predicted_value).strip()

    # Check if format is "ID X: material_name"
    if ":" in predicted_value:
        # Extract everything after the colon
        material_name = predicted_value.split(":", 1)[1].strip()
    else:
        material_name = predicted_value

    return material_name.lower().strip()


def calculate_accuracy_metrics(df_results: pd.DataFrame, property_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate accuracy metrics for classification results.
    Only considers rows where the LLM made a prediction (not "unknown").

    Args:
        df_results: DataFrame with classification results including ground truth
        property_name: Name of the property being classified (e.g., 'material')

    Returns:
        Tuple of (df_results with accuracy column added, summary metrics dict)
    """
    # Column names
    predicted_col = f'predicted_{property_name}'
    extracted_col = f'extracted_{property_name}'

    # Check if ground truth column exists
    if extracted_col not in df_results.columns:
        print(f"Warning: Ground truth column '{extracted_col}' not found. Skipping accuracy calculation.")
        return df_results, {}

    # Filter out rows where ground truth is missing
    df_with_ground_truth = df_results[df_results[extracted_col].notna() & (df_results[extracted_col] != '')].copy()

    if len(df_with_ground_truth) == 0:
        print("Warning: No valid ground truth data found. Skipping accuracy calculation.")
        return df_results, {}

    # Filter to only include rows where LLM made a prediction (not "unknown")
    df_with_ground_truth['predicted_name_temp'] = df_with_ground_truth[predicted_col].apply(extract_material_name)
    df_valid = df_with_ground_truth[df_with_ground_truth['predicted_name_temp'] != 'unknown'].copy()

    total_items_with_ground_truth = len(df_with_ground_truth)
    total_proposed = len(df_valid)
    llm_classification_rate = total_proposed / total_items_with_ground_truth if total_items_with_ground_truth > 0 else 0.0

    if total_proposed == 0:
        print(f"Warning: LLM did not make predictions for any items (all {total_items_with_ground_truth} items returned 'unknown').")
        return df_results, {
            'overall': {
                'total_items': total_items_with_ground_truth,
                'total_proposed': 0,
                'llm_classification_rate': 0.0,
                'total_predictions': 0,
                'correct_predictions': 0,
                'overall_accuracy': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'macro_f1': 0.0,
                'weighted_precision': 0.0,
                'weighted_recall': 0.0,
                'weighted_f1': 0.0
            }
        }

    print(f"\nCalculating accuracy metrics for {total_proposed} items where LLM made predictions (out of {total_items_with_ground_truth} total items with ground truth)...")
    print(f"LLM Classification Rate: {llm_classification_rate:.2%}")

    # Extract material names from predictions
    df_valid['predicted_name'] = df_valid[predicted_col].apply(extract_material_name)
    df_valid['actual_name'] = df_valid[extracted_col].apply(lambda x: str(x).lower().strip() if pd.notna(x) else "")

    # Calculate accuracy
    df_valid['prediction_accurate'] = df_valid['predicted_name'] == df_valid['actual_name']

    # Add accuracy column back to original dataframe
    df_results['prediction_accurate'] = False
    df_results.loc[df_valid.index, 'prediction_accurate'] = df_valid['prediction_accurate']

    # Calculate per-material metrics
    material_metrics = []

    # Get unique materials from both predicted and actual
    all_materials = set(df_valid['predicted_name'].unique()) | set(df_valid['actual_name'].unique())
    all_materials.discard('')  # Remove empty strings

    for material in sorted(all_materials):
        # True Positives: Predicted this material AND actually is this material
        tp = len(df_valid[(df_valid['predicted_name'] == material) & (df_valid['actual_name'] == material)])

        # False Positives: Predicted this material BUT actually is NOT this material
        fp = len(df_valid[(df_valid['predicted_name'] == material) & (df_valid['actual_name'] != material)])

        # False Negatives: Did NOT predict this material BUT actually IS this material
        fn = len(df_valid[(df_valid['predicted_name'] != material) & (df_valid['actual_name'] == material)])

        # True Negatives: Did NOT predict this material AND actually is NOT this material
        tn = len(df_valid[(df_valid['predicted_name'] != material) & (df_valid['actual_name'] != material)])

        # Calculate metrics
        predicted_count = tp + fp
        actual_count = tp + fn
        correct_predictions = tp

        # Precision: Of all predictions for this material, how many were correct?
        precision = tp / predicted_count if predicted_count > 0 else 0.0

        # Recall: Of all actual instances of this material, how many did we find?
        recall = tp / actual_count if actual_count > 0 else 0.0

        # Accuracy for this material: (TP + TN) / Total
        accuracy = (tp + tn) / len(df_valid) if len(df_valid) > 0 else 0.0

        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        material_metrics.append({
            'material': material,
            'predicted_count': predicted_count,
            'actual_count': actual_count,
            'correct_predictions': correct_predictions,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'accuracy': round(accuracy, 4)
        })

    df_material_metrics = pd.DataFrame(material_metrics)

    # Calculate overall metrics
    total_correct = df_valid['prediction_accurate'].sum()
    total_predictions = len(df_valid)
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0

    # Macro-averaged precision and recall (average across all materials)
    macro_precision = df_material_metrics['precision'].mean() if len(df_material_metrics) > 0 else 0.0
    macro_recall = df_material_metrics['recall'].mean() if len(df_material_metrics) > 0 else 0.0
    macro_f1 = df_material_metrics['f1_score'].mean() if len(df_material_metrics) > 0 else 0.0

    # Weighted-averaged metrics (weighted by actual_count)
    if df_material_metrics['actual_count'].sum() > 0:
        weighted_precision = (df_material_metrics['precision'] * df_material_metrics['actual_count']).sum() / df_material_metrics['actual_count'].sum()
        weighted_recall = (df_material_metrics['recall'] * df_material_metrics['actual_count']).sum() / df_material_metrics['actual_count'].sum()
        weighted_f1 = (df_material_metrics['f1_score'] * df_material_metrics['actual_count']).sum() / df_material_metrics['actual_count'].sum()
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    overall_metrics = {
        'total_items': total_items_with_ground_truth,
        'total_proposed': total_proposed,
        'llm_classification_rate': round(llm_classification_rate, 4),
        'total_predictions': total_predictions,
        'correct_predictions': int(total_correct),
        'overall_accuracy': round(overall_accuracy, 4),
        'macro_precision': round(macro_precision, 4),
        'macro_recall': round(macro_recall, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_precision': round(weighted_precision, 4),
        'weighted_recall': round(weighted_recall, 4),
        'weighted_f1': round(weighted_f1, 4)
    }

    return df_results, {'per_material': df_material_metrics, 'overall': overall_metrics}


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str, id_mapping: Dict[int, str] = None):
    """
    Save classification results to CSV with proper formatting.

    Args:
        results: List of classification result dictionaries
        output_file: Path to save CSV file
        id_mapping: Optional ID to name mapping for formatting
    """
    if not results:
        print("No results to save")
        return

    df_results = pd.DataFrame(results)

    # Apply ID to name mapping if provided
    if id_mapping and 'primary' in df_results.columns:
        df_results['primary'] = df_results['primary'].apply(lambda x: map_id_to_name(x, id_mapping) if x else x)

    # Format alternatives column
    if 'alternatives' in df_results.columns:
        df_results['alternatives'] = df_results['alternatives'].apply(
            lambda x: '; '.join([map_id_to_name(alt, id_mapping) if id_mapping and alt else alt for alt in x] if isinstance(x, list) else [])
        )

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
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description='Parse LLM Classification Results (API-only mode)')
    parser.add_argument('--json-file', required=True, help='Path to JSON results file')
    parser.add_argument('--csv-file', required=True, help='Path to original CSV test data')
    parser.add_argument('--property', required=True, help='Property that was classified (e.g., material)')
    parser.add_argument('--output-dir', default='.', help='Base directory for outputs')
    parser.add_argument('--timestamp', help='Timestamp for file naming (auto-generated if not provided)')
    parser.add_argument('--calculate-accuracy', action='store_true', default=False,
                       help='Calculate accuracy metrics using ground truth from original CSV')

    args = parser.parse_args()

    try:
        # Generate timestamp if not provided
        timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load JSON results
        print(f"Loading JSON results from: {args.json_file}")
        with open(args.json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Load original CSV data
        print(f"Loading CSV data from: {args.csv_file}")
        df_original = pd.read_csv(args.csv_file)

        # Convert garment_id to string for consistent matching
        df_original['garment_id'] = df_original['garment_id'].astype(str)

        # Create results DataFrame
        results_data = []
        for result in results:
            garment_id = str(result['garment_id'])

            # Find matching row in original data
            original_row = df_original[df_original['garment_id'] == garment_id]

            if not original_row.empty:
                row = original_row.iloc[0]

                # Build the result row
                result_row = {
                    'garment_id': garment_id,
                    'excel_image_link': row.get('excel_image_link', ''),
                    'brand': row.get('brand', ''),
                    'extracted_model': row.get('extracted_model', ''),
                    f'predicted_{args.property}': result.get('primary', ''),
                    'confidence': result.get('confidence', ''),
                    'alternative_materials': '; '.join(result.get('alternatives', [])),
                    'processing_time_seconds': result.get('processing_time_seconds', ''),
                    'success': result.get('success', True),
                    'title': row.get('title', ''),
                    'description': row.get('description', '')
                }

                # Add ground truth column if it exists (for accuracy calculation)
                extracted_col = f'extracted_{args.property}'
                if extracted_col in row.index:
                    result_row[extracted_col] = row.get(extracted_col, '')

                results_data.append(result_row)
            else:
                print(f"Warning: No matching data found for garment_id {garment_id}")

        # Create results DataFrame
        df_results = pd.DataFrame(results_data)

        # Calculate accuracy metrics if requested
        metrics = {}
        if args.calculate_accuracy:
            df_results, metrics = calculate_accuracy_metrics(df_results, args.property)

        # Create output directories
        visual_results_dir = Path(args.output_dir) / "visual_results"
        summary_results_dir = Path(args.output_dir) / "summary_results"
        visual_results_dir.mkdir(exist_ok=True)
        summary_results_dir.mkdir(exist_ok=True)

        # Generate output filenames
        base_name = f"classification_results_{args.property}_{timestamp}"
        csv_output = visual_results_dir / f"{base_name}.csv"

        # Save CSV results
        df_results.to_csv(csv_output, index=False, encoding='utf-8')
        print(f"Visual CSV results saved to: {csv_output}")

        # Save summary metrics if calculated
        if metrics and 'per_material' in metrics:
            summary_csv = summary_results_dir / f"summary_{args.property}_{timestamp}.csv"
            metrics['per_material'].to_csv(summary_csv, index=False, encoding='utf-8')
            print(f"Summary metrics saved to: {summary_csv}")

            # Save overall metrics to a separate file
            overall_csv = summary_results_dir / f"overall_metrics_{args.property}_{timestamp}.csv"
            df_overall = pd.DataFrame([metrics['overall']])
            df_overall.to_csv(overall_csv, index=False, encoding='utf-8')
            print(f"Overall metrics saved to: {overall_csv}")

        # Print summary
        successful = len(df_results[df_results['success'] == True])
        total = len(df_results)
        print(f"\nSummary:")
        print(f"Total processed: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")

        if successful > 0:
            avg_confidence = df_results[df_results['success'] == True]['confidence'].mean()
            avg_time = df_results['processing_time_seconds'].mean()
            print(f"Average confidence: {avg_confidence:.2f}")
            print(f"Average processing time: {avg_time:.2f}s")

        # Print accuracy metrics if calculated
        if metrics and 'overall' in metrics:
            print(f"\nAccuracy Metrics:")
            print(f"Total Items with Ground Truth: {metrics['overall']['total_items']}")
            print(f"Total Proposed by LLM: {metrics['overall']['total_proposed']}")
            print(f"LLM Classification Rate: {metrics['overall']['llm_classification_rate']:.2%}")
            print(f"---")
            print(f"Overall Accuracy (of proposed): {metrics['overall']['overall_accuracy']:.2%}")
            print(f"Macro Precision: {metrics['overall']['macro_precision']:.2%}")
            print(f"Macro Recall: {metrics['overall']['macro_recall']:.2%}")
            print(f"Macro F1: {metrics['overall']['macro_f1']:.2%}")
            print(f"Weighted Precision: {metrics['overall']['weighted_precision']:.2%}")
            print(f"Weighted Recall: {metrics['overall']['weighted_recall']:.2%}")
            print(f"Weighted F1: {metrics['overall']['weighted_f1']:.2%}")

        print(f"\nProcessing completed successfully!")
        print(f"Visual results available at: {csv_output}")

    except Exception as e:
        print(f"Error processing results: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
