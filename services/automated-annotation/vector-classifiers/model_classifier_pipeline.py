"""
Vector Classification Pipeline

This script:
1. Reads pre-computed image vectors from the image processing DynamoDB table
2. Queries Pinecone for visually similar images
3. Queries DynamoDB for model metadata
4. Performs majority voting on model/root_model
5. Returns classification result with confidence

Dependencies:
    pip install boto3 pinecone-client
"""

import os
import sys
import re
import unicodedata
import math
from collections import Counter
from decimal import Decimal
from typing import Any, Dict, Optional
def _normalize_matches(matches: list) -> list:
    """Convert Pinecone matches into plain dicts for downstream processing/JSON."""

    normalized = []
    for match in matches or []:
        if isinstance(match, dict):
            normalized.append(match)
            continue

        if hasattr(match, "to_dict"):
            normalized.append(match.to_dict())
            continue

        # Fall back to constructing a dict manually
        normalized.append(
            {
                "id": getattr(match, "id", None),
                "score": getattr(match, "score", None),
                "metadata": getattr(match, "metadata", None),
                "values": list(getattr(match, "values", []) or []),
            }
        )

    return normalized


import boto3

try:
    from . import pinecone_utils  # type: ignore
except ImportError:  #pragma: no cover - fallback when relative import unavailable
    import pinecone_utils  # type: ignore


def _decimal_to_float(value: Any) -> Any:
    """Recursively convert Decimal objects (from DynamoDB) into floats."""

    if isinstance(value, list):
        return [_decimal_to_float(v) for v in value]
    if isinstance(value, dict):
        return {k: _decimal_to_float(v) for k, v in value.items()}
    if isinstance(value, Decimal):
        return float(value)
    return value


def _get_image_processing_table_name(table_name: Optional[str] = None) -> str:
    """Resolve the DynamoDB table name that stores image processing results."""

    if table_name:
        return table_name

    env_table = os.getenv("IMAGE_PROCESSING_TABLE")
    if env_table:
        return env_table

    # Fallback to default naming convention if env not provided
    stage = os.getenv("STAGE", "dev")
    return f"truss-image-processing-{stage}"


def normalize_brand_to_namespace(brand: str) -> str:
    """
    Normalize brand name to Pinecone-compatible namespace format.

    Rules:
    - Lowercase
    - Only UTF-8 lowercase alphanumeric Latin characters and dashes
    - Map accented chars to non-accented ASCII (é -> e)
    - Replace spaces with "-"
    - Remove apostrophes

    Examples:
    - "Céline" -> "celine"
    - "Le Chiquito" -> "le-chiquito"
    - "Saint Laurent" -> "saint-laurent"
    - "D'ior" -> "dior"

    Args:
        brand: Brand name (may contain special characters)
        
    Returns:
        Normalized brand name suitable for Pinecone namespace
        
    Raises:
        ValueError: If brand name is empty or results in empty namespace after normalization
    """
    if not brand or not str(brand).strip():
        raise ValueError("Brand name cannot be empty")
    
    # Convert to string and strip whitespace
    brand_str = str(brand).strip()
    
    # Normalize Unicode characters (decompose accented chars)
    # NFKD = Normalization Form Compatibility Decomposition
    normalized = unicodedata.normalize('NFKD', brand_str)
    
    # Remove diacritics (accents) and convert to ASCII
    # This converts é -> e, ñ -> n, etc.
    ascii_brand = normalized.encode('ascii', 'ignore').decode('ascii')
    
    # Convert to lowercase
    ascii_brand = ascii_brand.lower()
    
    # Replace spaces with dashes
    ascii_brand = ascii_brand.replace(' ', '-')
    
    # Remove apostrophes (all types: straight, curly, etc.)
    apostrophes = ["'", "'", "'", '"', '"', '"']
    for apostrophe in apostrophes:
        ascii_brand = ascii_brand.replace(apostrophe, '')
    
    # Remove any characters that aren't lowercase alphanumeric or dashes
    # Keep only: a-z, 0-9, and -
    ascii_brand = re.sub(r'[^a-z0-9-]', '', ascii_brand)
    
    # Remove multiple consecutive dashes
    ascii_brand = re.sub(r'-+', '-', ascii_brand)
    
    # Remove leading/trailing dashes
    ascii_brand = ascii_brand.strip('-')
    
    if not ascii_brand:
        raise ValueError(f"Brand name '{brand}' resulted in empty namespace after normalization")
    
    return ascii_brand


def fetch_processing_record(processing_id: str, table_name: Optional[str] = None) -> Dict[str, Any]:
    """Fetch a processed image record (including vector) from DynamoDB."""

    if not processing_id:
        raise ValueError("processing_id is required")

    resolved_table = _get_image_processing_table_name(table_name)

    print(f"\n[STEP 1] Fetching vector from DynamoDB")
    print(f"  Table: {resolved_table}")
    print(f"  processing_id: {processing_id}")

    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(resolved_table)

    response = table.get_item(Key={"processingId": processing_id})
    item = response.get("Item")

    if not item:
        raise ValueError(f"No processing record found for processingId '{processing_id}'")

    if "imageVector" not in item or not item["imageVector"]:
        raise ValueError(
            "Processing record does not contain an image vector. "
            "Ensure the image was vectorized successfully before classification."
        )

    vector = _decimal_to_float(item["imageVector"])
    dimension = int(item.get("vectorDimension", len(vector)))

    vectorization_error = item.get("vectorizationError")
    if vectorization_error:
        raise ValueError(
            f"Processing record contains vectorization error: {vectorization_error}"
        )

    metadata = {
        "processedImage": item.get("processedImage"),
        "vectorizationTimings": _decimal_to_float(item.get("vectorizationTimings")),
        "timestamp": item.get("timestamp"),
        "stage": item.get("stage"),
    }

    print(f"  ✓ Retrieved vector with dimension {dimension}")

    return {
        "vector": vector,
        "dimension": dimension,
        "metadata": metadata,
        "raw_item": item,
    }


def query_pinecone(vector: list, k: int, namespace: str, index_name: str = pinecone_utils.DEFAULT_INDEX_NAME) -> list:
    """
    Query Pinecone for K nearest neighbors.
    
    Args:
        vector: Query vector
        k: Number of neighbors to retrieve
        namespace: Pinecone namespace (brand name)
        index_name: Pinecone index name
        
    Returns:
        List of match dictionaries with id and score
    """
    print(f"\n[STEP 2] Querying Pinecone for {k} nearest neighbors")
    print(f"  Index: {index_name}")
    print(f"  Namespace: {namespace}")
    
    matches = pinecone_utils.query_similar_vectors(
        vector=vector,
        top_k=k,
        namespace=namespace,
        index_name=index_name,
        include_metadata=True,
        include_values=False,
    )
    
    print(f"  ✓ Found {len(matches)} matches")
    
    # Print top 5 matches
    print(f"\n  Top matches:")
    for i, match in enumerate(matches[:5], 1):
        print(f"    {i}. ID: {match['id']}, Score: {match['score']:.4f}")
    
    return matches


def query_dynamodb(ids: list, table_name: str = None) -> Dict[str, Dict]:
    """
    Query DynamoDB for model and root_model metadata for given IDs.
    
    Args:
        ids: List of image IDs
        table_name: DynamoDB table name (defaults to DYNAMODB_MODEL_TABLE env var or "model_visual_classifier_nodes")
        
    Returns:
        Dictionary mapping id -> {model, root_model} (root_model may be None)
    """
    import os
    
    # Get table name from environment variable or use default
    if table_name is None:
        table_name = os.getenv("DYNAMODB_MODEL_TABLE", "model_visual_classifier_nodes")
    
    print(f"\n[STEP 3] Querying DynamoDB for model metadata")
    print(f"  Table: {table_name}")
    print(f"  Querying {len(ids)} IDs...")
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    results = {}
    
    for image_id in ids:
        try:
            response = table.get_item(Key={'id': str(image_id)})
            
            if 'Item' in response:
                item = response['Item']
                # Try both snake_case and camelCase field names
                model = item.get('model') or item.get('Model') or ''
                root_model = item.get('root_model') or item.get('rootModel') or item.get('RootModel') or None
                
                results[image_id] = {
                    'model': model,
                    'root_model': root_model
                }
            else:
                print(f"  ⚠️  ID {image_id} not found in DynamoDB")
                results[image_id] = {
                    'model': None,
                    'root_model': None
                }
        except Exception as e:
            print(f"  ✗ Error querying ID {image_id}: {e}")
            results[image_id] = {
                'model': None,
                'root_model': None
            }
    
    print(f"  ✓ Retrieved metadata for {len(results)} items")
    
    return results


def perform_voting(matches: list, metadata: Dict[str, Dict], k: int = 5) -> Dict[str, Any]:
    """
    Perform majority voting on model. root_model is included in output if available.
    
    Args:
        matches: List of Pinecone matches (ordered by similarity)
        metadata: Dictionary of id -> {model, root_model}
        k: Number of top results to use for voting
        
    Returns:
        Classification result dictionary
    """
    print(f"\n[STEP 4] Performing majority voting (K={k})")
    # Thresholds: model requires 4 votes, root_model requires 5 votes (was 5/6, then 3/4)
    required_model_votes = 4 if k >= 7 else math.ceil((4 / 7) * k)
    required_root_votes = 5 if k >= 7 else math.ceil((5 / 7) * k)
    
    # Take top K matches (already ordered by score)
    top_k_matches = matches[:k]
    
    # Extract models and root_models for top K, and build match details
    models = []
    root_models = []
    match_details = []
    
    print(f"\n  Top {k} results:")
    for i, match in enumerate(top_k_matches, 1):
        image_id = match['id']
        score = match['score']
        
        if image_id in metadata and metadata[image_id]['model']:
            model = metadata[image_id]['model']
            root_model = metadata[image_id].get('root_model')
            models.append(model)
            if root_model:
                root_models.append(root_model)
            print(f"    {i}. ID: {image_id}, Score: {score:.4f}, Model: {model}, Root: {root_model or 'N/A'}")
            match_details.append({
                "id": image_id,
                "score": round(score, 4),
                "model": model,
                "root_model": root_model
            })
        else:
            print(f"    {i}. ID: {image_id}, Score: {score:.4f}, Model: MISSING")
            match_details.append({
                "id": image_id,
                "score": round(score, 4),
                "model": None,
                "root_model": None
            })
    
    # Check if we have any valid model data
    if not models:
        return {
            "predicted_model": None,
            "predicted_root_model": None,
            "confidence": 0.0,
            "method": "no_data",
            "message": "No valid model metadata found for top K results",
            "metadata": {
                "match_details": match_details
            }
        }
    
    # Always vote on model
    model_counts = Counter([m for m in models if m])
    if not model_counts:
        return {
            "predicted_model": None,
            "predicted_root_model": None,
            "confidence": 0.0,
            "method": "no_data",
            "message": "No valid model metadata found for top K results",
            "metadata": {
                "match_details": match_details
            }
        }
    
    most_common_model, model_vote_count = model_counts.most_common(1)[0]
    model_confidence = (model_vote_count / k) * 100 if k else 0.0
    
    print(f"\n  Model voting results:")
    for model, count in model_counts.most_common():
        print(f"    {model}: {count}/{len(models)} ({(count/len(models))*100:.1f}%)")
    
    # If root_model is available, include the most common root_model in output
    predicted_root_model = None
    predicted_root_votes = 0
    root_model_counts = Counter(root_models) if root_models else Counter()
    if root_models:
        predicted_root_model, predicted_root_votes = root_model_counts.most_common(1)[0]
        root_confidence = (predicted_root_votes / k) * 100 if k else 0.0
        print(f"\n  Root model voting results:")
        for rm, count in root_model_counts.most_common():
            print(f"    {rm}: {count}/{k} ({(count/k)*100:.1f}%)")
        print(f"  Required root_model votes: {required_root_votes}/{k}")
        print(f"  Required model votes: {required_model_votes}/{k}")
    else:
        root_confidence = 0.0
        print("\n  Root model data missing for all results")
        print(f"  Required root_model votes: {required_root_votes}/{k}")
        print(f"  Required model votes: {required_model_votes}/{k}")

    if model_vote_count < required_model_votes:
        # Find the top model that didn't make it (second most common if exists)
        top_excluded_model = None
        top_excluded_model_votes = 0
        if len(model_counts) > 1:
            top_excluded_model, top_excluded_model_votes = model_counts.most_common(2)[1]
        
        return {
            "predicted_model": None,
            "predicted_root_model": None,
            "confidence": 0.0,
            "method": "insufficient_consensus",
            "message": (
                f"Model consensus not reached: {model_vote_count}/{k} votes "
                f"(requires {required_model_votes}/{k})"
            ),
            "metadata": {
                "match_details": match_details,
                "voting_details": {
                    "top_model": {
                        "name": most_common_model,
                        "votes": model_vote_count,
                        "required_votes": required_model_votes,
                        "confidence_percent": (model_vote_count / k) * 100 if k else 0.0,
                    },
                    "top_excluded_model": {
                        "name": top_excluded_model,
                        "votes": top_excluded_model_votes,
                        "confidence_percent": (top_excluded_model_votes / k) * 100 if k and top_excluded_model else 0.0,
                    } if top_excluded_model else None,
                    "all_model_votes": [
                        {
                            "model": model,
                            "votes": count,
                            "confidence_percent": (count / k) * 100 if k else 0.0,
                        }
                        for model, count in model_counts.most_common()
                    ],
                }
            },
        }

    if not root_models or predicted_root_votes < required_root_votes:
        # Find the top root_model that didn't make it (second most common if exists)
        top_excluded_root_model = None
        top_excluded_root_model_votes = 0
        if root_models and len(root_model_counts) > 1:
            top_excluded_root_model, top_excluded_root_model_votes = root_model_counts.most_common(2)[1]
        
        voting_metadata = {
            "match_details": match_details,
            "voting_details": {
                "top_model": {
                    "name": most_common_model,
                    "votes": model_vote_count,
                    "required_votes": required_model_votes,
                    "confidence_percent": (model_vote_count / k) * 100 if k else 0.0,
                },
                "top_root_model": {
                    "name": predicted_root_model,
                    "votes": predicted_root_votes,
                    "required_votes": required_root_votes,
                    "confidence_percent": (predicted_root_votes / k) * 100 if k else 0.0,
                } if root_models else None,
                "top_excluded_root_model": {
                    "name": top_excluded_root_model,
                    "votes": top_excluded_root_model_votes,
                    "confidence_percent": (top_excluded_root_model_votes / k) * 100 if k and top_excluded_root_model else 0.0,
                } if top_excluded_root_model else None,
                "all_model_votes": [
                    {
                        "model": model,
                        "votes": count,
                        "confidence_percent": (count / k) * 100 if k else 0.0,
                    }
                    for model, count in model_counts.most_common()
                ],
            }
        }
        
        if root_models:
            voting_metadata["voting_details"]["all_root_model_votes"] = [
                {
                    "root_model": rm,
                    "votes": count,
                    "confidence_percent": (count / k) * 100 if k else 0.0,
                }
                for rm, count in root_model_counts.most_common()
            ]
        
        return {
            "predicted_model": None,
            "predicted_root_model": None,
            "confidence": 0.0,
            "method": "insufficient_consensus",
            "message": (
                "Root model consensus not reached: "
                f"{predicted_root_votes}/{k} votes "
                f"(requires {required_root_votes}/{k})"
                if root_models
                else "Root model data unavailable for consensus check"
            ),
            "metadata": voting_metadata,
        }
    
    message = f"Model '{most_common_model}' has {model_vote_count}/{len(models)} votes"
    
    # Find the top excluded attributes for successful cases too (for completeness)
    top_excluded_model = None
    top_excluded_model_votes = 0
    if len(model_counts) > 1:
        top_excluded_model, top_excluded_model_votes = model_counts.most_common(2)[1]
    
    top_excluded_root_model = None
    top_excluded_root_model_votes = 0
    if root_models and len(root_model_counts) > 1:
        top_excluded_root_model, top_excluded_root_model_votes = root_model_counts.most_common(2)[1]
    
    result = {
        "predicted_model": most_common_model,
        "predicted_model_confidence": model_confidence,
        "predicted_root_model": predicted_root_model,
        "predicted_root_model_confidence": root_confidence,
        "confidence": min(model_confidence, root_confidence),
        "method": "threshold_voting",
        "message": message,
        "metadata": {
            "match_details": match_details,
            "voting_details": {
                "top_model": {
                    "name": most_common_model,
                    "votes": model_vote_count,
                    "required_votes": required_model_votes,
                    "confidence_percent": model_confidence,
                },
                "top_root_model": {
                    "name": predicted_root_model,
                    "votes": predicted_root_votes,
                    "required_votes": required_root_votes,
                    "confidence_percent": root_confidence,
                } if root_models else None,
                "top_excluded_model": {
                    "name": top_excluded_model,
                    "votes": top_excluded_model_votes,
                    "confidence_percent": (top_excluded_model_votes / k) * 100 if k and top_excluded_model else 0.0,
                } if top_excluded_model else None,
                "top_excluded_root_model": {
                    "name": top_excluded_root_model,
                    "votes": top_excluded_root_model_votes,
                    "confidence_percent": (top_excluded_root_model_votes / k) * 100 if k and top_excluded_root_model else 0.0,
                } if top_excluded_root_model else None,
                "all_model_votes": [
                    {
                        "model": model,
                        "votes": count,
                        "confidence_percent": (count / k) * 100 if k else 0.0,
                    }
                    for model, count in model_counts.most_common()
                ],
            }
        },
    }
    
    if root_models:
        result["metadata"]["voting_details"]["all_root_model_votes"] = [
            {
                "root_model": rm,
                "votes": count,
                "confidence_percent": (count / k) * 100 if k else 0.0,
            }
            for rm, count in root_model_counts.most_common()
        ]
    
    return result


def classify_image(processing_id: str, brand: str, k: int = 7) -> Dict[str, Any]:
    """
    Complete classification pipeline for a pre-processed image vector.

    Args:
        processing_id: ID of the processed image (from image processing pipeline)
        brand: Brand name (used as Pinecone namespace, lowercase)
        k: Number of nearest neighbors to retrieve

    Returns:
        Classification result dictionary with additional metadata
    """

    print("=" * 70)
    print("VISUAL CLASSIFICATION PIPELINE (Pre-computed vectors)")
    print("=" * 70)
    print(f"Processing ID: {processing_id}")
    print(f"Brand (namespace): {brand}")
    print(f"K (neighbors): {k}")

    if not brand:
        raise ValueError("'brand' is required for model classification")

    try:
        # Step 1: Fetch vector from DynamoDB
        processing_record = fetch_processing_record(processing_id)
        vector = processing_record["vector"]
        vector_dimension = processing_record["dimension"]

        # Step 2: Query Pinecone
        namespace = normalize_brand_to_namespace(brand)
        raw_matches = query_pinecone(vector, k, namespace)
        matches = _normalize_matches(raw_matches)

        if not matches:
            return {
                "processing_id": processing_id,
                "brand": brand,
                "k": k,
                "predicted_model": None,
                "predicted_root_model": None,
                "confidence": 0.0,
                "method": "no_matches",
                "message": "No matches found in Pinecone",
                "vector_dimension": vector_dimension,
                "vector_source": "image-processing-table",
                "matches": [],
                "metadata": processing_record["metadata"],
            }

        # Step 3: Query DynamoDB for metadata
        ids = [match["id"] for match in matches]
        metadata = query_dynamodb(ids)

        # Step 4: Perform voting
        classification_result = perform_voting(matches, metadata, k)

        # Print final result
        print(f"\n{'=' * 70}")
        print("CLASSIFICATION RESULT")
        print(f"{'=' * 70}")

        if classification_result.get("predicted_model"):
            print(f"✓ Predicted Model: {classification_result['predicted_model']}")
            print(f"  Confidence: {classification_result['confidence']:.1f}%")
            if classification_result.get("predicted_root_model"):
                print(f"  Root Model: {classification_result['predicted_root_model']}")
            print(f"  Method: {classification_result['method']}")
        else:
            print(f"✗ {classification_result['message']}")

        print(f"{'=' * 70}\n")

        # Merge metadata from classification_result with additional metadata
        # This preserves match_details and voting_details from perform_voting
        classification_metadata = classification_result.get("metadata", {})
        merged_metadata = {
            **classification_metadata,  # This includes match_details and voting_details
            "vector": processing_record["metadata"],
            "pinecone": {
                "namespace": namespace,
                "index_name": "mfc-classifier-bags-models",
            },
            "model_metadata_count": len(metadata),
        }
        
        # Ensure match_details is preserved (defensive check)
        if "match_details" not in merged_metadata and classification_metadata.get("match_details"):
            merged_metadata["match_details"] = classification_metadata["match_details"]
        
        # Debug: Log match_details count to verify it's being included
        match_details_count = len(merged_metadata.get("match_details", []))
        print(f"\n[DEBUG] Merged metadata includes {match_details_count} match_details entries")
        
        return {
            **classification_result,
            "processing_id": processing_id,
            "brand": brand,
            "k": k,
            "vector_dimension": vector_dimension,
            "vector_source": "image-processing-table",
            "matches": matches,
            "metadata": merged_metadata,
        }

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        
        # Re-raise the exception with a descriptive message so caller can handle appropriately
        error_message = str(e)
        if "AccessDeniedException" in error_message or "not authorized" in error_message:
            raise PermissionError(f"Vector classification failed - DynamoDB access denied: {error_message}") from e
        elif "No processing record found" in error_message:
            raise ValueError(f"Image not found for vector classification: {error_message}") from e
        elif "does not contain an image vector" in error_message:
            raise ValueError(f"Image not vectorized: {error_message}") from e
        else:
            raise RuntimeError(f"Vector classification failed: {error_message}") from e


if __name__ == "__main__":
    # Example usage:
    # python model_classifier_pipeline.py <processing_id> <brand> [k]

    if len(sys.argv) < 3:
        print(
            "Usage: python model_classifier_pipeline.py <processing_id> <brand> [k]"
        )
        sys.exit(1)

    processing_id_arg = sys.argv[1]
    brand_arg = sys.argv[2]
    k_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 7

    classification = classify_image(processing_id_arg, brand_arg, k_arg)

    import json

    print("\nJSON Result:")
    print(json.dumps(classification, indent=2))