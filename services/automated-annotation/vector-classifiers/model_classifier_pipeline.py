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
    
    # Take top K matches (already ordered by score)
    top_k_matches = matches[:k]
    
    # Extract models and root_models for top K
    models = []
    root_models = []
    
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
        else:
            print(f"    {i}. ID: {image_id}, Score: {score:.4f}, Model: MISSING")
    
    # Check if we have any valid model data
    if not models:
        return {
            "predicted_model": None,
            "predicted_root_model": None,
            "confidence": 0.0,
            "method": "no_data",
            "message": "No valid model metadata found for top K results"
        }
    
    # Always vote on model
    model_counts = Counter([m for m in models if m])
    if not model_counts:
        return {
            "predicted_model": None,
            "predicted_root_model": None,
            "confidence": 0.0,
            "method": "no_data",
            "message": "No valid model metadata found for top K results"
        }
    
    most_common_model, model_vote_count = model_counts.most_common(1)[0]
    model_confidence = (model_vote_count / len(models)) * 100 if models else 0.0
    
    print(f"\n  Model voting results:")
    for model, count in model_counts.most_common():
        print(f"    {model}: {count}/{len(models)} ({(count/len(models))*100:.1f}%)")
    
    # If root_model is available, include the most common root_model in output
    predicted_root_model = None
    if root_models:
        root_model_counts = Counter(root_models)
        predicted_root_model = root_model_counts.most_common(1)[0][0]
        print(f"\n  Root model (for reference): {predicted_root_model}")
    
    message = f"Model '{most_common_model}' has {model_vote_count}/{len(models)} votes"
    
    result = {
        "predicted_model": most_common_model,
        "predicted_model_confidence": model_confidence,
        "predicted_root_model": predicted_root_model,
        "confidence": model_confidence,
        "method": "model_voting",
        "message": message,
    }
    
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

        return {
            **classification_result,
            "processing_id": processing_id,
            "brand": brand,
            "k": k,
            "vector_dimension": vector_dimension,
            "vector_source": "image-processing-table",
            "matches": matches,
            "metadata": {
                "vector": processing_record["metadata"],
                "pinecone": {
                    "namespace": namespace,
                    "index_name": "mfc-classifier-bags-models",
                },
                "model_metadata_count": len(metadata),
            },
        }

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {
            "processing_id": processing_id,
            "brand": brand,
            "k": k,
            "predicted_model": None,
            "predicted_root_model": None,
            "confidence": 0.0,
            "method": "error",
            "message": str(e),
            "vector_source": "image-processing-table",
        }


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