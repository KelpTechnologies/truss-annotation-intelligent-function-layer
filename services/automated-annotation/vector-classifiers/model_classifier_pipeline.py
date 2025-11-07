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
from collections import Counter
from decimal import Decimal
from typing import Any, Dict, Optional

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
    Query DynamoDB for model and root_model for given IDs.
    
    Args:
        ids: List of image IDs
        table_name: DynamoDB table name (defaults to DYNAMODB_MODEL_TABLE env var or "model_visual_classifier_nodes")
        
    Returns:
        Dictionary mapping id -> {model, root_model}
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
                results[image_id] = {
                    'model': item.get('model', ''),
                    'root_model': item.get('root_model', '')
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
    Perform majority voting on model and root_model.
    
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
            root_model = metadata[image_id]['root_model']
            models.append(model)
            root_models.append(root_model)
            print(f"    {i}. ID: {image_id}, Score: {score:.4f}, Model: {model}, Root: {root_model}")
        else:
            print(f"    {i}. ID: {image_id}, Score: {score:.4f}, Model: MISSING")
    
    # Check if we have any valid data
    if not models:
        return {
            "predicted_model": None,
            "predicted_root_model": None,
            "confidence": 0.0,
            "method": "no_data",
            "message": "No valid metadata found for top K results"
        }
    
    # Count votes for models
    model_counts = Counter(models)
    most_common_model, model_vote_count = model_counts.most_common(1)[0]
    model_confidence = (model_vote_count / k) * 100
    
    print(f"\n  Model voting results:")
    for model, count in model_counts.most_common():
        print(f"    {model}: {count}/{k} ({(count/k)*100:.1f}%)")
    
    # Check if model has >=60% (>=3/5 for K=5)
    threshold_count = (k * 0.6)  # 60% threshold
    
    if model_vote_count >= threshold_count:
        print(f"\n  ✓ Model consensus found: {most_common_model} ({model_confidence:.1f}%)")
        return {
            "predicted_model": most_common_model,
            "predicted_root_model": None,
            "confidence": model_confidence,
            "method": "model_voting",
            "message": f"Model '{most_common_model}' has {model_vote_count}/{k} votes"
        }
    
    print(f"\n  ✗ No model consensus (best: {most_common_model} with {model_vote_count}/{k})")
    
    # Try root_model voting
    root_model_counts = Counter(root_models)
    most_common_root, root_vote_count = root_model_counts.most_common(1)[0]
    root_confidence = (root_vote_count / k) * 100
    
    print(f"\n  Root model voting results:")
    for root_model, count in root_model_counts.most_common():
        print(f"    {root_model}: {count}/{k} ({(count/k)*100:.1f}%)")
    
    if root_vote_count >= threshold_count:
        print(f"\n  ✓ Root model consensus found: {most_common_root} ({root_confidence:.1f}%)")
        return {
            "predicted_model": None,
            "predicted_root_model": most_common_root,
            "confidence": root_confidence,
            "method": "root_model_voting",
            "message": f"Root model '{most_common_root}' has {root_vote_count}/{k} votes"
        }
    
    print(f"\n  ✗ No root model consensus (best: {most_common_root} with {root_vote_count}/{k})")
    print(f"\n  ✗ No consensus found at 60% threshold")
    
    return {
        "predicted_model": None,
        "predicted_root_model": None,
        "confidence": 0.0,
        "method": "no_consensus",
        "message": "No consensus found - no model or root_model achieved 60% threshold"
    }


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
        namespace = brand.lower().strip()
        matches = query_pinecone(vector, k, namespace)

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

        if classification_result["predicted_model"]:
            print(f"✓ Predicted Model: {classification_result['predicted_model']}")
            print(f"  Confidence: {classification_result['confidence']:.1f}%")
            print(f"  Method: {classification_result['method']}")
        elif classification_result["predicted_root_model"]:
            print(f"✓ Predicted Root Model: {classification_result['predicted_root_model']}")
            print(f"  Confidence: {classification_result['confidence']:.1f}%")
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