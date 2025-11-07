"""
Visual Classification Pipeline

This script:
1. Vectorizes an input image
2. Queries Pinecone for similar images
3. Queries DynamoDB for model metadata
4. Performs majority voting on model/root_model
5. Returns classification result with confidence

Dependencies:
    pip install boto3 pandas pillow requests
"""

import requests
from typing import Dict, Any, Optional
from PIL import Image
from io import BytesIO
import boto3
from collections import Counter
import sys

# Import from tds package
try:
    from tds import pinecone_utils
except ImportError:
    print("ERROR: Could not import pinecone_utils from tds package")
    print("Make sure the tds package is installed and available")
    sys.exit(1)


def vectorize_image(image_path: str) -> Dict[str, Any]:
    """
    Vectorize a local image using the GPU Image Vectorization API.
    
    Args:
        image_path: Path to local image file
        
    Returns:
        Dictionary with vector and metadata
    """
    import os
    
    print(f"\n[STEP 1] Vectorizing image: {image_path}")
    
    # GPU API endpoint
    api_url = "https://image-vectorization-api-gpu-94434742359.us-central1.run.app/vectorize"
    
    # Get filename
    filename = os.path.basename(image_path)
    
    # Load image using PIL
    print(f"  Loading image with PIL...")
    img = Image.open(image_path)
    
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        print(f"  Converting from RGBA to RGB...")
        img = img.convert('RGB')
    
    # Always use JPEG format
    img_format = 'JPEG'
    content_type = 'image/jpeg'
    
    # Save to buffer and extract bytes
    print(f"  Preparing image for upload...")
    img_buffer = BytesIO()
    img.save(img_buffer, format=img_format)
    img_buffer.seek(0)
    image_bytes = img_buffer.getvalue()
    
    # Upload to API
    print(f"  Calling vectorization API...")
    files = {
        'file': (filename, image_bytes, content_type)
    }
    
    response = requests.post(api_url, files=files, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    
    print(f"  ✓ Image vectorized successfully")
    print(f"    - Vector dimension: {result['dimension']}")
    print(f"    - Processing time: {result['timings']['total_ms']:.2f}ms")
    
    return result


def query_pinecone(vector: list, k: int, namespace: str, index_name: str = "mfc-classifier-bags-models") -> list:
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
    
    # Query using pinecone_utils
    query_response = pinecone_utils.query_nearest_neighbors(
        query_input=vector,
        k=k,
        namespace=namespace,
        index_name=index_name
    )
    
    matches = query_response.get('matches', [])
    
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


def classify_image(image_path: str, brand: str = "jacquemus", k: int = 5) -> Dict[str, Any]:
    """
    Complete classification pipeline for an image.
    
    Args:
        image_path: Path to image file
        brand: Brand name (used as Pinecone namespace, lowercase)
        k: Number of nearest neighbors to retrieve
        
    Returns:
        Classification result dictionary
    """
    print("="*70)
    print("VISUAL CLASSIFICATION PIPELINE")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Brand (namespace): {brand}")
    print(f"K (neighbors): {k}")
    
    try:
        # Step 1: Vectorize image
        vectorization_result = vectorize_image(image_path)
        vector = vectorization_result['vector']
        
        # Step 2: Query Pinecone
        namespace = brand.lower().strip()
        matches = query_pinecone(vector, k, namespace)
        
        if not matches:
            return {
                "predicted_model": None,
                "predicted_root_model": None,
                "confidence": 0.0,
                "method": "no_matches",
                "message": "No matches found in Pinecone"
            }
        
        # Step 3: Query DynamoDB for metadata
        ids = [match['id'] for match in matches]
        metadata = query_dynamodb(ids)
        
        # Step 4: Perform voting
        result = perform_voting(matches, metadata, k)
        
        # Print final result
        print(f"\n{'='*70}")
        print("CLASSIFICATION RESULT")
        print(f"{'='*70}")
        
        if result['predicted_model']:
            print(f"✓ Predicted Model: {result['predicted_model']}")
            print(f"  Confidence: {result['confidence']:.1f}%")
            print(f"  Method: {result['method']}")
        elif result['predicted_root_model']:
            print(f"✓ Predicted Root Model: {result['predicted_root_model']}")
            print(f"  Confidence: {result['confidence']:.1f}%")
            print(f"  Method: {result['method']}")
        else:
            print(f"✗ {result['message']}")
        
        print(f"{'='*70}\n")
        
        return result
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "predicted_model": None,
            "predicted_root_model": None,
            "confidence": 0.0,
            "method": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = "le_baneto_1.jpg"  # Change this to your image path
    BRAND = "jacquemus"            # Brand name (used as namespace)
    K = 5                          # Number of neighbors for voting
    
    # You can override these from command line
    import sys
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        K = int(sys.argv[2])
    
    # Run classification
    result = classify_image(IMAGE_PATH, BRAND, K)
    
    # Print result as JSON for easy parsing
    import json
    print("\nJSON Result:")
    print(json.dumps(result, indent=2))