import json
import os
import traceback
import base64
import logging
import requests
import time
from datetime import datetime
import boto3

# Import DSL components from simplified package
from dsl import DSLAPIClient, ConfigLoader, LLMAnnotationAgent
from dsl import load_property_id_mapping_api

# Import structured logger for metrics (schema v2 compliant)
from structured_logger import StructuredLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize structured logger for metrics (REQUEST/RESPONSE/ERROR lifecycle events)
structured_logger = StructuredLogger(layer="aifl", service_name="automated-annotation")

CATEGORY_CONFIG = {
    "bags": {
        "root_type_id": 30,
        "default_brand": None,
        "properties": {
            "model": "model",
            "material": "material",
            "colour": "colour",
            "type": "type",
        },
    }
}

# Import vector classifier pipeline
try:
    import importlib.util
    import sys
    
    pipeline_path = os.path.join(
        os.path.dirname(__file__), 
        "vector-classifiers", 
        "model_classifier_pipeline.py"
    )
    
    if os.path.exists(pipeline_path):
        package_dir = os.path.dirname(pipeline_path)
        if package_dir not in sys.path:
            sys.path.insert(0, package_dir)

        spec = importlib.util.spec_from_file_location("model_classifier_pipeline", pipeline_path)
        pipeline_module = importlib.util.module_from_spec(spec)
        sys.modules["model_classifier_pipeline"] = pipeline_module
        spec.loader.exec_module(pipeline_module)
        classify_image = pipeline_module.classify_image
    else:
        classify_image = None
except Exception as e:
    logger.warning(f"Could not load vector classifier pipeline: {e}")
    classify_image = None


def _cors_headers(methods: str = "POST,GET,OPTIONS"):
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": methods,
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
    }


def _response(status_code: int, body: dict, methods: str = "POST,GET,OPTIONS"):
    return {
        "statusCode": status_code,
        "headers": _cors_headers(methods),
        "body": json.dumps(body),
    }


def _parse_body(event):
    logger.debug("Parsing request body")
    if not event.get("body"):
        logger.debug("No body found in event")
        return {}
    try:
        body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        logger.debug(f"Successfully parsed body: {json.dumps(body, default=str)[:500]}")  # Limit log size
        return body
    except Exception as e:
        logger.error(f"Failed to parse body: {str(e)}")
        return {}


def _build_text_metadata(payload: dict):
    if payload.get("text_metadata"):
        return payload["text_metadata"]
    parts = []
    for key in ["brand", "title", "description"]:
        value = payload.get(key)
        if value and str(value).strip():
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {value}")
    return "\n".join(parts) if parts else None


def _load_gcp_credentials_from_secrets():
    """Load GCP service account JSON from AWS Secrets Manager (centralized truss-platform-secrets)."""
    # Support both new TRUSS_SECRETS_ARN and legacy BIGQUERY_SECRET_ARN
    secret_arn = os.getenv("TRUSS_SECRETS_ARN") or os.getenv("BIGQUERY_SECRET_ARN")
    if not secret_arn:
        raise ValueError("TRUSS_SECRETS_ARN environment variable is required")
    
    logger.info(f"Loading GCP credentials from Secrets Manager: {secret_arn}")
    secrets_client = boto3.client('secretsmanager', region_name=os.getenv('AWS_REGION', 'eu-west-2'))
    
    response = secrets_client.get_secret_value(SecretId=secret_arn)
    secret_string = response.get('SecretString')
    
    if not secret_string:
        raise ValueError(f"Secret string is empty for secret {secret_arn}")
    
    # Parse as JSON to validate and extract
    try:
        secret_data = json.loads(secret_string)
        logger.debug(f"Secret parsed as JSON, keys: {list(secret_data.keys()) if isinstance(secret_data, dict) else 'not a dict'}")
        
        # If it's a dict, try to extract the service account JSON
        if isinstance(secret_data, dict):
            # NEW: Check for centralized 'bigquery' key (truss-platform-secrets structure)
            if 'bigquery' in secret_data:
                logger.info("Found BigQuery credentials under 'bigquery' key (centralized secrets)")
                bigquery_creds = secret_data['bigquery']
                # Build complete service account JSON
                service_account = {
                    "type": "service_account",
                    "project_id": bigquery_creds.get('project_id'),
                    "private_key_id": bigquery_creds.get('private_key_id'),
                    "private_key": bigquery_creds.get('private_key'),
                    "client_email": bigquery_creds.get('client_email'),
                    "client_id": bigquery_creds.get('client_id'),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{bigquery_creds.get('client_email')}"
                }
                return json.dumps(service_account)
            
            # Check if it's already a service account JSON (has 'type' and 'project_id')
            if secret_data.get('type') == 'service_account' and 'project_id' in secret_data:
                logger.info("Secret contains full service account JSON")
                # Return as JSON string (the whole dict is the service account)
                return json.dumps(secret_data)
            
            # Otherwise, try to find nested service account JSON
            for key in ['service_account_json', 'credentials', 'value', 'gcp_service_account']:
                if key in secret_data:
                    logger.info(f"Found nested service account JSON under key: {key}")
                    nested_value = secret_data[key]
                    # If it's a string, return as-is; if it's a dict, convert to JSON string
                    if isinstance(nested_value, str):
                        return nested_value
                    elif isinstance(nested_value, dict):
                        return json.dumps(nested_value)
                    break
            
            # If no nested key found and it's not a service account, use the whole dict
            logger.info("Using entire secret as service account JSON")
            return json.dumps(secret_data)
        else:
            # Not a dict, return as-is (should be JSON string already)
            logger.info("Secret is not a dict, returning as-is")
            return secret_string
    except json.JSONDecodeError as e:
        raise ValueError(f"Secret is not valid JSON: {str(e)}")


# Cache for centralized secrets
_cached_platform_secrets = None

def _get_platform_secrets():
    """Get all platform secrets from centralized truss-platform-secrets."""
    global _cached_platform_secrets
    if _cached_platform_secrets:
        return _cached_platform_secrets
    
    secret_arn = os.getenv("TRUSS_SECRETS_ARN") or os.getenv("BIGQUERY_SECRET_ARN")
    if not secret_arn:
        raise ValueError("TRUSS_SECRETS_ARN environment variable is required")
    
    logger.info(f"Loading platform secrets from: {secret_arn}")
    secrets_client = boto3.client('secretsmanager', region_name=os.getenv('AWS_REGION', 'eu-west-2'))
    
    response = secrets_client.get_secret_value(SecretId=secret_arn)
    secret_string = response.get('SecretString')
    
    if not secret_string:
        raise ValueError(f"Secret string is empty for secret {secret_arn}")
    
    _cached_platform_secrets = json.loads(secret_string)
    return _cached_platform_secrets


def _ensure_pinecone_api_key():
    """Ensure PINECONE_API_KEY environment variable is set from centralized secrets."""
    if os.getenv("PINECONE_API_KEY"):
        logger.info("PINECONE_API_KEY already set in environment")
        return
    
    logger.info("Loading Pinecone API key from centralized secrets")
    try:
        secrets = _get_platform_secrets()
        pinecone_key = secrets.get('pinecone', {}).get('api_key')
        if pinecone_key:
            os.environ["PINECONE_API_KEY"] = pinecone_key
            logger.info("PINECONE_API_KEY set from centralized secrets")
        else:
            logger.warning("Pinecone API key not found in centralized secrets")
    except Exception as e:
        logger.error(f"Failed to load Pinecone API key from secrets: {str(e)}")


def _ensure_gcp_adc():
    """Ensure Application Default Credentials are available from env or Secrets Manager."""
    logger.info("Setting up GCP Application Default Credentials")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    
    # Check if credentials file already exists and is valid
    if os.path.exists(creds_path) and os.path.getsize(creds_path) > 0:
        # Verify the existing file is actually valid JSON with required fields
        try:
            with open(creds_path, 'r') as f:
                existing_creds = json.load(f)
                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                if all(field in existing_creds for field in required_fields):
                    logger.info(f"GCP credentials file already exists and is valid at {creds_path} (size: {os.path.getsize(creds_path)} bytes)")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                    return
                else:
                    logger.warning(f"Existing credentials file at {creds_path} is missing required fields, will recreate")
        except Exception as e:
            logger.warning(f"Existing credentials file at {creds_path} is invalid ({str(e)}), will recreate")
        # If we get here, the file exists but is invalid, so we'll recreate it
    
    # Try to get from environment variable first
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    
    # If not in env, load from Secrets Manager
    if not sa_json:
        logger.info("GCP_SERVICE_ACCOUNT_JSON not in environment, loading from Secrets Manager")
        sa_json = _load_gcp_credentials_from_secrets()
    
    if not sa_json:
        raise ValueError("GCP credentials not available - cannot proceed with LLM classification")

    logger.info(f"Setting up GCP credentials at {creds_path}")
    logger.debug(f"GCP credentials length: {len(sa_json)} chars")
    
    # Parse JSON content
    if sa_json.strip().startswith("{"):
        content = sa_json
        logger.debug("GCP credentials are JSON format")
    else:
        try:
            content = base64.b64decode(sa_json).decode("utf-8")
            logger.debug("GCP credentials decoded from base64")
        except Exception as e:
            raise ValueError(f"Failed to decode base64 credentials: {str(e)}")

    # Validate JSON content and ensure private key has proper newlines
    try:
        creds_dict = json.loads(content)
        logger.debug("Credentials JSON is valid")
        
        # Ensure private key has actual newlines (not escaped \n)
        if 'private_key' in creds_dict:
            private_key = creds_dict['private_key']
            # Replace escaped newlines with actual newlines if needed
            if '\\n' in private_key and '\n' not in private_key:
                logger.debug("Converting escaped newlines in private key to actual newlines")
                creds_dict['private_key'] = private_key.replace('\\n', '\n')
                content = json.dumps(creds_dict)
                logger.debug("Private key newlines fixed")
            elif '\n' in private_key:
                logger.debug("Private key already has proper newlines")
        
        # Re-validate after potential modification
        json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"GCP credentials are not valid JSON: {str(e)}")

    # Ensure directory exists
    creds_dir = os.path.dirname(creds_path)
    os.makedirs(creds_dir, exist_ok=True)
    logger.debug(f"Created credentials directory: {creds_dir}")

    # Write credentials file with proper permissions (readable by owner only for security)
    with open(creds_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    # Set file permissions to 600 (read/write for owner only)
    os.chmod(creds_path, 0o600)
    
    # Verify file was written
    if not os.path.exists(creds_path):
        raise IOError(f"Failed to create credentials file at {creds_path}")
    
    file_size = os.path.getsize(creds_path)
    if file_size == 0:
        raise IOError(f"Credentials file is empty at {creds_path}")
    
    logger.info(f"GCP credentials written successfully to {creds_path} (size: {file_size} bytes, permissions: {oct(os.stat(creds_path).st_mode)[-3:]})")

    # Parse credentials to extract service account email and project
    try:
        creds_data = json.loads(content)
        service_account_email = creds_data.get("client_email", "unknown")
        logger.info(f"Using GCP service account: {service_account_email}")
    except Exception as e:
        logger.warning(f"Could not parse service account email from credentials: {str(e)}")
        service_account_email = "unknown"

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS={creds_path}")

    # Set project - try to get from env, then from credentials JSON, then fail
    vertexai_project = os.getenv("VERTEXAI_PROJECT")
    
    if not vertexai_project:
        # Try to extract project_id from the credentials JSON
        try:
            if 'creds_data' not in locals():
                creds_data = json.loads(content)
            vertexai_project = creds_data.get("project_id")
            if vertexai_project:
                logger.info(f"Extracted project_id from GCP credentials: {vertexai_project}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not extract project_id from credentials: {str(e)}")
    
    if not vertexai_project:
        raise ValueError("VERTEXAI_PROJECT environment variable is required (or project_id must be in GCP service account JSON)")
    
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        os.environ["GOOGLE_CLOUD_PROJECT"] = vertexai_project
        logger.info(f"Set GOOGLE_CLOUD_PROJECT={vertexai_project}")
    else:
        logger.debug(f"GOOGLE_CLOUD_PROJECT already set to {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    
    # Also set VERTEXAI_PROJECT if not already set
    if not os.getenv("VERTEXAI_PROJECT"):
        os.environ["VERTEXAI_PROJECT"] = vertexai_project
        logger.info(f"Set VERTEXAI_PROJECT={vertexai_project}")
    
    # Log important info for debugging IAM issues
    logger.info(f"GCP Configuration - Project: {vertexai_project}, Service Account: {service_account_email}")
    logger.warning(f"IMPORTANT: Service account '{service_account_email}' needs 'Vertex AI User' role (roles/aiplatform.user) in project '{vertexai_project}' to use Gemini models")
    
    # Verify credentials can be read by Google auth libraries
    try:
        from google.auth import default
        from google.auth.transport.requests import Request
        creds, project = default()
        logger.info(f"Successfully loaded credentials via google.auth.default() - Project: {project}")
        logger.info(f"Credentials type: {type(creds).__name__}")
        if hasattr(creds, 'service_account_email'):
            logger.info(f"Service account from credentials: {creds.service_account_email}")
    except Exception as e:
        logger.error(f"Failed to verify credentials with google.auth.default(): {str(e)}")
        logger.warning("This may indicate a problem with credential setup")
        
    logger.info("GCP Application Default Credentials setup completed successfully")


# Global context for request auth headers (set per-request in lambda_handler)
_request_auth_headers = {}

def _get_signed_image_url(image: str) -> str:
    """Get signed image URL from annotation-data-service-layer image-service."""
    global _request_auth_headers
    start_time = time.time()
    logger.info(f"Fetching signed image URL for image ID: {image}")
    
    api_base_url = os.getenv("ANNOTATION_DSL_API_BASE_URL")
    api_key = os.getenv("ANNOTATION_API_KEY")
    
    if not api_base_url:
        logger.error("ANNOTATION_DSL_API_BASE_URL environment variable is not set")
        raise ValueError("ANNOTATION_DSL_API_BASE_URL environment variable is required")
    
    url = f"{api_base_url.rstrip('/')}/images/processed/{image}"
    logger.debug(f"Image service URL: {url}")
    
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    # First try to use pass-through auth from original request
    if _request_auth_headers.get('x-api-key'):
        headers['x-api-key'] = _request_auth_headers['x-api-key']
        logger.debug("Using pass-through x-api-key from original request")
    elif _request_auth_headers.get('Authorization'):
        headers['Authorization'] = _request_auth_headers['Authorization']
        logger.debug("Using pass-through Authorization from original request")
    elif api_key:
        # Fall back to configured API key
        headers['x-api-key'] = api_key
        logger.debug("Using configured ANNOTATION_API_KEY")
    else:
        logger.warning("No authentication available for image service request")
        
    try:
        logger.debug(f"Making GET request to image service for image: {image}")
        response = requests.get(url, headers=headers, timeout=10)
        elapsed_time = time.time() - start_time
        logger.info(f"Image service response received in {elapsed_time:.2f}s - Status: {response.status_code}")
        
        response.raise_for_status()
        
        result = response.json()
        logger.debug(f"Image service response: {json.dumps(result, default=str)[:500]}")
        
        data = result.get('data', result)
        processed_image = data.get('processedImage', {})
        download_url = processed_image.get('downloadUrl')
        
        if not download_url:
            logger.error(f"No downloadUrl found in image service response for image '{image}'. Response: {json.dumps(result, default=str)[:500]}")
            raise ValueError(f"No downloadUrl found in image service response for image '{image}'")
        
        logger.info(f"Successfully retrieved signed image URL for image {image} in {elapsed_time:.2f}s")
        return download_url
    except requests.exceptions.RequestException as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Failed to fetch image URL for {image} after {elapsed_time:.2f}s: {str(e)}")
        raise


def _lookup_root_from_child(api_client: DSLAPIClient, property_type: str, value: str, brand: str = None, root_type: str = None, partition: str = "bags") -> dict:
    """Lookup root taxonomy value from child value using the knowledge service API."""
    start_time = time.time()
    logger.info(f"Looking up root for {property_type}='{value}' (brand={brand}, root_type={root_type}, partition={partition})")
    
    try:
        lookup_result = api_client.lookup_root(
            property_type=property_type,
            value=value,
            brand=brand,
            root_type=root_type,
            partition=partition
        )
        elapsed_time = time.time() - start_time
        logger.debug(f"Lookup API response received in {elapsed_time:.2f}s: {json.dumps(lookup_result, default=str)[:500]}")
        
        data = lookup_result.get('data', lookup_result)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
            logger.debug(f"Extracted first item from list response")
        elif not isinstance(data, dict):
            data = lookup_result
            logger.debug("Using lookup_result directly as data")
            
        if not data.get('found'):
            logger.info(f"No root found for {property_type}='{value}' (lookup completed in {elapsed_time:.2f}s)")
            return None
            
        result = {
            'root': data.get('root'),
            'root_id': data.get('root_id'),
            'child_id': data.get(f'{property_type}_id'),
        }
        
        if result['root'] and result['root_id']:
            logger.info(f"Found root: {result['root']} (ID: {result['root_id']}) for {property_type}='{value}' in {elapsed_time:.2f}s")
            return result
        else:
            logger.warning(f"Incomplete root lookup result for {property_type}='{value}': {result}")
            return None
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error during root lookup for {property_type}='{value}' after {elapsed_time:.2f}s: {str(e)}")
        raise


def _classify_model(payload: dict):
    """Classify model using a pre-computed image vector from the image-processing table."""
    start_time = time.time()
    logger.info(f"Starting model classification - payload: {json.dumps(payload, default=str)}")
    
    # Ensure Pinecone API key is available before classification
    _ensure_pinecone_api_key()
    
    if not classify_image:
        logger.error("Vector classifier pipeline not available")
        raise ValueError("Vector classifier pipeline not available")

    processing_id = payload.get("processing_id") or payload.get("processingId")
    if not processing_id:
        logger.error("Missing processing_id in payload")
        raise ValueError("'processing_id' is required for model classification")

    brand = payload.get("brand")
    if not brand:
        logger.error("Missing brand in payload")
        raise ValueError("'brand' is required for model classification")

    logger.info(f"Calling vector classifier for processing_id={processing_id}, brand={brand}")
    vector_start = time.time()
    result = classify_image(
        processing_id=processing_id,
        brand=brand,
        k=7,
    )
    vector_elapsed = time.time() - vector_start
    logger.info(f"Vector classification completed in {vector_elapsed:.2f}s")

    predicted_model = result.get("predicted_model")
    predicted_root_model = result.get("predicted_root_model")
    confidence = result.get("confidence", 0.0)
    
    logger.info(f"Vector classifier result - model: {predicted_model}, root_model: {predicted_root_model}, confidence: {confidence}")

    # Always lookup root from knowledge service
    root_lookup_result = None
    if predicted_model:
        logger.info(f"Looking up root for predicted model: {predicted_model}")
        api_base_url = os.getenv("DSL_API_BASE_URL")
        api_key = os.getenv("DSL_API_KEY")
        if not api_base_url:
            logger.error("DSL_API_BASE_URL not set")
            raise ValueError("DSL_API_BASE_URL environment variable is required")
        
        api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key, auth_headers=_request_auth_headers)
        category = payload.get("category", "bags")
        partition = category
        root_type = "Bags" if category == "bags" else None
        
        root_lookup_result = _lookup_root_from_child(
            api_client=api_client,
            property_type="model",
            value=predicted_model,
            brand=brand,
            root_type=root_type,
            partition=partition
        )
        
        # Use lookup result if available, otherwise use predicted_root_model
        root_model = root_lookup_result.get('root') if root_lookup_result else predicted_root_model
        root_model_id = root_lookup_result.get('root_id') if root_lookup_result else None
        model_id = root_lookup_result.get('child_id') if root_lookup_result else None
        
        if root_lookup_result:
            logger.info(f"Root lookup successful - root_model: {root_model}, root_model_id: {root_model_id}, model_id: {model_id}")
        else:
            logger.warning(f"Root lookup failed, using predicted_root_model: {predicted_root_model}")
    else:
        logger.warning("No predicted model from vector classifier, skipping root lookup")
        root_model = None
        root_model_id = None
        model_id = None

    total_elapsed = time.time() - start_time
    final_result = {
        "model": predicted_model,
        "model_id": model_id,
        "root_model": root_model,
        "root_model_id": root_model_id,
        "confidence": confidence,
    }
    logger.info(f"Model classification completed in {total_elapsed:.2f}s - Result: {json.dumps(final_result, default=str)}")
    return final_result


def _classify_property(category: str, target: str, request_payload: dict):
    """Classify a property using LLM."""
    start_time = time.time()
    logger.info(f"Starting property classification - category: {category}, target: {target}, payload keys: {list(request_payload.keys())}")
    
    category_config = CATEGORY_CONFIG.get(category)
    if not category_config:
        logger.error(f"Unsupported category: {category}")
        raise ValueError(f"Unsupported category '{category}'")

    property_map = category_config.get("properties", {})
    internal_property_name = property_map.get(target)
    if not internal_property_name:
        logger.error(f"Unsupported target '{target}' for category '{category}'")
        raise ValueError(f"Unsupported classification target '{target}' for category '{category}'")

    logger.info(f"Mapped target '{target}' to internal property '{internal_property_name}'")

    # Get image and lookup signed URL
    image = request_payload.get("image")
    if not image:
        logger.error("Missing image in request payload")
        raise ValueError("'image' is required for property classification")
    
    logger.info(f"Fetching signed URL for image: {image}")
    image_url = _get_signed_image_url(image)
    logger.info(f"Retrieved image URL (length: {len(image_url)} chars)")

    text_dump = request_payload.get("text_dump")
    if not text_dump:
        logger.error("Missing text_dump in request payload")
        raise ValueError("'text_dump' is required for property classification")

    logger.debug(f"Text dump length: {len(text_dump)} chars")

    classification_payload = {
        "property": internal_property_name,
        "root_type_id": category_config["root_type_id"],
        "image_url": image_url,
        "text_metadata": text_dump,
        "description": request_payload.get("description") or text_dump,
        "title": request_payload.get("title"),
        "brand": request_payload.get("brand"),
        "input_mode": request_payload.get("input_mode", "auto"),
        "resolve_names": request_payload.get("resolve_names", False),
        "garment_id": image,
    }
    
    logger.info(f"Calling LLM classifier for property '{internal_property_name}' (root_type_id: {category_config['root_type_id']})")
    raw_result = _classify_item(classification_payload)
    
    logger.info(f"LLM classification completed - primary: {raw_result.get('primary')}, confidence: {raw_result.get('confidence', 0.0)}")

    primary_value = raw_result.get("primary")
    if isinstance(primary_value, str) and primary_value.startswith("ID "):
        try:
            original_value = primary_value
            primary_value = primary_value.split(":", 1)[1].strip()
            logger.debug(f"Extracted value from ID format: '{original_value}' -> '{primary_value}'")
        except Exception as e:
            logger.warning(f"Failed to parse ID format value '{primary_value}': {str(e)}")

    # For color classifications, replace "Beige" with "Neutrals" as a safeguard
    if target and target.lower() in ['color', 'colour'] and isinstance(primary_value, str):
        if primary_value.lower() == 'beige':
            primary_value = 'Neutrals'
            logger.info(f"Replaced 'Beige' with 'Neutrals' for color classification in _classify_property")

    # Lookup root for model and material properties
    root_lookup_result = None
    if target in ["model", "material"] and primary_value:
        logger.info(f"Looking up root for {target}='{primary_value}'")
        api_base_url = os.getenv("DSL_API_BASE_URL")
        api_key = os.getenv("DSL_API_KEY")
        if not api_base_url:
            logger.error("DSL_API_BASE_URL not set")
            raise ValueError("DSL_API_BASE_URL environment variable is required")
        
        api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key, auth_headers=_request_auth_headers)
        partition = category
        root_type = "Bags" if category == "bags" else None
        brand = request_payload.get("brand") if target == "model" else None
        
        root_lookup_result = _lookup_root_from_child(
            api_client=api_client,
            property_type=target,
            value=primary_value,
            brand=brand,
            root_type=root_type,
            partition=partition
        )
        
        if root_lookup_result:
            logger.info(f"Root lookup successful for {target}: {root_lookup_result}")
        else:
            logger.warning(f"Root lookup returned None for {target}='{primary_value}'")

    # Build simplified response
    total_elapsed = time.time() - start_time
    if target == "model":
        result = {
            "model": primary_value,
            "model_id": root_lookup_result.get('child_id') if root_lookup_result else None,
            "root_model": root_lookup_result.get('root') if root_lookup_result else None,
            "root_model_id": root_lookup_result.get('root_id') if root_lookup_result else None,
            "confidence": raw_result.get("confidence", 0.0),
        }
    elif target == "material":
        result = {
            "material": primary_value,
            "material_id": root_lookup_result.get('child_id') if root_lookup_result else None,
            "root_material": root_lookup_result.get('root') if root_lookup_result else None,
            "root_material_id": root_lookup_result.get('root_id') if root_lookup_result else None,
            "confidence": raw_result.get("confidence", 0.0),
        }
    else:
        result = {
            target: primary_value,
            "confidence": raw_result.get("confidence", 0.0),
        }
    
    logger.info(f"Property classification completed in {total_elapsed:.2f}s - Result: {json.dumps(result, default=str)}")
    return result


def _classify_item(payload: dict):
    """Classify an item using LLM."""
    start_time = time.time()
    logger.info(f"Starting LLM classification - property: {payload.get('property')}, root_type_id: {payload.get('root_type_id')}")
    
    # Use ANNOTATION_DSL_API_BASE_URL for classifier configs, templates, and context data
    api_base_url = os.getenv("ANNOTATION_DSL_API_BASE_URL")
    api_key = os.getenv("ANNOTATION_API_KEY")
    if not api_base_url:
        logger.error("ANNOTATION_DSL_API_BASE_URL environment variable is not set")
        raise ValueError("ANNOTATION_DSL_API_BASE_URL environment variable is required")
    
    logger.debug(f"Using API base URL: {api_base_url}")

    property_name = payload.get("property")
    root_type_id = payload.get("root_type_id")
    if not property_name or root_type_id is None:
        logger.error(f"Missing required fields - property: {property_name}, root_type_id: {root_type_id}")
        raise ValueError("'property' and 'root_type_id' are required fields")

    input_mode = payload.get("input_mode", "auto")
    image_url = payload.get("image_url")
    garment_id = str(payload.get("garment_id") or "")
    brand = payload.get("brand")
    text_metadata = _build_text_metadata(payload)
    
    logger.info(f"Classification parameters - property: {property_name}, root_type_id: {root_type_id}, input_mode: {input_mode}, brand: {brand}")
    logger.debug(f"Text metadata length: {len(text_metadata) if text_metadata else 0} chars")

    # Ensure GCP Application Default Credentials are set up (required for Vertex AI)
    logger.info("Ensuring GCP Application Default Credentials")
    _ensure_gcp_adc()
    
    # Verify credentials are set up and readable
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    
    if not os.path.exists(creds_path):
        raise ValueError(f"GCP credentials file not found at {creds_path}")
    
    # Verify file is readable
    if not os.access(creds_path, os.R_OK):
        raise ValueError(f"GCP credentials file is not readable: {creds_path}")
    
    # Log credentials file details for debugging
    file_stat = os.stat(creds_path)
    logger.info(f"GCP credentials verified at {creds_path} (size: {file_stat.st_size} bytes, readable: {os.access(creds_path, os.R_OK)})")
    
    # Verify credentials file contains valid JSON with required fields
    try:
        with open(creds_path, 'r') as f:
            creds_check = json.load(f)
            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in creds_check]
            if missing_fields:
                logger.error(f"Credentials file missing required fields: {missing_fields}")
                raise ValueError(f"Credentials file missing required fields: {missing_fields}")
            
            # Verify private key format
            private_key = creds_check.get('private_key', '')
            if not private_key.startswith('-----BEGIN'):
                logger.warning("Private key may not be properly formatted")
            if '\\n' in private_key and '\n' not in private_key:
                logger.warning("Private key contains escaped newlines - may need conversion")
            
            logger.info(f"Credentials file validated - contains all required fields")
            logger.info(f"Service account: {creds_check.get('client_email')}, Project: {creds_check.get('project_id')}")
            logger.debug(f"Private key length: {len(private_key)} chars, starts with BEGIN: {private_key.startswith('-----BEGIN')}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse credentials file as JSON: {str(e)}")
        raise ValueError(f"Credentials file is not valid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to validate credentials file: {str(e)}")
        raise ValueError(f"Credentials file validation failed: {str(e)}")

    api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key, auth_headers=_request_auth_headers)
    config_loader = ConfigLoader(mode='api', api_client=api_client)

    logger.info(f"Loading classifier config for property '{property_name}', root_type_id {root_type_id}")
    config_start = time.time()
    api_response = config_loader.load_classifier_config(property_name, int(root_type_id))
    config_elapsed = time.time() - config_start
    logger.info(f"Classifier config loaded in {config_elapsed:.2f}s")
    
    config = api_response.get('data', api_response)
    logger.debug(f"Classifier config: {json.dumps(config, default=str)[:500]}")

    template_id = config.get('prompt_template')
    if not template_id:
        logger.error(f"No prompt_template in config: {json.dumps(config, default=str)}")
        raise ValueError("No prompt_template specified in configuration")
    
    logger.info(f"Loading prompt template: {template_id}")
    template_start = time.time()
    template = config_loader.load_prompt_template(template_id)
    template_elapsed = time.time() - template_start
    logger.info(f"Prompt template loaded in {template_elapsed:.2f}s")

    logger.info(f"Loading context data for property '{property_name}', root_type_id {root_type_id}")
    context_start = time.time()
    context_data = config_loader.load_context_data(property_name, int(root_type_id))
    context_elapsed = time.time() - context_start
    logger.info(f"Context data loaded in {context_elapsed:.2f}s")
    
    context_size = len(json.dumps(context_data, default=str)) if context_data else 0
    logger.debug(f"Context data size: {context_size} chars")

    model_name = config.get('model')
    if not model_name:
        raise ValueError("Model name not specified in classifier config")
    
    # Get project_id - try env var, then config, then extract from GCP credentials
    project_id = os.getenv('VERTEXAI_PROJECT')
    if not project_id:
        project_id = config.get('project_id')
    if not project_id:
        # Try to extract from GCP credentials file
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            try:
                with open(creds_path, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get('project_id')
                    if project_id:
                        logger.info(f"Extracted project_id from GCP credentials: {project_id}")
            except Exception as e:
                logger.warning(f"Could not read project_id from credentials file: {str(e)}")
    
    if not project_id:
        raise ValueError("VERTEXAI_PROJECT environment variable, project_id in config, or project_id in GCP credentials is required")
    
    # Get location - prioritize config from API, then env var, then use default (like old system)
    location = config.get('location')
    if not location:
        location = os.getenv('VERTEXAI_LOCATION')
    if not location:
        location = 'us-central1'  # Default VertexAI location (like old system)
        logger.info(f"Using default VertexAI location: {location}")
    else:
        logger.info(f"Using VertexAI location: {location}")
    
    logger.info(f"Initializing LLM classifier - model: {model_name}, project: {project_id}, location: {location}")
    classifier = LLMAnnotationAgent(
        model_name=model_name,
        project_id=project_id,
        location=location,
        prompt_template_path=None,
        context_mode=config.get('default_context_mode', 'full-context'),
        log_IO=False,
        config={
            'temperature': config.get('temperature', 0.1),
            'max_output_tokens': config.get('max_output_tokens', 1024),
        },
    )

    classifier.prompt_template = template
    logger.info(f"Calling LLM classifier with image_url (length: {len(image_url) if image_url else 0}), text_metadata: {bool(text_metadata)}")
    
    classify_start = time.time()
    result = classifier.classify(
        image_url=image_url,
        text_metadata=text_metadata,
        property_type=property_name,
        garment_id=garment_id,
        root_type_id=int(root_type_id),
        brand=brand,
        input_mode=input_mode,
        context_data=context_data,
    )
    classify_elapsed = time.time() - classify_start
    logger.info(f"LLM classification completed in {classify_elapsed:.2f}s - prediction_id: {result.prediction_id}, primary: {result.primary}, confidence: {result.confidence}")

    resolve_names = bool(payload.get("resolve_names", False))
    primary = result.primary
    alternatives = result.alternatives
    
    logger.debug(f"Raw result - primary: {primary}, alternatives: {alternatives}, confidence: {result.confidence}")

    if resolve_names:
        logger.info("Resolving names from ID mapping")
        id_map = load_property_id_mapping_api(context_data.get('data', context_data))

        def _fmt(val: str):
            try:
                if val.startswith("ID "):
                    pred_id = int(val.split("ID ")[1])
                    name = id_map.get(pred_id)
                    if name:
                        # For color classifications, replace "Beige" with "Neutrals"
                        if property_name and property_name.lower() in ['color', 'colour']:
                            if name.lower() == 'beige':
                                name = 'Neutrals'
                                logger.info(f"Replaced 'Beige' with 'Neutrals' for color classification")
                        return f"ID {pred_id}: {name}"
                    return val
            except Exception as e:
                logger.warning(f"Failed to format value '{val}': {str(e)}")
                return val
            return val

        primary = _fmt(primary)
        alternatives = [_fmt(a) for a in alternatives]
        logger.debug(f"Resolved names - primary: {primary}, alternatives: {alternatives}")

    total_elapsed = time.time() - start_time
    response = {
        "garment_id": garment_id,
        "property": property_name,
        "root_type_id": int(root_type_id),
        "prediction_id": result.prediction_id,
        "primary": primary,
        "alternatives": alternatives,
        "confidence": result.confidence,
        "reasoning": getattr(result, 'reasoning', None),
        "input_mode_used": input_mode,
        "has_text_metadata": bool(text_metadata),
        "image_url": image_url,
        "success": True,
    }
    
    logger.info(f"LLM classification completed in {total_elapsed:.2f}s - Result: {json.dumps(response, default=str)[:500]}")
    return response


def lambda_handler(event, context):
    global _request_auth_headers
    request_start_time = time.time()
    request_id = context.aws_request_id if context else "unknown"
    
    # Start structured logging for metrics (captures request timing)
    req_ctx = structured_logger.start_request(event)
    
    logger.info("=" * 80)
    logger.info(f"Lambda invocation started - Request ID: {request_id}")
    logger.info(f"Event method: {event.get('httpMethod', 'UNKNOWN')}, path: {event.get('path', 'UNKNOWN')}")
    logger.debug(f"Full event: {json.dumps(event, default=str)[:1000]}")
    
    # Extract auth headers from original request for pass-through to downstream services
    incoming_headers = event.get("headers") or {}
    _request_auth_headers = {}
    # Check both lowercase and mixed-case header names (API Gateway normalizes differently)
    for key in incoming_headers:
        key_lower = key.lower()
        if key_lower == 'authorization':
            _request_auth_headers['Authorization'] = incoming_headers[key]
            logger.debug("Captured Authorization header for pass-through")
        elif key_lower == 'x-api-key':
            _request_auth_headers['x-api-key'] = incoming_headers[key]
            logger.debug("Captured x-api-key header for pass-through")
    
    try:
        method = event.get("httpMethod", "GET")
        path = (event.get("path") or "").rstrip("/")
        
        # Strip custom domain base path prefix if present (e.g., /agents from api.trussarchive.io/agents/...)
        if path.startswith("/agents/"):
            path = path[7:]  # Remove "/agents" prefix, keep leading "/"
            logger.info(f"Stripped /agents prefix from path")
        
        logger.info(f"Processing {method} request to {path}")

        if method == "OPTIONS":
            logger.info("Handling OPTIONS request")
            response = _response(200, {"ok": True})
            structured_logger.log_response(req_ctx, status_code=200)
            return response

        segments = [segment for segment in path.strip("/").split("/") if segment]
        logger.debug(f"Path segments: {segments}")

        if (
            len(segments) >= 5
            and segments[0] == "automations"
            and segments[1] == "annotation"
            and segments[3] == "classify"
            and method == "POST"
        ):
            category = segments[2]
            target = segments[4]
            
            logger.info(f"Classification request - category: {category}, target: {target}")

            payload = _parse_body(event)
            logger.info(f"Parsed payload keys: {list(payload.keys())}")

            if category not in CATEGORY_CONFIG:
                logger.error(f"Unsupported category: {category}")
                response = _response(404, {"error": f"Unsupported category '{category}'"})
                structured_logger.log_response(req_ctx, status_code=404)
                return response

            try:
                if target == "model":
                    logger.info("Processing model classification request")
                    image = payload.get("image")
                    if not image:
                        logger.error("Missing image in model classification request")
                        raise ValueError("'image' is required for model classification")

                    brand = payload.get("brand") or CATEGORY_CONFIG[category].get("default_brand")
                    if not brand:
                        logger.error("Missing brand in model classification request")
                        raise ValueError("'brand' is required for model classification")

                    logger.info(f"Model classification - image: {image}, brand: {brand}, category: {category}")
                    result = _classify_model({
                        "processing_id": image,
                        "processingId": image,
                        "brand": brand,
                        "category": category,
                    })
                    
                    elapsed = time.time() - request_start_time
                    logger.info(f"Model classification request completed in {elapsed:.2f}s")
                    response = _response(
                        200,
                        {
                            "component_type": "model_classification_result",
                            "data": [result],
                            "metadata": {"category": category, "target": target},
                        },
                    )
                    structured_logger.log_response(req_ctx, status_code=200)
                    return response

                logger.info(f"Processing property classification - target: {target}")
                result = _classify_property(category, target, payload)
                
                elapsed = time.time() - request_start_time
                logger.info(f"Property classification request completed in {elapsed:.2f}s")
                response = _response(
                    200,
                    {
                        "component_type": "classification_result",
                        "data": [result],
                        "metadata": {"category": category, "target": target},
                    },
                )
                structured_logger.log_response(req_ctx, status_code=200)
                return response
            except ValueError as exc:
                elapsed = time.time() - request_start_time
                logger.error(f"Validation error after {elapsed:.2f}s: {str(exc)}")
                response = _response(400, {"error": str(exc)})
                structured_logger.log_error(req_ctx, exc, status_code=400)
                return response
            except PermissionError as exc:
                elapsed = time.time() - request_start_time
                logger.error(f"Permission error after {elapsed:.2f}s: {str(exc)}")
                response = _response(403, {"error": str(exc)})
                structured_logger.log_error(req_ctx, exc, status_code=403)
                return response
            except RuntimeError as exc:
                elapsed = time.time() - request_start_time
                logger.error(f"Runtime error after {elapsed:.2f}s: {str(exc)}")
                response = _response(500, {"error": str(exc)})
                structured_logger.log_error(req_ctx, exc, status_code=500)
                return response

        if path.endswith("/health"):
            logger.info("Processing health check request")
            api_base_url = os.getenv("DSL_API_BASE_URL")
            api_key = os.getenv("DSL_API_KEY")
            status = {"status": "unhealthy"}
            try:
                if api_base_url:
                    logger.debug("Performing DSL API health check")
                    client = DSLAPIClient(base_url=api_base_url, api_key=api_key, auth_headers=_request_auth_headers)
                    status = client.health_check()
                    logger.info(f"Health check result: {json.dumps(status, default=str)}")
                else:
                    logger.warning("DSL_API_BASE_URL or DSL_API_KEY not set, skipping health check")
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                status = {"status": "unhealthy", "error": str(e)}

            elapsed = time.time() - request_start_time
            logger.info(f"Health check completed in {elapsed:.2f}s")
            response = _response(200, {"component_type": "health_check", "data": [status], "metadata": {}}, methods="GET,OPTIONS")
            structured_logger.log_response(req_ctx, status_code=200)
            return response

        logger.warning(f"Endpoint not found - path: {path}, method: {method}")
        response = _response(404, {"error": "Endpoint not found"})
        structured_logger.log_response(req_ctx, status_code=404)
        return response

    except Exception as e:
        elapsed = time.time() - request_start_time
        logger.error(f"Unhandled exception after {elapsed:.2f}s: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        traceback.print_exc()
        # Log structured error for metrics (captures duration, error details)
        structured_logger.log_error(req_ctx, e, status_code=500)
        return _response(500, {"error": str(e)})
    finally:
        total_elapsed = time.time() - request_start_time
        logger.info(f"Lambda invocation completed in {total_elapsed:.2f}s - Request ID: {request_id}")
        logger.info("=" * 80)
