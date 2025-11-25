import json
import os
import traceback
import base64
import logging
import requests
import time
from datetime import datetime

# Import DSL components from simplified package
from dsl import DSLAPIClient, ConfigLoader, LLMAnnotationAgent
from dsl import load_property_id_mapping_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


def _ensure_gcp_adc():
    """Ensure Application Default Credentials are available from env."""
    logger.debug("Checking GCP Application Default Credentials")
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

    if not sa_json:
        logger.debug("No GCP_SERVICE_ACCOUNT_JSON found, skipping ADC setup")
        return

    try:
        logger.debug(f"Setting up GCP credentials at {creds_path}")
        if sa_json.strip().startswith("{"):
            content = sa_json
            logger.debug("GCP credentials are JSON format")
        else:
            try:
                content = base64.b64decode(sa_json).decode("utf-8")
                logger.debug("GCP credentials decoded from base64")
            except Exception as e:
                logger.warning(f"Failed to decode base64 credentials, using as-is: {str(e)}")
                content = sa_json

        os.makedirs(os.path.dirname(creds_path), exist_ok=True)
        with open(creds_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"GCP credentials written to {creds_path}")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        if os.getenv("VERTEXAI_PROJECT") and not os.getenv("GOOGLE_CLOUD_PROJECT"):
            os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("VERTEXAI_PROJECT")
            logger.debug(f"Set GOOGLE_CLOUD_PROJECT to {os.getenv('VERTEXAI_PROJECT')}")
    except Exception as e:
        logger.error(f"Failed to set up GCP credentials: {str(e)}")
        pass


def _get_signed_image_url(image: str) -> str:
    """Get signed image URL from annotation-data-service-layer image-service."""
    start_time = time.time()
    logger.info(f"Fetching signed image URL for image ID: {image}")
    
    api_base_url = os.getenv("ANNOTATION_API_BASE_URL")
    api_key = os.getenv("ANNOTATION_API_KEY")
    
    if not api_base_url:
        logger.error("ANNOTATION_API_BASE_URL environment variable is not set")
        raise ValueError("ANNOTATION_API_BASE_URL environment variable is required")
    
    url = f"{api_base_url.rstrip('/')}/images/processed/{image}"
    logger.debug(f"Image service URL: {url}")
    
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers['x-api-key'] = api_key
        logger.debug("API key provided for image service request")
    else:
        logger.warning("No API key provided for image service request")
        
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
        if not api_base_url or not api_key:
            logger.error("DSL_API_BASE_URL or DSL_API_KEY not set")
            raise ValueError("DSL_API_BASE_URL and DSL_API_KEY environment variables are required")
        
        api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key)
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

    # Lookup root for model and material properties
    root_lookup_result = None
    if target in ["model", "material"] and primary_value:
        logger.info(f"Looking up root for {target}='{primary_value}'")
        api_base_url = os.getenv("DSL_API_BASE_URL")
        api_key = os.getenv("DSL_API_KEY")
        if not api_base_url or not api_key:
            logger.error("DSL_API_BASE_URL or DSL_API_KEY not set")
            raise ValueError("DSL_API_BASE_URL and DSL_API_KEY environment variables are required")
        
        api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key)
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
    
    # Use ANNOTATION_API_BASE_URL for classifier configs, templates, and context data
    api_base_url = os.getenv("ANNOTATION_API_BASE_URL")
    api_key = os.getenv("ANNOTATION_API_KEY")
    if not api_base_url:
        logger.error("ANNOTATION_API_BASE_URL environment variable is not set")
        raise ValueError("ANNOTATION_API_BASE_URL environment variable is required")
    
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

    logger.debug("Ensuring GCP Application Default Credentials")
    _ensure_gcp_adc()

    api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key)
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

    model_name = config.get('model', 'gemini-2.5-flash-lite')
    project_id = os.getenv('VERTEXAI_PROJECT', config.get('project_id', 'truss-data-science'))
    location = os.getenv('VERTEXAI_LOCATION', config.get('location', 'us-central1'))
    
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
                    return f"ID {pred_id}: {name}" if name else val
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
    request_start_time = time.time()
    request_id = context.request_id if context else "unknown"
    
    logger.info("=" * 80)
    logger.info(f"Lambda invocation started - Request ID: {request_id}")
    logger.info(f"Event method: {event.get('httpMethod', 'UNKNOWN')}, path: {event.get('path', 'UNKNOWN')}")
    logger.debug(f"Full event: {json.dumps(event, default=str)[:1000]}")
    
    try:
        method = event.get("httpMethod", "GET")
        path = (event.get("path") or "").rstrip("/")
        
        logger.info(f"Processing {method} request to {path}")

        if method == "OPTIONS":
            logger.info("Handling OPTIONS request")
            return _response(200, {"ok": True})

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
                return _response(404, {"error": f"Unsupported category '{category}'"})

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
                    return _response(
                        200,
                        {
                            "component_type": "model_classification_result",
                            "data": [result],
                            "metadata": {"category": category, "target": target},
                        },
                    )

                logger.info(f"Processing property classification - target: {target}")
                result = _classify_property(category, target, payload)
                
                elapsed = time.time() - request_start_time
                logger.info(f"Property classification request completed in {elapsed:.2f}s")
                return _response(
                    200,
                    {
                        "component_type": "classification_result",
                        "data": [result],
                        "metadata": {"category": category, "target": target},
                    },
                )
            except ValueError as exc:
                elapsed = time.time() - request_start_time
                logger.error(f"Validation error after {elapsed:.2f}s: {str(exc)}")
                return _response(400, {"error": str(exc)})

        if path.endswith("/health"):
            logger.info("Processing health check request")
            api_base_url = os.getenv("DSL_API_BASE_URL")
            api_key = os.getenv("DSL_API_KEY")
            status = {"status": "unhealthy"}
            try:
                if api_base_url and api_key:
                    logger.debug("Performing DSL API health check")
                    client = DSLAPIClient(base_url=api_base_url, api_key=api_key)
                    status = client.health_check()
                    logger.info(f"Health check result: {json.dumps(status, default=str)}")
                else:
                    logger.warning("DSL_API_BASE_URL or DSL_API_KEY not set, skipping health check")
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                status = {"status": "unhealthy", "error": str(e)}

            elapsed = time.time() - request_start_time
            logger.info(f"Health check completed in {elapsed:.2f}s")
            return _response(200, {"component_type": "health_check", "data": [status], "metadata": {}}, methods="GET,OPTIONS")

        logger.warning(f"Endpoint not found - path: {path}, method: {method}")
        return _response(404, {"error": "Endpoint not found"})

    except Exception as e:
        elapsed = time.time() - request_start_time
        logger.error(f"Unhandled exception after {elapsed:.2f}s: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        traceback.print_exc()
        return _response(500, {"error": str(e)})
    finally:
        total_elapsed = time.time() - request_start_time
        logger.info(f"Lambda invocation completed in {total_elapsed:.2f}s - Request ID: {request_id}")
        logger.info("=" * 80)
