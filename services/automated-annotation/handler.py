import json
import os
import traceback
import base64
import logging
import requests

# Import DSL components from simplified package
from dsl import DSLAPIClient, ConfigLoader, LLMAnnotationAgent
from dsl import load_property_id_mapping_api

logger = logging.getLogger(__name__)

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
    if not event.get("body"):
        return {}
    try:
        return json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
    except Exception:
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
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

    if not sa_json:
        return

    try:
        if sa_json.strip().startswith("{"):
            content = sa_json
        else:
            try:
                content = base64.b64decode(sa_json).decode("utf-8")
            except Exception:
                content = sa_json

        os.makedirs(os.path.dirname(creds_path), exist_ok=True)
        with open(creds_path, "w", encoding="utf-8") as f:
            f.write(content)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        if os.getenv("VERTEXAI_PROJECT") and not os.getenv("GOOGLE_CLOUD_PROJECT"):
            os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("VERTEXAI_PROJECT")
    except Exception:
        pass


def _get_signed_image_url(image: str) -> str:
    """Get signed image URL from annotation-data-service-layer image-service."""
    api_base_url = os.getenv("ANNOTATION_API_BASE_URL")
    api_key = os.getenv("ANNOTATION_API_KEY")
    
    if not api_base_url:
        raise ValueError("ANNOTATION_API_BASE_URL environment variable is required")
    
    url = f"{api_base_url.rstrip('/')}/images/processed/{image}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers['x-api-key'] = api_key
        
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    result = response.json()
    data = result.get('data', result)
    processed_image = data.get('processedImage', {})
    download_url = processed_image.get('downloadUrl')
    
    if not download_url:
        raise ValueError(f"No downloadUrl found in image service response for image '{image}'")
    
    return download_url


def _lookup_root_from_child(api_client: DSLAPIClient, property_type: str, value: str, brand: str = None, root_type: str = None, partition: str = "bags") -> dict:
    """Lookup root taxonomy value from child value using the knowledge service API."""
    lookup_result = api_client.lookup_root(
        property_type=property_type,
        value=value,
        brand=brand,
        root_type=root_type,
        partition=partition
    )
    
    data = lookup_result.get('data', lookup_result)
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    elif not isinstance(data, dict):
        data = lookup_result
        
    if not data.get('found'):
        return None
        
    result = {
        'root': data.get('root'),
        'root_id': data.get('root_id'),
        'child_id': data.get(f'{property_type}_id'),
    }
    
    return result if result['root'] and result['root_id'] else None


def _classify_model(payload: dict):
    """Classify model using a pre-computed image vector from the image-processing table."""
    if not classify_image:
        raise ValueError("Vector classifier pipeline not available")

    processing_id = payload.get("processing_id") or payload.get("processingId")
    if not processing_id:
        raise ValueError("'processing_id' is required for model classification")

    brand = payload.get("brand")
    if not brand:
        raise ValueError("'brand' is required for model classification")

    result = classify_image(
        processing_id=processing_id,
        brand=brand,
        k=7,
    )

    predicted_model = result.get("predicted_model")
    predicted_root_model = result.get("predicted_root_model")

    # Always lookup root from knowledge service
    root_lookup_result = None
    if predicted_model:
        api_base_url = os.getenv("DSL_API_BASE_URL")
        api_key = os.getenv("DSL_API_KEY")
        if not api_base_url or not api_key:
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
    else:
        root_model = None
        root_model_id = None
        model_id = None

    return {
        "model": predicted_model,
        "model_id": model_id,
        "root_model": root_model,
        "root_model_id": root_model_id,
        "confidence": result.get("confidence", 0.0),
    }


def _classify_property(category: str, target: str, request_payload: dict):
    """Classify a property using LLM."""
    category_config = CATEGORY_CONFIG.get(category)
    if not category_config:
        raise ValueError(f"Unsupported category '{category}'")

    property_map = category_config.get("properties", {})
    internal_property_name = property_map.get(target)
    if not internal_property_name:
        raise ValueError(f"Unsupported classification target '{target}' for category '{category}'")

    # Get image and lookup signed URL
    image = request_payload.get("image")
    if not image:
        raise ValueError("'image' is required for property classification")
    
    image_url = _get_signed_image_url(image)

    text_dump = request_payload.get("text_dump")
    if not text_dump:
        raise ValueError("'text_dump' is required for property classification")

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

    raw_result = _classify_item(classification_payload)

    primary_value = raw_result.get("primary")
    if isinstance(primary_value, str) and primary_value.startswith("ID "):
        try:
            primary_value = primary_value.split(":", 1)[1].strip()
        except Exception:
            pass

    # Lookup root for model and material properties
    root_lookup_result = None
    if target in ["model", "material"] and primary_value:
        api_base_url = os.getenv("DSL_API_BASE_URL")
        api_key = os.getenv("DSL_API_KEY")
        if not api_base_url or not api_key:
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

    # Build simplified response
    if target == "model":
        return {
            "model": primary_value,
            "model_id": root_lookup_result.get('child_id') if root_lookup_result else None,
            "root_model": root_lookup_result.get('root') if root_lookup_result else None,
            "root_model_id": root_lookup_result.get('root_id') if root_lookup_result else None,
            "confidence": raw_result.get("confidence", 0.0),
        }
    elif target == "material":
        return {
            "material": primary_value,
            "material_id": root_lookup_result.get('child_id') if root_lookup_result else None,
            "root_material": root_lookup_result.get('root') if root_lookup_result else None,
            "root_material_id": root_lookup_result.get('root_id') if root_lookup_result else None,
            "confidence": raw_result.get("confidence", 0.0),
        }
    else:
        return {
            target: primary_value,
            "confidence": raw_result.get("confidence", 0.0),
        }


def _classify_item(payload: dict):
    """Classify an item using LLM."""
    # Use ANNOTATION_API_BASE_URL for classifier configs, templates, and context data
    api_base_url = os.getenv("ANNOTATION_API_BASE_URL")
    api_key = os.getenv("ANNOTATION_API_KEY")
    if not api_base_url:
        raise ValueError("ANNOTATION_API_BASE_URL environment variable is required")

    property_name = payload.get("property")
    root_type_id = payload.get("root_type_id")
    if not property_name or root_type_id is None:
        raise ValueError("'property' and 'root_type_id' are required fields")

    input_mode = payload.get("input_mode", "auto")
    image_url = payload.get("image_url")
    garment_id = str(payload.get("garment_id") or "")
    brand = payload.get("brand")
    text_metadata = _build_text_metadata(payload)

    _ensure_gcp_adc()

    api_client = DSLAPIClient(base_url=api_base_url, api_key=api_key)
    config_loader = ConfigLoader(mode='api', api_client=api_client)

    api_response = config_loader.load_classifier_config(property_name, int(root_type_id))
    config = api_response.get('data', api_response)

    template_id = config.get('prompt_template')
    if not template_id:
        raise ValueError("No prompt_template specified in configuration")
    template = config_loader.load_prompt_template(template_id)

    context_data = config_loader.load_context_data(property_name, int(root_type_id))

    classifier = LLMAnnotationAgent(
        model_name=config.get('model', 'gemini-2.5-flash-lite'),
        project_id=os.getenv('VERTEXAI_PROJECT', config.get('project_id', 'truss-data-science')),
        location=os.getenv('VERTEXAI_LOCATION', config.get('location', 'us-central1')),
        prompt_template_path=None,
        context_mode=config.get('default_context_mode', 'full-context'),
        log_IO=False,
        config={
            'temperature': config.get('temperature', 0.1),
            'max_output_tokens': config.get('max_output_tokens', 1024),
        },
    )

    classifier.prompt_template = template

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

    resolve_names = bool(payload.get("resolve_names", False))
    primary = result.primary
    alternatives = result.alternatives

    if resolve_names:
        id_map = load_property_id_mapping_api(context_data.get('data', context_data))

        def _fmt(val: str):
            try:
                if val.startswith("ID "):
                    pred_id = int(val.split("ID ")[1])
                    name = id_map.get(pred_id)
                    return f"ID {pred_id}: {name}" if name else val
            except Exception:
                return val
            return val

        primary = _fmt(primary)
        alternatives = [_fmt(a) for a in alternatives]

    return {
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


def lambda_handler(event, context):
    try:
        method = event.get("httpMethod", "GET")
        path = (event.get("path") or "").rstrip("/")

        if method == "OPTIONS":
            return _response(200, {"ok": True})

        segments = [segment for segment in path.strip("/").split("/") if segment]

        if (
            len(segments) >= 5
            and segments[0] == "automations"
            and segments[1] == "annotation"
            and segments[3] == "classify"
            and method == "POST"
        ):
            category = segments[2]
            target = segments[4]

            payload = _parse_body(event)

            if category not in CATEGORY_CONFIG:
                return _response(404, {"error": f"Unsupported category '{category}'"})

            try:
                if target == "model":
                    image = payload.get("image")
                    if not image:
                        raise ValueError("'image' is required for model classification")

                    brand = payload.get("brand") or CATEGORY_CONFIG[category].get("default_brand")
                    if not brand:
                        raise ValueError("'brand' is required for model classification")

                    result = _classify_model({
                        "processing_id": image,
                        "processingId": image,
                        "brand": brand,
                        "category": category,
                    })
                    return _response(
                        200,
                        {
                            "component_type": "model_classification_result",
                            "data": [result],
                            "metadata": {"category": category, "target": target},
                        },
                    )

                result = _classify_property(category, target, payload)
                return _response(
                    200,
                    {
                        "component_type": "classification_result",
                        "data": [result],
                        "metadata": {"category": category, "target": target},
                    },
                )
            except ValueError as exc:
                return _response(400, {"error": str(exc)})

        if path.endswith("/health"):
            api_base_url = os.getenv("DSL_API_BASE_URL")
            api_key = os.getenv("DSL_API_KEY")
            status = {"status": "unhealthy"}
            try:
                if api_base_url and api_key:
                    client = DSLAPIClient(base_url=api_base_url, api_key=api_key)
                    status = client.health_check()
            except Exception as e:
                status = {"status": "unhealthy", "error": str(e)}

            return _response(200, {"component_type": "health_check", "data": [status], "metadata": {}}, methods="GET,OPTIONS")

        return _response(404, {"error": "Endpoint not found"})

    except Exception as e:
        traceback.print_exc()
        return _response(500, {"error": str(e)})
