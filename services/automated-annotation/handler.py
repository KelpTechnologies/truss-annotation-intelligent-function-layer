import json
import os
import traceback
import base64

# Import DSL components from simplified package
from dsl import DSLAPIClient, ConfigLoader, LLMAnnotationAgent
from dsl import load_property_id_mapping_api


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
    """Ensure Application Default Credentials are available from env.

    Expects env:
      - GCP_SERVICE_ACCOUNT_JSON: JSON (or base64 of JSON) for a service_account
      - GOOGLE_APPLICATION_CREDENTIALS: path (defaults to /tmp/gcp_sa.json)
      - VERTEXAI_PROJECT, VERTEXAI_LOCATION (optional)
    """
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")

    if not sa_json:
        # Nothing to write; user may have mounted credentials another way
        return

    try:
        if sa_json.strip().startswith("{"):
            content = sa_json
        else:
            try:
                content = base64.b64decode(sa_json).decode("utf-8")
            except Exception:
                content = sa_json

        # Write to creds_path
        os.makedirs(os.path.dirname(creds_path), exist_ok=True)
        with open(creds_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Ensure ADC picks it up
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        # Mirror project to GOOGLE_CLOUD_PROJECT if provided
        if os.getenv("VERTEXAI_PROJECT") and not os.getenv("GOOGLE_CLOUD_PROJECT"):
            os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("VERTEXAI_PROJECT")
    except Exception:
        # Do not fail classification if credentials write fails; VertexAI will error later
        pass


def _classify_item(payload: dict):
    api_base_url = os.getenv("DSL_API_BASE_URL")
    api_key = os.getenv("DSL_API_KEY")
    if not api_base_url or not api_key:
        raise ValueError("DSL_API_BASE_URL and DSL_API_KEY environment variables are required")

    property_name = payload.get("property")
    root_type_id = payload.get("root_type_id")
    if not property_name or root_type_id is None:
        raise ValueError("'property' and 'root_type_id' are required fields")

    input_mode = payload.get("input_mode", "auto")
    image_url = payload.get("image_url")
    garment_id = str(payload.get("garment_id") or "")
    brand = payload.get("brand")
    text_metadata = _build_text_metadata(payload)

    # Ensure Google ADC is available before initializing VertexAI client
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

        if path.endswith("/classify") and method == "POST":
            payload = _parse_body(event)
            result = _classify_item(payload)
            return _response(200, {"component_type": "classification_result", "data": [result], "metadata": {}})

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
