#!/usr/bin/env python3
"""
Test: model_test - Model classification tests
Supports API mode (HTTP) and local mode (direct Python invocation)

Usage:
    API mode:   TEST_MODE=api STAGING_API_URL=https://... DSL_API_KEY=... python model_test.py
    Local mode: TEST_MODE=local python model_test.py

Note: Model classification requires brand to select brand-specific taxonomy.
"""

import os
import sys
import math
from pathlib import Path

MODE = os.environ.get("TEST_MODE", "api")

# Brand to config_id mapping (for local mode)
BRAND_CONFIG_MAP = {
    "gucci": "classifier-model-bags-gucci-full-taxo",
    "louis vuitton": "classifier-model-bags-louisvuitton-full-taxo",
    "prada": "classifier-model-bags-prada-full-taxo",
    "chanel": "classifier-model-bags-chanel-full-taxo",
}


# === API MODE ===
def classify_via_api(text_dump: dict, target: str = "model") -> dict:
    """Classify via HTTP API."""
    import requests

    BASE_URL = os.environ.get("STAGING_API_URL") or os.environ.get("API_BASE_URL")
    API_KEY = os.environ.get("DSL_API_KEY") or os.environ.get("API_KEY") or ""

    if not BASE_URL:
        raise ValueError("Set STAGING_API_URL or API_BASE_URL env var")

    BASE_URL = BASE_URL.rstrip("/")

    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/{target}",
        json={"text_dump": text_dump, "input_mode": "text-only"},
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )

    assert response.status_code == 200, f"API error {response.status_code}: {response.text}"
    return response.json()["data"][0]


# === LOCAL MODE ===
def classify_via_local(text_dump: dict, target: str = "model") -> dict:
    """Classify via direct Python invocation."""
    # Add repo to path
    repo_root = Path(__file__).parent.parent
    service_path = repo_root / "services" / "automated-annotation"
    sys.path.insert(0, str(service_path))

    from agent_architecture import LLMAnnotationAgent
    from agent_orchestration.csv_config_loader import ConfigLoader
    from agent_architecture.validation import AgentStatus

    # Get brand from text_dump
    brand = text_dump.get("brand", "").lower()
    if not brand:
        return {
            "model": None,
            "model_id": None,
            "root_model": None,
            "root_model_id": None,
            "confidence": 0
        }

    # Get config_id for brand
    config_id = BRAND_CONFIG_MAP.get(brand)
    if not config_id:
        # Try partial match
        for b, cid in BRAND_CONFIG_MAP.items():
            if b in brand or brand in b:
                config_id = cid
                break

    if not config_id:
        return {
            "model": None,
            "model_id": None,
            "root_model": None,
            "root_model_id": None,
            "confidence": 0
        }

    # Load config
    config_loader = ConfigLoader(mode='dynamo', env='staging')
    full_config = config_loader.load_full_agent_config(config_id)
    agent = LLMAnnotationAgent(full_config=full_config, log_IO=False)

    # Format text input
    text_parts = []
    for k, v in text_dump.items():
        if v:
            text_parts.append(f"{k}: {v}")
    formatted_text = "\n".join(text_parts)

    # Execute
    result = agent.execute({
        'item_id': 'test-local',
        'text_input': formatted_text,
        'input_mode': 'text-only'
    })

    # Extract result
    if result.status == AgentStatus.SUCCESS and result.result:
        prediction_id = result.result.get('prediction_id')
        schema_content = result.schema.get('schema_content', {}) if result.schema else {}

        # Get primary name
        primary_name = None
        root_name = None
        root_id = None

        if prediction_id and str(prediction_id) in schema_content:
            item = schema_content[str(prediction_id)]
            primary_name = item.get('name')
            # Get root from schema metadata
            root_id = item.get('parent_id') or prediction_id
            if str(root_id) in schema_content:
                root_name = schema_content[str(root_id)].get('name')
            else:
                root_name = primary_name
                root_id = prediction_id

        confidence = result.result.get('scores', [{}])[0].get('score', 0)

        return {
            "model": primary_name,
            "model_id": prediction_id,
            "root_model": root_name,
            "root_model_id": root_id,
            "confidence": confidence
        }
    else:
        # Unknown/failed
        return {
            "model": None,
            "model_id": None,
            "root_model": None,
            "root_model_id": None,
            "confidence": 0
        }


# === UNIFIED INTERFACE ===
def classify(text_dump: dict, target: str = "model") -> dict:
    if MODE == "local":
        return classify_via_local(text_dump, target)
    return classify_via_api(text_dump, target)


# === HELPERS ===
def is_nan(value):
    """Check if value is NaN/null/empty."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ("nan", ""):
        return True
    return False


def assert_model(result: dict, expected_model, expected_id, expected_root, expected_root_id, case: str):
    """Assert model classification result."""
    if expected_model is None:
        assert is_nan(result.get("model")), f"{case}: expected NaN model, got {result.get('model')}"
        assert is_nan(result.get("model_id")), f"{case}: expected NaN model_id, got {result.get('model_id')}"
    else:
        assert result.get("model") == expected_model, f"{case}: expected '{expected_model}', got '{result.get('model')}'"
        assert result.get("model_id") == expected_id, f"{case}: expected id {expected_id}, got {result.get('model_id')}"
        if expected_root:
            assert result.get("root_model") == expected_root, f"{case}: expected root '{expected_root}', got '{result.get('root_model')}'"
        if expected_root_id:
            assert result.get("root_model_id") == expected_root_id, f"{case}: expected root_id {expected_root_id}, got {result.get('root_model_id')}"


# === TESTS ===
def test_unknown_no_info():
    """TEST 1: Unknown - no information"""
    print("TEST 1: Unknown - no info")

    r1 = classify({"brand": "Gucci", "Title": ""})
    assert_model(r1, None, None, None, None, "empty title")

    r2 = classify({"brand": "Prada", "Title": "leather bag"})
    assert_model(r2, None, None, None, None, "generic")

    print("  PASSED")


def test_unknown_irrelevant():
    """TEST 2: Unknown - irrelevant info (no model indicators)"""
    print("TEST 2: Unknown - irrelevant")

    r1 = classify({"brand": "Gucci", "Title": "Gucci handbag excellent condition authentic"})
    assert_model(r1, None, None, None, None, "no model hints")

    r2 = classify({"brand": "Louis Vuitton", "Title": "Louis Vuitton tote bag monogram canvas"})
    assert_model(r2, None, None, None, None, "generic LV tote")

    print("  PASSED")


def test_simple_correct():
    """TEST 3: Simple correct examples"""
    print("TEST 3: Simple correct")

    r1 = classify({"brand": "Gucci", "Title": "Gucci Soho Disco Crossbody Black Leather"})
    assert_model(r1, "Soho Disco", 1056, "Soho Disco", 1056, "soho disco")

    r2 = classify({"brand": "Louis Vuitton", "Title": "Louis Vuitton Neverfull MM Monogram"})
    assert_model(r2, "Neverfull", 445, "Neverfull", 445, "neverfull")

    print("  PASSED")


def test_primary_secondary():
    """TEST 4: Primary vs secondary (model variants)"""
    print("TEST 4: Model variants")

    r1 = classify({"brand": "Gucci", "Title": "Calfskin Jackie 1961 Small Shoulder Bag"})
    assert_model(r1, "Jackie 1961", 307, "Jackie", 306, "jackie 1961")

    r2 = classify({"brand": "Gucci", "Title": "Bamboo Night Clutch Black Calfskin"})
    assert_model(r2, "Bamboo", 50, "Bamboo", 50, "bamboo")

    print("  PASSED")


def test_noise_extraction():
    """TEST 5: Extract from noise"""
    print("TEST 5: Noise extraction")

    r1 = classify({"brand": "Prada", "Title": "AUTHENTIC GUARANTEED Fast Shipping A+++ Seller Rated 5 Stars Re-Edition 2005 Nylon Mini Top Handle RARE COLOR"})
    assert_model(r1, "Re-Edition 2005", 524, "Re-Edition", 526, "re-edition 2005")

    r2 = classify({"brand": "Chanel", "Title": "limited edition cruise 2019 collection exclusive VIP client gift classic flap bag medium caviar leather gold hardware"})
    # Note: "Classic Flap / Classic 11.12" is the full name
    assert r2.get("root_model") in ["Timeless/Classic", "Classic Flap"], f"classic flap root: got '{r2.get('root_model')}'"

    print("  PASSED")


def test_ground_truth_leprix():
    """Ground truth from leprix dataset"""
    print("TEST GT: Leprix")

    cases = [
        ({"brand": "Gucci", "Title": "Balenciaga Medium Calfskin Hacker Project Jackie 1961 - very good condition"}, "Jackie 1961", 307, "Jackie", 306),
        ({"brand": "Gucci", "Title": "Calfskin Bamboo Jackie Hobo - Used Excellent"}, "Jackie", 306, "Jackie", 306),
        ({"brand": "Gucci", "Title": "Tricolor Leather Soho Disco Crossbody AA"}, "Soho Disco", 1056, "Soho Disco", 1056),
        # Prada Tessuto - material not model, expect unknown
        ({"brand": "Prada", "Title": "Tessuto Zip Top Crossbody - AB"}, None, None, None, None),
    ]

    for text_dump, exp_model, exp_id, exp_root, exp_root_id in cases:
        r = classify(text_dump)
        if exp_model is None:
            assert is_nan(r.get("model")), f"leprix: expected NaN model, got '{r.get('model')}'"
        else:
            assert r.get("model") == exp_model, f"leprix: expected '{exp_model}', got '{r.get('model')}'"

    print("  PASSED")


def test_ground_truth_italian():
    """Ground truth from Italian brands dataset"""
    print("TEST GT: Italian")

    cases = [
        ({"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Nano Papillon Monogram Vintage"}, "Papillon", 483),
        ({"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - MANHATTAN GM Monogram"}, "Manhattan", 397),
        # Murakami is collaboration name not model, expect unknown
        ({"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Portafoglio Murakami Clip"}, None, None),
    ]

    for text_dump, exp_model, exp_id in cases:
        r = classify(text_dump)
        if exp_model is None:
            assert is_nan(r.get("model")), f"italian: expected NaN model, got '{r.get('model')}'"
        else:
            assert r.get("model") == exp_model, f"italian: expected '{exp_model}', got '{r.get('model')}'"

    print("  PASSED")


if __name__ == "__main__":
    print(f"Mode: {MODE}")
    try:
        test_unknown_no_info()
        test_unknown_irrelevant()
        test_simple_correct()
        test_primary_secondary()
        test_noise_extraction()
        test_ground_truth_leprix()
        test_ground_truth_italian()
        print("PASSED")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
