#!/usr/bin/env python3
"""
Test: material_test - Material classification tests
Supports API mode (HTTP) and local mode (direct Python invocation)

Usage:
    API mode:   TEST_MODE=api STAGING_API_URL=https://... DSL_API_KEY=... python material_test.py
    Local mode: TEST_MODE=local python material_test.py
"""

import os
import sys
import math
from pathlib import Path

MODE = os.environ.get("TEST_MODE", "api")

# === API MODE ===
def classify_via_api(text_dump: dict, target: str = "material") -> dict:
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
def classify_via_local(text_dump: dict, target: str = "material") -> dict:
    """Classify via direct Python invocation."""
    # Add repo to path
    repo_root = Path(__file__).parent.parent
    service_path = repo_root / "services" / "automated-annotation"
    sys.path.insert(0, str(service_path))

    from agent_architecture import LLMAnnotationAgent
    from agent_orchestration.csv_config_loader import ConfigLoader
    from agent_architecture.validation import AgentStatus

    # Load config
    config_loader = ConfigLoader(mode='dynamo', env='staging')
    full_config = config_loader.load_full_agent_config('classifier-material-bags')
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
            "material": primary_name,
            "material_id": prediction_id,
            "root_material": root_name,
            "root_material_id": root_id,
            "confidence": confidence
        }
    else:
        # Unknown/failed
        return {
            "material": None,
            "material_id": None,
            "root_material": None,
            "root_material_id": None,
            "confidence": 0
        }


# === UNIFIED INTERFACE ===
def classify(text_dump: dict, target: str = "material") -> dict:
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


def assert_material(result: dict, expected_material, expected_id, expected_root, expected_root_id, case: str):
    """Assert material classification result."""
    if expected_material is None:
        assert is_nan(result.get("material")), f"{case}: expected NaN material, got {result.get('material')}"
        assert is_nan(result.get("material_id")), f"{case}: expected NaN material_id, got {result.get('material_id')}"
    else:
        assert result.get("material") == expected_material, f"{case}: expected '{expected_material}', got '{result.get('material')}'"
        assert result.get("material_id") == expected_id, f"{case}: expected id {expected_id}, got {result.get('material_id')}"
        assert result.get("root_material") == expected_root, f"{case}: expected root '{expected_root}', got '{result.get('root_material')}'"
        assert result.get("root_material_id") == expected_root_id, f"{case}: expected root_id {expected_root_id}, got {result.get('root_material_id')}"


# === TESTS ===
def test_unknown_no_info():
    """TEST 1: Unknown - no information"""
    print("TEST 1: Unknown - no info")

    r1 = classify({"material": "", "Title": ""})
    assert_material(r1, None, None, None, None, "empty")

    r2 = classify({"material": "", "Title": "item"})
    assert_material(r2, None, None, None, None, "generic")

    print("  PASSED")


def test_unknown_irrelevant():
    """TEST 2: Unknown - irrelevant info"""
    print("TEST 2: Unknown - irrelevant")

    r1 = classify({"material": "", "Title": "Beautiful vintage accessory great condition"})
    assert_material(r1, None, None, None, None, "no material hints")

    r2 = classify({"material": "", "Title": "Perfect gift for her birthday present idea"})
    assert_material(r2, None, None, None, None, "gift text")

    print("  PASSED")


def test_simple_correct():
    """TEST 3: Simple correct examples"""
    print("TEST 3: Simple correct")

    r1 = classify({"material": "Canvas", "Title": "Louis Vuitton Monogram Canvas Neverfull"})
    assert_material(r1, "Canvas", 2, "Canvas", 2, "canvas")

    r2 = classify({"material": "Calfskin", "Title": "Crossbody Louis Vuitton Neverfull Calfskin"})
    assert_material(r2, "Calfskin", 47, "Leather", 1, "calfskin")

    print("  PASSED")


def test_primary_secondary():
    """TEST 4: Primary vs secondary"""
    print("TEST 4: Primary vs secondary")

    r1 = classify({"material": "Calfskin", "Title": "Canvas tote bag with calfskin leather trim"})
    assert_material(r1, "Calfskin", 47, "Leather", 1, "calfskin trim")

    r2 = classify({"material": "Patent Leather", "Title": "Patent leather clutch with Cotton interior lining"})
    assert_material(r2, "Patent Leather", 10, "Leather", 1, "patent leather")

    print("  PASSED")


def test_noise_extraction():
    """TEST 5: Extract from noise"""
    print("TEST 5: Noise extraction")

    r1 = classify({"material": "Nylon", "Title": "RARE 2019 Limited Edition Holiday Collection Exclusive VIP Gift Set Premium Quality Designer Fashion Accessory Nylon Crossbody Excellent Condition"})
    assert_material(r1, "Nylon", 5, "Nylon", 5, "nylon in noise")

    r2 = classify({"material": "Saffiano Leather", "Title": "authentic guaranteed 100% real deal fast shipping free returns great seller A+++ saffiano leather wallet brand new with tags"})
    # Note: Saffiano -> Leather root
    assert r2.get("root_material") == "Leather", f"saffiano root: expected Leather, got {r2.get('root_material')}"

    print("  PASSED")


def test_ground_truth_leprix():
    """Ground truth from leprix dataset"""
    print("TEST GT: Leprix")

    cases = [
        ({"brand": "Gucci", "Title": "Balenciaga Medium Calfskin Hacker Project Jackie 1961 - very good condition"}, "Leather", 1),
        ({"brand": "Gucci", "Title": "Bicolor Calfskin Jackie 1961 Wallet On Chain - new with tags"}, "Leather", 1),
        ({"brand": "Prada", "Title": "Tessuto Zip Top Crossbody - AB"}, "Leather", 1),
    ]

    for text_dump, exp_root, exp_root_id in cases:
        r = classify(text_dump)
        assert r.get("root_material") == exp_root, f"leprix: expected root '{exp_root}', got '{r.get('root_material')}'"

    print("  PASSED")


def test_ground_truth_italian():
    """Ground truth from Italian brands dataset"""
    print("TEST GT: Italian")

    cases = [
        ({"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Nano Papillon Monogram Vintage", "Tags": "bauletto, borsa di lusso, borsa marrone, borsa ottime condizioni, borsa second-hand, borsa tela, borsa usata, Louis vuitton, Monogram, nano, Nanopapillon, Papillon, papillon monogram, Papillon vintage"}, "Canvas", 2),
        ({"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - MANHATTAN GM Monogram", "Tags": "bauletto, borsa di lusso, borsa marrone, borsa ottime condizioni, Borsa pelle, borsa second-hand, borsa tela, borsa usata, Louis vuitton, manhattan gm, Monogram"}, "Leather", 1),
        ({"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Portafoglio Murakami Clip", "Tags": "limited-edition, Louis vuitton, Monogram, Multicolor Monogram, Murakami, Murakami Monogram, Portafoglio, Portafoglio Clip, portafoglio di lusso, takeshi Murakami, unisex"}, "Leather", 1),
    ]

    for text_dump, exp_root, exp_root_id in cases:
        r = classify(text_dump)
        assert r.get("root_material") == exp_root, f"italian: expected root '{exp_root}', got '{r.get('root_material')}'"

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
