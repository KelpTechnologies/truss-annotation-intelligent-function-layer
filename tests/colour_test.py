#!/usr/bin/env python3
"""
Test: colour_test - Colour classification tests
Supports API mode (HTTP) and local mode (direct Python invocation)

Usage:
    API mode:   TEST_MODE=api STAGING_API_URL=https://... DSL_API_KEY=... python colour_test.py
    Local mode: TEST_MODE=local python colour_test.py
"""

import os
import sys
import math
from pathlib import Path

MODE = os.environ.get("TEST_MODE", "api")

# === API MODE ===
def classify_via_api(text_dump: dict, target: str = "colour") -> dict:
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
def classify_via_local(text_dump: dict, target: str = "colour") -> dict:
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
    full_config = config_loader.load_full_agent_config('classifier-colour-bags')
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
        if prediction_id and str(prediction_id) in schema_content:
            item = schema_content[str(prediction_id)]
            primary_name = item.get('name')

        confidence = result.result.get('scores', [{}])[0].get('score', 0)

        return {
            "colour": primary_name,
            "colour_id": prediction_id,
            "confidence": confidence
        }
    else:
        # Unknown/failed
        return {
            "colour": None,
            "colour_id": None,
            "confidence": 0
        }


# === UNIFIED INTERFACE ===
def classify(text_dump: dict, target: str = "colour") -> dict:
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


def assert_colour(result: dict, expected_colour, expected_id, case: str):
    """Assert colour classification result."""
    if expected_colour is None:
        assert is_nan(result.get("colour")), f"{case}: expected NaN colour, got {result.get('colour')}"
        assert is_nan(result.get("colour_id")), f"{case}: expected NaN colour_id, got {result.get('colour_id')}"
    else:
        assert result.get("colour") == expected_colour, f"{case}: expected '{expected_colour}', got '{result.get('colour')}'"
        assert result.get("colour_id") == expected_id, f"{case}: expected id {expected_id}, got {result.get('colour_id')}"


# === TESTS ===
def test_unknown_no_info():
    """TEST 1: Unknown - no information"""
    print("TEST 1: Unknown - no info")

    r1 = classify({"colour": "", "Title": ""})
    assert_colour(r1, None, None, "empty")

    r2 = classify({"colour": "", "Title": "handbag"})
    assert_colour(r2, None, None, "generic")

    print("  PASSED")


def test_unknown_irrelevant():
    """TEST 2: Unknown - irrelevant info"""
    print("TEST 2: Unknown - irrelevant")

    r1 = classify({"colour": "", "Title": "Designer leather crossbody excellent condition"})
    assert_colour(r1, None, None, "no colour hints")

    r2 = classify({"colour": "", "Title": "Vintage authentic luxury tote bag with dust cover"})
    assert_colour(r2, None, None, "vintage text")

    print("  PASSED")


def test_simple_correct():
    """TEST 3: Simple correct examples"""
    print("TEST 3: Simple correct")

    r1 = classify({"colour": "Black", "Title": "Gucci Black Leather Soho Disco Crossbody"})
    assert_colour(r1, "Black", 1, "black")

    r2 = classify({"colour": "Red", "Title": "Prada Red Saffiano Galleria Tote"})
    assert_colour(r2, "Red", 7, "red")

    print("  PASSED")


def test_primary_secondary():
    """TEST 4: Primary vs secondary (bicolor)"""
    print("TEST 4: Primary vs secondary")

    r1 = classify({"colour": "White", "Title": "White leather bag with black trim accents"})
    assert_colour(r1, "White", 4, "white primary")

    r2 = classify({"colour": "neutrals", "Title": "Chanel Beige and Black Bicolor Flap Bag"})
    assert_colour(r2, "Neutrals", 5, "neutrals bicolor")

    print("  PASSED")


def test_noise_extraction():
    """TEST 5: Extract from noise"""
    print("TEST 5: Noise extraction")

    r1 = classify({"colour": "Navy", "Title": "SUPER RARE 2020 Cruise Collection Runway Edition Celebrity Favorite Must-Have Statement Piece Navy Blue Shoulder Bag LIMITED - buy now 100% authentic used good condition, made in italy"})
    assert_colour(r1, "Blue", 11, "navy->blue")

    r2 = classify({"colour": "Pink", "Title": "authentic guaranteed 100% original with receipt certificate included dustbag box ribbon pink leather wallet gift ready. 100% authenic contanct me to buy"})
    assert_colour(r2, "Pink", 2, "pink")

    print("  PASSED")


def test_ground_truth_leprix():
    """Ground truth from leprix dataset"""
    print("TEST GT: Leprix")

    cases = [
        ({"brand": "Gucci", "Title": "Balenciaga Medium Calfskin Hacker Project Jackie 1961 - very good condition", "Colour": "Black"}, "Black", 1),
        ({"brand": "Gucci", "Title": "Bicolor Calfskin Jackie 1961 Wallet On Chain - new with tags", "Colour": "White"}, "White", 4),
        # Row 3 - no explicit colour in text, expected NaN
        ({"brand": "Gucci", "Title": "Tricolor Leather Soho Disco Crossbody AA"}, None, None),
    ]

    for text_dump, exp_colour, exp_id in cases:
        r = classify(text_dump)
        if exp_colour is None:
            assert is_nan(r.get("colour")), f"leprix: expected NaN colour, got '{r.get('colour')}'"
        else:
            assert r.get("colour") == exp_colour, f"leprix: expected '{exp_colour}', got '{r.get('colour')}'"

    print("  PASSED")


def test_ground_truth_italian():
    """Ground truth from Italian brands dataset"""
    print("TEST GT: Italian")

    cases = [
        # Monogram -> typically Brown
        ({"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Nano Papillon Monogram Vintage"}, "Brown", 3),
        ({"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - MANHATTAN GM Monogram"}, "Brown", 3),
        # rosa = pink in Italian
        ({"brand": "Prada", "Title": "PRADA - Occhiale da sole lente a specchio rosa"}, "Pink", 2),
    ]

    for text_dump, exp_colour, exp_id in cases:
        r = classify(text_dump)
        assert r.get("colour") == exp_colour, f"italian: expected '{exp_colour}', got '{r.get('colour')}'"

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
