#!/usr/bin/env python3
"""
Test: colour_test - Colour classification tests (TEXT-ONLY)

Simulates Lambda routing: text-only -> classifier-colour-bags-text

Usage:
    python colour_test.py --base-url https://... --api-key ...
    python colour_test.py --mode local
"""

import argparse
import sys
import math
import json
import os
import tempfile
from pathlib import Path

# Load .env at initialization
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Set cross-platform temp path for GCP credentials (before lambda imports)
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(tempfile.gettempdir(), "gcp_sa.json")

parser = argparse.ArgumentParser(description="Colour classification tests")
parser.add_argument("--mode", "-m", default="api", choices=["api", "local"], help="Test mode")
parser.add_argument("--base-url", "-u", help="API base URL (or set STAGING_API_URL)")
parser.add_argument("--api-key", "-k", help="API key (or set DSL_API_KEY)")
args = parser.parse_args()

MODE = args.mode
BASE_URL = args.base_url or os.environ.get("STAGING_API_URL") or os.environ.get("API_BASE_URL")
BASE_URL = BASE_URL.rstrip("/") if BASE_URL else None
API_KEY = args.api_key or os.environ.get("DSL_API_KEY") or os.environ.get("API_KEY") or ""
RESULTS = []

# Text-only config (matches Lambda routing: text-only -> classifier-{prop}-bags-text)
CONFIG_ID = "classifier-colour-bags-text"


def classify_via_api(text_dump: dict) -> dict:
    """Classify via HTTP API - simulates Lambda endpoint."""
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required or set STAGING_API_URL in .env")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/colour",
        json={"text_dump": text_dump, "input_mode": "text-only"},
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    return response.json()["data"][0]


def classify_via_local(text_dump: dict) -> dict:
    """
    Classify via direct Python - simulates Lambda routing.

    Lambda routing logic (from agent_orchestration_api_handler.py):
    1. detect_input_mode() -> "text-only" (no image, has text)
    2. get_config_id_for_input_mode(base, "text-only") -> "classifier-colour-bags-text"
    3. classify_for_api() loads config and runs agent
    """
    repo_root = Path(__file__).parent.parent
    service_path = repo_root / "services" / "automated-annotation"
    sys.path.insert(0, str(service_path))

    # Ensure GCP credentials are set up
    from core.utils.credentials import ensure_gcp_adc
    ensure_gcp_adc()

    from agent_orchestration.classifier_api_orchestration import classify_for_api

    # Format text_dump as JSON string (matches Lambda behavior)
    text_input = json.dumps(text_dump, indent=2)

    # Call orchestration function directly (simulates Lambda)
    result = classify_for_api(
        config_id=CONFIG_ID,
        text_input=text_input,
        input_mode="text-only",
        env="staging"
    )

    return {
        "colour": result.get("primary_name"),
        "colour_id": result.get("primary_id"),
        "confidence": result.get("confidence", 0),
        "success": result.get("success", False),
        "reasoning": result.get("reasoning", "")
    }


def classify(text_dump: dict) -> dict:
    return classify_via_local(text_dump) if MODE == "local" else classify_via_api(text_dump)


def is_nan(value):
    """Check if value represents NaN/null/empty (legacy helper)."""
    if value is None: return True
    if isinstance(value, float) and math.isnan(value): return True
    if isinstance(value, str) and value.lower() in ("nan", ""): return True
    return False


def is_unknown(value):
    """
    Check if value represents 'unknown' classification result.

    Classifier returns ID 0 + name "Unknown" when colour can't be identified.
    - colour_id: 0 (int)
    - colour: "Unknown" (str)
    """
    if value is None: return True
    if value == 0: return True
    if isinstance(value, str) and value.lower() in ("unknown", "nan", ""): return True
    if isinstance(value, float) and math.isnan(value): return True
    return False


def run_case(name: str, text_dump: dict, exp_colour, exp_id):
    """
    Run a test case comparing expected vs actual classification result.

    For unknown cases (exp_colour=None), expects:
    - colour_id: 0 or None
    - colour: "Unknown", "", or None
    """
    try:
        r = classify(text_dump)
        got_colour, got_id = r.get("colour"), r.get("colour_id")

        if exp_colour is None:
            # Unknown expected: check ID=0/None and name="Unknown"/None/""
            passed = is_unknown(got_id) and is_unknown(got_colour)
            exp_str = "Unknown(0)"
            got_str = f"{got_colour}({got_id})"
        else:
            passed = (got_colour == exp_colour and got_id == exp_id)
            exp_str = f"{exp_colour}({exp_id})"
            got_str = f"{got_colour}({got_id})"

        input_short = text_dump.get("Title", "")[:40] or str(text_dump)[:40]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: '{input_short}' | exp={exp_str} got={got_str}")
        RESULTS.append((name, passed, None))
    except Exception as e:
        print(f"  [ERR ] {name}: {e}")
        RESULTS.append((name, False, str(e)))


# === TESTS ===
def test_unknown_no_info():
    print("TEST 1: Unknown - no info")
    run_case("empty", {"colour": "", "Title": ""}, None, None)
    run_case("generic", {"colour": "", "Title": "handbag"}, None, None)

def test_unknown_irrelevant():
    print("TEST 2: Unknown - irrelevant")
    run_case("no-hints", {"colour": "", "Title": "Designer leather crossbody excellent condition"}, None, None)
    run_case("vintage", {"colour": "", "Title": "Vintage authentic luxury tote bag with dust cover"}, None, None)

def test_simple_correct():
    print("TEST 3: Simple correct")
    run_case("black", {"colour": "Black", "Title": "Gucci Black Leather Soho Disco Crossbody"}, "Black", 1)
    run_case("red", {"colour": "Red", "Title": "Prada Red Saffiano Galleria Tote"}, "Red", 7)

def test_primary_secondary():
    print("TEST 4: Primary vs secondary (bicolor)")
    run_case("white", {"colour": "White", "Title": "White leather bag with black trim accents"}, "White", 4)
    run_case("neutrals", {"colour": "neutrals", "Title": "Chanel Beige and Black Bicolor Flap Bag"}, "Neutrals", 5)

def test_noise_extraction():
    print("TEST 5: Noise extraction")
    run_case("navy", {"colour": "Navy", "Title": "SUPER RARE 2020 Cruise Collection Runway Edition Celebrity Favorite Must-Have Statement Piece Navy Blue Shoulder Bag LIMITED - buy now 100% authentic used good condition, made in italy"}, "Blue", 11)
    run_case("pink", {"colour": "Pink", "Title": "authentic guaranteed 100% original with receipt certificate included dustbag box ribbon pink leather wallet gift ready. 100% authenic contanct me to buy"}, "Pink", 2)

def test_ground_truth_leprix():
    print("TEST GT: Leprix")
    run_case("leprix1", {"brand": "Gucci", "Title": "Balenciaga Medium Calfskin Hacker Project Jackie 1961 - very good condition", "Colour": "Black"}, "Black", 1)
    run_case("leprix2", {"brand": "Gucci", "Title": "Bicolor Calfskin Jackie 1961 Wallet On Chain - new with tags", "Colour": "White"}, "White", 4)
    run_case("leprix3", {"brand": "Gucci", "Title": "Tricolor Leather Soho Disco Crossbody AA"}, None, None)

def test_ground_truth_italian():
    print("TEST GT: Italian")
    run_case("ital1", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Nano Papillon Monogram Vintage"}, "Brown", 3)
    run_case("ital2", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - MANHATTAN GM Monogram"}, "Brown", 3)
    run_case("ital3", {"brand": "Prada", "Title": "PRADA - Occhiale da sole lente a specchio rosa"}, "Pink", 2)


if __name__ == "__main__":
    print(f"=== COLOUR TESTS (mode={MODE}, config={CONFIG_ID}) ===\n")
    test_unknown_no_info()
    test_unknown_irrelevant()
    test_simple_correct()
    test_primary_secondary()
    test_noise_extraction()
    test_ground_truth_leprix()
    test_ground_truth_italian()

    passed = [r for r in RESULTS if r[1]]
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n=== SUMMARY: {len(passed)}/{len(RESULTS)} passed ===")
    if failed:
        print("Failed:")
        for name, _, err in failed:
            print(f"  - {name}" + (f" ({err})" if err else ""))
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
