#!/usr/bin/env python3
"""
Test: model_test - Model classification tests (TEXT-ONLY)

Simulates Lambda routing for model (brand-specific):
  text-only -> classifier-model-bags-{brand}-full-taxo

Usage:
    python model_test.py --base-url https://... --api-key ...
    python model_test.py --mode local
"""

import argparse
import sys
import math
import json
import re
import unicodedata
import os
from pathlib import Path

# Load .env from tests folder
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    if key and key not in os.environ:  # Don't override existing
                        os.environ[key] = value

load_env()

parser = argparse.ArgumentParser(description="Model classification tests")
parser.add_argument("--mode", "-m", default="api", choices=["api", "local"], help="Test mode")
parser.add_argument("--base-url", "-u", help="API base URL (or set STAGING_API_URL)")
parser.add_argument("--api-key", "-k", help="API key (or set DSL_API_KEY)")
args = parser.parse_args()

MODE = args.mode
BASE_URL = args.base_url or os.environ.get("STAGING_API_URL") or os.environ.get("API_BASE_URL")
BASE_URL = BASE_URL.rstrip("/") if BASE_URL else None
API_KEY = args.api_key or os.environ.get("DSL_API_KEY") or os.environ.get("API_KEY") or ""
RESULTS = []


def safe_mapping(brand: str) -> str:
    """
    Normalize brand name to config-compatible format.
    (Copied from classifier_model_orchestration.py)
    """
    normalized = unicodedata.normalize('NFKD', brand)
    ascii_str = normalized.encode('ASCII', 'ignore').decode('ASCII')
    cleaned = re.sub(r'[^a-zA-Z0-9]+', '', ascii_str)
    return cleaned.lower()


def get_model_config_id(brand: str) -> str:
    """Get config_id for brand-specific model classifier."""
    safe_brand = safe_mapping(brand)
    return f"classifier-model-bags-{safe_brand}-full-taxo"


def classify_via_api(text_dump: dict) -> dict:
    """Classify via HTTP API - simulates Lambda endpoint."""
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required or set STAGING_API_URL in .env")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/model",
        json={"text_dump": text_dump, "input_mode": "text-only"},
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    return response.json()["data"][0]


def classify_via_local(text_dump: dict) -> dict:
    """
    Classify via direct Python - simulates Lambda routing.

    Lambda routing logic for model (from agent_orchestration_api_handler.py):
    1. detect_input_mode() -> "text-only" (no image, has text)
    2. execute_model_classification_for_api() uses get_model_config_id(brand)
    3. Config: classifier-model-bags-{brand}-full-taxo
    """
    repo_root = Path(__file__).parent.parent
    service_path = repo_root / "services" / "automated-annotation"
    sys.path.insert(0, str(service_path))

    # Ensure GCP credentials are set up
    from core.utils.credentials import ensure_gcp_adc
    ensure_gcp_adc()

    from agent_orchestration.classifier_api_orchestration import classify_for_api

    brand = text_dump.get("brand", "")
    if not brand or not str(brand).strip():
        return {"model": None, "model_id": None, "root_model": None, "root_model_id": None}

    config_id = get_model_config_id(brand)

    # Format text_dump as JSON string (matches Lambda behavior)
    text_input = json.dumps(text_dump, indent=2)

    # Call orchestration function directly (simulates Lambda)
    try:
        result = classify_for_api(
            config_id=config_id,
            text_input=text_input,
            input_mode="text-only",
            env="staging"
        )

        return {
            "model": result.get("primary_name"),
            "model_id": result.get("primary_id"),
            "root_model": result.get("root_model_name"),
            "root_model_id": result.get("root_model_id"),
            "confidence": result.get("confidence", 0),
            "success": result.get("success", False),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        # Config not found for brand
        return {"model": None, "model_id": None, "root_model": None, "root_model_id": None, "error": str(e)}


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

    Classifier returns ID 0 + name "Unknown" when model can't be identified.
    - model_id: 0 (int)
    - model: "Unknown" (str)
    - root_model_id: None
    - root_model: None
    """
    if value is None: return True
    if value == 0: return True
    if isinstance(value, str) and value.lower() in ("unknown", "nan", ""): return True
    if isinstance(value, float) and math.isnan(value): return True
    return False


def run_case(name: str, text_dump: dict, exp_model, exp_id, exp_root=None, exp_root_id=None):
    """
    Run a test case comparing expected vs actual classification result.

    For unknown cases (exp_model=None), expects:
    - model_id: 0 or None
    - model: "Unknown", "", or None
    - root_model_id: None
    - root_model: None
    """
    try:
        r = classify(text_dump)
        got_model, got_id = r.get("model"), r.get("model_id")
        got_root, got_root_id = r.get("root_model"), r.get("root_model_id")

        if exp_model is None:
            # Unknown expected: check ID=0/None and name="Unknown"/None/""
            passed = is_unknown(got_id) and is_unknown(got_model)
            exp_str = "Unknown(0)->None"
            got_str = f"{got_model}({got_id})->{got_root}({got_root_id})"
        else:
            passed = (got_model == exp_model and got_id == exp_id)
            if exp_root and passed:
                passed = (got_root == exp_root and got_root_id == exp_root_id)
            exp_str = f"{exp_model}({exp_id})" + (f"->{exp_root}({exp_root_id})" if exp_root else "")
            got_str = f"{got_model}({got_id})->{got_root}({got_root_id})"

        input_short = text_dump.get("Title", "")[:40] or str(text_dump)[:40]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: '{input_short}' | exp={exp_str} got={got_str}")
        RESULTS.append((name, passed, None))
    except Exception as e:
        print(f"  [ERR ] {name}: {e}")
        RESULTS.append((name, False, str(e)))


def run_root_check(name: str, text_dump: dict, allowed_roots: list):
    try:
        r = classify(text_dump)
        got_root = r.get("root_model")
        passed = got_root in allowed_roots
        input_short = text_dump.get("Title", "")[:40] or str(text_dump)[:40]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: '{input_short}' | exp_root in {allowed_roots} got={got_root}")
        RESULTS.append((name, passed, None))
    except Exception as e:
        print(f"  [ERR ] {name}: {e}")
        RESULTS.append((name, False, str(e)))


# === TESTS ===
def test_unknown_no_info():
    print("TEST 1: Unknown - no info")
    run_case("empty", {"brand": "Gucci", "Title": ""}, None, None)
    run_case("generic", {"brand": "Prada", "Title": "leather bag"}, None, None)

def test_unknown_irrelevant():
    print("TEST 2: Unknown - irrelevant")
    run_case("no-hints", {"brand": "Gucci", "Title": "Gucci handbag excellent condition authentic"}, None, None)
    run_case("generic-lv", {"brand": "Louis Vuitton", "Title": "Louis Vuitton tote bag monogram canvas"}, None, None)

def test_simple_correct():
    print("TEST 3: Simple correct")
    run_case("soho", {"brand": "Gucci", "Title": "Gucci Soho Disco Crossbody Black Leather"}, "Soho Disco", 1056, "Soho Disco", 1056)
    run_case("neverfull", {"brand": "Louis Vuitton", "Title": "Louis Vuitton Neverfull MM Monogram"}, "Neverfull", 445, "Neverfull", 445)

def test_primary_secondary():
    print("TEST 4: Model variants")
    run_case("jackie1961", {"brand": "Gucci", "Title": "Calfskin Jackie 1961 Small Shoulder Bag"}, "Jackie 1961", 307, "Jackie", 306)
    run_case("bamboo", {"brand": "Gucci", "Title": "Bamboo Night Clutch Black Calfskin"}, "Bamboo", 50, "Bamboo", 50)

def test_noise_extraction():
    print("TEST 5: Noise extraction")
    run_case("reedition", {"brand": "Prada", "Title": "AUTHENTIC GUARANTEED Fast Shipping A+++ Seller Rated 5 Stars Re-Edition 2005 Nylon Mini Top Handle RARE COLOR"}, "Re-Edition 2005", 524, "Re-Edition", 526)
    run_root_check("classicflap", {"brand": "Chanel", "Title": "limited edition cruise 2019 collection exclusive VIP client gift classic flap bag medium caviar leather gold hardware"}, ["Timeless/Classic", "Classic Flap", "Classic Flap / Classic 11.12"])

def test_ground_truth_leprix():
    print("TEST GT: Leprix")
    run_case("leprix1", {"brand": "Gucci", "Title": "Balenciaga Medium Calfskin Hacker Project Jackie 1961 - very good condition"}, "Jackie 1961", 307, "Jackie", 306)
    run_case("leprix2", {"brand": "Gucci", "Title": "Calfskin Bamboo Jackie Hobo - Used Excellent"}, "Jackie", 306, "Jackie", 306)
    run_case("leprix3", {"brand": "Gucci", "Title": "Tricolor Leather Soho Disco Crossbody AA"}, "Soho Disco", 1056, "Soho Disco", 1056)
    run_case("leprix4", {"brand": "Prada", "Title": "Tessuto Zip Top Crossbody - AB"}, None, None)

def test_ground_truth_italian():
    print("TEST GT: Italian")
    run_case("ital1", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Nano Papillon Monogram Vintage"}, "Papillon", 483, "Papillon", 483)
    run_case("ital2", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - MANHATTAN GM Monogram"}, "Manhattan", 397, "Manhattan", 397)
    run_case("ital3", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Portafoglio Murakami Clip"}, None, None)


if __name__ == "__main__":
    print(f"=== MODEL TESTS (mode={MODE}) ===\n")
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
