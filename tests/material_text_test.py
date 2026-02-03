#!/usr/bin/env python3
"""
Test: material_test - Material classification tests (TEXT-ONLY)

Simulates Lambda routing: text-only -> classifier-material-bags-text

Usage:
    python material_text_test.py --base-url https://... --api-key ...
    python material_text_test.py --mode local

Environment variables (loaded from .env):
    API mode:   STAGING_API_URL, DSL_API_KEY
    Local mode: AWS_REGION, TRUSS_SECRETS_ARN, VERTEXAI_PROJECT, VERTEXAI_LOCATION
"""

import argparse
import sys
import os
import math
import json
from pathlib import Path

# Load .env at initialization
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Set cross-platform temp path for GCP credentials (before lambda imports)
import tempfile
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(tempfile.gettempdir(), "gcp_sa.json")

parser = argparse.ArgumentParser(description="Material classification tests")
parser.add_argument("--mode", "-m", default="api", choices=["api", "local"], help="Test mode")
parser.add_argument("--base-url", "-u", help="API base URL (or set STAGING_API_URL)")
parser.add_argument("--api-key", "-k", help="API key (or set DSL_API_KEY)")
args = parser.parse_args()

MODE = args.mode
BASE_URL = args.base_url or os.getenv("STAGING_API_URL")
BASE_URL = BASE_URL.rstrip("/") if BASE_URL else None
API_KEY = args.api_key or os.getenv("DSL_API_KEY") or ""
RESULTS = []

# === ENV VALIDATION ===
def validate_env():
    """Validate required env vars based on mode. Exit early if missing."""
    missing = []

    if MODE == "api":
        if not BASE_URL:
            missing.append("STAGING_API_URL (or --base-url)")
    else:  # local mode
        required_local = [
            ("AWS_REGION", os.getenv("AWS_REGION")),
            ("TRUSS_SECRETS_ARN", os.getenv("TRUSS_SECRETS_ARN")),
            ("VERTEXAI_PROJECT", os.getenv("VERTEXAI_PROJECT")),
            ("VERTEXAI_LOCATION", os.getenv("VERTEXAI_LOCATION")),
        ]
        for name, val in required_local:
            if not val:
                missing.append(name)

    if missing:
        print(f"ERROR: Missing required environment variables for {MODE} mode:")
        for var in missing:
            print(f"  - {var}")
        print(f"\nCopy tests/.env.example to tests/.env and fill in values.")
        sys.exit(1)

validate_env()

# Text-only config (matches Lambda routing: text-only -> classifier-{prop}-bags-text)
CONFIG_ID = "classifier-material-bags-text"


def classify_via_api(text_dump: dict) -> dict:
    """Classify via HTTP API - simulates Lambda endpoint."""
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required for api mode")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/material",
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
    2. get_config_id_for_input_mode(base, "text-only") -> "classifier-material-bags-text"
    3. classify_for_api() loads config and runs agent
    """
    repo_root = Path(__file__).parent.parent
    service_path = repo_root / "services" / "automated-annotation"
    sys.path.insert(0, str(service_path))

    # Initialize GCP credentials from AWS Secrets Manager
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
        "material": result.get("primary_name"),
        "material_id": result.get("primary_id"),
        "root_material": result.get("root_material_name"),
        "root_material_id": result.get("root_material_id"),
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

    Classifier returns ID 0 + name "Unknown" when material can't be identified.
    - material_id: 0 (int)
    - material: "Unknown" (str)
    - root_material_id: None
    - root_material: None
    """
    if value is None: return True
    if value == 0: return True
    if isinstance(value, str) and value.lower() in ("unknown", "nan", ""): return True
    if isinstance(value, float) and math.isnan(value): return True
    return False


def run_case(name: str, text_dump: dict, exp_mat, exp_id, exp_root, exp_root_id):
    """
    Run a test case comparing expected vs actual classification result.

    For unknown cases (exp_mat=None), expects:
    - material_id: 0 or None
    - material: "Unknown", "", or None
    - root_material_id: None
    - root_material: None
    """
    try:
        r = classify(text_dump)
        got_mat, got_id = r.get("material"), r.get("material_id")
        got_root, got_root_id = r.get("root_material"), r.get("root_material_id")

        if exp_mat is None:
            # Unknown expected: check ID=0/None and name="Unknown"/None/""
            passed = is_unknown(got_id) and is_unknown(got_mat)
            exp_str = "Unknown(0)->None"
            got_str = f"{got_mat}({got_id})->{got_root}({got_root_id})"
        else:
            passed = (got_mat == exp_mat and got_id == exp_id and got_root == exp_root and got_root_id == exp_root_id)
            exp_str = f"{exp_mat}({exp_id})->{exp_root}({exp_root_id})"
            got_str = f"{got_mat}({got_id})->{got_root}({got_root_id})"

        input_short = text_dump.get("Title", "")[:40] or str(text_dump)[:40]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: '{input_short}' | exp={exp_str} got={got_str}")
        RESULTS.append((name, passed, None))
    except Exception as e:
        print(f"  [ERR ] {name}: {e}")
        RESULTS.append((name, False, str(e)))


def run_root_only(name: str, text_dump: dict, exp_root, exp_root_id):
    try:
        r = classify(text_dump)
        got_root, got_root_id = r.get("root_material"), r.get("root_material_id")
        passed = (got_root == exp_root)
        input_short = text_dump.get("Title", "")[:40] or str(text_dump)[:40]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: '{input_short}' | exp_root={exp_root} got={got_root}")
        RESULTS.append((name, passed, None))
    except Exception as e:
        print(f"  [ERR ] {name}: {e}")
        RESULTS.append((name, False, str(e)))


# === TESTS ===
def test_unknown_no_info():
    print("TEST 1: Unknown - no info")
    run_case("empty", {"material": "", "Title": ""}, None, None, None, None)
    run_case("generic", {"material": "", "Title": "item"}, None, None, None, None)

def test_unknown_irrelevant():
    print("TEST 2: Unknown - irrelevant")
    run_case("no-hints", {"material": "", "Title": "Beautiful vintage accessory great condition"}, None, None, None, None)
    run_case("gift-text", {"material": "", "Title": "Perfect gift for her birthday present idea"}, None, None, None, None)

def test_simple_correct():
    print("TEST 3: Simple correct")
    run_case("canvas", {"material": "Canvas", "Title": "Louis Vuitton Monogram Canvas Neverfull"}, "Canvas", 2, "Canvas", 2)
    run_case("calfskin", {"material": "Calfskin", "Title": "Crossbody Louis Vuitton Neverfull Calfskin"}, "Calfskin", 47, "Leather", 1)

def test_primary_secondary():
    print("TEST 4: Primary vs secondary")
    run_case("calfskin-trim", {"material": "Calfskin", "Title": "Canvas tote bag with calfskin leather trim"}, "Calfskin", 47, "Leather", 1)
    run_case("patent", {"material": "Patent Leather", "Title": "Patent leather clutch with Cotton interior lining"}, "Patent Leather", 10, "Leather", 1)

def test_noise_extraction():
    print("TEST 5: Noise extraction")
    run_case("nylon", {"material": "Nylon", "Title": "RARE 2019 Limited Edition Holiday Collection Exclusive VIP Gift Set Premium Quality Designer Fashion Accessory Nylon Crossbody Excellent Condition"}, "Nylon", 5, "Nylon", 5)
    run_root_only("saffiano", {"material": "Saffiano Leather", "Title": "authentic guaranteed 100% real deal fast shipping free returns great seller A+++ saffiano leather wallet brand new with tags"}, "Leather", 1)

def test_ground_truth_leprix():
    print("TEST GT: Leprix")
    run_root_only("leprix1", {"brand": "Gucci", "Title": "Balenciaga Medium Calfskin Hacker Project Jackie 1961 - very good condition"}, "Leather", 1)
    run_root_only("leprix2", {"brand": "Gucci", "Title": "Bicolor Calfskin Jackie 1961 Wallet On Chain - new with tags"}, "Leather", 1)
    run_root_only("leprix3", {"brand": "Prada", "Title": "Tessuto Zip Top Crossbody - AB"}, "Leather", 1)

def test_ground_truth_italian():
    print("TEST GT: Italian")
    run_root_only("ital1", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Nano Papillon Monogram Vintage", "Tags": "bauletto, borsa di lusso, borsa marrone, borsa ottime condizioni, borsa second-hand, borsa tela, borsa usata, Louis vuitton, Monogram, nano, Nanopapillon, Papillon, papillon monogram, Papillon vintage"}, "Canvas", 2)
    run_root_only("ital2", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - MANHATTAN GM Monogram", "Tags": "bauletto, borsa di lusso, borsa marrone, borsa ottime condizioni, Borsa pelle, borsa second-hand, borsa tela, borsa usata, Louis vuitton, manhattan gm, Monogram"}, "Leather", 1)
    run_root_only("ital3", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Portafoglio Murakami Clip", "Tags": "limited-edition, Louis vuitton, Monogram, Multicolor Monogram, Murakami, Murakami Monogram, Portafoglio, Portafoglio Clip, portafoglio di lusso, takeshi Murakami, unisex"}, "Leather", 1)


if __name__ == "__main__":
    print(f"=== MATERIAL TESTS (mode={MODE}, config={CONFIG_ID}) ===\n")
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
