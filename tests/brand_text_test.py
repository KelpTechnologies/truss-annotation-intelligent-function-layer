#!/usr/bin/env python3
"""
Test: brand_text_test - Brand classification tests (TEXT-ONLY)

Simulates Lambda routing: text-only -> classifier-brand-bags-text

Usage:
    python brand_text_test.py --base-url https://... --api-key ...
    python brand_text_test.py --mode local
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

parser = argparse.ArgumentParser(description="Brand classification tests")
parser.add_argument("--mode", "-m", default="api", choices=["api", "local"], help="Test mode")
parser.add_argument("--base-url", "-u", help="API base URL (or set STAGING_API_URL)")
parser.add_argument("--api-key", "-k", help="API key (or set DSL_API_KEY)")
args = parser.parse_args()

MODE = args.mode
BASE_URL = args.base_url or os.environ.get("STAGING_API_URL") or os.environ.get("API_BASE_URL")
BASE_URL = BASE_URL.rstrip("/") if BASE_URL else None
API_KEY = args.api_key or os.environ.get("DSL_API_KEY") or os.environ.get("API_KEY") or ""
RESULTS = []

# Brand uses two-agent workflow: brand-extraction-v1 + brand-classification-v1


def classify_via_api(text_dump: dict) -> dict:
    """Classify via HTTP API - simulates Lambda endpoint."""
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required or set STAGING_API_URL in .env")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/brand",
        json={"text_dump": text_dump, "input_mode": "text-only"},
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    return response.json()["data"][0]


def classify_via_local(text_dump: dict) -> dict:
    """
    Classify via direct Python - uses two-agent brand classification workflow.

    Brand classification uses:
    1. Agent 1 (brand-extraction-v1): Extracts brand candidates, searches BigQuery
    2. Agent 2 (brand-classification-v1): Classifies which brand applies
    """
    repo_root = Path(__file__).parent.parent
    service_path = repo_root / "services" / "automated-annotation"
    sys.path.insert(0, str(service_path))

    # Ensure GCP credentials are set up
    from core.utils.credentials import ensure_gcp_adc
    ensure_gcp_adc()

    from agent_orchestration.brand_classification_orchestration import run_brand_classification_workflow

    # Build raw text from text_dump fields
    raw_text = json.dumps(text_dump, indent=2)

    # Run two-agent workflow
    result = run_brand_classification_workflow(
        raw_text=raw_text,
        name=text_dump.get("Title"),
        env="staging",
        verbose=False
    )

    # Map workflow result to expected format
    if result.get("workflow_status") == "success":
        return {
            "brand": result.get("final_brand"),
            "brand_id": result.get("final_brand_id"),
            "confidence": result.get("confidence", 0),
            "success": True,
            "reasoning": result.get("agent2_result", {}).get("reasoning", "")
        }
    else:
        return {
            "brand": "Unknown",
            "brand_id": 0,
            "confidence": 0,
            "success": False,
            "reasoning": result.get("error", "Brand classification failed")
        }


def classify(text_dump: dict) -> dict:
    return classify_via_local(text_dump) if MODE == "local" else classify_via_api(text_dump)


def is_unknown(value):
    """
    Check if value represents 'unknown' classification result.

    Classifier returns ID 0 + name "Unknown" when brand can't be identified.
    - brand_id: 0 (int)
    - brand: "Unknown" (str)
    """
    if value is None: return True
    if value == 0: return True
    if isinstance(value, str) and value.lower() in ("unknown", "nan", ""): return True
    if isinstance(value, float) and math.isnan(value): return True
    return False


def run_case(name: str, text_dump: dict, exp_brand, exp_id):
    """
    Run a test case comparing expected vs actual classification result.

    For unknown cases (exp_brand=None), expects:
    - brand_id: 0 or None
    - brand: "Unknown", "", or None
    """
    try:
        r = classify(text_dump)
        got_brand, got_id = r.get("brand"), r.get("brand_id")

        if exp_brand is None:
            passed = is_unknown(got_id) and is_unknown(got_brand)
            exp_str = "Unknown(0)"
        else:
            passed = (got_brand == exp_brand and got_id == exp_id)
            exp_str = f"{exp_brand}({exp_id})"

        got_str = f"{got_brand}({got_id})"
        RESULTS.append({"name": name, "passed": passed, "expected": exp_str, "actual": got_str, "error": None})
    except Exception as e:
        RESULTS.append({"name": name, "passed": False, "expected": f"{exp_brand}({exp_id})", "actual": None, "error": str(e)})


# === TESTS ===
def test_unknown_no_info():
    run_case("empty", {"brand": "", "Title": ""}, None, None)
    run_case("generic", {"brand": "", "Title": "bag"}, None, None)

def test_unknown_irrelevant():
    run_case("no-hints", {"brand": "", "Title": "Beautiful vintage leather handbag great condition"}, None, None)
    run_case("designer", {"brand": "", "Title": "Designer crossbody purse authentic luxury item"}, None, None)

def test_simple_correct():
    run_case("lv", {"brand": "Louis Vuitton", "Title": "Louis Vuitton Neverfull MM Monogram Canvas"}, "Louis Vuitton", 4210)
    run_case("coach", {"brand": "Coach", "Title": "Coach Tabby Shoulder Bag in Signature Canvas"}, "Coach", 1555)

def test_brand_mismatch():
    run_case("hermes", {"brand": "Unknown", "Title": "Hermès Birkin 35 Togo Leather Gold Hardware"}, "Hermès", 3074)
    run_case("chloe", {"brand": "", "Title": "Chloé Marcie Medium Saddle Bag Tan Calfskin"}, "Chloé", 1462)

def test_without_accents():
    run_case("hermes-no-accent", {"brand": "", "Title": "Hermes Kelly 28 Epsom Leather Palladium Hardware"}, "Hermès", 3074)
    run_case("chloe-no-accent", {"brand": "", "Title": "Chloe Woody Medium Tote Canvas and Leather"}, "Chloé", 1462)

def test_with_accents():
    run_case("hermes-accent", {"brand": "", "Title": "Hermès Constance 24 Evercolor Leather"}, "Hermès", 3074)
    run_case("chloe-accent", {"brand": "", "Title": "Chloé Faye Small Shoulder Bag Motty Grey"}, "Chloé", 1462)

def test_brand_in_noise():
    run_case("lv-noise", {"brand": "", "Title": "RARE 2019 Limited Edition Holiday Collection Exclusive VIP Gift Set Premium Quality Louis Vuitton Speedy 25 Excellent Condition Fast Ship"}, "Louis Vuitton", 4210)
    run_case("coach-noise", {"brand": "", "Title": "authentic guaranteed 100% real deal fast shipping free returns great seller A+++ Coach Chelsea Crossbody brand new with tags NWT"}, "Coach", 1555)

def test_abbreviations():
    run_case("lv-abbrev", {"brand": "", "Title": "LV Pochette Accessoires Monogram"}, "Louis Vuitton", 4210)
    run_case("hermes-caps", {"brand": "", "Title": "HERMES Garden Party 36 Negonda Leather"}, "Hermès", 3074)

def test_ground_truth_leprix():
    run_case("leprix1", {"brand": "Louis Vuitton", "Title": "Louis Vuitton Neverfull GM Damier Ebene - excellent condition"}, "Louis Vuitton", 4210)
    run_case("leprix2", {"brand": "Hermès", "Title": "Hermes Evelyne III PM Clemence Gold - very good condition"}, "Hermès", 3074)
    run_case("leprix3", {"brand": "Coach", "Title": "Coach Willow Tote Colorblock Signature Canvas - new with tags"}, "Coach", 1555)

def test_ground_truth_italian():
    run_case("ital1", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Nano Papillon Monogram Vintage", "Tags": "bauletto, borsa di lusso, Louis vuitton, Monogram, nano, Nanopapillon, Papillon"}, "Louis Vuitton", 4210)
    run_case("ital2", {"brand": "Chloé", "Title": "CHLOE - Marcie Small Crossbody Tan", "Tags": "borsa di lusso, borsa pelle, Chloe, crossbody, marcie, tan"}, "Chloé", 1462)
    run_case("ital3", {"brand": "Hermès", "Title": "HERMES - Picotin Lock 18 Clemence Noir", "Tags": "borsa di lusso, borsa pelle, Hermes, picotin, picotin lock, noir"}, "Hermès", 3074)


def print_summary():
    """Print final test summary with detailed failure info."""
    passed = [r for r in RESULTS if r["passed"]]
    failed = [r for r in RESULTS if not r["passed"]]

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {len(passed)}/{len(RESULTS)} passed")
    print(f"{'='*60}")

    if failed:
        print("\nFailed tests:")
        print("-" * 60)
        for f in failed:
            print(f"  test_ref: {f['name']}")
            print(f"  expected: {f['expected']}")
            print(f"  actual:   {f['actual']}")
            if f['error']:
                print(f"  error:    {f['error']}")
            print("-" * 60)

    if passed:
        print("\nPassed tests:")
        for p in passed:
            print(f"  ✓ {p['name']}")

    return len(failed) == 0


if __name__ == "__main__":
    print(f"=== BRAND TESTS (mode={MODE}, workflow=brand-extraction-v1+brand-classification-v1) ===\n")
    test_unknown_no_info()
    test_unknown_irrelevant()
    test_simple_correct()
    test_brand_mismatch()
    test_without_accents()
    test_with_accents()
    test_brand_in_noise()
    test_abbreviations()
    test_ground_truth_leprix()
    test_ground_truth_italian()

    all_passed = print_summary()

    if all_passed:
        print("\nPASSED")
        sys.exit(0)
    else:
        print("\nFAILED")
        sys.exit(1)
