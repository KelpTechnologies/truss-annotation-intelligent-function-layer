#!/usr/bin/env python3
"""
Test: condition_tests - Condition classification tests (TEXT-ONLY)

Simulates Lambda routing: text-only -> classifier-condition-bags-text

Usage:
    python condition_tests.py --base-url https://... --api-key ...
    python condition_tests.py --mode local
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

parser = argparse.ArgumentParser(description="Condition classification tests")
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
CONFIG_ID = "classifier-condition-bags-text"


def classify_via_api(text_dump: dict) -> dict:
    """Classify via HTTP API - simulates Lambda endpoint."""
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required or set STAGING_API_URL in .env")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/condition",
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
    2. get_config_id_for_input_mode(base, "text-only") -> "classifier-condition-bags-text"
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
        "condition": result.get("primary_name"),
        "condition_id": result.get("primary_id"),
        "confidence": result.get("confidence", 0),
        "success": result.get("success", False),
        "reasoning": result.get("reasoning", "")
    }


def classify(text_dump: dict) -> dict:
    return classify_via_local(text_dump) if MODE == "local" else classify_via_api(text_dump)


def is_unknown(value):
    """
    Check if value represents 'unknown' classification result.

    Classifier returns ID 0 + name "Unknown" when condition can't be identified.
    - condition_id: 0 (int)
    - condition: "Unknown" (str)
    """
    if value is None: return True
    if value == 0: return True
    if isinstance(value, str) and value.lower() in ("unknown", "nan", ""): return True
    if isinstance(value, float) and math.isnan(value): return True
    return False


def run_case(name: str, text_dump: dict, exp_condition, exp_id):
    """
    Run a test case comparing expected vs actual classification result.

    For unknown cases (exp_condition=None), expects:
    - condition_id: 0 or None
    - condition: "Unknown", "", or None
    """
    try:
        r = classify(text_dump)
        got_condition, got_id = r.get("condition"), r.get("condition_id")

        if exp_condition is None:
            # Unknown expected: check ID=0/None and name="Unknown"/None/""
            passed = is_unknown(got_id) and is_unknown(got_condition)
            exp_str = "Unknown(0)"
            got_str = f"{got_condition}({got_id})"
        else:
            passed = (got_condition == exp_condition and got_id == exp_id)
            exp_str = f"{exp_condition}({exp_id})"
            got_str = f"{got_condition}({got_id})"

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
    run_case("empty", {"condition": "", "Title": ""}, None, None)
    run_case("generic", {"condition": "", "Title": "Louis Vuitton Neverfull"}, None, None)

def test_unknown_irrelevant():
    print("TEST 2: Unknown - irrelevant")
    run_case("no-hints", {"condition": "", "Title": "Beautiful vintage Chanel flap bag black caviar"}, None, None)
    run_case("hermes", {"condition": "", "Title": "Hermès Birkin 35 Togo Gold Hardware"}, None, None)

def test_simple_correct():
    print("TEST 3: Simple correct")
    run_case("excellent", {"condition": "", "Title": "Louis Vuitton Speedy 25 - excellent condition"}, "Used Excellent", 2)
    run_case("very-good", {"condition": "", "Title": "Coach Tabby Shoulder Bag very good condition"}, "Used Very Good", 3)
    run_case("good", {"condition": "", "Title": "Prada Galleria Saffiano - good condition"}, "Used Good", 4)

def test_brand_new():
    print("TEST 4: Brand new variations")
    run_case("bnwt", {"condition": "", "Title": "Chloé Marcie Bag - brand new with tags"}, "Brand New", 1)
    run_case("nwt", {"condition": "", "Title": "Hermès Kelly 28 NWT never used"}, "Brand New", 1)
    run_case("bnwt-sealed", {"condition": "", "Title": "Louis Vuitton Pochette BNWT sealed"}, "Brand New", 1)
    run_case("nib", {"condition": "", "Title": "Coach Crossbody new in box NIB"}, "Brand New", 1)

def test_japanese_grades():
    print("TEST 5: Japanese/Resale grades (AA, AB)")
    run_case("aa-1", {"condition": "", "Title": "Louis Vuitton Neverfull MM Monogram - AA"}, "Used Excellent", 2)
    run_case("ab-1", {"condition": "", "Title": "Hermès Evelyne III PM Clemence - AB"}, "Used Very Good", 3)
    run_case("aa-2", {"condition": "", "Title": "Chanel Classic Flap Medium Caviar grade AA"}, "Used Excellent", 2)
    run_case("ab-2", {"condition": "", "Title": "Gucci Marmont Camera Bag grade AB"}, "Used Very Good", 3)

def test_extended_grades():
    print("TEST 6: Extended grades (S, A, B, C)")
    run_case("grade-s", {"condition": "", "Title": "Prada Re-Edition 2005 Nylon - grade S"}, "Brand New", 1)
    run_case("grade-a", {"condition": "", "Title": "Balenciaga City Bag Lambskin - grade A"}, "Used Excellent", 2)
    run_case("grade-b", {"condition": "", "Title": "Celine Luggage Nano - grade B"}, "Used Very Good", 3)
    run_case("grade-c", {"condition": "", "Title": "Fendi Baguette - grade C"}, "Used Good", 4)

def test_descriptive_language():
    print("TEST 7: Descriptive language")
    run_case("mint", {"condition": "", "Title": "Dior Saddle Bag - mint condition like new"}, "Used Excellent", 2)
    run_case("minor-wear", {"condition": "", "Title": "Bottega Veneta Pouch - shows minor signs of wear"}, "Used Good", 4)
    run_case("visible-wear", {"condition": "", "Title": "Givenchy Antigona - visible scratches and wear on corners"}, "Used Fair", 5)
    run_case("heavy-wear", {"condition": "", "Title": "YSL Loulou - heavy wear needs restoration"}, "Used Poor", 6)

def test_condition_in_noise():
    print("TEST 8: Condition in noise")
    run_case("noise-excellent", {"condition": "", "Title": "RARE LIMITED 2020 Holiday Louis Vuitton Speedy Bandouliere 25 EXCELLENT CONDITION fast shipping A+++ seller authentic guaranteed"}, "Used Excellent", 2)
    run_case("noise-ab", {"condition": "", "Title": "100% authentic real Hermès Constance 24 AB very good great deal free returns trusted seller"}, "Used Very Good", 3)

def test_poor_fair():
    print("TEST 9: Poor/Fair indicators")
    run_case("fair", {"condition": "", "Title": "Chanel Boy Bag - fair condition visible patina"}, "Used Fair", 5)
    run_case("poor", {"condition": "", "Title": "Louis Vuitton Keepall 55 - poor condition for repair/parts"}, "Used Poor", 6)
    run_case("well-loved", {"condition": "", "Title": "Coach Legacy Bag well loved showing age"}, "Used Fair", 5)

def test_ground_truth_leprix():
    print("TEST GT: Leprix")
    run_case("leprix1", {"brand": "Gucci", "Title": "Balenciaga Medium Calfskin Hacker Project Jackie 1961 - very good condition"}, "Used Very Good", 3)
    run_case("leprix2", {"brand": "Gucci", "Title": "Bicolor Calfskin Jackie 1961 Wallet On Chain - new with tags"}, "Brand New", 1)
    run_case("leprix3", {"brand": "Prada", "Title": "Tessuto Zip Top Crossbody - AB"}, "Used Very Good", 3)

def test_ground_truth_italian():
    print("TEST GT: Italian")
    run_case("ital1", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Nano Papillon Monogram Vintage", "Tags": "borsa ottime condizioni, borsa second-hand, borsa usata"}, "Used Excellent", 2)
    run_case("ital2", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - MANHATTAN GM Monogram", "Tags": "borsa ottime condizioni, borsa second-hand, borsa usata"}, "Used Excellent", 2)
    run_case("ital3", {"brand": "Hermès", "Title": "HERMES - Birkin 30 Togo Gold - AA", "Tags": "borsa di lusso, condizioni eccellenti, Hermes, birkin"}, "Used Excellent", 2)

def test_ground_truth_grades():
    print("TEST GT: Grade codes")
    run_case("grade1", {"brand": "Chanel", "Title": "Chanel Classic Double Flap Medium - AA grade pristine"}, "Used Excellent", 2)
    run_case("grade2", {"brand": "Hermès", "Title": "Hermes Picotin Lock 18 Clemence - AB some light marks"}, "Used Very Good", 3)
    run_case("grade3", {"brand": "Louis Vuitton", "Title": "Louis Vuitton Keepall 50 Monogram - grade B normal wear"}, "Used Very Good", 3)
    run_case("grade4", {"brand": "Gucci", "Title": "Gucci Dionysus Small GG Supreme - S rank unused"}, "Brand New", 1)


if __name__ == "__main__":
    print(f"=== CONDITION TESTS (mode={MODE}, config={CONFIG_ID}) ===\n")
    test_unknown_no_info()
    test_unknown_irrelevant()
    test_simple_correct()
    test_brand_new()
    test_japanese_grades()
    test_extended_grades()
    test_descriptive_language()
    test_condition_in_noise()
    test_poor_fair()
    test_ground_truth_leprix()
    test_ground_truth_italian()
    test_ground_truth_grades()

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
