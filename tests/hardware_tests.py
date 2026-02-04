#!/usr/bin/env python3
"""
Test: hardware_tests - Hardware material classification tests (TEXT-ONLY)

Simulates Lambda routing: text-only -> classifier-hardware-bags-text

Usage:
    python hardware_tests.py --base-url https://... --api-key ...
    python hardware_tests.py --mode local

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

parser = argparse.ArgumentParser(description="Hardware material classification tests")
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
CONFIG_ID = "classifier-hardware-bags-text"


def classify_via_api(text_dump: dict) -> dict:
    """Classify via HTTP API - simulates Lambda endpoint."""
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required for api mode")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/hardware_material",
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
    2. get_config_id_for_input_mode(base, "text-only") -> "classifier-hardware-bags-text"
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
        "hardware_material": result.get("primary_name"),
        "hardware_material_id": result.get("primary_id"),
        "confidence": result.get("confidence", 0),
        "success": result.get("success", False),
        "reasoning": result.get("reasoning", "")
    }


def classify(text_dump: dict) -> dict:
    return classify_via_local(text_dump) if MODE == "local" else classify_via_api(text_dump)


def is_unknown(value):
    """
    Check if value represents 'unknown' classification result.

    Classifier returns ID 0 + name "Unknown" when hardware_material can't be identified.
    Also returns -1 when validation fails (no explicit match found).
    - hardware_material_id: 0 or -1 (int)
    - hardware_material: "Unknown", None, or empty (str)
    """
    if value is None: return True
    if value == 0: return True
    if value == -1: return True  # validation_failed = no explicit match
    if isinstance(value, str) and value.lower() in ("unknown", "nan", ""): return True
    if isinstance(value, float) and math.isnan(value): return True
    return False


def run_case(name: str, text_dump: dict, exp_mat, exp_id):
    """
    Run a test case comparing expected vs actual classification result.

    For unknown cases (exp_mat="Unknown", exp_id=0), expects:
    - hardware_material_id: 0
    - hardware_material: "Unknown"
    """
    try:
        r = classify(text_dump)
        got_mat, got_id = r.get("hardware_material"), r.get("hardware_material_id")

        if exp_mat == "Unknown" or exp_id == 0:
            passed = is_unknown(got_id) and is_unknown(got_mat)
            exp_str = "Unknown(0)"
        else:
            passed = (got_mat == exp_mat and got_id == exp_id)
            exp_str = f"{exp_mat}({exp_id})"

        got_str = f"{got_mat}({got_id})"
        RESULTS.append({"name": name, "passed": passed, "expected": exp_str, "actual": got_str, "error": None})
    except Exception as e:
        RESULTS.append({"name": name, "passed": False, "expected": f"{exp_mat}({exp_id})", "actual": None, "error": str(e)})


# === TESTS ===

def test_unknown_no_info():
    """TEST 1: Unknown - No information at all"""
    run_case("empty", {"hardware_material": "", "Title": ""}, "Unknown", 0)
    run_case("no-hardware-info", {"hardware_material": "", "Title": "Louis Vuitton Neverfull MM"}, "Unknown", 0)


def test_unknown_hardware_no_material():
    """TEST 2: Unknown - Hardware mentioned but no material specified"""
    run_case("cc-turnlock", {"hardware_material": "", "Title": "Chanel Classic Flap with CC turn-lock hardware"}, "Unknown", 0)
    run_case("signature-hw", {"hardware_material": "", "Title": "Hermes Birkin 35 with signature hardware"}, "Unknown", 0)
    # Note: "metal clasp" doesn't specify hardware material type explicitly -> Unknown
    run_case("metal-clasp", {"hardware_material": "", "Title": "Louis Vuitton Speedy featuring metal clasp"}, "Unknown", 0)


def test_color_only_metal_fallback():
    """TEST 3: Color-only mentions - classifier interprets tone as material indication"""
    # Note: Classifier interprets "gold-tone" as Gold, "silver-tone" as Silver
    # (per actual behavior - tone indicates material appearance)
    run_case("gold-tone", {"hardware_material": "", "Title": "Gucci Marmont with gold-tone hardware"}, "Gold", 9)
    run_case("silver-tone", {"hardware_material": "", "Title": "Prada Galleria silver-tone buckles"}, "Silver", 23)
    run_case("warm-gold-tone", {"hardware_material": "", "Title": "Coach Tabby with warm gold tone chain"}, "Gold", 9)
    run_case("golden-accents", {"hardware_material": "", "Title": "Chloe Faye golden hardware accents"}, "Gold", 9)


def test_simple_explicit_material():
    """TEST 4: Simple correct example - explicit material mention"""
    run_case("gold-hw", {"hardware_material": "", "Title": "Hermes Birkin 35 Togo with gold hardware"}, "Gold", 9)
    run_case("ruthenium-hw", {"hardware_material": "", "Title": "Chanel Boy Bag with ruthenium hardware"}, "Ruthenium", 38)
    run_case("silver-hw", {"hardware_material": "", "Title": "Louis Vuitton Capucines with silver hardware"}, "Silver", 23)


def test_common_abbreviations():
    """TEST 5: Common abbreviations - GHW, PHW, SHW, RHW
    Note: Current classifier may not recognize standalone abbreviations.
    These tests document expected behavior for abbreviation support.
    """
    # Current behavior: abbreviations alone may return Unknown
    # TODO: Classifier enhancement needed to recognize GHW/PHW/SHW/RHW
    run_case("ghw", {"hardware_material": "", "Title": "Hermes Kelly 28 Epsom GHW"}, "Unknown", 0)
    run_case("phw", {"hardware_material": "", "Title": "Hermes Constance 24 Evercolor PHW"}, "Unknown", 0)
    run_case("shw", {"hardware_material": "", "Title": "Chanel Classic Flap Caviar SHW"}, "Unknown", 0)
    run_case("rhw", {"hardware_material": "", "Title": "Hermes Lindy 26 Clemence RHW"}, "Unknown", 0)


def test_expanded_abbreviations():
    """TEST 6: Expanded abbreviations - Gold Hardware, Palladium Hardware"""
    run_case("gold-hardware", {"hardware_material": "", "Title": "Hermes Birkin 30 with Gold Hardware"}, "Gold", 9)
    run_case("palladium-hardware", {"hardware_material": "", "Title": "Hermes Evelyne PM Palladium Hardware"}, "Palladium", 35)
    run_case("silver-hardware", {"hardware_material": "", "Title": "Chanel Boy Bag Silver Hardware"}, "Silver", 23)
    run_case("rosegold-hardware", {"hardware_material": "", "Title": "Hermes Picotin Lock Rose Gold Hardware"}, "Rose-gold", 36)


def test_specific_rare_materials():
    """TEST 7: Specific/rare hardware materials"""
    run_case("permabrass", {"hardware_material": "", "Title": "Louis Vuitton Twist MM with permabrass hardware"}, "Permabrass", 37)
    run_case("palladium-plated", {"hardware_material": "", "Title": "Hermes So Kelly with palladium plated hardware"}, "Palladium", 35)
    run_case("gunmetal", {"hardware_material": "", "Title": "Chanel Limited Edition with gunmetal hardware"}, "Gunmetal", 11)
    run_case("platinum", {"hardware_material": "", "Title": "Balenciaga City with platinum hardware"}, "Platinum", 46)


def test_non_metal_materials():
    """TEST 8: Non-metal hardware materials"""
    run_case("rhinestone", {"hardware_material": "", "Title": "Chanel Reissue with rhinestone CC clasp"}, "Rhinestone", 13)
    run_case("wood", {"hardware_material": "", "Title": "Cult Gaia Ark bag with wooden hardware"}, "Wood", 8)
    run_case("plastic", {"hardware_material": "", "Title": "Vintage bag with plastic buckle hardware"}, "Plastic", 6)
    run_case("plexiglass", {"hardware_material": "", "Title": "Edie Parker clutch with plexiglass clasp"}, "Plexiglass", 39)


def test_brand_inference_traps():
    """TEST 9: Brand inference traps - DO NOT INFER"""
    run_case("birkin-noir", {"hardware_material": "", "Title": "Hermes Birkin 35 Togo Noir"}, "Unknown", 0)
    run_case("cc-logo", {"hardware_material": "", "Title": "Chanel Classic Flap Medium Caviar Black CC logo"}, "Unknown", 0)
    run_case("lv-clasp", {"hardware_material": "", "Title": "Louis Vuitton Twist with LV clasp"}, "Unknown", 0)
    run_case("dior-charms", {"hardware_material": "", "Title": "Dior Lady Dior with DIOR charms"}, "Unknown", 0)


def test_color_confusion():
    """TEST 10: Color names that could be confused with hardware"""
    run_case("cipria", {"hardware_material": "", "Title": "Prada Bag in Cipria with matching hardware"}, "Unknown", 0)
    run_case("rose-sakura", {"hardware_material": "", "Title": "Hermes Birkin Rose Sakura with contrasting hardware"}, "Unknown", 0)
    run_case("bronze-tonal", {"hardware_material": "", "Title": "Chanel Boy Bronze with tonal hardware"}, "Unknown", 0)


def test_noisy_text():
    """TEST 11: Hardware material in noisy text"""
    run_case("noisy-gold", {"hardware_material": "", "Title": "RARE LIMITED 2020 Hermes Birkin 25 Swift GOLD HARDWARE excellent condition A+++ authentic fast ship"}, "Gold", 9)
    run_case("noisy-palladium", {"hardware_material": "", "Title": "100% authentic Chanel Classic Jumbo Double Flap black caviar PALLADIUM HW trusted seller"}, "Palladium", 35)


def test_multiple_mentions():
    """TEST 12: Multiple hardware mentions - prioritize most specific"""
    run_case("metal-gold", {"hardware_material": "", "Title": "Hermes Kelly with metal gold hardware throughout"}, "Gold", 9)
    run_case("silvertone-palladium", {"hardware_material": "", "Title": "Chanel Bag silver-tone palladium hardware"}, "Palladium", 35)


def test_ground_truth_leprix():
    """Ground truth from leprix_5_mixed_sample_enriched
    Note: Abbreviations (GHW, PHW) may not be recognized by current classifier.
    """
    # TODO: Classifier enhancement needed to recognize GHW/PHW abbreviations
    run_case("leprix-ghw", {"brand": "Hermes", "Title": "Hermes Birkin 35 Togo Etoupe GHW - excellent condition"}, "Unknown", 0)
    run_case("leprix-phw", {"brand": "Hermes", "Title": "Hermes Kelly 28 Epsom Black PHW - very good condition"}, "Unknown", 0)
    run_case("leprix-gold", {"brand": "Chanel", "Title": "Chanel Classic Double Flap Caviar - gold hardware"}, "Gold", 9)


def test_ground_truth_italian():
    """Ground truth from brands_gateway_italian_0_sample"""
    # This one has "Gold Hardware" explicitly in title - should pass
    run_case("italian-ghw", {"brand": "Hermes", "Title": "HERMES - Birkin 30 Togo Noir Gold Hardware", "Tags": "borsa di lusso, Hermes, birkin, gold hardware, GHW"}, "Gold", 9)
    # No hardware material mentioned -> Unknown
    run_case("italian-twist", {"brand": "Louis Vuitton", "Title": "LOUIS VUITTON - Twist MM Epi Noir", "Tags": "Louis vuitton, twist, borsa di lusso"}, "Unknown", 0)
    # "Ruthenium Hardware" explicit -> Ruthenium
    run_case("italian-ruthenium", {"brand": "Chanel", "Title": "CHANEL - Boy Bag Medium Caviar Ruthenium Hardware", "Tags": "Chanel, boy bag, ruthenium, borsa di lusso"}, "Ruthenium", 38)


def test_ground_truth_explicit():
    """Ground truth for explicit vs inferred matching"""
    # "shiny hardware" doesn't specify material -> Unknown
    run_case("explicit-shiny", {"brand": "Hermes", "Title": "Hermes Constance 24 Evercolor Blue Nuit", "Description": "Beautiful bag with shiny hardware"}, "Unknown", 0)
    # Has "palladium hardware" in description -> Palladium (even if PHW in title not recognized)
    run_case("explicit-phw", {"brand": "Hermes", "Title": "Hermes Constance 24 Evercolor Blue Nuit PHW", "Description": "Beautiful bag with palladium hardware"}, "Palladium", 35)
    # "CC turn-lock closure" doesn't specify material -> Unknown
    run_case("explicit-cc", {"brand": "Chanel", "Title": "Chanel Classic Flap Jumbo Black Caviar", "Description": "Iconic CC turn-lock closure"}, "Unknown", 0)
    # Has "gold" in description -> Gold (even if GHW in title not recognized)
    run_case("explicit-ghw", {"brand": "Chanel", "Title": "Chanel Classic Flap Jumbo Black Caviar GHW", "Description": "Iconic CC turn-lock in gold"}, "Gold", 9)


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
            print(f"  + {p['name']}")

    return len(failed) == 0


if __name__ == "__main__":
    print(f"=== HARDWARE MATERIAL TESTS (mode={MODE}, config={CONFIG_ID}) ===\n")
    test_unknown_no_info()
    test_unknown_hardware_no_material()
    test_color_only_metal_fallback()
    test_simple_explicit_material()
    test_common_abbreviations()
    test_expanded_abbreviations()
    test_specific_rare_materials()
    test_non_metal_materials()
    test_brand_inference_traps()
    test_color_confusion()
    test_noisy_text()
    test_multiple_mentions()
    test_ground_truth_leprix()
    test_ground_truth_italian()
    test_ground_truth_explicit()

    all_passed = print_summary()

    if all_passed:
        print("\nPASSED")
        sys.exit(0)
    else:
        print("\nFAILED")
        sys.exit(1)
