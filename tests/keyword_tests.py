#!/usr/bin/env python3
"""
Test: keyword_tests - Keyword classification tests (TEXT-ONLY)

Simulates Lambda routing: text-only -> classifier-keywords-bags-text

Usage:
    python keyword_tests.py --base-url https://... --api-key ...
    python keyword_tests.py --mode local

Environment variables (loaded from .env):
    API mode:   STAGING_API_URL, DSL_API_KEY
    Local mode: AWS_REGION, TRUSS_SECRETS_ARN, VERTEXAI_PROJECT, VERTEXAI_LOCATION
"""

import argparse
import sys
import os
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

parser = argparse.ArgumentParser(description="Keyword classification tests")
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

# Text-only config
CONFIG_ID = "classifier-keywords-bags-text"


def classify_via_api(text_dump: dict) -> dict:
    """Classify via HTTP API - simulates Lambda endpoint."""
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required for api mode")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/keywords",
        json={"text_dump": text_dump, "input_mode": "text-only"},
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    return response.json()["data"][0]


def classify_via_local(text_dump: dict) -> dict:
    """
    Classify via direct Python using keyword-specific orchestration.

    The keyword classifier expects:
    - general_input_text: text fields to process (Title, Description, etc.)
    - text_to_avoid: existing_classifications to avoid duplicating
    """
    repo_root = Path(__file__).parent.parent
    service_path = repo_root / "services" / "automated-annotation"
    sys.path.insert(0, str(service_path))

    # Initialize GCP credentials from AWS Secrets Manager
    from core.utils.credentials import ensure_gcp_adc
    ensure_gcp_adc()

    from agent_orchestration.csv_config_loader import ConfigLoader
    from agent_orchestration.keyword_classifier_orchestration import (
        run_keyword_classification,
        extract_keyword_result
    )

    # Split text_dump into general_input_text and text_to_avoid
    existing_classifications = text_dump.pop("existing_classifications", {})
    general_input_text = text_dump  # Remaining fields: Title, Description, Tags, etc.

    # Load config
    config_loader = ConfigLoader(mode='dynamo', env='staging', fallback_env='staging')
    full_config = config_loader.load_full_agent_config('classifier-keywords-bags-text')

    # Run keyword classification
    result = run_keyword_classification(
        general_input_text=general_input_text,
        text_to_avoid=existing_classifications,
        full_config=full_config,
        item_id="test"
    )

    # Convert keywords list to dict format for test comparison
    keywords_dict = {}
    for i, kw in enumerate(result.get("keywords", []), 1):
        if isinstance(kw, dict):
            keywords_dict[f"keyword_{i}"] = {
                "keyword": kw.get("keyword", ""),
                "confidence": kw.get("confidence", 0.0)
            }

    return {
        "keywords": keywords_dict,
        "reasoning": result.get("reasoning", ""),
        "success": result.get("success", False),
    }


def classify(text_dump: dict) -> dict:
    return classify_via_local(text_dump) if MODE == "local" else classify_via_api(text_dump)


def keywords_match(got_kw: dict, exp_kw: dict) -> bool:
    """
    Check if extracted keywords match expected keywords.

    For empty expected: got should also be empty {}
    For non-empty: check keyword values match (ignore confidence scores)
    """
    if not exp_kw:
        return not got_kw or got_kw == {}

    # Check all expected keywords are present with matching values
    for key in exp_kw:
        if key not in got_kw:
            return False
        exp_val = exp_kw[key].get("keyword", "").lower()
        got_val = got_kw[key].get("keyword", "").lower() if isinstance(got_kw[key], dict) else str(got_kw[key]).lower()
        if exp_val != got_val:
            return False
    return True


def run_case(name: str, text_dump: dict, exp_keywords: dict, exp_reasoning_contains: str = None):
    """
    Run a test case comparing expected vs actual keyword classification result.

    For empty keywords (exp_keywords={}), expects empty keywords object.
    For non-empty, checks keyword values match (confidence can vary).
    """
    try:
        r = classify(text_dump)
        got_kw = r.get("keywords", {})
        got_reasoning = r.get("reasoning", "")

        passed = keywords_match(got_kw, exp_keywords)

        # Format for display
        exp_str = json.dumps(exp_keywords) if exp_keywords else "{}"
        got_str = json.dumps(got_kw) if got_kw else "{}"

        RESULTS.append({
            "name": name,
            "passed": passed,
            "expected": exp_str,
            "actual": got_str,
            "reasoning": got_reasoning,
            "error": None
        })
    except Exception as e:
        RESULTS.append({
            "name": name,
            "passed": False,
            "expected": json.dumps(exp_keywords),
            "actual": None,
            "reasoning": None,
            "error": str(e)
        })


def run_case_empty(name: str, text_dump: dict):
    """Run test expecting empty keywords result."""
    run_case(name, text_dump, {})


def run_case_no_bad_keywords(name: str, text_dump: dict, bad_keywords: list):
    """
    Run test expecting NO bad keywords (condition, accessories, listing-specific).

    Passes if none of the bad_keywords appear in the result.
    Allows other valid keywords to be extracted.
    """
    try:
        r = classify(text_dump)
        got_kw = r.get("keywords", {})
        got_reasoning = r.get("reasoning", "")

        # Extract actual keyword values
        got_values = []
        for key in sorted(got_kw.keys()):
            if isinstance(got_kw[key], dict):
                got_values.append(got_kw[key].get("keyword", "").lower())
            else:
                got_values.append(str(got_kw[key]).lower())

        # Check no bad keywords are present
        bad_lower = [k.lower() for k in bad_keywords]
        found_bad = [bad for bad in bad_lower if any(bad in got for got in got_values)]

        passed = len(found_bad) == 0

        exp_str = f"should NOT contain: {bad_keywords}"
        got_str = f"got: {got_values}" + (f" (found bad: {found_bad})" if found_bad else "")

        RESULTS.append({
            "name": name,
            "passed": passed,
            "expected": exp_str,
            "actual": got_str,
            "reasoning": got_reasoning,
            "error": None
        })
    except Exception as e:
        RESULTS.append({
            "name": name,
            "passed": False,
            "expected": f"should NOT contain: {bad_keywords}",
            "actual": None,
            "reasoning": None,
            "error": str(e)
        })


def run_case_has_keywords(name: str, text_dump: dict, expected_keyword_values: list):
    """
    Run test expecting specific keywords (order-independent, confidence flexible).

    expected_keyword_values: list of expected keyword strings
    """
    try:
        r = classify(text_dump)
        got_kw = r.get("keywords", {})
        got_reasoning = r.get("reasoning", "")

        # Extract actual keyword values
        got_values = []
        for key in sorted(got_kw.keys()):
            if isinstance(got_kw[key], dict):
                got_values.append(got_kw[key].get("keyword", "").lower())
            else:
                got_values.append(str(got_kw[key]).lower())

        # Check at least one expected keyword is present
        exp_lower = [k.lower() for k in expected_keyword_values]
        found_any = any(any(exp in got for got in got_values) for exp in exp_lower)

        exp_str = f"contains any of: {expected_keyword_values}"
        got_str = f"got: {got_values}"

        RESULTS.append({
            "name": name,
            "passed": found_any,
            "expected": exp_str,
            "actual": got_str,
            "reasoning": got_reasoning,
            "error": None
        })
    except Exception as e:
        RESULTS.append({
            "name": name,
            "passed": False,
            "expected": f"contains: {expected_keyword_values}",
            "actual": None,
            "reasoning": None,
            "error": str(e)
        })


# === TESTS ===
def test_no_info():
    """TEST 1: No information at all - expect empty or minimal keywords

    Note: With empty Title, classifier may still extract inherent model features.
    We check that no condition/accessory/listing words are extracted.
    """
    # For truly empty input, any result is acceptable as long as no bad keywords
    bad_words = ["worn", "scratched", "dustbag", "box", "serial", "authenticated"]

    run_case_no_bad_keywords("empty-lv", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Neverfull", "material": "Canvas", "colour": "Brown", "condition": "Used Excellent", "size": "MM"},
        "Title": ""
    }, bad_words)

    run_case_no_bad_keywords("empty-chanel", {
        "existing_classifications": {"brand": "Chanel", "model": "Classic Flap", "material": "Caviar", "colour": "Black", "condition": "Used Very Good", "size": "Medium"},
        "Title": "Chanel Classic Flap"
    }, bad_words)


def test_already_classified():
    """TEST 2: Text contains ONLY already-classified attributes

    Note: Classifier may still extract inherent design features not explicitly in existing_classifications.
    We check that no condition/accessory/listing words are extracted.
    """
    bad_words = ["worn", "scratched", "dustbag", "box", "serial", "authenticated", "excellent condition", "used good"]

    run_case_no_bad_keywords("dup-hermes", {
        "existing_classifications": {"brand": "Hermès", "model": "Birkin", "material": "Togo", "root_material": "Leather", "hardware": "Gold", "root_hardware": "Gold", "colour": "Noir", "condition": "Used Excellent", "size": "35"},
        "Title": "Hermès Birkin 35 Togo Noir Gold Hardware Excellent Condition"
    }, bad_words)

    run_case_no_bad_keywords("dup-lv", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Speedy", "material": "Monogram Canvas", "root_material": "Canvas", "colour": "Brown", "condition": "Used Good", "size": "25"},
        "Title": "Louis Vuitton Speedy 25 Monogram Canvas Brown"
    }, bad_words)


def test_construction_keywords():
    """TEST 3: Construction details - quilted, woven, perforated"""
    run_case_has_keywords("quilted", {
        "existing_classifications": {"brand": "Chanel", "model": "Classic Flap", "material": "Lambskin", "colour": "Black", "hardware": "Gold", "condition": "Used Very Good", "size": "Medium"},
        "Title": "Chanel Classic Flap Medium Black Lambskin Quilted Gold Hardware"
    }, ["quilted"])

    run_case_has_keywords("intrecciato", {
        "existing_classifications": {"brand": "Bottega Veneta", "model": "Cassette", "material": "Leather", "colour": "Green", "condition": "Used Excellent", "size": "Medium"},
        "Title": "Bottega Veneta Cassette Medium Green Leather Intrecciato Woven"
    }, ["intrecciato", "woven"])

    run_case_has_keywords("perforated", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Capucines", "material": "Taurillon", "colour": "Black", "hardware": "Silver", "condition": "Brand New", "size": "MM"},
        "Title": "Louis Vuitton Capucines MM Black Taurillon Perforated Leather Silver Hardware"
    }, ["perforated"])


def test_silhouette_structure():
    """TEST 4: Silhouette and structure keywords"""
    run_case_has_keywords("bucket", {
        "existing_classifications": {"brand": "Hermès", "model": "Picotin", "material": "Clemence", "colour": "Gold", "hardware": "Palladium", "condition": "Used Very Good", "size": "18"},
        "Title": "Hermès Picotin Lock 18 Clemence Gold PHW Bucket Bag Open Top"
    }, ["bucket", "open top"])

    run_case_has_keywords("structured", {
        "existing_classifications": {"brand": "Celine", "model": "Luggage", "material": "Calfskin", "colour": "Black", "condition": "Used Good", "size": "Nano"},
        "Title": "Celine Luggage Nano Black Calfskin Structured Boxy Silhouette Top Handle"
    }, ["structured", "boxy", "top handle"])

    run_case_has_keywords("slouchy", {
        "existing_classifications": {"brand": "Saint Laurent", "model": "Lou Lou", "material": "Leather", "colour": "Beige", "hardware": "Gold", "condition": "Used Excellent", "size": "Medium"},
        "Title": "Saint Laurent Lou Lou Medium Beige Leather Y Quilted Slouchy Gold Hardware"
    }, ["quilted", "slouchy"])


def test_price_impact():
    """TEST 5: Price-impacting features - limited edition, vintage, collaboration"""
    run_case_has_keywords("murakami", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Neverfull", "material": "Canvas", "colour": "Multicolor", "condition": "Used Very Good", "size": "MM"},
        "Title": "Louis Vuitton Neverfull MM Multicolor Canvas Murakami Collaboration Limited Edition 2003"
    }, ["murakami", "collaboration", "limited edition"])

    run_case_has_keywords("seasonal", {
        "existing_classifications": {"brand": "Chanel", "model": "Classic Flap", "material": "Tweed", "colour": "Pink", "hardware": "Gold", "condition": "Used Excellent", "size": "Medium"},
        "Title": "Chanel Classic Flap Medium Pink Tweed Gold Hardware Seasonal Cruise Collection"
    }, ["seasonal", "cruise", "collection"])

    run_case_has_keywords("vintage", {
        "existing_classifications": {"brand": "Hermès", "model": "Kelly", "material": "Box", "colour": "Rouge", "hardware": "Gold", "condition": "Used Good", "size": "28"},
        "Title": "Hermès Kelly 28 Box Rouge Gold Hardware Vintage 1971 Sellier"
    }, ["vintage", "sellier", "1971"])


def test_closure_mechanism():
    """TEST 6: Closure and mechanism keywords"""
    run_case_has_keywords("double-flap", {
        "existing_classifications": {"brand": "Chanel", "model": "Classic Flap", "material": "Caviar", "colour": "Black", "hardware": "Gold", "condition": "Used Excellent", "size": "Jumbo"},
        "Title": "Chanel Classic Flap Jumbo Black Caviar Gold Hardware Double Flap Turn Lock CC Closure"
    }, ["double flap", "turn lock"])

    run_case_has_keywords("h-buckle", {
        "existing_classifications": {"brand": "Hermès", "model": "Constance", "material": "Epsom", "colour": "Blue", "hardware": "Palladium", "condition": "Used Very Good", "size": "24"},
        "Title": "Hermès Constance 24 Epsom Blue Palladium H Buckle Push Lock Closure"
    }, ["H buckle", "push lock"])

    run_case_has_keywords("twist-lock", {
        "existing_classifications": {"brand": "Fendi", "model": "Peekaboo", "material": "Leather", "colour": "Grey", "hardware": "Silver", "condition": "Brand New", "size": "Medium"},
        "Title": "Fendi Peekaboo Medium Grey Leather Silver Hardware Twist Lock Compartments"
    }, ["twist lock", "compartments"])


def test_condition_ignore():
    """TEST 7: CONDITION/WEAR - MUST IGNORE (condition words should NOT be extracted)"""
    # These tests check that condition/wear words are NOT extracted as keywords
    condition_words = ["worn", "scratched", "scuffed", "stained", "peeling", "faded", "tarnished", "loose", "marks", "creased", "discolored", "patina", "wear"]

    run_case_no_bad_keywords("worn-corners", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Speedy", "material": "Canvas", "colour": "Brown", "condition": "Used Fair", "size": "30"},
        "Title": "Louis Vuitton Speedy 30 Canvas worn corners scratched hardware scuffed vachetta stained interior"
    }, condition_words)

    run_case_no_bad_keywords("peeling", {
        "existing_classifications": {"brand": "Chanel", "model": "Boy", "material": "Lambskin", "colour": "Black", "hardware": "Ruthenium", "condition": "Used Poor", "size": "Medium"},
        "Title": "Chanel Boy Medium Black Lambskin Ruthenium peeling leather faded hardware tarnished chain loose stitching"
    }, condition_words)

    run_case_no_bad_keywords("marks", {
        "existing_classifications": {"brand": "Hermès", "model": "Birkin", "material": "Togo", "colour": "Etoupe", "hardware": "Gold", "condition": "Used Good", "size": "35"},
        "Title": "Hermès Birkin 35 Togo Etoupe Gold marks on exterior handle out of shape creased corners discolored glazing"
    }, condition_words)


def test_accessories_ignore():
    """TEST 8: ACCESSORIES - MUST IGNORE (dustbag, box, receipt etc.)

    Note: Some accessories like clochette, lock keys are integral parts of the bag.
    We only exclude generic listing accessories (dustbag, box, receipt, etc.)
    """
    # Core accessory words that should never be extracted
    accessory_words = ["dustbag", "dust bag", "box", "receipt", "tags", "booklet", "dust cover"]

    run_case_no_bad_keywords("dustbag", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Neverfull", "material": "Canvas", "colour": "Brown", "condition": "Used Excellent", "size": "GM"},
        "Title": "Louis Vuitton Neverfull GM Canvas comes with dustbag original box receipt tags booklet pouch"
    }, accessory_words)

    run_case_no_bad_keywords("orange-box", {
        "existing_classifications": {"brand": "Hermès", "model": "Kelly", "material": "Togo", "colour": "Gold", "hardware": "Palladium", "condition": "Brand New", "size": "25"},
        "Title": "Hermès Kelly 25 Togo Gold Palladium includes orange box dust cover lock keys clochette rain cover"
    }, accessory_words)


def test_listing_specific_ignore():
    """TEST 9: LISTING-SPECIFIC DETAILS - MUST IGNORE (serial numbers, ownership history)"""
    # These tests check that listing-specific words are NOT extracted as keywords
    listing_words = ["serial number", "date code", "authenticated", "entrupy", "one owner", "purchased", "rarely worn", "limited use"]

    run_case_no_bad_keywords("serial", {
        "existing_classifications": {"brand": "Chanel", "model": "Classic Flap", "material": "Caviar", "colour": "Black", "hardware": "Gold", "condition": "Used Excellent", "size": "Medium"},
        "Title": "Chanel Classic Flap Medium Black Caviar Gold serial number 12345678 date code 25 series authenticated by Entrupy"
    }, listing_words)

    run_case_no_bad_keywords("ownership", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Alma", "material": "Epi", "colour": "Black", "condition": "Used Very Good", "size": "PM"},
        "Title": "Louis Vuitton Alma PM Epi Black limited use rarely worn one owner purchased 2019 Paris store"
    }, listing_words)


def test_duplicate_material():
    """TEST 10: Duplicate avoidance - material mentioned again (extract only NEW features)"""
    run_case_has_keywords("lizard-trim", {
        "existing_classifications": {"brand": "Hermès", "model": "Birkin", "material": "Togo", "root_material": "Leather", "colour": "Noir", "hardware": "Gold", "root_hardware": "Gold", "condition": "Used Excellent", "size": "30"},
        "Title": "Hermès Birkin 30 Togo Leather Noir Black Gold Hardware GHW Touch Lizard Trim"
    }, ["lizard", "touch", "trim"])

    run_case_has_keywords("chevron", {
        "existing_classifications": {"brand": "Chanel", "model": "Boy", "material": "Caviar", "root_material": "Leather", "colour": "Navy", "hardware": "Ruthenium", "root_hardware": "Ruthenium", "condition": "Used Very Good", "size": "Old Medium"},
        "Title": "Chanel Boy Old Medium Navy Blue Caviar Leather Ruthenium Hardware Quilted Chevron Pattern"
    }, ["chevron", "quilted"])


def test_duplicate_brand_model():
    """TEST 11: Duplicate avoidance - brand/model (extract design features only)"""
    run_case_has_keywords("s-lock", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Pochette Métis", "material": "Canvas", "colour": "Brown", "hardware": "Gold", "condition": "Used Good", "size": "One Size"},
        "Title": "Louis Vuitton Pochette Métis Monogram Canvas Brown Gold Hardware Crossbody Flap Bag S Lock"
    }, ["S lock", "crossbody", "flap"])

    run_case_has_keywords("cannage", {
        "existing_classifications": {"brand": "Dior", "model": "Lady Dior", "material": "Lambskin", "colour": "Black", "hardware": "Silver", "condition": "Used Excellent", "size": "Medium"},
        "Title": "Christian Dior Lady Dior Medium Black Lambskin Silver Hardware Cannage Pattern DIOR Charms"
    }, ["cannage", "charms"])


def test_duplicate_condition():
    """TEST 12: Duplicate avoidance - condition (extract design features only)"""
    run_case_has_keywords("tiger-head", {
        "existing_classifications": {"brand": "Gucci", "model": "Dionysus", "material": "Canvas", "colour": "Beige", "hardware": "Silver", "condition": "Brand New", "size": "Small"},
        "Title": "Gucci Dionysus Small Beige Canvas Silver Hardware Brand New With Tags BNWT Tiger Head Closure"
    }, ["tiger head"])

    run_case_has_keywords("double-zip", {
        "existing_classifications": {"brand": "Prada", "model": "Galleria", "material": "Saffiano", "colour": "Black", "hardware": "Gold", "condition": "Used Excellent", "size": "Medium"},
        "Title": "Prada Galleria Medium Black Saffiano Gold Excellent Condition Double Zip Structured"
    }, ["double zip", "structured"])


def test_duplicate_size():
    """TEST 13: Duplicate avoidance - size (extract design features only)"""
    run_case_has_keywords("perforated-h", {
        "existing_classifications": {"brand": "Hermès", "model": "Evelyne", "material": "Clemence", "colour": "Gold", "hardware": "Palladium", "condition": "Used Very Good", "size": "PM"},
        "Title": "Hermès Evelyne PM Clemence Gold Palladium Perforated H Logo Crossbody Adjustable Strap"
    }, ["perforated", "crossbody", "adjustable strap"])

    run_case_has_keywords("bandouliere", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Keepall", "material": "Canvas", "colour": "Brown", "hardware": "Gold", "condition": "Used Good", "size": "55"},
        "Title": "Louis Vuitton Keepall 55 Bandouliere Monogram Canvas Brown Gold 55cm Duffle Travel Bag"
    }, ["bandouliere", "duffle", "travel"])


def test_noisy_text():
    """TEST 14: Valid design elements with noisy text"""
    run_case_has_keywords("giant-studs", {
        "existing_classifications": {"brand": "Balenciaga", "model": "City", "material": "Lambskin", "colour": "Black", "hardware": "Silver", "condition": "Used Very Good", "size": "Medium"},
        "Title": "RARE 100% AUTH Balenciaga City Medium Black Lambskin Silver Hardware GIANT STUDS Motorcycle Bag A+++ Seller Fast Ship"
    }, ["giant studs", "motorcycle"])

    run_case_has_keywords("saddle-whipstitch", {
        "existing_classifications": {"brand": "Chloe", "model": "Marcie", "material": "Calfskin", "colour": "Tan", "condition": "Used Excellent", "size": "Medium"},
        "Title": "authentic guaranteed Chloe Marcie Medium Tan Calfskin excellent Saddle Bag Whipstitch Detail Braided Handle free returns"
    }, ["saddle", "whipstitch", "braided"])


def test_multiple_keywords():
    """TEST 15: Multiple valid keywords - ranking test"""
    run_case_has_keywords("chanel19", {
        "existing_classifications": {"brand": "Chanel", "model": "19", "material": "Lambskin", "colour": "Black", "hardware": "Mixed Metal", "condition": "Used Excellent", "size": "Large"},
        "Title": "Chanel 19 Large Black Lambskin Mixed Metal Hardware Quilted Diamond Pattern Chain Woven Leather Strap Magnetic Flap CC Turn Lock"
    }, ["quilted", "diamond", "chain", "magnetic flap"])

    run_case_has_keywords("lv-twist", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Twist", "material": "Epi", "colour": "Pink", "hardware": "Silver", "condition": "Brand New", "size": "MM"},
        "Title": "Louis Vuitton Twist MM Pink Epi Silver Hardware Embossed Monogram Flower LV Lock Crossbody Chain Strap Limited Edition"
    }, ["limited edition", "embossed", "LV lock"])


def test_ground_truth_leprix():
    """Ground truth from leprix sample data"""
    run_case_has_keywords("leprix-flap", {
        "existing_classifications": {"brand": "Chanel", "model": "Classic Flap", "material": "Caviar", "root_material": "Leather", "hardware": "Gold", "colour": "Black", "condition": "Used Excellent", "size": "Medium"},
        "Title": "Chanel Classic Flap Medium Black Caviar Gold Hardware Double Flap Quilted Diamond"
    }, ["double flap", "quilted"])

    # Birkin with Taurillon Clemence - may extract hardware/material details
    # We just check no bad keywords are present
    bad_words = ["worn", "scratched", "dustbag", "box", "serial", "authenticated", "very good"]
    run_case_no_bad_keywords("leprix-birkin", {
        "existing_classifications": {"brand": "Hermès", "model": "Birkin", "material": "Togo", "root_material": "Leather", "hardware": "Palladium", "colour": "Etoupe", "condition": "Used Very Good", "size": "35"},
        "Title": "Hermès Birkin 35 Togo Etoupe Palladium Hardware PHW Taurillon Clemence"
    }, bad_words)

    run_case_has_keywords("leprix-speedy", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Speedy", "material": "Canvas", "root_material": "Canvas", "hardware": "Gold", "colour": "Brown", "condition": "Used Good", "size": "25"},
        "Title": "Louis Vuitton Speedy 25 Bandouliere Monogram Canvas Brown Gold Hardware Crossbody Strap"
    }, ["bandouliere", "crossbody"])


def test_ground_truth_italian():
    """Ground truth from Italian sample data"""
    run_case_has_keywords("ital-metis", {
        "existing_classifications": {"brand": "Louis Vuitton", "model": "Pochette Métis", "material": "Canvas", "root_material": "Canvas", "hardware": "Gold", "colour": "Brown", "condition": "Used Excellent", "size": "One Size"},
        "Title": "LOUIS VUITTON - Pochette Métis Monogram Canvas S Lock Crossbody",
        "Tags": "borsa di lusso, Louis vuitton, pochette, monogram, crossbody"
    }, ["S lock", "crossbody"])

    run_case_has_keywords("ital-dionysus", {
        "existing_classifications": {"brand": "Gucci", "model": "Dionysus", "material": "Canvas", "root_material": "Canvas", "hardware": "Silver", "colour": "Beige", "condition": "Used Very Good", "size": "Small"},
        "Title": "GUCCI - Dionysus Small GG Supreme Canvas Tiger Head",
        "Tags": "borsa di lusso, Gucci, dionysus, GG supreme, tiger head"
    }, ["tiger head", "GG supreme"])


def test_condition_vs_design():
    """Ground truth: condition vs design feature distinction"""
    run_case_has_keywords("boy-quilted", {
        "existing_classifications": {"brand": "Chanel", "model": "Boy", "material": "Calfskin", "hardware": "Ruthenium", "colour": "Black", "condition": "Used Good", "size": "Medium"},
        "Title": "Chanel Boy Medium Black Calfskin Ruthenium Quilted Stitch Detail Chain Strap",
        "Description": "Shows normal wear on corners"
    }, ["quilted", "chain strap"])

    run_case_has_keywords("kelly-sellier", {
        "existing_classifications": {"brand": "Hermès", "model": "Kelly", "material": "Epsom", "hardware": "Palladium", "colour": "Blue", "condition": "Used Fair", "size": "28"},
        "Title": "Hermès Kelly 28 Epsom Blue Palladium Sellier Rigid Structure",
        "Description": "Corner wear visible, handle shows patina, interior stained"
    }, ["sellier", "rigid"])


def test_duplicate_avoidance():
    """Ground truth: already-classified attribute avoidance"""
    run_case_has_keywords("galleria", {
        "existing_classifications": {"brand": "Prada", "model": "Galleria", "material": "Saffiano", "root_material": "Leather", "hardware": "Gold", "colour": "Black", "condition": "Used Excellent", "size": "Medium"},
        "Title": "Prada Galleria Medium Saffiano Leather Black Gold Hardware Double Zip Tote",
        "Description": "Saffiano leather exterior, gold tone hardware, medium size"
    }, ["double zip", "tote"])

    run_case_has_keywords("lady-dior", {
        "existing_classifications": {"brand": "Dior", "model": "Lady Dior", "material": "Lambskin", "root_material": "Leather", "hardware": "Silver", "colour": "Pink", "condition": "Brand New", "size": "Small"},
        "Title": "Dior Lady Dior Small Pink Lambskin Silver Hardware Cannage Quilted DIOR Charms",
        "Description": "Pink lambskin with silver hardware, brand new condition"
    }, ["cannage", "DIOR charms"])


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
            if f.get('reasoning'):
                print(f"  reasoning: {f['reasoning'][:100]}...")
            if f['error']:
                print(f"  error:    {f['error']}")
            print("-" * 60)

    if passed:
        print("\nPassed tests:")
        for p in passed:
            print(f"  + {p['name']}")

    return len(failed) == 0


if __name__ == "__main__":
    print(f"=== KEYWORD TESTS (mode={MODE}, config={CONFIG_ID}) ===\n")

    test_no_info()
    test_already_classified()
    test_construction_keywords()
    test_silhouette_structure()
    test_price_impact()
    test_closure_mechanism()
    test_condition_ignore()
    test_accessories_ignore()
    test_listing_specific_ignore()
    test_duplicate_material()
    test_duplicate_brand_model()
    test_duplicate_condition()
    test_duplicate_size()
    test_noisy_text()
    test_multiple_keywords()
    test_ground_truth_leprix()
    test_ground_truth_italian()
    test_condition_vs_design()
    test_duplicate_avoidance()

    all_passed = print_summary()

    if all_passed:
        print("\nPASSED")
        sys.exit(0)
    else:
        print("\nFAILED")
        sys.exit(1)
