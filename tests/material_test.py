#!/usr/bin/env python3
"""
Test: material_test
Tests material classification endpoint with various inputs.

Usage:
    python material_test.py --api-key <key>
    python material_test.py --api-key <key> --base-url <url>
"""

import argparse
import os
import sys
import math
import requests

parser = argparse.ArgumentParser(description="Test material classification endpoint")
parser.add_argument("--api-key", "-k", help="API key for x-api-key header")
parser.add_argument("--base-url", "-u", help="Base URL for API")
args = parser.parse_args()

BASE_URL = args.base_url or os.environ.get("API_BASE_URL") or os.environ.get("STAGING_API_URL")
API_KEY = args.api_key or os.environ.get("TEST_API_KEY") or os.environ.get("API_KEY") or ""

if not BASE_URL:
    print("ERROR: Provide --base-url or set API_BASE_URL/STAGING_API_URL env var")
    sys.exit(1)

# Remove trailing slash from BASE_URL if present
BASE_URL = BASE_URL.rstrip("/")

ENDPOINT = "/automations/annotation/bags/classify/material"

print(f"Using: {BASE_URL}{ENDPOINT}")


def is_nan_equivalent(value):
    """Check if value represents NaN/null/empty."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ("nan", ""):
        return True
    return False


# Test cases: (input_text_dump, expected_material, expected_material_id, expected_root_material, expected_root_material_id)
TEST_CASES = [
    # Case 1: Canvas -> Canvas (root is itself)
    (
        {"material": "Canvas", "Title": "Crossbody Louis Vuitton Neverfull"},
        "Canvas", 2, "Canvas", 2
    ),
    # Case 2: Calfskin -> Leather (root is parent)
    (
        {"material": "Calfskin", "Title": "Crossbody Louis Vuitton Neverfull"},
        "Calfskin", 47, "Leather", 1
    ),
    # Case 3: Empty input -> all NaN
    (
        {"material": "", "Title": ""},
        None, None, None, None  # Expect NaN equivalents
    ),
    # Case 4: Title only, no material -> all NaN
    (
        {"material": "", "Title": "Louis vuitton bag"},
        None, None, None, None  # Expect NaN equivalents
    ),
]


def run_test_case(case_num, text_dump, expected_material, expected_material_id, expected_root, expected_root_id):
    """Run a single test case."""
    response = requests.post(
        f"{BASE_URL}{ENDPOINT}",
        json={
            "text_dump": text_dump,
            "input_mode": "text-only"
        },
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )

    assert response.status_code == 200, \
        f"Case {case_num}: Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    result = data["data"][0]

    # Check material
    actual_material = result.get("material")
    if expected_material is None:
        assert is_nan_equivalent(actual_material), \
            f"Case {case_num}: Expected material NaN, got {actual_material}"
    else:
        assert actual_material == expected_material, \
            f"Case {case_num}: Expected material '{expected_material}', got '{actual_material}'"

    # Check material_id
    actual_material_id = result.get("material_id")
    if expected_material_id is None:
        assert is_nan_equivalent(actual_material_id), \
            f"Case {case_num}: Expected material_id NaN, got {actual_material_id}"
    else:
        assert actual_material_id == expected_material_id, \
            f"Case {case_num}: Expected material_id {expected_material_id}, got {actual_material_id}"

    # Check root_material
    actual_root = result.get("root_material")
    if expected_root is None:
        assert is_nan_equivalent(actual_root), \
            f"Case {case_num}: Expected root_material NaN, got {actual_root}"
    else:
        assert actual_root == expected_root, \
            f"Case {case_num}: Expected root_material '{expected_root}', got '{actual_root}'"

    # Check root_material_id
    actual_root_id = result.get("root_material_id")
    if expected_root_id is None:
        assert is_nan_equivalent(actual_root_id), \
            f"Case {case_num}: Expected root_material_id NaN, got {actual_root_id}"
    else:
        assert actual_root_id == expected_root_id, \
            f"Case {case_num}: Expected root_material_id {expected_root_id}, got {actual_root_id}"

    # Describe result
    if expected_material:
        print(f"Case {case_num} PASSED  ({expected_material} -> {expected_root})")
    else:
        print(f"Case {case_num} PASSED  (empty -> NaN)")


if __name__ == "__main__":
    try:
        for i, (text_dump, mat, mat_id, root, root_id) in enumerate(TEST_CASES, 1):
            run_test_case(i, text_dump, mat, mat_id, root, root_id)
        print("PASSED")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"ERROR: Request failed - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
