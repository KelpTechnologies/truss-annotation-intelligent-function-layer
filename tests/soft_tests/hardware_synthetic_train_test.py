#!/usr/bin/env python3
"""
Test: hardware_synthetic_train_test - Hardware classification against synthetic dataset C

Tests hardware classifier against synthetic CSV with marketplace-style listings.
Each row has listing_title + notes with hardware descriptions and ground truth.

Usage:
    conda run -n pyds python tests/soft_tests/hardware_synthetic_train_test.py -m local
    conda run -n pyds python tests/soft_tests/hardware_synthetic_train_test.py -m local --workers 5
    conda run -n pyds python tests/soft_tests/hardware_synthetic_train_test.py -m api
"""

import argparse
import sys
import math
import json
import os
import csv
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Load .env at initialization
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Set cross-platform temp path for GCP credentials (before lambda imports)
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(tempfile.gettempdir(), "gcp_sa.json")

parser = argparse.ArgumentParser(description="Hardware synthetic dataset tests")
parser.add_argument("--mode", "-m", default="local", choices=["api", "local"], help="Test mode (default: local)")
parser.add_argument("--base-url", "-u", help="API base URL (or set STAGING_API_URL)")
parser.add_argument("--api-key", "-k", help="API key (or set DSL_API_KEY)")
parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers (default 4)")
parser.add_argument("--verbose", "-v", action="store_true", help="Show INFO logs (suppressed by default)")
args = parser.parse_args()

# Suppress verbose logs unless --verbose
if not args.verbose:
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("agent_orchestration", "agent_architecture", "core", "google_genai",
                 "httpx", "botocore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

MODE = args.mode
BASE_URL = args.base_url or os.environ.get("STAGING_API_URL") or os.environ.get("API_BASE_URL")
BASE_URL = BASE_URL.rstrip("/") if BASE_URL else None
API_KEY = args.api_key or os.environ.get("DSL_API_KEY") or os.environ.get("API_KEY") or ""
RESULTS = []
RESULTS_LOCK = threading.Lock()

CONFIG_ID = "classifier-hardware-bags"

DATA_DIR = Path(__file__).resolve().parent.parent / "test_input_data"

_local_imports_ready = False


def _ensure_local_imports():
    global _local_imports_ready
    if _local_imports_ready:
        return
    repo_root = Path(__file__).resolve().parent.parent.parent
    service_path = repo_root / "services" / "automated-annotation"
    if str(service_path) not in sys.path:
        sys.path.insert(0, str(service_path))
    from core.utils.credentials import ensure_gcp_adc
    ensure_gcp_adc()
    _local_imports_ready = True


def classify_via_api(text_dump: dict) -> dict:
    """Classify via HTTP API."""
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required or set STAGING_API_URL in .env")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/hardware",
        json={"text_dump": text_dump, "input_mode": "text-only"},
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    return response.json()["data"][0]


def classify_via_local(text_dump: dict) -> dict:
    """Classify via direct Python - simulates Lambda routing with proper config resolution."""
    _ensure_local_imports()
    from agent_orchestration.classifier_api_orchestration import classify_for_api
    from core.orchestration_api_handlers.agent_orchestration_api_handler import get_config_id_for_input_mode

    config_id = get_config_id_for_input_mode(CONFIG_ID, "text-only")
    text_input = json.dumps(text_dump, indent=2)
    result = classify_for_api(
        config_id=config_id,
        text_input=text_input,
        input_mode="text-only",
        env="staging"
    )
    return {
        "hardware": result.get("primary_name"),
        "hardware_id": result.get("primary_id"),
        "confidence": result.get("confidence", 0),
        "success": result.get("success", False),
        "reasoning": result.get("reasoning", "")
    }


def classify(text_dump: dict) -> dict:
    return classify_via_local(text_dump) if MODE == "local" else classify_via_api(text_dump)


def is_unknown(value):
    if value is None: return True
    if value == 0: return True
    if isinstance(value, str) and value.lower() in ("unknown", "nan", "", "null", "none"): return True
    if isinstance(value, float) and math.isnan(value): return True
    return False


def run_case(name: str, text_dump: dict, exp_hardware) -> dict:
    """Run a single test case."""
    try:
        r = classify(text_dump)
        got_hardware = r.get("hardware")

        if exp_hardware is None:
            passed = is_unknown(got_hardware)
            exp_str = "None"
        else:
            # Case-insensitive match
            passed = (str(got_hardware).lower() == str(exp_hardware).lower()) if got_hardware else False
            exp_str = exp_hardware

        got_str = got_hardware or "None"
        return {
            "name": name, "passed": passed, "expected": exp_str, "actual": got_str,
            "confidence": r.get("confidence", 0), "reasoning": r.get("reasoning", ""),
            "error": None
        }
    except Exception as e:
        return {
            "name": name, "passed": False, "expected": exp_hardware or "None",
            "actual": None, "confidence": 0, "reasoning": "", "error": str(e)
        }


def _parse_expected(row):
    exp = row.get("ground_truth_hardware_text_classifier", "").strip()
    if exp.lower() in ("null", "", "unknown", "none"):
        return None
    return exp


def load_cases_C():
    csv_path = DATA_DIR / "synthetic_C_hardware.csv"
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            cases.append((f"synC-{i+1}", {
                "Title": row.get("listing_title", ""),
                "description": row.get("notes", ""),
            }, _parse_expected(row)))
    return cases


def run_cases_parallel(cases, max_workers):
    results = [None] * len(cases)

    def _worker(idx, name, text_dump, exp):
        return idx, run_case(name, text_dump, exp)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_worker, i, name, td, exp)
            for i, (name, td, exp) in enumerate(cases)
        ]
        done = 0
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            done += 1
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  [{done}/{len(cases)}] {result['name']}: {status} (exp={result['expected']} got={result['actual']})")

    return results


def print_summary():
    passed = [r for r in RESULTS if r["passed"]]
    failed = [r for r in RESULTS if not r["passed"]]

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {len(passed)}/{len(RESULTS)} passed")
    print(f"{'='*60}")

    if failed:
        print(f"\nFailed tests:")
        print("-" * 60)
        for f in failed:
            print(f"  test_ref: {f['name']}")
            print(f"  expected: {f['expected']}")
            print(f"  actual:   {f['actual']}")
            if f.get('reasoning'):
                print(f"  reasoning: {f['reasoning'][:120]}")
            if f['error']:
                print(f"  error:    {f['error']}")
            print("-" * 60)

    if passed:
        print(f"\nPassed tests:")
        for p in passed:
            print(f"  + {p['name']} (confidence={p.get('confidence', 0)})")

    return len(failed) == 0


def save_results_csv():
    logs_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"hardware_synthetic_{ts}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "passed", "expected", "actual", "confidence", "reasoning", "error"])
        writer.writeheader()
        for r in RESULTS:
            writer.writerow(r)

    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    max_workers = args.workers

    print(f"=== HARDWARE SYNTHETIC TESTS (mode={MODE}, config={CONFIG_ID}, workers={max_workers}) ===\n")

    if MODE == "local":
        _ensure_local_imports()

    cases = load_cases_C()
    print(f"--- CSV C: Online Marketplace ({len(cases)} cases) ---")

    print(f"\nRunning {len(cases)} cases with {max_workers} workers...\n")
    RESULTS = run_cases_parallel(cases, max_workers)

    all_passed = print_summary()
    save_results_csv()

    if all_passed:
        print("\nPASSED")
        sys.exit(0)
    else:
        print("\nFAILED")
        sys.exit(1)
