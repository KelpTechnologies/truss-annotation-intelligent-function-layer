#!/usr/bin/env python3
"""
Test: keyword_synthetic_eval_test - Keyword extraction evaluation across synthetic datasets

Runs keyword classifier across synthetic CSVs using ground truth values as text_to_avoid.
Generates a detailed output CSV for manual review of keyword quality.

For each row:
- general_input_text: built from listing text columns
- text_to_avoid: built from ground truth columns (brand, model, colour, condition, material, size)
- Output: extracted keywords, confidence, reasoning

Usage:
    conda run -n pyds python tests/soft_tests/keyword_synthetic_eval_test.py -m local
    conda run -n pyds python tests/soft_tests/keyword_synthetic_eval_test.py -m local --csv C
    conda run -n pyds python tests/soft_tests/keyword_synthetic_eval_test.py -m local --workers 4
"""

import argparse
import sys
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

parser = argparse.ArgumentParser(description="Keyword extraction evaluation on synthetic data")
parser.add_argument("--mode", "-m", default="local", choices=["api", "local"], help="Test mode (default: local)")
parser.add_argument("--base-url", "-u", help="API base URL (or set STAGING_API_URL)")
parser.add_argument("--api-key", "-k", help="API key (or set DSL_API_KEY)")
parser.add_argument("--csv", choices=["B", "C", "D", "E", "all"], default="all", help="Which CSV to test")
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


def build_text_to_avoid(row):
    """Build text_to_avoid from ground truth columns."""
    avoid = {}
    gt_map = {
        "brand": "ground_truth_brand_text_classifier",
        "model": "ground_truth_model_text_classifier",
        "colour": "ground_truth_colour_text_classifier",
        "condition": "ground_truth_condition_text_classifier",
        "material": "ground_truth_material_text_classifier",
        "size": "ground_truth_size_text_classifier",
    }
    for prop, col in gt_map.items():
        val = row.get(col, "").strip()
        if val and val.lower() not in ("null", "", "unknown", "none", "nan"):
            avoid[prop] = val
    return avoid


_keyword_config = None


def _get_keyword_config():
    global _keyword_config
    if _keyword_config is None:
        _ensure_local_imports()
        from agent_orchestration.csv_config_loader import ConfigLoader
        config_loader = ConfigLoader(mode='dynamo', env='staging', fallback_env='staging')
        _keyword_config = config_loader.load_full_agent_config('classifier-keywords-bags')
    return _keyword_config


def classify_keywords_local(general_input_text: dict, text_to_avoid: dict) -> dict:
    """Classify via direct Python."""
    _ensure_local_imports()
    from agent_orchestration.keyword_classifier_orchestration import run_keyword_classification

    result = run_keyword_classification(
        general_input_text=general_input_text,
        text_to_avoid=text_to_avoid,
        full_config=_get_keyword_config(),
        item_id="1"
    )
    return result


def classify_keywords_api(general_input_text: dict, text_to_avoid: dict) -> dict:
    """Classify via HTTP API."""
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required or set STAGING_API_URL in .env")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/keywords",
        json={
            "general_input_text": general_input_text,
            "text_to_avoid": text_to_avoid
        },
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    return response.json()["data"][0]


def classify_keywords(general_input_text: dict, text_to_avoid: dict) -> dict:
    if MODE == "local":
        return classify_keywords_local(general_input_text, text_to_avoid)
    return classify_keywords_api(general_input_text, text_to_avoid)


def run_case(name: str, general_input_text: dict, text_to_avoid: dict, raw_listing: str) -> dict:
    """Run a single keyword extraction case."""
    try:
        r = classify_keywords(general_input_text, text_to_avoid)
        keywords = r.get("keywords", [])
        kw1 = keywords[0].get("keyword", "") if len(keywords) > 0 else ""
        kw2 = keywords[1].get("keyword", "") if len(keywords) > 1 else ""
        kw3 = keywords[2].get("keyword", "") if len(keywords) > 2 else ""
        kw1_conf = keywords[0].get("confidence", 0.0) if len(keywords) > 0 else 0.0
        kw2_conf = keywords[1].get("confidence", 0.0) if len(keywords) > 1 else 0.0
        kw3_conf = keywords[2].get("confidence", 0.0) if len(keywords) > 2 else 0.0

        return {
            "name": name,
            "raw_listing": raw_listing,
            "text_to_avoid": json.dumps(text_to_avoid),
            "keyword_1": kw1,
            "keyword_1_confidence": kw1_conf,
            "keyword_2": kw2,
            "keyword_2_confidence": kw2_conf,
            "keyword_3": kw3,
            "keyword_3_confidence": kw3_conf,
            "keyword_count": len(keywords),
            "reasoning": r.get("reasoning", ""),
            "success": r.get("success", False),
            "error": None
        }
    except Exception as e:
        return {
            "name": name,
            "raw_listing": raw_listing,
            "text_to_avoid": json.dumps(text_to_avoid),
            "keyword_1": "", "keyword_1_confidence": 0.0,
            "keyword_2": "", "keyword_2_confidence": 0.0,
            "keyword_3": "", "keyword_3_confidence": 0.0,
            "keyword_count": 0,
            "reasoning": "",
            "success": False,
            "error": str(e)
        }


# === CASE LOADERS ===
# Each returns (name, general_input_text, text_to_avoid, raw_listing_text)

def load_cases_B():
    master = DATA_DIR / "synthetic_B.csv"
    cases = []
    with open(master, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            raw = row.get("item_name", "")
            general_input = {"title": raw}
            avoid = build_text_to_avoid(row)
            cases.append((f"synB-{i+1}", general_input, avoid, raw))
    return cases


def load_cases_C():
    master = DATA_DIR / "synthetic_C.csv"
    cases = []
    with open(master, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            title = row.get("listing_title", "")
            notes = row.get("notes", "")
            raw = f"{title} | {notes}" if notes else title
            general_input = {"title": title}
            if notes:
                general_input["description"] = notes
            avoid = build_text_to_avoid(row)
            cases.append((f"synC-{i+1}", general_input, avoid, raw))
    return cases


def load_cases_D():
    master = DATA_DIR / "synthetic_D.csv"
    cases = []
    with open(master, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            title = row.get("title", "")
            general_input = {"title": title}
            avoid = build_text_to_avoid(row)
            cases.append((f"synD-{i+1}", general_input, avoid, title))
    return cases


def load_cases_E():
    master = DATA_DIR / "synthetic_E.csv"
    cases = []
    with open(master, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            name = row.get("product_name", "")
            desc = row.get("description", "")
            raw = f"{name} | {desc}" if desc else name
            general_input = {"title": name}
            if desc:
                general_input["description"] = desc
            avoid = build_text_to_avoid(row)
            cases.append((f"synE-{i+1}", general_input, avoid, raw))
    return cases


def run_cases_parallel(cases, max_workers):
    results = [None] * len(cases)

    def _worker(idx, name, general_input, text_to_avoid, raw_listing):
        return idx, run_case(name, general_input, text_to_avoid, raw_listing)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_worker, i, name, gi, ta, raw)
            for i, (name, gi, ta, raw) in enumerate(cases)
        ]
        done = 0
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            done += 1
            kws = [result.get(f"keyword_{j}", "") for j in range(1, 4) if result.get(f"keyword_{j}")]
            kw_str = ", ".join(kws) if kws else "(none)"
            status = "OK" if result["success"] else "ERR"
            print(f"  [{done}/{len(cases)}] {result['name']}: {status} -> {kw_str}")

    return results


def print_summary():
    success = [r for r in RESULTS if r["success"]]
    failed = [r for r in RESULTS if not r["success"]]

    print(f"\n{'='*60}")
    print(f"KEYWORD EVAL SUMMARY: {len(success)}/{len(RESULTS)} succeeded")
    print(f"{'='*60}")

    # Per-CSV breakdown
    csv_labels = [
        ("synB", "CSV B (Consignment)"),
        ("synC", "CSV C (Marketplace)"),
        ("synD", "CSV D (Trade)"),
        ("synE", "CSV E (Wholesale)"),
    ]
    print("\nPer-CSV breakdown:")
    for prefix, label in csv_labels:
        csv_results = [r for r in RESULTS if r["name"].startswith(prefix)]
        if csv_results:
            csv_ok = sum(1 for r in csv_results if r["success"])
            avg_count = sum(r["keyword_count"] for r in csv_results) / len(csv_results)
            print(f"  {label}: {csv_ok}/{len(csv_results)} succeeded, avg {avg_count:.1f} keywords/item")

    # Keyword frequency
    all_keywords = []
    for r in RESULTS:
        for j in range(1, 4):
            kw = r.get(f"keyword_{j}", "")
            if kw:
                all_keywords.append(kw.lower())

    if all_keywords:
        from collections import Counter
        top = Counter(all_keywords).most_common(15)
        print(f"\nTop 15 keywords extracted:")
        for kw, count in top:
            print(f"  {kw}: {count}")

    if failed:
        print(f"\nFailed extractions ({len(failed)}):")
        for f in failed[:10]:
            print(f"  {f['name']}: {f.get('error', 'unknown')}")

    return len(failed) == 0


def save_results_csv():
    logs_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"keyword_synthetic_eval_{ts}.csv"

    fieldnames = [
        "name", "raw_listing", "text_to_avoid",
        "keyword_1", "keyword_1_confidence",
        "keyword_2", "keyword_2_confidence",
        "keyword_3", "keyword_3_confidence",
        "keyword_count", "reasoning", "success", "error"
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in RESULTS:
            writer.writerow(r)

    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    csv_filter = args.csv
    max_workers = args.workers

    print(f"=== KEYWORD SYNTHETIC EVAL (mode={MODE}, workers={max_workers}) ===\n")

    if MODE == "local":
        _ensure_local_imports()

    # Load config once upfront for local mode to verify it works
    if MODE == "local":
        _ensure_local_imports()
        from agent_orchestration.csv_config_loader import ConfigLoader
        config_loader = ConfigLoader(mode='dynamo', env='staging', fallback_env='staging')
        try:
            full_config = config_loader.load_full_agent_config('classifier-keywords-bags')
            print(f"Config loaded: model={full_config['model_config'].get('model')}")
        except Exception as e:
            print(f"ERROR loading config: {e}")
            sys.exit(1)

    all_cases = []
    loaders = {
        "B": ("CSV B: Consignment Store", load_cases_B),
        "C": ("CSV C: Online Marketplace", load_cases_C),
        "D": ("CSV D: Trade Platform", load_cases_D),
        "E": ("CSV E: Wholesale Inventory", load_cases_E),
    }

    for label, (desc, loader) in loaders.items():
        if csv_filter in (label, "all"):
            cases = loader()
            print(f"--- {desc} ({len(cases)} cases) ---")
            all_cases.extend(cases)

    print(f"\nRunning {len(all_cases)} keyword extractions with {max_workers} workers...\n")
    RESULTS = run_cases_parallel(all_cases, max_workers)

    print_summary()
    save_results_csv()

    success_count = sum(1 for r in RESULTS if r["success"])
    print(f"\nDone. {success_count}/{len(RESULTS)} items processed successfully.")
