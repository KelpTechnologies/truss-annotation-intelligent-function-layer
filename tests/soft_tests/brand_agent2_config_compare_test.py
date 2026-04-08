#!/usr/bin/env python3
"""
Brand Agent 2 Config Comparison Test
=====================================
Runs synthetic CSV A through the regex pipeline with two different Agent 2 configs.
Compares accuracy, latency, and output values.

Usage:
    conda run -n pyds python tests/soft_tests/brand_agent2_config_compare_test.py -m local
    conda run -n pyds python tests/soft_tests/brand_agent2_config_compare_test.py -m local -n 10 --verbose
"""

import argparse
import copy
import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(tempfile.gettempdir(), "gcp_sa.json")

parser = argparse.ArgumentParser(description="Brand Agent 2 config A vs B comparison on CSV A")
parser.add_argument("--mode", "-m", default="local", choices=["local"])
parser.add_argument("--workers", "-w", type=int, default=1)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--limit", "-n", type=int, default=None)
parser.add_argument("--csv", choices=["A", "B", "C", "D", "E", "all"], default="A", help="Which CSV(s) to test")
parser.add_argument("--sample", "-s", type=int, default=None, help="Random sample N from each CSV")
parser.add_argument("--config-a", type=str, default="brand-classification-v1", help="Config ID A (default, current)")
parser.add_argument("--config-b", type=str, default="brand-classification-v2", help="Config ID B (new)")
args = parser.parse_args()

if not args.verbose:
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("agent_orchestration", "agent_architecture", "core", "google_genai",
                 "httpx", "botocore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "tests" / "test_input_data"
if not DATA_DIR.exists():
    DATA_DIR = Path("C:/Users/dce/documents/TRUSS/Github/truss-annotation-intelligent-function-layer/tests/test_input_data")
SERVICE_PATH = REPO_ROOT / "services" / "automated-annotation"

sys.path.insert(0, str(SERVICE_PATH))

from core.utils.credentials import ensure_gcp_adc
ensure_gcp_adc()

from agent_orchestration.brand_classification_orchestration import (
    run_agent2_brand_classification,
    run_brand_classification_workflow,
)
from agent_orchestration.csv_config_loader import ConfigLoader
from agent_orchestration.regex_brand_lookup import BrandMasterIndex


def _parse_expected(row):
    exp = row.get("ground_truth_brand_text_classifier", "").strip()
    return None if exp.lower() in ("null", "", "unknown") else exp

def load_cases_A():
    cases = []
    with open(DATA_DIR / "synthetic_A_brand.csv", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            cases.append((f"synA-{i+1}", {
                "Title": row.get("title", ""),
                "description": row.get("description", ""),
            }, _parse_expected(row)))
    return cases

def load_cases_B():
    cases = []
    with open(DATA_DIR / "synthetic_B_brand.csv", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            cases.append((f"synB-{i+1}", {
                "brand": row.get("brand", ""),
                "Title": row.get("item_name", ""),
            }, _parse_expected(row)))
    return cases

def load_cases_C():
    cases = []
    with open(DATA_DIR / "synthetic_C_brand.csv", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            cases.append((f"synC-{i+1}", {
                "brand": row.get("brand_name", ""),
                "Title": row.get("listing_title", ""),
            }, _parse_expected(row)))
    return cases

def load_cases_D():
    cases = []
    with open(DATA_DIR / "synthetic_D_brand.csv", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            cases.append((f"synD-{i+1}", {
                "brand": row.get("brand", ""),
                "Title": row.get("title", ""),
            }, _parse_expected(row)))
    return cases

def load_cases_E():
    cases = []
    with open(DATA_DIR / "synthetic_E_brand.csv", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            cases.append((f"synE-{i+1}", {
                "Title": row.get("product_name", ""),
                "description": row.get("description", ""),
            }, _parse_expected(row)))
    return cases


def run_with_config(case_name, text_dump, exp_brand, brand_index, config_id, config_loader, verbose):
    """Run regex extraction → Agent 2 with a specific config. Returns timing + result."""
    raw_text = json.dumps(text_dump, indent=2)
    search_text = f"{text_dump.get('Title', '')} {raw_text}"

    # Step 1: regex extraction (same for both configs)
    t0 = time.perf_counter()
    regex_result = brand_index.search(search_text, verbose=verbose)
    extraction_ms = (time.perf_counter() - t0) * 1000

    matched_brands = regex_result.get("matched_brands", [])
    unique_proposed = len(matched_brands)

    if not matched_brands:
        return {
            "case": case_name, "config": config_id,
            "total_ms": round(extraction_ms, 1), "extraction_ms": round(extraction_ms, 1),
            "agent2_ms": 0, "unique_brands_proposed": 0,
            "final_brand": "Unknown", "final_brand_id": None, "confidence": 0,
            "expected": exp_brand or "Unknown", "passed": exp_brand is None,
            "reasoning": "No regex matches", "error": "No regex matches",
        }

    # Step 2: Agent 2 with specified config
    try:
        base_config = config_loader.load_full_agent_config(config_id)
    except Exception as e:
        return {
            "case": case_name, "config": config_id,
            "total_ms": round(extraction_ms, 1), "extraction_ms": round(extraction_ms, 1),
            "agent2_ms": 0, "unique_brands_proposed": unique_proposed,
            "final_brand": "Unknown", "final_brand_id": None, "confidence": 0,
            "expected": exp_brand or "Unknown", "passed": False,
            "reasoning": "", "error": f"Config load failed: {e}",
        }

    t1 = time.perf_counter()
    agent2_result = run_agent2_brand_classification(
        raw_text=raw_text,
        name=text_dump.get("Title"),
        matched_brands=matched_brands,
        env="staging",
        verbose=verbose,
        base_config=base_config,
    )
    agent2_ms = (time.perf_counter() - t1) * 1000
    total_ms = extraction_ms + agent2_ms

    final_brand = agent2_result.get("brand_name") or "Unknown"
    final_brand_id = agent2_result.get("prediction_id")
    confidence = agent2_result.get("confidence", 0)
    reasoning = agent2_result.get("reasoning", "")

    if exp_brand is None:
        passed = final_brand.lower() in ("unknown", "none", "", "null")
    else:
        passed = (final_brand == exp_brand)

    return {
        "case": case_name, "config": config_id,
        "total_ms": round(total_ms, 1), "extraction_ms": round(extraction_ms, 1),
        "agent2_ms": round(agent2_ms, 1), "unique_brands_proposed": unique_proposed,
        "final_brand": final_brand, "final_brand_id": final_brand_id,
        "confidence": confidence, "expected": exp_brand or "Unknown",
        "passed": passed, "reasoning": reasoning, "error": None,
    }


def main():
    loaders = {"A": load_cases_A, "B": load_cases_B, "C": load_cases_C,
                "D": load_cases_D, "E": load_cases_E}
    csv_keys = list(loaders.keys()) if args.csv == "all" else [args.csv]

    cases = []
    for key in csv_keys:
        csv_cases = loaders[key]()
        if args.sample and args.sample < len(csv_cases):
            csv_cases = random.sample(csv_cases, args.sample)
        cases.extend(csv_cases)

    if args.limit:
        cases = cases[:args.limit]
    print(f"Loaded {len(cases)} cases from CSV(s) {', '.join(csv_keys)}")
    print(f"Config A: {args.config_a}")
    print(f"Config B: {args.config_b}\n")

    print("Loading brand master index...")
    t0 = time.perf_counter()
    brand_index = BrandMasterIndex()
    print(f"  {brand_index.total_patterns} patterns loaded in {(time.perf_counter()-t0)*1000:.0f}ms\n")

    config_loader = ConfigLoader(mode='dynamo', env='staging', fallback_env='staging')

    # --- Config A ---
    print(f"{'='*70}")
    print(f"CONFIG A: {args.config_a}")
    print(f"{'='*70}")
    results_a = []
    for i, (name, td, exp) in enumerate(cases):
        r = run_with_config(name, td, exp, brand_index, args.config_a, config_loader, args.verbose)
        results_a.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{i+1}/{len(cases)}] {name}: {status} | {r['total_ms']:.0f}ms "
              f"(extract={r['extraction_ms']:.0f}ms agent2={r['agent2_ms']:.0f}ms) "
              f"| {r['unique_brands_proposed']} brands → {r['final_brand']}")

    # --- Config B ---
    print(f"\n{'='*70}")
    print(f"CONFIG B: {args.config_b}")
    print(f"{'='*70}")
    results_b = []
    for i, (name, td, exp) in enumerate(cases):
        r = run_with_config(name, td, exp, brand_index, args.config_b, config_loader, args.verbose)
        results_b.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{i+1}/{len(cases)}] {name}: {status} | {r['total_ms']:.0f}ms "
              f"(extract={r['extraction_ms']:.0f}ms agent2={r['agent2_ms']:.0f}ms) "
              f"| {r['unique_brands_proposed']} brands → {r['final_brand']}")

    # --- Comparison ---
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    total = len(cases)
    pass_a = sum(1 for r in results_a if r["passed"])
    pass_b = sum(1 for r in results_b if r["passed"])

    def avg(lst): return sum(lst) / len(lst) if lst else 0
    def med(lst):
        s = sorted(lst)
        n = len(s)
        return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2

    a2_ms_a = [r["agent2_ms"] for r in results_a if r["agent2_ms"] > 0]
    a2_ms_b = [r["agent2_ms"] for r in results_b if r["agent2_ms"] > 0]
    total_ms_a = [r["total_ms"] for r in results_a]
    total_ms_b = [r["total_ms"] for r in results_b]

    label_a = args.config_a
    label_b = args.config_b

    print(f"\n  Accuracy:  {label_a} {pass_a}/{total}  |  {label_b} {pass_b}/{total}")
    print(f"\n  --- Latency Breakdown (avg) ---")
    print(f"  {'Stage':<20} {label_a:>16} {label_b:>16}")
    print(f"  {'-'*52}")
    print(f"  {'Agent 2 (LLM)':<20} {avg(a2_ms_a):>15.0f}ms {avg(a2_ms_b):>15.0f}ms")
    print(f"  {'Total':<20} {avg(total_ms_a):>15.0f}ms {avg(total_ms_b):>15.0f}ms")
    print(f"\n  --- Latency Breakdown (median) ---")
    print(f"  {'Stage':<20} {label_a:>16} {label_b:>16}")
    print(f"  {'-'*52}")
    if a2_ms_a and a2_ms_b:
        print(f"  {'Agent 2 (LLM)':<20} {med(a2_ms_a):>15.0f}ms {med(a2_ms_b):>15.0f}ms")
    print(f"  {'Total':<20} {med(total_ms_a):>15.0f}ms {med(total_ms_b):>15.0f}ms")

    # Per-case diff
    diffs = []
    for ra, rb in zip(results_a, results_b):
        if ra["final_brand"] != rb["final_brand"]:
            diffs.append((ra["case"], ra["final_brand"], rb["final_brand"], ra["expected"]))

    if diffs:
        print(f"\n  Output differences ({len(diffs)} cases):")
        print(f"  {'Case':<12} {'Expected':<25} {label_a:<25} {label_b:<25}")
        print(f"  {'-'*87}")
        for case, brand_a, brand_b, exp in diffs:
            print(f"  {case:<12} {exp:<25} {brand_a or 'None':<25} {brand_b or 'None':<25}")
    else:
        print(f"\n  No output differences — both configs returned identical brands.")

    # Confidence comparison
    conf_a = [r["confidence"] for r in results_a if r["confidence"] and r["confidence"] > 0]
    conf_b = [r["confidence"] for r in results_b if r["confidence"] and r["confidence"] > 0]
    if conf_a and conf_b:
        print(f"\n  Confidence (avg): {label_a} {avg(conf_a):.3f}  |  {label_b} {avg(conf_b):.3f}")
        print(f"  Confidence (med): {label_a} {med(conf_a):.3f}  |  {label_b} {med(conf_b):.3f}")

    # Save results
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"brand_config_compare_{ts}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fields = ["case", "expected",
                  "a_brand", "a_brand_id", "a_confidence", "a_agent2_ms", "a_passed",
                  "b_brand", "b_brand_id", "b_confidence", "b_agent2_ms", "b_passed",
                  "brands_match"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ra, rb in zip(results_a, results_b):
            writer.writerow({
                "case": ra["case"], "expected": ra["expected"],
                "a_brand": ra["final_brand"], "a_brand_id": ra["final_brand_id"],
                "a_confidence": ra["confidence"], "a_agent2_ms": ra["agent2_ms"],
                "a_passed": ra["passed"],
                "b_brand": rb["final_brand"], "b_brand_id": rb["final_brand_id"],
                "b_confidence": rb["confidence"], "b_agent2_ms": rb["agent2_ms"],
                "b_passed": rb["passed"],
                "brands_match": ra["final_brand"] == rb["final_brand"],
            })

    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
