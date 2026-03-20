#!/usr/bin/env python3
"""
Brand Regex vs Agent 1 Comparison Test
=======================================
Runs synthetic CSV A through both pipelines and compares:
  - Latency (regex extraction vs Agent 1 LLM+DB)
  - Output values (final_brand, final_brand_id, confidence)
  - Unique brands proposed to Agent 2

Usage:
    conda run -n pyds python tests/soft_tests/brand_regex_vs_agent_test.py -m local
    conda run -n pyds python tests/soft_tests/brand_regex_vs_agent_test.py -m local --workers 2 --verbose
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(tempfile.gettempdir(), "gcp_sa.json")

parser = argparse.ArgumentParser(description="Brand regex vs agent1 comparison on CSV A")
parser.add_argument("--mode", "-m", default="local", choices=["local"], help="Test mode (local only)")
parser.add_argument("--workers", "-w", type=int, default=1, help="Parallel workers (default 1 for fair latency comparison)")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--regex-only", action="store_true", help="Only run regex pipeline (skip Agent 1, no GCP needed)")
parser.add_argument("--limit", "-n", type=int, default=None, help="Limit to first N cases")
args = parser.parse_args()

if not args.verbose:
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("agent_orchestration", "agent_architecture", "core", "google_genai",
                 "httpx", "botocore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# test_input_data is not tracked in git — try worktree first, fall back to main IFL repo
DATA_DIR = REPO_ROOT / "tests" / "test_input_data"
if not DATA_DIR.exists():
    DATA_DIR = Path("C:/Users/dce/documents/TRUSS/Github/truss-annotation-intelligent-function-layer/tests/test_input_data")
SERVICE_PATH = REPO_ROOT / "services" / "automated-annotation"

# Setup imports
sys.path.insert(0, str(SERVICE_PATH))

from core.utils.credentials import ensure_gcp_adc
ensure_gcp_adc()

from agent_orchestration.brand_classification_orchestration import run_brand_classification_workflow
from agent_orchestration.regex_brand_lookup import BrandMasterIndex


def load_cases_A():
    csv_path = DATA_DIR / "synthetic_A_brand.csv"
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            exp = row.get("ground_truth_brand_text_classifier", "").strip()
            exp = None if exp.lower() in ("null", "", "unknown") else exp
            cases.append((f"synA-{i+1}", {
                "Title": row.get("title", ""),
                "description": row.get("description", ""),
            }, exp))
    return cases


def run_single(case_name, text_dump, exp_brand, use_regex, brand_index, verbose):
    """Run one case through a pipeline, return timing + result."""
    raw_text = json.dumps(text_dump, indent=2)

    t0 = time.perf_counter()
    result = run_brand_classification_workflow(
        raw_text=raw_text,
        name=text_dump.get("Title"),
        env="staging",
        verbose=verbose,
        brand_index=brand_index,
        use_regex=use_regex,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    success = result.get("workflow_status") == "success"
    final_brand = result.get("final_brand") if success else "Unknown"
    final_brand_id = result.get("final_brand_id") if success else None
    confidence = result.get("confidence", 0) if success else 0

    # Count unique brands proposed to Agent 2
    agent1_res = result.get("agent1_result", {})
    matched_brands = agent1_res.get("matched_brands", [])
    unique_proposed = len(matched_brands)

    extraction_method = result.get("extraction_method", "unknown")
    regex_ms = agent1_res.get("regex_elapsed_ms", None)
    stage_timing = result.get("stage_timing", {})

    passed = False
    if exp_brand is None:
        passed = final_brand is None or (isinstance(final_brand, str) and final_brand.lower() in ("unknown", "none", "", "null"))
    else:
        passed = (final_brand == exp_brand)

    return {
        "case": case_name,
        "extraction_method": extraction_method,
        "total_ms": round(elapsed_ms, 1),
        "extraction_ms": stage_timing.get("extraction_ms"),
        "agent2_ms": stage_timing.get("agent2_ms"),
        "regex_ms": round(regex_ms, 2) if regex_ms is not None else None,
        "unique_brands_proposed": unique_proposed,
        "final_brand": final_brand,
        "final_brand_id": final_brand_id,
        "confidence": confidence,
        "expected": exp_brand or "Unknown",
        "passed": passed,
        "error": result.get("error"),
    }


def main():
    cases = load_cases_A()
    if args.limit:
        cases = cases[:args.limit]
    print(f"Loaded {len(cases)} cases from CSV A\n")

    # Load brand index once
    print("Loading brand master index...")
    t0 = time.perf_counter()
    brand_index = BrandMasterIndex()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  {brand_index.total_patterns} patterns loaded in {load_ms:.0f}ms\n")

    max_workers = args.workers
    verbose = args.verbose

    # --- Run REGEX pipeline ---
    print(f"{'='*70}")
    print(f"PIPELINE 1: REGEX CSV LOOKUP → Agent 2")
    print(f"{'='*70}")
    regex_results = []
    for i, (name, td, exp) in enumerate(cases):
        r = run_single(name, td, exp, use_regex=True, brand_index=brand_index, verbose=verbose)
        regex_results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{i+1}/{len(cases)}] {name}: {status} | {r['total_ms']:.0f}ms | "
              f"{r['unique_brands_proposed']} brands → {r['final_brand']}")

    # --- Run AGENT1 pipeline ---
    print(f"\n{'='*70}")
    print(f"PIPELINE 2: AGENT 1 LLM+DB → Agent 2")
    print(f"{'='*70}")
    agent_results = []
    for i, (name, td, exp) in enumerate(cases):
        r = run_single(name, td, exp, use_regex=False, brand_index=None, verbose=verbose)
        agent_results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{i+1}/{len(cases)}] {name}: {status} | {r['total_ms']:.0f}ms | "
              f"{r['unique_brands_proposed']} brands → {r['final_brand']}")

    # --- Comparison ---
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    regex_pass = sum(1 for r in regex_results if r["passed"])
    agent_pass = sum(1 for r in agent_results if r["passed"])
    total = len(cases)

    def avg(lst): return sum(lst) / len(lst) if lst else 0
    def med(lst):
        s = sorted(lst)
        n = len(s)
        return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2
    def safe_list(results, key): return [r[key] for r in results if r[key] is not None]

    regex_total_ms = [r["total_ms"] for r in regex_results]
    agent_total_ms = [r["total_ms"] for r in agent_results]
    regex_extract_ms = safe_list(regex_results, "extraction_ms")
    agent_extract_ms = safe_list(agent_results, "extraction_ms")
    regex_agent2_ms = safe_list(regex_results, "agent2_ms")
    agent_agent2_ms = safe_list(agent_results, "agent2_ms")
    regex_brands_proposed = [r["unique_brands_proposed"] for r in regex_results]
    agent_brands_proposed = [r["unique_brands_proposed"] for r in agent_results]

    print(f"\n  Accuracy:        Regex {regex_pass}/{total}  |  Agent1 {agent_pass}/{total}")
    print(f"\n  --- Latency Breakdown (avg) ---")
    print(f"  {'Stage':<20} {'Regex':>10} {'Agent1':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Extraction':<20} {avg(regex_extract_ms):>9.0f}ms {avg(agent_extract_ms):>9.0f}ms")
    print(f"  {'Agent 2 (LLM)':<20} {avg(regex_agent2_ms):>9.0f}ms {avg(agent_agent2_ms):>9.0f}ms")
    print(f"  {'Total':<20} {avg(regex_total_ms):>9.0f}ms {avg(agent_total_ms):>9.0f}ms")
    print(f"\n  --- Latency Breakdown (median) ---")
    print(f"  {'Stage':<20} {'Regex':>10} {'Agent1':>10}")
    print(f"  {'-'*40}")
    if regex_extract_ms and agent_extract_ms:
        print(f"  {'Extraction':<20} {med(regex_extract_ms):>9.0f}ms {med(agent_extract_ms):>9.0f}ms")
    if regex_agent2_ms and agent_agent2_ms:
        print(f"  {'Agent 2 (LLM)':<20} {med(regex_agent2_ms):>9.0f}ms {med(agent_agent2_ms):>9.0f}ms")
    print(f"  {'Total':<20} {med(regex_total_ms):>9.0f}ms {med(agent_total_ms):>9.0f}ms")
    print(f"\n  Brands proposed (avg): Regex {avg(regex_brands_proposed):.1f}  |  Agent1 {avg(agent_brands_proposed):.1f}")
    print(f"  Brands proposed (med): Regex {med(regex_brands_proposed):.0f}  |  Agent1 {med(agent_brands_proposed):.0f}")

    # Per-case diff
    diffs = []
    for rx, ag in zip(regex_results, agent_results):
        if rx["final_brand"] != ag["final_brand"]:
            diffs.append((rx["case"], rx["final_brand"], ag["final_brand"], rx["expected"]))

    if diffs:
        print(f"\n  Output differences ({len(diffs)} cases):")
        print(f"  {'Case':<12} {'Expected':<25} {'Regex':<25} {'Agent1':<25}")
        print(f"  {'-'*87}")
        for case, rx_brand, ag_brand, exp in diffs:
            print(f"  {case:<12} {exp:<25} {rx_brand or 'None':<25} {ag_brand or 'None':<25}")
    else:
        print(f"\n  No output differences — both pipelines returned identical brands.")

    # Save detailed results
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"brand_regex_vs_agent_{ts}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fields = ["case", "expected", "regex_brand", "regex_brand_id", "regex_confidence",
                  "regex_ms", "regex_brands_proposed", "regex_passed",
                  "agent_brand", "agent_brand_id", "agent_confidence",
                  "agent_ms", "agent_brands_proposed", "agent_passed", "brands_match"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rx, ag in zip(regex_results, agent_results):
            writer.writerow({
                "case": rx["case"],
                "expected": rx["expected"],
                "regex_brand": rx["final_brand"],
                "regex_brand_id": rx["final_brand_id"],
                "regex_confidence": rx["confidence"],
                "regex_ms": rx["total_ms"],
                "regex_brands_proposed": rx["unique_brands_proposed"],
                "regex_passed": rx["passed"],
                "agent_brand": ag["final_brand"],
                "agent_brand_id": ag["final_brand_id"],
                "agent_confidence": ag["confidence"],
                "agent_ms": ag["total_ms"],
                "agent_brands_proposed": ag["unique_brands_proposed"],
                "agent_passed": ag["passed"],
                "brands_match": rx["final_brand"] == ag["final_brand"],
            })

    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
