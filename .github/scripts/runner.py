#!/usr/bin/env python3
"""
Lightweight API Contract Test Runner

Discovers and runs test files (.py, .js) in tests/ folders.
Outputs results as JSON for GitHub Actions to post as PR comments.

Usage:
    python runner.py [--base-url URL] [--output FORMAT]

Options:
    --base-url      Base URL for API tests (default: from env STAGING_API_URL)
    --output        Output format: json, markdown, or summary (default: markdown)
    --dir           Directory to search for tests (default: current directory)
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional


@dataclass
class TestResult:
    name: str
    file: str
    passed: bool
    duration_ms: float
    output: str
    error: Optional[str] = None


@dataclass
class TestRun:
    timestamp: str
    base_url: str
    total: int
    passed: int
    failed: int
    results: list[TestResult]


def discover_tests(base_dir: Path) -> list[tuple[Path, Path]]:
    """
    Find all test pairs (.txt + .py/.js) in tests/ folders.
    Returns list of (txt_file, code_file) tuples.
    """
    test_pairs = []
    
    for tests_dir in base_dir.rglob("tests"):
        if not tests_dir.is_dir():
            continue
        
        # Find all .txt files
        txt_files = {f.stem: f for f in tests_dir.glob("*.txt")}
        
        # Match with .py or .js files
        for stem, txt_file in txt_files.items():
            py_file = tests_dir / f"{stem}.py"
            js_file = tests_dir / f"{stem}.js"
            
            if py_file.exists():
                test_pairs.append((txt_file, py_file))
            elif js_file.exists():
                test_pairs.append((txt_file, js_file))
            else:
                print(f"Warning: {txt_file} has no matching .py or .js file", file=sys.stderr)
    
    return sorted(test_pairs, key=lambda x: str(x[1]))


def run_python_test(test_file: Path, base_url: str) -> TestResult:
    """Execute a Python test file."""
    start = datetime.now()
    
    env = os.environ.copy()
    env["STAGING_API_URL"] = base_url
    env["API_BASE_URL"] = base_url
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=test_file.parent
        )
        
        duration = (datetime.now() - start).total_seconds() * 1000
        
        return TestResult(
            name=test_file.stem,
            file=str(test_file),
            passed=result.returncode == 0,
            duration_ms=round(duration, 2),
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None
        )
    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name=test_file.stem,
            file=str(test_file),
            passed=False,
            duration_ms=round(duration, 2),
            output="",
            error="Test timed out after 60 seconds"
        )
    except Exception as e:
        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name=test_file.stem,
            file=str(test_file),
            passed=False,
            duration_ms=round(duration, 2),
            output="",
            error=str(e)
        )


def run_js_test(test_file: Path, base_url: str) -> TestResult:
    """Execute a JavaScript test file."""
    start = datetime.now()
    
    env = os.environ.copy()
    env["STAGING_API_URL"] = base_url
    env["API_BASE_URL"] = base_url
    
    try:
        result = subprocess.run(
            ["node", str(test_file)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=test_file.parent
        )
        
        duration = (datetime.now() - start).total_seconds() * 1000
        
        return TestResult(
            name=test_file.stem,
            file=str(test_file),
            passed=result.returncode == 0,
            duration_ms=round(duration, 2),
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None
        )
    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name=test_file.stem,
            file=str(test_file),
            passed=False,
            duration_ms=round(duration, 2),
            output="",
            error="Test timed out after 60 seconds"
        )
    except Exception as e:
        duration = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            name=test_file.stem,
            file=str(test_file),
            passed=False,
            duration_ms=round(duration, 2),
            output="",
            error=str(e)
        )


def run_test(txt_file: Path, code_file: Path, base_url: str) -> TestResult:
    """Run a single test based on file extension."""
    if code_file.suffix == ".py":
        return run_python_test(code_file, base_url)
    elif code_file.suffix == ".js":
        return run_js_test(code_file, base_url)
    else:
        return TestResult(
            name=code_file.stem,
            file=str(code_file),
            passed=False,
            duration_ms=0,
            output="",
            error=f"Unknown file type: {code_file.suffix}"
        )


def format_markdown(test_run: TestRun) -> str:
    """Format test results as markdown for PR comment."""
    lines = []
    
    # Header with summary
    if test_run.failed == 0:
        lines.append(f"## ✅ All Tests Passed ({test_run.passed}/{test_run.total})")
    else:
        lines.append(f"## ❌ Tests Failed ({test_run.failed}/{test_run.total} failed)")
    
    lines.append("")
    lines.append(f"**API:** `{test_run.base_url}`")
    lines.append(f"**Time:** {test_run.timestamp}")
    lines.append("")
    
    # Results table
    lines.append("| Status | Test | Duration |")
    lines.append("|--------|------|----------|")
    
    for r in test_run.results:
        status = "✅" if r.passed else "❌"
        lines.append(f"| {status} | `{r.name}` | {r.duration_ms}ms |")
    
    # Details for failed tests
    failed = [r for r in test_run.results if not r.passed]
    if failed:
        lines.append("")
        lines.append("### Failures")
        lines.append("")
        
        for r in failed:
            lines.append(f"<details>")
            lines.append(f"<summary><code>{r.name}</code></summary>")
            lines.append("")
            lines.append(f"**File:** `{r.file}`")
            lines.append("")
            if r.error:
                lines.append("**Error:**")
                lines.append("```")
                lines.append(r.error.strip())
                lines.append("```")
            if r.output:
                lines.append("**Output:**")
                lines.append("```")
                lines.append(r.output.strip())
                lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")
    
    return "\n".join(lines)


def format_summary(test_run: TestRun) -> str:
    """Format test results as one-line summary."""
    if test_run.failed == 0:
        return f"✅ {test_run.passed}/{test_run.total} tests passed"
    else:
        return f"❌ {test_run.failed}/{test_run.total} tests failed"


def main():
    parser = argparse.ArgumentParser(description="Run API contract tests")
    parser.add_argument("--base-url", default=os.environ.get("STAGING_API_URL", "http://localhost:3000"))
    parser.add_argument("--output", choices=["json", "markdown", "summary"], default="markdown")
    parser.add_argument("--dir", default=".")
    args = parser.parse_args()
    
    base_dir = Path(args.dir).resolve()
    
    # Discover tests
    test_pairs = discover_tests(base_dir)
    
    if not test_pairs:
        print("No tests found", file=sys.stderr)
        sys.exit(0)
    
    # Run tests
    results = []
    for txt_file, code_file in test_pairs:
        result = run_test(txt_file, code_file, args.base_url)
        results.append(result)
        
        # Print progress
        status = "✓" if result.passed else "✗"
        print(f"{status} {result.name} ({result.duration_ms}ms)", file=sys.stderr)
    
    # Build test run summary
    test_run = TestRun(
        timestamp=datetime.now().isoformat(),
        base_url=args.base_url,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        results=results
    )
    
    # Output results
    if args.output == "json":
        print(json.dumps(asdict(test_run), indent=2))
    elif args.output == "markdown":
        print(format_markdown(test_run))
    else:
        print(format_summary(test_run))
    
    # Exit with failure code if any tests failed
    sys.exit(1 if test_run.failed > 0 else 0)


if __name__ == "__main__":
    main()
