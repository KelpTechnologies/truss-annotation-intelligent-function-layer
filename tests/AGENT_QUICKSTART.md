# Agent Quickstart: Test Development

## Summary

1. Eval tests repo state (glob *.txt, *.py)
2. Read: README.md, TXT_TO_CODE_GUIDE.md, project_readme.md
3. Map txt→py files, identify gaps
4. Build missing py files from txt specs
5. Validate existing py files (check imports, run local mode)
6. Use parallel subagents per txt file
7. Output terminal report with changes summary
8. Print local test commands
9. Update project context (project_readme.md) with learnings

---

## Step 1: Evaluate Test Repo State

```bash
# Glob all test files
Glob: tests/*.txt
Glob: tests/*.py
```

Expected structure:
```
tests/
├── README.md              # Framework overview
├── TXT_TO_CODE_GUIDE.md   # Conversion guide (CRITICAL)
├── project_readme.md      # Project-specific patterns
├── .env.example           # Env vars template
├── .env                   # Local env vars (git-ignored)
├── test_input_data/       # CSV fixtures
├── <name>_test.txt        # Test specs
└── <name>_test.py         # Test implementations
```

---

## Step 2: Read Documentation

Read in order (stop if file missing):
1. `tests/README.md` - framework workflow
2. `tests/TXT_TO_CODE_GUIDE.md` - txt→code conversion rules
3. `tests/project_readme.md` - project-specific response wrappers, patterns

Key takeaways to extract:
- Response format: `response["data"][0]` wrapper
- Unknown values: `{property}_id: 0`, `{property}: "Unknown"`
- Auth: `x-api-key` header with `DSL_API_KEY`
- Modes: `--mode api` (staging) vs `--mode local` (direct lambda)

---

## Step 3: Identify txt↔py Mapping

Create mapping table:

| txt file | py file | status |
|----------|---------|--------|
| colour_test.txt | colour_test.py | EXISTS |
| model_test.txt | model_test.py | EXISTS |
| brand_text_test.txt | ? | MISSING |

Files ending in `_old.txt` are deprecated - ignore.

---

## Step 4: Build Missing Python Files

For each MISSING py file:

### 4.1 Parse the txt file sections

Extract from txt:
- `# INPUT`: endpoint, method, request body format
- `# OUTPUT`: expected response fields, unknown handling
- `# INTERNAL LOGIC`: business rules, edge cases
- Test case arrays: `text_dump_arr` with expected outputs

### 4.2 Use template structure

```python
#!/usr/bin/env python3
"""
Test: {filename} - {property} classification tests (TEXT-ONLY)

Usage:
    conda run -n pyds python tests/{filename}.py -m local
    conda run -n pyds python tests/{filename}.py -m api
"""

import argparse
import sys
import math
import json
import os
from pathlib import Path

# Load .env from tests folder
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = value

load_env()

parser = argparse.ArgumentParser(description="{Property} classification tests")
parser.add_argument("--mode", "-m", default="api", choices=["api", "local"])
parser.add_argument("--base-url", "-u", help="API base URL")
parser.add_argument("--api-key", "-k", help="API key")
args = parser.parse_args()

MODE = args.mode
BASE_URL = args.base_url or os.environ.get("STAGING_API_URL") or os.environ.get("API_BASE_URL")
BASE_URL = BASE_URL.rstrip("/") if BASE_URL else None
API_KEY = args.api_key or os.environ.get("DSL_API_KEY") or ""
RESULTS = []

# Config ID for text-only classification
CONFIG_ID = "classifier-{property}-bags-text"


def classify_via_api(text_dump: dict) -> dict:
    import requests
    if not BASE_URL:
        raise ValueError("--base-url required or set STAGING_API_URL")
    response = requests.post(
        f"{BASE_URL}/automations/annotation/bags/classify/{property}",
        json={"text_dump": text_dump, "input_mode": "text-only"},
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    return response.json()["data"][0]


def classify_via_local(text_dump: dict) -> dict:
    repo_root = Path(__file__).parent.parent
    service_path = repo_root / "services" / "automated-annotation"
    sys.path.insert(0, str(service_path))

    from core.utils.credentials import ensure_gcp_adc
    ensure_gcp_adc()

    from agent_orchestration.classifier_api_orchestration import classify_for_api

    text_input = json.dumps(text_dump, indent=2)
    result = classify_for_api(
        config_id=CONFIG_ID,
        text_input=text_input,
        input_mode="text-only",
        env="staging"
    )

    return {
        "{property}": result.get("primary_name"),
        "{property}_id": result.get("primary_id"),
        # Add root fields if applicable
    }


def classify(text_dump: dict) -> dict:
    return classify_via_local(text_dump) if MODE == "local" else classify_via_api(text_dump)


def is_unknown(value):
    if value is None: return True
    if value == 0: return True
    if isinstance(value, str) and value.lower() in ("unknown", "nan", ""): return True
    if isinstance(value, float) and math.isnan(value): return True
    return False


def run_case(name: str, text_dump: dict, exp_val, exp_id):
    # Implement test case runner (see material_test.py for pattern)
    pass


# === TESTS ===
# Add test functions matching txt sections


if __name__ == "__main__":
    print(f"=== {PROPERTY} TESTS (mode={MODE}) ===\n")
    # Call test functions
    # Print summary
```

### 4.3 Key patterns from existing tests

Study these files for patterns:
- `material_test.py` - full text-only classifier test
- `colour_test.py` - colour classification
- `model_test.py` - model classification

---

## Step 5: Validate Existing Tests

For each EXISTING py file:

### 5.1 Check lambda imports

Read the py file and trace imports:
```python
from agent_orchestration.classifier_api_orchestration import classify_for_api
from core.utils.credentials import ensure_gcp_adc
```

Verify these paths exist in:
```
services/automated-annotation/
├── agent_orchestration/classifier_api_orchestration.py
└── core/utils/credentials.py
```

### 5.2 Run local mode test

```bash
conda run -n pyds python tests/{name}_test.py -m local
```

Check for:
- Import errors (path issues)
- Missing env vars (validate_env failures)
- API errors (config not found)
- Test failures (assertion errors)

### 5.3 Document issues found

If errors occur, note:
- Error type (import, env, API, assertion)
- Root cause
- Fix required (if any)

---

## Step 5a: Parallel Subagent Execution

Launch one subagent per txt file using Task tool:

```
For txt files WITHOUT matching py:
  Task: "Build {name}_test.py from {name}_test.txt"
  - Read txt file
  - Generate py file using template
  - Run local test to validate

For txt files WITH matching py:
  Task: "Validate {name}_test.py"
  - Check imports resolve
  - Run local mode
  - Report errors
```

Example parallel launch:
```python
# Launch all subagents in single message
Task(prompt="Build brand_text_test.py from brand_text_test.txt spec", subagent_type="general-purpose")
Task(prompt="Validate colour_test.py - check imports, run local", subagent_type="general-purpose")
Task(prompt="Validate model_test.py - check imports, run local", subagent_type="general-purpose")
```

---

## Step 6: Output Report

Print terminal report:

```
=== TEST DEVELOPMENT REPORT ===

| File | Status | Changes | Notes |
|------|--------|---------|-------|
| colour_test.py | VALIDATED | None | All tests pass |
| model_test.py | VALIDATED | None | 2 failures (model mismatch) |
| brand_text_test.py | CREATED | +180 lines | New file from txt spec |
| material_test.py | VALIDATED | None | All tests pass |

Summary:
- Created: 1 file
- Validated: 3 files
- Failures found: 1 file (model_test.py)

Issues:
- model_test.py line 45: test_simple_model fails - got "Speedy 30" expected "Speedy 25"
```

---

## Step 7: Local Test Commands

Print commands for user:

```bash
# Run individual tests in local mode
conda run -n pyds python tests/material_test.py -m local
conda run -n pyds python tests/colour_test.py -m local
conda run -n pyds python tests/model_test.py -m local
conda run -n pyds python tests/brand_text_test.py -m local

# Run individual tests in API mode (requires staging URL)
conda run -n pyds python tests/material_test.py -m api

# Run all tests via runner (API mode only)
STAGING_API_URL=https://your-url python .github/scripts/runner.py
```

---

## Environment Setup

Before running local mode:

```bash
# 1. Copy env template
cp tests/.env.example tests/.env

# 2. Edit tests/.env with values:
#    - AWS_REGION=eu-west-2
#    - TRUSS_SECRETS_ARN=arn:aws:secretsmanager:...
#    - VERTEXAI_PROJECT=truss-data-science
#    - VERTEXAI_LOCATION=us-central1

# 3. Ensure AWS credentials configured
aws configure  # or set AWS_PROFILE
```

---

## Common Issues

### Import Error: No module named 'agent_orchestration'

Fix: sys.path not set correctly. Ensure:
```python
repo_root = Path(__file__).parent.parent
service_path = repo_root / "services" / "automated-annotation"
sys.path.insert(0, str(service_path))
```

### Missing GOOGLE_APPLICATION_CREDENTIALS

Fix: Call `ensure_gcp_adc()` before imports needing GCP:
```python
from core.utils.credentials import ensure_gcp_adc
ensure_gcp_adc()
```

### Test returns ID 0 when expecting value

This is correct for unknown cases. Use `is_unknown()` helper:
```python
def is_unknown(value):
    return value is None or value == 0 or value in ("Unknown", "unknown", "")
```

---

## Step 8: Update Project Context

After test development/validation complete, update `project_readme.md` with learnings.

### 8.1 Gather context to add

Collect from this session:
- New patterns discovered (response formats, edge cases)
- New test-specific context (endpoints, fields, behaviors)
- Gotchas or workarounds found during implementation
- Any corrections to existing documentation

### 8.2 Review existing project_readme.md

Read `tests/project_readme.md` and check for:
- **Outdated info**: patterns that no longer apply
- **Missing context**: gaps based on current test work
- **Incorrect examples**: code samples that don't match actual behavior
- **Incomplete sections**: areas that need expansion

### 8.3 Propose changes (if any)

If issues found, present to user:

```
=== PROJECT README UPDATES ===

SUGGESTED CHANGES:
1. [Section: X] - Issue: Y
   Why: Z
   Proposed fix: ...

2. [Section: X] - Issue: Y
   ...

Approve changes? (y/n)
```

Use AskUserQuestion tool for approval:
```python
AskUserQuestion(
    questions=[{
        "header": "Doc updates",
        "question": "Approve these project_readme.md changes?",
        "options": [
            {"label": "Yes, apply all", "description": "Apply all suggested changes"},
            {"label": "No, skip", "description": "Keep current documentation"}
        ],
        "multiSelect": False
    }]
)
```

### 8.4 Apply approved updates

On approval, append new context to relevant sections:
- Add new patterns under "Classification API Patterns"
- Add new env vars under "Environment Variables"
- Add new test examples under "Complete Classification Test Example"
- Add new gotchas under relevant sections

### 8.5 Context template for new tests

When adding context for a new test file, include:

```markdown
### {Property} Classification Context

**Endpoint**: `POST /automations/annotation/{category}/classify/{property}`
**Config ID**: `classifier-{property}-{category}-text`

**Response fields**:
| Field | Description |
|-------|-------------|
| `{property}` | Classified value |
| `{property}_id` | Taxonomy ID |

**Edge cases**:
- [List any discovered edge cases]

**Notes**:
- [Any implementation-specific notes]
```

---

## File Reference

| File | Purpose |
|------|---------|
| README.md | Framework overview |
| TXT_TO_CODE_GUIDE.md | txt→code conversion rules |
| project_readme.md | Project-specific patterns |
| .env.example | Env vars template |
| material_test.py | Reference implementation |
