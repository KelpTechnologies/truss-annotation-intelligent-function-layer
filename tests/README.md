# Lightweight API Contract Testing Framework

A minimal testing framework for validating API contracts across multiple repos. Tests are defined in natural language (`.txt`) and implemented in Python or JavaScript.

**Related docs:**
- [TXT_TO_CODE_GUIDE.md](./TXT_TO_CODE_GUIDE.md) - Detailed guide for converting .txt specs to test code
- [project_readme.md](./project_readme.md) - Project-specific API patterns and examples (if exists)

## Quick Start

### 1. Copy files to your repo

```bash
# Copy the workflow
mkdir -p .github/workflows .github/scripts
cp integration-tests.yml .github/workflows/
cp runner.py .github/scripts/

# Create your first test folder
mkdir -p your-lambda/tests
```

### 2. Add a test

**Data team creates: `your-lambda/tests/001-my-feature.txt`**
```
# INPUT
- API endpoint: POST /api/v1/my-endpoint
- Request body:
  - field1: "value1"
  - field2: 123

# OUTPUT
- HTTP status: 200
- Response body:
  - result: "expected_value"

# INTERNAL LOGIC
- Describe what should happen internally
- Business rules that must be respected
- Edge cases to handle
```

**Dev team creates: `your-lambda/tests/001-my-feature.py`**
```python
#!/usr/bin/env python3
import os
import sys
import requests

BASE_URL = os.environ.get("API_BASE_URL")

def test_my_feature():
    response = requests.post(
        f"{BASE_URL}/api/v1/my-endpoint",
        json={"field1": "value1", "field2": 123}
    )
    
    assert response.status_code == 200
    assert response.json()["result"] == "expected_value"
    print("✓ Test passed")

if __name__ == "__main__":
    try:
        test_my_feature()
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
```

### 3. Run tests

**Manually from GitHub UI:**
1. Go to Actions tab
2. Select "Integration Tests"
3. Click "Run workflow"
4. Enter staging URL and optional PR number

**Via API:**
```bash
curl -X POST \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/OWNER/REPO/dispatches \
  -d '{
    "event_type": "run-tests",
    "client_payload": {
      "base_url": "https://staging.example.com",
      "pr_number": "123"
    }
  }'
```

**Locally:**
```bash
STAGING_API_URL=https://staging.example.com python .github/scripts/runner.py
```

---

## File Structure

```
your-repo/
├── .github/
│   ├── workflows/
│   │   └── integration-tests.yml    # GitHub Action
│   └── scripts/
│       └── runner.py                # Test runner
├── tests/
│   ├── README.md                    # This file (generic)
│   ├── TXT_TO_CODE_GUIDE.md         # Conversion guide (generic)
│   ├── project_readme.md            # Project-specific patterns (optional)
│   ├── 001-feature-a.txt            # Test intent (data team)
│   ├── 001-feature-a.py             # Test code (dev team)
│   └── test_input_data/             # Test fixtures
├── services/
│   └── my-service/
│       └── handler.py
└── ...
```

---

## .txt File Format

Every `.txt` file MUST have these three sections:

### # INPUT
Describe what goes into the API:
- Endpoint (method + path)
- Request body / query params
- Headers
- Preconditions (database state, etc.)

### # OUTPUT
Describe what should come out:
- HTTP status code
- Response body structure
- Expected values

### # INTERNAL LOGIC
Describe the business rules:
- How inputs transform to outputs
- Edge cases
- What should NOT happen
- Which tables/services are involved

---

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  DATA TEAM                                                  │
│  1. Write .txt file with INPUT/OUTPUT/INTERNAL LOGIC        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  DEV TEAM                                                   │
│  2. Implement feature                                       │
│  3. Write .py/.js test matching the .txt spec               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  DATA TEAM                                                  │
│  4. Review test code to confirm it matches intent           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  CI/CD                                                      │
│  5. Deploy to staging                                       │
│  6. Trigger test run                                        │
│  7. Results posted to PR                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `STAGING_API_URL` | Base URL for the staging API |
| `API_BASE_URL` | Alternative to STAGING_API_URL |
| `TEST_AUTH_TOKEN` | Auth token for protected endpoints (optional) |

---

## Test Output

Tests results are posted as PR comments:

```
## ✅ All Tests Passed (3/3)

**API:** `https://staging.example.com`
**Time:** 2024-01-15T10:30:00

| Status | Test | Duration |
|--------|------|----------|
| ✅ | `001-bulk-discount` | 245ms |
| ✅ | `002-zero-quantity` | 123ms |
| ✅ | `001-context-retrieval` | 456ms |
```

Or if tests fail:

```
## ❌ Tests Failed (1/3 failed)

...

### Failures

<details>
<summary><code>002-zero-quantity</code></summary>

**Error:**
Expected status 400, got 200
</details>
```

---

## Upgrading to Automatic Triggers

Once stable, switch from manual/API triggers to automatic:

```yaml
# .github/workflows/integration-tests.yml
on:
  workflow_run:
    workflows: ["Deploy to Staging"]
    types: [completed]
```

This watches your deploy workflow and runs tests automatically after each staging deployment.

---

## Tips

- **Keep tests atomic**: One test file = one focused behavior
- **Name files clearly**: `001-bulk-discount-over-100-units.txt` > `001-test.txt`
- **Include edge cases in .txt**: The INTERNAL LOGIC section is where nuance lives
- **Test the contract, not implementation**: Focus on inputs and outputs, not how it works internally
