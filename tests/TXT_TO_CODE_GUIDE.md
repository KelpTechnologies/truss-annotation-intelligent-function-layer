# Test Implementation Guide for Coding Agents

This document explains how to convert natural language test definitions (`.txt` files) into executable test code (`.py` or `.js` files).

**Related docs:**
- [README.md](./README.md) - Framework overview and workflow
- [project_readme.md](./project_readme.md) - Project-specific API patterns (if exists)

---

## Overview

You will receive a `.txt` file with three sections: `# INPUT`, `# OUTPUT`, and `# INTERNAL LOGIC`. Your job is to create a matching test file that:

1. Makes the API call described in INPUT
2. Asserts the response matches OUTPUT
3. Validates any constraints from INTERNAL LOGIC

The test file must:
- Have the same filename stem as the `.txt` file (e.g., `001-my-test.txt` → `001-my-test.py`)
- Exit with code 0 on success, code 1 on failure
- Print clear error messages on failure

---

## Reading the .txt Format

### # INPUT Section

This describes what to send to the API. Extract:

| Look for | Maps to |
|----------|---------|
| "API endpoint: METHOD /path" | HTTP method and URL path |
| "Request body:" | JSON body for POST/PUT/PATCH |
| "Query parameters:" | URL query string |
| "Headers:" | HTTP headers |
| "Preconditions:" | Test setup or assumptions (document only, may not be testable) |

### # OUTPUT Section

This describes what the API should return. Extract:

| Look for | Maps to |
|----------|---------|
| "HTTP status: NNN" | Assert response status code |
| "Response body:" | Assert response JSON structure and values |
| Specific field values | Assert exact matches |
| Field types | Assert type checking |

### # INTERNAL LOGIC Section

This describes business rules and edge cases. Use this to:

| Look for | Maps to |
|----------|---------|
| Calculation rules | Verify computed values are correct |
| Boundary conditions | Add edge case assertions |
| "should NOT" statements | Add negative assertions |
| Error conditions | Add tests for error responses |
| Table/service references | Document in test comments (helps reviewers) |

---

## Python Test Template

```python
#!/usr/bin/env python3
"""
Test: {filename}
{Brief description from the .txt file}
"""

import os
import sys
import requests

# Get API base URL from environment
BASE_URL = os.environ.get("API_BASE_URL") or os.environ.get("STAGING_API_URL")

if not BASE_URL:
    print("ERROR: API_BASE_URL or STAGING_API_URL environment variable not set")
    sys.exit(1)

# Optional: Auth token if needed
AUTH_TOKEN = os.environ.get("TEST_AUTH_TOKEN", "")

# Collect test results (don't print during execution)
test_results = []  # List of {"name": str, "passed": bool, "expected": any, "actual": any, "error": str}


def record_result(name, passed, expected=None, actual=None, error=None):
    """Record test result for final summary."""
    test_results.append({
        "name": name,
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "error": error
    })


def test_main_case():
    """
    Primary test case from .txt spec.
    """
    test_name = "test_main_case"
    try:
        # --- Make the request (from INPUT section) ---
        response = requests.post(  # or .get(), .put(), .delete()
            f"{BASE_URL}/api/v1/endpoint",
            json={
                "field1": "value1",
                "field2": 123
            },
            headers={
                "Authorization": f"Bearer {AUTH_TOKEN}"  # if needed
            }
        )

        # --- Check status code (from OUTPUT section) ---
        if response.status_code != 200:
            record_result(test_name, False, expected=200, actual=response.status_code,
                         error=f"Status code mismatch: {response.text}")
            return

        data = response.json()

        # --- Check response body (from OUTPUT section) ---
        if data.get("expected_field") != "expected_value":
            record_result(test_name, False, expected="expected_value",
                         actual=data.get("expected_field"), error="Field value mismatch")
            return

        # --- Check business rules (from INTERNAL LOGIC section) ---
        expected_total = data["subtotal"] - data["discount"]
        if data["total"] != expected_total:
            record_result(test_name, False, expected=expected_total,
                         actual=data["total"], error="Total calculation incorrect")
            return

        record_result(test_name, True)
    except Exception as e:
        record_result(test_name, False, error=str(e))


def test_edge_case():
    """
    Edge case from INTERNAL LOGIC section.
    Example: "If quantity is 0 or negative, return 400 error"
    """
    test_name = "test_edge_case"
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/endpoint",
            json={
                "field1": "value1",
                "field2": -1  # Invalid value
            }
        )

        if response.status_code != 400:
            record_result(test_name, False, expected=400, actual=response.status_code,
                         error="Expected 400 for invalid input")
            return

        record_result(test_name, True)
    except Exception as e:
        record_result(test_name, False, error=str(e))


def print_summary():
    """Print final test summary with detailed failure info."""
    passed = [r for r in test_results if r["passed"]]
    failed = [r for r in test_results if not r["passed"]]

    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {len(passed)}/{len(test_results)} passed")
    print(f"{'='*50}")

    if failed:
        print("\nFailed tests:")
        print("-" * 50)
        for f in failed:
            print(f"  test_ref: {f['name']}")
            print(f"  expected: {f['expected']}")
            print(f"  actual:   {f['actual']}")
            if f['error']:
                print(f"  error:    {f['error']}")
            print("-" * 50)

    if passed:
        print("\nPassed tests:")
        for p in passed:
            print(f"  ✓ {p['name']}")

    return len(failed) == 0


if __name__ == "__main__":
    # Run all tests (don't exit early on failure)
    test_main_case()
    test_edge_case()  # Add more as needed

    # Print summary at end
    all_passed = print_summary()

    if all_passed:
        print("\nPASSED")
        sys.exit(0)
    else:
        print("\nFAILED")
        sys.exit(1)
```

---

## JavaScript Test Template

```javascript
#!/usr/bin/env node
/**
 * Test: {filename}
 * {Brief description from the .txt file}
 */

const BASE_URL = process.env.API_BASE_URL || process.env.STAGING_API_URL;
const AUTH_TOKEN = process.env.TEST_AUTH_TOKEN || "";

if (!BASE_URL) {
  console.error("ERROR: API_BASE_URL or STAGING_API_URL environment variable not set");
  process.exit(1);
}

// Collect test results (don't print during execution)
const testResults = []; // Array of {name, passed, expected, actual, error}

function recordResult(name, passed, expected = null, actual = null, error = null) {
  testResults.push({ name, passed, expected, actual, error });
}

async function testMainCase() {
  const testName = "testMainCase";
  try {
    // --- Make the request (from INPUT section) ---
    const response = await fetch(`${BASE_URL}/api/v1/endpoint`, {
      method: "POST", // or "GET", "PUT", "DELETE"
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${AUTH_TOKEN}` // if needed
      },
      body: JSON.stringify({
        field1: "value1",
        field2: 123
      })
    });

    // --- Check status code (from OUTPUT section) ---
    if (response.status !== 200) {
      const text = await response.text();
      recordResult(testName, false, 200, response.status, `Status mismatch: ${text}`);
      return;
    }

    const data = await response.json();

    // --- Check response body (from OUTPUT section) ---
    if (data.expected_field !== "expected_value") {
      recordResult(testName, false, "expected_value", data.expected_field, "Field value mismatch");
      return;
    }

    // --- Check business rules (from INTERNAL LOGIC section) ---
    const expectedTotal = data.subtotal - data.discount;
    if (data.total !== expectedTotal) {
      recordResult(testName, false, expectedTotal, data.total, "Total calculation incorrect");
      return;
    }

    recordResult(testName, true);
  } catch (error) {
    recordResult(testName, false, null, null, error.message);
  }
}

async function testEdgeCase() {
  const testName = "testEdgeCase";
  try {
    const response = await fetch(`${BASE_URL}/api/v1/endpoint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        field1: "value1",
        field2: -1 // Invalid value
      })
    });

    if (response.status !== 400) {
      recordResult(testName, false, 400, response.status, "Expected 400 for invalid input");
      return;
    }

    recordResult(testName, true);
  } catch (error) {
    recordResult(testName, false, null, null, error.message);
  }
}

function printSummary() {
  const passed = testResults.filter(r => r.passed);
  const failed = testResults.filter(r => !r.passed);

  console.log("\n" + "=".repeat(50));
  console.log(`TEST SUMMARY: ${passed.length}/${testResults.length} passed`);
  console.log("=".repeat(50));

  if (failed.length > 0) {
    console.log("\nFailed tests:");
    console.log("-".repeat(50));
    for (const f of failed) {
      console.log(`  test_ref: ${f.name}`);
      console.log(`  expected: ${f.expected}`);
      console.log(`  actual:   ${f.actual}`);
      if (f.error) console.log(`  error:    ${f.error}`);
      console.log("-".repeat(50));
    }
  }

  if (passed.length > 0) {
    console.log("\nPassed tests:");
    for (const p of passed) {
      console.log(`  ✓ ${p.name}`);
    }
  }

  return failed.length === 0;
}

async function main() {
  // Run all tests (don't exit early on failure)
  await testMainCase();
  await testEdgeCase(); // Add more as needed

  // Print summary at end
  const allPassed = printSummary();

  if (allPassed) {
    console.log("\nPASSED");
    process.exit(0);
  } else {
    console.log("\nFAILED");
    process.exit(1);
  }
}

main();
```

---

## Mapping Rules

### HTTP Methods

| .txt says | Use |
|-----------|-----|
| "POST /path" | `requests.post()` / `fetch(..., {method: "POST"})` |
| "GET /path" | `requests.get()` / `fetch(..., {method: "GET"})` |
| "PUT /path" | `requests.put()` / `fetch(..., {method: "PUT"})` |
| "PATCH /path" | `requests.patch()` / `fetch(..., {method: "PATCH"})` |
| "DELETE /path" | `requests.delete()` / `fetch(..., {method: "DELETE"})` |

### Request Body

| .txt says | Code |
|-----------|------|
| `field: "string"` | `"field": "string"` |
| `field: 123` | `"field": 123` |
| `field: 123.45` | `"field": 123.45` |
| `field: true/false` | `"field": true` / `"field": false` |
| `field: ["a", "b"]` | `"field": ["a", "b"]` |
| `field: null` | `"field": null` |

### Query Parameters

For GET requests with query params:

```
# INPUT
- API endpoint: GET /api/v1/search
- Query parameters:
  - q: "search term"
  - limit: 10
```

**Python:**
```python
response = requests.get(
    f"{BASE_URL}/api/v1/search",
    params={"q": "search term", "limit": 10}
)
```

**JavaScript:**
```javascript
const url = new URL("/api/v1/search", BASE_URL);
url.searchParams.set("q", "search term");
url.searchParams.set("limit", "10");
const response = await fetch(url.toString());
```

### Assertions

| .txt says | Python | JavaScript |
|-----------|--------|------------|
| "HTTP status: 200" | `assert response.status_code == 200` | `if (response.status !== 200) throw ...` |
| "field: 123" | `assert data["field"] == 123` | `if (data.field !== 123) throw ...` |
| "field: array" | `assert isinstance(data["field"], list)` | `if (!Array.isArray(data.field)) throw ...` |
| "field exists" | `assert "field" in data` | `if (!("field" in data)) throw ...` |
| "max N items" | `assert len(data["items"]) <= N` | `if (data.items.length > N) throw ...` |
| "sorted by X desc" | Check `data[i].X >= data[i+1].X` | Check `data[i].X >= data[i+1].X` |

---

## Handling Common Patterns

### Decimal/Money Values

When .txt specifies monetary values like `1275.00`:

```python
# Use approximate comparison for floats
assert abs(data["total"] - 1275.00) < 0.01, f"Expected ~1275.00, got {data['total']}"

# Or check string formatting if API returns strings
assert data["total"] == "1275.00"
```

### Array Validation

When .txt says "array of objects with fields X, Y, Z":

```python
assert isinstance(data["items"], list)
for item in data["items"]:
    assert "X" in item, f"Missing field X in {item}"
    assert "Y" in item, f"Missing field Y in {item}"
    assert "Z" in item, f"Missing field Z in {item}"
```

### Sorting Validation

When .txt says "sorted by relevance_score descending":

```python
scores = [item["relevance_score"] for item in data["items"]]
assert scores == sorted(scores, reverse=True), "Items not sorted by relevance_score descending"
```

### Threshold Validation

When .txt says "only return items with score > 0.7":

```python
for item in data["items"]:
    assert item["score"] > 0.7, f"Item below threshold: {item['score']}"
```

### Error Response Validation

When .txt says "return 400 error if X":

```python
def test_invalid_input_returns_400():
    response = requests.post(
        f"{BASE_URL}/api/v1/endpoint",
        json={"field": "invalid_value"}
    )
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
```

---

## What NOT to Test

From the INTERNAL LOGIC section, some things are **documentation only** and cannot be directly tested:

| .txt says | Action |
|-----------|--------|
| "Query table X, not table Y" | Add as code comment for reviewer |
| "Uses algorithm Z internally" | Add as code comment for reviewer |
| "Performance: under 500ms" | Optional: add timing assertion if critical |
| "Caching behavior" | Usually not testable via API |

Example comment:

```python
def test_context_retrieval():
    """
    NOTE: Per spec, this should query user_contexts table, NOT user_history.
    Reviewer: please verify implementation uses correct table.
    """
    # ... test code ...
```

---

## Multiple Test Cases

If the .txt file implies multiple scenarios, create separate test functions:

```python
def test_bulk_discount_100_plus_units():
    """100+ units should get 15% discount"""
    ...

def test_bulk_discount_50_to_99_units():
    """50-99 units should get 10% discount"""
    ...

def test_bulk_discount_under_50_units():
    """Under 50 units should get no discount"""
    ...

def test_zero_quantity_returns_error():
    """Zero or negative quantity should return 400"""
    ...
```

---

---

## Project-Specific Patterns

For API patterns specific to your project (response wrappers, authentication, endpoint structures), create a `project_readme.md` file in your tests directory.

---

## Implementation Rules

### 1. Never Modify Production Code

Tests must NEVER alter code outside the `tests/` folder. All adaptations for bugs or issues must be handled within test code.

If a test is impossible to run:
```python
def test_feature_x():
    """BLOCKED: Requires feature Y which is not implemented"""
    print("SKIP: Cannot run - requires unimplemented feature Y")
    sys.exit(1)  # Fail explicitly with reason
```

### 2. Validate Environment Variables Early

All env vars must be validated at script startup, BEFORE any imports that depend on them:

```python
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Cross-platform temp path (before lambda imports)
import tempfile
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(tempfile.gettempdir(), "gcp_sa.json")

def validate_env():
    missing = []
    if MODE == "local":
        for var in ["AWS_REGION", "TRUSS_SECRETS_ARN", "VERTEXAI_PROJECT"]:
            if not os.getenv(var):
                missing.append(var)
    if missing:
        print(f"ERROR: Missing: {', '.join(missing)}")
        sys.exit(1)

validate_env()  # MUST be before imports that use these vars
```

### 3. Map Output Expectations to Actual Code Behavior

When .txt expects "null" but code returns structured data, match the CODE behavior while preserving the INTENT:

| .txt says | Code actually returns | Test assertion |
|-----------|----------------------|----------------|
| `null` / `NaN` | `{"id": "0", "name": "unknown"}` | Check for id=0 or name="unknown" |
| `material: null` | `{"material": None, "material_id": None}` | `is_nan(result.get("material"))` |
| `error` | `{"success": false, "error": "..."}` | `assert not result.get("success")` |

```python
def is_nan(value):
    """Check if value represents 'no result' per .txt spec"""
    if value is None: return True
    if isinstance(value, float) and math.isnan(value): return True
    if isinstance(value, str) and value.lower() in ("nan", "", "unknown"): return True
    return False
```

### 4. Report Design Decisions

After implementing, document deviations from .txt in a brief report:

```python
"""
DESIGN DECISIONS (deviations from .txt spec):
- .txt expects null for unknown, code returns {"id": 0, "name": "unknown"} -> checking for id=0
- .txt expects single string, code returns list -> checking first element
- Added root_material check not in .txt (required by current API contract)
"""
```

**Print this at test completion if deviations exist:**
```python
if __name__ == "__main__":
    # ... run tests ...
    print("\n--- DESIGN NOTES ---")
    print("- 'null' mapped to id=0 per current API behavior")
    print("- root_material assertion added (not in .txt)")
```

---

## Checklist Before Submitting

1. [ ] Filename matches `.txt` file (e.g., `001-feature.txt` → `001-feature.py`)
2. [ ] Reads `API_BASE_URL` or `STAGING_API_URL` from environment
3. [ ] **Validates all required env vars at startup, exits early if missing**
4. [ ] Exits with code 0 on success, code 1 on failure
5. [ ] Tests the exact endpoint from INPUT section
6. [ ] Asserts status code from OUTPUT section
7. [ ] Asserts all response fields from OUTPUT section (mapped to actual code behavior)
8. [ ] Includes edge cases from INTERNAL LOGIC section
9. [ ] **Collects results during execution, prints summary at end (no mid-test prints)**
10. [ ] **Failed test summary includes: test_ref, expected, actual, error**
11. [ ] Comments reference relevant parts of INTERNAL LOGIC for reviewer
12. [ ] Prints "PASSED" or "FAILED" after summary
13. [ ] **No changes to code outside tests/ folder**
14. [ ] **Design decisions report printed if deviations from .txt exist**

---

## Example Conversion

### Input: `001-bulk-discount.txt`

```
# INPUT
- API endpoint: POST /api/v1/calculate-price
- Request body:
  - quantity: 150
  - unit_price: 10.00

# OUTPUT
- HTTP status: 200
- Response body:
  - subtotal: 1500.00
  - discount_percent: 15
  - total: 1275.00

# INTERNAL LOGIC
- 100+ units gets 15% discount
- Discount applies to subtotal
- If quantity <= 0, return 400
```

### Output: `001-bulk-discount.py`

```python
#!/usr/bin/env python3
"""
Test: 001-bulk-discount
Verifies bulk discount pricing for 100+ unit orders.
"""

import os
import sys
import requests

BASE_URL = os.environ.get("API_BASE_URL") or os.environ.get("STAGING_API_URL")

if not BASE_URL:
    print("ERROR: API_BASE_URL or STAGING_API_URL environment variable not set")
    sys.exit(1)

# Collect test results
test_results = []


def record_result(name, passed, expected=None, actual=None, error=None):
    test_results.append({"name": name, "passed": passed, "expected": expected, "actual": actual, "error": error})


def test_150_units_gets_15_percent_discount():
    """From INPUT/OUTPUT: 150 units at $10 = $1275 after 15% discount"""
    test_name = "test_150_units_gets_15_percent_discount"
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/calculate-price",
            json={"quantity": 150, "unit_price": 10.00}
        )

        if response.status_code != 200:
            record_result(test_name, False, 200, response.status_code, response.text)
            return

        data = response.json()

        if data.get("subtotal") != 1500.00:
            record_result(test_name, False, 1500.00, data.get("subtotal"), "subtotal mismatch")
            return
        if data.get("discount_percent") != 15:
            record_result(test_name, False, 15, data.get("discount_percent"), "discount_percent mismatch")
            return
        if data.get("total") != 1275.00:
            record_result(test_name, False, 1275.00, data.get("total"), "total mismatch")
            return

        # INTERNAL LOGIC: verify discount applies to subtotal
        expected_total = data["subtotal"] * (1 - data["discount_percent"] / 100)
        if abs(data["total"] - expected_total) >= 0.01:
            record_result(test_name, False, expected_total, data["total"], "discount calculation mismatch")
            return

        record_result(test_name, True)
    except Exception as e:
        record_result(test_name, False, error=str(e))


def test_zero_quantity_returns_400():
    """From INTERNAL LOGIC: quantity <= 0 should return 400"""
    test_name = "test_zero_quantity_returns_400"
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/calculate-price",
            json={"quantity": 0, "unit_price": 10.00}
        )

        if response.status_code != 400:
            record_result(test_name, False, 400, response.status_code, "Expected 400 for zero quantity")
            return

        record_result(test_name, True)
    except Exception as e:
        record_result(test_name, False, error=str(e))


def test_negative_quantity_returns_400():
    """From INTERNAL LOGIC: quantity <= 0 should return 400"""
    test_name = "test_negative_quantity_returns_400"
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/calculate-price",
            json={"quantity": -5, "unit_price": 10.00}
        )

        if response.status_code != 400:
            record_result(test_name, False, 400, response.status_code, "Expected 400 for negative quantity")
            return

        record_result(test_name, True)
    except Exception as e:
        record_result(test_name, False, error=str(e))


def print_summary():
    passed = [r for r in test_results if r["passed"]]
    failed = [r for r in test_results if not r["passed"]]

    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {len(passed)}/{len(test_results)} passed")
    print(f"{'='*50}")

    if failed:
        print("\nFailed tests:")
        print("-" * 50)
        for f in failed:
            print(f"  test_ref: {f['name']}")
            print(f"  expected: {f['expected']}")
            print(f"  actual:   {f['actual']}")
            if f['error']:
                print(f"  error:    {f['error']}")
            print("-" * 50)

    if passed:
        print("\nPassed tests:")
        for p in passed:
            print(f"  ✓ {p['name']}")

    return len(failed) == 0


if __name__ == "__main__":
    test_150_units_gets_15_percent_discount()
    test_zero_quantity_returns_400()
    test_negative_quantity_returns_400()

    all_passed = print_summary()

    if all_passed:
        print("\nPASSED")
        sys.exit(0)
    else:
        print("\nFAILED")
        sys.exit(1)
```
