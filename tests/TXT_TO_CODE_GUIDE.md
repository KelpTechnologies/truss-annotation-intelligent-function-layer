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


def test_main_case():
    """
    Primary test case from .txt spec.
    """
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
    assert response.status_code == 200, \
        f"Expected status 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    
    # --- Check response body (from OUTPUT section) ---
    assert data["expected_field"] == "expected_value", \
        f"Expected 'expected_value', got {data.get('expected_field')}"
    
    # --- Check business rules (from INTERNAL LOGIC section) ---
    # Example: verify calculation
    assert data["total"] == data["subtotal"] - data["discount"], \
        f"Total calculation incorrect: {data}"
    
    print("✓ Main test case passed")


def test_edge_case():
    """
    Edge case from INTERNAL LOGIC section.
    Example: "If quantity is 0 or negative, return 400 error"
    """
    response = requests.post(
        f"{BASE_URL}/api/v1/endpoint",
        json={
            "field1": "value1",
            "field2": -1  # Invalid value
        }
    )
    
    assert response.status_code == 400, \
        f"Expected status 400 for invalid input, got {response.status_code}"
    
    print("✓ Edge case passed")


if __name__ == "__main__":
    try:
        test_main_case()
        test_edge_case()  # Add more as needed
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

async function testMainCase() {
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
    throw new Error(`Expected status 200, got ${response.status}: ${text}`);
  }

  const data = await response.json();

  // --- Check response body (from OUTPUT section) ---
  if (data.expected_field !== "expected_value") {
    throw new Error(`Expected 'expected_value', got ${data.expected_field}`);
  }

  // --- Check business rules (from INTERNAL LOGIC section) ---
  // Example: verify calculation
  if (data.total !== data.subtotal - data.discount) {
    throw new Error(`Total calculation incorrect: ${JSON.stringify(data)}`);
  }

  console.log("✓ Main test case passed");
}

async function testEdgeCase() {
  // Edge case from INTERNAL LOGIC section
  // Example: "If quantity is 0 or negative, return 400 error"
  
  const response = await fetch(`${BASE_URL}/api/v1/endpoint`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      field1: "value1",
      field2: -1 // Invalid value
    })
  });

  if (response.status !== 400) {
    throw new Error(`Expected status 400 for invalid input, got ${response.status}`);
  }

  console.log("✓ Edge case passed");
}

async function main() {
  try {
    await testMainCase();
    await testEdgeCase(); // Add more as needed
    console.log("PASSED");
    process.exit(0);
  } catch (error) {
    console.error(`FAILED: ${error.message}`);
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

## Checklist Before Submitting

1. [ ] Filename matches `.txt` file (e.g., `001-feature.txt` → `001-feature.py`)
2. [ ] Reads `API_BASE_URL` or `STAGING_API_URL` from environment
3. [ ] Exits with code 0 on success, code 1 on failure
4. [ ] Tests the exact endpoint from INPUT section
5. [ ] Asserts status code from OUTPUT section
6. [ ] Asserts all response fields from OUTPUT section
7. [ ] Includes edge cases from INTERNAL LOGIC section
8. [ ] Error messages are clear and include actual vs expected values
9. [ ] Comments reference relevant parts of INTERNAL LOGIC for reviewer
10. [ ] Prints "PASSED" on success, "FAILED: {reason}" on failure

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


def test_150_units_gets_15_percent_discount():
    """From INPUT/OUTPUT: 150 units at $10 = $1275 after 15% discount"""
    response = requests.post(
        f"{BASE_URL}/api/v1/calculate-price",
        json={"quantity": 150, "unit_price": 10.00}
    )
    
    assert response.status_code == 200, \
        f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    
    assert data["subtotal"] == 1500.00, \
        f"Expected subtotal 1500.00, got {data.get('subtotal')}"
    assert data["discount_percent"] == 15, \
        f"Expected discount_percent 15, got {data.get('discount_percent')}"
    assert data["total"] == 1275.00, \
        f"Expected total 1275.00, got {data.get('total')}"
    
    # INTERNAL LOGIC: verify discount applies to subtotal
    expected_total = data["subtotal"] * (1 - data["discount_percent"] / 100)
    assert abs(data["total"] - expected_total) < 0.01, \
        f"Total doesn't match subtotal minus discount"
    
    print("✓ 150 units discount test passed")


def test_zero_quantity_returns_400():
    """From INTERNAL LOGIC: quantity <= 0 should return 400"""
    response = requests.post(
        f"{BASE_URL}/api/v1/calculate-price",
        json={"quantity": 0, "unit_price": 10.00}
    )
    
    assert response.status_code == 400, \
        f"Expected 400 for zero quantity, got {response.status_code}"
    
    print("✓ Zero quantity error test passed")


def test_negative_quantity_returns_400():
    """From INTERNAL LOGIC: quantity <= 0 should return 400"""
    response = requests.post(
        f"{BASE_URL}/api/v1/calculate-price",
        json={"quantity": -5, "unit_price": 10.00}
    )
    
    assert response.status_code == 400, \
        f"Expected 400 for negative quantity, got {response.status_code}"
    
    print("✓ Negative quantity error test passed")


if __name__ == "__main__":
    try:
        test_150_units_gets_15_percent_discount()
        test_zero_quantity_returns_400()
        test_negative_quantity_returns_400()
        print("PASSED")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
```
