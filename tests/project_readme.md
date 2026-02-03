# Project-Specific Test Patterns

This document contains patterns specific to the **Truss Annotation Intelligent Function Layer** API.

For general test implementation guidance, see:
- [TXT_TO_CODE_GUIDE.md](./TXT_TO_CODE_GUIDE.md) - Converting .txt specs to test code
- [README.md](./README.md) - Framework overview and workflow

---

## Classification API Patterns

### Response Wrapper Format

All classification endpoints return a wrapped response:

```json
{
  "component_type": "classification_result",
  "data": [{ /* actual result */ }],
  "metadata": {"category": "bags", "target": "material"}
}
```

Extract the result via `response["data"][0]`.

### Input Format (text_dump)

Classification endpoints accept `text_dump` containing product fields:

```python
{
    "text_dump": {
        "material": "Calfskin",
        "Title": "Louis Vuitton Neverfull",
        "description": "..."  # optional
    },
    "input_mode": "text-only"
}
```

### NaN/Null Handling

API may return various null representations. Use a helper:

```python
import math

def is_nan_equivalent(value):
    """Check if value represents NaN/null/empty."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ("nan", ""):
        return True
    return False
```

### Data-Driven Test Pattern

For multiple test cases, use a list of tuples:

```python
TEST_CASES = [
    # (text_dump, expected_field1, expected_field2, ...)
    ({"material": "Canvas"}, "Canvas", 2, "Canvas", 2),
    ({"material": "Calfskin"}, "Calfskin", 47, "Leather", 1),
    ({"material": ""}, None, None, None, None),  # NaN expected
]

def run_test_case(case_num, text_dump, exp1, exp2, exp3, exp4):
    response = requests.post(f"{BASE_URL}{ENDPOINT}", json={"text_dump": text_dump, "input_mode": "text-only"})
    result = response.json()["data"][0]
    # assertions...

for i, args in enumerate(TEST_CASES, 1):
    run_test_case(i, *args)
```

### Classification Endpoint Reference

| Endpoint Pattern | Description |
|------------------|-------------|
| `POST /automations/annotation/{category}/classify/{property}` | Generic classification |
| Examples: `/bags/classify/material`, `/bags/classify/colour`, `/watches/classify/model` | |

### Common Response Fields (material example)

| Field | Description |
|-------|-------------|
| `material` | Classified material name |
| `material_id` | Material taxonomy ID |
| `root_material` | Root/parent material name |
| `root_material_id` | Root material taxonomy ID |
| `confidence` | Classification confidence (0-1) |
| `success`, `validation_passed` | Error detection fields |

---

## Complete Classification Test Example

### Input: `material_test.txt`

```
inputs:
text_dump_arr:
[{"material": "Canvas", "Title": "..."},
 {"material": "Calfskin", "Title": "..."},
 {"material": "", "Title": ""},
 {"material": "", "Title": "Louis vuitton bag"}]

expected output:
material,material_id,root_material,root_material_id
Canvas,2,Canvas,2
Calfskin,47,Leather,1
NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN
```

### Output: `material_test.py`

See `tests/material_test.py` for the complete implementation using:
- Data-driven `TEST_CASES` array
- `is_nan_equivalent()` helper for null checking
- Response wrapper extraction `data["data"][0]`
- x-api-key header for authentication

---

## Environment Variables (Project-Specific)

| Variable | Description |
|----------|-------------|
| `STAGING_API_URL` | Base URL for staging API (e.g., `https://staging-api.truss.com`) |
| `API_BASE_URL` | Alternative to STAGING_API_URL |
| `X_API_KEY` | API key for `x-api-key` header authentication |

---

## Running Tests Locally

```bash
# Set environment variables
export STAGING_API_URL=https://staging-api.example.com
export X_API_KEY=your-api-key

# Run all tests
python .github/scripts/runner.py

# Run a specific test
python tests/material_test.py
```
