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

### Unknown/Unidentified Classification Handling

When classifier cannot identify a property (e.g., no material indicators in text), it returns:
- `{property}_id`: `0` (ID 0 = "unknown" convention in schema)
- `{property}`: `"Unknown"` (name lookup for ID 0)
- `root_{property}_id`: `None` (root lookup skipped for ID 0)
- `root_{property}`: `None`

Use helper to check for unknown state:

```python
import math

def is_unknown(value):
    """
    Check if value represents 'unknown' classification result.

    Classifier returns ID 0 + name "Unknown" when property can't be identified.
    """
    if value is None: return True
    if value == 0: return True
    if isinstance(value, str) and value.lower() in ("unknown", "nan", ""): return True
    if isinstance(value, float) and math.isnan(value): return True
    return False
```

### Legacy NaN/Null Handling (deprecated)

Older code may use various null representations:

```python
def is_nan_equivalent(value):
    """Check if value represents NaN/null/empty (legacy)."""
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
    ({"material": ""}, None, None, None, None),  # Unknown expected: ID=0, name="Unknown"
]

def run_test_case(case_num, text_dump, exp_mat, exp_id, exp_root, exp_root_id):
    response = requests.post(f"{BASE_URL}{ENDPOINT}", json={"text_dump": text_dump, "input_mode": "text-only"})
    result = response.json()["data"][0]
    if exp_mat is None:
        # Unknown case: check for ID=0 or name="Unknown"
        assert is_unknown(result.get("material_id"))
        assert is_unknown(result.get("material"))
    else:
        assert result.get("material") == exp_mat
        assert result.get("material_id") == exp_id

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
| `material` | Classified material name (`"Unknown"` if unidentifiable) |
| `material_id` | Material taxonomy ID (`0` if unidentifiable) |
| `root_material` | Root/parent material name (`None` if unidentifiable) |
| `root_material_id` | Root material taxonomy ID (`None` if unidentifiable) |
| `confidence` | Classification confidence (0-1) |
| `success`, `validation_passed` | Error detection fields |

### Unknown State Convention

When classifier cannot identify a property:

| Field | Value |
|-------|-------|
| `{property}_id` | `0` |
| `{property}` | `"Unknown"` |
| `root_{property}_id` | `None` |
| `root_{property}` | `None` |

This is distinct from API errors (`success: false`). Unknown means classification succeeded but no match found.

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

### API Mode
| Variable | Description |
|----------|-------------|
| `STAGING_API_URL` | Base URL for staging API |
| `DSL_API_KEY` | API key for `x-api-key` header |

### Local Mode (Direct Lambda Invocation)
| Variable | Description |
|----------|-------------|
| `AWS_REGION` | AWS region for Secrets Manager |
| `TRUSS_SECRETS_ARN` | ARN containing GCP service account JSON |
| `VERTEXAI_PROJECT` | GCP project ID for Gemini models |
| `VERTEXAI_LOCATION` | GCP region (e.g., `us-central1`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP SA JSON (auto-set if missing) |

---

## BigQuery/GCP Connection (Local Mode)

Local tests need GCP credentials for BigQuery taxonomy lookups and Vertex AI (Gemini) models.

### How It Works

1. **Credentials stored in AWS Secrets Manager** under `TRUSS_SECRETS_ARN`
2. **Test sets cross-platform temp path** before lambda imports:
   ```python
   import tempfile
   if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
       os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(tempfile.gettempdir(), "gcp_sa.json")
   ```
3. **`ensure_gcp_adc()` fetches and writes credentials**:
   ```python
   from core.utils.credentials import ensure_gcp_adc
   ensure_gcp_adc()  # Fetches from Secrets Manager, writes to temp path
   ```

### Credential Flow

```
TRUSS_SECRETS_ARN (AWS)
        │
        ▼
ensure_gcp_adc() fetches GCP SA JSON
        │
        ▼
Writes to GOOGLE_APPLICATION_CREDENTIALS path
        │
        ▼
BigQuery client uses ADC (Application Default Credentials)
```

### Platform Paths

| Platform | GOOGLE_APPLICATION_CREDENTIALS |
|----------|-------------------------------|
| Linux/Lambda | `/tmp/gcp_sa.json` |
| Windows | `C:\Users\<user>\AppData\Local\Temp\gcp_sa.json` |
| macOS | `/var/folders/.../gcp_sa.json` |

### Required AWS Permissions

Test runner needs IAM permissions to read from Secrets Manager:
- `secretsmanager:GetSecretValue` on `TRUSS_SECRETS_ARN`

---

## Running Tests Locally

### Setup
```bash
# Copy and fill in environment variables
cp tests/.env.example tests/.env
# Edit tests/.env with your values
```

### API Mode (calls staging endpoint)
```bash
# Requires: STAGING_API_URL, DSL_API_KEY
conda run -n pyds python tests/material_test.py -m api
```

### Local Mode (direct lambda invocation)
```bash
# Requires: AWS_REGION, TRUSS_SECRETS_ARN, VERTEXAI_PROJECT, VERTEXAI_LOCATION
# Also needs AWS credentials configured (aws configure)
conda run -n pyds python tests/material_test.py -m local
conda run -n pyds python tests/material_text_test.py -m local
```

### Run All Tests
```bash
STAGING_API_URL=https://staging.example.com python .github/scripts/runner.py
```
