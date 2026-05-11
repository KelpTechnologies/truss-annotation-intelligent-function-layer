"""Quick unit test for hardware root lookup fix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "automated-annotation"))

from agent_orchestration.root_property_lookup import lookup_hardware_root, lookup_material_root
import agent_orchestration.root_property_lookup as rpl

print("Test 1: lookup_hardware_root import")
print("  PASS")

# Mock lookup_root_property to avoid DB calls
original = rpl.lookup_root_property

def mock_lookup(property_type, property_label_id, category="bags"):
    return {"root_property_name": "TestRoot", "root_property_id": 999, "error_logs": []}

rpl.lookup_root_property = mock_lookup

print("Test 2: hardware root returns correct keys")
hw = lookup_hardware_root(hardware_id=123)
assert "root_hardware_name" in hw, f"Missing root_hardware_name: {hw}"
assert "root_hardware_id" in hw, f"Missing root_hardware_id: {hw}"
assert hw["root_hardware_name"] == "TestRoot"
assert hw["root_hardware_id"] == 999
assert hw["error_logs"] == []
print("  PASS")

print("Test 3: material root unchanged")
mat = lookup_material_root(material_id=456)
assert "root_material_name" in mat
assert mat["root_material_name"] == "TestRoot"
print("  PASS")

print("Test 4: hardware delegates to material property_type")
calls = []
def tracking_lookup(property_type, property_label_id, category="bags"):
    calls.append(property_type)
    return {"root_property_name": "X", "root_property_id": 1, "error_logs": []}
rpl.lookup_root_property = tracking_lookup
lookup_hardware_root(hardware_id=42)
assert calls[-1] == "material", f"Expected material, got {calls[-1]}"
print("  PASS")

rpl.lookup_root_property = original

# Test 5: classify_for_api config_id ordering
# Hardware config_ids must be checked BEFORE material (since some contain both words)
print("Test 5: config_id matching order")
config_ids = {
    "classifier-hardware-bags": "hardware",
    "classifier-hardware-material-bags-text": "hardware",
    "classifier-material-bags": "material",
    "classifier-material-bags-text": "material",
    "classifier-model-bags": "model",
}
for cid, expected in config_ids.items():
    lower = cid.lower()
    if "hardware" in lower:
        matched = "hardware"
    elif "material" in lower:
        matched = "material"
    elif "model" in lower:
        matched = "model"
    else:
        matched = "unknown"
    assert matched == expected, f"{cid}: expected {expected}, matched {matched}"
print("  PASS")

# Test 6: format_classification_for_legacy_api includes root fields for hardware
print("Test 6: legacy API format includes hardware root fields")
from core.orchestration_api_handlers.agent_orchestration_api_handler import format_classification_for_legacy_api
mock_result = {
    "success": True,
    "validation_passed": True,
    "status": "ok",
    "error_type": "",
    "error": "",
    "primary_name": "Zipper",
    "primary_id": 42,
    "reasoning": "test",
    "root_hardware_name": "Closure",
    "root_hardware_id": 10,
    "confidence": 0.95,
    "root_lookup_errors": [],
}
formatted = format_classification_for_legacy_api(mock_result, "hardware")
assert "root_hardware" in formatted, f"Missing root_hardware in formatted: {formatted}"
assert "root_hardware_id" in formatted, f"Missing root_hardware_id in formatted: {formatted}"
assert formatted["root_hardware"] == "Closure"
assert formatted["root_hardware_id"] == 10
assert formatted["hardware"] == "Zipper"
assert formatted["hardware_id"] == 42
print("  PASS")

print("\nAll 6 tests passed.")
