#!/usr/bin/env python3
"""Quick API test - single call to condition + material endpoints, full output comparison."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BASE_URL = (os.getenv("STAGING_API_URL") or "").rstrip("/")
API_KEY = os.getenv("DSL_API_KEY") or ""

if not BASE_URL:
    print("ERROR: Set STAGING_API_URL in tests/.env")
    exit(1)

import requests

TEXT_DUMP = {
    "material": "calfskin",
    "Title": "Louis Vuitton Neverfull MM Monogram vintage bag",
    "brand": "Louis Vuitton"
}

def call_endpoint(prop: str) -> dict:
    url = f"{BASE_URL}/automations/annotation/bags/classify/{prop}"
    resp = requests.post(
        url,
        json={"text_dump": TEXT_DUMP, "input_mode": "text-only"},
        headers={"x-api-key": API_KEY} if API_KEY else {}
    )
    return {"status": resp.status_code, "body": resp.json() if resp.ok else resp.text}

print(f"=== QUICK API TEST ===")
print(f"URL: {BASE_URL}")
print(f"Input: {json.dumps(TEXT_DUMP, indent=2)}\n")

for prop in ["material", "condition", "colour", "hardware"]:
    print(f"--- {prop.upper()} ---")
    result = call_endpoint(prop)
    print(json.dumps(result, indent=2))
    print()
