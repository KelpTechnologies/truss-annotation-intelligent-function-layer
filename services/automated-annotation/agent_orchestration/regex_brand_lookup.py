"""
Regex Brand Lookup
==================
Deterministic CSV-based brand matching. Replaces Agent 1 LLM extraction + DB lookup.

Loads a brand_master CSV (cols: brand_name, display_brand_name, brand_id) and does
case-insensitive substring matching against input text. Results are deduplicated on
display_brand_name and ordered by match start position.

NOTE: brand_id values from the CSV may be out of sync with the production database.
Currently brand_id is not used downstream after output, so this should be safe.
"""

import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


# Default CSV path — override via csv_path param or set before import
DEFAULT_CSV_PATH = Path(__file__).parent.parent / "agent_utils" / "brand_master_export.csv"


class BrandMasterIndex:
    """Preloaded brand master for fast regex lookup."""

    def __init__(self, csv_path: Optional[str] = None):
        path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
        if not path.exists():
            raise FileNotFoundError(f"Brand master CSV not found: {path}")

        df = pd.read_csv(path)
        required = {"brand_name", "display_brand_name", "brand_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Brand master CSV missing columns: {missing}")

        # Drop rows with null brand_name
        df = df.dropna(subset=["brand_name"])

        # Build lookup: list of (pattern_lower, brand_name, display_brand_name, brand_id)
        self.entries = []
        for _, row in df.iterrows():
            self.entries.append((
                str(row["brand_name"]).lower(),
                str(row["brand_name"]),
                str(row["display_brand_name"]),
                int(row["brand_id"]) if pd.notna(row["brand_id"]) else None,
            ))

        self.total_patterns = len(self.entries)

    def search(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Case-insensitive substring match of all brand_name patterns against text.
        Deduplicates on display_brand_name, orders by earliest match start position.

        Returns dict matching Agent 1 output shape:
            success, matched_brands [{id, brand}], search_terms_used, etc.
        """
        t0 = time.perf_counter()
        text_lower = text.lower()

        # Collect matches: display_brand_name -> (earliest_start_pos, brand_id)
        seen: Dict[str, tuple] = {}  # display_brand_name -> (start_pos, brand_id)

        for pattern_lower, brand_name, display_brand_name, brand_id in self.entries:
            if not pattern_lower:
                continue
            pos = text_lower.find(pattern_lower)
            if pos != -1:
                if display_brand_name not in seen or pos < seen[display_brand_name][0]:
                    seen[display_brand_name] = (pos, brand_id)

        # Sort by start position
        sorted_matches = sorted(seen.items(), key=lambda x: x[1][0])

        # Build output in same shape as Agent 1 DB results
        matched_brands = []
        for display_name, (pos, brand_id) in sorted_matches:
            matched_brands.append({
                "id": brand_id,       # NOTE: may be out of sync with production DB
                "brand": display_name,
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if verbose:
            print(f"  Regex lookup: {len(matched_brands)} unique brands matched "
                  f"from {self.total_patterns} patterns in {elapsed_ms:.1f}ms")

        return {
            "success": len(matched_brands) > 0,
            "extracted_candidates": [],
            "search_terms_used": [],
            "matched_brands": matched_brands,
            "regex_match_count": len(matched_brands),
            "regex_elapsed_ms": elapsed_ms,
            "error": None if matched_brands else "No regex matches found",
        }
