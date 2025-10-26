"""Utility helpers: JSON loading, simple input validation, and small helpers for charts/reports.
"""
import json
from typing import Tuple, Dict, Any


def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_input(data: Dict[str, Any]) -> Tuple[bool, str]:
    # Very small validation: ensure objective and constraints exist
    if "objective" not in data:
        return False, "Missing 'objective'"
    obj = data["objective"]
    if "coeffs" not in obj:
        return False, "Objective missing 'coeffs'"
    if not isinstance(obj.get("coeffs"), list):
        return False, "Objective 'coeffs' must be a list"
    # constraints is optional but if present must be a list
    if "constraints" in data and not isinstance(data["constraints"], list):
        return False, "'constraints' must be a list"
    return True, "ok"
