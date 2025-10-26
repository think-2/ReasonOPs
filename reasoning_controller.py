"""Controller layer: classification + adaptive schema orchestration.

Pure logic (no Streamlit calls). Manages an in-memory conversation state,
detects problem type, extends/merges schemas via `qwen_api` and `schema_library`,
and suggests clarifying questions for missing fields.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import List, Dict, Optional

from qwen_api import classify_problem, generate_schema
from schema_library import get_base_schema


class ConversationState:
    """Keeps running chat history and evolving schema."""

    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = []
        self.problem_type: Optional[str] = None
        self.schema: Optional[Dict] = None

    def add_user(self, content: str) -> None:
        """Append a user message to the conversation history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Append an assistant message to the conversation history."""
        self.messages.append({"role": "assistant", "content": content})

    def summarize(self) -> str:
        """Return a short summary formed from the last up to 4 turns' content."""
        texts = [m.get("content", "") for m in self.messages[-4:]]
        return " — ".join(texts)


def detect_or_confirm_problem(state: ConversationState) -> str:
    """Classify the problem type if not already set and return it.

    Uses the first user message as the input to the classifier.
    """
    if state.problem_type:
        return state.problem_type

    # Find the first user message
    for m in state.messages:
        if m.get("role") == "user" and m.get("content"):
            user_text = m["content"]
            break
    else:
        raise ValueError("No user message found to classify")

    label = classify_problem(user_text)
    state.problem_type = label.strip()
    return state.problem_type


def merge_schema(old: Dict, new: Dict) -> Dict:
    """Merge two schema dicts field-wise without duplicates.

    Existing fields (by label) are preserved; missing keys are filled from the new
    schema. The returned schema is a deep copy and does not mutate inputs.
    """
    old = deepcopy(old) if old else {}
    new = deepcopy(new) if new else {}

    old_fields = old.get("fields", [])
    new_fields = new.get("fields", [])

    merged: List[Dict] = []
    seen = set()

    def label_key(f: Dict) -> str:
        return (f.get("label") or "").strip().lower()

    # Start with old fields to preserve user-provided defaults/order
    for f in old_fields:
        lk = label_key(f)
        if lk:
            seen.add(lk)
        merged.append(deepcopy(f))

    # Add/merge new fields
    for nf in new_fields:
        lk = label_key(nf)
        if not lk:
            continue
        if lk in seen:
            # merge missing keys from nf into corresponding merged field
            for existing in merged:
                if label_key(existing) == lk:
                    for k, v in nf.items():
                        if k not in existing or existing.get(k) in (None, ""):
                            existing[k] = deepcopy(v)
                    break
        else:
            merged.append(deepcopy(nf))
            seen.add(lk)

    result = deepcopy(new) if new else {}
    result["fields"] = merged
    # keep problem_type if present in old
    if old.get("problem_type"):
        result.setdefault("problem_type", old.get("problem_type"))
    return result


def update_schema(state: ConversationState) -> Dict:
    """Use Qwen to extend or correct the current schema and write it back to state.

    Returns the updated schema.
    """
    base_schema = get_base_schema(state.problem_type) if state.problem_type else {}
    # If state.schema already exists, use it as the base to preserve defaults
    base_for_generation = state.schema or base_schema

    generated = generate_schema(state.messages, base_schema=base_for_generation)
    # If generator returned an error, avoid overwriting existing schema
    if isinstance(generated, dict) and generated.get("error"):
        # Keep existing schema if present, otherwise return empty dict with error
        if state.schema:
            return state.schema
        state.schema = {"error": generated.get("error"), "raw": generated.get("raw")}
        return state.schema

    merged = merge_schema(state.schema or {}, generated or {})
    state.schema = merged
    return state.schema


def find_missing_fields(schema: Dict) -> List[str]:
    """Return list of field labels that have null/empty defaults.

    A field is considered missing when the 'default' key is absent or its value
    is None, an empty string, or an empty list/dict. Numeric zero is treated as present.
    """
    if not schema or "fields" not in schema:
        return []
    missing: List[str] = []
    for f in schema.get("fields", []):
        label = f.get("label") or ""
        if not label:
            continue
        if "default" not in f:
            # treat example as acceptable if present
            if f.get("example"):
                continue
            missing.append(label)
            continue
        val = f.get("default")
        if val is None:
            missing.append(label)
            continue
        if isinstance(val, str) and val.strip() == "":
            missing.append(label)
            continue
        if isinstance(val, (list, dict)) and len(val) == 0:
            missing.append(label)
            continue
    return missing


def next_question(schema: Dict) -> str:
    """Generate a natural clarifying question for the next missing field.

    Uses simple heuristics based on label keywords.
    """
    missing = find_missing_fields(schema)
    if not missing:
        return "No missing fields detected."
    label = missing[0]
    lower = label.lower()
    # heuristics
    if "product" in lower:
        return f"What are the product names (e.g. A, B, C) for '{label}'?"
    if "profit" in lower or "price" in lower or "value" in lower:
        return f"What are the profit (or price) values corresponding to '{label}'?"
    if "resource" in lower or "capacity" in lower or "available" in lower:
        return f"What are the available capacities or amounts for '{label}'?"
    if "time" in lower or "duration" in lower or "deadline" in lower:
        return f"Please provide the time/duration values for '{label}'."
    # fallback
    return f"Please provide values for '{label}'."


if __name__ == "__main__":
    # Smoke test / demo
    state = ConversationState()
    state.add_user("We make 3 products A,B,C with limited labor and material. Maximize profit.")
    try:
        detect_or_confirm_problem(state)
    except Exception as e:
        print("Classification error:", e)
    schema = update_schema(state)
    print("Detected:", state.problem_type)
    print(json.dumps(schema, indent=2))
    missing = find_missing_fields(state.schema or {})
    if missing:
        print("Missing:", missing)
        print("Next question:", next_question(state.schema or {}))
