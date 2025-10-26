"""Minimal schema library for ReasonOPs.

Provides base templates and a registry for quick initialization of adaptive schemas.
"""
from copy import deepcopy
from typing import Dict


profit_optimization_schema: Dict = {
    "problem_type": "profit_optimization",
    "fields": [
        {"label": "Products", "type": "list[str]", "example": ["A", "B", "C"]},
        {"label": "Profit per unit", "type": "list[float]", "example": [10, 15, 25]},
        {"label": "Resources", "type": "list[str]", "example": ["Labor", "Material"]},
        {"label": "Available capacity", "type": "list[float]", "example": [200, 150]},
    ],
}


_REGISTRY: Dict[str, Dict] = {
    "profit_optimization": profit_optimization_schema,
}


def get_base_schema(problem_type: str) -> Dict:
    """Return a deep copy of a base schema template for the given problem_type.

    This returns a copy to avoid accidental mutation of the canonical template.
    """
    template = _REGISTRY.get(problem_type)
    return deepcopy(template) if template is not None else {}
