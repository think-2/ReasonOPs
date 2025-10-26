"""Minimal LP solver stub using PuLP if available.
Provides a solve_lp(data) function that expects a dict with
- 'objective': { 'sense': 'max'|'min', 'coeffs': [..] }
- 'constraints': [ { 'coeffs': [..], 'sense': '<='|'='|'>=', 'rhs': number } ]

This is intentionally small; replace with full modeling later.
"""
from typing import Dict, Any
import re

try:
    import pulp
    _HAS_PULP = True
except Exception:
    _HAS_PULP = False


def parse_list(value: str) -> list[Any]:
    """Parse comma-separated user input into list of strings or floats.

    Returns a list of floats if all items parse as float, otherwise strings.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        return [value]
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        return []
    # try to convert to float where possible
    parsed: list[Any] = []
    all_float = True
    for it in items:
        try:
            num = float(it)
            parsed.append(num)
        except ValueError:
            parsed.append(it)
            all_float = False
    # if mixed types, keep parsed as-is
    return parsed


def solve_profit_lp(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build and solve a simple linear profit-maximization LP.

    Expects keys: Products, Profit per unit, Resources, Available capacity.
    """
    products = parse_list(form_data.get("Products", ""))
    profits = parse_list(form_data.get("Profit per unit", ""))
    resources = parse_list(form_data.get("Resources", ""))
    caps = parse_list(form_data.get("Available capacity", ""))

    # Normalize products to strings
    products = [str(p) for p in products]
    # profits/caps may be floats or strings; convert where possible
    def to_float_list(lst):
        out = []
        for v in lst:
            try:
                out.append(float(v))
            except Exception:
                out.append(0.0)
        return out

    profits_f = to_float_list(profits)
    caps_f = to_float_list(caps)

    if not products:
        return {"status": "NoProducts", "objective": None, "allocations": {}}

    if not _HAS_PULP:
        # Fallback deterministic stub
        allocations = {p: 0.0 for p in products}
        return {"status": "NoSolver", "objective": 0.0, "allocations": allocations}

    m = pulp.LpProblem("ProfitMax", pulp.LpMaximize)
    x = {p: pulp.LpVariable(re.sub(r"\s+", "_", p), lowBound=0) for p in products}

    # objective
    if profits_f and len(profits_f) >= len(products):
        m += pulp.lpSum(profits_f[i] * x[p] for i, p in enumerate(products))
    elif profits_f and len(profits_f) == len(products):
        m += pulp.lpSum(profits_f[i] * x[p] for i, p in enumerate(products))
    else:
        # if no profits, set zero objective
        m += pulp.lpSum(0.0 * x[p] for p in products)

    # simple resource constraint: sum of all products <= cap for each resource
    for i, r in enumerate(resources):
        cap = caps_f[i] if i < len(caps_f) else 0.0
        m += (pulp.lpSum(1.0 * x[p] for p in products) <= cap), f"Resource_{r}"

    m.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[m.status] if hasattr(pulp, "LpStatus") else str(m.status)
    objective = None
    try:
        if m.status == 1:
            objective = round(pulp.value(m.objective), 2)
    except Exception:
        objective = None

    allocations = {}
    for p in products:
        varname = re.sub(r"\s+", "_", p)
        val = x[p].value()
        allocations[p] = round(val, 2) if val is not None else 0.0

    return {"status": status, "objective": objective, "allocations": allocations}
