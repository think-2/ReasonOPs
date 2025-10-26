"""Minimal LP solver stub using PuLP if available.
Provides a solve_lp(data) function that expects a dict with
- 'objective': { 'sense': 'max'|'min', 'coeffs': [..] }
- 'constraints': [ { 'coeffs': [..], 'sense': '<='|'='|'>=', 'rhs': number } ]

This is intentionally small; replace with full modeling later.
"""
from typing import Dict, Any

try:
    import pulp
    _HAS_PULP = True
except Exception:
    _HAS_PULP = False


def solve_lp(data: Dict[str, Any]) -> Dict[str, Any]:
    """Solve a tiny LP or return a deterministic stub if pulp missing.

    Returns a dict with variable values and objective value.
    """
    if not _HAS_PULP:
        # Return a deterministic stub for now
        n = len(data.get("objective", {}).get("coeffs", []))
        vars_values = {f"x{i}": 0.0 for i in range(n)}
        return {"status": "stub_no_pulp", "vars": vars_values, "objective": 0.0}

    # Minimal model creation
    sense = data.get("objective", {}).get("sense", "max")
    coeffs = data.get("objective", {}).get("coeffs", [])
    n = len(coeffs)

    if sense.lower().startswith("max"):
        prob = pulp.LpProblem("ReasonOPs", pulp.LpMaximize)
    else:
        prob = pulp.LpProblem("ReasonOPs", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(n)]
    prob += pulp.lpSum([coeffs[i] * x[i] for i in range(n)])

    for c in data.get("constraints", []):
        coeffs_c = c.get("coeffs", [])
        sense_c = c.get("sense", "<=")
        rhs = c.get("rhs", 0)
        expr = pulp.lpSum([coeffs_c[i] * x[i] for i in range(n)])
        if sense_c == "<=":
            prob += expr <= rhs
        elif sense_c == ">=":
            prob += expr >= rhs
        else:
            prob += expr == rhs

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    result = {"status": pulp.LpStatus[prob.status], "vars": {}, "objective": pulp.value(prob.objective)}
    for i, var in enumerate(x):
        result["vars"][var.name] = var.value()
    return result
