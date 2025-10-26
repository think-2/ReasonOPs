"""Minimal LP solvers using PuLP if available.

Provides two entrypoints:
- solve_lp(data): generic coefficient-matrix based interface.
- solve_profit_lp(products, profits, resources, availability, consumption):
  higher-level profit-maximization helper suitable for MVP UI.
"""
from typing import Dict, Any, List, Mapping, Optional

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


def solve_profit_lp(
    products: List[str],
    profits: Mapping[str, float],
    resources: List[str],
    availability: Mapping[str, float],
    consumption: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, Any]:
    """Maximize profit sum(profit[p] * x[p]) subject to resource constraints.

    If `consumption` is provided, it should be a dict resource->product->coef
    indicating units of resource used per one unit of product. If omitted, we
    default to 1 unit per product for each resource (simple capacity cap).
    """
    if not _HAS_PULP:
        alloc = {p: 0.0 for p in products}
        return {"status": "stub_no_pulp", "allocations": alloc, "objective": 0.0}

    prob = pulp.LpProblem("ProfitMax", pulp.LpMaximize)
    x = {p: pulp.LpVariable(p, lowBound=0) for p in products}

    # Objective
    prob += pulp.lpSum((profits.get(p, 0.0) * x[p]) for p in products)

    # Constraints
    if consumption:
        for r in resources:
            cons_r = consumption.get(r, {})
            prob += pulp.lpSum((cons_r.get(p, 0.0) * x[p]) for p in products) <= availability.get(r, 0.0)
    else:
        # Default: each product consumes 1 unit of each resource
        for r in resources:
            prob += pulp.lpSum(x[p] for p in products) <= availability.get(r, 0.0)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    allocations = {p: x[p].value() for p in products}
    objective = float(pulp.value(prob.objective)) if prob.objective is not None else 0.0
    return {"status": status, "allocations": allocations, "objective": round(objective, 4)}
