"""Utility helpers: JSON loading, validation, and charting.
"""
import json
from typing import Tuple, Dict, Any

try:
    import pandas as pd  # type: ignore
    import plotly.express as px  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


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


def plot_allocation_chart(allocations: Dict[str, float]):
    """Return a Plotly bar chart for allocations if Plotly is available.

    If Plotly is not installed, return None and let the caller handle fallback.
    """
    if not _HAS_PLOTLY:
        return None
    df = pd.DataFrame([{"Product": k, "Units": v} for k, v in allocations.items()])
    fig = px.bar(df, x="Product", y="Units", title="Optimal Production Mix")
    fig.update_layout(yaxis_title="Units", xaxis_title="Product")
    return fig
