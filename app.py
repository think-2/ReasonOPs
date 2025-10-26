"""Streamlit app entrypoint for ReasonOPs

Two flows:
- JSON-based LP solve (load sample or upload file) via solve_lp
- Qwen-assisted flow: analyze text → schema → dynamic inputs → solve_profit_lp
"""
import json
import streamlit as st

from solver import solve_lp, solve_profit_lp
from utils import load_json_file, validate_input, plot_allocation_chart
from qwen_api import qwen_reason

st.set_page_config(page_title="ReasonOPs", layout="centered")

# Sidebar logo (tolerate missing/invalid image files)
try:
    st.sidebar.image("assets/logo.png", width=120)
except Exception:
    st.sidebar.markdown("### ReasonOPs")
st.title("ReasonOPs — Production Profit Optimizer (Qwen-Powered)")

tab1, tab2 = st.tabs(["JSON LP", "Qwen-Assisted"])

with tab1:
    st.markdown("Upload a JSON input file describing a small LP, or load the sample input.")
    uploaded = st.file_uploader("Upload input JSON", type=["json"], key="upload_json")
    if uploaded is not None:
        data = json.load(uploaded)
    else:
        if st.button("Load sample input"):
            data = load_json_file("assets/sample_inputs.json")
        else:
            data = None

    if data is not None:
        valid, msg = validate_input(data)
        if not valid:
            st.error(f"Invalid input: {msg}")
        else:
            st.write("Input:")
            st.json(data)
            if st.button("Solve LP"):
                with st.spinner("Solving..."):
                    solution = solve_lp(data)
                st.success("Solved")
                st.json(solution)

with tab2:
    st.subheader("Describe your production setup")
    desc = st.text_area(
        "Natural language description",
        placeholder=(
            "e.g., Two products A/B, profit per unit differs, with labor and material "
            "availability. Infer fields and suggest a schema."
        ),
        key="qwen_desc",
    )

    if "q_schema" not in st.session_state:
        st.session_state.q_schema = None
    if "q_products" not in st.session_state:
        st.session_state.q_products = []
    if "q_resources" not in st.session_state:
        st.session_state.q_resources = []

    if st.button("Analyze with Qwen"):
        with st.spinner("Contacting Qwen for schema…"):
            text = qwen_reason(f"Extract structured LP fields (products, resources) from: {desc}")
        if text.startswith("Error:"):
            st.error(text)
        else:
            # Heuristic: attempt to parse JSON embedded in text; if it fails, fallback
            parsed = None
            try:
                # Try to find JSON block or parse whole text
                candidate = text
                start = candidate.find("{")
                end = candidate.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = candidate[start : end + 1]
                parsed = json.loads(candidate)
            except Exception:
                parsed = None

            products = (parsed or {}).get("products") or ["Product A", "Product B"]
            resources = (parsed or {}).get("resources") or ["Labor", "Material"]
            st.session_state.q_schema = parsed or {}
            st.session_state.q_products = products
            st.session_state.q_resources = resources
            st.success("Schema inferred. Adjust values below and optimize.")

    if st.session_state.q_products:
        st.markdown("### Define profits and availability")
        cols = st.columns(2)
        profits = {}
        with cols[0]:
            st.markdown("#### Profit per product")
            for p in st.session_state.q_products:
                profits[p] = st.number_input(f"Profit per unit — {p}", min_value=0.0, value=10.0, step=1.0, key=f"prof_{p}")
        availability = {}
        with cols[1]:
            st.markdown("#### Resource availability")
            for r in st.session_state.q_resources:
                availability[r] = st.number_input(f"Available — {r}", min_value=0.0, value=100.0, step=1.0, key=f"avail_{r}")

        if st.button("Optimize Profit"):
            with st.spinner("Solving optimization…"):
                result = solve_profit_lp(
                    st.session_state.q_products,
                    profits,
                    st.session_state.q_resources,
                    availability,
                )
            if result.get("status") not in {"Optimal", "optimal"}:
                st.warning(f"Solver status: {result.get('status')}")
            st.success(f"Max Profit: {result.get('objective')}")
            st.json(result.get("allocations"))

            fig = plot_allocation_chart(result.get("allocations", {}))
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            with st.spinner("Explaining with Qwen…"):
                explanation = qwen_reason(
                    "Explain why this allocation is optimal (brief): " + json.dumps(result)
                )
            if explanation.startswith("Error:"):
                st.info("Explanation unavailable: " + explanation)
            else:
                st.markdown("### Reasoning")
                st.write(explanation)

st.sidebar.markdown(
    "## Notes\n"
    "- JSON LP tab supports manual coefficient models.\n"
    "- Qwen tab infers schema from description and optimizes a simple profit model.\n"
    "- Set QWEN_API_KEY in env or .streamlit/secrets.toml."
)
