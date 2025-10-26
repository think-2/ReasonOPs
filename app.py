"""Streamlit app entrypoint for ReasonOPs

Minimal UI that loads sample input and runs a placeholder LP solver.
"""
import json
import os
import streamlit as st
from solver import solve_lp
from utils import load_json_file, validate_input

st.set_page_config(page_title="ReasonOPs", layout="centered")

st.sidebar.image("assets/logo.png", width=120)
st.title("ReasonOPs — LP Solver + Qwen Integration (stub)")

st.markdown("Upload a JSON input file describing a small LP, or load the sample input.")

uploaded = st.file_uploader("Upload input JSON", type=["json"])
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
            st.success("Solved (stub)")
            st.json(solution)

st.sidebar.markdown("## Notes\nThis is a scaffolded project. Replace solver and Qwen API stubs with real implementations.")
