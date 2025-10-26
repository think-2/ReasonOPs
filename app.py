"""Streamlit chat UI that ties the reasoning controller and solver together.

This app persists a ConversationState in `st.session_state.state`, lets a user
chat naturally, builds an adaptive input form from the current schema, asks
clarifying questions, and calls the solver when requested.
"""
import json
import time

import streamlit as st
import pandas as pd
import plotly.express as px

from reasoning_controller import (
    ConversationState,
    detect_or_confirm_problem,
    update_schema,
    find_missing_fields,
    next_question,
)
from qwen_api import qwen_chat
from solver import solve_profit_lp


st.set_page_config(page_title="Production Profit Optimizer", page_icon="📈", layout="centered")
st.title("🤖 Reasoning-Driven Production Optimizer")
st.info("⏳ The app may take 20-30 s to start after being idle.")


if "state" not in st.session_state:
    st.session_state.state = ConversationState()
state: ConversationState = st.session_state.state


with st.sidebar:
    st.image("assets/logo.png", width=120)


user_msg = st.chat_input("Describe your production scenario…")
if user_msg:
    # append and show user message
    state.add_user(user_msg)
    st.chat_message("user").write(user_msg)

    # reasoning animation + detect and update schema
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("🧠 Analyzing your problem…")
        time.sleep(0.8)
        try:
            detect_or_confirm_problem(state)
            update_schema(state)
            placeholder.write(f"Detected problem type: **{state.problem_type}**")
        except Exception as e:
            placeholder.write(f"Error during reasoning: {e}")

# Render adaptive form (pre-filled) if schema present
schema = state.schema or {}
form_data = {}
if schema.get("fields"):
    st.markdown("### 🧾 Confirm or adjust the detected parameters")
    st.info("🧠 Values inferred from your description; adjust if needed.")
    for f in schema["fields"]:
        label = f.get("label", "")
        ftype = f.get("type", "text")
        default = f.get("default") or f.get("example") or ""
        if "number" in ftype or ftype in ("float", "int"):
            try:
                val = float(default) if default not in (None, "") else 0.0
            except Exception:
                val = 0.0
            form_data[label] = st.number_input(label, value=val)
        elif "list" in ftype:
            prefill = ",".join(map(str, default)) if isinstance(default, list) else str(default)
            form_data[label] = st.text_input(label, value=prefill)
        else:
            form_data[label] = st.text_input(label, value=str(default))

    # persist form data
    st.session_state.form_data = form_data

# Clarification questions
missing = find_missing_fields(schema)
if missing:
    q = next_question(schema)
    st.chat_message("assistant").write(q)

st.write("")

with st.expander("Current conversation & schema"):
    st.markdown("**Conversation (last turns):**")
    st.write(state.summarize())
    st.markdown("**Current schema:**")
    st.json(state.schema or {})

if st.button("🚀 Optimize"):
    if not st.session_state.get("form_data"):
        st.warning("Fill the form first.")
    else:
        with st.spinner("Running optimization..."):
            result = solve_profit_lp(st.session_state.form_data)
        if result.get("status") != "Optimal":
            st.error("Optimization failed or infeasible.")
        else:
            st.success(f"✅ Optimal Profit: ${result.get('objective')}")
            st.json(result.get("allocations", {}))

            # Visualization
            df = pd.DataFrame(list(result.get("allocations", {}).items()), columns=["Product", "Units"])
            if not df.empty:
                st.plotly_chart(px.bar(df, x="Product", y="Units", title="Optimal Production Mix"))

            # Qwen reasoning explanation
            explain_prompt = f"""
We solved a profit-optimization LP and obtained these results:
{json.dumps(result, indent=2)}
Explain concisely why this allocation is optimal, which constraints are binding,
and suggest one 'what-if' adjustment to improve profit.
"""
            with st.spinner("🧠 Generating explanation..."):
                explanation = qwen_chat([{"role": "user", "content": explain_prompt}], max_tokens=400)
            st.markdown("### 🤖 Explanation")
            st.write(explanation)
            st.chat_message("assistant").write("Would you like me to explain why this mix is optimal?")
            st.balloons()

