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


st.set_page_config(page_title="Production Profit Optimizer", page_icon="📈", layout="wide")
st.markdown("<h3 style='text-align:center;'>🧮 AI-Driven Operations Reasoner</h3>", unsafe_allow_html=True)
st.info("⏳ App may take 20–30 s to start after idle. Please wait while the reasoning engine loads.")


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

    # reasoning with loading feedback
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking through your constraints..."):
            try:
                detect_or_confirm_problem(state)
                update_schema(state)
            except Exception as e:
                st.error(f"Error during reasoning: {e}")
        st.success(f"Detected: {state.problem_type}")
        # optional typed message flair
        def type_text(text: str, delay: float = 0.01):
            ph = st.empty()
            typed = ""
            for ch in text:
                typed += ch
                ph.markdown(typed)
                time.sleep(delay)
            return

        type_text(f"Detected problem type: {state.problem_type}")

schema = state.schema or {}
form_data = {}
if schema.get("fields"):
    st.markdown("### 🧾 Confirm or adjust the detected parameters")
    st.info("🧠 Values inferred from your description; adjust if needed.")
    # load previous form values if present to persist across reruns
    prev = st.session_state.get("form_data", {})
    for f in schema["fields"]:
        label = f.get("label", "")
        ftype = f.get("type", "text")
        default = f.get("default") or f.get("example") or ""
        # prefer previous value if present
        pre_value = prev.get(label, None)
        if pre_value is not None:
            default = pre_value

        if "number" in ftype or ftype in ("float", "int"):
            try:
                val = float(default) if default not in (None, "") else 0.0
            except Exception:
                val = 0.0
            form_data[label] = st.number_input(label, value=val, key=f"num_{label}")
        elif "list" in ftype:
            prefill = ",".join(map(str, default)) if isinstance(default, list) else str(default)
            form_data[label] = st.text_input(label, value=prefill, key=f"list_{label}")
        else:
            form_data[label] = st.text_input(label, value=str(default), key=f"txt_{label}")

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

            # What-if simulation section
            st.markdown("### 🔁 What-If Simulation")
            factor = st.slider("Change resource capacity (%)", 50, 150, 100, 10)
            if st.button("Re-simulate"):
                st.info(f"Scaling capacities by {factor}%...")
                # try to scale capacities in a copy of form_data
                fd = dict(st.session_state.form_data)
                cap_label = "Available capacity"
                caps = fd.get(cap_label, "")
                try:
                    vals = [float(x.strip()) for x in str(caps).split(",") if x.strip()]
                    scaled = [str(round(v * factor / 100.0, 4)) for v in vals]
                    fd[cap_label] = ",".join(scaled)
                    with st.spinner("Re-running optimization with scaled capacities..."):
                        res2 = solve_profit_lp(fd)
                    if res2.get("status") == "Optimal":
                        st.success(f"Resimulated Optimal Profit: ${res2.get('objective')}")
                        st.json(res2.get("allocations", {}))
                    else:
                        st.error("Resimulation infeasible or failed.")
                except Exception:
                    st.error("Could not parse capacities for resimulation.")

