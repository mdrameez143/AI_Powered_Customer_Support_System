"""
Minimal Streamlit UI. Talks to the FastAPI backend over HTTP so the UI and
API stay decoupled — the UI is a thin client, not a second implementation
of the logic. Run the API first, then this.
"""
from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Powered Support System", page_icon="🎫", layout="wide")
st.title("🎫 AI Powered Support System")

with st.sidebar:
    st.markdown("### Backend status")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        st.success(f"Connected — {health['rows_loaded']} tickets loaded")
        st.caption(f"LLM provider: {health['llm_provider']}")
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot reach API at {API_BASE}\n\n{e}")

tab_query, tab_anomalies = st.tabs(["💬 Ask a question", "🚨 Anomalies"])

with tab_query:
    st.markdown("Ask a natural-language question about the ticket dataset.")
    examples = [
        "How many tickets are currently open?",
        "Which agent has the lowest average customer rating?",
        "Show me all Critical tickets not resolved within 12 hours.",
        "What is the average customer rating for Technical category tickets?",
    ]
    cols = st.columns(len(examples))
    picked = None
    for c, ex in zip(cols, examples):
        if c.button(ex, use_container_width=True):
            picked = ex

    question = st.text_input("Your question", value=picked or "", placeholder="e.g. How many tickets are open?")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{API_BASE}/query", json={"question": question}, timeout=60)
                if resp.status_code != 200:
                    st.error(resp.json().get("detail", resp.text))
                else:
                    data = resp.json()
                    st.markdown(f"**Answer:** {data['answer']}")
                    with st.expander("Show computed result / query plan"):
                        st.write("Matched rows:", data.get("matched_rows"))
                        st.json({"query_spec": data.get("query_spec"), "result": data.get("result")})
            except Exception as e:  # noqa: BLE001
                st.error(f"Request failed: {e}")

with tab_anomalies:
    st.markdown("Detected anomalies in the current dataset.")
    if st.button("Refresh anomalies"):
        st.session_state.pop("anomaly_report", None)

    if "anomaly_report" not in st.session_state:
        try:
            st.session_state["anomaly_report"] = requests.get(f"{API_BASE}/anomalies", timeout=30).json()
        except Exception as e:  # noqa: BLE001
            st.error(f"Request failed: {e}")
            st.session_state["anomaly_report"] = None

    report = st.session_state.get("anomaly_report")
    if report:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"⏱️ Resolution-time outliers ({report['resolution_time_outliers']['count']})")
            st.caption(report["resolution_time_outliers"]["description"])
            st.dataframe(report["resolution_time_outliers"]["tickets"], use_container_width=True)
        with col2:
            st.subheader(f"🔥 Stale high-priority tickets ({report['stale_high_priority']['count']})")
            st.caption(report["stale_high_priority"]["description"])
            st.dataframe(report["stale_high_priority"]["tickets"], use_container_width=True)
