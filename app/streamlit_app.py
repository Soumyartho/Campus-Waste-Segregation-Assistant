"""Campus Waste Segregation Assistant - Streamlit entry point.

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Campus Waste Segregation Assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import common  # noqa: E402  (must follow set_page_config)

common.inject_css()

pages = [
    st.Page("pages/classify.py", title="Classify", icon=":material/photo_camera:", default=True),
    st.Page("pages/batch.py", title="Batch", icon=":material/photo_library:"),
    st.Page("pages/analytics.py", title="Analytics", icon=":material/monitoring:"),
    st.Page("pages/insights.py", title="Model insights", icon=":material/insights:"),
    st.Page("pages/about.py", title="About", icon=":material/info:"),
]

with st.sidebar:
    st.markdown("### ♻️ Waste Segregation Assistant")
    predictor = common.get_predictor()
    if predictor is None:
        st.caption("Model: not loaded")
    else:
        st.caption(f"Model: `{predictor.model_name}` · CPU · tau {predictor.tau:.2f}")

st.navigation(pages).run()
