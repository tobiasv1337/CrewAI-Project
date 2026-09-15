"""Server entry point: ``uv run streamlit run app.py``."""

from pathlib import Path

import streamlit as st

from ui.web_branding import branding_routes


app = st.App(
    Path(__file__).resolve().with_name("study_manager.py"),
    routes=branding_routes(st.get_option("server.baseUrlPath")),
)
