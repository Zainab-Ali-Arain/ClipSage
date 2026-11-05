"""
ClipSage - a study assistant for six Computer Science subjects, built from YouTube lectures.

Run:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from clipsage.ui.common import handle_pending_navigation, init_state, inject_css
from clipsage.ui.home import dashboard_page, saved_page, search_page, settings_page
from clipsage.ui.math_page import math_page
from clipsage.ui.ml_page import ml_page
from clipsage.ui.subject_page import subject_page
from clipsage.ui.videos import videos_page

st.set_page_config(page_title="ClipSage", page_icon="\U0001F393", layout="wide", initial_sidebar_state="collapsed")
init_state()
inject_css()

pages = {
    "dashboard": st.Page(dashboard_page, title="Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True),
    "subject": st.Page(subject_page, title="Subject", icon=":material/school:", url_path="subject"),
    "search": st.Page(search_page, title="Search", icon=":material/search:", url_path="search"),
    "videos": st.Page(videos_page, title="Video Library", icon=":material/smart_display:", url_path="videos"),
    "math": st.Page(math_page, title="Mathematics Playground", icon=":material/calculate:", url_path="math"),
    "ml": st.Page(ml_page, title="ML Algorithm Playground", icon=":material/science:", url_path="ml"),
    "saved": st.Page(saved_page, title="Saved Data", icon=":material/bookmark:", url_path="saved"),
    "settings": st.Page(settings_page, title="Settings", icon=":material/settings:", url_path="settings"),
}
st.session_state["pages"] = pages

st.logo(str(Path(__file__).parent / "clipsage" / "ui" / "logo.svg"), size="large")
nav = st.navigation(list(pages.values()), position="hidden")   # menu is drawn below; Subject and Search open from the dashboard
with st.sidebar:
    st.page_link(pages["dashboard"], label="Dashboard", icon=":material/home:")
    current = st.session_state.get("subject")
    if current:
        st.page_link(pages["subject"], label=f"Back to {current}", icon=":material/school:")
    st.caption("Practice")
    st.page_link(pages["videos"], label="Video Library", icon=":material/smart_display:")
    st.page_link(pages["math"], label="Mathematics Playground", icon=":material/calculate:")
    st.page_link(pages["ml"], label="ML Algorithm Playground", icon=":material/science:")
    st.caption("You")
    st.page_link(pages["saved"], label="Saved Data", icon=":material/bookmark:")
    st.page_link(pages["settings"], label="Settings", icon=":material/settings:")
handle_pending_navigation()
nav.run()
