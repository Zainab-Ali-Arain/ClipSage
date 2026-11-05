"""
ClipSage - a study assistant for six Computer Science subjects, built from YouTube lectures.

Run:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from clipsage.subjects import load_subjects
from clipsage.ui.common import handle_pending_navigation, init_state, inject_css
from clipsage.ui.home import dashboard_page, saved_page, search_page, settings_page
from clipsage.ui.math_page import math_page
from clipsage.ui.ml_page import ml_page
from clipsage.ui.subject_page import subject_page
from clipsage.ui.videos import videos_page

st.set_page_config(page_title="ClipSage", page_icon="\U0001F393", layout="wide", initial_sidebar_state="expanded")
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

SUBJECT_NAMES = [s["name"] for s in load_subjects()]


def _subject_changed() -> None:
    choice = st.session_state["subject_select"]
    if choice in SUBJECT_NAMES:
        st.session_state["subject"] = choice
        st.session_state["subject_section"] = "Overview"
        st.session_state["_goto"] = "subject"
    else:
        st.session_state["_goto"] = "dashboard"


st.logo(str(Path(__file__).parent / "clipsage" / "ui" / "logo.svg"), size="large")
with st.sidebar:
    current = st.session_state.get("subject")
    st.session_state["subject_select"] = current if current in SUBJECT_NAMES else "All subjects (Dashboard)"
    st.selectbox("Subject", ["All subjects (Dashboard)"] + SUBJECT_NAMES, key="subject_select", on_change=_subject_changed)
    if current in SUBJECT_NAMES:
        st.page_link(pages["subject"], label=f"Open {current}", icon=":material/arrow_forward:")
    st.page_link(pages["dashboard"], label="Back to dashboard", icon=":material/home:")

nav = st.navigation({
    "Learn": [pages["dashboard"], pages["subject"], pages["search"], pages["videos"]],
    "Practice": [pages["math"], pages["ml"]],
    "You": [pages["saved"], pages["settings"]],
})
handle_pending_navigation()
nav.run()
