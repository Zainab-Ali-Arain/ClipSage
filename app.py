"""
ClipSage - a bilingual study assistant built from YouTube lectures.

Run:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from clipsage.i18n import LANGUAGES, tl
from clipsage.ui.common import TL, handle_pending_navigation, init_state, inject_css, remember_language
from clipsage.ui.home import dashboard_page, explore_page, progress_page, saved_page, settings_page
from clipsage.ui.learn import learn_page, practice_page
from clipsage.ui.math_page import math_page
from clipsage.ui.ml_page import ml_page
from clipsage.ui.videos import videos_page

st.set_page_config(page_title="ClipSage", page_icon="\U0001F393", layout="wide", initial_sidebar_state="expanded")
init_state()
inject_css()

PAGES = [
    ("dashboard", dashboard_page, "nav_dashboard", ":material/dashboard:", True),
    ("explore", explore_page, "nav_explore", ":material/explore:", False),
    ("videos", videos_page, "nav_videos", ":material/smart_display:", False),
    ("learn", learn_page, "nav_learn", ":material/menu_book:", False),
    ("practice", practice_page, "nav_practice", ":material/edit_note:", False),
    ("math", math_page, "nav_math", ":material/calculate:", False),
    ("ml", ml_page, "nav_ml", ":material/science:", False),
    ("saved", saved_page, "nav_saved", ":material/bookmark:", False),
    ("progress", progress_page, "nav_progress", ":material/trending_up:", False),
    ("settings", settings_page, "nav_settings", ":material/settings:", False),
]

# Navigation labels: English when "Both" is selected (keeps the menu short), otherwise the chosen language.
nav_lang = "en" if st.session_state.get("lang") == "both" else st.session_state.get("lang", "en")
pages = {key: st.Page(fn, title=tl(label, nav_lang), icon=icon, url_path=key, default=default)
         for key, fn, label, icon, default in PAGES}
st.session_state["pages"] = pages

st.logo(str(Path(__file__).parent / "clipsage" / "ui" / "logo.svg"), size="large")
with st.sidebar:
    st.selectbox(TL("language"), list(LANGUAGES), format_func=LANGUAGES.get, key="lang", on_change=remember_language)
    st.caption("Learn from real lecture moments · Asli lecture moments se seekhein")

nav = st.navigation({
    "ClipSage": [pages["dashboard"], pages["explore"], pages["videos"], pages["learn"]],
    "Practice": [pages["practice"], pages["math"], pages["ml"]],
    "You": [pages["saved"], pages["progress"], pages["settings"]],
})
handle_pending_navigation()
nav.run()
