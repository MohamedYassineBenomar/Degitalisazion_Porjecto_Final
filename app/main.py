"""
AlgarveMar — Streamlit entry point.

Streamlit auto-discovers any *.py file inside `app/pages/` and turns it
into a navigable page in the sidebar. This file is the landing page
shown by default when the user runs:

    streamlit run app/main.py
"""

from pathlib import Path

import streamlit as st

from utils import init_db

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call on the page).
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AlgarveMar — Albufeira",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Make sure the bookings DB exists before any page tries to use it.
init_db()

# ---------------------------------------------------------------------------
# Brand palette injected as CSS so every page shares the same look.
# Deep navy + sea blue + sandy accent. Kept inline so we have a single file
# to ship and don't depend on external assets.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --hm-navy:  #0a3d62;
            --hm-sea:   #3c91b3;
            --hm-sky:   #d9eef7;
            --hm-sand:  #f5e6d3;
        }
        .stApp {
            background: linear-gradient(180deg, var(--hm-sky) 0%, #ffffff 320px);
        }
        h1, h2, h3 { color: var(--hm-navy); }
        .hm-hero {
            background: linear-gradient(135deg, var(--hm-navy) 0%, var(--hm-sea) 100%);
            color: white;
            padding: 56px 48px;
            border-radius: 16px;
            margin-bottom: 28px;
            box-shadow: 0 8px 24px rgba(10, 61, 98, 0.15);
        }
        .hm-hero h1 { color: white; margin: 0 0 8px 0; font-size: 2.6rem; }
        .hm-hero p  { color: rgba(255,255,255,0.92); font-size: 1.1rem; margin: 0; }
        .hm-tag     { display:inline-block; background: var(--hm-sand); color: var(--hm-navy);
                      padding: 4px 12px; border-radius: 999px; font-size: 0.85rem;
                      font-weight: 600; margin-bottom: 12px; }
        .hm-card {
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(10, 61, 98, 0.08);
            border-left: 4px solid var(--hm-sea);
            margin-bottom: 16px;
        }
        .hm-card h3 { margin-top: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hm-hero">
        <span class="hm-tag">★★★★ · ALBUFEIRA, PORTUGAL</span>
        <h1>🌊 AlgarveMar</h1>
        <p>An Algarve stay, priced fairly by AI — every day, every room.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Welcome content + quick links
# ---------------------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown("### Welcome aboard")
    st.write(
        "AlgarveMar is a 4-star, 150-room hotel right on the seafront in "
        "**Albufeira (Algarve, Portugal)**. Book directly with us and skip the third-party "
        "fees — our prices are set every day by an AI model that learns "
        "from real demand patterns, so you always see a fair rate."
    )

    st.markdown(
        """
        <div class="hm-card">
            <h3>Why book direct</h3>
            <ul>
                <li><b>No commissions:</b> the price you see is the price the hotel keeps.</li>
                <li><b>Best-rate guarantee:</b> our direct rate is never beaten by Booking.com.</li>
                <li><b>Smart pricing:</b> our model adjusts daily for season, weekday and demand.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("### Get started")
    st.info(
        "Use **Book Your Stay** in the sidebar to pick your dates and see "
        "the AI-recommended price for your room. "
        "Hotel staff: **Manager Dashboard** (Prophet) and **ML Dashboard** "
        "(LightGBM) for the analytics views — password `admin123`."
    )
    st.markdown(
        """
        <div class="hm-card">
            <h3>At a glance</h3>
            <p>🏖️ 50 m to the beach<br>
               🛏️ 150 rooms · 3 categories<br>
               🍷 Sea-view restaurant<br>
               🏊 Rooftop infinity pool</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### How it works")
_pipeline_png = DOCS_DIR / "infographic_pipeline.png"
if _pipeline_png.exists():
    st.image(str(_pipeline_png), use_container_width=True)

st.caption(
    "MVP demo — final project for the Digitalization course. "
    "AI pricing powered by **two parallel models** (Meta's Prophet + Microsoft's "
    "LightGBM with Fourier seasonality features) trained on 2+ years of historic "
    "ADR data, with an explicit price-elasticity layer (η = −0.7) on every "
    "'with AI' KPI."
)
