"""
Manager Dashboard.

Hotel staff sign in with a hard-coded password (DEMO ONLY — production
would use proper auth such as OIDC/SSO via Streamlit's auth or Auth0).

Once in, they see:
  1. Today's AI-recommended price vs the old static rate
  2. A 90-day price forecast with confidence band
  3. Four headline KPIs (bookings, revenue, avg price, occupancy)
  4. The 20 most recent bookings
  5. Daily revenue across booked nights
"""

from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sibling `utils` importable when run from the pages/ folder.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import (  # noqa: E402
    OLD_STATIC_PRICE,
    TOTAL_ROOMS,
    compute_kpis,
    get_all_bookings,
    predict_prices,
    revenue_per_day,
)

# ---------------------------------------------------------------------------
# DEMO PASSWORD — production would use a real auth backend.
# ---------------------------------------------------------------------------
DEMO_PASSWORD = "admin123"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FORECAST_CSV = PROJECT_ROOT / "data" / "forecast.csv"
HISTORY_CSV = PROJECT_ROOT / "data" / "daily_prices.csv"

# ---------------------------------------------------------------------------
# Page config + shared brand styles (kept consistent with the guest page).
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Manager — HotelMar", page_icon="🌊", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --hm-navy:  #0a3d62;
            --hm-sea:   #3c91b3;
            --hm-sky:   #d9eef7;
            --hm-sand:  #f5e6d3;
            --hm-up:    #2e7d32;
            --hm-down:  #c0392b;
        }
        .stApp {
            background: linear-gradient(180deg, var(--hm-sky) 0%, #ffffff 320px);
        }
        h1, h2, h3 { color: var(--hm-navy); }
        .hm-hero {
            background: linear-gradient(135deg, var(--hm-navy) 0%, var(--hm-sea) 100%);
            color: white;
            padding: 36px 40px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(10, 61, 98, 0.15);
        }
        .hm-hero h1 { color: white; margin: 0 0 4px 0; font-size: 2rem; }
        .hm-hero p  { color: rgba(255,255,255,0.92); margin: 0; }
        .hm-logo {
            background: linear-gradient(135deg, var(--hm-navy) 0%, var(--hm-sea) 100%);
            color: white;
            padding: 16px 18px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 18px;
        }
        .hm-logo .wave { font-size: 1.6rem; display:block; }
        .hm-logo .name { font-size: 1.2rem; font-weight: 700; letter-spacing: 0.04em; }
        .hm-logo .role { font-size: 0.78rem; opacity: 0.85; letter-spacing: 0.08em; text-transform: uppercase; }
        .hm-priceCard {
            background: white;
            padding: 28px 32px;
            border-radius: 14px;
            box-shadow: 0 4px 16px rgba(10, 61, 98, 0.10);
            border-left: 6px solid var(--hm-sea);
        }
        .hm-priceCard .label { color:#5b6b78; font-size:0.95rem; margin:0 0 4px 0; }
        .hm-priceCard .value { color: var(--hm-navy); font-size: 3rem; font-weight: 700;
                               line-height: 1.1; margin: 4px 0 8px 0; }
        .hm-priceCard .delta-up   { color: var(--hm-up); font-weight: 600; }
        .hm-priceCard .delta-down { color: var(--hm-down); font-weight: 600; }
        .hm-priceCard .sub { color:#5b6b78; font-size:0.9rem; margin:0; }
        .hm-section { margin-top: 28px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — logo + refresh control (visible regardless of auth state).
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="hm-logo">
            <span class="wave">🌊</span>
            <div class="name">HotelMar</div>
            <div class="role">Manager Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("↻ Refresh Data", use_container_width=True):
        # Drop any cached predictions / cached_data so KPIs and charts
        # reflect the latest bookings and any retrained model.
        st.cache_data.clear()
        st.rerun()
    st.caption("Demo password: `admin123`")


# ---------------------------------------------------------------------------
# Password gate.
# ---------------------------------------------------------------------------
if "manager_authed" not in st.session_state:
    st.session_state.manager_authed = False

if not st.session_state.manager_authed:
    st.markdown(
        """
        <div class="hm-hero">
            <h1>🔒 Manager sign-in</h1>
            <p>Enter the management password to access pricing and booking analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pwd = st.text_input("Password", type="password", placeholder="admin123")
    if st.button("Sign in", type="primary"):
        if pwd == DEMO_PASSWORD:
            st.session_state.manager_authed = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.caption(
        "DEMO ONLY: the password is hard-coded in the source. "
        "A production deployment must use a proper auth backend "
        "(OIDC/SSO, Auth0, or Streamlit's built-in auth)."
    )
    st.stop()


# =============================================================================
# AUTHED — the rest of the page only renders after a successful sign-in.
# =============================================================================

st.markdown(
    """
    <div class="hm-hero">
        <h1>Manager Dashboard</h1>
        <p>AI-recommended pricing · live KPIs · booking analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data once per render so all sections share the same snapshot.
bookings_df = get_all_bookings()
kpis = compute_kpis(bookings_df)


# -----------------------------------------------------------------------------
# Section 1 — Today's Recommended Price
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("Today's recommended price")

today = date.today()
today_pred = predict_prices(today, today + timedelta(days=1))
ai_price = float(today_pred["yhat"].iloc[0]) if not today_pred.empty else 0.0

diff_eur = ai_price - OLD_STATIC_PRICE
diff_pct = (diff_eur / OLD_STATIC_PRICE) * 100.0 if OLD_STATIC_PRICE else 0.0
delta_class = "delta-up" if diff_eur >= 0 else "delta-down"
arrow = "▲" if diff_eur >= 0 else "▼"

c1, c2 = st.columns([1.4, 1])
with c1:
    st.markdown(
        f"""
        <div class="hm-priceCard">
            <p class="label">AI suggestion · Standard Sea View · {today.strftime('%A %d %b %Y')}</p>
            <div class="value">€{ai_price:,.2f}</div>
            <p class="{delta_class}">
                {arrow} €{abs(diff_eur):,.2f} ({diff_pct:+.1f}%) vs old static price (€{OLD_STATIC_PRICE:.0f})
            </p>
            <p class="sub">Apply this rate across all available rooms tonight to maximize RevPAR.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.metric("Old static price", f"€{OLD_STATIC_PRICE:.2f}",
              help="The flat seasonal rate that HotelMar used before AI pricing.")
    st.metric("AI suggestion", f"€{ai_price:,.2f}",
              delta=f"{diff_pct:+.1f}% vs static")


# -----------------------------------------------------------------------------
# Section 2 — 90-Day Price Forecast
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("90-day price forecast")

history = pd.read_csv(HISTORY_CSV, parse_dates=["ds"])
forecast = pd.read_csv(FORECAST_CSV, parse_dates=["ds"])

last_train = history["ds"].max()
forecast_only = forecast[forecast["ds"] > last_train]

fig = go.Figure()
# Confidence band on the forecast portion.
fig.add_trace(go.Scatter(
    x=list(forecast_only["ds"]) + list(forecast_only["ds"][::-1]),
    y=list(forecast_only["yhat_upper"]) + list(forecast_only["yhat_lower"][::-1]),
    fill="toself",
    fillcolor="rgba(230,126,34,0.18)",
    line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    name="80% interval",
    showlegend=True,
))
# Historical actuals.
fig.add_trace(go.Scatter(
    x=history["ds"], y=history["y"],
    mode="lines",
    line=dict(color="#3c91b3", width=2),
    name="Historical (actual)",
))
# Forecast.
fig.add_trace(go.Scatter(
    x=forecast_only["ds"], y=forecast_only["yhat"],
    mode="lines",
    line=dict(color="#e67e22", width=2.5),
    name="Forecast (AI)",
))
fig.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Date", gridcolor="#e9eef3"),
    yaxis=dict(title="Price (EUR)", gridcolor="#e9eef3"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Blue line: real ADR observed in the historical training window. "
    "Orange line: 90-day forward AI forecast. Shaded band: 80% prediction interval."
)


# -----------------------------------------------------------------------------
# Section 3 — KPIs
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("Key performance indicators")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total bookings", f"{kpis['total_bookings']:,}")
k2.metric("Total revenue", f"€{kpis['total_revenue']:,.2f}")
k3.metric("Avg price / night", f"€{kpis['avg_price']:,.2f}")
k4.metric("Avg occupancy", f"{kpis['avg_occupancy']*100:.1f}%",
          help=f"Room-nights sold ÷ ({TOTAL_ROOMS} rooms × days in booking window).")

if bookings_df.empty:
    st.info(
        "No bookings yet. Use the **Book Your Stay** page to add some, "
        "then come back and click **↻ Refresh Data** in the sidebar."
    )


# -----------------------------------------------------------------------------
# Section 4 — Recent Bookings
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("Recent bookings")

if bookings_df.empty:
    st.write("_No bookings recorded yet._")
else:
    recent = bookings_df.head(20).copy()
    recent["check_in"] = recent["check_in"].dt.strftime("%d %b %Y")
    recent["check_out"] = recent["check_out"].dt.strftime("%d %b %Y")
    recent["created_at"] = recent["created_at"].dt.strftime("%d %b %Y %H:%M")
    recent["total_price"] = recent["total_price"].round(2)
    recent = recent.rename(columns={
        "id": "Ref #", "guest_name": "Guest", "email": "Email",
        "check_in": "Check-in", "check_out": "Check-out",
        "room_type": "Room", "total_price": "Total (EUR)",
        "nights": "Nights", "created_at": "Booked at",
    })
    st.dataframe(
        recent[["Ref #", "Guest", "Email", "Check-in", "Check-out",
                "Nights", "Room", "Total (EUR)", "Booked at"]],
        hide_index=True,
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Section 5 — Revenue by Day
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("Revenue by day")

rev = revenue_per_day(bookings_df)
if rev.empty:
    st.write("_Nothing to chart yet — make a booking first._")
else:
    bar = go.Figure()
    bar.add_trace(go.Bar(
        x=rev["date"], y=rev["revenue"],
        marker_color="#3c91b3",
        name="Revenue (EUR)",
        hovertemplate="%{x|%a %d %b %Y}<br><b>€%{y:,.2f}</b><extra></extra>",
    ))
    bar.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Stay night", gridcolor="#e9eef3"),
        yaxis=dict(title="Revenue (EUR)", gridcolor="#e9eef3"),
        showlegend=False,
    )
    st.plotly_chart(bar, use_container_width=True)
    st.caption(
        f"Daily revenue computed by spreading each booking's total over "
        f"its stay nights. Total across {len(rev):,} stay nights: "
        f"€{rev['revenue'].sum():,.2f}."
    )
