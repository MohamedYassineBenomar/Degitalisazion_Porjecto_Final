"""
Machine Learning Dashboard.

Parallel to 2_Manager_Dashboard.py. Same data, same demo bookings,
same hold-out window — but every price prediction comes from a
**LightGBM with Fourier features** (gradient-boosted decision trees) instead of Prophet.
Lets the jury compare a classical-ML approach with the time-series
baseline side by side.

Visual language:
  - Prophet dashboard uses ORANGE for the AI traces.
  - This page uses PURPLE so a glance tells you which model is in
    front of you.
"""

from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import (  # noqa: E402
    DEMO_DATE,
    PRICE_ELASTICITY,
    TOTAL_ROOMS,
    VARIABLE_COST_PER_ROOM_NIGHT,
    compute_naive_kpis_ml,
    compute_elasticity_adjusted_kpis_ml,
    compute_static_baseline_kpis,
    get_all_bookings,
    historical_monthly_avg,
    predict_prices_ml,
)

DEMO_PASSWORD = "admin123"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_FORECAST_CSV = PROJECT_ROOT / "data" / "forecast_ml.csv"
ML_BLIND_TEST_CSV = PROJECT_ROOT / "data" / "blind_test_predictions_ml.csv"
HISTORY_CSV = PROJECT_ROOT / "data" / "daily_prices.csv"
PROPHET_BLIND_TEST_CSV = PROJECT_ROOT / "data" / "blind_test_predictions.csv"
ELASTICITY_INFOGRAPHIC = PROJECT_ROOT / "docs" / "infographic_elasticity.png"

# ---------------------------------------------------------------------------
# Shared helper used by both KPI tables on this page.
# ---------------------------------------------------------------------------
def diff_cell(static_v: float, ai_v: float, fmt: str, mode: str = "default") -> str:
    """Render the right-hand 'Difference' cell. Same modes as in the
    Prophet dashboard; documented there."""
    raw = ai_v - static_v
    if abs(raw) < 0.005:
        return '<td class="diff-cell diff-flat">no change</td>'
    display_diff = -raw if mode == "cost" else raw
    sign = "+" if display_diff > 0 else "−"
    body = fmt.format(abs(display_diff))
    if mode == "orange":
        cls = "diff-orange"
    elif mode == "headline":
        cls = "diff-headline"
    else:
        cls = "diff-up" if display_diff > 0 else "diff-down"
    return f'<td class="diff-cell {cls}">{sign}{body}</td>'


st.set_page_config(page_title="ML Manager — AlgarveMar", page_icon="🤖", layout="wide")

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
            --hm-ml:    #8e44ad;        /* purple, distinguishes ML from Prophet's orange */
        }
        .stApp {
            background: linear-gradient(180deg, var(--hm-sky) 0%, #ffffff 320px);
        }
        h1, h2, h3 { color: var(--hm-navy); }
        .hm-hero {
            background: linear-gradient(135deg, var(--hm-navy) 0%, var(--hm-ml) 100%);
            color: white;
            padding: 36px 40px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(10, 61, 98, 0.15);
        }
        .hm-hero h1 { color: white; margin: 0 0 4px 0; font-size: 2rem; }
        .hm-hero p  { color: rgba(255,255,255,0.92); margin: 0; }
        .hm-logo {
            background: linear-gradient(135deg, var(--hm-navy) 0%, var(--hm-ml) 100%);
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
            background: white; padding: 28px 32px; border-radius: 14px;
            box-shadow: 0 4px 16px rgba(10, 61, 98, 0.10);
            border-left: 6px solid var(--hm-ml);
        }
        .hm-priceCard .label { color:#5b6b78; font-size:0.95rem; margin:0 0 4px 0; }
        .hm-priceCard .value { color: var(--hm-navy); font-size: 3rem; font-weight: 700;
                               line-height: 1.1; margin: 4px 0 8px 0; }
        .hm-priceCard .delta-up   { color: var(--hm-up); font-weight: 600; }
        .hm-priceCard .delta-down { color: var(--hm-down); font-weight: 600; }
        .hm-priceCard .sub { color:#5b6b78; font-size:0.9rem; margin:0; }
        .hm-section { margin-top: 28px; }
        .hm-chartKey {
            background: white; padding: 16px 20px; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(10, 61, 98, 0.06);
            border-left: 4px solid var(--hm-ml); margin-top: 12px;
        }
        .hm-chartKey ul { margin: 6px 0 0 0; padding-left: 18px; }
        .hm-chartKey li { margin-bottom: 6px; color: #1c2b3a; }
        .hm-chartKey li b { color: var(--hm-navy); }
        .hm-swatch {
            display: inline-block; width: 22px; height: 4px;
            vertical-align: middle; border-radius: 2px;
            margin-right: 8px; margin-bottom: 2px;
        }
        .hm-swatch.dashed { height: 0; border-top: 3px dashed currentColor; background: transparent; }
        .hm-swatch.band { height: 10px; opacity: 0.35; }

        .hm-compare {
            background: white; padding: 4px 0; border-radius: 14px;
            box-shadow: 0 4px 16px rgba(10, 61, 98, 0.08); overflow: hidden;
        }
        .hm-compare table { width: 100%; border-collapse: collapse; font-size: 0.98rem; }
        .hm-compare th {
            text-align: left; padding: 14px 18px;
            background: var(--hm-sky); color: var(--hm-navy);
            font-weight: 600; font-size: 0.95rem;
            white-space: normal; vertical-align: bottom;
            border-bottom: 2px solid var(--hm-ml);
        }
        .hm-compare th small {
            display: block; font-weight: 400; color: #5b6b78;
            font-size: 0.78rem; margin-top: 2px;
        }
        .hm-compare td {
            padding: 16px 18px; border-bottom: 1px solid #eef3f7;
            vertical-align: middle; white-space: normal; word-break: keep-all;
        }
        .hm-compare tr:last-child td { border-bottom: none; }
        .hm-compare .metric-name { font-weight: 600; color: var(--hm-navy); white-space: normal; }
        .hm-compare .static-cell { color: #5b6b78; font-variant-numeric: tabular-nums; }
        .hm-compare .ai-cell { color: var(--hm-navy); font-weight: 700; font-variant-numeric: tabular-nums; }
        .hm-compare .lift-flat { color: #95a5a6; font-weight: 400; }

        .hm-compare .diff-cell {
            font-weight: 700; color: #000;
            font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap;
        }
        .hm-compare .diff-up   { background: #d4f5dc; }
        .hm-compare .diff-down { background: #fbd7d7; }
        .hm-compare .diff-flat { background: #f0f0f0; color: #000; }
        .hm-compare .diff-orange {
            background: #ffd9a8 !important; color: #000 !important;
        }
        .hm-compare .diff-headline {
            background: #1E7C2F !important; color: #fff !important; font-weight: 800 !important;
        }
        .hm-compare tr.hm-row-profit td {
            background: #f7fbf8; border-top: 1px solid #d3e8d4;
            border-bottom: 1px solid #d3e8d4; font-weight: 700;
        }
        .hm-compare tr.hm-row-profit .metric-name { color: var(--hm-up); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — same logo card pattern as the Prophet dashboard.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="hm-logo">
            <span class="wave">🤖</span>
            <div class="name">AlgarveMar</div>
            <div class="role">ML · Manager Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("↻ Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Demo password: `admin123`")


# ---------------------------------------------------------------------------
# Password gate (separate session_state key from the Prophet page so the
# two dashboards don't share auth — keeps the demo behaviour explicit).
# ---------------------------------------------------------------------------
if "ml_manager_authed" not in st.session_state:
    st.session_state.ml_manager_authed = False

if not st.session_state.ml_manager_authed:
    st.markdown(
        """
        <div class="hm-hero">
            <h1>🔒 ML Manager sign-in</h1>
            <p>Same demo credentials as the main manager dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pwd = st.text_input(
        "Password", type="password",
        value=DEMO_PASSWORD,  # pre-filled for demo convenience
    )
    if st.button("Sign in", type="primary"):
        if pwd == DEMO_PASSWORD:
            st.session_state.ml_manager_authed = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.caption(
        "DEMO ONLY: the password is hard-coded in source. Production "
        "would use a real auth backend (OIDC/SSO, Auth0, etc)."
    )
    st.stop()


# =============================================================================
# AUTHED — the rest of the page only renders after sign-in.
# =============================================================================

st.markdown(
    """
    <div class="hm-hero">
        <h1>ML Manager Dashboard 🤖</h1>
        <p>Same demo, same data — pricing predictions by a LightGBM with Fourier features (gradient-boosted decision trees)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Shared data loads.
bookings_df = get_all_bookings()
history = pd.read_csv(HISTORY_CSV, parse_dates=["ds"])


# -----------------------------------------------------------------------------
# Section 1 — Today's ML-Recommended Price
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader(f"ML Recommendation for {DEMO_DATE.strftime('%A %d %B %Y')}")

today_pred = predict_prices_ml(DEMO_DATE, DEMO_DATE + timedelta(days=1))
ml_price = float(today_pred["yhat"].iloc[0]) if not today_pred.empty else 0.0

baseline = historical_monthly_avg(DEMO_DATE.month)
month_name = DEMO_DATE.strftime("%B")
diff_eur = ml_price - baseline
diff_pct = (diff_eur / baseline) * 100.0 if baseline else 0.0
delta_class = "delta-up" if diff_eur >= 0 else "delta-down"
arrow = "▲" if diff_eur >= 0 else "▼"

c1, c2 = st.columns([1.4, 1])
with c1:
    st.markdown(
        f"""
        <div class="hm-priceCard">
            <p class="label">ML suggestion · Standard Sea View · {DEMO_DATE.strftime('%A %d %b %Y')}</p>
            <div class="value">€{ml_price:,.2f}</div>
            <p class="{delta_class}">
                {arrow} €{abs(diff_eur):,.2f} ({diff_pct:+.1f}%) vs historical {month_name} avg (€{baseline:.2f})
            </p>
            <p class="sub">Generated by a LightGBM model — 18 features incl. Fourier seasonality + lag-365.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.metric(
        f"Historical {month_name} avg",
        f"€{baseline:,.2f}",
        help=f"Average ADR across all {month_name} days in the training data.",
    )
    st.metric("ML suggestion", f"€{ml_price:,.2f}",
              delta=f"{diff_pct:+.1f}% vs {month_name} avg")


# -----------------------------------------------------------------------------
# Section 2 — Blind-test forecast vs same period last year (Jul-Aug 2017)
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("Blind-test forecast vs same period last year (Jul – Aug 2017)")

bt_pred = pd.read_csv(ML_BLIND_TEST_CSV, parse_dates=["ds"])

yoy_window_start = bt_pred["ds"].min() - pd.DateOffset(years=1)
yoy_window_end = bt_pred["ds"].max() - pd.DateOffset(years=1)
yoy_2016 = history[
    (history["ds"] >= yoy_window_start) & (history["ds"] <= yoy_window_end)
].copy()
yoy_2016["ds_shifted"] = yoy_2016["ds"] + pd.DateOffset(years=1)

fig_bt_chart = go.Figure()
fig_bt_chart.add_trace(go.Scatter(
    x=list(bt_pred["ds"]) + list(bt_pred["ds"][::-1]),
    y=list(bt_pred["yhat_upper"]) + list(bt_pred["yhat_lower"][::-1]),
    fill="toself",
    fillcolor="rgba(142,68,173,0.16)",
    line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    name="~80% interval",
    showlegend=True,
))
fig_bt_chart.add_trace(go.Scatter(
    x=yoy_2016["ds_shifted"], y=yoy_2016["y"],
    mode="lines",
    line=dict(color="#27ae60", width=1.6, dash="dash"),
    name="Same period last year (2016 actuals)",
    hovertemplate="%{x|%d %b 2017} (was %{x|%d %b 2016})<br>€%{y:.2f}<extra></extra>",
))
fig_bt_chart.add_trace(go.Scatter(
    x=bt_pred["ds"], y=bt_pred["y_actual"],
    mode="lines",
    line=dict(color="#3c91b3", width=2),
    name="Actual (2017)",
))
fig_bt_chart.add_trace(go.Scatter(
    x=bt_pred["ds"], y=bt_pred["yhat"],
    mode="lines",
    line=dict(color="#8e44ad", width=2.5),
    name="ML blind-test prediction",
))
fig_bt_chart.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.92)", bordercolor="#e5edf2",
                borderwidth=1, font=dict(size=12)),
    xaxis=dict(title="Date (2017)", gridcolor="#e9eef3"),
    yaxis=dict(title="Price (EUR)", gridcolor="#e9eef3"),
    hovermode="x unified",
)
st.plotly_chart(fig_bt_chart, use_container_width=True)

# Compute MAPE from the loaded blind-test predictions for the caption.
ml_err = bt_pred["y_actual"] - bt_pred["yhat"]
ml_mae = float(ml_err.abs().mean())
ml_mape = float((ml_err.abs() / bt_pred["y_actual"].abs() * 100).mean())

st.caption(
    f"Same 62-day blind-test slice as the Prophet dashboard. "
    f"**Blue:** real 2017 ADR. **Purple:** what the LightGBM "
    f"model predicted using only data up to 30 June 2017. **Green "
    f"dashed:** 2016 actuals on the same calendar dates. "
    f"**ML MAPE on this window: {ml_mape:.2f}%** (Prophet: 5.75% on "
    f"the same data — Prophet's smooth decomposition fits the August "
    f"peak more tightly than the tree ensemble's discrete splits)."
)


# -----------------------------------------------------------------------------
# Section 2c — The math behind "higher price -> fewer bookings"
# Plain-language + LaTeX walk-through of the price-elasticity formula
# that drives every "with AI / with ML" KPI on this page.
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("📐 The math: how we model \"higher price → fewer bookings\"")

st.markdown(
    "When the AI raises a room's price above the static rulebook, **some "
    "guests will refuse to book at the higher rate**. That's basic economics — "
    "a demand curve slopes downward. We capture this drop using the standard "
    "economic measure called **price elasticity of demand**."
)

st.markdown("**Step 1 — Definition of price elasticity of demand**")
st.latex(r"\eta \;=\; \frac{\%\,\Delta Q}{\%\,\Delta P} \;=\; \frac{\Delta Q / Q}{\Delta P / P}")
st.markdown(
    "- **η (eta)** is the elasticity coefficient.\n"
    "- **ΔQ / Q** is the *percentage* change in quantity sold (bookings).\n"
    "- **ΔP / P** is the *percentage* change in price.\n\n"
    "For any normal good, elasticity is **negative**: a price rise causes "
    "a quantity drop. The size of η tells you *how sensitive* the market is."
)

st.markdown("**Step 2 — The value we use for AlgarveMar**")
st.latex(r"\eta \;=\; -0.7 \quad \text{(4-star mid-range hotel, industry literature)}")
st.markdown(
    "Empirical studies report elasticity between **−0.4 and −0.8** for "
    "mid-range hotels. We chose **−0.7** (conservative end — more guests "
    "walk away). **Interpretation:** a **10 % price hike loses 7 % of "
    "bookings**; a **20 % hike loses 14 %**, and so on."
)

st.markdown("**Step 3 — Solving the formula for retained bookings**")
st.markdown(
    "From the elasticity definition we can isolate the *retention ratio* — "
    "the fraction of original bookings we keep at the new price:"
)
st.latex(r"\frac{Q_{\text{kept}}}{Q_{\text{original}}} \;=\; 1 \;+\; \eta \cdot \frac{\Delta P}{P}")

st.markdown("**Step 4 — Per-booking retention (what the dashboard actually computes)**")
st.markdown(
    "For every individual booking $i$ in the demo dataset, we compute:"
)
st.latex(r"""
\text{pct\_change}_i \;=\; \frac{P_{\text{AI},\,i} - P_{\text{static},\,i}}{P_{\text{static},\,i}}
""")
st.latex(r"""
\text{retention}_i \;=\; \max\!\bigl(0,\; \min(1,\; 1 + \eta \cdot \text{pct\_change}_i)\bigr)
""")
st.markdown(
    "- We **clip** the result to the interval **[0, 1]** because a booking "
    "can't fall below 0 guests, and we don't model demand *expansion* "
    "when prices drop (a more sophisticated version could).\n"
    "- The **revenue** contribution of booking $i$ becomes "
    "$\\text{retention}_i \\times P_{\\text{AI},\\,i}$.\n"
    "- Summing across all bookings gives the **realistic** revenue, room-"
    "night and profit numbers in the comparison tables below."
)

st.markdown("**Worked example — one booking**")
st.markdown(
    "Suppose for a particular booking the static rulebook says **€154.53/night** "
    "and the AI says **€185.50/night**. Plugging into the formulas:"
)
st.latex(r"""
\text{pct\_change} \;=\; \frac{185.50 - 154.53}{154.53} \;=\; +0.200
\quad (= +20\%)
""")
st.latex(r"""
\text{retention} \;=\; 1 + (-0.7) \cdot 0.200 \;=\; 1 - 0.14 \;=\; \mathbf{0.86}
""")
st.markdown(
    "**86 %** of guests still book at the higher price; **14 % walk away** "
    "and the hotel loses those reservations and their downstream variable costs."
)

# Visual companion to the math above — the demand curve, with the
# +20 % worked example highlighted on it. Generated by
# scripts/05_render_infographics.py.
if ELASTICITY_INFOGRAPHIC.exists():
    st.image(str(ELASTICITY_INFOGRAPHIC), use_container_width=True)

st.markdown("**Why the ML page shows fewer retained bookings than the Prophet page**")
st.markdown(
    "On this page (LightGBM) the **average ΔP / P across all demo bookings "
    "is around +30 %**, vs roughly **+20 % on the Prophet page**. Same "
    "elasticity (η = −0.7), bigger price uplift → bigger demand haircut:\n\n"
    "- Prophet's average retention ≈ **0.82** (~82 % of guests book)\n"
    "- LightGBM's average retention ≈ **0.75** (~75 % of guests book)\n\n"
    "The **formula is identical**; only the AI's price suggestion differs. "
    "Which model's prices a real hotel should actually charge is a "
    "**revenue-vs-volume** trade-off — exactly the conversation the "
    "comparison tables below help the manager have."
)

if not bookings_df.empty:
    BLIND_START = pd.Timestamp("2017-07-01")
    BLIND_END = pd.Timestamp("2017-08-31")
    blind_df = bookings_df[
        (bookings_df["check_in"] >= BLIND_START)
        & (bookings_df["check_in"] <= BLIND_END)
    ].copy()

    if blind_df.empty:
        st.info("No demo bookings fall inside Jul – Aug 2017.")
    else:
        bt_static = compute_static_baseline_kpis(blind_df)
        bt_ml = compute_elasticity_adjusted_kpis_ml(blind_df)

        bt_d_bookings  = diff_cell(bt_static["total_bookings"],       bt_ml["total_bookings"],       "{:,.0f}")
        bt_d_revenue   = diff_cell(bt_static["total_revenue"],        bt_ml["total_revenue"],        "€{:,.2f}")
        bt_d_costs     = diff_cell(bt_static["total_variable_cost"],  bt_ml["total_variable_cost"],  "€{:,.2f}", mode="cost")
        bt_d_margin    = diff_cell(bt_static["gross_margin"]*100,     bt_ml["gross_margin"]*100,     "{:.1f}%")
        bt_d_price     = diff_cell(bt_static["avg_price"],            bt_ml["avg_price"],            "€{:,.2f}", mode="orange")
        bt_d_occupancy = diff_cell(bt_static["avg_occupancy"]*100,    bt_ml["avg_occupancy"]*100,    "{:.1f}%")
        bt_d_netprofit = diff_cell(bt_static["gross_profit"],         bt_ml["gross_profit"],         "€{:,.2f}", mode="headline")

        st.markdown(
            f"""
            <div class="hm-compare">
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Without AI<small>static seasonal rulebook</small></th>
                            <th>With ML<small>LightGBM + Fourier,<br>elasticity-adjusted (η = {PRICE_ELASTICITY})</small></th>
                            <th>Difference<small>ML − static</small></th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="metric-name">Total bookings</td>
                            <td class="static-cell">{bt_static['total_bookings']:,}</td>
                            <td class="ai-cell">{bt_ml['total_bookings']:,}</td>
                            {bt_d_bookings}
                        </tr>
                        <tr>
                            <td class="metric-name">Total revenue</td>
                            <td class="static-cell">€{bt_static['total_revenue']:,.2f}</td>
                            <td class="ai-cell">€{bt_ml['total_revenue']:,.2f}</td>
                            {bt_d_revenue}
                        </tr>
                        <tr>
                            <td class="metric-name">Variable operating costs<small style="color:#5b6b78;font-weight:400;">€{VARIABLE_COST_PER_ROOM_NIGHT:.0f}/room-night × room-nights sold</small></td>
                            <td class="static-cell">€{bt_static['total_variable_cost']:,.2f}<br><span class="lift-flat" style="font-size:0.82rem;">{bt_static['room_nights']:,.0f} room-nights</span></td>
                            <td class="ai-cell">€{bt_ml['total_variable_cost']:,.2f}<br><span class="lift-flat" style="font-size:0.82rem;">{bt_ml['room_nights']:,.0f} room-nights</span></td>
                            {bt_d_costs}
                        </tr>
                        <tr>
                            <td class="metric-name">Gross margin</td>
                            <td class="static-cell">{bt_static['gross_margin']*100:.1f}%</td>
                            <td class="ai-cell">{bt_ml['gross_margin']*100:.1f}%</td>
                            {bt_d_margin}
                        </tr>
                        <tr>
                            <td class="metric-name">Avg price&nbsp;/&nbsp;night</td>
                            <td class="static-cell">€{bt_static['avg_price']:,.2f}</td>
                            <td class="ai-cell">€{bt_ml['avg_price']:,.2f}</td>
                            {bt_d_price}
                        </tr>
                        <tr>
                            <td class="metric-name">Avg occupancy</td>
                            <td class="static-cell">{bt_static['avg_occupancy']*100:.1f}%</td>
                            <td class="ai-cell">{bt_ml['avg_occupancy']*100:.1f}%</td>
                            {bt_d_occupancy}
                        </tr>
                        <tr class="hm-row-profit">
                            <td class="metric-name">Net profit<small style="color:#5b6b78;font-weight:400;">= Revenue − variable costs − fixed overhead<br>(fixed overhead cancels)</small></td>
                            <td class="static-cell"></td>
                            <td class="ai-cell"></td>
                            {bt_d_netprofit}
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        bt_profit_lift = bt_ml["gross_profit"] - bt_static["gross_profit"]
        bt_profit_pct = (
            bt_profit_lift / bt_static["gross_profit"] * 100
            if bt_static["gross_profit"] > 0 else 0
        )
        st.caption(
            f"**Scope.** Same {len(blind_df):,} demo bookings as the Prophet "
            f"dashboard's blind-test table, re-priced here using the **ML "
            f"model's** per-night predictions. On this slice the ML model "
            f"lifts net profit by **€{bt_profit_lift:,.2f} "
            f"({bt_profit_pct:+.1f}%)**. (Prophet's lift on the same data "
            f"was around +€73K / +8.4% on the full demo window — the two "
            f"models give different signals because the ML tree splits "
            f"price more aggressively than Prophet's smooth seasonality.)"
        )


# -----------------------------------------------------------------------------
# Section 4 — 90-day ML forecast vs same period last year
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("90-day ML forecast vs same period last year")

forecast = pd.read_csv(ML_FORECAST_CSV, parse_dates=["ds"])
last_train = history["ds"].max()
forecast_only = forecast  # already future-only

yoy_window_start = forecast_only["ds"].min() - pd.DateOffset(years=1)
yoy_window_end = forecast_only["ds"].max() - pd.DateOffset(years=1)
yoy_data = history[
    (history["ds"] >= yoy_window_start) & (history["ds"] <= yoy_window_end)
].copy()
yoy_data["ds_shifted"] = yoy_data["ds"] + pd.DateOffset(years=1)

all_dates = (
    pd.concat([history["ds"], forecast_only["ds"]])
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)
static_rulebook = all_dates.dt.month.map(
    {m: historical_monthly_avg(m) for m in range(1, 13)}
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(forecast_only["ds"]) + list(forecast_only["ds"][::-1]),
    y=list(forecast_only["yhat_upper"]) + list(forecast_only["yhat_lower"][::-1]),
    fill="toself",
    fillcolor="rgba(142,68,173,0.16)",
    line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    name="~80% interval",
    showlegend=True,
))
fig.add_trace(go.Scatter(
    x=all_dates, y=static_rulebook,
    mode="lines",
    line=dict(color="#7f8c8d", width=1.6, dash="dash"),
    name="Static rulebook (without AI)",
))
fig.add_trace(go.Scatter(
    x=yoy_data["ds_shifted"], y=yoy_data["y"],
    mode="lines",
    line=dict(color="#27ae60", width=1.6, dash="dash"),
    name="Same period last year (2016 actuals)",
))
fig.add_trace(go.Scatter(
    x=history["ds"], y=history["y"],
    mode="lines",
    line=dict(color="#3c91b3", width=2),
    name="Historical (actual)",
))
fig.add_trace(go.Scatter(
    x=forecast_only["ds"], y=forecast_only["yhat"],
    mode="lines",
    line=dict(color="#8e44ad", width=2.5),
    name="ML forecast (with ML)",
))
fig.update_layout(
    height=460,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.92)", bordercolor="#e5edf2",
                borderwidth=1, font=dict(size=12)),
    xaxis=dict(title="Date", gridcolor="#e9eef3"),
    yaxis=dict(title="Price (EUR)", gridcolor="#e9eef3"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <div class="hm-chartKey">
        <b>What you're seeing</b>
        <ul>
            <li><span class="hm-swatch" style="background:#3c91b3;"></span>
                <b>Historical (actual)</b> — 793 days of real ADR (Jul 2015 – Aug 2017).</li>
            <li><span class="hm-swatch dashed" style="color:#7f8c8d;"></span>
                <b>Static rulebook</b> — flat monthly average; the same baseline as the Prophet page.</li>
            <li><span class="hm-swatch" style="background:#8e44ad;"></span>
                <b>ML forecast</b> — LightGBM predictions for the next 90 days.</li>
            <li><span class="hm-swatch band" style="background:#8e44ad;"></span>
                <b>~80% interval</b> — ±1.28·RMSE band; trees don't ship native intervals so this is an approximation.</li>
            <li><span class="hm-swatch dashed" style="color:#27ae60;"></span>
                <b>Same period last year (2016)</b> — actuals shifted forward 12 months for context.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Section 5 — KPIs on the full demo window, ML-priced
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("Key performance indicators — without AI vs with ML")

if bookings_df.empty:
    st.info(
        "No bookings yet. Use the **Book Your Stay** page to add some, "
        "then come back and click **↻ Refresh Data** in the sidebar."
    )
else:
    static_kpis = compute_static_baseline_kpis(bookings_df)
    ml_kpis = compute_elasticity_adjusted_kpis_ml(bookings_df)

    d_bookings  = diff_cell(static_kpis["total_bookings"],       ml_kpis["total_bookings"],       "{:,.0f}")
    d_revenue   = diff_cell(static_kpis["total_revenue"],        ml_kpis["total_revenue"],        "€{:,.2f}")
    d_costs     = diff_cell(static_kpis["total_variable_cost"],  ml_kpis["total_variable_cost"],  "€{:,.2f}", mode="cost")
    d_margin    = diff_cell(static_kpis["gross_margin"]*100,     ml_kpis["gross_margin"]*100,     "{:.1f}%")
    d_price     = diff_cell(static_kpis["avg_price"],            ml_kpis["avg_price"],            "€{:,.2f}", mode="orange")
    d_occupancy = diff_cell(static_kpis["avg_occupancy"]*100,    ml_kpis["avg_occupancy"]*100,    "{:.1f}%")
    d_netprofit = diff_cell(static_kpis["gross_profit"],         ml_kpis["gross_profit"],         "€{:,.2f}", mode="headline")

    st.markdown(
        f"""
        <div class="hm-compare">
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Without AI<small>static seasonal rulebook</small></th>
                        <th>With ML<small>LightGBM + Fourier,<br>elasticity-adjusted (η = {PRICE_ELASTICITY})</small></th>
                        <th>Difference<small>ML − static</small></th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="metric-name">Total bookings</td>
                        <td class="static-cell">{static_kpis['total_bookings']:,}</td>
                        <td class="ai-cell">{ml_kpis['total_bookings']:,}</td>
                        {d_bookings}
                    </tr>
                    <tr>
                        <td class="metric-name">Total revenue</td>
                        <td class="static-cell">€{static_kpis['total_revenue']:,.2f}</td>
                        <td class="ai-cell">€{ml_kpis['total_revenue']:,.2f}</td>
                        {d_revenue}
                    </tr>
                    <tr>
                        <td class="metric-name">Variable operating costs<small style="color:#5b6b78;font-weight:400;">€{VARIABLE_COST_PER_ROOM_NIGHT:.0f}/room-night × room-nights sold</small></td>
                        <td class="static-cell">€{static_kpis['total_variable_cost']:,.2f}<br><span class="lift-flat" style="font-size:0.82rem;">{static_kpis['room_nights']:,.0f} room-nights</span></td>
                        <td class="ai-cell">€{ml_kpis['total_variable_cost']:,.2f}<br><span class="lift-flat" style="font-size:0.82rem;">{ml_kpis['room_nights']:,.0f} room-nights</span></td>
                        {d_costs}
                    </tr>
                    <tr>
                        <td class="metric-name">Gross margin</td>
                        <td class="static-cell">{static_kpis['gross_margin']*100:.1f}%</td>
                        <td class="ai-cell">{ml_kpis['gross_margin']*100:.1f}%</td>
                        {d_margin}
                    </tr>
                    <tr>
                        <td class="metric-name">Avg price&nbsp;/&nbsp;night</td>
                        <td class="static-cell">€{static_kpis['avg_price']:,.2f}</td>
                        <td class="ai-cell">€{ml_kpis['avg_price']:,.2f}</td>
                        {d_price}
                    </tr>
                    <tr>
                        <td class="metric-name">Avg occupancy</td>
                        <td class="static-cell">{static_kpis['avg_occupancy']*100:.1f}%</td>
                        <td class="ai-cell">{ml_kpis['avg_occupancy']*100:.1f}%</td>
                        {d_occupancy}
                    </tr>
                    <tr class="hm-row-profit">
                        <td class="metric-name">Net profit</td>
                        <td class="static-cell"></td>
                        <td class="ai-cell"></td>
                        {d_netprofit}
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Same full demo window (152 days, Jul–Nov 2017), same 2,494 "
        "bookings, **but re-priced with the LightGBM model** "
        "instead of Prophet. The ML model predicts higher absolute "
        "prices in the September-November forecast window — which "
        "means more guests walk away under the elasticity model, "
        "so the net-profit lift over static is smaller than Prophet's."
    )


# -----------------------------------------------------------------------------
# Section 6 — Model-vs-model comparison (Prophet vs ML on the same blind test)
# -----------------------------------------------------------------------------
st.markdown('<div class="hm-section"></div>', unsafe_allow_html=True)
st.subheader("Prophet vs ML — side by side on the same blind test")

prophet_bt = pd.read_csv(PROPHET_BLIND_TEST_CSV, parse_dates=["ds"])

# Compute Prophet metrics
p_err = prophet_bt["y_actual"] - prophet_bt["yhat"]
p_mae = float(p_err.abs().mean())
p_rmse = float((p_err ** 2).mean() ** 0.5)
p_mape = float((p_err.abs() / prophet_bt["y_actual"].abs() * 100).mean())

# Compute ML metrics
m_rmse = float((ml_err ** 2).mean() ** 0.5)

mc1, mc2 = st.columns(2)
with mc1:
    st.markdown("### 🟠 Prophet (Meta)")
    st.metric("MAE", f"€{p_mae:.2f}")
    st.metric("RMSE", f"€{p_rmse:.2f}")
    st.metric("MAPE", f"{p_mape:.2f}%", delta="EXCELLENT (<10%)", delta_color="off")
    st.caption(
        "Time-series-native: decomposes the signal into trend + yearly + "
        "weekly seasonality. Smooth predictions, native uncertainty "
        "intervals, but few features available for extensions."
    )
with mc2:
    st.markdown("### 🟣 LightGBM + Fourier (Microsoft)")
    st.metric("MAE", f"€{ml_mae:.2f}")
    st.metric("RMSE", f"€{m_rmse:.2f}")
    rating = "EXCELLENT (<10%)" if ml_mape < 10 else "GOOD (10–20%)"
    st.metric("MAPE", f"{ml_mape:.2f}%", delta=rating, delta_color="off")
    st.caption(
        "Modern gradient-boosted decision trees with **18 engineered "
        "features**: 8 plain date features (year, month, day_of_week, "
        "day_of_year, week_of_year, is_weekend, days_since_start, "
        "day_of_month) + 6 yearly Fourier harmonics (sin/cos × 3) "
        "+ 2 weekly Fourier harmonics (sin/cos × 2) + same-day-last-year "
        "lag (y_lag_365). LightGBM is the gold-standard tabular ML "
        "library (Microsoft, top of most Kaggle leaderboards). Scales "
        "naturally with new features (weather, events, competitor "
        "rates) — that's its operational advantage over Prophet."
    )

st.caption(
    "**Which is better?** On *this dataset* — short (793 days), "
    "univariate (no covariates), strong yearly + weekly seasonality "
    "— Prophet keeps a small edge because its Bayesian smooth-spline "
    "decomposition fits the August peak more tightly than LightGBM's "
    "discrete splits. The gap closed from **~5 pp** (the original GBR "
    "MAPE of 10.27% vs Prophet's 5.75%) **down to ~2 pp** once we "
    "added Fourier features and switched the booster to LightGBM. "
    "**In production** with weather, events, occupancy and competitor "
    "rates available as covariates, LightGBM scales naturally — Prophet "
    "would need extra regressors and re-tuning. For an MVP with date + "
    "price only, Prophet is the right baseline; the LightGBM page "
    "proves the architecture is model-agnostic and ready to swap as "
    "real-world features arrive."
)
