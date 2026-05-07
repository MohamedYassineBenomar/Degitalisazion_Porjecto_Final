"""
Guest-facing booking page.

Flow:
  1. Guest picks dates, room type and party size.
  2. "Check Price" runs the AI model on each night and shows a quote.
  3. Guest enters name + email and clicks "Confirm Booking".
  4. We save to SQLite and show a booking reference.
"""

from datetime import date, timedelta
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `from utils import ...` work when Streamlit launches this file
# directly from the pages/ folder.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import (  # noqa: E402  (import after sys.path tweak)
    BASELINE_OCCUPANCY,
    ROOM_TYPES,
    apply_occupancy_adjustment,
    booking_reference,
    predict_prices,
    save_booking,
)

# ---------------------------------------------------------------------------
# Page config + reuse the same brand styling as the landing page.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Book your stay — HotelMar", page_icon="🌊", layout="wide")

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
            padding: 40px 40px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(10, 61, 98, 0.15);
        }
        .hm-hero h1 { color: white; margin: 0 0 6px 0; font-size: 2.2rem; }
        .hm-hero p  { color: rgba(255,255,255,0.92); margin: 0; }
        .hm-quote {
            background: white;
            padding: 28px;
            border-radius: 14px;
            box-shadow: 0 4px 16px rgba(10, 61, 98, 0.12);
            border-left: 5px solid var(--hm-sea);
        }
        .hm-quote-total {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--hm-navy);
            margin: 8px 0 4px 0;
        }
        .hm-pernight {
            color: #5b6b78;
            font-size: 0.95rem;
            margin: 0;
        }
        .hm-success {
            background: #e8f5e9;
            border-left: 5px solid #2e7d32;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .hm-success h3 { color: #2e7d32; margin-top: 0; }
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
        <h1>Book your stay</h1>
        <p>4-star sea-view hotel in Sitges · AI-recommended pricing · no booking fees</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Booking inputs
# ---------------------------------------------------------------------------
today = date.today()

c1, c2, c3, c4 = st.columns([1, 1, 1.2, 0.8])
with c1:
    check_in = st.date_input(
        "Check-in",
        value=today + timedelta(days=14),
        min_value=today,
        format="DD/MM/YYYY",
    )
with c2:
    check_out = st.date_input(
        "Check-out",
        value=today + timedelta(days=17),
        min_value=today + timedelta(days=1),
        format="DD/MM/YYYY",
    )
with c3:
    room_type = st.selectbox(
        "Room type",
        list(ROOM_TYPES.keys()),
        index=0,
        help="Standard Sea View · Superior (+30%) · Suite (+80%)",
    )
with c4:
    guests = st.number_input("Guests", min_value=1, max_value=4, value=2, step=1)

# ---------------------------------------------------------------------------
# Check Price button
# ---------------------------------------------------------------------------
check = st.button("🔍 Check Price", type="primary", use_container_width=False)

if check:
    if check_out <= check_in:
        st.error("Check-out must be after check-in.")
    else:
        with st.spinner("Asking the AI for tonight's price..."):
            prices = predict_prices(check_in, check_out)

        # Apply room-type multiplier and the (currently flat) occupancy hook.
        multiplier = ROOM_TYPES[room_type]
        prices = prices.copy()
        prices["price"] = prices["yhat"].apply(
            lambda p: apply_occupancy_adjustment(p * multiplier)
        )

        total = float(prices["price"].sum())

        # Stash the quote in session_state so the "Confirm Booking" form
        # below can read it without recomputing.
        st.session_state.quote = {
            "check_in": check_in,
            "check_out": check_out,
            "room_type": room_type,
            "guests": int(guests),
            "nights": int(len(prices)),
            "breakdown": prices,
            "total": total,
            "multiplier": multiplier,
        }

# ---------------------------------------------------------------------------
# Quote display + confirm form (shown only once a price has been computed)
# ---------------------------------------------------------------------------
if "quote" in st.session_state:
    q = st.session_state.quote

    st.markdown("### Your quote")

    left, right = st.columns([2, 1])

    with left:
        breakdown = q["breakdown"].copy()
        breakdown["Date"] = breakdown["ds"].dt.strftime("%a %d %b %Y")
        breakdown["Price (EUR)"] = breakdown["price"].round(2)
        st.dataframe(
            breakdown[["Date", "Price (EUR)"]],
            hide_index=True,
            use_container_width=True,
        )
        avg = q["total"] / q["nights"]
        st.caption(
            f"Average: €{avg:.2f}/night · "
            f"Room multiplier: ×{q['multiplier']:.2f} ({q['room_type']}) · "
            f"Occupancy assumption: {BASELINE_OCCUPANCY:.0%}"
        )

    with right:
        st.markdown(
            f"""
            <div class="hm-quote">
                <div style="color:#5b6b78; font-size:0.9rem;">Total for {q['nights']} night(s)</div>
                <div class="hm-quote-total">€{q['total']:.2f}</div>
                <p class="hm-pernight">€{q['total']/q['nights']:.2f} per night · {q['guests']} guest(s)</p>
                <hr style="border:none;border-top:1px solid #e5edf2;margin:14px 0;">
                <div style="font-size:0.9rem;color:#5b6b78;">
                    <b>{q['room_type']}</b><br>
                    {q['check_in'].strftime('%d %b %Y')} → {q['check_out'].strftime('%d %b %Y')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Confirm form -----------------------------------------------------
    st.markdown("### Confirm your booking")
    with st.form("confirm_booking"):
        cf1, cf2 = st.columns(2)
        with cf1:
            guest_name = st.text_input("Full name", placeholder="Maria García")
        with cf2:
            email = st.text_input("Email", placeholder="maria@example.com")

        st.caption(
            "By confirming, you agree to the hotel's cancellation terms. "
            "This is an MVP demo — no real charge will be made."
        )

        submitted = st.form_submit_button(
            "✓ Confirm Booking", type="primary", use_container_width=False
        )

        if submitted:
            if not guest_name.strip() or not email.strip():
                st.error("Please enter both your name and email.")
            elif "@" not in email:
                st.error("That doesn't look like a valid email.")
            else:
                booking_id = save_booking(
                    guest_name=guest_name.strip(),
                    email=email.strip(),
                    check_in=q["check_in"],
                    check_out=q["check_out"],
                    room_type=q["room_type"],
                    total_price=q["total"],
                )
                ref = booking_reference(booking_id)

                st.markdown(
                    f"""
                    <div class="hm-success">
                        <h3>Booking confirmed ✓</h3>
                        <p>Thank you, <b>{guest_name}</b>. We've sent a confirmation
                        to <b>{email}</b>.</p>
                        <p>Your booking reference is <b style="font-size:1.2rem;">{ref}</b> —
                        keep it handy for check-in.</p>
                        <p style="margin-bottom:0;">
                            {q['nights']} night(s) · {q['room_type']} ·
                            <b>€{q['total']:.2f}</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.balloons()
                # Clear the quote so the page is ready for a new search.
                del st.session_state.quote
