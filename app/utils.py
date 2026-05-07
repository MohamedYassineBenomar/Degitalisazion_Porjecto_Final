"""
Shared helpers for the HotelMar Streamlit app.

Anything that touches the model or the database lives here so the
page files stay focused on UI.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import pickle
import sqlite3

import pandas as pd
import streamlit as st

# --- Paths -----------------------------------------------------------------

# The app folder is one level below the project root, so go up once.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "price_model.pkl"
DB_PATH = PROJECT_ROOT / "data" / "bookings.db"

# Last day the model has actually seen during training. Anything after
# this is extrapolation. Hard-coded because the dataset is fixed.
LAST_TRAIN_DATE = pd.Timestamp("2017-08-31")


# --- Pricing config --------------------------------------------------------

# Room types and how each one prices relative to a "Standard Sea View" room.
# A Suite is 80% more expensive than a Standard, etc.
ROOM_TYPES: dict[str, float] = {
    "Standard Sea View": 1.0,
    "Superior": 1.3,
    "Suite": 1.8,
}

# Placeholder hotel-wide occupancy. In a real system this would come from
# the PMS (property management system) live. For Step 3 we treat 65% as
# "neutral" and don't shift the price. Step 4 will make it dynamic.
BASELINE_OCCUPANCY = 0.65

# What HotelMar used to charge per night with the old "static seasonal
# pricing" rulebook — the baseline we compare AI suggestions against.
OLD_STATIC_PRICE = 110.0

# How many rooms the hotel has, used as the denominator for occupancy.
TOTAL_ROOMS = 150


def apply_occupancy_adjustment(price: float, occupancy: float = BASELINE_OCCUPANCY) -> float:
    """Hook point for dynamic occupancy pricing. For now: no change."""
    # Placeholder: when occupancy > baseline we'd push prices up; when
    # under-booked we'd discount. Implemented in Step 4.
    return price


# --- Model -----------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the AI pricing model...")
def load_model():
    """Load the trained Prophet model. Cached for the whole session."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def predict_prices(check_in: date, check_out: date) -> pd.DataFrame:
    """
    Predict the per-night ADR for every night between check_in (inclusive)
    and check_out (exclusive). Returns a DataFrame with columns:
        ds   : the date (Timestamp)
        yhat : the model's predicted price (float, EUR)
    """
    model = load_model()

    # The "stay nights" are check_in, check_in+1, ..., check_out-1.
    nights = pd.date_range(
        start=pd.Timestamp(check_in),
        end=pd.Timestamp(check_out) - pd.Timedelta(days=1),
        freq="D",
    )
    if len(nights) == 0:
        return pd.DataFrame(columns=["ds", "yhat"])

    target = nights[-1]

    if target <= LAST_TRAIN_DATE:
        # Date is inside the model's training range — predict directly
        # on the requested days.
        future = pd.DataFrame({"ds": nights})
    else:
        # Date is in the future — Prophet needs a contiguous frame from
        # training end up to the target date.
        days_ahead = (target - LAST_TRAIN_DATE).days
        future = model.make_future_dataframe(periods=days_ahead, freq="D")

    forecast = model.predict(future)
    # Filter to just the dates we care about.
    out = forecast[forecast["ds"].isin(nights)][["ds", "yhat"]]
    return out.sort_values("ds").reset_index(drop=True)


# --- Database --------------------------------------------------------------

def init_db() -> None:
    """Create the bookings table if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                guest_name  TEXT    NOT NULL,
                email       TEXT    NOT NULL,
                check_in    TEXT    NOT NULL,
                check_out   TEXT    NOT NULL,
                room_type   TEXT    NOT NULL,
                total_price REAL    NOT NULL,
                created_at  TEXT    NOT NULL
            )
            """
        )


def save_booking(
    guest_name: str,
    email: str,
    check_in: date,
    check_out: date,
    room_type: str,
    total_price: float,
) -> int:
    """Insert a booking row and return its auto-generated id."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO bookings
                (guest_name, email, check_in, check_out, room_type, total_price, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                guest_name,
                email,
                check_in.isoformat(),
                check_out.isoformat(),
                room_type,
                round(float(total_price), 2),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return cur.lastrowid


def booking_reference(booking_id: int) -> str:
    """Format an internal id as a guest-friendly reference (HM-000123)."""
    return f"HM-{booking_id:06d}"


# --- Analytics (used by the manager dashboard) -----------------------------

def get_all_bookings() -> pd.DataFrame:
    """Return all bookings as a DataFrame, with parsed dates and a
    derived `nights` column. Empty DataFrame if the table is empty."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT id, guest_name, email, check_in, check_out, "
            "room_type, total_price, created_at "
            "FROM bookings ORDER BY id DESC",
            conn,
        )
    if df.empty:
        return df
    df["check_in"] = pd.to_datetime(df["check_in"])
    df["check_out"] = pd.to_datetime(df["check_out"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["nights"] = (df["check_out"] - df["check_in"]).dt.days
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute the four headline KPIs shown on the manager dashboard.

    - total_bookings : count of confirmed booking rows
    - total_revenue  : sum of total_price across all bookings (EUR)
    - avg_price      : average price per *room-night* sold (EUR)
    - avg_occupancy  : room-nights sold / (TOTAL_ROOMS x days_in_window)

    days_in_window covers the period from the earliest check-in to the
    latest check-out so the metric reflects "how full have we been over
    the booked horizon".
    """
    if df.empty:
        return {
            "total_bookings": 0,
            "total_revenue": 0.0,
            "avg_price": 0.0,
            "avg_occupancy": 0.0,
        }

    total_bookings = int(len(df))
    total_revenue = float(df["total_price"].sum())
    room_nights = int(df["nights"].sum())
    avg_price = total_revenue / room_nights if room_nights else 0.0

    window_days = max((df["check_out"].max() - df["check_in"].min()).days, 1)
    avg_occupancy = room_nights / (TOTAL_ROOMS * window_days)

    return {
        "total_bookings": total_bookings,
        "total_revenue": total_revenue,
        "avg_price": avg_price,
        "avg_occupancy": avg_occupancy,
    }


def revenue_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """Spread each booking's total revenue evenly across its stay nights
    and return a DataFrame [date, revenue] aggregated by stay-night."""
    if df.empty:
        return pd.DataFrame(columns=["date", "revenue"])

    rows = []
    for _, b in df.iterrows():
        if b["nights"] <= 0:
            continue
        per_night = b["total_price"] / b["nights"]
        nights = pd.date_range(b["check_in"], b["check_out"] - pd.Timedelta(days=1), freq="D")
        for d in nights:
            rows.append({"date": d, "revenue": per_night})

    out = pd.DataFrame(rows)
    return out.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
