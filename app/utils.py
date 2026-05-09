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

# Surface a build marker so Streamlit Cloud's deploy log shows which
# revision is live; also forces a full container restart on bump
# (Cloud sometimes hot-reloads page files without re-importing
# sibling modules like this one, leaving stale symbol tables behind).
APP_BUILD = "v1.10.1-reorder-blind-test-first"

# The app folder is one level below the project root, so go up once.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "price_model.pkl"
DB_PATH = PROJECT_ROOT / "data" / "bookings.db"
HISTORY_CSV = PROJECT_ROOT / "data" / "daily_prices.csv"

# Last day the model has actually seen during training. Anything after
# this is extrapolation. Hard-coded because the dataset is fixed.
LAST_TRAIN_DATE = pd.Timestamp("2017-08-31")
LAST_FORECAST_DATE = pd.Timestamp("2017-11-29")  # 90 days past training

# Fixed demo date inside the model's forecast window.
# In production this would be datetime.today().
DEMO_DATE = date(2017, 9, 15)


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

# Price elasticity of demand. Empirical literature on mid-range / 4-star
# hotels lands roughly in [-0.4, -0.8]; we use -0.7, which is the more
# conservative end (more guests refuse higher prices). Used by the
# manager dashboard's "with AI — realistic" column.
PRICE_ELASTICITY = -0.7

# Variable operating cost per occupied room-night, EUR. Industry breakdown
# for a 4-star hotel: housekeeping ~€15, supplies ~€5, energy ~€8,
# laundry ~€7, breakfast ~€8 = €43. These are costs that scale with
# occupancy — when AI prices push some guests away, we incur fewer of
# them, which partly offsets the lost revenue.
VARIABLE_COST_PER_ROOM_NIGHT = 43.0


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

# Demo seed config — exposed at module level so seed_demo_bookings()
# can be reused both by init_db() (auto-seed on empty DB) and by
# scripts/03_seed_demo_bookings.py.
DEMO_SEED_WINDOW_START = date(2017, 7, 1)
DEMO_SEED_WINDOW_END = date(2017, 11, 29)
DEMO_SEED_N_BOOKINGS = 2500

_DEMO_GUEST_POOL = [
    ("Maria Garcia", "maria.garcia@example.com"),
    ("Liam O'Connor", "liam.oconnor@example.com"),
    ("Sofia Rossi", "sofia.rossi@example.it"),
    ("Hans Muller", "hans.muller@example.de"),
    ("Elena Petrov", "elena.petrov@example.com"),
    ("Carlos Mendes", "carlos.mendes@example.pt"),
    ("Yuki Tanaka", "yuki.tanaka@example.jp"),
    ("Anna Kowalski", "anna.k@example.pl"),
    ("Pierre Dubois", "p.dubois@example.fr"),
    ("Aisha Khan", "aisha.khan@example.com"),
    ("Olivia Smith", "olivia.smith@example.co.uk"),
    ("Marco Bianchi", "marco.b@example.it"),
    ("Ingrid Larsen", "ingrid.l@example.no"),
    ("Tom Hansen", "tom.hansen@example.dk"),
    ("Beatriz Silva", "beatriz.s@example.br"),
    ("Lucas Schmidt", "lucas.s@example.de"),
    ("Nora Lindqvist", "nora.l@example.se"),
    ("Diego Fernandez", "diego.f@example.es"),
    ("Hannah Becker", "h.becker@example.de"),
    ("Rafael Costa", "rafael.c@example.pt"),
    ("Chloe Martin", "chloe.m@example.fr"),
    ("Stefan Novak", "s.novak@example.cz"),
    ("Sara Berg", "sara.berg@example.se"),
    ("Kenji Saito", "k.saito@example.jp"),
    ("Layla Ahmed", "layla.a@example.com"),
    ("Sebastian Klein", "s.klein@example.de"),
    ("Camila Vargas", "c.vargas@example.es"),
    ("Mateo Romano", "m.romano@example.it"),
    ("Freya Eriksen", "freya.e@example.dk"),
    ("Jonas Weber", "jonas.w@example.ch"),
]
_DEMO_MONTH_WEIGHTS = {7: 5, 8: 6, 9: 4, 10: 2, 11: 1}
_DEMO_NIGHTS = ([1, 2, 3, 4, 5], [1, 3, 4, 3, 2])
_DEMO_ROOM_WEIGHTS = [6, 3, 1]


def _create_bookings_table(conn: sqlite3.Connection) -> None:
    """Idempotent: create the bookings table if it doesn't exist."""
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


def seed_demo_bookings(verbose: bool = False) -> int:
    """Wipe and re-populate the bookings table with the demo dataset.

    Used in two places:
    - by init_db(), automatically, when the deployed app boots and finds
      an empty table (Streamlit Cloud has ephemeral storage and our
      bookings.db is gitignored, so a fresh container has nothing).
    - by scripts/03_seed_demo_bookings.py for local development.

    Returns the number of rows inserted. Deterministic (random.seed=42)
    so the dashboard looks the same across cold starts.
    """
    import random  # local imports to keep top of file lean
    import pickle
    from datetime import datetime, timedelta as _td

    if not MODEL_PATH.exists():
        if verbose:
            print(f"  [skip] no model at {MODEL_PATH} — run scripts/02_train_model.py first")
        return 0

    rng = random.Random(42)

    if verbose:
        print(f"Demo window: {DEMO_SEED_WINDOW_START} -> {DEMO_SEED_WINDOW_END}")
        print(f"Loading model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    all_dates = pd.date_range(DEMO_SEED_WINDOW_START, DEMO_SEED_WINDOW_END, freq="D")
    pred = model.predict(pd.DataFrame({"ds": all_dates}))
    price_lookup = dict(zip(pred["ds"], pred["yhat"]))
    if verbose:
        print(f"  cached predictions for {len(price_lookup)} days")

    # Weighted check-in pool: more demand in summer.
    weighted_dates = []
    for d in all_dates:
        weighted_dates.extend([d] * _DEMO_MONTH_WEIGHTS.get(d.month, 1))

    rooms = list(ROOM_TYPES.keys())
    nights_choices, nights_weights = _DEMO_NIGHTS

    rows = []
    for _ in range(DEMO_SEED_N_BOOKINGS):
        guest = rng.choice(_DEMO_GUEST_POOL)
        check_in_ts = rng.choice(weighted_dates)
        nights = rng.choices(nights_choices, weights=nights_weights, k=1)[0]
        check_out_ts = check_in_ts + pd.Timedelta(days=nights)
        if check_out_ts.date() > DEMO_SEED_WINDOW_END:
            check_out_ts = pd.Timestamp(DEMO_SEED_WINDOW_END)
            nights = (check_out_ts - check_in_ts).days
            if nights < 1:
                continue

        room = rng.choices(rooms, weights=_DEMO_ROOM_WEIGHTS, k=1)[0]
        multiplier = ROOM_TYPES[room]

        stay_nights = pd.date_range(
            check_in_ts, check_out_ts - pd.Timedelta(days=1), freq="D"
        )
        total = float(sum(
            apply_occupancy_adjustment(price_lookup[d] * multiplier)
            for d in stay_nights
        ))

        # Backdate created_at so timestamps look organic.
        days_lead = rng.randint(7, 90)
        created = check_in_ts - pd.Timedelta(days=days_lead, hours=rng.randint(0, 23))

        rows.append((
            guest[0], guest[1],
            check_in_ts.date().isoformat(),
            check_out_ts.date().isoformat(),
            room,
            round(total, 2),
            created.to_pydatetime().isoformat(timespec="seconds"),
        ))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        _create_bookings_table(conn)
        conn.execute("DELETE FROM bookings")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='bookings'")
        conn.executemany(
            "INSERT INTO bookings "
            "(guest_name, email, check_in, check_out, room_type, total_price, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    if verbose:
        print(f"Inserted {len(rows):,} bookings into {DB_PATH}")
    return len(rows)


def init_db() -> None:
    """Make sure the bookings table exists. If it's empty AND a trained
    model is available, auto-seed the demo dataset.

    This is what makes the Streamlit Cloud deployment work end-to-end:
    bookings.db is gitignored, so a fresh container starts with no
    database file. On first boot, init_db() detects the empty table
    and populates ~2,500 demo bookings (~5 seconds one-time cost).
    Subsequent calls do nothing (cheap row-count check)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        _create_bookings_table(conn)
        n = conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0]
    if n == 0 and MODEL_PATH.exists():
        seed_demo_bookings(verbose=False)


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


def _empty_kpi_dict() -> dict:
    return {
        "total_bookings": 0,
        "room_nights": 0,
        "total_revenue": 0.0,
        "total_variable_cost": 0.0,
        "gross_profit": 0.0,
        "gross_margin": 0.0,
        "avg_price": 0.0,
        "avg_occupancy": 0.0,
    }


def _kpi_dict(
    total_bookings: float,
    room_nights: float,
    total_revenue: float,
    window_days: int,
) -> dict:
    """Build the standard KPI dict from raw aggregates. All three KPI
    helpers (static / naive / elasticity-adjusted) flow through here so
    the cost-and-profit math stays identical."""
    total_variable_cost = room_nights * VARIABLE_COST_PER_ROOM_NIGHT
    gross_profit = total_revenue - total_variable_cost
    gross_margin = gross_profit / total_revenue if total_revenue > 0 else 0.0
    avg_price = total_revenue / room_nights if room_nights > 0 else 0.0
    avg_occupancy = room_nights / (TOTAL_ROOMS * window_days) if window_days > 0 else 0.0

    return {
        "total_bookings": int(round(total_bookings)),
        "room_nights": room_nights,
        "total_revenue": total_revenue,
        "total_variable_cost": total_variable_cost,
        "gross_profit": gross_profit,
        "gross_margin": gross_margin,
        "avg_price": avg_price,
        "avg_occupancy": avg_occupancy,
    }


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute the headline KPIs shown on the manager dashboard.

    - total_bookings      : count of confirmed booking rows
    - room_nights         : total stay-nights sold
    - total_revenue       : sum of total_price across all bookings (EUR)
    - total_variable_cost : room_nights × VARIABLE_COST_PER_ROOM_NIGHT
    - gross_profit        : revenue minus variable cost
    - gross_margin        : gross_profit / total_revenue (0..1)
    - avg_price           : average price per room-night sold (EUR)
    - avg_occupancy       : room-nights sold / (TOTAL_ROOMS × days_in_window)

    days_in_window covers the period from the earliest check-in to the
    latest check-out so the metric reflects "how full have we been over
    the booked horizon".
    """
    if df.empty:
        return _empty_kpi_dict()

    total_bookings = int(len(df))
    total_revenue = float(df["total_price"].sum())
    room_nights = int(df["nights"].sum())
    window_days = max((df["check_out"].max() - df["check_in"].min()).days, 1)

    return _kpi_dict(total_bookings, room_nights, total_revenue, window_days)


@st.cache_data(show_spinner=False)
def historical_monthly_avg(month: int) -> float:
    """Average historical ADR across every day of the given month (1-12)
    in the training data. Used as the "old static pricing" baseline on
    the manager dashboard — more honest than a single flat rate because
    a real seasonal rulebook varies by month."""
    df = pd.read_csv(HISTORY_CSV, parse_dates=["ds"])
    return float(df[df["ds"].dt.month == month]["y"].mean())


def compute_elasticity_adjusted_kpis(
    bookings_df: pd.DataFrame,
    elasticity: float = PRICE_ELASTICITY,
) -> dict:
    """Re-state the AI-priced KPIs after applying price elasticity of
    demand: when AI charges more than the static rulebook, some guests
    walk away. Returns the same KPI shape as compute_kpis().

    Per-booking model:
        static_price_i  = sum(historical monthly avg × room mult × nights)
        ai_price_i      = bookings.total_price (Prophet × room mult × nights)
        pct_change_i    = (ai_price_i - static_price_i) / static_price_i
        retention_i     = max(0, 1 + elasticity × pct_change_i)
        kept_revenue_i  = retention_i × ai_price_i

    Aggregating retention × revenue across bookings gives the
    realistic AI revenue with demand response. Bookings, room-nights,
    and occupancy are scaled by the same retention factor.
    """
    if bookings_df.empty:
        return _empty_kpi_dict()

    monthly = {m: historical_monthly_avg(m) for m in range(1, 13)}

    total_revenue = 0.0
    expected_bookings = 0.0
    expected_room_nights = 0.0

    for _, b in bookings_df.iterrows():
        multiplier = ROOM_TYPES[b["room_type"]]
        nights = pd.date_range(
            b["check_in"], b["check_out"] - pd.Timedelta(days=1), freq="D"
        )
        n_nights = len(nights)
        if n_nights == 0:
            continue

        static_price = sum(monthly[d.month] * multiplier for d in nights)
        ai_price = float(b["total_price"])
        if static_price <= 0:
            continue

        pct_change = (ai_price - static_price) / static_price
        # Cap retention at [0, 1] — AI prices are higher overall, so we
        # don't model a hypothetical "expansion" effect from cheap days.
        retention = min(1.0, max(0.0, 1.0 + elasticity * pct_change))

        total_revenue += retention * ai_price
        expected_bookings += retention
        expected_room_nights += retention * n_nights

    window_days = max(
        (bookings_df["check_out"].max() - bookings_df["check_in"].min()).days, 1
    )

    return _kpi_dict(
        expected_bookings, expected_room_nights, total_revenue, window_days
    )


def compute_static_baseline_kpis(bookings_df: pd.DataFrame) -> dict:
    """Re-price every booking under the OLD static seasonal rulebook
    and return the same KPI shape as compute_kpis().

    The rulebook charges one flat rate per calendar month — the
    historical average ADR for that month. Demand is held constant
    (same 2,494 reservations), so this is a counterfactual: 'what
    would these same bookings have brought in if HotelMar had stayed
    on the old rulebook instead of switching to AI?'
    """
    if bookings_df.empty:
        return _empty_kpi_dict()

    monthly = {m: historical_monthly_avg(m) for m in range(1, 13)}

    # Expand each booking to one row per stay-night and price it at the
    # monthly average × room multiplier.
    nightly_prices = []
    for _, b in bookings_df.iterrows():
        multiplier = ROOM_TYPES[b["room_type"]]
        nights = pd.date_range(
            b["check_in"], b["check_out"] - pd.Timedelta(days=1), freq="D"
        )
        for d in nights:
            nightly_prices.append(monthly[d.month] * multiplier)

    total_revenue = float(sum(nightly_prices))
    total_bookings = int(len(bookings_df))
    room_nights = len(nightly_prices)
    window_days = max(
        (bookings_df["check_out"].max() - bookings_df["check_in"].min()).days, 1
    )

    return _kpi_dict(total_bookings, room_nights, total_revenue, window_days)


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
