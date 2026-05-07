"""
Step 5 — Seed demo bookings for the manager dashboard.

Wipes data/bookings.db and re-populates it with realistic-looking
reservations spread across the demo window (1 Jul 2017 -> 29 Nov 2017),
which is exactly the model's forecast range. Prices on each booking
are drawn from the trained Prophet model, so the totals on the
dashboard match the AI quotes a guest would have seen.

Why so many rows? Each booking row represents one party booking ONE
room for N nights. To reach a realistic 30-45% house-wide occupancy
on a 150-room hotel across a ~152-day window we need on the order of
22,500 capacity room-nights x 35% / ~3 nights per stay ~= 2,500 rows.
The "Recent bookings" panel still shows only the 20 most recent.

Run from the project root:
    python scripts/03_seed_demo_bookings.py
"""

import pickle
import random
import sqlite3
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Make the app/ helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from utils import (  # noqa: E402
    DB_PATH,
    MODEL_PATH,
    ROOM_TYPES,
    TOTAL_ROOMS,
    apply_occupancy_adjustment,
    init_db,
)

random.seed(42)  # deterministic seeds make the dashboard reproducible

# ---------------------------------------------------------------------------
# Demo window — exactly the dates the AI model can confidently price.
# ---------------------------------------------------------------------------
WINDOW_START = date(2017, 7, 1)
WINDOW_END = date(2017, 11, 29)
WINDOW_DAYS = (WINDOW_END - WINDOW_START).days + 1
N_BOOKINGS = 2500  # see module docstring for why

# Seasonal weighting: more demand in summer, less as autumn rolls in.
# Weight per month — higher = more bookings will land in that month.
MONTH_WEIGHTS = {7: 5, 8: 6, 9: 4, 10: 2, 11: 1}

# Stay-length distribution: most guests stay 2-4 nights.
NIGHTS_CHOICES = [1, 2, 3, 4, 5]
NIGHTS_WEIGHTS = [1, 3, 4, 3, 2]

# Room mix: most book Standard, a few Superior, fewer Suites.
ROOM_NAMES = list(ROOM_TYPES.keys())
ROOM_WEIGHTS = [6, 3, 1]

GUEST_POOL = [
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


def main() -> None:
    print(f"Demo window: {WINDOW_START} -> {WINDOW_END}  ({WINDOW_DAYS} days)")
    print(f"Target rows : {N_BOOKINGS}")
    print(f"Hotel       : {TOTAL_ROOMS} rooms")

    # ------------------------------------------------------------------
    # Pre-compute the model's predicted ADR for every day in the demo
    # window, ONCE. Calling model.predict() per booking would be slow.
    # ------------------------------------------------------------------
    print("\nLoading production model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    all_dates = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    pred = model.predict(pd.DataFrame({"ds": all_dates}))
    price_lookup = dict(zip(pred["ds"], pred["yhat"]))
    print(f"  cached predictions for {len(price_lookup)} days")

    # ------------------------------------------------------------------
    # Fresh DB.
    # ------------------------------------------------------------------
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM bookings")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='bookings'")

    # Pre-build the weighted check-in date pool so np.random doesn't
    # over-sample the long tail months.
    weighted_dates = []
    for d in all_dates:
        weighted_dates.extend([d] * MONTH_WEIGHTS.get(d.month, 1))

    rows = []
    for _ in range(N_BOOKINGS):
        guest = random.choice(GUEST_POOL)
        check_in_ts = random.choice(weighted_dates)
        nights = random.choices(NIGHTS_CHOICES, weights=NIGHTS_WEIGHTS, k=1)[0]
        check_out_ts = check_in_ts + pd.Timedelta(days=nights)

        # Cap the stay so it doesn't run past the end of the demo window.
        if check_out_ts.date() > WINDOW_END:
            check_out_ts = pd.Timestamp(WINDOW_END)
            nights = (check_out_ts - check_in_ts).days
            if nights < 1:
                continue

        room = random.choices(ROOM_NAMES, weights=ROOM_WEIGHTS, k=1)[0]
        multiplier = ROOM_TYPES[room]

        # Sum up nightly prices using the cached lookup.
        stay_nights = pd.date_range(check_in_ts, check_out_ts - pd.Timedelta(days=1), freq="D")
        total = float(sum(
            apply_occupancy_adjustment(price_lookup[d] * multiplier)
            for d in stay_nights
        ))

        # Backdate created_at so the timeline looks like bookings flowed
        # in over the months leading up to the stay.
        days_lead = random.randint(7, 90)
        created = check_in_ts - pd.Timedelta(days=days_lead, hours=random.randint(0, 23))

        rows.append((
            guest[0], guest[1],
            check_in_ts.date().isoformat(),
            check_out_ts.date().isoformat(),
            room,
            round(total, 2),
            created.to_pydatetime().isoformat(timespec="seconds"),
        ))

    # ------------------------------------------------------------------
    # Bulk insert + report.
    # ------------------------------------------------------------------
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT INTO bookings "
            "(guest_name, email, check_in, check_out, room_type, total_price, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    print(f"\nInserted {len(rows):,} bookings into {DB_PATH}")

    # Quick KPI-style readout.
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT check_in, check_out, total_price FROM bookings", conn,
            parse_dates=["check_in", "check_out"],
        )

    df["nights"] = (df["check_out"] - df["check_in"]).dt.days
    room_nights = int(df["nights"].sum())
    capacity = TOTAL_ROOMS * WINDOW_DAYS
    occupancy = room_nights / capacity

    print("\n=== Seed summary ===")
    print(f"  total bookings  : {len(df):,}")
    print(f"  total revenue   : EUR {df['total_price'].sum():,.2f}")
    print(f"  total nights    : {room_nights:,}")
    print(f"  capacity        : {capacity:,} room-nights "
          f"({TOTAL_ROOMS} rooms x {WINDOW_DAYS} days)")
    print(f"  avg occupancy   : {occupancy*100:.1f}%")
    print(f"  avg price/night : EUR {df['total_price'].sum() / room_nights:.2f}")


if __name__ == "__main__":
    main()
