"""
Step 5 — Seed demo bookings for the manager dashboard.

Thin CLI wrapper around utils.seed_demo_bookings() so the same
seeding logic is used both:
  - from scripts/03_seed_demo_bookings.py (manual local seeding), and
  - from utils.init_db() (auto-seed on empty DB at app boot — what
    keeps the Streamlit Cloud deployment populated).

Run from the project root:
    python scripts/03_seed_demo_bookings.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Make the app/ helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from utils import (  # noqa: E402
    DB_PATH,
    DEMO_SEED_WINDOW_END,
    DEMO_SEED_WINDOW_START,
    TOTAL_ROOMS,
    seed_demo_bookings,
)


def main() -> None:
    print(f"Demo window : {DEMO_SEED_WINDOW_START} -> {DEMO_SEED_WINDOW_END}")
    print(f"Hotel       : {TOTAL_ROOMS} rooms")
    print()

    n = seed_demo_bookings(verbose=True)
    if n == 0:
        print("\nSeeding skipped (no trained model). Run scripts/02_train_model.py first.")
        return

    # Quick KPI-style readout to confirm occupancy lands in the target band.
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT check_in, check_out, total_price FROM bookings",
            conn,
            parse_dates=["check_in", "check_out"],
        )

    df["nights"] = (df["check_out"] - df["check_in"]).dt.days
    room_nights = int(df["nights"].sum())
    window_days = (DEMO_SEED_WINDOW_END - DEMO_SEED_WINDOW_START).days + 1
    capacity = TOTAL_ROOMS * window_days
    occupancy = room_nights / capacity

    print("\n=== Seed summary ===")
    print(f"  total bookings  : {len(df):,}")
    print(f"  total revenue   : EUR {df['total_price'].sum():,.2f}")
    print(f"  total nights    : {room_nights:,}")
    print(f"  capacity        : {capacity:,} room-nights "
          f"({TOTAL_ROOMS} rooms x {window_days} days)")
    print(f"  avg occupancy   : {occupancy*100:.1f}%")
    print(f"  avg price/night : EUR {df['total_price'].sum() / room_nights:.2f}")


if __name__ == "__main__":
    main()
