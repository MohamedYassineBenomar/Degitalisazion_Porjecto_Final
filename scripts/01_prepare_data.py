"""
Step 1 — Data preparation pipeline for HotelMar.

Reads the raw Kaggle Hotel Booking Demand CSV, keeps only Resort Hotel
non-cancelled bookings, builds a clean daily price series (one row per
arrival date with the average ADR), and writes it to data/daily_prices.csv
in the (ds, y) format Prophet expects.

Run from the project root:
    python scripts/01_prepare_data.py
"""

from pathlib import Path
import pandas as pd

# Paths are resolved relative to the project root, which is the parent
# of this script's folder. This way the script works regardless of where
# you call it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "hotel_bookings.csv"
OUT_CSV = PROJECT_ROOT / "data" / "daily_prices.csv"

# Sanity bounds for ADR (average daily rate, in EUR). The raw dataset
# contains zeros (free stays / data errors) and a few extreme outliers
# that would distort the forecast, so we cap the range.
MIN_PRICE = 10.0
MAX_PRICE = 500.0


def main() -> None:
    print(f"Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV)
    print(f"  raw rows: {len(df):,}")

    # 1. Resort Hotel only.
    df = df[df["hotel"] == "Resort Hotel"]
    print(f"  after Resort Hotel filter: {len(df):,}")

    # 2. Drop cancellations — we only want bookings that actually
    #    generated revenue.
    df = df[df["is_canceled"] == 0]
    print(f"  after dropping cancellations: {len(df):,}")

    # 3. Build a real date column from the three arrival_date_* fields.
    #    arrival_date_month is a month name ("January", "February", ...).
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"]
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )

    # 4. Group by date and average the ADR across all bookings on that day.
    daily = (
        df.groupby("arrival_date", as_index=False)["adr"]
        .mean()
        .rename(columns={"arrival_date": "ds", "adr": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    print(f"  daily rows before price cap: {len(daily):,}")

    # 5. Drop unrealistic prices.
    daily = daily[(daily["y"] >= MIN_PRICE) & (daily["y"] <= MAX_PRICE)]
    daily = daily.reset_index(drop=True)
    print(f"  daily rows after price cap [{MIN_PRICE}, {MAX_PRICE}]: {len(daily):,}")

    # 6. Save.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")

    # 7. Summary report.
    print("\n=== Summary ===")
    print(f"Total days     : {len(daily):,}")
    print(f"Date range     : {daily['ds'].min().date()}  ->  {daily['ds'].max().date()}")
    print(f"Min price (EUR): {daily['y'].min():.2f}")
    print(f"Avg price (EUR): {daily['y'].mean():.2f}")
    print(f"Max price (EUR): {daily['y'].max():.2f}")
    print("\nFirst 10 rows:")
    print(daily.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
