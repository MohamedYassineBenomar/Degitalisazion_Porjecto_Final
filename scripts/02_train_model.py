"""
Step 2 — Train the Prophet pricing model for HotelMar.

Reads the cleaned daily price series produced by 01_prepare_data.py,
fits a Prophet model with yearly + weekly seasonality, saves the
trained model as a pickle, and generates a 90-day forward forecast.

Run from the project root:
    python scripts/02_train_model.py
"""

from pathlib import Path
import pickle
import warnings

import matplotlib

# Use a non-interactive backend so the script can save plots
# without needing an X display (works on a headless server too).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

# Prophet emits a lot of harmless info-level chatter; quiet it down.
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "daily_prices.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "price_model.pkl"
FORECAST_CSV = PROJECT_ROOT / "data" / "forecast.csv"
FORECAST_PLOT = PROJECT_ROOT / "data" / "forecast_plot.png"

FORECAST_DAYS = 90


def main() -> None:
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["ds"])
    print(f"  training points: {len(df):,}")
    print(f"  date range     : {df['ds'].min().date()}  ->  {df['ds'].max().date()}")

    # Build the Prophet model.
    # - yearly_seasonality=True: captures summer-vs-winter price cycles
    #   (a resort hotel obviously has strong yearly seasonality).
    # - weekly_seasonality=True: captures weekend vs weekday patterns.
    # - daily_seasonality=False: we have one value per day, so an
    #   intra-day component would be meaningless.
    # - interval_width=0.80: the yhat_lower/yhat_upper band will cover
    #   ~80% of likely outcomes (a sensible default for pricing decisions).
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
    )

    print("\nTraining Prophet (this takes a few seconds) ...")
    model.fit(df)
    print("  done.")

    # Save the model so we can load it later from the Streamlit app
    # without retraining every time.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  model saved -> {MODEL_PATH}")

    # Build a future dataframe and predict.
    future = model.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
    forecast = model.predict(future)

    # Keep only the columns we actually need downstream.
    forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_out.to_csv(FORECAST_CSV, index=False)
    print(f"  forecast saved -> {FORECAST_CSV}")

    # Save the standard Prophet plot (historical points + fitted curve
    # + future forecast with uncertainty band).
    fig = model.plot(forecast)
    plt.title("HotelMar — daily ADR forecast (history + 90 days ahead)")
    plt.xlabel("Date")
    plt.ylabel("Price (EUR)")
    fig.tight_layout()
    fig.savefig(FORECAST_PLOT, dpi=120)
    plt.close(fig)
    print(f"  plot saved     -> {FORECAST_PLOT}")

    # Summary report.
    last_train = df["ds"].max().date()
    future_only = forecast_out[forecast_out["ds"] > df["ds"].max()].reset_index(drop=True)

    print("\n=== Training summary ===")
    print(f"Training points    : {len(df):,}")
    print(f"Last training date : {last_train}")
    print(f"Forecast horizon   : {FORECAST_DAYS} days "
          f"({future_only['ds'].min().date()}  ->  {future_only['ds'].max().date()})")
    print(f"Forecast avg price : {future_only['yhat'].mean():.2f} EUR")
    print(f"Forecast min price : {future_only['yhat'].min():.2f} EUR  "
          f"on {future_only.loc[future_only['yhat'].idxmin(), 'ds'].date()}")
    print(f"Forecast max price : {future_only['yhat'].max():.2f} EUR  "
          f"on {future_only.loc[future_only['yhat'].idxmax(), 'ds'].date()}")

    print("\nFirst 10 forecasted days:")
    sample = future_only.head(10).copy()
    sample["ds"] = sample["ds"].dt.date
    sample[["yhat", "yhat_lower", "yhat_upper"]] = sample[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].round(2)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
