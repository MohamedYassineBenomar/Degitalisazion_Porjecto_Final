"""
Step 4 — Train a classical-ML alternative to the Prophet pricing model.

Parallel to scripts/02_train_model.py — same daily price series, same
80 / 20 chronological hold-out (last 62 days blind), same artefact
layout — but the algorithm is a **Gradient Boosting Regressor**
(scikit-learn). Features are engineered explicitly from the date
column (no Prophet-style decomposition):

    year, month, day_of_month, day_of_week, day_of_year,
    week_of_year, is_weekend, days_since_start

This script trains the model twice:

  Phase A — Evaluation
    Fit on the first 731 days, predict the last 62 (blind), report
    MAE / RMSE / MAPE. Saves data/test_evaluation_ml.png and
    data/blind_test_predictions_ml.csv.

  Phase B — Production
    Refit on the full 793 days, save as models/price_model_ml.pkl,
    and generate a 90-day forward forecast as data/forecast_ml.csv
    plus a static history+forecast plot.

Run from the project root:
    python scripts/04_train_ml_model.py
"""

from pathlib import Path
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "daily_prices.csv"
ML_MODEL_PATH = PROJECT_ROOT / "models" / "price_model_ml.pkl"
ML_FORECAST_CSV = PROJECT_ROOT / "data" / "forecast_ml.csv"
ML_FORECAST_PLOT = PROJECT_ROOT / "data" / "forecast_plot_ml.png"
ML_BLIND_TEST_CSV = PROJECT_ROOT / "data" / "blind_test_predictions_ml.csv"
ML_EVAL_PLOT = PROJECT_ROOT / "data" / "test_evaluation_ml.png"

FORECAST_DAYS = 90
TEST_DAYS = 62
TRAIN_ORIGIN = pd.Timestamp("2015-07-01")


def date_features(dates) -> pd.DataFrame:
    """Build the feature matrix from dates alone. Mirrors what Prophet
    decomposes internally (trend + yearly + weekly) but as explicit
    columns a tree ensemble can learn from."""
    # Accept Series, DatetimeIndex, list-like; normalise to a Series.
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    return pd.DataFrame({
        "year":              dates.dt.year,
        "month":             dates.dt.month,
        "day_of_month":      dates.dt.day,
        "day_of_week":       dates.dt.dayofweek,
        "day_of_year":       dates.dt.dayofyear,
        "week_of_year":      dates.dt.isocalendar().week.astype(int),
        "is_weekend":        (dates.dt.dayofweek >= 5).astype(int),
        "days_since_start":  (dates - TRAIN_ORIGIN).dt.days.astype(int),
    })


def build_model() -> GradientBoostingRegressor:
    """Gradient Boosting with hyperparameters chosen for small (~800-row)
    tabular regression — modest depth, conservative learning rate,
    enough trees to converge."""
    return GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=3,
        subsample=0.85,
        random_state=42,
    )


def quality_label(mape: float) -> str:
    if mape < 10:
        return "EXCELLENT (<10%)"
    if mape < 20:
        return "GOOD (10-20%)"
    if mape < 50:
        return "REASONABLE (20-50%)"
    return "POOR (>50%)"


def main() -> None:
    print(f"Loading {INPUT_CSV} ...")
    df = (
        pd.read_csv(INPUT_CSV, parse_dates=["ds"])
        .sort_values("ds")
        .reset_index(drop=True)
    )
    print(f"  total points : {len(df):,}")
    print(f"  date range   : {df['ds'].min().date()}  ->  {df['ds'].max().date()}")

    # ------------------------------------------------------------------
    # Phase A — Hold-out evaluation
    # ------------------------------------------------------------------
    split_idx = max(1, len(df) - TEST_DAYS)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\n=== Phase A: Hold-out evaluation (~2-year train / last {TEST_DAYS}-day blind test) ===")
    print(f"  train: {len(train_df):3d} days  "
          f"{train_df['ds'].min().date()} -> {train_df['ds'].max().date()}")
    print(f"  test : {len(test_df):3d} days  "
          f"{test_df['ds'].min().date()} -> {test_df['ds'].max().date()}")

    print("\nTraining Gradient Boosting on the train slice only ...")
    X_train = date_features(train_df["ds"])
    y_train = train_df["y"].values
    X_test = date_features(test_df["ds"])
    y_test = test_df["y"].values

    eval_model = build_model()
    eval_model.fit(X_train, y_train)
    print("  done.")

    # Predict on the blind test slice.
    yhat_test = eval_model.predict(X_test)

    # Metrics.
    err = y_test - yhat_test
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((abs_err / np.abs(y_test) * 100).mean())

    print("\n=== Hold-out test results ===")
    print(f"  MAE  (avg absolute error)  : {mae:7.2f} EUR")
    print(f"  RMSE (root mean sq error)  : {rmse:7.2f} EUR")
    print(f"  MAPE (avg % error)         : {mape:7.2f} %")
    print(f"  -> quality                 : {quality_label(mape)}")

    # Simple confidence-band proxy: ±1.28×RMSE ~ 80% (z-score for normal).
    # Tree models don't ship native intervals; this is a defensible
    # approximation good enough to plot uncertainty visually.
    bound = 1.28 * rmse

    # Persist blind-test predictions for the dashboard.
    blind_out = pd.DataFrame({
        "ds": test_df["ds"].values,
        "y_actual": y_test,
        "yhat": yhat_test,
        "yhat_lower": yhat_test - bound,
        "yhat_upper": yhat_test + bound,
    })
    blind_out.to_csv(ML_BLIND_TEST_CSV, index=False)
    print(f"  blind-test csv  -> {ML_BLIND_TEST_CSV}")

    # Eval plot: training history (grey), test actual (blue), test
    # predicted (purple), 80%-band as a purple shaded region. We use
    # purple for the ML model to distinguish it visually from Prophet's
    # orange.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train_df["ds"], train_df["y"], color="#bdc3c7", linewidth=0.7,
            label=f"Train history ({len(train_df)} days)")
    ax.plot(test_df["ds"], y_test, color="#3c91b3", linewidth=1.4,
            label="Test — actual")
    ax.plot(test_df["ds"], yhat_test, color="#8e44ad", linewidth=1.6,
            label="Test — predicted (Gradient Boosting)")
    ax.fill_between(
        test_df["ds"], yhat_test - bound, yhat_test + bound,
        color="#8e44ad", alpha=0.15, label="~80% interval (±1.28·RMSE)",
    )
    ax.axvline(train_df["ds"].max(), color="#34495e", linestyle="--",
               linewidth=1.0, alpha=0.6)
    ax.set_title(
        "Held-out evaluation — Gradient Boosting on Resort Hotel ADR\n"
        f"MAE {mae:.2f} EUR · RMSE {rmse:.2f} EUR · MAPE {mape:.2f}%  ({quality_label(mape)})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ML_EVAL_PLOT, dpi=120)
    plt.close(fig)
    print(f"  eval plot saved -> {ML_EVAL_PLOT}")

    # ------------------------------------------------------------------
    # Phase B — Production model (refit on the full dataset)
    # ------------------------------------------------------------------
    print("\n=== Phase B: Production model (refit on FULL dataset) ===")
    print(f"Training Gradient Boosting on all {len(df)} days ...")
    X_full = date_features(df["ds"])
    y_full = df["y"].values
    prod_model = build_model()
    prod_model.fit(X_full, y_full)
    print("  done.")

    ML_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ML_MODEL_PATH, "wb") as f:
        pickle.dump(prod_model, f)
    print(f"  model saved   -> {ML_MODEL_PATH}")

    # 90-day forecast.
    last_train_date = df["ds"].max()
    future_dates = pd.date_range(
        last_train_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D"
    )
    X_future = date_features(future_dates)
    yhat_future = prod_model.predict(X_future)

    # Update the band estimate using a fresh in-sample residual.
    in_sample_pred = prod_model.predict(X_full)
    in_sample_rmse = float(np.sqrt(((y_full - in_sample_pred) ** 2).mean()))
    prod_bound = 1.28 * max(in_sample_rmse, rmse)  # never tighter than test RMSE

    forecast_out = pd.DataFrame({
        "ds": future_dates,
        "yhat": yhat_future,
        "yhat_lower": yhat_future - prod_bound,
        "yhat_upper": yhat_future + prod_bound,
    })
    forecast_out.to_csv(ML_FORECAST_CSV, index=False)
    print(f"  forecast saved-> {ML_FORECAST_CSV}")

    # Production plot.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["ds"], df["y"], color="#3c91b3", linewidth=0.8,
            label="Historical (actual)")
    ax.plot(future_dates, yhat_future, color="#8e44ad", linewidth=2,
            label="ML forecast (90 days)")
    ax.fill_between(future_dates, yhat_future - prod_bound,
                    yhat_future + prod_bound, color="#8e44ad", alpha=0.15,
                    label="~80% interval")
    ax.set_title("HotelMar — ML daily ADR forecast (history + 90 days ahead)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ML_FORECAST_PLOT, dpi=120)
    plt.close(fig)
    print(f"  plot saved    -> {ML_FORECAST_PLOT}")

    # ------------------------------------------------------------------
    # Defense summary
    # ------------------------------------------------------------------
    fut_only = forecast_out
    feature_imp = pd.Series(
        prod_model.feature_importances_, index=X_full.columns
    ).sort_values(ascending=False)

    print("\n" + "=" * 64)
    print(" DEFENSE SUMMARY — Gradient Boosting evaluation & production")
    print("=" * 64)
    print(f"  Dataset           : {len(df):,} days "
          f"({df['ds'].min().date()} -> {df['ds'].max().date()})")
    print(f"  Split             : {len(train_df):,} train days / "
          f"{len(test_df):,} test days (~2 months, blind, chronological)")
    print()
    print("  Hold-out metrics (lower is better):")
    print(f"    MAE             : {mae:7.2f} EUR per night")
    print(f"    RMSE            : {rmse:7.2f} EUR per night")
    print(f"    MAPE            : {mape:7.2f} %  -> {quality_label(mape)}")
    print()
    print("  Feature importances (gain-weighted):")
    for name, val in feature_imp.items():
        bar = "█" * int(val * 50)
        print(f"    {name:18s} {val*100:5.1f}%  {bar}")
    print()
    print(f"  Production model  : refit on the full {len(df):,} days")
    print(f"  90-day forecast   : {fut_only['ds'].min().date()} -> "
          f"{fut_only['ds'].max().date()}  "
          f"(avg EUR {fut_only['yhat'].mean():.2f})")
    print("=" * 64)


if __name__ == "__main__":
    main()
