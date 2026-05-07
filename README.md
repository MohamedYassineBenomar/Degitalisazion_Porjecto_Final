# HotelMar — AI-Powered Direct Booking & Dynamic Pricing

Final project for the Digitalization course.

## What this is

HotelMar is a 4-star, 150-room hotel in Sitges (Spain). Today, ~75% of its
bookings come through Booking.com, costing roughly €180,000/year in
commissions. Static seasonal pricing also leaves money on the table.

This MVP is a small web app that addresses both problems:

1. **Direct booking page** — guests can book on the hotel's own site, so
   the hotel keeps 100% of the revenue.
2. **AI price recommender** — a Prophet time-series model suggests the
   optimal daily room rate to maximize **RevPAR** (Revenue Per
   Available Room).

## Tech stack

- Python 3.11+
- [Streamlit](https://streamlit.io/) — web UI
- [Prophet](https://facebook.github.io/prophet/) — time-series forecasting
- Pandas — data cleaning
- SQLite — booking storage
- Plotly — charts
- Streamlit Community Cloud — free hosting

## Data

Source: Kaggle "Hotel Booking Demand" dataset (~119k bookings).
We use only **Resort Hotel** rows and exclude cancellations.
The price column is `adr` (Average Daily Rate).

## Project structure

```
HotelMar/
├── data/              # raw CSV + cleaned daily_prices.csv
├── scripts/           # one-off scripts (data prep, model training)
├── app/               # streamlit app pages
├── models/            # saved Prophet model (.pkl)
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start

```bash
# 1. Create the virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare the daily price series
python scripts/01_prepare_data.py
```

## Roadmap (milestones)

| Tag    | What's done                              |
| ------ | ---------------------------------------- |
| v0.1   | Project setup + data preparation         |
| v0.2   | Prophet model trained & saved            |
| v0.3   | Guest-facing booking page                |
| v0.4   | Manager dashboard with price suggestions |
| v1.0   | Deployed to Streamlit Community Cloud    |
