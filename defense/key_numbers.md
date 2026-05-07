# HotelMar — Key Numbers Cheat Sheet

> Glance at this **one minute before walking in**. Everything you need
> to recall under pressure is here.

---

## 🏨 The business (where we are today)

| | |
|---|---:|
| Annual revenue | **€6.2 M** |
| Rooms | **150** |
| Occupancy (avg) | **52 %** |
| OTA share of bookings | **75 %** |
| Booking.com commission rate | **~15 %** |
| Annual commissions paid | **€180 K** |
| Static pricing rulebook | one summer rate, one winter rate |

---

## 🤖 The model (what Prophet learned)

| | |
|---|---:|
| Training data | Kaggle "Hotel Booking Demand" — Resort Hotel only |
| Training rows after cleaning | **793 days** (2015-07-01 → 2017-08-31) |
| Train/test split | **80 / 20 chronological** (634 / 159 days) |
| MAE (avg absolute error) | **€22.26 / night** |
| RMSE | **€26.22 / night** |
| **MAPE** | **16.99 %  →  GOOD (10–20 %)** |
| Forecast horizon | 90 days (Sep 1 → Nov 29 2017) |
| August premium learned | **+€95** above yearly avg |
| Saturday premium learned | **+€6.34**  (Friday +€5.88 ≈ **+€10 weekend lift**) |
| Underlying trend learned | **+€26 / year** of price growth |
| Quality bracket | "Solidly good — useful enough to set live prices, with manager judgment on high-variance days." |

---

## 📊 The demo (numbers on screen)

### Manager dashboard, today's price

| | |
|---|---:|
| Demo date | **Friday 15 September 2017** |
| AI suggestion (Standard Sea View) | **€133.23** |
| Historical September avg baseline | **€98.92** |
| Delta | **+€34.31  (+34.7 %)** |

### Default guest stay (Fri 22 → Mon 25 Sep 2017, Standard, 2 guests)

| Date | Price |
|---|---:|
| Fri 22 Sep 2017 | **€125.87** |
| Sat 23 Sep 2017 | **€125.57** |
| Sun 24 Sep 2017 | **€116.81** |
| **Total (3 nights)** | **€368.25** |

### Dashboard KPIs (from 2,494 seeded bookings)

| | |
|---|---:|
| Total bookings | **2,494** |
| Total revenue (demo window) | **€1,447,807.89** |
| Avg price per night | **€185.50** |
| Avg occupancy | **34.5 %** |
| Demo window | Jul 1 → Nov 29 2017 (152 days) |

---

## 💰 Impact (the close)

| | |
|---|---:|
| RevPAR uplift from AI pricing (industry benchmark) | **+22 %** |
| ↳ Additional revenue/year | **+€280 K** |
| Direct-channel commission savings (50 % shift) | **+€90 K** |
| **Total annual impact** | ****€370 K****  |
| As % of €6.2 M base | **~6 %** |
| MVP build cost | **~€8 K** |
| Production run cost | **<€100 / month  (~€1-2 K / year)** |
| Payback | **weeks, not years** |

---

## 🛠️ The stack (one line each)

- **Python 3.11** — the language
- **Streamlit 1.57** — the web UI (one Python file per page)
- **Prophet 1.3** — Meta's time-series forecaster
- **Pandas 3.0 + NumPy 2.4** — data handling
- **SQLite** — local booking ledger (PostgreSQL in production)
- **Plotly 6.7** — interactive charts
- **Streamlit Community Cloud** — free hosting, autoscaling, TLS

---

## 🔐 Compliance one-liners

- **GDPR**: minimal PII (name, email), Art. 6(1)(b) lawful basis
  (contract), retention + deletion in production checklist.
- **EU AI Act**: limited-risk system (not high-risk); transparency
  ("AI-recommended pricing" disclosed), human oversight (manager
  applies the rate), evaluation documented.
- **No automated decision-making affecting consumer rights** — pricing
  is a *recommendation*, not autonomous.

---

## 🎯 If they only remember three numbers

> **MAPE 16.99 %**     — the model is honest
>
> **€370 K / year**    — the annual impact on a €6.2 M business
>
> **<€100 / month**    — what it costs to run

---

## Live URL + password (write on your hand)

> <https://degitalisazionporjectofinal-jbqw9jfbm45spzoaf9dqep.streamlit.app/>
>
> Manager password: **`admin123`**
