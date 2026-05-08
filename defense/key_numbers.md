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
| Training data | Kaggle "Hotel Booking Demand" — Resort Hotel + nearby markets only |
| Country filter | **{ESP, PRT, FRA, ITA, AND}** — Spain + 4 closest neighbours |
| Filtered bookings (rows) | 15,079 of 28,938 Resort non-cancelled (52 %) |
| Training rows after aggregation | **792 days** (2015-07-01 → 2017-08-31) |
| Train/test split | **80 / 20 chronological** (633 / 159 days) |
| MAE (avg absolute error) | **€24.37 / night** |
| RMSE | **€29.41 / night** |
| **MAPE** | **18.51 %  →  GOOD (10–20 %)** |
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

### Dashboard KPIs (from 2,494 seeded bookings, demo window 152 days)

| Metric | Without AI | With AI naive | With AI realistic (η=−0.7) |
|---|---:|---:|---:|
| Total bookings | 2,494 | 2,494 | **2,040** |
| Room-nights sold | 7,805 | 7,805 | **6,390** |
| Total revenue | €1,206,069 | €1,447,808 | **€1,218,391** |
| Variable op. costs (€43/rn) | €335,615 | €335,615 | **€274,769** |
| **Gross profit** | **€870,454** | **€1,112,193** | **€943,622** |
| Gross margin | 72.2 % | 76.8 % | **77.4 %** |
| Avg price / night | €154.53 | €185.50 | €190.67 |
| Avg occupancy | 34.5 % | 34.5 % | 28.2 % |

**Realistic profit lift: +€73 K (+8.4 %)** = €12 K from pricing
   + €61 K from cost savings on the 1,415 fewer room-nights served.

---

## 💰 Impact (the close — gross-profit version)

| | |
|---|---:|
| Variable cost per room-night (industry standard) | **€43** |
| Realistic profit lift on the **demo** (152-day back-test) | **+€73 K** |
| ↳ from smarter pricing (revenue effect) | +€12 K |
| ↳ from lower variable costs (fewer rooms × €43) | +€61 K |
| Annualization factor (€6.2 M ÷ demo €1.21 M static) | **≈ 5.1×** |
| **Realistic annual gross-profit lift (AI alone)** | **~€370 K** |
| Direct-channel commission savings (50 % shift) | **+€90 K** |
| **Total combined annual impact** | **~€450-500 K** |
| As % of €6.2 M base | **~7 %** |
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

> **MAPE 18.51 %**            — the model is honest
>
> **+€360 K / year gross profit**  — realistic AI lift, scaled to €6.2 M
>
> **<€100 / month**            — what it costs to run

(And one bonus: the realistic profit lift comes ~85 % from
**cost discipline** — fewer low-margin rooms sold at higher prices —
not the headline revenue uplift. That's the unglamorous truth of
revenue management.)

---

## Live URL + password (write on your hand)

> <https://degitalisazionporjectofinal-jbqw9jfbm45spzoaf9dqep.streamlit.app/>
>
> Manager password: **`admin123`**
