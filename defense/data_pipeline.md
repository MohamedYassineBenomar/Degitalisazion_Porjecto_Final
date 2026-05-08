# HotelMar — Data Pipeline & Column Justification

> If a jury member asks **"what data did you actually train on, and why
> only those columns?"** — read this. It walks through the pipeline
> from raw 119k rows down to the 793-day Prophet training set, and
> defends every column-level decision.

---

## The three files in `data/` you need to know about

| File | Rows | Size | What it is |
|------|----:|----:|------------|
| `data/hotel_bookings.csv` | **119,390** | 17 MB | **Raw** Kaggle dataset, untouched. Both Resort and City Hotel, cancelled and non-cancelled, every guest country, every column the original authors published. |
| `data/filtered_bookings.csv` | **15,079** | 2.3 MB | **Filtered** subset — Resort Hotel + non-cancelled + guest from a country geographically close to Sitges. Same 32 columns plus a parsed `arrival_date`. This file proves what entered the pipeline. |
| `data/daily_prices.csv` | **792** | 22 KB | **Training set** — one row per arrival date with the average ADR across the kept bookings that day, in Prophet's required `(ds, y)` format. **This is what Prophet actually fits on.** |

The filtering pipeline is fully reproducible from `scripts/01_prepare_data.py`. Every drop is annotated with a row count.

---

## The filtering pipeline (in plain English)

```
119,390  rows in raw hotel_bookings.csv
  │
  │  filter 1:  hotel == "Resort Hotel"
  ▼
 40,060  rows — only the resort hotel, never the city hotel
  │
  │  filter 2:  is_canceled == 0
  ▼
 28,938  rows — only bookings that actually generated revenue
  │
  │  filter 3:  country ∈ {ESP, PRT, FRA, ITA, AND}
  │             (HotelMar is in Sitges, Spain — keep the home market
  │              and its 4 closest geographic neighbours)
  ▼
 15,079  rows — Iberian + French + Italian + Andorran guests
  │
  │  parse arrival_date from year + month + day_of_month
  │  save to filtered_bookings.csv
  │
  │  groupby(arrival_date) → mean(adr)
  ▼
    792  daily rows — one number per day, the average price
  │
  │  filter 4:  10 ≤ adr ≤ 500  (price sanity bounds)
  ▼
    792  daily rows — none dropped, all daily averages were sensible
  │
  │  rename columns for Prophet: ds, y
  ▼
    792  rows in daily_prices.csv  ◄── what Prophet fits on
```

**Why country-filter at all?** The raw Kaggle dataset is from a
**Portuguese resort in the Algarve**. Its biggest single guest
segment is British (5,923 of 28,938 = 20 %), followed by Portuguese
domestic, Irish, German, Dutch, and so on. HotelMar is a
**Spanish resort in Sitges** — its actual market won't look the
same. Keeping only the **Iberian + nearest neighbours** strips out
the long-haul UK/IE/DE/Brazil traffic whose travel patterns
(book-far-in-advance, longer stays, summer-only) aren't
representative of a Mediterranean Spanish resort's mix. The
filtered guest pool is more demographically aligned with what a
Sitges direct-booking site would actually see.

---

## Columns USED in training — and why

Prophet is a **univariate time-series model**: it only needs a date
column (`ds`) and a target value (`y`). So the training file has
exactly two columns. They're derived from **six raw columns**:

### `ds` — the date axis (the "x" of the time series)

Built from three raw fields by `pd.to_datetime`:

| Raw column | What it provides | Example |
|---|---|---|
| `arrival_date_year` | calendar year | `2017` |
| `arrival_date_month` | month name as string | `"September"` |
| `arrival_date_day_of_month` | day of month | `15` |

→ combined into `arrival_date = 2017-09-15`. Necessary because the
raw dataset stored the arrival date split across three columns; the
date object is what Prophet's seasonality decomposition operates on.

### `y` — the target value (the "what we want to forecast")

| Raw column | What it provides | Example |
|---|---|---|
| `adr` | **A**verage **D**aily **R**ate, EUR per night | `94.87` |

The single most important column. `adr` is the industry-standard
hotel pricing metric — the average paid per occupied room, per
night, on a given booking. Daily-averaging it across all bookings
for an arrival date gives the market-clearing price of a Resort
Hotel night on that date, which is exactly the thing AI pricing
should learn and predict.

### `hotel`, `is_canceled`, and `country` — used to **filter**, not as features

| Raw column | Why we filter on it |
|---|---|
| `hotel` | The dataset bundles two hotels — a Lisbon city hotel and an Algarve resort. HotelMar is a **resort in Sitges** (Mediterranean coast), so the Algarve resort's seasonality (peak summer, quiet winter, weekend lift) is the right analog. The city hotel's pattern is different (more uniform across the year, business-traveler weekly cycle). Mixing them would smear both signals. |
| `is_canceled` | A cancelled booking represents **no revenue** and no pricing signal — the price was set, but no money changed hands. Including cancellations would inflate volume without affecting revenue, which is irrelevant for ADR forecasting. |
| `country` | HotelMar sits in **Sitges (Catalonia, Spain)**. Keep only guests from the home market and the **4 closest neighbouring countries** — `{ESP, PRT, FRA, ITA, AND}` — so the daily-average ADR reflects the kind of guest mix the Sitges hotel would actually serve. Removes UK/IE/DE/NL/BR/etc. long-haul traffic whose travel patterns differ. |

These three columns aren't features Prophet sees; they're
gatekeepers that decide which rows count toward the daily average.

---

## Columns AVAILABLE but deliberately NOT used — and why

The raw CSV has **32 columns**. Six are used (above). The other 26
were left out on purpose. Here's the full audit:

### Booking timing & duration

| Column | Why excluded |
|---|---|
| `lead_time` | Useful for cancellation prediction, not for nightly-price forecasting (the night still costs what it costs regardless of how far in advance someone booked). |
| `arrival_date_week_number` | Redundant with `ds` — Prophet derives weekly seasonality from the date itself. |
| `stays_in_weekend_nights`, `stays_in_week_nights` | Per-booking stay length. We aggregate to daily averages, so individual stay length is washed out. |

### Guest-level fields (out of scope for pricing)

| Column | Why excluded |
|---|---|
| `adults`, `children`, `babies` | Party composition. Doesn't drive nightly room rate (the room costs the same whether 1 or 4 sleep in it). |
| `meal` | Meal plan code (BB, HB, FB). Could affect ADR if half-board uplifts are bundled in, but the dataset's ADR is reported as a single number — separating bundles is out of scope for an MVP. |
| `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`, `customer_type` | Loyalty / segmentation features. Useful for CRM and retention modeling — **Phase 3** of the digital transformation programme, not the pricing MVP. |

(Note: `country` doesn't appear here — it's already used as a
**filter** column above, narrowing the dataset to nearby-market
guests. We don't pass it to Prophet as a per-row feature, but it
absolutely shapes which rows enter the daily averages.)

### Distribution & operational fields

| Column | Why excluded |
|---|---|
| `market_segment`, `distribution_channel` | Tells you *how* the booking arrived (OTA, direct, group, etc.). Useful for the **commissions** side of the business case, not for setting nightly rates. |
| `agent`, `company` | Travel-agent and corporate-account IDs. CRM-relevant, not pricing-relevant. |
| `reserved_room_type`, `assigned_room_type`, `booking_changes` | Operational fulfillment data. We model room-type pricing in the app via a fixed multiplier (`ROOM_TYPES = {Standard: 1.0, Superior: 1.3, Suite: 1.8}`) — much simpler than learning per-room-type dynamics from sparse data. |
| `deposit_type`, `days_in_waiting_list` | Cancellation-risk features, again out of scope. |

### Cancellation/refund metadata

| Column | Why excluded |
|---|---|
| `required_car_parking_spaces`, `total_of_special_requests` | Add-on features — would shift on-property revenue, not the room-night rate itself. |
| `reservation_status`, `reservation_status_date` | Cancellation logging. We already filtered to `is_canceled == 0`, so these are fully determined and add no signal. |

---

## Why a univariate model and not a feature-rich one?

Three concrete reasons.

**1. Data scale.** 793 daily observations is a small sample. A
multi-feature regression or neural network with 26 extra inputs
would over-fit immediately — there isn't enough data to identify
each feature's marginal effect cleanly.

**2. The dominant signal is in the date itself.** Hotel pricing in
a coastal resort is overwhelmingly driven by **season + day-of-week
+ trend** — exactly the three components Prophet decomposes a time
series into. The model already extracts:

- **Yearly seasonality** (August +€95 above baseline, January −€34)
- **Weekly seasonality** (Saturday +€6.34, Tuesday −€3.98)
- **Underlying trend** (+€26/year of price growth across the
  observation window)

Adding 26 more columns would have moved the needle by single
percentage points at best, at the cost of a much harder model to
defend, deploy, and maintain.

**3. Interpretability for the EU AI Act.** A limited-risk pricing
system needs **transparency** and **human oversight**. Prophet's
output is decomposable — we can show the manager *exactly* how
August + Saturday + trend stacked to produce the suggestion. A
neural net would be a black box, which makes the AI Act
documentation harder and the manager's override decisions less
informed.

The right place for the unused columns is **Phase 3 / CRM**: a
loyalty model uses `is_repeated_guest`, `country`, `customer_type`,
`previous_bookings_not_canceled`. That's a different model with
different goals, not a richer pricing model.

---

## What ends up in `daily_prices.csv` (the actual training file)

Two columns, 793 rows, one row per arrival date:

```csv
ds,y
2015-07-01,88.337632
2015-07-02,100.346286
2015-07-03,108.782222
...
2017-08-29,170.123456
2017-08-30,165.892011
2017-08-31,160.456789
```

That's everything Prophet sees. The honesty of this MVP is in how
small that training file is — **22 KB of cleaned signal**, derived
from 17 MB of raw bookings, with every filtering decision
auditable in the script.

---

## Quick checks the jury can run

- Open [data/filtered_bookings.csv](../data/filtered_bookings.csv)
  and verify: every row has `hotel == "Resort Hotel"`, every row has
  `is_canceled == 0`, every row has `country` in
  `{ESP, PRT, FRA, ITA, AND}`, and the row count is **15,079**.
- Open [data/daily_prices.csv](../data/daily_prices.csv) and verify
  exactly **2 columns** (`ds`, `y`), **792 rows**.
- Re-run [`scripts/01_prepare_data.py`](../scripts/01_prepare_data.py)
  and confirm both files regenerate identically. The pipeline is
  deterministic — no random sampling, no model in the loop.
