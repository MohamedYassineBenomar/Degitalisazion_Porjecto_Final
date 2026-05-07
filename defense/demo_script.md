# HotelMar — 5-Minute Demo Script

> Live URL: <https://degitalisazionporjectofinal-jbqw9jfbm45spzoaf9dqep.streamlit.app/>
> Manager password: `admin123`

**Before you start (30 seconds before, off-clock)**
- Open the live URL in a browser tab. Make sure the landing page is fully
  rendered.
- Open a *second* tab on the **Manager Dashboard** page and pre-enter
  the password — DON'T sign in yet. This saves 5 seconds during the demo
  if Streamlit Cloud has cold-started.
- Have a third tab on **Book Your Stay** ready as a fallback.

---

## (0:00 – 0:30) The problem — landing page

**You click:** show the landing page (first browser tab).

**On screen:**
- Navy/sea gradient hero with "🌊 HotelMar — A Mediterranean stay,
  priced fairly by AI"
- "★★★★ · SITGES, SPAIN" badge
- "Welcome aboard" copy + "Why book direct" card listing
  *No commissions · Best-rate guarantee · Smart pricing*

**You say (~30 seconds):**

> "HotelMar is a 4-star, 150-room hotel in Sitges. It runs €6.2 million
> a year — a healthy business — but it has two structural problems.
> First, **75 % of its bookings come through Booking.com**, which
> charges roughly 15 % commission. That's **€180,000 a year** going to
> a third party. Second, the hotel still uses **static seasonal pricing
> set in a spreadsheet** — one rate for high season, one for low — so
> demand spikes like long weekends and trade-show weeks leave money on
> the table. This MVP fixes both: a direct-booking page that bypasses
> the OTAs, and an AI model that prices every night dynamically. Let me
> show you the guest-facing side first."

---

## (0:30 – 2:00) The guest flow — direct booking

**You click:** sidebar → **Book Your Stay**.

**On screen:**
- "Book your stay" hero
- Four inputs in a row: **Check-in 22/09/2017**, **Check-out 25/09/2017**,
  **Room type Standard Sea View**, **Guests 2** (these are the defaults —
  don't change them, they were chosen to land on a Friday-to-Monday weekend
  so the Saturday premium is visible)
- Caption: "Demo dates limited to the AI model's forecast window…"

**You say (~20 seconds):**

> "A guest lands on our site, picks dates — let's say a long weekend
> in Sitges, Friday the 22nd to Monday the 25th of September 2017.
> Standard Sea View room, two guests. The booking date range is
> deliberately pinned to 2017 — I'll explain why in a moment. Click
> **Check Price**."

**You click:** the dark-navy **🔍 Check Price** button.

**On screen** (after ~1 second):
- "Your quote" section appears
- Per-night table:
  - Fri 22 Sep 2017 — **€125.87**
  - Sat 23 Sep 2017 — **€125.57**
  - Sun 24 Sep 2017 — **€116.81**
- Big total card on the right: **€368.25** for 3 nights
- Caption: "Average: €122.75/night · Room multiplier ×1.00 · Occupancy 65 %"

**You say (~50 seconds):**

> "The Prophet model returned a price for **each individual night**.
> Notice what it learned. Friday and Saturday come out around **€125 and
> a half**. Sunday drops to **€116.81** — that's **about €9 less**. The
> model figured out the weekend premium without anyone telling it: it
> simply observed in two-plus years of training data that
> Friday-Saturday demand consistently runs higher than Sunday-Monday. A
> static spreadsheet with one weekend rate and one weekday rate would
> miss this — it would either flatten Friday and Saturday together, or
> price all three nights at one number and either over-charge Sunday or
> under-charge Saturday. This is the core value: **per-night pricing
> based on real demand patterns**, not a manager's gut feel. Total for
> the stay: **€368.25**, no commissions, paid direct to the hotel."

**You click:** scroll down to "Confirm your booking" form.

**You type:**
- **Full name:** `Jury Demo` (or your own name)
- **Email:** `jury@example.com`

**You click:** the **✓ Confirm Booking** button.

**On screen:**
- Green success card: "Booking confirmed ✓"
- Reference like **HM-002495** (the next id after the 2,494 seeded rows)
- Balloons animation
- Message: "Thank you, Jury Demo. We've sent a confirmation to
  jury@example.com."

**You say (~10 seconds):**

> "Booking confirmed. The hotel keeps **a hundred percent** of that
> €368 — no Booking.com slice. The reservation is now in our SQLite
> database, ready for the manager to see in the dashboard. Note the
> reference number — **HM dash zero-zero-two-four-nine-five** — we'll
> find this row in a moment."

> 💡 **Memorize that reference number** — the exact number depends on
> how many bookings already exist. Just glance at the success card
> and read it out loud.

---

## (2:00 – 4:00) The manager dashboard

**You click:** sidebar → **Manager Dashboard**.

**On screen:**
- Sidebar shows the navy "HotelMar / Manager Console" logo card
- Lock-icon hero: "🔒 Manager sign-in"
- Password field
- Caption: "DEMO ONLY: the password is hard-coded in the source.
  A production deployment must use a proper auth backend."

**You type:** `admin123` (or paste, if you pre-loaded).

**You click:** **Sign in**.

**On screen:** the full dashboard renders top-to-bottom.

**You say (~15 seconds):**

> "I sign in as the revenue manager. Demo password admin one-two-three —
> the in-page notice makes clear that production would use proper
> single-sign-on. I'll walk you through the five sections from top
> to bottom."

### Section 1 — Today's recommended price

**Point at:** the big card on the left.

**On screen:**
- Heading: "AI Recommendation for Friday 15 September 2017"
- Headline number: **€133.23**
- Green delta: **▲ €34.31 (+34.7 %) vs historical September avg (€98.92)**
- Two metric cards on the right reinforcing the same numbers

**You say (~30 seconds):**

> "The model is telling the manager that **for tonight — Friday the 15th
> of September 2017 — the right price is €133.23**. The honest baseline
> we compare against is the **historical September average across our
> training data: €98.92**. So the AI is suggesting we charge **34.7 %
> more than the old static seasonal rate**. Why? Because the model has
> picked up that **mid-September is still warm tail-end of summer**,
> Fridays carry a weekend premium, and the underlying demand trend was
> upward across the years it observed. Three signals, one number,
> automatic."

### Section 2 — 90-day price forecast

**Point at:** the Plotly chart.

**On screen:**
- Blue line: **historical actual ADR**, ~793 days, three obvious summer peaks
- Orange line at the right edge: **90-day forecast**
- Soft orange band: **80 % prediction interval**

**You say (~25 seconds):**

> "Below is what the model has learned. Each summer peak sits **above
> €200**, each winter trough drops below €60 — that's the Mediterranean
> seasonal cycle. The orange line on the right is the **forward
> forecast**: the rate the manager can use for **strategic capacity
> planning, group bookings, OTA contract negotiations**. The shaded
> band is the **80 % confidence interval** — when the band widens,
> the model is telling the manager 'I'm less sure here, use judgment.'"

### Section 3 — KPIs

**Point at:** the four metric cards.

**On screen:**
- **Total bookings: 2,494**
- **Total revenue: €1,447,807.89**
- **Avg price/night: €185.50**
- **Avg occupancy: 34.5 %**

**You say (~20 seconds):**

> "Four headline KPIs across the demo window: just under **two thousand
> five hundred bookings**, **one and a half million euros** of revenue,
> **a hundred and eighty-five euros** average per night — that's
> **above the static rate**, the AI is pricing up — and **occupancy
> at 34.5 %**, which is the seasonal mid-point we'd expect for
> July-November. These would refresh live as new bookings come in."

### Section 4 — Recent bookings (find your booking!)

**You click:** the column header **"Booked at"** to sort descending
(or just look at the top — your booking is the most recent).

**Point at:** the top row of the table.

**On screen:** the booking you just made — `HM-002495 / Jury Demo /
jury@example.com / 22 Sep 2017 / 25 Sep 2017 / 3 / Standard Sea
View / 368.25 / [today's timestamp]`.

**You say (~15 seconds):**

> "And there's the booking I just made — **HM dash zero-zero-two-four-
> nine-five**, three nights, three hundred and sixty-eight euros, time-
> stamped seconds ago. From guest click to manager visibility:
> end-to-end in under two minutes."

### Section 5 — Revenue by day (don't dwell)

**Point at:** the bar chart.

**On screen:** ~152 bars in HotelMar sea-blue, peaking in summer,
falling toward autumn.

**You say (~10 seconds):**

> "Bottom of the page: revenue per stay-night across the booked window —
> the same Mediterranean seasonal shape, expressed in money."

---

## (4:00 – 5:00) Recap and impact

**You click:** scroll back to the top of the dashboard, or switch to
the landing tab.

**You say (~60 seconds, the close):**

> "So what does this MVP buy HotelMar in a year?
>
> First, **AI pricing**. Industry benchmarks for revenue-management
> systems land between **15 and 25 % RevPAR uplift**. We'll model 22 %.
> On a €6.2 million revenue base that's **roughly €280,000 in
> additional revenue per year** — same rooms, same nights, just priced
> at the right number.
>
> Second, the **direct channel**. Today 75 % of bookings flow through
> Booking.com at 15 % commission, which is the €180,000 line in the
> business case. If the direct page captures even **half** of that
> traffic — modest, conservative — that's another **€90,000 saved
> annually**.
>
> **Combined: €370,000 a year** of impact on a €6.2 million business.
> About **six percent of revenue**, recovered. The MVP costs roughly
> **€8,000** to build and well under **€100 a month** to run on
> Streamlit Cloud or a small VPS. Payback measured in weeks, not years.
>
> The model itself is **honest**: 80/20 chronological hold-out,
> **MAPE 16.99 %** — solidly in the 'good' band. The architecture is
> small, transparent, and explainable. **Prophet, not deep learning** —
> for this volume of data and this audience that's a deliberate choice,
> and I'm happy to defend it.
>
> That's HotelMar. Thank you — happy to take questions."

---

## Backup plan if something goes wrong

| Problem | What to do |
|---|---|
| Cold-start splash on first hit | "Streamlit Cloud is warming up — first hit takes 10–15 seconds, that's the only catch of the free tier. While it loads I'll show you the architecture diagram in the document." |
| Booking confirm doesn't show ref | Refresh the page, the row is still in the DB. Skip ahead to the dashboard and find any recent row. |
| Dashboard shows old number | Click **↻ Refresh Data** in the sidebar — the cache clears. |
| App crashes with red error | Open the README screenshot section ([docs/dashboard.png](../docs/dashboard.png)) and walk through that instead. Apologise once, move on. |
| Wi-Fi dies | The seeded local screenshots in `docs/` cover the entire flow. |

## Timing safety net

If you hit the 4-minute mark and you've only just finished the booking,
**skip Section 5** of the dashboard (revenue-by-day) and go straight
to the recap. The numbers in the recap are the part the jury will
remember.
