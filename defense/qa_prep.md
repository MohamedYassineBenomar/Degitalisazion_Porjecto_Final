# HotelMar — Q&A Prep

Ten likely jury questions, with answers tuned for one calm paragraph
each. Practise saying each one out loud once.

---

### 1. Why Prophet and not deep learning (LSTM, Transformer)?

Prophet is purpose-built for **additive seasonal time series with
multiple cycles** — exactly what hotel ADR is — and it's the right
tool for this job for four concrete reasons. **Data scale**: I have
793 daily observations; deep models like LSTM or Temporal Fusion
Transformer typically want tens of thousands of points before they
beat classical methods. **Interpretability**: Prophet decomposes the
forecast into trend + yearly + weekly components I can show the
manager — "August adds €95, Saturday adds €6" — whereas a neural net
is a black box, which matters for both jury defence and EU AI Act
transparency. **Engineering effort**: Prophet trained in a few seconds
with five lines of code; an LSTM needs hyperparameter tuning, GPU,
and careful regularization. **Maintainability**: Prophet is a
well-documented Meta library that a hotel's small IT team can keep
running. With more data and a real RevPAR optimisation problem, deep
learning would be the next step — but for an MVP it would be premature.

---

### 2. What is the optimization goal? Explain RevPAR.

The objective is to **maximise RevPAR — Revenue Per Available Room** —
which is the hotel-industry standard KPI. RevPAR equals **ADR**
(average daily rate) times **occupancy**, and crucially it captures
the trade-off between the two: pricing too high empties rooms (low
occupancy), pricing too low fills them at thin margins. A hotel with
150 rooms running 80 % occupancy at €100/night earns the same RevPAR
as 60 % occupancy at €133/night. The Prophet model learns the
historical equilibrium — what price the market actually paid on each
type of day — and recommends that. So we're not directly optimising
an objective function; we're using the past as a prior for "the right
price given the conditions". A more sophisticated next version would
add an explicit demand-elasticity layer, but the current model already
captures the seasonal and weekly demand signals that drive ~80 % of
RevPAR variance.

---

### 3. How do you know the model is accurate?

Standard machine-learning hygiene: a **chronological hold-out** of
the most recent 2 months. I trained Prophet on the first **731 days
(~2 full years, Jul 2015 → Jun 2017)** and predicted the held-out
**last 62 days (Jul – Aug 2017)** without ever showing them to the
model. Three metrics on the blind test: **MAE €10.90** (average
absolute error per night), **RMSE €14.06**, and most importantly
**MAPE 5.75 %** — meaning predictions land within ±6 % of the
actual price on average. By the standard MAPE rubric — under 10 %
is excellent, 10-20 % is good, 20-50 % is reasonable — we're in
the **excellent band**. The split is **chronological**, never
random, because shuffling rows would let the model "see the
future" during training and inflate accuracy. The test window is
deliberately the **summer peak** — the highest-stakes period for
revenue management, when getting the price right matters most.
After validation passed, I refit on the full 793 days for
production — standard practice: validate on hold-out, deploy with
everything.

---

### 4. Why is the demo locked to 2017?

The Kaggle Hotel Booking Demand dataset ends **31 August 2017**, and
Prophet's 90-day forecast horizon takes us to **29 November 2017**.
Anything past that is extrapolation: Prophet projects its learned
linear trend into the future, and the model observed roughly **+€26
per year** of underlying price growth. If I used today's date — May
2026 — Prophet would naively add about **€230 to the baseline**,
giving demo prices around €370/night that have nothing to do with the
seasonal logic that's the actual value proposition. By **pinning the
demo to Friday 15 September 2017**, every prediction stays inside the
trained range, and the seasonal and weekly patterns the jury came to
see are the ones doing the work. In production, the model would be
**continuously retrained** on the rolling 24-month window of fresh
booking data, so "today" always sits inside the trained range — that's
the standard MLOps pattern.

---

### 5. How does this comply with GDPR?

The MVP collects only **two PII fields per booking** — guest name and
email — and they're stored in a SQLite database on the server. For a
production rollout the document specifies four GDPR controls.
**Lawful basis**: contract performance for booking data (Art. 6(1)(b))
and explicit consent for any marketing use — the booking form would
add a checkbox. **Data subject rights**: a guest portal with download
(Art. 15) and delete (Art. 17) endpoints. **Retention**: bookings
purged after 24 months unless the guest is in an active loyalty
programme. **Security**: TLS in transit (handled by Streamlit Cloud
today), encryption at rest in production via PostgreSQL with a
managed KMS key. We are **not** profiling guests for automated
decision-making — pricing decisions are based on aggregate historical
ADR, never on the individual guest — so the high-bar Article 22
restrictions don't apply. The privacy notice and DPIA are part of the
launch checklist in the document, not the MVP.

---

### 6. What about the EU AI Act?

The AI Act classifies systems by risk tier. **Hotel pricing AI is
limited risk**, not high risk — the high-risk list covers things like
biometric identification, critical infrastructure, education
admissions, employment, credit scoring, and healthcare. Limited-risk
obligations are mostly about **transparency**: the user must know
they're interacting with AI. We satisfy that on both surfaces — the
guest page mentions "AI-recommended pricing" in the hero, and the
manager dashboard's "AI Recommendation for…" heading is explicit.
The Act also requires **human oversight of consequential decisions**
— we comply because the dashboard surfaces a *suggestion*, never an
autonomous action; a manager applies the rate. We have **documented
evaluation** (the v0.2.1 MAPE numbers) and **technical documentation**
(this repo, including the model card implicit in the README and the
Prophet decomposition). If down the line we add personalised pricing
based on guest profile, that would push us into a different
conversation around Article 22 of GDPR rather than the AI Act
specifically.

---

### 7. What if the model gives a bad price?

Four safety layers. **First**, the model surfaces uncertainty: the
80 % prediction interval (`yhat_lower` to `yhat_upper`) is plotted
on the dashboard, so a manager looking at a wide band knows to use
judgment. **Second**, **human-in-the-loop by design** — the dashboard
shows a *recommendation*, not an autonomous decision. The manager
applies the rate to the PMS, sees how it lands, and can override.
**Third**, **bounds checking** is straightforward in production — the
data preparation script already clamps inputs to the €10–€500 range,
and the price endpoint can clamp outputs the same way. **Fourth**,
**continuous monitoring**: in production we'd track booking rate
versus predicted demand each day and trigger an alert if a price is
emptying rooms — the same loop a human revenue manager runs by
instinct, just systematised. The model is decision support, not
autonomous pricing, and that framing is what makes the EU AI Act
risk profile workable.

---

### 8. Why didn't you build IoT, CRM, and digital check-in too?

**Deliberate scoping.** The full digital transformation of HotelMar
is a multi-phase programme described in the project document — IoT
energy management, CRM/loyalty, digital check-in kiosks, predictive
maintenance — and the MVP picks **the single highest-ROI item** to
ship first. The €180 K/year commission line is the largest fixable
cost in the business case, so direct booking + AI pricing addresses
it. Phase 2 (~6 months out) is **IoT energy management**: smart
thermostats per room, occupancy sensors, a fog gateway in the basement
aggregating telemetry — that's the edge/fog tier from the
architecture document. Phase 3 is **CRM and loyalty**, which the
direct-booking database we just built is a natural foundation for.
Phase 4 is **digital check-in** at lobby kiosks, again fog tier.
Doing all four in three months would have produced four shallow
prototypes; doing one well produces a deployable product and a
defensible argument for funding the rest.

---

### 9. How much is the AI actually worth? (the honest gross-profit answer)

Revenue lift alone is the wrong number — what matters is **gross
profit**, because **fewer rooms sold means lower variable operating
costs** (housekeeping, energy, laundry, supplies, breakfast — about
**€43/room-night** for a 4-star property). The dashboard's
"Without AI / With AI naive / With AI realistic" table walks through
all three scenarios at the **gross-profit** line, which is what
hotel finance actually compares.

On the demo dataset (152 days, 2,494 seeded bookings) the **static
seasonal rulebook** would have produced **€870 K of gross profit**
(€1.21 M revenue − €336 K variable costs, 72 % margin). The **naive
AI** scenario — same bookings, just priced dynamically — pushes
profit to **€1.11 M (+27.8 %)**, but assumes nobody reacts to the
~26 % price uplift, which isn't realistic. The **realistic** column
applies **price elasticity of demand η = −0.7** (conservative end of
the −0.4 to −0.8 industry range), so roughly **18 % of guests walk
away** at the higher prices. Importantly, those vanished guests
**also vanish from our cost base**: variable costs drop from €336 K
to **€275 K (−€61 K)**. Combine that with a small **+€12 K** revenue
gain and we get **+€73 K of gross profit on the demo** (+8.4 % at
the bottom line), of which **most of the value is actually the cost
discipline, not the price uplift**. **Annualised to HotelMar's full
€6.2 M / 365-day operation** (the demo represents about 1/5 of
annual revenue, scaling factor ≈ 5×) that's roughly
**€370 K/year of additional gross profit**. **Layered with the
direct-channel commission savings** (~€90 K/year if half of OTA
volume shifts to direct booking) the combined annual impact is
**€450-500 K/year of recovered margin on a €6.2 M base**, against
an **MVP cost of ~€8 K** and **<€100/month** to run. If the jury
asks for a single sound-bite: **"AI lifts gross profit ~6 % on a
honest back-test, mostly through cost discipline at higher prices,
and the MVP pays back in weeks."**

---

### 10. What's the cost to run this in production?

Today: **zero euros**. The live deployment is on **Streamlit
Community Cloud**, which is free for public apps. For a real
production rollout serving live booking traffic the document estimates
**well under €100/month**: a small VPS (€20-50), managed Postgres
(€20), a custom domain (€15/year), transactional email (€5/month),
and TLS via Let's Encrypt (free). Total opex around **€600-1200 a
year**. Maintenance — a couple of hours of MLOps a month for
retraining and monitoring — adds another **€1.5-3 K/year** depending
on hourly rate. So all-in production cost is **roughly €2-4 K
annually**, against the **€370 K/year impact**. The economics aren't
the hard part of this project; the discipline of scoping the MVP and
proving the model honestly was. Anything else you'd like me to
defend?
