# HotelMar — Architecture Summary

A one-page map of the **cloud → fog → edge → mist** digital architecture
described in the project document, and which parts the MVP actually
delivers today versus which are scoped as future phases.

---

## The four-tier model in one sentence

Modern digital systems are layered by **distance to the data**: mist
is the sensor itself, edge is the device computing on the sensor data,
fog is the local-network gateway aggregating across edge devices, and
cloud is the off-site SaaS that ingests and acts on it all.

```
     ┌─────────────────────────────────────────────────────────┐
     │  CLOUD          SaaS, training, central database        │  ← MVP
     │  (off-site)     long-term storage, analytics,           │     LIVES
     │                 model training & deployment             │     HERE
     └────────────────────────┬────────────────────────────────┘
                              │  internet
     ┌────────────────────────┴────────────────────────────────┐
     │  FOG            on-prem gateway / lobby server          │  Phase 3-4
     │  (in-hotel)     digital check-in kiosks, local cache,   │
     │                 PMS bridge, low-latency UX              │
     └────────────────────────┬────────────────────────────────┘
                              │  Wi-Fi / wired LAN
     ┌────────────────────────┴────────────────────────────────┐
     │  EDGE           per-device compute                      │  Phase 2
     │  (per room)     smart thermostat, room hub, occupancy   │
     │                 sensor logic, motor controllers         │
     └────────────────────────┬────────────────────────────────┘
                              │  Zigbee / BLE / wired
     ┌────────────────────────┴────────────────────────────────┐
     │  MIST           the sensors themselves                  │  Phase 2
     │  (raw signal)   PIR, temperature, humidity, door,       │
     │                 keycard reader, water-leak              │
     └─────────────────────────────────────────────────────────┘
```

---

## What the MVP built (cloud tier — green-lit, live today)

The MVP is **entirely cloud-tier SaaS** — and that's the right starting
point because it's where the highest-ROI problem lives (the €180 K
commission bleed and the static-pricing leakage).

| Concern | What's running |
|---|---|
| Web UI | Streamlit 1.57 deployed on Streamlit Community Cloud |
| Compute | Single Python container, autoscaled by the platform |
| Storage | SQLite for bookings (will become managed PostgreSQL in production) |
| AI | Prophet 1.3 model, trained offline, served via pickle |
| Auth | Hard-coded password (placeholder — production: OIDC/SSO) |
| Observability | Streamlit's built-in logs + GitHub commit history |
| Cost | €0/month on the free tier; ≤ €100/month productionised |

This tier delivers: **direct booking page, AI price recommender,
manager analytics dashboard, persistent booking ledger**.

---

## What the document scopes for future phases

### Phase 2 — IoT energy management (mist + edge + fog)

**Goal:** cut the hotel's **energy bill** (electricity is the second-
largest variable cost after commissions) by automating heating, cooling,
and lighting based on real occupancy.

| Tier | Component | Why it lives there |
|---|---|---|
| **Mist** | PIR motion sensors, door-magnet sensors, temperature/humidity probes per room | Raw analog signal — no network needed, just produce data |
| **Edge** | Smart thermostat with local rules ("if door open > 30 s, pause AC"), keycard hub | Sub-second decisions can't tolerate a round-trip to the cloud |
| **Fog** | A small server in the basement aggregating telemetry from all 150 rooms, batching before sending up | Cuts WAN traffic by ~95 %, survives internet outages, runs LAN-only ML for anomaly detection |
| **Cloud** | Trends dashboard, ML training on aggregated data, integration with the pricing model (e.g. "rooms 301-350 are HVAC-fault-prone, price them lower") | Cross-room learning, long-term storage, board-level reporting |

### Phase 3 — CRM and loyalty (cloud, with fog touchpoints)

The **direct-booking database** the MVP just created is the seed.
Phase 3 adds: guest preference profile, repeat-stay detection,
personalised offers, marketing automation. Mostly cloud, but the
**fog tier** matters for in-room tablets and lobby check-in screens
that need to greet returning guests in milliseconds.

### Phase 4 — Digital check-in (fog tier in the lobby)

**Self-service kiosks at reception** running on-prem (fog) for
sub-second response, syncing booking state with the cloud SaaS the
MVP built. ID scan happens on the kiosk (edge), document upload to
cloud, room key issued via the keycard hub (edge). Pure fog/edge use
case — latency-sensitive, wants to keep working when the WAN dies.

---

## Why this layering matters

The architecture is **deliberate, not buzzword bingo**.

- **Cloud only would fail** for IoT energy: 150 rooms × 4 sensors × 1
  reading/sec = 600 readings/sec to the WAN, 24/7. Fog + edge collapse
  that to a few aggregated signals per minute.
- **Edge/fog only would fail** for pricing: the model needs cross-hotel,
  cross-year data to learn seasonality; only the cloud has that view.
- **Mist alone is just sensors**; turning data into decisions requires
  every tier above to do its job.

The MVP is the **first production-grade slice** of this stack. It
ships the highest-ROI tier (cloud SaaS), validates the architecture
with a real working example, and creates a foundation the later
phases plug into without re-platforming.
