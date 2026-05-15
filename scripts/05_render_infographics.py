"""
Step 5 — Render brand-matched infographics for the README and the live app.

Generates three PNG infographics in docs/:

  - docs/infographic_value_prop.png   wide hero for the landing page:
        "75% OTAs · €180K commissions"  →  AI pricing + direct booking
        →  "+€370K/year recovered margin"
  - docs/infographic_pipeline.png      4-step data → model → price flow
        for the landing page
  - docs/infographic_elasticity.png    price-elasticity demand curve
        for the ML dashboard maths section

Pure matplotlib + AlgarveMar brand palette — deterministic, regenerable.
Run from the project root:
    python scripts/05_render_infographics.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = PROJECT_ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

# AlgarveMar brand palette (matches app/main.py CSS + dashboards).
NAVY = "#0a3d62"
SEA = "#3c91b3"
SKY = "#d9eef7"
SAND = "#f5e6d3"
UP = "#2e7d32"
DOWN = "#c0392b"
ML_PURPLE = "#8e44ad"
ORANGE = "#e67e22"
INK = "#1c2b3a"
MUTED = "#5b6b78"
WHITE = "#ffffff"

# Use a clean sans-serif system font.
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


# ---------------------------------------------------------------------------
# Helper: rounded card with a header band, used by several infographics.
# ---------------------------------------------------------------------------
def _card(ax, x, y, w, h, fill=WHITE, edge="#cdd9e3", lw=1.2, radius=0.06):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)


def _arrow(ax, x1, y1, x2, y2, color=SEA, lw=3):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=22,
        color=color, linewidth=lw,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Infographic 1 — Value proposition (landing-page hero strip)
# ---------------------------------------------------------------------------
def render_value_prop():
    fig, ax = plt.subplots(figsize=(14, 4.2), dpi=120)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Background fill — same sky-to-white gradient as the app.
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(
        grad, extent=(0, 14, 0, 4.2), aspect="auto",
        cmap=plt.cm.colors.LinearSegmentedColormap.from_list("hm_bg", [SKY, WHITE]),
        zorder=-1,
    )

    # --- LEFT: The Problem -------------------------------------------------
    _card(ax, 0.3, 0.4, 4.0, 3.4, fill=WHITE, edge="#e8c8c8", lw=1.5, radius=0.15)
    ax.text(2.3, 3.5, "Today", ha="center", va="center",
            fontsize=11, color=DOWN, fontweight="bold")
    ax.text(2.3, 2.95, "75%", ha="center", va="center",
            fontsize=46, color=NAVY, fontweight="bold")
    ax.text(2.3, 2.40, "of bookings via OTAs", ha="center", va="center",
            fontsize=11, color=INK)
    ax.text(2.3, 1.95, "(Booking.com et al.)", ha="center", va="center",
            fontsize=9.5, color=MUTED, style="italic")
    ax.plot([0.9, 3.7], [1.65, 1.65], color="#e8c8c8", lw=1)
    ax.text(2.3, 1.30, "≈ €180,000", ha="center", va="center",
            fontsize=20, color=DOWN, fontweight="bold")
    ax.text(2.3, 0.85, "lost to commissions / year", ha="center", va="center",
            fontsize=10, color=MUTED)

    # --- MIDDLE: The AI ---------------------------------------------------
    # Big arrow with two stacked labels.
    _arrow(ax, 4.5, 2.1, 9.5, 2.1, color=SEA, lw=4)
    ax.text(7.0, 3.0, "Direct booking", ha="center", va="center",
            fontsize=13, color=NAVY, fontweight="bold")
    ax.text(7.0, 2.55, "+ AI-recommended pricing", ha="center", va="center",
            fontsize=13, color=NAVY, fontweight="bold")
    ax.text(7.0, 1.35, "Prophet  +  LightGBM", ha="center", va="center",
            fontsize=10.5, color=MUTED, style="italic")
    ax.text(7.0, 0.95, "two parallel models", ha="center", va="center",
            fontsize=9.5, color=MUTED)

    # --- RIGHT: The Outcome -----------------------------------------------
    _card(ax, 9.7, 0.4, 4.0, 3.4, fill=WHITE, edge="#c8e8d0", lw=1.5, radius=0.15)
    ax.text(11.7, 3.5, "With AlgarveMar AI", ha="center", va="center",
            fontsize=11, color=UP, fontweight="bold")
    ax.text(11.7, 2.95, "+€370K", ha="center", va="center",
            fontsize=46, color=UP, fontweight="bold")
    ax.text(11.7, 2.40, "annual gross-profit lift", ha="center", va="center",
            fontsize=11, color=INK)
    ax.text(11.7, 1.95, "(realistic, elasticity-adjusted)", ha="center", va="center",
            fontsize=9.5, color=MUTED, style="italic")
    ax.plot([10.3, 13.1], [1.65, 1.65], color="#c8e8d0", lw=1)
    ax.text(11.7, 1.30, "≈ 6% of €6.2M revenue", ha="center", va="center",
            fontsize=14, color=NAVY, fontweight="bold")
    ax.text(11.7, 0.85, "MVP cost: ~€8K · runs <€100/mo", ha="center", va="center",
            fontsize=10, color=MUTED)

    out = DOCS / "infographic_value_prop.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Icon helpers — drawn primitives so we don't depend on emoji-capable fonts.
# Each draws a vector icon centred on (cx, cy) sized to ~0.55 unit radius.
# ---------------------------------------------------------------------------
def _icon_collect(ax, cx, cy, color):
    # Three ascending bars (a bar chart for "data collected").
    widths = 0.20
    heights = [0.30, 0.55, 0.80]
    base = cy - 0.45
    offsets = [-0.30, 0.0, 0.30]
    for ox, h in zip(offsets, heights):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx + ox - widths / 2, base), widths, h,
            boxstyle="round,pad=0.005,rounding_size=0.04",
            facecolor=color, edgecolor=color,
        ))


def _icon_train(ax, cx, cy, color):
    # A simple "neural net" — two columns of nodes + connecting lines.
    left_x, right_x = cx - 0.35, cx + 0.35
    left_ys = [cy + 0.30, cy, cy - 0.30]
    right_ys = [cy + 0.18, cy - 0.18]
    for ly in left_ys:
        for ry in right_ys:
            ax.plot([left_x, right_x], [ly, ry], color=color, lw=1.0, alpha=0.55, zorder=2)
    for ly in left_ys:
        ax.add_patch(mpatches.Circle((left_x, ly), 0.10, color=color, zorder=3))
    for ry in right_ys:
        ax.add_patch(mpatches.Circle((right_x, ry), 0.10, color=color, zorder=3))


def _icon_predict(ax, cx, cy, color):
    # A trending-up line with an arrowhead — forecast / prediction.
    xs = np.linspace(cx - 0.55, cx + 0.45, 50)
    ys = cy - 0.30 + 1.10 * (xs - (cx - 0.55)) ** 1.3 / (0.95 ** 1.3) * 0.55
    ys = np.clip(ys, cy - 0.45, cy + 0.45)
    ax.plot(xs, ys, color=color, lw=2.6, zorder=2)
    ax.annotate(
        "", xy=(cx + 0.55, cy + 0.35), xytext=(cx + 0.40, cy + 0.18),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4, mutation_scale=18),
    )
    # Three dotted "forecast" baseline points.
    for ox in (-0.45, -0.15, 0.15):
        ax.add_patch(mpatches.Circle((cx + ox, cy - 0.45), 0.04, color=color, alpha=0.5))


def _icon_earn(ax, cx, cy, color):
    # A coin: circle with an € glyph (DejaVu Sans supports €).
    ax.add_patch(mpatches.Circle((cx, cy), 0.50,
                                  facecolor=color, edgecolor=color, alpha=0.18, zorder=2))
    ax.add_patch(mpatches.Circle((cx, cy), 0.50,
                                  facecolor="none", edgecolor=color, lw=2.5, zorder=3))
    ax.text(cx, cy, "€", ha="center", va="center",
            fontsize=30, fontweight="bold", color=color, zorder=4)


# ---------------------------------------------------------------------------
# Infographic 2 — Pipeline (4-step "how it works")
# ---------------------------------------------------------------------------
def render_pipeline():
    fig, ax = plt.subplots(figsize=(14, 3.6), dpi=120)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Light background.
    ax.add_patch(mpatches.Rectangle((0, 0), 14, 3.6, color=WHITE, zorder=-2))

    # Title.
    ax.text(7, 3.25, "How AlgarveMar's AI pricing works",
            ha="center", va="center", fontsize=15,
            color=NAVY, fontweight="bold")

    # 4 steps, each a 3.0-wide card centred at:
    # x = 1.5, 5.0, 8.5, 12.0
    steps = [
        # (x_center, icon_drawer, headline, sub, accent_color)
        (1.5,  _icon_collect, "Collect",   "119,390 raw bookings\n→ 793 daily prices",      NAVY),
        (5.0,  _icon_train,   "Train",     "Prophet (Bayesian) +\nLightGBM (Microsoft)",   SEA),
        (8.5,  _icon_predict, "Predict",   "Per-night price for\nany date, any room",      ORANGE),
        (12.0, _icon_earn,    "Earn",      "+€73K demo profit\n(realistic, η = −0.7)",     UP),
    ]
    half_w = 1.45
    box_y, box_h = 0.4, 2.4

    for i, (cx, icon_fn, head, sub, accent) in enumerate(steps):
        _card(ax, cx - half_w, box_y, 2 * half_w, box_h,
              fill=WHITE, edge="#cdd9e3", lw=1.2, radius=0.10)
        # Top accent stripe.
        ax.add_patch(mpatches.Rectangle(
            (cx - half_w + 0.05, box_y + box_h - 0.15),
            2 * half_w - 0.10, 0.10, color=accent, zorder=2
        ))
        # Drawn vector icon (font-agnostic).
        icon_fn(ax, cx, box_y + box_h - 0.85, accent)
        ax.text(cx, box_y + box_h - 1.65, head, ha="center", va="center",
                fontsize=13, fontweight="bold", color=NAVY)
        ax.text(cx, box_y + 0.45, sub, ha="center", va="center",
                fontsize=10, color=INK, linespacing=1.5)

    # Arrows between cards.
    for i in range(3):
        cx_left = steps[i][0] + half_w + 0.05
        cx_right = steps[i + 1][0] - half_w - 0.05
        y = box_y + box_h / 2
        _arrow(ax, cx_left, y, cx_right, y, color=SEA, lw=2.5)

    out = DOCS / "infographic_pipeline.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Infographic 3 — Price elasticity curve (for ML dashboard math section)
# ---------------------------------------------------------------------------
def render_elasticity():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    ax.set_facecolor(WHITE)

    elasticity = -0.7
    pct_changes = np.linspace(0, 0.50, 200)
    retention = np.clip(1 + elasticity * pct_changes, 0, 1)

    # Shaded "guests retained" region.
    ax.fill_between(pct_changes * 100, 0, retention * 100,
                    color=UP, alpha=0.18, label="Guests who still book")
    # Shaded "lost to price" region.
    ax.fill_between(pct_changes * 100, retention * 100, 100,
                    color=DOWN, alpha=0.12, label="Guests who walk away")

    ax.plot(pct_changes * 100, retention * 100,
            color=NAVY, linewidth=3.0, label=f"Retention = 1 + η·ΔP/P    (η = {elasticity})")

    # Annotated waypoints.
    waypoints = [(10, 0.93), (20, 0.86), (30, 0.79), (40, 0.72)]
    for pct, ret in waypoints:
        ax.plot(pct, ret * 100, "o", color=NAVY, markersize=8, zorder=5)
        ax.annotate(
            f"+{pct}% price → {int(ret*100)}% retained",
            xy=(pct, ret * 100), xytext=(pct + 4, ret * 100 + 5),
            fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
        )

    # The worked example from the dashboard text.
    ax.annotate(
        "Example: AI prices €185.50 vs static €154.53\n→ +20% price hike → 86% of guests stay,\n   14% walk away",
        xy=(20, 86), xytext=(28, 50),
        fontsize=10.5, color=NAVY, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc=SAND, ec=NAVY, lw=1),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4),
    )

    ax.set_xlabel("Price change vs static rulebook  (Δ P / P)", fontsize=12, color=INK)
    ax.set_ylabel("Bookings retained  (%)", fontsize=12, color=INK)
    ax.set_xlim(0, 50)
    ax.set_ylim(50, 102)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"+{int(v)}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.6, color="#cdd9e3")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cdd9e3")
    ax.spines["bottom"].set_color("#cdd9e3")

    ax.set_title(
        "Price elasticity of demand  —  why higher prices lose bookings\n"
        f"AlgarveMar's assumption: η = {elasticity} (4-star mid-range hotel)",
        fontsize=13, color=NAVY, fontweight="bold", pad=14,
    )

    ax.legend(loc="lower left", fontsize=10, frameon=True,
              facecolor=WHITE, edgecolor="#cdd9e3")

    fig.tight_layout()
    out = DOCS / "infographic_elasticity.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    print("Rendering AlgarveMar infographics ...")
    render_value_prop()
    render_pipeline()
    render_elasticity()
    print("done.")


if __name__ == "__main__":
    main()
