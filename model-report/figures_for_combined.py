"""Plain-language walk-forward charts for the combined report.

The track-report figures keep their version labels. These copies do not.
Numbers are the stored walk-forward scores, not a new fit.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1f4e79"
TEAL = "#2f6f6a"
SLATE = "#5c6b7a"
OUT = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d0d5dd",
    "axes.grid": True,
    "grid.color": "#e6e8ec",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def grouped(series, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    xpos = np.arange(len(MONTHS))
    width = 0.8 / len(series)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for offset, (label, values, color) in zip(offsets, series):
        ax.bar(xpos + offset, values, width, color=color, label=label)
    ax.set_xticks(xpos)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3)
    save(fig, name)


grouped(
    [
        ("List price", [7.46, 7.58, 7.15, 7.10, 7.24, 7.25], SLATE),
        ("No community", [3.98, 4.26, 4.44, 4.18, 4.42, 4.61], TEAL),
        ("List-aware", [3.97, 4.16, 4.38, 4.14, 4.37, 4.55], NAVY),
    ],
    "MdAPE (%)",
    "List-aware walk-forward error",
    "report_la_mdape.png",
)

grouped(
    [
        ("12-month anchor", [16.57, 15.43, 15.61, 14.79, 15.02, 15.28], SLATE),
        ("Gap model", [7.49, 7.44, 7.86, 7.19, 7.64, 7.40], TEAL),
        ("Off-market", [7.42, 7.27, 8.17, 7.10, 7.44, 7.39], NAVY),
    ],
    "MdAPE (%)",
    "Off-market walk-forward error",
    "report_om_mdape.png",
)

la_signed = [1.31, 1.33, 0.20, 0.46, 1.31, 0.92]
om_signed = [0.63, -2.50, -4.44, -1.15, 1.65, 0.98]
fig, ax = plt.subplots(figsize=(9.4, 3.9))
ax.plot(MONTHS, la_signed, color=NAVY, lw=1.8, marker="o", label="List-aware")
ax.plot(MONTHS, om_signed, color=TEAL, lw=1.8, marker="o", label="Off-market")
ax.axhline(0, color=SLATE, lw=0.8)
ax.set_ylabel("Median signed error (%)")
ax.set_title("Walk-forward bias by sale month")
ax.legend(frameon=False, ncol=2)
save(fig, "report_signed.png")


def gain_chart(pairs, title, name, figsize):
    labels = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, values, color=NAVY, height=0.72)
    ax.set_xlabel("Share of split gain (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.22)
    for y, v in enumerate(values):
        ax.text(v + 0.35, y, f"{v:.1f}", va="center", ha="left", fontsize=11, color=NAVY)
    ax.tick_params(axis="y", labelsize=12, length=0)
    ax.tick_params(axis="x", labelsize=10)
    save(fig, name)


gain_chart(
    [
        ("Peer ask", 21.6),
        ("Community", 18.6),
        ("Log ask", 9.6),
        ("City lag", 9.4),
        ("Recent discount", 7.5),
        ("Month factor", 6.3),
        ("Log size", 3.2),
    ],
    "List-aware",
    "report_la_gain.png",
    (6.4, 3.6),
)

gain_chart(
    [
        ("Lot missing", 33.3),
        ("Building price", 17.8),
        ("Log size", 12.9),
        ("Community", 12.4),
        ("Log frontage", 8.3),
        ("Washrooms", 5.3),
        ("Lot depth", 3.0),
        ("Community $/sqft", 2.0),
        ("Remarks score", 0.7),
    ],
    "Off-market",
    "report_om_gain.png",
    (6.4, 4.2),
)

fields = ["Lot frontage", "Lot depth", "External estimate", "Every other field"]
blank = [58.2, 58.2, 0.9, 0.0]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.barh(fields[::-1], blank[::-1], color=NAVY, height=0.62)
ax.set_xlabel("Share of the 150,000 training rows left blank")
ax.set_xlim(0, 78)
ax.set_title("Which fields are missing")
for y, v in enumerate(blank[::-1]):
    ax.text(v + 1.4, y, f"{v:.1f}%", va="center", ha="left", fontsize=10, color=NAVY)
ax.tick_params(axis="y", length=0)
save(fig, "report_lot_missing.png")
