"""Charts for the research presentation.

EDA and the residual forecast are refit with the same specification as
reproduce_v22_figures.py. Walk-forward bars are the stored month scores.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import data
from features import square_feet

OUT = Path(__file__).resolve().parent / "figures"
NAVY = "#1f4e79"
TEAL = "#2f6f6a"
SLATE = "#5c6b7a"
BAND = "#f4e6d4"

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


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def shade(ax):
    ax.axvspan(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-05-01"), color=BAND, zorder=0)


def numeric_design(frame):
    missing_lot = frame["lot_front"].isna()
    X = pd.DataFrame(index=frame.index)
    X["log_list_price"] = np.log(frame["list_price"])
    X["log_square_feet"] = np.log(frame["square_feet"])
    X["lot_missing"] = missing_lot.astype(float)
    X["log_lot_front"] = np.log(frame["lot_front"].where(~missing_lot, 1.0))
    X["lot_depth"] = frame["lot_depth"].fillna(0.0)
    for column in ["bedrooms", "washrooms", "parking_total", "garage", "latitude", "longitude"]:
        X[column] = frame[column].astype(float)
    return X


def eda_and_sarimax():
    listings = data.load("listings").copy()
    listings["square_feet"] = square_feet(listings["house_area"])
    holdout = pd.Timestamp("2025-12-01")
    train = listings[listings["date_sold"] < holdout].copy()

    month = listings["date_sold"].dt.to_period("M")
    med = listings.groupby(month)["sold_price"].median() / 1000
    x = med.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    shade(ax)
    ax.plot(x, med.values, color=NAVY, lw=1.7)
    ax.set_ylabel("Median sold price ($ thousands)")
    ax.set_xlabel("Sale month")
    ax.set_title("Monthly median sold price")
    ax.legend(
        handles=[Patch(facecolor=BAND, edgecolor="none", label="Dec 2025–May 2026")],
        frameon=False, fontsize=9, loc="upper right",
    )
    save(fig, "pres_median_price.png")

    pct = 100 * (listings["sold_price"] - listings["list_price"]) / listings["list_price"]
    by = pct.groupby(month)
    mean, std = by.mean(), by.std()
    x = mean.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    shade(ax)
    ax.fill_between(x, mean - std, mean + std, color=NAVY, alpha=0.07, linewidth=0)
    ax.plot(x, mean.values, color=NAVY, lw=2.4)
    ax.axhline(0, color=SLATE, lw=0.8, ls="--")
    ax.set_ylabel("Mean (sold − list) / list (%)")
    ax.set_xlabel("Sale month")
    ax.set_title("Monthly mean close versus the ask")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=NAVY, lw=2.4, label="Monthly mean"),
            Patch(facecolor=NAVY, alpha=0.18, label="± 1 s.d. across homes"),
            Patch(facecolor=BAND, edgecolor="none", label="Evaluation window"),
        ],
        frameon=False, fontsize=8, loc="lower left",
    )
    save(fig, "pres_close_vs_list.png")

    train_X = sm.add_constant(numeric_design(train), has_constant="add")
    log_model = sm.OLS(np.log(train["sold_price"]), train_X).fit()
    print("elasticity", round(float(log_model.params["log_list_price"]), 3))
    print("r2", round(float(log_model.rsquared), 3))
    monthly = pd.Series(log_model.resid, index=train.index)
    monthly = monthly.groupby(train["date_sold"].dt.to_period("M")).mean()
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.asfreq("MS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(
            monthly.astype(float),
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 12),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)
    forecast = fit.get_forecast(steps=6).predicted_mean
    print("params", {k: round(float(v), 4) for k, v in fit.params.items()})
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    shade(ax)
    ax.plot(monthly.index, monthly.values, color=NAVY, lw=1.6, label="Monthly mean residual")
    ax.plot(fit.fittedvalues.index, fit.fittedvalues.values, color=SLATE, lw=1.2, ls="--", label="SARIMAX fitted")
    ax.plot(forecast.index, forecast.values, color=TEAL, lw=1.7, marker="o", ms=4, label="Six-month forecast")
    ax.axhline(0, color=SLATE, lw=0.6)
    ax.set_ylabel("Mean log residual")
    ax.set_xlabel("Sale month")
    ax.set_title("Log-linear residual and its forecast")
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    save(fig, "pres_sarimax.png")


def grouped(series, title, name):
    months = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    xpos = np.arange(len(months))
    width = 0.36
    ax.bar(xpos - width / 2, series[0][1], width, color=SLATE, label=series[0][0])
    ax.bar(xpos + width / 2, series[1][1], width, color=NAVY, label=series[1][0])
    ax.set_xticks(xpos)
    ax.set_xticklabels(months, rotation=20, ha="right")
    ax.set_ylabel("MdAPE (%)")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save(fig, name)


def walk_forward():
    grouped(
        [
            ("Asking price", [7.46, 7.58, 7.15, 7.10, 7.24, 7.25]),
            ("List-aware", [3.97, 4.16, 4.38, 4.14, 4.37, 4.55]),
        ],
        "List-aware walk-forward MdAPE",
        "pres_la_mdape.png",
    )
    grouped(
        [
            ("12-month anchor", [16.57, 15.43, 15.61, 14.79, 15.02, 15.28]),
            ("Off-market", [7.42, 7.27, 8.17, 7.10, 7.44, 7.39]),
        ],
        "Off-market walk-forward MdAPE",
        "pres_om_mdape.png",
    )


def list_vs_sold():
    listings = data.load("listings")
    corr = float(listings["list_price"].corr(listings["sold_price"]))
    print("corr", round(corr, 3), "n", len(listings))
    limit = float(listings[["list_price", "sold_price"]].stack().quantile(0.995))
    shown = listings[(listings["list_price"] <= limit) & (listings["sold_price"] <= limit)]
    rng = np.random.default_rng(0)
    take = rng.choice(len(shown), size=min(8000, len(shown)), replace=False)
    sample = shown.iloc[take]
    fig, ax = plt.subplots(figsize=(5.15, 4.7))
    ax.scatter(
        sample["list_price"] / 1e6,
        sample["sold_price"] / 1e6,
        s=9, alpha=0.28, c=NAVY, linewidths=0, rasterized=True,
    )
    hi = limit / 1e6
    ax.plot([0, hi], [0, hi], color=TEAL, lw=1.35, ls="--", label="Sold = list", zorder=3)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("List price ($ millions)")
    ax.set_ylabel("Sold price ($ millions)")
    ax.set_title("List price and sold price")
    ax.text(
        0.05, 0.95, f"corr = {corr:.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=13, color=NAVY, fontweight="bold",
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    save(fig, "pres_list_vs_sold.png")


def signed_by_decile():
    frame = pd.read_csv(ROOT / "outputs" / "validation_predictions.csv")
    frame["signed"] = 100 * (frame["predicted_price"] - frame["sold_price"]) / frame["sold_price"]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for track, color, label in (
        ("off_market", NAVY, "Off-market"),
        ("list_aware", TEAL, "List-aware"),
    ):
        part = frame[frame["track"] == track].copy()
        part["decile"] = pd.qcut(part["sold_price"], 10, labels=False) + 1
        med = part.groupby("decile")["signed"].median()
        price = part.groupby("decile")["sold_price"].median() / 1e6
        ax.plot(med.index, med.values, color=color, lw=1.8, marker="o", ms=4.5, label=label)
        print(label, {int(i): round(float(v), 2) for i, v in med.items()})
        print("  prices", {int(i): round(float(v), 2) for i, v in price.items()})
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Sold-price decile (1 = cheapest)")
    ax.set_ylabel("Median signed error (%)")
    ax.set_title("Walk-forward signed error")
    ax.legend(frameon=False)
    save(fig, "pres_signed_decile.png")


if __name__ == "__main__":
    eda_and_sarimax()
    walk_forward()
    signed_by_decile()
