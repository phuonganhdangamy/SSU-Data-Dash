"""Reproduce developmental diagnostics and figures for list-aware model LA-V22.

Headline walk-forward scores are taken from notebooks/list_aware_models.ipynb.
This script refits the single December 2025 holdout once, for segmented error,
calibration, and feature-gain charts, and redraws the EDA charts the report uses.

Run from the repository root:

    py -3 model-report/reproduce_v22_figures.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import data
from features import square_feet

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NAVY = "#1f4e79"
TEAL = "#2f6f6a"
AMBER = "#c47b2b"
SLATE = "#5c6b7a"
RED = "#8c3a3a"

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


def percentage_scores(predicted, sold):
    signed = 100 * (np.asarray(predicted) - np.asarray(sold)) / np.asarray(sold)
    absolute = np.abs(signed)
    return {
        "MdAPE": round(float(np.median(absolute)), 2),
        "MAPE": round(float(np.mean(absolute)), 2),
        "median_signed_error": round(float(np.median(signed)), 2),
        "p90_APE": round(float(np.quantile(absolute, 0.90)), 2),
        "p95_APE": round(float(np.quantile(absolute, 0.95)), 2),
    }


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


def month_ord(dates):
    return (dates.dt.year * 12 + dates.dt.month).to_numpy()


def past_median_table(reference, values, group_columns):
    history = reference.loc[:, group_columns].copy()
    history["month_ord"] = month_ord(reference["date_sold"])
    history["value"] = np.asarray(values, dtype=float)
    history = history.dropna()
    query_ords = np.arange(int(history["month_ord"].min()), int(history["month_ord"].max()) + 8)
    pieces = []
    for key, group in history.groupby(group_columns, observed=True, sort=False):
        group = group.sort_values("month_ord")
        sale_ords = group["month_ord"].to_numpy()
        vals = group["value"].to_numpy()
        ends = np.searchsorted(sale_ords, query_ords, side="left")
        medians = np.full(len(query_ords), np.nan)
        last_end = -1
        last_med = np.nan
        for i, end in enumerate(ends):
            end = int(end)
            if end == 0:
                continue
            if end != last_end:
                last_med = float(np.median(vals[:end]))
                last_end = end
            medians[i] = last_med
        key = key if isinstance(key, tuple) else (key,)
        piece = {"month_ord": query_ords, "median": medians, "count": ends}
        for column, value in zip(group_columns, key):
            piece[column] = value
        pieces.append(pd.DataFrame(piece))
    return pd.concat(pieces, ignore_index=True)


def lookup_past(query, table, group_columns):
    keys = query.loc[:, list(group_columns)].copy()
    keys["month_ord"] = month_ord(query["date_listed"])
    merged = keys.merge(table, on=list(group_columns) + ["month_ord"], how="left")
    if len(merged) != len(query):
        raise ValueError("past-median lookup changed the number of rows")
    median = merged["median"].to_numpy(dtype=float)
    count = merged["count"].fillna(0).to_numpy(dtype=float)
    return median, count


def recent_ratio_table(reference, group_columns, window=3):
    history = reference.loc[:, group_columns].copy()
    history["month_ord"] = month_ord(reference["date_sold"])
    history["value"] = (reference["sold_price"] / reference["list_price"]).to_numpy(dtype=float)
    history = history.dropna()
    query_ords = np.arange(int(history["month_ord"].min()), int(history["month_ord"].max()) + 8)
    pieces = []
    for key, group in history.groupby(group_columns, observed=True, sort=False):
        group = group.sort_values("month_ord")
        sale_ords = group["month_ord"].to_numpy()
        vals = group["value"].to_numpy()
        right = np.searchsorted(sale_ords, query_ords, side="left")
        left = np.searchsorted(sale_ords, query_ords - window, side="left")
        counts = right - left
        medians = np.full(len(query_ords), np.nan)
        for i, (lo, hi, count) in enumerate(zip(left, right, counts)):
            if count > 0:
                medians[i] = float(np.median(vals[int(lo):int(hi)]))
        key = key if isinstance(key, tuple) else (key,)
        piece = {"month_ord": query_ords, "median": medians, "count": counts}
        for column, value in zip(group_columns, key):
            piece[column] = value
        pieces.append(pd.DataFrame(piece))
    return pd.concat(pieces, ignore_index=True)


def save(fig, name):
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    listings = data.load("listings").copy()
    listings["square_feet"] = square_feet(listings["house_area"])
    holdout_start = pd.Timestamp("2025-12-01")
    train = listings[listings["date_sold"] < holdout_start].copy()
    valid = listings[listings["date_sold"] >= holdout_start].copy()

    ratio = listings["sold_price"] / listings["list_price"]
    pct = 100 * (listings["sold_price"] - listings["list_price"]) / listings["list_price"]
    by_month = pct.groupby(listings["date_sold"].dt.to_period("M"))
    mean = by_month.mean()
    std = by_month.std()
    x = mean.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.fill_between(x, mean - std, mean + std, color=NAVY, alpha=0.15, linewidth=0, label="Mean ± 1 s.d.")
    ax.plot(x, mean, color=NAVY, lw=1.8, label="Monthly mean")
    ax.axhline(0, color=SLATE, lw=0.8, ls="--")
    ax.set_ylabel("Sold price vs list (%)")
    ax.set_xlabel("Sale month")
    ax.set_title("Close relative to asking price")
    ax.legend(frameon=False)
    save(fig, "monthly_close_vs_list.png")

    limit = listings[["list_price", "sold_price"]].stack().quantile(0.995)
    shown = listings[(listings["list_price"] <= limit) & (listings["sold_price"] <= limit)]
    rng = np.random.default_rng(0)
    take = rng.choice(len(shown), size=min(8000, len(shown)), replace=False)
    sample = shown.iloc[take]
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(sample["list_price"] / 1e6, sample["sold_price"] / 1e6, s=8, alpha=0.25, c=NAVY, linewidths=0)
    ax.plot([0, limit / 1e6], [0, limit / 1e6], color=AMBER, lw=1.4, label="Sold = list")
    ax.set_xlim(0, limit / 1e6)
    ax.set_ylim(0, limit / 1e6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("List price ($ millions)")
    ax.set_ylabel("Sold price ($ millions)")
    ax.set_title("List price against sold price")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "list_vs_sold.png")

    # Upstream log-linear model and the listing-month factor.
    train_X = sm.add_constant(numeric_design(train), has_constant="add")
    log_model = sm.OLS(np.log(train["sold_price"]), train_X).fit()
    log_resid = pd.Series(log_model.resid, index=train.index)
    monthly = log_resid.groupby(train["date_sold"].dt.to_period("M")).mean()
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.asfreq("MS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        residual_model = SARIMAX(
            monthly.astype(float),
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 12),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=200)
    forecast = residual_model.get_forecast(steps=6).predicted_mean

    def list_month_factor(frame):
        listed = frame["date_listed"].dt.to_period("M").dt.to_timestamp()
        previous = listed - pd.DateOffset(months=1)
        values = listed.map(forecast)
        lagged = previous.map(monthly)
        return values.fillna(lagged).fillna(0.0).to_numpy()

    log_ratio = np.log(train["sold_price"] / train["list_price"])
    ratio_by_month = log_ratio.groupby(train["date_sold"].dt.to_period("M")).median()
    lag_by_list_month = ratio_by_month.shift(1)
    last_train_month = ratio_by_month.index.max()
    latest_ratio = float(ratio_by_month.iloc[-1])

    def market_lag(frame):
        keys = frame["date_listed"].dt.to_period("M")
        lag = keys.map(lag_by_list_month).astype(float)
        lag = lag.mask(keys > last_train_month, latest_ratio)
        return lag.fillna(0.0).to_numpy()

    def tree_frame(frame):
        X = numeric_design(frame)
        listed_month = frame["date_listed"].dt.month.to_numpy()
        X["listed_month_sin"] = np.sin(2 * np.pi * listed_month / 12)
        X["listed_month_cos"] = np.cos(2 * np.pi * listed_month / 12)
        X["market_lag"] = market_lag(frame)
        return X

    category_levels = {
        column: pd.Index(train[column].dropna().unique())
        for column in ["house_type_name", "house_style", "municipality"]
    }

    def home_attributes(frame):
        X = tree_frame(frame)
        X["lot_area"] = frame["lot_front"] * frame["lot_depth"]
        X["area_is_range"] = frame["house_area"].astype(str).str.contains("-", regex=False).astype(float)
        for column, levels in category_levels.items():
            X[column] = pd.Categorical(frame[column], categories=levels)
        return X

    sold_pps = (train["sold_price"] / train["square_feet"]).to_numpy()
    peer_table = past_median_table(train, sold_pps, ["community", "house_type_name"])
    type_pps_table = past_median_table(train, sold_pps, ["house_type_name"])

    def peer_ask(frame):
        list_pps = (frame["list_price"] / frame["square_feet"]).to_numpy(dtype=float)
        type_median, _ = lookup_past(frame, type_pps_table, ["house_type_name"])
        community_median, community_count = lookup_past(frame, peer_table, ["community", "house_type_name"])
        chosen = np.where(community_count >= 30, community_median, type_median)
        out = np.full(len(frame), np.nan)
        ok = np.isfinite(chosen) & (chosen > 0)
        out[ok] = list_pps[ok] / chosen[ok]
        return out, community_count

    def design(frame):
        X = home_attributes(frame)
        X["peer_ask"], peer_count = peer_ask(frame)
        X["list_month_factor"] = list_month_factor(frame)
        return X, peer_count

    community_ratios = recent_ratio_table(train, ["community", "house_type_name"])
    type_ratios = recent_ratio_table(train, ["house_type_name"])

    def recent_discount(frame):
        community_median, community_count = lookup_past(frame, community_ratios, ["community", "house_type_name"])
        type_median, type_count = lookup_past(frame, type_ratios, ["house_type_name"])
        chosen = np.where(
            community_count >= 30,
            community_median,
            np.where(type_count >= 30, type_median, np.nan),
        )
        return chosen, community_count, type_count

    x_train, train_peer_count = design(train)
    x_valid, valid_peer_count = design(valid)
    train_discount, train_comm_count, train_type_count = recent_discount(train)
    valid_discount, valid_comm_count, valid_type_count = recent_discount(valid)
    x_train["recent_discount"] = train_discount
    x_valid["recent_discount"] = valid_discount
    community_levels = pd.Index(train["community"].dropna().unique())
    x_train["community"] = pd.Categorical(train["community"], categories=community_levels)
    x_valid["community"] = pd.Categorical(valid["community"], categories=community_levels)

    categories = ["house_type_name", "house_style", "municipality", "community"]
    gap = np.log(train["sold_price"].to_numpy() / train["list_price"].to_numpy())
    model = LGBMRegressor(
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=-1,
        verbosity=-1,
        num_leaves=127,
        learning_rate=0.03,
        n_estimators=700,
    )
    model.fit(x_train, gap, categorical_feature=categories)
    pred = valid["list_price"].to_numpy() * np.exp(model.predict(x_valid))
    sold = valid["sold_price"].to_numpy()
    dev_scores = percentage_scores(pred, sold)
    list_scores = percentage_scores(valid["list_price"], sold)
    print("developmental V22", dev_scores)
    print("developmental list", list_scores)

    signed = 100 * (pred - sold) / sold
    ape = np.abs(signed)
    valid = valid.copy()
    valid["pred"] = pred
    valid["ape"] = ape
    valid["signed"] = signed
    valid["pred_ratio"] = pred / valid["list_price"].to_numpy()
    valid["actual_ratio"] = sold / valid["list_price"].to_numpy()

    # Walk-forward list-price benchmark, same months as the notebook.
    walk_rows = []
    list_parts = []
    sold_parts = []
    for month in pd.period_range("2025-12", "2026-05", freq="M"):
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        part_sold = homes["sold_price"].to_numpy()
        part_list = homes["list_price"].to_numpy()
        list_parts.append(part_list)
        sold_parts.append(part_sold)
        walk_rows.append({"month": str(month), "n": int(len(homes)), **percentage_scores(part_list, part_sold)})
    walk_list = percentage_scores(np.concatenate(list_parts), np.concatenate(sold_parts))
    print("walk-forward list", walk_list)

    decile = pd.qcut(valid["sold_price"], 10, labels=[f"D{i}" for i in range(1, 11)])
    decile_table = (
        valid.groupby(decile, observed=True)
        .agg(
            n=("sold_price", "size"),
            sold_min=("sold_price", "min"),
            sold_max=("sold_price", "max"),
            MdAPE=("ape", "median"),
            MAPE=("ape", "mean"),
            signed=("signed", "median"),
        )
        .reset_index(names="decile")
    )
    for column in ["MdAPE", "MAPE", "signed", "sold_min", "sold_max"]:
        decile_table[column] = decile_table[column].round(2 if column in {"MdAPE", "MAPE", "signed"} else 0)

    type_table = (
        valid.groupby("house_type_name")
        .agg(n=("ape", "size"), MdAPE=("ape", "median"), MAPE=("ape", "mean"), signed=("signed", "median"))
        .sort_values("n", ascending=False)
        .reset_index()
    )
    for column in ["MdAPE", "MAPE", "signed"]:
        type_table[column] = type_table[column].round(2)

    muni_table = (
        valid.groupby("municipality")
        .agg(n=("ape", "size"), MdAPE=("ape", "median"), signed=("signed", "median"))
        .sort_values("n", ascending=False)
        .reset_index()
    )
    for column in ["MdAPE", "signed"]:
        muni_table[column] = muni_table[column].round(2)

    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=model.feature_name_)
    gain = (100 * gain / gain.sum()).sort_values(ascending=False).round(2)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(decile_table["decile"], decile_table["MdAPE"], color=NAVY, width=0.72)
    ax.axhline(dev_scores["MdAPE"], color=AMBER, lw=1.2, ls="--", label=f"Overall MdAPE {dev_scores['MdAPE']:.2f}%")
    ax.set_ylabel("MdAPE (%)")
    ax.set_xlabel("Sold-price decile (D1 lowest)")
    ax.set_title("Developmental absolute error by price decile")
    ax.legend(frameon=False)
    save(fig, "error_by_decile.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    order = type_table.sort_values("MdAPE")
    ax.barh(order["house_type_name"], order["MdAPE"], color=TEAL)
    ax.axvline(dev_scores["MdAPE"], color=AMBER, lw=1.2, ls="--")
    ax.set_xlabel("MdAPE (%)")
    ax.set_title("Developmental absolute error by property type")
    save(fig, "error_by_type.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(signed, bins=60, range=(-30, 30), color=NAVY, edgecolor="white")
    ax.axvline(0, color=SLATE, lw=0.8, ls="--")
    ax.axvline(np.median(signed), color=AMBER, lw=1.4, label=f"Median signed {np.median(signed):+.2f}%")
    ax.set_xlabel("Signed percentage error  (predicted − sold) / sold")
    ax.set_ylabel("Validation sales")
    ax.set_title("Developmental error distribution")
    ax.legend(frameon=False)
    save(fig, "error_distribution.png")

    valid["bin"] = pd.qcut(valid["pred_ratio"], 10, duplicates="drop")
    cal = valid.groupby("bin", observed=True).agg(
        predicted=("pred_ratio", "median"),
        actual=("actual_ratio", "median"),
        n=("pred_ratio", "size"),
    )
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.plot(cal["predicted"], cal["actual"], "o-", color=NAVY, lw=1.4, ms=6)
    lo = min(cal["predicted"].min(), cal["actual"].min()) - 0.01
    hi = max(cal["predicted"].max(), cal["actual"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=AMBER, lw=1.2, label="Predicted = realized")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Median predicted sold / list")
    ax.set_ylabel("Median realized sold / list")
    ax.set_title("Calibration of the close ratio")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "calibration_ratio.png")

    top = gain.head(12).sort_values()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(top.index, top.values, color=NAVY)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("LightGBM gain, developmental V22 fit")
    save(fig, "feature_gain.png")

    months = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
    v22 = [3.97, 4.16, 4.38, 4.14, 4.37, 4.55]
    v21 = [3.98, 4.26, 4.44, 4.18, 4.42, 4.61]
    list_m = [row["MdAPE"] for row in walk_rows]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    xpos = np.arange(len(months))
    width = 0.26
    ax.bar(xpos - width, list_m, width, color=SLATE, label="List price")
    ax.bar(xpos, v21, width, color=TEAL, label="V21")
    ax.bar(xpos + width, v22, width, color=NAVY, label="V22")
    ax.set_xticks(xpos)
    ax.set_xticklabels(months)
    ax.set_ylabel("MdAPE (%)")
    ax.set_title("Walk-forward median absolute percentage error")
    ax.legend(frameon=False, ncol=3)
    save(fig, "walkforward_mdape.png")

    bias = [1.31, 1.33, 0.20, 0.46, 1.31, 0.92]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    colors = [NAVY if value >= 0 else RED for value in bias]
    ax.bar(months, bias, color=colors)
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_ylabel("Median signed error (%)")
    ax.set_title("V22 walk-forward bias by sale month")
    save(fig, "walkforward_bias.png")

    coverage = {
        "train_peer_community_share": round(float((train_peer_count >= 30).mean()), 4),
        "valid_peer_community_share": round(float((valid_peer_count >= 30).mean()), 4),
        "train_discount_community_share": round(float((train_comm_count >= 30).mean()), 4),
        "train_discount_type_share": round(float(((train_comm_count < 30) & (train_type_count >= 30)).mean()), 4),
        "train_discount_missing_share": round(float(((train_comm_count < 30) & (train_type_count < 30)).mean()), 4),
        "valid_discount_community_share": round(float((valid_comm_count >= 30).mean()), 4),
        "valid_discount_type_share": round(float(((valid_comm_count < 30) & (valid_type_count >= 30)).mean()), 4),
        "valid_discount_missing_share": round(float(((valid_comm_count < 30) & (valid_type_count < 30)).mean()), 4),
        "valid_community_unseen": round(float(x_valid["community"].isna().mean()), 4),
        "list_month_factor_valid_median": round(float(np.median(x_valid["list_month_factor"])), 4),
    }

    summary = {
        "n_listings": int(len(listings)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_community": int(listings["community"].nunique()),
        "n_style": int(listings["house_style"].nunique()),
        "n_municipality": int(listings["municipality"].nunique()),
        "n_type": int(listings["house_type_name"].nunique()),
        "corr_list_sold": round(float(listings["list_price"].corr(listings["sold_price"])), 3),
        "median_pct_vs_list": round(float(pct.median()), 2),
        "mean_pct_vs_list": round(float(pct.mean()), 2),
        "std_pct_vs_list": round(float(pct.std()), 2),
        "ratio_p10": round(float(ratio.quantile(0.10)), 3),
        "ratio_p50": round(float(ratio.quantile(0.50)), 3),
        "ratio_p90": round(float(ratio.quantile(0.90)), 3),
        "share_below_list": round(float((ratio < 1).mean()), 4),
        "share_above_list": round(float((ratio > 1).mean()), 4),
        "ols_r2": round(float(log_model.rsquared), 3),
        "ols_list_elasticity": round(float(log_model.params["log_list_price"]), 3),
        "resid_skew": round(float(pd.Series(log_model.resid).skew()), 2),
        "dev_v22": dev_scores,
        "dev_list": list_scores,
        "walk_list": walk_list,
        "walk_list_months": walk_rows,
        "coverage": coverage,
        "gain": gain.round(2).to_dict(),
        "forecast_log_residual": {str(k.date()): round(float(v), 4) for k, v in forecast.items()},
        "type_mix_train": train["house_type_name"].value_counts(normalize=True).round(3).to_dict(),
        "deciles": decile_table.to_dict(orient="records"),
        "types": type_table.to_dict(orient="records"),
        "municipalities": muni_table.to_dict(orient="records"),
    }
    out = Path(__file__).resolve().parent / "v22_diagnostics.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
