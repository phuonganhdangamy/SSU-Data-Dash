"""Developmental diagnostics for off-market model OM-V9.

The adopted specification is the V9 direct log-price model:
remarks score, shift weight 1, 12-month community price, extra remark flags.
Walk-forward scores in the report are the stored notebook outputs.
This script refits the December 2025 holdout once.

    py -3 model-report/reproduce_v9_figures.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

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

GAP_CATEGORIES = ["house_type_name", "house_style", "community", "municipality"]
PHRASE_PATTERNS = [
    ("as_is", r"as-is|as is"),
    ("turnkey", r"turnkey"),
    ("potential", r"potential"),
    ("original", r"original"),
    ("finished", r"finished"),
    ("renovated", r"renovat"),
    ("needs_work", r"needs work"),
    ("stunning", r"stunning|beautiful|gorgeous|spectacular"),
]
EXTRA_PHRASES = PHRASE_PATTERNS + [
    ("investor", r"\binvestors?\b"),
    ("motivated", r"\bmotivated\b"),
    ("redone", r"\bredone\b|\bredo\b"),
    ("dated", r"\bdated\b"),
    ("worn", r"\bworn\b"),
]
LGB = dict(
    num_leaves=127,
    learning_rate=0.03,
    n_estimators=700,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=-1,
    verbosity=-1,
)
TEXT_ALPHA = 0.3
WINDOW = 12


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


def save(fig, name):
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.name, flush=True)


def numeric_design(frame):
    missing_lot = frame["lot_front"].isna()
    X = pd.DataFrame(index=frame.index)
    X["log_square_feet"] = np.log(frame["square_feet"])
    X["lot_missing"] = missing_lot.astype(float)
    X["log_lot_front"] = np.log(frame["lot_front"].where(~missing_lot, 1.0))
    X["lot_depth"] = frame["lot_depth"].fillna(0.0)
    for column in ["bedrooms", "washrooms", "parking_total", "garage", "latitude", "longitude"]:
        X[column] = frame[column].astype(float)
    return X


def tree_frame(frame):
    X = numeric_design(frame)
    listed_month = frame["date_listed"].dt.month.to_numpy()
    X["listed_month_sin"] = np.sin(2 * np.pi * listed_month / 12)
    X["listed_month_cos"] = np.cos(2 * np.pi * listed_month / 12)
    return X


def month_ord(dates):
    return (dates.dt.year * 12 + dates.dt.month).to_numpy()


def recent_psf_table(reference, group_columns, window):
    history = reference.loc[:, group_columns].copy()
    history["month_ord"] = month_ord(reference["date_sold"])
    history["value"] = (reference["sold_price"] / reference["square_feet"]).to_numpy(dtype=float)
    history = history.replace([np.inf, -np.inf], np.nan).dropna()
    last_month = int(history["month_ord"].max())
    query_ords = np.arange(int(history["month_ord"].min()) - 24, last_month + 18)
    pieces = []
    for key, group in history.groupby(group_columns, observed=True, sort=False):
        group = group.sort_values("month_ord")
        sale_ords = group["month_ord"].to_numpy()
        values = group["value"].to_numpy()
        medians = np.full(len(query_ords), np.nan)
        counts = np.zeros(len(query_ords), dtype=float)
        for i, query in enumerate(query_ords):
            end = int(np.searchsorted(sale_ords, query, side="left"))
            if end == 0:
                continue
            start = int(np.searchsorted(sale_ords, query - window, side="left"))
            if start >= end:
                last = int(sale_ords[end - 1])
                start = int(np.searchsorted(sale_ords, last - window + 1, side="left"))
            medians[i] = float(np.median(values[start:end]))
            counts[i] = end - start
        key = key if isinstance(key, tuple) else (key,)
        piece = {"month_ord": query_ords, "median": medians, "count": counts}
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


def psf_bundle(reference, window):
    usable = reference[(reference["sold_price"] > 0) & (reference["square_feet"] > 0)].copy()
    usable["_all"] = "all"
    return {
        "frame": usable,
        "fallback": float(np.median(usable["sold_price"] / usable["square_feet"])),
        "community": recent_psf_table(usable, ["community", "house_type_name"], window),
        "type": recent_psf_table(usable, ["house_type_name"], window),
        "city": recent_psf_table(usable, ["_all"], window),
    }


def price_level(frame, bundle):
    query = frame.copy()
    query["_all"] = "all"
    community_median, community_count = lookup_past(query, bundle["community"], ["community", "house_type_name"])
    type_median, type_count = lookup_past(query, bundle["type"], ["house_type_name"])
    city_median, _ = lookup_past(query, bundle["city"], ["_all"])
    level = np.where(community_count >= 30, community_median, np.where(type_count >= 30, type_median, city_median))
    level = np.where(np.isfinite(level) & (level > 0), level, bundle["fallback"])
    return level, community_count >= 30


def city_bundle(reference, window):
    usable = reference[(reference["sold_price"] > 0) & (reference["square_feet"] > 0)].copy()
    usable["_all"] = "all"
    return {
        "fallback": float(np.median(usable["sold_price"] / usable["square_feet"])),
        "city": recent_psf_table(usable, ["_all"], window),
    }


def city_level(frame, bundle):
    query = frame.copy()
    query["_all"] = "all"
    median, _ = lookup_past(query, bundle["city"], ["_all"])
    return np.where(np.isfinite(median) & (median > 0), median, bundle["fallback"])


def category_levels(reference):
    return {column: pd.Index(reference[column].dropna().unique()) for column in GAP_CATEGORIES}


def prepare(reference):
    return psf_bundle(reference, WINDOW), city_bundle(reference, 3), city_bundle(reference, 12), category_levels(reference)


def _pin_key(frame):
    lat = np.round(frame["latitude"].to_numpy(dtype=float), 6)
    lon = np.round(frame["longitude"].to_numpy(dtype=float), 6)
    typ = frame["house_type_name"].astype(str).to_numpy()
    return pd.Series([f"{a:.6f}|{b:.6f}|{c}" for a, b, c in zip(lat, lon, typ)], index=frame.index)


def _index_tables(types, ords, log_psf):
    frame = pd.DataFrame({"typ": types, "ord": ords, "log_psf": log_psf})
    monthly = frame.groupby(["typ", "ord"], sort=False)["log_psf"].median()
    tables = {}
    for typ, series in monthly.groupby(level=0):
        values = series.droplevel(0).sort_index()
        span = np.arange(int(values.index.min()), int(values.index.max()) + 1 + 24)
        level = values.reindex(span).ffill()
        known = level.rolling(3, min_periods=1).median().shift(1)
        tables[typ] = (span.astype(np.int32), known.to_numpy(dtype=float))
    return tables


def _index_at(tables, types, ords):
    out = np.full(len(ords), np.nan, dtype=float)
    if len(ords) == 0:
        return out
    type_arr = np.asarray(types)
    order = np.argsort(type_arr, kind="mergesort")
    sorted_types = type_arr[order]
    breaks = np.flatnonzero(np.r_[True, sorted_types[1:] != sorted_types[:-1], True])
    for start, stop in zip(breaks[:-1], breaks[1:]):
        typ = sorted_types[start]
        table = tables.get(typ)
        if table is None:
            continue
        span, known = table
        pos = order[start:stop]
        query = ords[pos]
        loc = np.searchsorted(span, query)
        in_span = (loc < len(span)) & (loc >= 0) & (span[np.clip(loc, 0, len(span) - 1)] == query)
        out[pos[in_span]] = known[loc[in_span]]
        after = query > span[-1]
        if np.any(after):
            finite = np.isfinite(known)
            if np.any(finite):
                out[pos[after]] = known[np.flatnonzero(finite)[-1]]
    return out


def _groups(keys, dates, factor, ml_num):
    order = np.lexsort((dates, keys))
    keys, dates, factor, ml_num = keys[order], dates[order], factor[order], ml_num[order]
    groups = {}
    if len(keys) == 0:
        return groups
    breaks = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    for start, stop in zip(breaks[:-1], breaks[1:]):
        groups[keys[start]] = (dates[start:stop], factor[start:stop], ml_num[start:stop])
    return groups


def building_history(reference):
    usable = reference[(reference["sold_price"] > 0) & (reference["square_feet"] > 0)].copy()
    usable = usable.dropna(subset=["latitude", "longitude", "house_type_name", "date_sold", "ml_num"])
    usable = usable.sort_values(["date_sold", "ml_num"])
    psf = usable["sold_price"].to_numpy(dtype=float) / usable["square_feet"].to_numpy(dtype=float)
    log_psf = np.log(psf)
    ords = (usable["date_sold"].dt.year * 12 + usable["date_sold"].dt.month).to_numpy(dtype=np.int32)
    types = usable["house_type_name"].astype(str).to_numpy()
    tables = _index_tables(types, ords, log_psf)
    sale_index = _index_at(tables, types, ords)
    keep = np.isfinite(sale_index) & np.isfinite(psf) & (psf > 0)
    usable = usable.loc[keep]
    factor = psf[keep] * np.exp(-sale_index[keep])
    dates = usable["date_sold"].to_numpy(dtype="datetime64[ns]")
    ml_num = usable["ml_num"].astype(str).to_numpy()
    return {"index": tables, "building": _groups(_pin_key(usable).to_numpy(), dates, factor, ml_num), "ids": set(ml_num)}


def building_features(homes, history):
    n = len(homes)
    count = np.zeros(n, dtype=np.int32)
    median_psf = np.full(n, np.nan)
    listed = homes["date_listed"].to_numpy(dtype="datetime64[ns]")
    mls = homes["ml_num"].astype(str).to_numpy()
    list_ord = (homes["date_listed"].dt.year * 12 + homes["date_listed"].dt.month).to_numpy(dtype=np.int32)
    index_now = _index_at(history["index"], homes["house_type_name"].astype(str).to_numpy(), list_ord)
    keys = _pin_key(homes).to_numpy()
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    if len(sorted_keys):
        breaks = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True])
        groups = history["building"]
        for start, stop in zip(breaks[:-1], breaks[1:]):
            group = groups.get(sorted_keys[start])
            if group is None:
                continue
            g_dates, g_factor, g_ml = group
            pos = order[start:stop]
            rights = np.searchsorted(g_dates, listed[pos], side="left")
            for j, right in enumerate(rights):
                p = pos[j]
                if right <= 0 or not np.isfinite(index_now[p]):
                    continue
                keep = g_ml[:right] != mls[p]
                if not np.any(keep):
                    continue
                kept_factor = g_factor[:right][keep]
                finite = np.isfinite(kept_factor)
                if not np.any(finite):
                    continue
                kept_factor = kept_factor[finite]
                count[p] = len(kept_factor)
                median_psf[p] = np.exp(index_now[p]) * float(np.median(kept_factor))
    return pd.DataFrame({"count": count, "median_psf": median_psf}, index=homes.index)


def _remarks_cleaner(communities):
    names = sorted({str(name).lower() for name in communities if len(str(name)) >= 4}, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b") if names else None

    def clean(text):
        text = re.sub(r"\d[\d,\.]*", " NUM ", str(text).lower())
        if pattern is not None:
            text = pattern.sub(" COMMUNITY ", text)
        return text

    return clean


def _remarks_vectorizer(communities):
    return TfidfVectorizer(
        preprocessor=_remarks_cleaner(communities),
        ngram_range=(1, 2),
        min_df=40,
        max_features=3000,
        token_pattern=r"(?u)\b[a-zA-Z]+\b",
        sublinear_tf=True,
    )


def _oof_by_time(texts, communities, target, dates):
    order = np.argsort(dates, kind="mergesort")
    score = np.full(len(target), np.nan)
    n = len(target)
    edges = [int(n * fraction) for fraction in (0.40, 0.55, 0.70, 0.85, 1.00)]
    start = edges[0]
    for end in edges[1:]:
        train_rows = order[:start]
        test_rows = order[start:end]
        if len(train_rows) < 1000 or len(test_rows) == 0:
            start = end
            continue
        vectorizer = _remarks_vectorizer(communities[train_rows])
        model = Ridge(alpha=TEXT_ALPHA)
        model.fit(vectorizer.fit_transform(texts[train_rows]), target[train_rows])
        score[test_rows] = model.predict(vectorizer.transform(texts[test_rows]))
        start = end
    return score


def fit_remarks(history):
    texts = history["remarks"].fillna("").astype(str).to_numpy()
    communities = history["community"].astype(str).to_numpy()
    target = np.log(history["sold_price"].to_numpy(dtype=float))
    dates = history["date_sold"].to_numpy(dtype="datetime64[ns]")
    oof = _oof_by_time(texts, communities, target, dates)
    vectorizer = _remarks_vectorizer(communities)
    ridge = Ridge(alpha=TEXT_ALPHA)
    ridge.fit(vectorizer.fit_transform(texts), target)
    return {"vectorizer": vectorizer, "ridge": ridge, "oof": pd.Series(oof, index=history.index), "ids": set(history["ml_num"].astype(str))}


def apply_remarks(model, homes):
    if homes.index.equals(model["oof"].index):
        return model["oof"]
    if set(homes["ml_num"].astype(str)) & model["ids"]:
        raise RuntimeError("remarks score uses a sale to predict itself")
    matrix = model["vectorizer"].transform(homes["remarks"].fillna("").astype(str))
    return pd.Series(model["ridge"].predict(matrix), index=homes.index)


def direct_matrix(frame, design_pack, building, text):
    bundle, city3, city12, levels = design_pack
    level, _ = price_level(frame, bundle)
    X = tree_frame(frame)
    X["log_community_psf"] = np.log(level)
    X["log_city_psf_3"] = np.log(city_level(frame, city3))
    X["log_city_psf_12"] = np.log(city_level(frame, city12))
    for column in GAP_CATEGORIES:
        X[column] = pd.Categorical(frame[column].to_numpy(), categories=levels[column])
    remarks = frame["remarks"].fillna("").str.lower()
    for name, pattern in EXTRA_PHRASES:
        X[name] = remarks.str.contains(pattern, regex=True).astype(float).to_numpy()
    matched = building.loc[frame.index]
    count = matched["count"].to_numpy(dtype=float)
    median = matched["median_psf"].to_numpy(dtype=float)
    sqft = frame["square_feet"].to_numpy(dtype=float)
    building_price = np.full(len(frame), np.nan)
    ok = (count >= 1) & np.isfinite(median) & (median > 0) & np.isfinite(sqft) & (sqft > 0)
    building_price[ok] = np.log(median[ok] * sqft[ok])
    X["building_sales"] = count
    X["building_log_price"] = building_price
    if text is not None:
        X["remarks_score"] = text.loc[frame.index].to_numpy(dtype=float)
    return X


def direct_fit(frame, design_pack, building, text):
    model = LGBMRegressor(**LGB)
    model.fit(direct_matrix(frame, design_pack, building, text), np.log(frame["sold_price"].to_numpy(dtype=float)), categorical_feature=GAP_CATEGORIES)
    return model


def direct_predict(model, frame, design_pack, building, text):
    return np.exp(model.predict(direct_matrix(frame, design_pack, building, text)))


def main():
    listings = data.load("listings").copy()
    listings["square_feet"] = square_feet(listings["house_area"])
    test = data.load("off_market").copy()
    test["square_feet"] = square_feet(test["house_area"])
    holdout = pd.Timestamp("2025-12-01")
    train = listings[listings["date_sold"] < holdout].copy()
    valid = listings[listings["date_sold"] >= holdout].copy()
    print(f"train {len(train):,} valid {len(valid):,} test {len(test):,}", flush=True)

    monthly = listings.groupby(listings["date_sold"].dt.to_period("M"))["sold_price"].median()
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    ax.plot(monthly.index, monthly / 1e3, color=NAVY, lw=1.8)
    ax.axvspan(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-05-31"), color=AMBER, alpha=0.15, label="Validation window")
    ax.set_ylabel("Median sold price ($ thousands)")
    ax.set_xlabel("Sale month")
    ax.set_title("Price level the off-market model has to carry")
    ax.legend(frameon=False)
    save(fig, "om_monthly_price.png")

    series = monthly.dropna()
    adf_p = adfuller(series, regression="ct", autolag="AIC")[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, _, crit = kpss(series, regression="ct", nlags="auto")
    fit = STL(series, period=12, robust=True).fit()
    strength = max(0.0, 1 - np.var(fit.resid) / np.var(fit.seasonal + fit.resid))
    print(f"ADF trend p {adf_p:.4f} KPSS p {kpss_p:.3f} seasonal strength {strength:.2f}", flush=True)

    print("building histories", flush=True)
    early = train[train["date_sold"] < "2025-09-01"]
    check = train[train["date_sold"] >= "2025-09-01"]
    early_hist = building_history(early)
    train_hist = building_history(train)
    early_pack, train_pack = prepare(early), prepare(train)
    early_building = building_features(early, early_hist)
    check_building = building_features(check, early_hist)
    train_building = building_features(train, train_hist)
    valid_building = building_features(valid, train_hist)
    print("remarks", flush=True)
    early_remarks = fit_remarks(early)
    train_remarks = fit_remarks(train)
    text = {
        "early": apply_remarks(early_remarks, early),
        "check": apply_remarks(early_remarks, check),
        "train": apply_remarks(train_remarks, train),
        "valid": apply_remarks(train_remarks, valid),
    }
    print("shift model", flush=True)
    shift_model = direct_fit(early, early_pack, early_building, text["early"])
    shift_hat = direct_predict(shift_model, check, early_pack, check_building, text["check"])
    shift = float(np.median(np.log(shift_hat) - np.log(check["sold_price"].to_numpy(dtype=float))))
    print("full model", flush=True)
    model = direct_fit(train, train_pack, train_building, text["train"])
    base = direct_predict(model, valid, train_pack, valid_building, text["valid"])
    pred = base * np.exp(-shift)
    sold = valid["sold_price"].to_numpy()
    scores = percentage_scores(pred, sold)
    level, community_hit = price_level(valid, train_pack[0])
    anchor = level * valid["square_feet"].to_numpy(dtype=float)
    anchor_scores = percentage_scores(anchor, sold)
    print("V9", scores, "shift", round(100 * (np.exp(shift) - 1), 2), flush=True)
    print("V1 anchor", anchor_scores, "community share", round(float(community_hit.mean()), 3), flush=True)

    valid = valid.copy()
    valid["pred"] = pred
    valid["ape"] = np.abs(100 * (pred - sold) / sold)
    valid["signed"] = 100 * (pred - sold) / sold
    building_cover = float((valid_building["count"] >= 1).mean())

    decile = pd.qcut(valid["sold_price"], 10, labels=[f"D{i}" for i in range(1, 11)])
    decile_table = (
        valid.groupby(decile, observed=True)
        .agg(n=("sold_price", "size"), sold_min=("sold_price", "min"), sold_max=("sold_price", "max"), MdAPE=("ape", "median"), MAPE=("ape", "mean"), signed=("signed", "median"))
        .reset_index(names="decile")
    )
    type_table = (
        valid.groupby("house_type_name")
        .agg(n=("ape", "size"), MdAPE=("ape", "median"), MAPE=("ape", "mean"), signed=("signed", "median"))
        .sort_values("n", ascending=False)
        .reset_index()
    )
    muni_table = (
        valid.groupby("municipality")
        .agg(n=("ape", "size"), MdAPE=("ape", "median"), signed=("signed", "median"))
        .sort_values("n", ascending=False)
        .reset_index()
    )
    for table, cols in ((decile_table, ["MdAPE", "MAPE", "signed"]), (type_table, ["MdAPE", "MAPE", "signed"]), (muni_table, ["MdAPE", "signed"])):
        for column in cols:
            table[column] = table[column].round(2)
    decile_table["sold_min"] = decile_table["sold_min"].round(0)
    decile_table["sold_max"] = decile_table["sold_max"].round(0)

    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=model.feature_name_)
    gain = (100 * gain / gain.sum()).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(decile_table["decile"], decile_table["MdAPE"], color=NAVY, width=0.72)
    ax.axhline(scores["MdAPE"], color=AMBER, lw=1.2, ls="--", label=f"Overall MdAPE {scores['MdAPE']:.2f}%")
    ax.set_ylabel("MdAPE (%)")
    ax.set_xlabel("Sold-price decile (D1 lowest)")
    ax.set_title("Developmental absolute error by price decile")
    ax.legend(frameon=False)
    save(fig, "om_error_by_decile.png")

    order = type_table.sort_values("MdAPE")
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.barh(order["house_type_name"], order["MdAPE"], color=TEAL)
    ax.axvline(scores["MdAPE"], color=AMBER, lw=1.2, ls="--")
    ax.set_xlabel("MdAPE (%)")
    ax.set_title("Developmental absolute error by property type")
    save(fig, "om_error_by_type.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(valid["signed"], bins=60, range=(-40, 40), color=NAVY, edgecolor="white")
    ax.axvline(0, color=SLATE, lw=0.8, ls="--")
    ax.axvline(valid["signed"].median(), color=AMBER, lw=1.4, label=f"Median signed {valid['signed'].median():+.2f}%")
    ax.set_xlabel("Signed percentage error  (predicted − sold) / sold")
    ax.set_ylabel("Validation sales")
    ax.set_title("Developmental error distribution")
    ax.legend(frameon=False)
    save(fig, "om_error_distribution.png")

    valid["bin"] = pd.qcut(valid["pred"], 10, duplicates="drop")
    cal = valid.groupby("bin", observed=True).agg(predicted=("pred", "median"), actual=("sold_price", "median"))
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.plot(cal["predicted"] / 1e6, cal["actual"] / 1e6, "o-", color=NAVY, lw=1.4, ms=6)
    lo = min(cal["predicted"].min(), cal["actual"].min()) / 1e6 * 0.95
    hi = max(cal["predicted"].max(), cal["actual"].max()) / 1e6 * 1.05
    ax.plot([lo, hi], [lo, hi], color=AMBER, lw=1.2, label="Predicted = sold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Median predicted price ($ millions)")
    ax.set_ylabel("Median sold price ($ millions)")
    ax.set_title("Calibration by predicted-price bin")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "om_calibration.png")

    top = gain.head(12).sort_values()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(top.index, top.values, color=NAVY)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("LightGBM gain, developmental V9 fit")
    save(fig, "om_feature_gain.png")

    months = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
    v1 = [16.57, 15.43, 15.61, 14.79, 15.02, 15.28]
    v7 = [7.49, 7.44, 7.86, 7.19, 7.64, 7.40]
    v9 = [7.42, 7.27, 8.17, 7.10, 7.44, 7.39]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    xpos = np.arange(len(months))
    width = 0.26
    ax.bar(xpos - width, v1, width, color=SLATE, label="V1 community anchor")
    ax.bar(xpos, v7, width, color=TEAL, label="V7 gap model")
    ax.bar(xpos + width, v9, width, color=NAVY, label="V9 direct")
    ax.set_xticks(xpos)
    ax.set_xticklabels(months)
    ax.set_ylabel("MdAPE (%)")
    ax.set_title("Walk-forward median absolute percentage error")
    ax.legend(frameon=False, ncol=3)
    save(fig, "om_walkforward_mdape.png")

    bias = [0.63, -2.50, -4.44, -1.15, 1.65, 0.98]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    colors = [NAVY if value >= 0 else "#8c3a3a" for value in bias]
    ax.bar(months, bias, color=colors)
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_ylabel("Median signed error (%)")
    ax.set_title("V9 walk-forward bias by sale month")
    save(fig, "om_walkforward_bias.png")

    summary = {
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "test_listed_min": str(test["date_listed"].min().date()),
        "test_listed_max": str(test["date_listed"].max().date()),
        "has_list_price_on_test": bool("list_price" in test.columns),
        "n_community": int(listings["community"].nunique()),
        "n_style": int(listings["house_style"].nunique()),
        "n_municipality": int(listings["municipality"].nunique()),
        "dev": scores,
        "anchor": anchor_scores,
        "community_share_valid": round(float(community_hit.mean()), 4),
        "building_cover_valid": round(building_cover, 4),
        "shift_percent": round(100 * (float(np.exp(shift)) - 1), 2),
        "adf_p": round(float(adf_p), 4),
        "kpss_p": None if kpss_p >= 0.10 else round(float(kpss_p), 3),
        "kpss_stat": round(float(kpss_stat), 3),
        "kpss_10": round(float(crit["10%"]), 3),
        "seasonal_strength": round(float(strength), 2),
        "seasonal_swing": round(float(fit.seasonal.groupby(fit.seasonal.index.month).mean().max() - fit.seasonal.groupby(fit.seasonal.index.month).mean().min()), 0),
        "gain": {k: round(float(v), 2) for k, v in gain.items()},
        "deciles": decile_table.to_dict(orient="records"),
        "types": type_table.to_dict(orient="records"),
        "municipalities": muni_table.to_dict(orient="records"),
        "lot_missing_train": round(float(listings["lot_front"].isna().mean()), 4),
        "lot_missing_test": round(float(test["lot_front"].isna().mean()), 4),
    }
    out = Path(__file__).resolve().parent / "om_v9_diagnostics.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out.name, flush=True)


if __name__ == "__main__":
    main()
