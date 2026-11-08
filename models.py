"""The models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() returns the submission model for each track:

    list_aware   LA-V22, a gradient-boosted model of the sold-to-list gap
    off_market   OM-V9, a gradient-boosted model of log sold price

List price is an input only for the list-aware model. Both models build every
comp, peer price, and text score from sales that closed before the home was
listed. The level shift in OM-V9 is measured on the last three months of the
training sales, then applied to a model fit on all of them.

ListPriceBaseline and NearestSalesBaseline are the starter baselines.
They are not used for the submission.
"""
import re
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from features import area_is_range, home_numbers, lot_area, remark_flags, square_feet


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        return ListAwareModel()
    if track == "off_market":
        return OffMarketModel()
    raise ValueError(f"unknown track: {track}")


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. The submission uses ListAwareModel
    and OffMarketModel instead.
    """

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes)))


# Rows below this use the full LA-V22 / OM-V9 specification. The starter's tiny
# pipeline test is far smaller, so it gets a short tree fit that only checks the
# plumbing. Every real training slice is well above this.
_FULL_ROWS = 8000
_LGB = dict(
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
_V22_CATEGORIES = ["house_type_name", "house_style", "municipality", "community"]
_OM_CATEGORIES = ["house_type_name", "house_style", "community", "municipality"]
_TEXT_ALPHA = 0.3
_PSF_WINDOW = 12
_PEER_COUNT = 30
_HORIZON = 24
_FORECAST_STEPS = 18


def _lgb_params(n_rows):
    params = dict(_LGB)
    if n_rows < _FULL_ROWS:
        params.update(n_estimators=30, num_leaves=15, min_child_samples=2, n_jobs=1)
    return params


def _with_square_feet(frame):
    frame = frame.copy()
    frame["square_feet"] = square_feet(frame["house_area"])
    return frame


def _month_ord(dates):
    return (dates.dt.year * 12 + dates.dt.month).to_numpy()


def _as_key(key):
    return key if isinstance(key, tuple) else (key,)


def _empty_table(group_columns):
    return pd.DataFrame(columns=[*group_columns, "month_ord", "median", "count"])


def _past_median_table(reference, values, group_columns):
    """Median of `values` over every earlier sale in the group, by listing month."""
    history = reference.loc[:, group_columns].copy()
    history["month_ord"] = _month_ord(reference["date_sold"])
    history["value"] = np.asarray(values, dtype=float)
    history = history.dropna()
    if history.empty:
        return _empty_table(group_columns)
    for column in group_columns:
        history[column] = history[column].astype(str)
    history["month_ord"] = history["month_ord"].astype(np.int64)
    query_ords = np.arange(int(history["month_ord"].min()), int(history["month_ord"].max()) + _HORIZON)
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
        piece = {"month_ord": query_ords, "median": medians, "count": ends}
        for column, value in zip(group_columns, _as_key(key)):
            piece[column] = value
        pieces.append(pd.DataFrame(piece))
    return pd.concat(pieces, ignore_index=True)


def _lookup_past(query, table, group_columns):
    keys = query.loc[:, list(group_columns)].copy()
    for column in group_columns:
        keys[column] = keys[column].astype(str)
    keys["month_ord"] = _month_ord(query["date_listed"]).astype(np.int64)
    merged = keys.merge(table, on=list(group_columns) + ["month_ord"], how="left")
    if len(merged) != len(query):
        raise ValueError("past-median lookup changed the number of rows")
    median = merged["median"].to_numpy(dtype=float)
    count = merged["count"].fillna(0).to_numpy(dtype=float)
    return median, count


def _recent_ratio_table(reference, group_columns, window=3):
    """Median sold/list ratio over the `window` months before the listing month."""
    history = reference.loc[:, group_columns].copy()
    history["month_ord"] = _month_ord(reference["date_sold"])
    history["value"] = (reference["sold_price"] / reference["list_price"]).to_numpy(dtype=float)
    history = history.replace([np.inf, -np.inf], np.nan).dropna()
    if history.empty:
        return _empty_table(group_columns)
    for column in group_columns:
        history[column] = history[column].astype(str)
    history["month_ord"] = history["month_ord"].astype(np.int64)
    query_ords = np.arange(int(history["month_ord"].min()), int(history["month_ord"].max()) + _HORIZON)
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
        piece = {"month_ord": query_ords, "median": medians, "count": counts}
        for column, value in zip(group_columns, _as_key(key)):
            piece[column] = value
        pieces.append(pd.DataFrame(piece))
    return pd.concat(pieces, ignore_index=True)


def _recent_psf_table(reference, group_columns, window):
    """Median sold $/sqft over the `window` months before the listing month."""
    history = reference.loc[:, group_columns].copy()
    history["month_ord"] = _month_ord(reference["date_sold"])
    history["value"] = (reference["sold_price"] / reference["square_feet"]).to_numpy(dtype=float)
    history = history.replace([np.inf, -np.inf], np.nan).dropna()
    if history.empty:
        return _empty_table(group_columns)
    for column in group_columns:
        history[column] = history[column].astype(str)
    history["month_ord"] = history["month_ord"].astype(np.int64)
    last_month = int(history["month_ord"].max())
    query_ords = np.arange(int(history["month_ord"].min()) - _HORIZON, last_month + _HORIZON)
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
        piece = {"month_ord": query_ords, "median": medians, "count": counts}
        for column, value in zip(group_columns, _as_key(key)):
            piece[column] = value
        pieces.append(pd.DataFrame(piece))
    return pd.concat(pieces, ignore_index=True)


def _psf_bundle(reference, window):
    usable = reference[(reference["sold_price"] > 0) & (reference["square_feet"] > 0)].copy()
    usable["_all"] = "all"
    return {
        "fallback": float(np.median(usable["sold_price"] / usable["square_feet"])),
        "community": _recent_psf_table(usable, ["community", "house_type_name"], window),
        "type": _recent_psf_table(usable, ["house_type_name"], window),
        "city": _recent_psf_table(usable, ["_all"], window),
    }


def _price_level(frame, bundle):
    query = frame.copy()
    query["_all"] = "all"
    community_median, community_count = _lookup_past(query, bundle["community"], ["community", "house_type_name"])
    type_median, type_count = _lookup_past(query, bundle["type"], ["house_type_name"])
    city_median, _ = _lookup_past(query, bundle["city"], ["_all"])
    level = np.where(
        community_count >= _PEER_COUNT,
        community_median,
        np.where(type_count >= _PEER_COUNT, type_median, city_median),
    )
    return np.where(np.isfinite(level) & (level > 0), level, bundle["fallback"])


def _city_bundle(reference, window):
    usable = reference[(reference["sold_price"] > 0) & (reference["square_feet"] > 0)].copy()
    usable["_all"] = "all"
    return {
        "fallback": float(np.median(usable["sold_price"] / usable["square_feet"])),
        "city": _recent_psf_table(usable, ["_all"], window),
    }


def _city_level(frame, bundle):
    query = frame.copy()
    query["_all"] = "all"
    median, _ = _lookup_past(query, bundle["city"], ["_all"])
    return np.where(np.isfinite(median) & (median > 0), median, bundle["fallback"])


def _category_levels(reference):
    return {column: pd.Index(reference[column].dropna().unique()) for column in _OM_CATEGORIES}


def _prepare(reference):
    return (
        _psf_bundle(reference, _PSF_WINDOW),
        _city_bundle(reference, 3),
        _city_bundle(reference, 12),
        _category_levels(reference),
    )


def _pin_key(frame):
    lat = np.round(frame["latitude"].to_numpy(dtype=float), 6)
    lon = np.round(frame["longitude"].to_numpy(dtype=float), 6)
    kind = frame["house_type_name"].astype(str).to_numpy()
    return np.array([f"{a:.6f}|{b:.6f}|{c}" for a, b, c in zip(lat, lon, kind)])


def _index_tables(types, ords, log_psf):
    frame = pd.DataFrame({"typ": types, "ord": ords, "log_psf": log_psf})
    monthly = frame.groupby(["typ", "ord"], sort=False)["log_psf"].median()
    tables = {}
    for kind, series in monthly.groupby(level=0):
        values = series.droplevel(0).sort_index()
        span = np.arange(int(values.index.min()), int(values.index.max()) + 1 + _HORIZON)
        level = values.reindex(span).ffill()
        known = level.rolling(3, min_periods=1).median().shift(1)
        tables[kind] = (span.astype(np.int32), known.to_numpy(dtype=float))
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
        kind = sorted_types[start]
        table = tables.get(kind)
        if table is None:
            continue
        span, known = table
        pos = order[start:stop]
        query = ords[pos]
        loc = np.searchsorted(span, query)
        in_span = (loc < len(span)) & (span[np.clip(loc, 0, len(span) - 1)] == query)
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


def _building_history(reference):
    """Same-building sales. A comp has to close before the home is listed."""
    usable = reference[(reference["sold_price"] > 0) & (reference["square_feet"] > 0)].copy()
    usable = usable.dropna(subset=["latitude", "longitude", "house_type_name", "date_sold", "ml_num"])
    usable = usable.sort_values(["date_sold", "ml_num"])
    if usable.empty:
        return {"index": {}, "building": {}, "ids": set()}
    psf = usable["sold_price"].to_numpy(dtype=float) / usable["square_feet"].to_numpy(dtype=float)
    log_psf = np.log(psf)
    ords = (usable["date_sold"].dt.year * 12 + usable["date_sold"].dt.month).to_numpy(dtype=np.int32)
    types = usable["house_type_name"].astype(str).to_numpy()
    tables = _index_tables(types, ords, log_psf)
    sale_index = _index_at(tables, types, ords)
    keep = np.isfinite(sale_index) & np.isfinite(psf) & (psf > 0)
    usable = usable.iloc[np.flatnonzero(keep)]
    factor = psf[keep] * np.exp(-sale_index[keep])
    dates = usable["date_sold"].to_numpy(dtype="datetime64[ns]")
    ml_num = usable["ml_num"].astype(str).to_numpy()
    return {
        "index": tables,
        "building": _groups(_pin_key(usable), dates, factor, ml_num),
        "ids": set(ml_num),
    }


def _building_features(homes, history):
    n = len(homes)
    count = np.zeros(n, dtype=np.int32)
    median_psf = np.full(n, np.nan)
    if n == 0 or not history["building"]:
        return pd.DataFrame({"count": count, "median_psf": median_psf}, index=homes.index)
    listed = homes["date_listed"].to_numpy(dtype="datetime64[ns]")
    mls = homes["ml_num"].astype(str).to_numpy()
    list_ord = (homes["date_listed"].dt.year * 12 + homes["date_listed"].dt.month).to_numpy(dtype=np.int32)
    index_now = _index_at(history["index"], homes["house_type_name"].astype(str).to_numpy(), list_ord)
    keys = _pin_key(homes)
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
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


def _remarks_vectorizer(communities, n_docs):
    min_df = 40 if n_docs >= _FULL_ROWS else 1
    return TfidfVectorizer(
        preprocessor=_remarks_cleaner(communities),
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=3000,
        token_pattern=r"(?u)\b[a-zA-Z]+\b",
        sublinear_tf=True,
    )


def _oof_remarks(texts, communities, target, dates, n_docs):
    """Score each training remark from sales that closed earlier, never from itself."""
    order = np.argsort(dates, kind="mergesort")
    score = np.full(len(target), np.nan)
    edges = [int(len(target) * fraction) for fraction in (0.40, 0.55, 0.70, 0.85, 1.00)]
    start = edges[0]
    for end in edges[1:]:
        train_rows = order[:start]
        test_rows = order[start:end]
        if len(train_rows) < 1000 or len(test_rows) == 0:
            start = end
            continue
        vectorizer = _remarks_vectorizer(communities[train_rows], n_docs)
        model = Ridge(alpha=_TEXT_ALPHA)
        model.fit(vectorizer.fit_transform(texts[train_rows]), target[train_rows])
        score[test_rows] = model.predict(vectorizer.transform(texts[test_rows]))
        start = end
    return score


def _fit_remarks(history):
    texts = history["remarks"].fillna("").astype(str).to_numpy()
    communities = history["community"].astype(str).to_numpy()
    target = np.log(history["sold_price"].to_numpy(dtype=float))
    dates = history["date_sold"].to_numpy(dtype="datetime64[ns]")
    n_docs = len(history)
    oof = _oof_remarks(texts, communities, target, dates, n_docs)
    vectorizer = _remarks_vectorizer(communities, n_docs)
    ridge = Ridge(alpha=_TEXT_ALPHA)
    ridge.fit(vectorizer.fit_transform(texts), target)
    return {
        "vectorizer": vectorizer,
        "ridge": ridge,
        "oof": pd.Series(oof, index=history.index),
        "ids": set(history["ml_num"].astype(str)),
    }


def _apply_remarks(model, homes):
    if homes.index.equals(model["oof"].index):
        return model["oof"]
    if set(homes["ml_num"].astype(str)) & model["ids"]:
        raise RuntimeError("remarks score uses a sale to predict itself")
    matrix = model["vectorizer"].transform(homes["remarks"].fillna("").astype(str))
    return pd.Series(model["ridge"].predict(matrix), index=homes.index)


def _direct_matrix(frame, design_pack, building, text):
    bundle, city3, city12, levels = design_pack
    X = home_numbers(frame, include_list=False)
    X["log_community_psf"] = np.log(_price_level(frame, bundle))
    X["log_city_psf_3"] = np.log(_city_level(frame, city3))
    X["log_city_psf_12"] = np.log(_city_level(frame, city12))
    for column in _OM_CATEGORIES:
        X[column] = pd.Categorical(frame[column].to_numpy(), categories=levels[column])
    for name, values in remark_flags(frame).items():
        X[name] = values.to_numpy()
    matched = building.loc[frame.index]
    count = matched["count"].to_numpy(dtype=float)
    median = matched["median_psf"].to_numpy(dtype=float)
    sqft = frame["square_feet"].to_numpy(dtype=float)
    building_price = np.full(len(frame), np.nan)
    ok = (count >= 1) & np.isfinite(median) & (median > 0) & np.isfinite(sqft) & (sqft > 0)
    building_price[ok] = np.log(median[ok] * sqft[ok])
    X["building_sales"] = count
    X["building_log_price"] = building_price
    X["remarks_score"] = text.loc[frame.index].to_numpy(dtype=float)
    return X


def _direct_fit(frame, design_pack, building, text):
    model = LGBMRegressor(**_lgb_params(len(frame)))
    model.fit(
        _direct_matrix(frame, design_pack, building, text),
        np.log(frame["sold_price"].to_numpy(dtype=float)),
        categorical_feature=_OM_CATEGORIES,
    )
    return model


def _direct_predict(model, frame, design_pack, building, text):
    return np.exp(model.predict(_direct_matrix(frame, design_pack, building, text)))


def _early_cut(train):
    """First day of the three-month window used to measure OM-V9's level shift.

    Walk-forward scores a month from sales that closed before it, and measures the
    shift on the three months before that month. The same rule, on whatever sales
    are in `train`, is three months before the month after the last sale.
    """
    last = pd.Timestamp(train["date_sold"].max()).to_period("M")
    return (last - 2).to_timestamp()


class ListAwareModel:
    """LA-V22. Predicts the log gap between sold price and list price.

    The price is list price times exp(gap). Community is a category. Peer asks,
    the recent sold-to-list discount, and the listing-month factor are rebuilt
    from the training sales only.
    """

    def fit(self, train):
        train = _with_square_feet(train)
        print(f"LA-V22: fitting on {len(train):,} sales", flush=True)
        self.lag_map, self.last_month, self.latest_ratio = self._market_tables(train)
        sold_psf = (train["sold_price"] / train["square_feet"]).to_numpy()
        self.peer_psf = _past_median_table(train, sold_psf, ["community", "house_type_name"])
        self.type_psf = _past_median_table(train, sold_psf, ["house_type_name"])
        self.community_ratios = _recent_ratio_table(train, ["community", "house_type_name"])
        self.type_ratios = _recent_ratio_table(train, ["house_type_name"])
        self.monthly, self.forecast = self._residual_forecast(train)
        self.levels = {
            column: pd.Index(train[column].dropna().unique())
            for column in ["house_type_name", "house_style", "municipality"]
        }
        self.community_levels = pd.Index(train["community"].dropna().unique())
        gap = np.log(train["sold_price"].to_numpy(dtype=float) / train["list_price"].to_numpy(dtype=float))
        self.model = LGBMRegressor(**_lgb_params(len(train)))
        self.model.fit(self._matrix(train), gap, categorical_feature=_V22_CATEGORIES)
        print("LA-V22: ready", flush=True)
        return self

    def predict(self, homes):
        if len(homes) == 0:
            return np.empty(0, dtype=float)
        homes = _with_square_feet(homes)
        gap = self.model.predict(self._matrix(homes))
        return homes["list_price"].to_numpy(dtype=float) * np.exp(gap)

    def _market_tables(self, past):
        log_ratio = np.log(past["sold_price"].to_numpy(dtype=float) / past["list_price"].to_numpy(dtype=float))
        by_month = pd.Series(log_ratio, index=past.index).groupby(past["date_sold"].dt.to_period("M")).median()
        return by_month.shift(1), by_month.index.max(), float(by_month.iloc[-1])

    def _residual_forecast(self, past):
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        design = sm.add_constant(self._ols_design(past), has_constant="add")
        ols = sm.OLS(np.log(past["sold_price"].to_numpy(dtype=float)), design).fit()
        monthly = pd.Series(ols.resid, index=past.index).groupby(past["date_sold"].dt.to_period("M")).mean()
        monthly.index = monthly.index.to_timestamp()
        monthly = monthly.asfreq("MS")
        forecast = pd.Series(dtype=float)
        if int(monthly.notna().sum()) >= 36:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                fitted = SARIMAX(
                    monthly.astype(float),
                    order=(1, 0, 0),
                    seasonal_order=(1, 0, 0, 12),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)
            forecast = fitted.get_forecast(steps=_FORECAST_STEPS).predicted_mean
        monthly.index = pd.to_datetime(monthly.index).to_period("M").to_timestamp()
        if len(forecast):
            forecast.index = pd.to_datetime(forecast.index).to_period("M").to_timestamp()
        return monthly, forecast

    def _ols_design(self, frame):
        return home_numbers(frame, include_list=True).drop(
            columns=["listed_month_sin", "listed_month_cos"]
        )

    def _market_lag(self, frame):
        keys = frame["date_listed"].dt.to_period("M")
        lag = keys.map(self.lag_map).astype(float)
        lag = lag.mask(keys > self.last_month, self.latest_ratio)
        return lag.fillna(0.0).to_numpy()

    def _list_month_factor(self, frame):
        listed = frame["date_listed"].dt.to_period("M").dt.to_timestamp()
        previous = listed - pd.DateOffset(months=1)
        return listed.map(self.forecast).fillna(previous.map(self.monthly)).fillna(0.0).to_numpy()

    def _peer_ask(self, frame):
        list_psf = (frame["list_price"] / frame["square_feet"]).to_numpy(dtype=float)
        type_median, _ = _lookup_past(frame, self.type_psf, ["house_type_name"])
        community_median, community_count = _lookup_past(frame, self.peer_psf, ["community", "house_type_name"])
        chosen = np.where(community_count >= _PEER_COUNT, community_median, type_median)
        out = np.full(len(frame), np.nan)
        ok = np.isfinite(chosen) & (chosen > 0)
        out[ok] = list_psf[ok] / chosen[ok]
        return out

    def _recent_discount(self, frame):
        community_median, community_count = _lookup_past(
            frame, self.community_ratios, ["community", "house_type_name"]
        )
        type_median, type_count = _lookup_past(frame, self.type_ratios, ["house_type_name"])
        return np.where(
            community_count >= _PEER_COUNT,
            community_median,
            np.where(type_count >= _PEER_COUNT, type_median, np.nan),
        )

    def _matrix(self, frame):
        X = home_numbers(frame, include_list=True)
        X["market_lag"] = self._market_lag(frame)
        X["lot_area"] = lot_area(frame)
        X["area_is_range"] = area_is_range(frame)
        for column, levels in self.levels.items():
            X[column] = pd.Categorical(frame[column].to_numpy(), categories=levels)
        X["peer_ask"] = self._peer_ask(frame)
        X["list_month_factor"] = self._list_month_factor(frame)
        X["recent_discount"] = self._recent_discount(frame)
        X["community"] = pd.Categorical(frame["community"].to_numpy(), categories=self.community_levels)
        return X


class OffMarketModel:
    """OM-V9. Predicts log sold price directly, with no list price.

    Adopted specification: remarks score, one global level shift at full weight,
    the 12-month community price, and the extra remark flags. The shift is the
    median log miss of a model fit before the last three training months and
    scored on those three months.
    """

    def fit(self, train):
        train = _with_square_feet(train)
        early_cut = _early_cut(train)
        early = train[train["date_sold"] < early_cut]
        check = train[train["date_sold"] >= early_cut]
        self.shift = 0.0
        if len(early) >= 200 and len(check) >= 50:
            print(f"OM-V9: level shift from sales on or after {early_cut:%Y-%m-%d}", flush=True)
            early_history = _building_history(early)
            early_pack = _prepare(early)
            early_building = _building_features(early, early_history)
            check_building = _building_features(check, early_history)
            print("OM-V9: remarks score for the shift window", flush=True)
            early_remarks = _fit_remarks(early)
            shift_model = _direct_fit(
                early, early_pack, early_building, _apply_remarks(early_remarks, early)
            )
            shift_hat = _direct_predict(
                shift_model, check, early_pack, check_building, _apply_remarks(early_remarks, check)
            )
            error = np.log(shift_hat) - np.log(check["sold_price"].to_numpy(dtype=float))
            if np.isfinite(error).all():
                self.shift = float(np.median(error))
        print(
            f"OM-V9: fitting on {len(train):,} sales, shift {100 * (np.exp(self.shift) - 1):+.2f}%",
            flush=True,
        )
        self.history = _building_history(train)
        self.pack = _prepare(train)
        self.building = _building_features(train, self.history)
        print("OM-V9: remarks score for the full fit", flush=True)
        self.remarks = _fit_remarks(train)
        self.model = _direct_fit(train, self.pack, self.building, _apply_remarks(self.remarks, train))
        print("OM-V9: ready", flush=True)
        return self

    def predict(self, homes):
        if len(homes) == 0:
            return np.empty(0, dtype=float)
        homes = _with_square_feet(homes)
        building = _building_features(homes, self.history)
        text = _apply_remarks(self.remarks, homes)
        base = _direct_predict(self.model, homes, self.pack, building, text)
        return base * np.exp(-self.shift)
