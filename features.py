"""Columns for the two submission models.

Anything that can be read off the home is built here. Anything that depends on
earlier sales is built in models.py during fit, from those sales only, and then
looked up for the home being priced. These functions are given the columns of a
test file, so they never see sold_price or date_sold. List price is present
only for the list-aware track.

LA-V22 predicts the log gap between sold price and list price. On top of
home_numbers(include_list=True) it adds, from past sales:

- market_lag: last month's median log sold-to-list ratio
- lot_area, area_is_range: the lot, and whether house_area was given as a range
- house_type_name, house_style, municipality, community: categories learned in training
- peer_ask: this home's list price per square foot, divided by the past median
  for its community and type (type, when the community has fewer than 30 sales)
- list_month_factor: where this listing month sits in the recent residual path
- recent_discount: median sold-to-list ratio over the previous three months,
  for the community and type, then the type, then missing

OM-V9 predicts log sold price and does not use list price. On top of
home_numbers(include_list=False) and remark_flags() it adds, from past sales:

- log_community_psf: 12-month median sold dollars per square foot, community and
  type, falling back to the type and then the city below 30 sales
- log_city_psf_3, log_city_psf_12: the same median for the whole city
- house_type_name, house_style, community, municipality: training categories
- building_sales, building_log_price: earlier sales at the same pin and type
- remarks_score: a ridge on the remark text, fit on earlier sales
"""
import numpy as np
import pandas as pd

# Phrases kept for OM-V9. The first eight were the base set. The last five
# lowered the July-November error, so they stayed.
REMARK_PHRASES = [
    ("as_is", r"as-is|as is"),
    ("turnkey", r"turnkey"),
    ("potential", r"potential"),
    ("original", r"original"),
    ("finished", r"finished"),
    ("renovated", r"renovat"),
    ("needs_work", r"needs work"),
    ("stunning", r"stunning|beautiful|gorgeous|spectacular"),
    ("investor", r"\binvestors?\b"),
    ("motivated", r"\bmotivated\b"),
    ("redone", r"\bredone\b|\bredo\b"),
    ("dated", r"\bdated\b"),
    ("worn", r"\bworn\b"),
]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def home_numbers(frame, include_list):
    """Size, lot, rooms, location, and listing-month seasonality.

    The frame must already have a square_feet column. Missing lot frontage is
    its own column, and the log frontage of a missing lot is log(1), so a condo
    is not given a made-up lot. include_list adds log list price for LA-V22.
    """
    missing_lot = frame["lot_front"].isna()
    X = pd.DataFrame(index=frame.index)
    if include_list:
        X["log_list_price"] = np.log(frame["list_price"].to_numpy(dtype=float))
    X["log_square_feet"] = np.log(frame["square_feet"].to_numpy(dtype=float))
    X["lot_missing"] = missing_lot.astype(float)
    X["log_lot_front"] = np.log(frame["lot_front"].where(~missing_lot, 1.0).to_numpy(dtype=float))
    X["lot_depth"] = frame["lot_depth"].fillna(0.0).to_numpy(dtype=float)
    for column in ["bedrooms", "washrooms", "parking_total", "garage", "latitude", "longitude"]:
        X[column] = frame[column].to_numpy(dtype=float)
    listed_month = frame["date_listed"].dt.month.to_numpy()
    X["listed_month_sin"] = np.sin(2 * np.pi * listed_month / 12)
    X["listed_month_cos"] = np.cos(2 * np.pi * listed_month / 12)
    return X


def lot_area(frame):
    """Frontage times depth. Missing for condos, which the tree can leave missing."""
    return (frame["lot_front"] * frame["lot_depth"]).to_numpy(dtype=float)


def area_is_range(frame):
    """1 when house_area was written as a range rather than a single number."""
    return frame["house_area"].astype(str).str.contains("-", regex=False).astype(float).to_numpy()


def remark_flags(frame):
    """1 when the remark contains a phrase OM-V9 keeps."""
    remarks = frame["remarks"].fillna("").str.lower()
    columns = {
        name: remarks.str.contains(pattern, regex=True).astype(float).to_numpy()
        for name, pattern in REMARK_PHRASES
    }
    return pd.DataFrame(columns, index=frame.index)
