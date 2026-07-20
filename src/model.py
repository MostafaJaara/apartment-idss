"""Models for the apartment IDSS.

Two components:

1. FairRentModel -- a gradient-boosted regression that predicts the
   market rent of a listing from its attributes (bedrooms, bathrooms,
   square feet, amenity flags, city). The residual between asking price
   and predicted market price becomes a "deal score": listings priced
   below their predicted market rent score higher. This is the
   data-driven part of the system and is retrained whenever the
   listings file is refreshed.

2. suitability_score -- converts a student's stated preferences and
   weights into a 0-100 score per listing. The weights are user-facing
   sliders: moving them reorders the ranking, which is what makes the
   interface load-bearing rather than cosmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .preprocess import AMENITY_FLAGS, haversine_km

NUMERIC = ["bedrooms", "bathrooms", "square_feet"]
FLAGS = [f"has_{f.lower().replace(' ', '_')}" for f in AMENITY_FLAGS]
CATEG = ["state"]


class FairRentModel:
    """Predicts market rent; exposes a per-listing deal score."""

    def __init__(self, random_state: int = 436):
        self.pipe = Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("num", "passthrough", NUMERIC + FLAGS),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEG),
                        ]
                    ),
                ),
                ("gbm", HistGradientBoostingRegressor(random_state=random_state)),
            ]
        )
        self.mae_: float | None = None

    def fit(self, df: pd.DataFrame) -> "FairRentModel":
        X, y = df[NUMERIC + FLAGS + CATEG], df["price"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=436
        )
        self.pipe.fit(X_tr, y_tr)
        self.mae_ = float(mean_absolute_error(y_te, self.pipe.predict(X_te)))
        return self

    def predict_market_rent(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(df[NUMERIC + FLAGS + CATEG])

    def deal_score(self, df: pd.DataFrame) -> np.ndarray:
        """0-100. 50 = priced at market; >50 = below market (a deal)."""
        predicted = self.predict_market_rent(df)
        rel = (predicted - df["price"].to_numpy()) / np.maximum(predicted, 1.0)
        return np.clip(50 + 100 * rel, 0, 100)


def apply_filters(
    df: pd.DataFrame,
    max_budget: float,
    min_bedrooms: int,
    max_km: float,
    campus_lat: float,
    campus_lon: float,
    need_cats: bool = False,
    need_dogs: bool = False,
) -> pd.DataFrame:
    """Hard constraints. Listings failing any of these are not shown."""
    out = df.copy()
    out["distance_km"] = haversine_km(
        out["latitude"].astype(float),
        out["longitude"].astype(float),
        campus_lat,
        campus_lon,
    )
    mask = (
        (out["price"] <= max_budget)
        & (out["bedrooms"] >= min_bedrooms)
        & (out["distance_km"] <= max_km)
    )
    if need_cats:
        mask &= out["cats_ok"]
    if need_dogs:
        mask &= out["dogs_ok"]
    return out[mask]


def suitability_score(
    df: pd.DataFrame,
    max_budget: float,
    max_km: float,
    deal: np.ndarray,
    w_price: float,
    w_distance: float,
    w_space: float,
    w_deal: float,
    wanted_amenities: list[str] | None = None,
    w_amenities: float = 0.0,
) -> pd.Series:
    """Weighted 0-100 suitability score.

    Component scores are each 0-100:
      price:     cheaper relative to budget is better (linear)
      distance:  closer to campus is better (linear within max_km)
      space:     square feet, scaled against the 90th pct of the pool
      deal:      FairRentModel deal score (below-market pricing)
      amenities: fraction of the student's wanted amenities present

    Weights come straight from UI sliders and are normalized here, so
    the sliders always trade off against each other.
    """
    price_s = 100 * (1 - df["price"] / max_budget).clip(0, 1)
    dist_s = 100 * (1 - df["distance_km"] / max_km).clip(0, 1)
    sf_ref = max(float(df["square_feet"].quantile(0.9)), 1.0)
    space_s = 100 * (df["square_feet"] / sf_ref).clip(0, 1)

    if wanted_amenities:
        cols = [f"has_{a.lower().replace(' ', '_')}" for a in wanted_amenities]
        amen_s = 100 * df[cols].mean(axis=1)
    else:
        amen_s = pd.Series(0.0, index=df.index)
        w_amenities = 0.0

    weights = np.array([w_price, w_distance, w_space, w_deal, w_amenities], float)
    if weights.sum() == 0:
        weights = np.array([1, 1, 1, 1, 0], float)
    weights = weights / weights.sum()

    parts = np.vstack([price_s, dist_s, space_s, deal, amen_s])
    return pd.Series(weights @ parts, index=df.index).round(1)
