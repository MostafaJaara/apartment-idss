"""Smoke tests: pipeline cleans, model trains, scoring reorders on weight change."""
import numpy as np
from src.preprocess import load_clean
from src.model import FairRentModel, apply_filters, suitability_score
from src.campuses import CAMPUSES


def test_end_to_end():
    df = load_clean()
    assert len(df) > 8000
    assert df["latitude"].notna().all()

    model = FairRentModel().fit(df)
    assert model.mae_ is not None and model.mae_ < 600  # sanity bound

    city, state, lat, lon = CAMPUSES["UT Austin (Austin, TX)"]
    pool = df[(df.cityname == city) & (df.state == state)]
    cand = apply_filters(pool, 1600, 1, 12, lat, lon)
    assert len(cand) > 20

    deal = model.deal_score(cand)
    s_price = suitability_score(cand, 1600, 12, deal, 1, 0, 0, 0)
    s_dist = suitability_score(cand, 1600, 12, deal, 0, 1, 0, 0)
    # Load-bearing check: changing weights changes the top-ranked listing set
    top_price = set(s_price.sort_values(ascending=False).head(5).index)
    top_dist = set(s_dist.sort_values(ascending=False).head(5).index)
    assert top_price != top_dist
