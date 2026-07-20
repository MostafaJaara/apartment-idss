# ApartMatch — an IDSS for choosing an off-campus apartment

MSCI 436 (Decision Support Systems) project.

**The decision supported:** a university student with a fixed budget must
decide which apartments to shortlist and go view. A wrong choice costs a
year locked into an overpriced lease, a long commute, or weeks of wasted
searching. ApartMatch ranks real rental listings by a suitability score
the student controls, and flags listings priced below an ML estimate of
market rent.

## Why this is an IDSS and not a report

- **Load-bearing interactivity:** the student sets hard constraints
  (budget, bedrooms, max distance, pet policy) and importance weights
  (low rent vs. short commute vs. space vs. below-market value vs.
  amenities). Moving a weight slider re-ranks every listing, changing
  which apartments make the shortlist — verified by an automated test.
- **Evolving data:** listings churn daily. `scripts/update_data.py`
  re-ingests a fresh export, re-cleans it, and retrains the fair-rent
  model, reporting held-out MAE each time.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run tests: `python -m pytest tests/`
Refresh data + retrain: `python scripts/update_data.py [new_listings.csv]`

## Data

UCI Machine Learning Repository #555, "Apartment for Rent Classified" —
10,000 US rental listings from classified sites (semicolon-delimited,
latin-1). After cleaning: **9,730 rows**. Known limitations, surfaced in
the app itself:

- Amenities are self-reported; missing ≠ absent.
- 42% of listings omit a pet policy.
- No furnished/unfurnished flag exists in this source (a deviation from
  our original proposal; replaced with pet-policy filters and amenity
  preferences).
- Prices are a snapshot; the refresh pipeline addresses staleness in
  deployment.

## Model

1. **Fair-rent model** — `HistGradientBoostingRegressor` predicting
   monthly rent from bedrooms, bathrooms, square feet, amenity flags,
   and state. Held-out MAE ≈ **$292/month**. Its residual becomes a
   0–100 *deal score* (>50 = priced below predicted market rent).
2. **Suitability score** — a transparent weighted composite of price
   fit, commute distance (haversine to a selected campus), space, deal
   score, and amenity match. Weights come directly from UI sliders and
   are normalized, so criteria genuinely trade off.

The composite is deliberately interpretable: a student can see *why* a
listing ranks where it does, which a black-box ranker would not allow.

## Repository layout

```
app.py                  Streamlit IDSS (run this)
src/preprocess.py       cleaning pipeline (documented steps)
src/model.py            fair-rent model + suitability scoring
src/campuses.py         campus coordinate presets
scripts/update_data.py  nightly refresh + retrain entry point
tests/test_pipeline.py  end-to-end smoke test incl. load-bearing check
data/raw/               UCI 10K listings snapshot
```

## Operationalization

- **Access:** Streamlit Community Cloud (free tier) or any small VM;
  one `streamlit run` process, no GPU required.
- **Pipeline continuity:** nightly cron/GitHub Action runs
  `update_data.py`; the app serves the newest cleaned file via cache
  invalidation.
- **Retraining:** on every refresh (< 5 s on this data size); MAE is
  logged so drift is visible.
- **Infrastructure:** Python 3.10+, pandas, scikit-learn, Streamlit.
  Memory footprint < 500 MB.
