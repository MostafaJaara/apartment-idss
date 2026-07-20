# ApartMatch

MSCI 436 (Decision Support Systems) project.

An interactive tool that helps a university student decide **which apartments are worth viewing**, given a budget, a campus, and personal priorities.

## The decision it supports

Finding an off-campus apartment is a real choice with real costs:

- Too expensive → a year stuck in a lease you can't afford
- Too far → a long daily commute
- Too much browsing → weeks of searching with no shortlist

ApartMatch does not "pick for you." It helps you **narrow thousands of listings into a shortlist of 3–5 places to go see**, using rules you control.

## How the logic works

The system has two layers: **hard rules** (must-haves) and **soft preferences** (what matters more).

### 1. Hard constraints (must pass)

Only listings that pass every rule below stay in the pool:

| Constraint | Meaning |
|---|---|
| Campus / city | Must be near the campus you selected |
| Max budget | Monthly rent must be ≤ your limit |
| Bedrooms | Must have at least the bedrooms you need |
| Max distance | Must be within your commute limit |
| Pets (optional) | If checked, must explicitly allow cats and/or dogs |

Anything that fails is removed completely — it never gets ranked.

### 2. Soft preferences (how remaining listings are ranked)

Surviving listings get a **suitability score from 0–100**. That score is a mix of five factors. You set how important each one is with the sidebar sliders:

| Factor | Higher score when… |
|---|---|
| Low rent | Asking price is lower relative to your budget |
| Short commute | Closer to campus |
| More space | Larger square footage (compared to other matches) |
| Below-market deal | Asking rent looks cheap vs. what similar units usually cost |
| Amenity match | More of your selected amenities are listed |

Weights are normalized, so raising one factor automatically reduces the relative influence of the others. **Moving a slider re-ranks the shortlist** — that interactivity is the point of the system.

### 3. Fair-rent / "deal" check

Separately, the system estimates a typical market rent for each listing from size, bedrooms/bathrooms, amenities, and location.

- If asking price is **below** that estimate → higher deal score (possible bargain)
- If asking price is **near** the estimate → around the middle (at market)
- If asking price is **above** the estimate → lower deal score (possibly overpriced)

This is one input to the ranking, not the final decision. You decide how much "deal hunting" matters by adjusting that slider.

### 4. What you do with the output

1. Set constraints and weights
2. Review the ranked shortlist (top ~15)
3. Use the price-vs-commute chart to spot cheap+close options your weights might undervalue
4. Shortlist the top 3–5 and book viewings
5. Prefer listings that stay near the top even when you tweak weights — those are more robust choices

## Why this is decision support (not just a report)

- Your inputs change which apartments appear and in what order
- You can see the trade-offs (cheap vs. close vs. spacious vs. good deal)
- The ranking is explainable: each factor is scored, then combined with your weights
- Data can be refreshed and the fair-rent estimate relearned as listings change

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

Optional:

```bash
python -m pytest tests/                          # run checks
python scripts/update_data.py [new_listings.csv] # refresh data + retrain
```

## Data (and caveats that affect decisions)

Source: UCI "Apartment for Rent Classified" (~10,000 US listings). After cleaning: **~9,730** usable rows.

Important limitations (also shown in the app):

- Amenities are self-reported — missing does **not** mean the amenity is absent
- Many listings omit pet policy; pet filters only keep listings that **explicitly** allow pets
- There is no furnished/unfurnished field in this dataset
- Prices are a snapshot; refresh the data periodically for up-to-date decisions

## Project layout

```
app.py                  Interactive UI (run this)
src/preprocess.py       Cleaning steps for raw listings
src/model.py            Fair-rent estimate + suitability scoring
src/campuses.py         Campus locations used for commute distance
scripts/update_data.py  Refresh listings and retrain fair-rent model
tests/test_pipeline.py  End-to-end checks (including weight re-ranking)
data/raw/               Listings snapshot
```
