"""ApartMatch -- an Intelligent Decision Support System for choosing
an off-campus apartment.

Run with:  streamlit run app.py

MSCI 436 project. The decision supported: which apartment should this
student shortlist and go view, given a fixed budget, a campus to reach,
and personal trade-offs between price, commute, space, and value.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.campuses import CAMPUSES
from src.model import FairRentModel, apply_filters, suitability_score
from src.preprocess import AMENITY_FLAGS, load_clean

st.set_page_config(page_title="ApartMatch IDSS", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_clean()


@st.cache_resource
def get_model(df_hash: int) -> FairRentModel:
    return FairRentModel().fit(get_data())


df = get_data()
model = get_model(len(df))

st.title("ApartMatch — off-campus apartment decision support")
st.caption(
    f"{len(df):,} cleaned listings · fair-rent model MAE on held-out "
    f"listings: ${model.mae_:,.0f}/mo"
)

# ---------------- Sidebar: the student's decision inputs ----------------
with st.sidebar:
    st.header("Your constraints")
    campus = st.selectbox("Campus", list(CAMPUSES))
    city, state, clat, clon = CAMPUSES[campus]

    max_budget = st.slider("Max monthly budget ($)", 400, 3500, 1400, 50)
    min_beds = st.selectbox("Bedrooms (minimum)", [0, 1, 2, 3, 4], index=1)
    max_km = st.slider("Max distance from campus (km)", 1, 40, 10)

    st.subheader("Pets")
    need_cats = st.checkbox("Must allow cats")
    need_dogs = st.checkbox("Must allow dogs")

    st.subheader("Nice-to-have amenities")
    wanted = st.multiselect("Select any", AMENITY_FLAGS, default=["Parking"])

    st.header("What matters most?")
    st.caption("These weights re-rank every listing — move them and watch the shortlist change.")
    w_price = st.slider("Low rent", 0.0, 1.0, 0.35, 0.05)
    w_dist = st.slider("Short commute", 0.0, 1.0, 0.30, 0.05)
    w_space = st.slider("More space", 0.0, 1.0, 0.10, 0.05)
    w_deal = st.slider("Below-market deal (ML)", 0.0, 1.0, 0.15, 0.05)
    w_amen = st.slider("Amenity match", 0.0, 1.0, 0.10, 0.05)

# ---------------- Filter, score, rank ----------------
pool = df[(df["cityname"] == city) & (df["state"] == state)]
cand = apply_filters(pool, max_budget, min_beds, max_km, clat, clon, need_cats, need_dogs)

if cand.empty:
    st.warning(
        "No listings satisfy these constraints. Try raising the budget or "
        "the distance limit — the counts below show which constraint binds."
    )
    st.write(
        {
            "in city": len(pool),
            "within budget": int((pool["price"] <= max_budget).sum()),
            "enough bedrooms": int((pool["bedrooms"] >= min_beds).sum()),
        }
    )
    st.stop()

deal = model.deal_score(cand)
cand = cand.assign(
    deal_score=deal.round(1),
    market_rent=model.predict_market_rent(cand).round(0),
)
cand["suitability"] = suitability_score(
    cand, max_budget, max_km, deal,
    w_price, w_dist, w_space, w_deal, wanted, w_amen,
)
ranked = cand.sort_values("suitability", ascending=False)

# ---------------- Decision panel ----------------
top = ranked.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches found", f"{len(ranked):,}")
c2.metric("Top suitability", f"{top['suitability']:.0f}/100")
c3.metric("Top pick rent", f"${top['price']:,.0f}/mo",
          f"${top['price'] - top['market_rent']:,.0f} vs market")
c4.metric("Top pick commute", f"{top['distance_km']:.1f} km")

st.subheader("Ranked shortlist")
st.caption(
    "Shortlist the top 3-5 and book viewings. A listing that stays in the "
    "top 5 across your weight settings is a robust choice."
)
show = ranked.head(15)[
    ["suitability", "title", "price", "market_rent", "deal_score",
     "bedrooms", "bathrooms", "square_feet", "distance_km", "address"]
].rename(columns={"market_rent": "predicted market rent"})
st.dataframe(show, use_container_width=True, hide_index=True)

# ---------------- Decision-tied visualizations ----------------
left, right = st.columns(2)

with left:
    st.subheader("Price vs. commute trade-off")
    st.caption(
        "Each dot is a candidate. Bottom-left dominates: cheap and close. "
        "Size = suitability. Use this to spot listings your weights undervalue."
    )
    chart_df = ranked.head(100)[
        ["distance_km", "price", "suitability", "title"]
    ].copy()
    st.scatter_chart(
        chart_df, x="distance_km", y="price", size="suitability",
        use_container_width=True,
    )

with right:
    st.subheader("Asking price vs. predicted market rent (top 10)")
    st.caption(
        "Bars below the line are priced under the model's market estimate — "
        "candidates to act on quickly."
    )
    top10 = ranked.head(10).copy()
    top10["label"] = top10["title"].str.slice(0, 24)
    bar_df = top10.set_index("label")[["price", "market_rent"]]
    st.bar_chart(bar_df, use_container_width=True)

st.subheader("Map of candidates")
st.map(ranked.head(200)[["latitude", "longitude"]].astype(float))

with st.expander("Data caveats the decision-maker should know"):
    st.markdown(
        "- Amenities are self-reported in classified ads; a missing amenity "
        "may still exist (treated here as 'not listed').\n"
        "- 42% of listings do not state a pet policy; the pet filters only "
        "keep listings that explicitly allow pets.\n"
        "- Listings snapshot is from the UCI dataset; in deployment the "
        "pipeline refreshes nightly from classified-site APIs.\n"
        "- The fair-rent model MAE is shown in the header; treat deal "
        "scores within a few points of 50 as 'at market'."
    )
