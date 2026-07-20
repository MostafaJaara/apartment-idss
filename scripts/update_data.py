"""Refresh the listings file and retrain the fair-rent model.

In deployment this runs nightly (cron / GitHub Actions). It:
  1. Ingests a new raw listings export (same schema as the UCI file) --
     in production this would call classified-site APIs or a scraper.
  2. Re-runs the cleaning pipeline.
  3. Retrains the FairRentModel and reports held-out MAE, so a drop in
     accuracy after a refresh is visible before the model is served.

Usage:
    python scripts/update_data.py path/to/new_listings.csv
    python scripts/update_data.py            # revalidates the current file
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import FairRentModel  # noqa: E402
from src.preprocess import RAW_PATH, clean, load_raw  # noqa: E402


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_PATH
    print(f"Ingesting {src} ...")
    raw = load_raw(src)
    df = clean(raw)
    print(f"  raw rows: {len(raw):,} -> clean rows: {len(df):,}")

    if src != RAW_PATH:
        RAW_PATH.write_bytes(src.read_bytes())
        print(f"  installed as {RAW_PATH}")

    model = FairRentModel().fit(df)
    print(f"  retrained fair-rent model, held-out MAE: ${model.mae_:,.0f}/mo")
    print("Done. Restart the app (or clear Streamlit cache) to serve fresh data.")


if __name__ == "__main__":
    main()
