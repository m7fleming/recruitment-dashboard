# Transfer Decision Support Dashboard

A Streamlit dashboard for Premier League recruitment analysis. Players are grouped
into position-specific style archetypes by K-Means clustering, and market value is
predicted per position, with players ranked by how underpriced they look relative to
same-league peers at a comparable value level.

Built on 4,749 player-seasons from the top five European leagues (2023/24 to 2025/26),
combining Sofascore per-90 performance data with Transfermarkt valuations. The
dashboard shows the 2025/26 season only; earlier seasons trained the models.

## Quickstart

```
git clone https://github.com/m7fleming/recruitment-dashboard.git
cd recruitment-dashboard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
On Windows, use `.venv\Scripts\activate` in place of the `source` command.

Built and tested on Python 3.14. Earlier versions will likely work, but the pinned
dependency versions in `requirements.txt` were resolved against 3.14.

No database required. The dashboard reads the CSVs in `data_processed/` by default.

## Optional: Postgres backend

The project was originally built against PostgreSQL, and that path still works:

```
createdb football_recruitment
cp .env.example .env          # set DATABASE_URL
python3 scripts/load_to_postgres.py
DATA_SOURCE=postgres streamlit run app.py
```

Both backends return an identical dataframe.

## How it works

**Clustering.** Per position, features are standardised, reduced by PCA (~80% variance
retained), then clustered with K-Means, K chosen by the elbow on inertia. Market value
is excluded from the feature set, so archetypes are derived from playing style alone.
Ward hierarchical clustering was run as a stability check (ARI 0.40 to 0.51). Sixteen
archetypes result, plus one goalkeeper bucket retained as a documented artifact rather
than a real style.

**Valuation.** `log1p(market_value_eur)` is modelled per position, comparing linear
regression, Random Forest and XGBoost on an 80/20 split. Predictions shown in the
dashboard are out-of-fold cross-validated, Duan-smearing corrected for the
back-transform, then detrended against value and league.

| Position | Best model | Test R² |
|---|---|---|
| Goalkeeper | XGBoost | 0.427 |
| Defender | Linear | 0.349 |
| Midfielder | Linear | 0.352 |
| Forward | Linear | 0.367 |

## Limitations

- **R² of 0.22 to 0.43** means real uncertainty around every individual estimate.
  Predicted values are indicative, not valuations.
- **A 900-minute minimum** (roughly ten matches) is applied before modelling. This is
  standard practice for per-90 stability, but it removes exactly the low-minute,
  out-of-favour players who are often the most available and most mispriced. The tool
  systematically cannot see them.
- **Squad strength is uncontrolled.** Players at dominant clubs score above their
  league average, plausibly because weaker domestic opposition inflates per-90 output.
  Controlling for league did not resolve this.
- **Positions are broad** (G/D/M/F), so within-position role variation is captured by
  clustering rather than by the position label itself.
- **Historical valuations are partially patched.** Transfermarkt stamps a player's
  current valuation across past seasons; roughly 70% of affected rows were corrected
  from a historical scrape, the rest were not.
- **Rankings are least reliable at the value extremes**, where regression to the mean
  is strongest.

## Data

Derived from Sofascore and Transfermarkt. Redistributed here for non-commercial
demonstration of the tool only. The MIT licence covers the code in this repository,
not the underlying data.

## Contributing

Contributions by pull request.
