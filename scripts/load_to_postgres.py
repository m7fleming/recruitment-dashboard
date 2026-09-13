"""
Loads data_processed/all_players_model_ready.csv, player_cluster_labels.csv, and
player_predictions.csv into the four Postgres tables defined in scripts/schema.sql (players,
player_season_stats, player_cluster_labels, player_predictions) -- everything app.py reads.

Drops and recreates all four tables on every run, so it's safe to re-run after regenerating any
of the three CSVs (e.g. after scripts/data_prep.py, clustering.ipynb, or regression.ipynb). The
source CSVs are the only thing treated as authoritative; nothing in Postgres is hand-edited.

player_cluster_labels.csv and player_predictions.csv are optional -- if either hasn't been
generated yet (clustering.ipynb / regression.ipynb haven't been run), that table is skipped with
a warning rather than failing the whole load.

Requires DATABASE_URL in the environment or a .env file, e.g.:
    DATABASE_URL=postgresql://localhost:5432/football_recruitment

Run: python3 scripts/load_to_postgres.py
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / 'data_processed' / 'all_players_model_ready.csv'
CLUSTER_LABELS_CSV_PATH = REPO_ROOT / 'data_processed' / 'player_cluster_labels.csv'
PREDICTIONS_CSV_PATH = REPO_ROOT / 'data_processed' / 'player_predictions.csv'
SCHEMA_PATH = REPO_ROOT / 'scripts' / 'schema.sql'

# Columns pulled onto the players table (bio attributes, treated as stable per player_id).
PLAYER_SOURCE_COLS = {
    'player id': 'player_id',
    'Name': 'name',
    'name_key': 'name_key',
    'dob': 'dob',
    'country': 'country',
    'preferred_foot': 'preferred_foot',
    'height': 'height_cm',
    'weight': 'weight_kg',
}

# Non-metric columns pulled onto player_season_stats; every other CSV column is a per-90 metric
# and is loaded as-is (see schema.sql's "KNOWN DATA QUALITY ISSUES" comment).
FACT_SOURCE_COLS = {
    'player id': 'player_id',
    'league': 'league',
    'season': 'season',
    'Team': 'team_name',
    'team id': 'sofascore_team_id',
    'position': 'position',
    'market_value_eur': 'market_value_eur',
    'contract_until': 'contract_until',
    'appearances_total': 'appearances_total',
    'minutesPlayed_total': 'minutes_played_total',
}


def load_dataframe():
    df = pd.read_csv(CSV_PATH)
    print(f'Loaded {CSV_PATH}: {df.shape}')

    df['dob'] = pd.to_datetime(df['dob'], utc=True).dt.date
    df['contract_until'] = pd.to_datetime(df['contract_until'], utc=True).dt.date
    df['height'] = df['height'].round().astype('Int64')
    df['weight'] = df['weight'].round().astype('Int64')
    df['appearances_total'] = df['appearances_total'].astype('Int64')
    df['minutesPlayed_total'] = df['minutesPlayed_total'].astype('Int64')
    df['team id'] = df['team id'].astype('Int64')

    return df


def build_players(df):
    players = (
        df[list(PLAYER_SOURCE_COLS)]
        .rename(columns=PLAYER_SOURCE_COLS)
        .drop_duplicates(subset='player_id', keep='first')
        .reset_index(drop=True)
    )
    print(f'Built players: {players.shape[0]} rows')
    return players


def build_player_season_stats(df):
    metric_cols = [c for c in df.columns if c not in PLAYER_SOURCE_COLS and c not in FACT_SOURCE_COLS]
    fact = df[list(FACT_SOURCE_COLS) + metric_cols].rename(columns=FACT_SOURCE_COLS)
    print(f'Built player_season_stats: {fact.shape[0]} rows, {len(metric_cols)} metric columns')
    return fact


def build_cluster_labels():
    if not CLUSTER_LABELS_CSV_PATH.exists():
        print(f'SKIPPING player_cluster_labels: {CLUSTER_LABELS_CSV_PATH} not found '
              f'(run clustering.ipynb first)')
        return None
    df = pd.read_csv(CLUSTER_LABELS_CSV_PATH)
    labels = df[['player id', 'league', 'season', 'position', 'kmeans_cluster_raw', 'cluster_label']].rename(
        columns={'player id': 'player_id'}
    )
    print(f'Built player_cluster_labels: {labels.shape[0]} rows')
    return labels


def build_predictions():
    if not PREDICTIONS_CSV_PATH.exists():
        print(f'SKIPPING player_predictions: {PREDICTIONS_CSV_PATH} not found '
              f'(run regression.ipynb first)')
        return None
    df = pd.read_csv(PREDICTIONS_CSV_PATH)
    preds = df[['player id', 'league', 'season', 'position', 'market_value_eur',
                'predicted_market_value_eur', 'log_residual', 'undervalued_score',
                'market_value_data_quality_flag', 'model_used', 'model_test_r2']].rename(
        columns={'player id': 'player_id'}
    )
    preds['market_value_eur'] = preds['market_value_eur'].round().astype('Int64')
    print(f'Built player_predictions: {preds.shape[0]} rows')
    return preds


def main():
    load_dotenv(REPO_ROOT / '.env')
    database_url = os.environ['DATABASE_URL']
    engine = create_engine(database_url)

    with engine.begin() as conn:
        conn.execute(text('DROP TABLE IF EXISTS player_predictions'))
        conn.execute(text('DROP TABLE IF EXISTS player_cluster_labels'))
        conn.execute(text('DROP TABLE IF EXISTS player_season_stats'))
        conn.execute(text('DROP TABLE IF EXISTS players'))
        conn.execute(text(SCHEMA_PATH.read_text()))
    print('Recreated schema from scripts/schema.sql')

    df = load_dataframe()
    players = build_players(df)
    fact = build_player_season_stats(df)
    cluster_labels = build_cluster_labels()
    predictions = build_predictions()

    players.to_sql('players', engine, if_exists='append', index=False)
    fact.to_sql('player_season_stats', engine, if_exists='append', index=False)
    if cluster_labels is not None:
        cluster_labels.to_sql('player_cluster_labels', engine, if_exists='append', index=False)
    if predictions is not None:
        predictions.to_sql('player_predictions', engine, if_exists='append', index=False)

    with engine.connect() as conn:
        n_players = conn.execute(text('SELECT count(*) FROM players')).scalar()
        n_fact = conn.execute(text('SELECT count(*) FROM player_season_stats')).scalar()
        n_clusters = conn.execute(text('SELECT count(*) FROM player_cluster_labels')).scalar()
        n_preds = conn.execute(text('SELECT count(*) FROM player_predictions')).scalar()
    print(f'Loaded: players={n_players}, player_season_stats={n_fact}, '
          f'player_cluster_labels={n_clusters}, player_predictions={n_preds}')


if __name__ == '__main__':
    main()
