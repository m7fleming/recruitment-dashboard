"""
CSV backend for app.py: reproduces exactly what the Postgres query returns,
reading the three data_processed CSVs directly. Used when DATA_SOURCE is not
set to 'postgres', so the dashboard runs on a clone with no database.

Mirrors scripts/load_to_postgres.py's column renames and column selection,
and scripts/schema.sql's join keys. Any change there must be made here too.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / 'data_processed'

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

JOIN_KEYS = ['player_id', 'league', 'season', 'position']


def load_from_csv(current_season):
    df = pd.read_csv(DATA_DIR / 'all_players_model_ready.csv')
    df['dob'] = pd.to_datetime(df['dob'], utc=True).dt.date
    df['contract_until'] = pd.to_datetime(df['contract_until'], utc=True).dt.date
    df['height'] = df['height'].round().astype('Int64')
    df['weight'] = df['weight'].round().astype('Int64')
    df['appearances_total'] = df['appearances_total'].astype('Int64')
    df['minutesPlayed_total'] = df['minutesPlayed_total'].astype('Int64')
    df['team id'] = df['team id'].astype('Int64')

    metric_cols = [c for c in df.columns
                   if c not in PLAYER_SOURCE_COLS and c not in FACT_SOURCE_COLS]
    fact = df[list(FACT_SOURCE_COLS) + metric_cols].rename(columns=FACT_SOURCE_COLS)

    bio = (df[list(PLAYER_SOURCE_COLS)]
           .rename(columns=PLAYER_SOURCE_COLS)
           .drop_duplicates(subset='player_id', keep='first'))
    bio = bio[['player_id', 'name', 'dob', 'country',
               'preferred_foot', 'height_cm', 'weight_kg']]

    out = fact[fact['season'] == current_season].merge(bio, on='player_id', how='inner')

    clusters = pd.read_csv(DATA_DIR / 'player_cluster_labels.csv')
    clusters = clusters[['player id', 'league', 'season', 'position',
                         'kmeans_cluster_raw', 'cluster_label']].rename(
        columns={'player id': 'player_id'})
    out = out.merge(clusters, on=JOIN_KEYS, how='left')

    preds = pd.read_csv(DATA_DIR / 'player_predictions.csv')
    preds = preds[['player id', 'league', 'season', 'position',
                   'predicted_market_value_eur', 'log_residual', 'undervalued_score',
                   'market_value_data_quality_flag', 'model_used',
                   'model_test_r2']].rename(columns={'player id': 'player_id'})
    out = out.merge(preds, on=JOIN_KEYS, how='left')

    return out