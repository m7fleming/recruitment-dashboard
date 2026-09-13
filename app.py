"""
Streamlit dashboard for the transfer decision support system.

Reads from data_processed/*.csv by default. Set DATA_SOURCE=postgres to read from
Postgres instead (see scripts/schema.sql and scripts/load_to_postgres.py) -- both
paths return an identical dataframe. Never retrains or re-fits anything:
market_value_eur / predicted_market_value_eur come from regression.ipynb's
out-of-fold cross-validated predictions, and models/*.joblib are loaded only to show
what's actually powering those numbers, not to re-score data live.

Run: streamlit run app.py
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

from csv_backend import load_from_csv

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / 'models'

CURRENT_SEASON = '25/26'

POSITION_ORDER = ['G', 'D', 'M', 'F']
POSITION_NAMES = {'G': 'Goalkeeper', 'D': 'Defender', 'M': 'Midfielder', 'F': 'Forward'}
POSITION_COLORS = {'G': 'darkorange', 'D': 'yellow', 'M': 'limegreen', 'F': 'dodgerblue'}

# Per-90 columns worth labelling explicitly wherever they appear in the UI, so a viewer doesn't
# mistake them for season totals
# Friendlier display names for a few specific columns; stat_label() below handles the general
# "is this a per-90 rate" labelling for everything else.
STAT_DISPLAY_NAMES = {
    'goals_per90': 'Goals', 'assists_per90': 'Assists',
    'expectedGoals_per90': 'xG', 'expectedAssists_per90': 'xA',
    'goals_overperformance_per90': 'Goals overperformance',
    'assists_overperformance_per90': 'Assists overperformance',
}

# Columns that are true totals or percentages, not per-90 rates -- everything else in this
# dataset's Sofascore-derived metric columns is a per-90 rate despite plain-sounding names like
# "totalShots" or "tackles", so the default assumption below is
# "per 90" and these are the deliberate exceptions.
NOT_PER90 = {
    # 'minutes_played_total' is the Postgres column name (renamed from the source CSV's
    # 'minutesPlayed_total' by scripts/load_to_postgres.py) -- this dict must match what's
    # actually in `df`, not the pre-rename CSV name, or this column silently gets mislabeled.
    'minutes_played_total', 'appearances_total', 'age', 'rating', 'totalRating', 'countRating',
    'height_cm', 'weight_kg',
}


def stat_label(col):
    """Human-readable, correctly-labelled name for a stat column -- explicit about per-90 rates
    so a viewer never mistakes one for a season total."""
    name = STAT_DISPLAY_NAMES.get(col, col)
    if col in NOT_PER90 or 'Percentage' in col:
        return name
    return f'{name} (per 90)'


# From clustering.ipynb's own per-cluster profiling -- the human-assigned interpretation of what
# each cluster represents, not something derivable from the data at runtime. Keep in sync with
# that notebook's "Summary" section if clustering.ipynb is ever re-run with different K per
# position. `abbrev` is None for clusters that aren't real playing styles (G_low_signal) --
# deliberately not given a short name, per the naming review that reclassified it.
CLUSTER_INFO = {
    'G_0': {
        'abbrev': 'HVD', 'name': 'High Volume Distributor',
        'description': 'High passing volume, especially long balls out from the back.',
    },
    'G_2': {
        'abbrev': 'SHST', 'name': 'Shot Stopper',
        'description': 'High pass accuracy, clean-sheet associated. The largest goalkeeper '
                        'group.',
    },
    'G_3': {
        'abbrev': 'SWK', 'name': 'Sweeper Keeper',
        'description': 'Aggressive, engages in duels and dribbles outside the box.',
    },
    'G_low_signal': {
        'abbrev': None, 'name': 'Low Signal (not a real style)',
        'description': (
            'Not a real playing style. This bucket combines THREE confirmed data artifacts '
            'from clustering.ipynb, not two: a near-zero-variance statistical artifact (a '
            'handful of keepers with one incidental shot event each), a single extreme '
            'outlier player, and a 21-player cluster originally proposed as a '
            '"creative/playmaking goalkeeper" archetype that scrutiny found to be driven by a '
            'single incidental assist or big-chance event for 71% of its members, with no '
            'coherent secondary signal and almost no season-to-season persistence for the same '
            'player (1 of 15 eligible members recurred). Do not present this as a goalkeeper '
            'archetype.'
        ),
    },
    'D_0': {
        'abbrev': 'AFB', 'name': 'Attacking Full-Back',
        'description': 'Crosses, duels, and possession loss from advanced positions.',
    },
    'D_1': {
        'abbrev': 'BPD', 'name': 'Ball-Playing Defender',
        'description': 'Deep possession-recycling centre-back, high passing volume from own '
                        'half.',
    },
    'D_2': {
        'abbrev': 'CWB', 'name': 'Creative Wing-Back',
        'description': 'Elevated expected assists and key passes for a defender.',
    },
    'D_3': {
        'abbrev': 'STP', 'name': 'Stopper Centre-Back',
        'description': 'Traditional, aerially-dominant. The largest defender group.',
    },
    'D_4': {
        'abbrev': 'PTD', 'name': 'Penalty-Taking Defender',
        'description': (
            'A real, retained archetype (n=18) -- 67% of members took 2+ penalties in a '
            'season (not a single incidental event) and share a coherent secondary '
            'goal-threat profile (elevated xG, goal conversion, shots on target). '
            'Caveat, stated prominently rather than as a footnote: only 1 of 13 multi-season '
            'members recurs in this cluster across consecutive seasons. Penalty duty is a '
            'squad-level role that can rotate between players season to season (a new '
            'signing arrives, a manager reassigns it) -- so membership here may reflect this '
            'season\'s squad role more than a durable individual trait. Treat it as a real '
            'but unstable label, not a fixed playing style like the other defender clusters.'
        ),
    },
    'M_0': {
        'abbrev': 'AP', 'name': 'Attacking Playmaker',
        'description': 'High combined goals+assists and expected assists.',
    },
    'M_1': {
        'abbrev': 'DLP', 'name': 'Deep-Lying Playmaker',
        'description': 'High passing volume, touches.',
    },
    'M_2': {
        'abbrev': 'BWM', 'name': 'Ball-Winning Midfielder',
        'description': 'Clearances, interceptions, tackles. The largest midfielder group.',
    },
    'M_3': {
        'abbrev': 'SS', 'name': 'Shadow Striker',
        'description': 'Shoots from inside the box, higher turnover risk (dispossessed, '
                        'offside).',
    },
    'F_0': {
        'abbrev': 'P', 'name': 'Poacher',
        'description': 'High xG and goals from inside the box.',
    },
    'F_1': {
        'abbrev': 'IC', 'name': 'Inside Creator',
        'description': 'Build-up passing in the final third rather than direct goal threat.',
    },
    'F_2': {
        'abbrev': 'PF', 'name': 'Pressing Forward',
        'description': 'High ball recovery and tackling work-rate.',
    },
    'F_3': {
        'abbrev': 'TM', 'name': 'Target Man',
        'description': 'Physical, aerially-dominant. The largest forward group.',
    },
}

# Top-8 distinguishing per-90 stats per cluster, by z-score vs. the position average -- computed
# directly from clustering.ipynb's own feature set and methodology (see that notebook's
# profile_clusters() function), not re-derived live in this app. G_low_signal deliberately has
# no entry: its "distinguishing" stats are the confirmed artifacts documented above, not a real
# profile worth surfacing as recruitment-relevant metrics.
CLUSTER_KEY_STATS = {
    'G_0': ['totalOppositionHalfPasses', 'inaccuratePasses', 'possessionLost', 'totalLongBalls',
            'accurateOppositionHalfPasses', 'goalKicks', 'accurateFinalThirdPasses',
            'accurateLongBalls'],
    'G_2': ['accuratePassesPercentage', 'accurateOwnHalfPasses', 'cleanSheet',
            'totalOwnHalfPasses', 'accurateLongBallsPercentage', 'accuratePasses',
            'goals_overperformance_per90', 'dispossessed'],
    'G_3': ['duelLost', 'successfulDribbles', 'totalContest', 'fouls', 'penaltyConceded',
            'groundDuelsWon', 'saves', 'savedShotsFromOutsideTheBox'],
    'D_0': ['possessionLost', 'groundDuelsWon', 'dispossessed', 'totalCross', 'dribbledPast',
            'totalContest', 'tackles', 'tacklesWon'],
    'D_1': ['accuratePasses', 'totalPasses', 'accurateOwnHalfPasses', 'totalOwnHalfPasses',
            'touches', 'accurateOppositionHalfPasses', 'accuratePassesPercentage', 'rating'],
    'D_2': ['expectedAssists_per90', 'keyPasses', 'totalAttemptAssist', 'totalCross',
            'bigChancesCreated', 'accurateCrosses', 'goalsAssistsSum', 'shotsFromOutsideTheBox'],
    'D_3': ['clearances', 'aerialDuelsWon', 'aerialLost', 'aerialDuelsWonPercentage',
            'totalDuelsWonPercentage', 'totalLongBalls', 'goalKicks', 'goalsConceded'],
    'D_4': ['penaltiesTaken', 'penaltyGoals', 'penaltyConversion', 'attemptPenaltyPost',
            'attemptPenaltyTarget', 'expectedGoals_per90', 'rightFootGoals', 'goals_per90'],
    'M_0': ['goalsAssistsSum', 'expectedAssists_per90', 'keyPasses', 'totalAttemptAssist',
            'goals_per90', 'expectedGoals_per90', 'shotsOnTarget', 'totalShots'],
    'M_1': ['totalPasses', 'accuratePasses', 'touches', 'accurateOwnHalfPasses',
            'totalOwnHalfPasses', 'accurateOppositionHalfPasses', 'totalOppositionHalfPasses',
            'accurateLongBalls'],
    'M_2': ['clearances', 'interceptions', 'tackles', 'tacklesWon', 'aerialDuelsWonPercentage',
            'totalDuelsWonPercentage', 'goalsConceded', 'goalsConcededInsideTheBox'],
    'M_3': ['shotsFromInsideTheBox', 'duelLost', 'dispossessed', 'offsides', 'totalContest',
            'totalShots', 'bigChancesMissed', 'shotsOnTarget'],
    'F_0': ['expectedGoals_per90', 'goals_per90', 'goalsFromInsideTheBox', 'goalsAssistsSum',
            'shotsOnTarget', 'shotsFromInsideTheBox', 'rightFootGoals', 'bigChancesMissed'],
    'F_1': ['totalOppositionHalfPasses', 'accurateOppositionHalfPasses', 'totalPasses',
            'accurateFinalThirdPasses', 'accuratePasses', 'keyPasses', 'totalAttemptAssist',
            'touches'],
    'F_2': ['ballRecovery', 'groundDuelsWon', 'tackles', 'tacklesWon', 'dribbledPast',
            'totalContest', 'scoringFrequency', 'interceptions'],
    'F_3': ['aerialLost', 'aerialDuelsWon', 'duelLost', 'fouls', 'clearances', 'offsides',
            'goalsConceded', 'goalsConcededInsideTheBox'],
}

# Non-style columns to exclude when building the similarity feature set for Find Similar Players
# -- the same exclusion logic as clustering.ipynb's EXCLUDE set (identifiers, bio, market value,
# playing-time volume, near-constant per-90-of-a-count fields), translated to this app's actual
# (post-Postgres-rename) column names so similarity is computed on the same per-90 style profile
# clustering itself uses, not re-derived or hand-picked here.
SIMILARITY_FEATURE_EXCLUDE = {
    'player_id', 'league', 'season', 'position', 'team_name', 'sofascore_team_id',
    'dob', 'country', 'preferred_foot', 'height_cm', 'weight_kg', 'contract_until',
    'market_value_eur', 'appearances_total', 'minutes_played_total',
    'appearances', 'matchesStarted', 'totwAppearances', 'outfielderBlocks',
    'name', 'kmeans_cluster_raw', 'cluster_label', 'predicted_market_value_eur',
    'log_residual', 'undervalued_score', 'market_value_data_quality_flag',
    'model_used', 'model_test_r2', 'age',
}


@st.cache_resource
def get_engine():
    load_dotenv(REPO_ROOT / '.env')
    return create_engine(os.environ['DATABASE_URL'])


@st.cache_data(ttl=600)
def load_data():
    # Only the current season is browsable/searchable -- earlier seasons exist purely to train
    # the model (regression.ipynb still uses all three), not as recruitment targets. Filtering
    # here, at the single point everything else reads from, means every tab/view (browse,
    # undervalued, cluster peer comparison) is automatically scoped to current-season players
    # without needing to repeat the filter.
    if os.environ.get('DATA_SOURCE', 'csv').lower() == 'postgres':
        query = """
            SELECT
                s.*,
                p.name, p.dob, p.country, p.preferred_foot, p.height_cm, p.weight_kg,
                cl.kmeans_cluster_raw, cl.cluster_label,
                pr.predicted_market_value_eur, pr.log_residual, pr.undervalued_score,
                pr.market_value_data_quality_flag, pr.model_used, pr.model_test_r2
            FROM player_season_stats s
            JOIN players p USING (player_id)
            LEFT JOIN player_cluster_labels cl USING (player_id, league, season, position)
            LEFT JOIN player_predictions pr USING (player_id, league, season, position)
            WHERE s.season = %(current_season)s
        """
        df = pd.read_sql(query, get_engine(), params={'current_season': CURRENT_SEASON})
    else:
        df = load_from_csv(CURRENT_SEASON)

    season_start_year = df['season'].str.split('/').str[0].astype(int) + 2000
    season_start = pd.to_datetime(season_start_year.astype(str) + '-08-01', utc=True)
    dob = pd.to_datetime(df['dob'], utc=True, errors='coerce')
    df['age'] = ((season_start - dob).dt.days / 365.25).round(1)

    return df

@st.cache_data(ttl=600)
def load_model_metadata():
    path = MODELS_DIR / 'model_metadata.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_model(position):
    path = MODELS_DIR / f'{position}_model.joblib'
    if not path.exists():
        return None
    return joblib.load(path)


def format_eur(value):
    if pd.isna(value):
        return 'n/a'
    return f'€{value:,.0f}'


def cluster_abbrev(cluster_label):
    """Short abbreviation for a cluster label (e.g. 'AFB'), or a plain-language stand-in for
    clusters that aren't real playing styles."""
    if pd.isna(cluster_label):
        return '—'
    info = CLUSTER_INFO.get(cluster_label)
    if not info:
        return cluster_label
    return info['abbrev'] or 'Low signal'


def cluster_name(cluster_label):
    info = CLUSTER_INFO.get(cluster_label)
    return info['name'] if info else cluster_label


def cluster_description(cluster_label):
    if pd.isna(cluster_label):
        return 'No cluster assignment (player not present in clustering.ipynb\'s output).'
    info = CLUSTER_INFO.get(cluster_label)
    if not info:
        return f'Unrecognised cluster label: {cluster_label}'
    return info['description']


def render_selectable_table(table, row_ids, key_prefix, column_config=None):
    """st.dataframe with single-row selection that jumps to Player Profile on select.

    `row_ids` must be a sequence of stable IDs (e.g. player_id) aligned 1:1 with `table`'s row
    order -- position i in `row_ids` identifies `table.iloc[i]`. Used to key the widget by the
    current result set's identity, not just a fixed string.

    This matters beyond convenience: Streamlit's soft rerun re-sends a dataframe widget's last
    frontend selection index into session_state *before* the script body runs on every rerun,
    same as the filter-widget issue the Reset Filters button hit. If the filters change and the
    table shrinks (or just reorders), a stale selection index can point past the end of the new
    table -- IndexError -- or, even if still in-bounds by coincidence, silently refer to a
    completely different row than the one the user actually clicked. Suffixing the key with a
    hash of the row identities means any change to the underlying result set produces a brand
    new widget identity with no frontend state to restore, so a stale selection can't survive a
    filter change at all. The bounds check below is defense in depth on top of that, not a
    substitute for it -- a key collision (same row set, different order) is still conceivable.
    """
    sig = hash(tuple(row_ids))
    event = st.dataframe(
        table, use_container_width=True, hide_index=True, column_config=column_config or {},
        on_select='rerun', selection_mode='single-row', key=f'{key_prefix}_{sig}',
    )
    if event and event.selection.rows:
        idx = event.selection.rows[0]
        if idx < len(table) and 'Name' in table.columns:
            picked_name = table.iloc[idx]['Name']
            st.session_state['jump_to_player'] = picked_name
            st.info(f'Selected **{picked_name}** -- open the Player Profile tab to view their '
                    f'full profile.', icon='\U0001f464')


st.set_page_config(page_title='Transfer Decision Support', layout='wide')
st.title('Transfer Decision Support Dashboard')
st.caption(
    f'Player search, style archetypes, and predicted-vs-actual market value, built on top of '
    f'clustering.ipynb and regression.ipynb. All per-90 stats are labelled explicitly -- they are '
    f'rates, not season totals. Showing {CURRENT_SEASON} only -- earlier seasons trained the '
    f'model but aren\'t recruitment targets.'
)

try:
    df = load_data()
except Exception as e:
    if os.environ.get('DATA_SOURCE', 'csv').lower() == 'postgres':
        st.error(
            f'Could not load data from Postgres: {e}\n\n'
            f'Check DATABASE_URL in .env, that Postgres is running, and that '
            f'`python3 scripts/load_to_postgres.py` has been run.'
        )
    else:
        st.error(
            f'Could not load data from CSV: {e}\n\n'
            f'Check that data_processed/ contains all_players_model_ready.csv, '
            f'player_cluster_labels.csv and player_predictions.csv.'
        )
    st.stop()

model_metadata = load_model_metadata()

SIMILARITY_FEATURE_COLS = [c for c in df.columns if c not in SIMILARITY_FEATURE_EXCLUDE]

# --- Cluster key: accessible without cluttering any single view --------------------------------
with st.expander('\U0001f4d6 Cluster key: abbreviations, names, and descriptions'):
    for pos in POSITION_ORDER:
        st.markdown(f'**{POSITION_NAMES[pos]}s**')
        for label, info in CLUSTER_INFO.items():
            if not label.startswith(pos):
                continue
            title = info['name'] if info['abbrev'] is None else f"{info['abbrev']} -- {info['name']}"
            st.markdown(f'- **{title}** (`{label}`): {info["description"]}')

# --- Sidebar filters -----------------------------------------------------------------------
st.sidebar.header('Filters')
# Resetting via st.session_state.pop(key) + st.rerun() doesn't reliably clear widgets --
# Streamlit's soft rerun re-sends each widget's current frontend value into session_state before
# the script body runs, which silently undoes the pop. The robust fix is to give every filter
# widget a key that includes a counter, and bump the counter on reset -- that forces genuinely
# new widget identities next run, which have no frontend state to restore and fall back to
# `default=`.
if 'filter_reset_gen' not in st.session_state:
    st.session_state['filter_reset_gen'] = 0
if st.sidebar.button('Reset filters', use_container_width=True):
    st.session_state['filter_reset_gen'] += 1
    st.rerun()
gen = st.session_state['filter_reset_gen']

leagues = st.sidebar.multiselect(
    'League', sorted(df['league'].unique()), default=[], key=f'filter_league_{gen}'
)
positions = st.sidebar.multiselect(
    'Position', POSITION_ORDER, default=[], format_func=lambda p: f'{p} -- {POSITION_NAMES[p]}',
    key=f'filter_position_{gen}',
)

age_min = float(np.floor(df['age'].min() * 2) / 2)
age_max = float(np.ceil(df['age'].max() * 2) / 2)
age_range = st.sidebar.slider(
    'Age range', age_min, age_max, (age_min, age_max), step=0.5, key=f'filter_age_{gen}'
)

value_min = int(df['market_value_eur'].min() // 1000 * 1000)
value_max = int(-(-df['market_value_eur'].max() // 1000) * 1000)  # round up to nearest 1000
budget_range = st.sidebar.slider(
    'Budget (market value, EUR)', value_min, value_max, (value_min, value_max),
    step=50_000, format='€%,d', key=f'filter_budget_{gen}',
)

# Cluster options depend on which position(s) are in scope -- a cluster label like D_0 only
# means something once you know it's a defender cluster, so don't offer M_2 alongside it unless
# midfielders are actually in scope. With no position filter set, all four positions are in
# scope, so all clusters are offered.
positions_in_scope = positions if positions else POSITION_ORDER
cluster_options = sorted(c for c in CLUSTER_INFO if c.split('_')[0] in positions_in_scope)


def cluster_filter_label(c):
    info = CLUSTER_INFO[c]
    return info['name'] if info['abbrev'] is None else f"{info['abbrev']} -- {info['name']}"


selected_clusters = st.sidebar.multiselect(
    'Cluster / playing style', cluster_options, default=[],
    format_func=cluster_filter_label, key=f'filter_cluster_{gen}',
)

filtered = df.copy()
if leagues:
    filtered = filtered[filtered['league'].isin(leagues)]
if positions:
    filtered = filtered[filtered['position'].isin(positions)]
if selected_clusters:
    filtered = filtered[filtered['cluster_label'].isin(selected_clusters)]
filtered = filtered[filtered['age'].between(*age_range)]
# NaN market_value_eur (unknown, not out of budget) always passes -- don't silently hide players
# with no known value just because a budget filter is active.
filtered = filtered[
    filtered['market_value_eur'].between(*budget_range) | filtered['market_value_eur'].isna()
]

st.sidebar.caption(f'{len(filtered)} of {len(df)} player-seasons match current filters.')

if model_metadata:
    with st.sidebar.expander('About the prediction models'):
        for pos in POSITION_ORDER:
            meta = model_metadata.get(pos)
            if not meta:
                continue
            model_obj = load_model(pos)
            loaded_note = type(model_obj).__name__ if model_obj is not None else 'not saved'
            st.markdown(
                f"**{pos} -- {POSITION_NAMES[pos]}**: {meta['model_type']} "
                f"(loaded: `{loaded_note}`), test R²={meta['test_r2']:.2f}, "
                f"trained on {meta['n_players']} player-seasons, {meta['n_features']} features."
            )
        st.caption(
            'Predictions shown throughout this dashboard are out-of-fold cross-validated '
            '(regression.ipynb) -- not produced by re-scoring with the models loaded here, which '
            'are fit on all data and would be in-sample. Loaded here for provenance only.'
        )

tab_browse, tab_undervalued, tab_profile, tab_similar = st.tabs([
    '\U0001f50d Search & Browse', '\U0001f4b0 Undervalued Players', '\U0001f464 Player Profile',
    '\U0001f501 Find Similar Players',
])

# --- Tab 1: Search & Browse -----------------------------------------------------------------
with tab_browse:
    st.subheader('Search & Browse')
    name_query = st.text_input('Search by player name', '')
    view = filtered
    if name_query:
        view = view[view['name'].str.contains(name_query, case=False, na=False)]

    # Base columns are always offered; goals/assists (per 90) are deliberately left out of the
    # default set (they're already prominent in the curated cluster stat lists below and were
    # cluttering the default view) but stay available to add back manually.
    BASE_BROWSE_COLS = {
        'name': 'Name', 'team_name': 'Team', 'league': 'League', 'season': 'Season',
        'position': 'Pos', 'age': 'Age', 'cluster_label': 'Cluster',
        'market_value_eur': 'Market Value (EUR)',
        'predicted_market_value_eur': 'Predicted (indicative)',
        'rating': 'Rating',
    }

    def top2(cluster_labels):
        seen = []
        for c in cluster_labels:
            for feat in CLUSTER_KEY_STATS.get(c, [])[:2]:
                if feat not in seen:
                    seen.append(feat)
        return seen

    all_extra_stat_options = sorted({f for feats in CLUSTER_KEY_STATS.values() for f in feats[:2]})
    extra_default = top2(selected_clusters)

    selectable_options = list(BASE_BROWSE_COLS) + [
        c for c in all_extra_stat_options if c not in BASE_BROWSE_COLS
    ]
    default_selection = list(BASE_BROWSE_COLS) + extra_default

    chosen_cols = st.multiselect(
        'Columns to show', selectable_options, default=default_selection,
        format_func=lambda c: BASE_BROWSE_COLS.get(c, stat_label(c)),
        help='Includes every cluster\'s top 2 distinguishing stats as optional columns -- '
             'filtering to a specific cluster above adds that cluster\'s top 2 here '
             'automatically.',
        key=f'browse_columns_{tuple(sorted(selected_clusters))}',
    )
    st.caption('Per-90 stats (and every other per-90 column in this dashboard) are rates, not '
               'season totals -- labelled explicitly.')

    cols_to_use = chosen_cols or list(BASE_BROWSE_COLS)
    # Sort on the raw (pre-rename) view first, keeping player_id aligned to the final row order --
    # row_ids below must correspond 1:1 with table's rows regardless of which columns were chosen
    # to display, since 'player_id' itself is never one of them.
    sort_raw_col = 'name' if 'name' in cols_to_use else cols_to_use[0]
    view_sorted = view.sort_values(sort_raw_col).reset_index(drop=True)
    row_ids = view_sorted['player_id'].tolist()

    display_names = [BASE_BROWSE_COLS.get(c, stat_label(c)) for c in cols_to_use]
    table = view_sorted[cols_to_use].rename(columns=dict(zip(cols_to_use, display_names)))
    if 'Cluster' in table.columns:
        table['Cluster'] = table['Cluster'].apply(cluster_abbrev)

    column_config = {}
    for raw_col in cols_to_use:
        label = BASE_BROWSE_COLS.get(raw_col, stat_label(raw_col))
        if raw_col in ('market_value_eur', 'predicted_market_value_eur'):
            column_config[label] = st.column_config.NumberColumn(label, format='€%,d')
        elif raw_col == 'age':
            column_config[label] = st.column_config.NumberColumn(label, format='%.1f')
        elif raw_col not in BASE_BROWSE_COLS:
            column_config[label] = st.column_config.NumberColumn(label, format='%.2f')

    render_selectable_table(table, row_ids, key_prefix='browse_table', column_config=column_config)

# --- Tab 2: Undervalued Players (the key view) -----------------------------------------------
with tab_undervalued:
    st.subheader('Ranked by Undervalued Score')
    st.markdown(
        'How much more (or less) undervalued a player looks *than other players in the same '
        'league at a similar price* -- the primary number below. Positive means the model rates '
        'them above what their price alone would suggest for a player at that value level in '
        'that league. Predicted market value is shown alongside as supporting context, not as '
        'the ranking itself -- see the methodology below -- treat it as **indicative**, not a '
        'precise valuation.'
    )

    n_flagged = int(filtered['market_value_data_quality_flag'].fillna(False).sum())
    scored = filtered[
        filtered['predicted_market_value_eur'].notna()
        & ~filtered['market_value_data_quality_flag'].fillna(False)
    ].copy()
    if n_flagged:
        st.caption(
            f'{n_flagged} player(s) excluded from this tab: confirmed `market_value_eur` '
            f'data-quality issues (e.g. a stale scraped value), not a model or ranking problem -- '
            f'Kept elsewhere in the dashboard, just not ranked here.'
        )
    scored = scored.sort_values('undervalued_score', ascending=False)

    with st.expander('⚠️ Methodology and known limitations -- read before trusting this ranking'):
        st.markdown(
            '**Why not just rank by predicted-minus-actual?** Investigated 2026-09 after a '
            'report that Cole Palmer\'s prediction looked implausibly low. Raw predicted-vs-actual '
            'residuals correlate -0.70 to -0.78 with actual value at every position -- a direct '
            'consequence of the models\' R² (0.22-0.43): cheap players get systematically '
            '*over*-predicted and expensive players systematically *under*-predicted (regression '
            'to the mean), not because of genuine mispricing. Ranking by the raw residual '
            'reproduced this: the naive top-20 had a median actual value of EUR1m against a '
            'EUR10m dataset median -- surfacing statistical shrinkage, not real recruitment '
            'leads. **The ranking is least reliable at the value extremes** for this reason. '
            '`undervalued_score` removes the value-dependent trend so cheap players are no '
            'longer structurally over-represented at the top.\n\n'
            '**Also controlled for league (2026-08-31).** An earlier version of this ranking '
            'left league uncontrolled, which mechanically inflated one league\'s share of the '
            'top of the list (England Premier League reached 62% of the 25/26 top-50 against a '
            '21.6% dataset share) -- not because PL players were genuinely more undervalued, but '
            'because PL is the highest-value league and the old detrend, fit pooling all five '
            'leagues, under-predicted how negative PL\'s residual should be at PL\'s own value '
            'range.\n\n'
            '**How `undervalued_score` is calculated:**\n'
            '1. Out-of-fold cross-validated log-value prediction per player (never fit on that '
            'player\'s own row), from a model that already includes `league` as a feature.\n'
            '2. **Duan\'s smearing correction** for the `expm1` back-transform, which otherwise '
            'systematically underestimates value (54-104% depending on position, after the fix '
            'below) -- a separate, purely multiplicative bias that does *not* explain the '
            'pattern above. Uses a **5%/95% winsorized** version of the standard estimator: a '
            'plain mean is not robust to a handful of extreme individual prediction errors, and '
            'un-winsorized it was inflating defenders\' correction furthest (2.0x plain-mean vs. '
            '1.3-1.5x for the other three positions) off a small number of outliers -- producing '
            'implausible predictions like EUR190m+ for a EUR20m player before this fix.\n'
            '3. A per-position, **league-controlled linear value-detrend**: `log_residual` '
            'regressed against actual value *and* league (one shared slope on value, plus a '
            'separate intercept per league), and players ranked by their deviation from that '
            'trend rather than the raw residual. This is what actually removes the '
            'value-correlated bias (confirmed: post-detrend correlation with value is 0.0000 at '
            'every position, by construction, and mean `undervalued_score` is 0.0 per league on '
            'the full fit pool).\n\n'
            '**The trade-off this creates**: `undervalued_score` now answers "underpriced '
            'relative to *same-league* peers at a comparable value level" -- not a cross-league '
            'comparison. The cost is that a cross-league bargain-hunting question (e.g. "is this '
            'Ligue 1 midfielder a better buy than that Bundesliga one") can no longer be read '
            'directly off `undervalued_score` -- that kind of comparison now requires your own '
            'judgement about how the leagues translate, using `predicted_market_value_eur` '
            '(which still reflects each league\'s valuation premium) or the raw scatter plot '
            'below as supporting context.\n\n'
            '**A separate, still-open bias**: clubs with unusually strong squads (e.g. Bayern '
            'Munich, Manchester City, PSG, Real Madrid) show `undervalued_score` well above their '
            'own league\'s average -- plausibly weaker domestic opposition inflating their '
            'players\' per-90 stats. Controlling for league did **not** fix this, and if anything '
            'made it more visible -- it is a club-level effect nested inside league that a '
            'league intercept cannot capture. A squad-strength/team-context feature is the '
            'candidate fix, currently on hold pending a scope decision. Until then, '
            'treat heavy representation from these clubs in the ranking as a known, unresolved '
            'artifact, not a confirmed recruitment signal.\n\n'
            'Even after all of the above, `predicted_market_value_eur` is indicative, not '
            'precise -- R²=0.22-0.43 means real uncertainty remains around every individual '
            'point estimate. That\'s why `undervalued_score` (not the predicted value) is what '
            'this tab ranks by; treat any single number as a starting point for scouting, not a '
            'verdict.'
        )

    top_n = st.slider('Show top N most undervalued', 5, 50, 15)
    table_cols = {
        'name': 'Name', 'team_name': 'Team', 'league': 'League', 'season': 'Season',
        'position': 'Pos', 'cluster_label': 'Cluster',
        'undervalued_score': 'Undervalued Score',
        'market_value_eur': 'Actual (EUR)', 'predicted_market_value_eur': 'Predicted (indicative)',
    }
    top_rows = scored.head(top_n)
    row_ids = top_rows['player_id'].tolist()
    top = top_rows[list(table_cols)].rename(columns=table_cols).reset_index(drop=True)
    top['Cluster'] = top['Cluster'].apply(cluster_abbrev)
    render_selectable_table(
        top, row_ids, key_prefix='undervalued_table',
        column_config={
            'Actual (EUR)': st.column_config.NumberColumn('Actual (EUR)', format='€%,d'),
            'Predicted (indicative)': st.column_config.NumberColumn(
                'Predicted (indicative)', format='€%,d'
            ),
            'Undervalued Score': st.column_config.NumberColumn('Undervalued Score', format='%.2f'),
        },
    )

    with st.expander('Predicted vs. actual scatter (illustrative -- not the ranking basis)'):
        st.caption(
            'This raw predicted-vs-actual view is exactly the comparison that\'s biased toward '
            'cheap players (see methodology above); the table above uses the corrected '
            'undervalued_score, not this plot. Shown for transparency, not as a second ranking.'
        )
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for pos in POSITION_ORDER:
            sub = scored[scored['position'] == pos]
            ax.scatter(sub['market_value_eur'] / 1_000_000,
                       sub['predicted_market_value_eur'] / 1_000_000,
                       alpha=0.5, s=16, color=POSITION_COLORS[pos], label=pos)
        lims = [0, max(scored['market_value_eur'].max(), scored['predicted_market_value_eur'].max()) / 1_000_000]
        ax.plot(lims, lims, color='black', linestyle='--', linewidth=1, label='predicted = actual')
        ax.set_xlabel('Actual Market Value (EUR millions)')
        ax.set_ylabel('Predicted Market Value (EUR millions, indicative)')
        ax.set_title('Predicted vs. Actual (current filters)')
        ax.legend(fontsize=8)
        ax.grid(False)
        st.pyplot(fig)

# --- Tab 3: Player Profile -------------------------------------------------------------------
with tab_profile:
    st.subheader('Player Profile')
    all_names = sorted(df['name'].unique())
    # A row selected in Search & Browse jumps here: pre-seed the selectbox's own state on the
    # rerun that follows a selection, then drop the one-shot flag so a later manual change in
    # this tab isn't overridden by a stale jump target.
    if 'jump_to_player' in st.session_state:
        st.session_state['profile_player_select'] = st.session_state.pop('jump_to_player')
    selected_name = st.selectbox(
        'Select a player', all_names, index=None, placeholder='Type to search...',
        key='profile_player_select',
    )

    if selected_name is None:
        st.info('Select a player above to see their cluster profile, predicted value, and how '
                'they compare to their cluster.')
    else:
        player_rows = df[df['name'] == selected_name].sort_values('season')
        season_options = player_rows['season'] + ' -- ' + player_rows['league']
        selected_season_label = st.selectbox('Season', season_options)
        row = player_rows[season_options == selected_season_label].iloc[0]

        pos = row['position']
        # Undervalued Score leads (primary claim); predicted value is supporting context, labelled
        # "indicative" -- see the Undervalued Players tab for why neither is a precise valuation.
        c1, c2, c3, c4 = st.columns(4)
        if row.get('market_value_data_quality_flag'):
            c1.metric('Undervalued Score', 'n/a')
            c1.caption('⚠️ Excluded: known market_value_eur data-quality issue')
        elif pd.notna(row.get('undervalued_score')):
            c1.metric('Undervalued Score', f"{row['undervalued_score']:+.2f}",
                      help='Value- and league-adjusted -- this is what the Undervalued Players tab '
                           'ranks by. Positive means more undervalued than other players in the '
                           'same league at a similar price.')
        c1.metric('Position', f"{pos} -- {POSITION_NAMES.get(pos, pos)}")
        c2.metric('Actual Market Value', format_eur(row['market_value_eur']))
        c2.metric('Predicted (indicative)', format_eur(row['predicted_market_value_eur']),
                  help='Out-of-fold model prediction, calibration-corrected but still an '
                       'indicative estimate, not a precise valuation (R²=0.22-0.43).')
        c3.metric('Team', row['team_name'])
        c3.metric('Age', f"{row['age']:.1f}")
        if pd.notna(row['log_residual']):
            gap_pct = (np.expm1(row['log_residual'])) * 100
            c4.metric('Raw Predicted vs. Actual', f'{gap_pct:+.0f}%',
                      help='(predicted - actual) / actual -- unadjusted gap, biased toward cheap '
                           'players at the value extremes. Undervalued Score (left) corrects for '
                           'this; prefer it over this raw figure.')

        cluster_label = row['cluster_label']
        abbrev = cluster_abbrev(cluster_label)
        title = cluster_name(cluster_label) if pd.notna(cluster_label) else 'No assignment'
        st.markdown(f"#### Cluster: {title}" + (f" ({abbrev})" if abbrev not in ('—', 'Low signal') else ''))
        st.caption(f'Raw label: `{cluster_label}`')
        st.info(cluster_description(cluster_label))

        st.markdown('#### Key stats vs. cluster average')
        st.caption(
            'Unlike the Undervalued Players tab, this comparison doesn\'t depend on the '
            'regression models\' absolute-value calibration at all -- it\'s just this player\'s '
            'own stats and market value against their cluster peers\' -- so it stays reliable '
            'even for players at the value extremes, where the predicted-value ranking is '
            'weakest. Shows every stat that distinguishes this specific cluster, not a generic '
            'per-position list.'
        )
        cluster_peers = df[(df['cluster_label'] == cluster_label) & (df['position'] == pos)]
        key_stats = CLUSTER_KEY_STATS.get(cluster_label, [])
        if cluster_label == 'G_low_signal' or len(cluster_peers) < 3:
            st.caption(
                'Cluster comparison skipped: this cluster is a documented low-signal artifact '
                'bucket (see the cluster key above), not a meaningful peer group to compare '
                'against.'
                if cluster_label == 'G_low_signal' else
                'Too few peers in this cluster for a meaningful comparison.'
            )
        else:
            comp_rows = []
            for stat in key_stats:
                if stat not in df.columns:
                    continue
                comp_rows.append({
                    'stat': stat_label(stat),
                    'player': row[stat],
                    'cluster_avg': cluster_peers[stat].mean(),
                })
            comp_df = pd.DataFrame(comp_rows).set_index('stat')

            fig, ax = plt.subplots(figsize=(9, 4))
            x = np.arange(len(comp_df))
            width = 0.35
            ax.bar(x - width / 2, comp_df['player'], width, label=selected_name, color='dodgerblue')
            ax.bar(x + width / 2, comp_df['cluster_avg'], width,
                   label=f'{abbrev} average', color='lightgray')
            ax.set_xticks(x)
            ax.set_xticklabels(comp_df.index, rotation=30, ha='right', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(False)
            plt.tight_layout()
            st.pyplot(fig)

            value_percentile = (cluster_peers['market_value_eur'] < row['market_value_eur']).mean() * 100
            st.caption(
                f"Market value percentile within {abbrev} (n={len(cluster_peers)}): "
                f"{value_percentile:.0f}th."
            )

        with st.expander(f'Full stats: every available metric for {selected_name}'):
            hidden_elsewhere = {
                'player_id', 'league', 'season', 'position', 'team_name', 'sofascore_team_id',
                'market_value_eur', 'predicted_market_value_eur', 'undervalued_score',
                'log_residual', 'market_value_data_quality_flag', 'model_used', 'model_test_r2',
                'kmeans_cluster_raw', 'cluster_label', 'age', 'name', 'contract_until',
            }
            all_stat_cols = [c for c in df.columns if c not in hidden_elsewhere]

            def _round(v):
                return round(v, 2) if isinstance(v, float) else v

            full_stats_df = pd.DataFrame({
                'Stat': [stat_label(c) for c in all_stat_cols],
                'Value': [_round(row[c]) for c in all_stat_cols],
            })
            st.caption(f'{len(all_stat_cols)} metrics, including ones not specific to this '
                       f'player\'s cluster.')
            st.dataframe(full_stats_df, use_container_width=True, hide_index=True)

# --- Tab 4: Find Similar Players (replacement-finder) ---------------------------------------
with tab_similar:
    st.subheader('Find Similar Players')
    st.markdown(
        'Pick a reference player to find **statistically similar** players at the same '
        'position -- a starting point for identifying realistic replacements or alternatives, '
        'not a scouting judgement. Similarity is computed purely from standardised (z-scored) '
        'per-90 stats -- the same feature set `clustering.ipynb` uses -- via Euclidean distance; '
        'it knows nothing about a player\'s reputation, injury history, or how they\'d actually '
        'fit a squad. Deliberately **not** PCA space (which is how the clusters themselves were '
        'built): raw standardised stats are directly interpretable to a non-technical user, '
        'which matters more here than exact consistency with the clustering methodology.'
    )

    all_names_sim = sorted(df['name'].unique())
    ref_name = st.selectbox(
        'Reference player', all_names_sim, index=None, placeholder='Type to search...',
        key='similar_ref_player',
    )

    if ref_name is None:
        st.info('Select a player above to find statistically similar alternatives.')
    else:
        ref_candidates = df[df['name'] == ref_name]
        if len(ref_candidates) > 1:
            # Two different real players can share a display name -- disambiguate by club,
            # the same way Player Profile disambiguates by season (a different collision there).
            club_options = ref_candidates['team_name'] + ' -- ' + ref_candidates['league']
            picked_club = st.selectbox('Multiple players share this name -- which one?', club_options)
            ref_row = ref_candidates[club_options == picked_club].iloc[0]
        else:
            ref_row = ref_candidates.iloc[0]

        ref_position = ref_row['position']
        ref_cluster = ref_row['cluster_label']
        ref_player_id = ref_row['player_id']
        ref_abbrev = cluster_abbrev(ref_cluster)
        ref_cluster_title = cluster_name(ref_cluster) if pd.notna(ref_cluster) else 'No assignment'

        st.markdown(
            f"#### Reference: {ref_name} -- {ref_row['team_name']} ({POSITION_NAMES[ref_position]}, "
            f"{ref_cluster_title}{f' / {ref_abbrev}' if ref_abbrev not in ('—', 'Low signal') else ''})"
        )

        # Similarity is computed on the FULL same-position pool, independent of the sidebar's
        # league/age/budget filters -- a player's statistical similarity to the reference
        # shouldn't change just because a scout tightened their budget; only which candidates are
        # *shown* should. Cluster/position sidebar filters are deliberately not applied here at
        # all: this page enforces its own hard same-position restriction and does its own
        # cluster split, which would conflict with (or be made redundant by) those two filters.
        pos_pool = df[df['position'] == ref_position].copy()
        X = pos_pool[SIMILARITY_FEATURE_COLS].fillna(0.0)
        stds = X.std(ddof=0).replace(0, np.nan)
        Z = (X - X.mean()) / stds
        Z = Z.fillna(0.0)

        ref_iloc = pos_pool.index.get_loc(ref_row.name)
        ref_z = Z.iloc[ref_iloc]
        distance = np.sqrt(((Z - ref_z) ** 2).sum(axis=1))
        pos_pool['distance'] = distance.values

        others = pos_pool[pos_pool['player_id'] != ref_player_id].copy()
        max_dist = others['distance'].max()
        others['similarity'] = 100 * (1 - others['distance'] / max_dist) if max_dist > 0 else 100.0

        # Scout filters: same controls as the sidebar (league, age, budget), applied here
        # explicitly rather than reusing `filtered` -- `filtered` also carries the sidebar's
        # Position/Cluster filters, which this page must not inherit (it enforces its own).
        candidates = others
        if leagues:
            candidates = candidates[candidates['league'].isin(leagues)]
        candidates = candidates[candidates['age'].between(*age_range)]
        candidates = candidates[
            candidates['market_value_eur'].between(*budget_range) | candidates['market_value_eur'].isna()
        ]
        n_excluded_by_filters = len(others) - len(candidates)
        st.caption(
            f'{len(others)} other {POSITION_NAMES[ref_position].lower()}s in the 25/26 dataset; '
            f'{len(candidates)} match the sidebar\'s league/age/budget filters'
            + (f' ({n_excluded_by_filters} filtered out)' if n_excluded_by_filters else '') + '.'
        )

        key_stats = CLUSTER_KEY_STATS.get(ref_cluster, [])
        is_low_signal = pd.isna(ref_cluster) or ref_cluster == 'G_low_signal'

        def build_result_table(rows):
            cols = {
                'name': 'Name', 'team_name': 'Team', 'league': 'League', 'age': 'Age',
                'cluster_label': 'Cluster', 'similarity': 'Similarity',
                'market_value_eur': 'Actual (EUR)',
                'predicted_market_value_eur': 'Predicted (indicative)',
                'undervalued_score': 'Undervalued Score',
            }
            extra_cols = [s for s in key_stats if s in rows.columns]
            all_cols = list(cols) + extra_cols
            out = rows.sort_values('similarity', ascending=False)[all_cols].rename(
                columns={**cols, **{s: stat_label(s) for s in extra_cols}}
            ).reset_index(drop=True)
            out['Cluster'] = out['Cluster'].apply(cluster_abbrev)
            cfg = {
                'Age': st.column_config.NumberColumn('Age', format='%.1f'),
                'Similarity': st.column_config.NumberColumn(
                    'Similarity', format='%.1f',
                    help='0-100, rescaled from Euclidean distance in standardised per-90 stat '
                         'space among all same-position players -- 100 would be an identical '
                         'stat profile. Not a percentage of anything, and not affected by the '
                         'sidebar filters below.',
                ),
                'Actual (EUR)': st.column_config.NumberColumn('Actual (EUR)', format='€%,d'),
                'Predicted (indicative)': st.column_config.NumberColumn(
                    'Predicted (indicative)', format='€%,d'
                ),
                'Undervalued Score': st.column_config.NumberColumn('Undervalued Score', format='%.2f'),
            }
            for s in extra_cols:
                cfg[stat_label(s)] = st.column_config.NumberColumn(stat_label(s), format='%.2f')
            return out, cfg

        if key_stats:
            ref_stat_line = ' | '.join(f'{stat_label(s)}: {ref_row[s]:.2f}' for s in key_stats)
            st.markdown(f"**{ref_name}'s own values on {ref_abbrev}'s distinguishing stats** "
                        f"(compare the same columns in the tables below): {ref_stat_line}")

        if is_low_signal:
            st.warning(
                f'`{ref_cluster if pd.notna(ref_cluster) else "no cluster"}` is not a real '
                f'playing style (see the cluster key above), so a "closest within cluster" list '
                f'would be meaningless -- it would just be the artifact bucket\'s other members, '
                f'not players who share a genuine style with {ref_name}. Showing one combined '
                f'list of the closest same-position players instead, with no cluster split and '
                f'no distinguishing-stats comparison (there isn\'t a real archetype to compare '
                f'against).',
                icon='⚠️',
            )
            table, cfg = build_result_table(candidates)
            st.dataframe(table, use_container_width=True, hide_index=True, column_config=cfg)
        else:
            within = candidates[candidates['cluster_label'] == ref_cluster]
            outside = candidates[candidates['cluster_label'] != ref_cluster]

            st.markdown(f'#### Closest within {ref_abbrev} ({len(within)})')
            if within.empty:
                st.caption('No players match both the same cluster and the current sidebar '
                           'filters.')
            else:
                table, cfg = build_result_table(within)
                st.dataframe(table, use_container_width=True, hide_index=True, column_config=cfg)

            st.markdown(f'#### Closest outside {ref_abbrev} ({len(outside)})')
            st.caption(
                'Deliberately shown, not just the within-cluster list: cluster boundaries are '
                'soft (Ward hierarchical clustering agreed with K-Means at only ARI 0.40-0.51 '
                'against these same clusters), so a statistically similar player may sit just '
                'the other side of a cluster boundary rather than genuinely differ in style.'
            )
            if outside.empty:
                st.caption('No players outside this cluster match the current sidebar filters.')
            else:
                table, cfg = build_result_table(outside)
                st.dataframe(table, use_container_width=True, hide_index=True, column_config=cfg)
