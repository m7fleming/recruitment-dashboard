-- Schema for the dissertation player dataset (data_processed/all_players_model_ready.csv).
--
-- Two tables, one row-grain each:
--   players             one row per real-world player (bio attributes, stable across seasons)
--   player_season_stats one row per (player, league, season) performance record
--
-- KNOWN DATA QUALITY ISSUES (as of the 2026-08-30 all_players_model_ready.csv) -- read before
-- trusting these columns in a dashboard or model:
--
-- 1. RESOLVED 2026-08-30: team_name/sofascore_team_id used to be sourced from Transfermarkt's
--    'Team' column, which reflects each player's CURRENT club as of scrape time, not their club
--    during the labeled historical season (e.g. Julian Alvarez showed his 2024 Atletico Madrid
--    move under a 2023/24 Premier League/Man City row). Fixed in full_pipeline_scrape.ipynb
--    Part 9 by sourcing club identity from Sofascore's own 'team'/'team id' fields instead, which
--    are correctly season-scoped (verified against the raw Sofascore scrape). Still no `teams`
--    table/foreign key here -- not needed for a single denormalized text column, and keeps this
--    schema a direct mirror of the corrected CSV.
--
-- 2. market_value_eur and contract_until are still sourced from Transfermarkt and likely have the
--    same "current, not historical" problem as issue #1 did -- Transfermarkt doesn't expose a
--    per-season market value in this scrape (the 'Market value history' column came back empty).
--    Left as-is for now; treat market_value_eur as a current-value proxy applied uniformly across
--    a player's season rows, not a true season-specific value, until this is revisited.
--
-- 3. Most numeric columns in player_season_stats (everything except appearances_total,
--    minutes_played_total, and the *_per90/*_overperformance_per90 columns) are per-90-minute
--    rates from Sofascore, not season totals, despite names like "tackles" or "yellowCards"
--    suggesting a count. Confirmed e.g. "appearances" = 1.04 for a player with 34 real
--    appearances (appearances_total). data_prep.py only renamed goals/assists/xG/xA to make
--    this explicit; the other ~100 columns keep Sofascore's original names as-is here to stay a
--    faithful mirror of the CSV.

CREATE TABLE players (
    player_id       INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    name_key        TEXT NOT NULL,
    dob             DATE,
    country         TEXT,
    preferred_foot  TEXT,
    height_cm       SMALLINT,
    weight_kg       SMALLINT
);

CREATE TABLE player_season_stats (
    player_id       INTEGER NOT NULL REFERENCES players (player_id),
    league          TEXT NOT NULL,
    season          TEXT NOT NULL,

    team_name        TEXT,
    sofascore_team_id INTEGER,

    position            TEXT,
    market_value_eur     BIGINT,
    contract_until       DATE,
    appearances_total    INTEGER,
    minutes_played_total INTEGER,

    -- see "KNOWN DATA QUALITY ISSUES" #3 above -- per-90 rates unless noted otherwise
    "accurateChippedPasses" DOUBLE PRECISION,
    "accurateCrosses" DOUBLE PRECISION,
    "accurateCrossesPercentage" DOUBLE PRECISION,
    "accurateFinalThirdPasses" DOUBLE PRECISION,
    "accurateLongBalls" DOUBLE PRECISION,
    "accurateLongBallsPercentage" DOUBLE PRECISION,
    "accurateOppositionHalfPasses" DOUBLE PRECISION,
    "accurateOwnHalfPasses" DOUBLE PRECISION,
    "accuratePasses" DOUBLE PRECISION,
    "accuratePassesPercentage" DOUBLE PRECISION,
    "aerialDuelsWon" DOUBLE PRECISION,
    "aerialDuelsWonPercentage" DOUBLE PRECISION,
    "aerialLost" DOUBLE PRECISION,
    "appearances" DOUBLE PRECISION,
    "assists_per90" DOUBLE PRECISION,
    "attemptPenaltyMiss" DOUBLE PRECISION,
    "attemptPenaltyPost" DOUBLE PRECISION,
    "attemptPenaltyTarget" DOUBLE PRECISION,
    "ballRecovery" DOUBLE PRECISION,
    "bigChancesCreated" DOUBLE PRECISION,
    "bigChancesMissed" DOUBLE PRECISION,
    "blockedShots" DOUBLE PRECISION,
    "cleanSheet" DOUBLE PRECISION,
    "clearances" DOUBLE PRECISION,
    "countRating" DOUBLE PRECISION,
    "crossesNotClaimed" DOUBLE PRECISION,
    "directRedCards" DOUBLE PRECISION,
    "dispossessed" DOUBLE PRECISION,
    "dribbledPast" DOUBLE PRECISION,
    "duelLost" DOUBLE PRECISION,
    "errorLeadToGoal" DOUBLE PRECISION,
    "errorLeadToShot" DOUBLE PRECISION,
    "expectedAssists_per90" DOUBLE PRECISION,
    "expectedGoals_per90" DOUBLE PRECISION,
    "fouls" DOUBLE PRECISION,
    "freeKickGoal" DOUBLE PRECISION,
    "goalConversionPercentage" DOUBLE PRECISION,
    "goalKicks" DOUBLE PRECISION,
    "goals_per90" DOUBLE PRECISION,
    "goalsAssistsSum" DOUBLE PRECISION,
    "goalsConceded" DOUBLE PRECISION,
    "goalsConcededInsideTheBox" DOUBLE PRECISION,
    "goalsConcededOutsideTheBox" DOUBLE PRECISION,
    "goalsFromInsideTheBox" DOUBLE PRECISION,
    "goalsFromOutsideTheBox" DOUBLE PRECISION,
    "goalsPrevented" DOUBLE PRECISION,
    "groundDuelsWon" DOUBLE PRECISION,
    "groundDuelsWonPercentage" DOUBLE PRECISION,
    "headedGoals" DOUBLE PRECISION,
    "highClaims" DOUBLE PRECISION,
    "hitWoodwork" DOUBLE PRECISION,
    "inaccuratePasses" DOUBLE PRECISION,
    "interceptions" DOUBLE PRECISION,
    "keyPasses" DOUBLE PRECISION,
    "leftFootGoals" DOUBLE PRECISION,
    "matchesStarted" DOUBLE PRECISION,
    "offsides" DOUBLE PRECISION,
    "outfielderBlocks" DOUBLE PRECISION,
    "ownGoals" DOUBLE PRECISION,
    "passToAssist" DOUBLE PRECISION,
    "penaltiesTaken" DOUBLE PRECISION,
    "penaltyConceded" DOUBLE PRECISION,
    "penaltyConversion" DOUBLE PRECISION,
    "penaltyFaced" DOUBLE PRECISION,
    "penaltyGoals" DOUBLE PRECISION,
    "penaltySave" DOUBLE PRECISION,
    "penaltyWon" DOUBLE PRECISION,
    "possessionLost" DOUBLE PRECISION,
    "possessionWonAttThird" DOUBLE PRECISION,
    "punches" DOUBLE PRECISION,
    "rating" DOUBLE PRECISION,
    "redCards" DOUBLE PRECISION,
    "rightFootGoals" DOUBLE PRECISION,
    "runsOut" DOUBLE PRECISION,
    "savedShotsFromInsideTheBox" DOUBLE PRECISION,
    "savedShotsFromOutsideTheBox" DOUBLE PRECISION,
    "saves" DOUBLE PRECISION,
    "savesCaught" DOUBLE PRECISION,
    "savesParried" DOUBLE PRECISION,
    "scoringFrequency" DOUBLE PRECISION,
    "setPieceConversion" DOUBLE PRECISION,
    "shotFromSetPiece" DOUBLE PRECISION,
    "shotsFromInsideTheBox" DOUBLE PRECISION,
    "shotsFromOutsideTheBox" DOUBLE PRECISION,
    "shotsOffTarget" DOUBLE PRECISION,
    "shotsOnTarget" DOUBLE PRECISION,
    "successfulDribbles" DOUBLE PRECISION,
    "successfulDribblesPercentage" DOUBLE PRECISION,
    "successfulRunsOut" DOUBLE PRECISION,
    "tackles" DOUBLE PRECISION,
    "tacklesWon" DOUBLE PRECISION,
    "tacklesWonPercentage" DOUBLE PRECISION,
    "totalAttemptAssist" DOUBLE PRECISION,
    "totalChippedPasses" DOUBLE PRECISION,
    "totalContest" DOUBLE PRECISION,
    "totalCross" DOUBLE PRECISION,
    "totalDuelsWon" DOUBLE PRECISION,
    "totalDuelsWonPercentage" DOUBLE PRECISION,
    "totalLongBalls" DOUBLE PRECISION,
    "totalOppositionHalfPasses" DOUBLE PRECISION,
    "totalOwnHalfPasses" DOUBLE PRECISION,
    "totalPasses" DOUBLE PRECISION,
    "totalRating" DOUBLE PRECISION,
    "totalShots" DOUBLE PRECISION,
    "totwAppearances" DOUBLE PRECISION,
    "touches" DOUBLE PRECISION,
    "wasFouled" DOUBLE PRECISION,
    "yellowCards" DOUBLE PRECISION,
    "yellowRedCards" DOUBLE PRECISION,

    -- derived in data_prep.py: goals_per90 - expectedGoals_per90 (and assists equivalent)
    "goals_overperformance_per90" DOUBLE PRECISION,
    "assists_overperformance_per90" DOUBLE PRECISION,

    PRIMARY KEY (player_id, league, season)
);

CREATE INDEX idx_player_season_stats_league_season ON player_season_stats (league, season);
CREATE INDEX idx_player_season_stats_position ON player_season_stats (position);

COMMENT ON TABLE player_season_stats IS
    'Grain: one row per player per league per season. See schema.sql header comment for known '
    'data quality caveats on market_value_eur/contract_until and the per-90 metric columns.';

-- Dashboard tables (app.py). Both are pure exports from a notebook -- never written to directly,
-- always safe to drop/reload alongside the two tables above.

CREATE TABLE player_cluster_labels (
    player_id            INTEGER NOT NULL REFERENCES players (player_id),
    league                TEXT NOT NULL,
    season                TEXT NOT NULL,
    position              TEXT NOT NULL,
    kmeans_cluster_raw    INTEGER NOT NULL,
    cluster_label         TEXT NOT NULL,

    PRIMARY KEY (player_id, league, season)
);

CREATE INDEX idx_player_cluster_labels_cluster_label ON player_cluster_labels (cluster_label);

COMMENT ON TABLE player_cluster_labels IS
    'From clustering.ipynb''s data_processed/player_cluster_labels.csv. cluster_label is the '
    'regression-ready feature -- the two unreliable goalkeeper clusters (a near-zero-variance '
    'artifact and a singleton outlier) are already collapsed into G_other_low_signal, which is '
    'not a real playing-style archetype and should be displayed/handled as such. See CLAUDE.md '
    'data flow step 8.';

CREATE TABLE player_predictions (
    player_id                       INTEGER NOT NULL REFERENCES players (player_id),
    league                           TEXT NOT NULL,
    season                           TEXT NOT NULL,
    position                         TEXT NOT NULL,
    market_value_eur                 BIGINT,
    predicted_market_value_eur       DOUBLE PRECISION,
    log_residual                     DOUBLE PRECISION,
    undervalued_score                DOUBLE PRECISION,
    market_value_data_quality_flag   BOOLEAN NOT NULL DEFAULT FALSE,
    model_used                       TEXT NOT NULL,
    model_test_r2                    DOUBLE PRECISION,

    PRIMARY KEY (player_id, league, season)
);

CREATE INDEX idx_player_predictions_undervalued_score ON player_predictions (undervalued_score);

COMMENT ON TABLE player_predictions IS
    'From regression.ipynb''s data_processed/player_predictions.csv. predicted_market_value_eur '
    'is an out-of-fold cross-validated prediction (never fit on the row it predicts) with two '
    'calibration corrections applied (see regression.ipynb''s "Calibration diagnostics" section): '
    'Duan''s smearing estimator (naive expm1 of a log-scale prediction systematically '
    'underestimates E[value] by 60-290% depending on position) and a per-position linear '
    'value-detrend, since raw residuals correlate ~-0.8 with actual value at every position '
    '(regression to the mean from R^2=0.15-0.4: cheap players get over-predicted, expensive '
    'players under-predicted). '
    'log_residual = predicted_log_market_value - actual_log_market_value (calibrated but NOT '
    'value-detrended -- do not rank by this column, it reproduces the value-dependent bias). '
    'undervalued_score is log_residual with the position''s value-trend subtracted out --'
    ' uncorrelated with value by construction, and the column app.py actually ranks the '
    'Undervalued Players tab by: positive means more undervalued than other players at a '
    'similar price, not just "predicted higher than actual" (which the raw log_residual '
    'conflates with simply being cheap). market_value_data_quality_flag marks rows (currently '
    'Jude Bellingham 23/24, Vinicius Junior 24/25) with a confirmed-wrong market_value_eur, '
    'excluded from every calibration fit above and from the dashboard''s ranking -- kept in the '
    'table rather than dropped, for transparency.';
