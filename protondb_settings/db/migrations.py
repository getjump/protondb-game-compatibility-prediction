"""Database schema creation — single source of truth from PLAN.md."""

from __future__ import annotations

import sqlite3

_SCHEMA_SQL = """
-- === Core (filled by worker) ===

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- === Pipeline durability ===

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step        TEXT NOT NULL,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    total_items INTEGER,
    processed   INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'running',
    error       TEXT,
    dump_tag    TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipeline_step_status ON pipeline_runs(step, status);

CREATE TABLE IF NOT EXISTS games (
    app_id     INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reports (
    id               TEXT PRIMARY KEY,
    app_id           INTEGER NOT NULL REFERENCES games(app_id),
    timestamp        TEXT NOT NULL,

    -- systemInfo
    gpu              TEXT,
    gpu_driver       TEXT,
    cpu              TEXT,
    ram              TEXT,
    os               TEXT,
    ram_mb           INTEGER,
    kernel           TEXT,
    window_manager   TEXT,

    -- responses: main
    type             TEXT,
    variant          TEXT,
    proton_version   TEXT,
    custom_proton_version TEXT,
    verdict          TEXT,
    tried_oob        TEXT,
    verdict_oob      TEXT,
    tinker_override  TEXT,

    -- responses: funnel
    installs         TEXT,
    opens            TEXT,
    starts_play      TEXT,
    duration         TEXT,
    extra            TEXT,

    -- fault fields
    audio_faults     TEXT,
    graphical_faults TEXT,
    input_faults     TEXT,
    performance_faults TEXT,
    stability_faults TEXT,
    windowing_faults TEXT,
    save_game_faults TEXT,
    significant_bugs TEXT,

    -- followUp JSON
    followup_audio_faults_json      TEXT,
    followup_graphical_faults_json  TEXT,
    followup_input_faults_json      TEXT,
    followup_performance_faults_json TEXT,
    followup_stability_faults_json  TEXT,
    followup_windowing_faults_json  TEXT,
    followup_save_game_faults_json  TEXT,
    followup_anticheat_json         TEXT,
    followup_control_cust_json      TEXT,

    -- customizationsUsed
    cust_winetricks       INTEGER,
    cust_protontricks     INTEGER,
    cust_config_change    INTEGER,
    cust_custom_prefix    INTEGER,
    cust_custom_proton    INTEGER,
    cust_lutris           INTEGER,
    cust_media_foundation INTEGER,
    cust_protonfixes      INTEGER,
    cust_native2proton    INTEGER,
    cust_not_listed       INTEGER,

    -- launchFlagsUsed
    flag_use_wine_d3d11      INTEGER,
    flag_disable_esync       INTEGER,
    flag_enable_nvapi        INTEGER,
    flag_disable_fsync       INTEGER,
    flag_use_wine_d9vk       INTEGER,
    flag_large_address_aware INTEGER,
    flag_disable_d3d11       INTEGER,
    flag_hide_nvidia         INTEGER,
    flag_game_drive          INTEGER,
    flag_no_write_watch      INTEGER,
    flag_no_xim              INTEGER,
    flag_old_gl_string       INTEGER,
    flag_use_seccomp         INTEGER,
    flag_fullscreen_integer_scaling INTEGER,

    -- text fields
    launch_options   TEXT,
    concluding_notes TEXT,

    -- notes
    notes_verdict          TEXT,
    notes_audio_faults     TEXT,
    notes_graphical_faults TEXT,
    notes_performance_faults TEXT,
    notes_stability_faults TEXT,
    notes_windowing_faults TEXT,
    notes_input_faults     TEXT,
    notes_significant_bugs TEXT,
    notes_save_game_faults TEXT,
    notes_extra            TEXT,
    notes_launch_flags     TEXT,
    notes_customizations   TEXT,
    notes_variant          TEXT,
    notes_proton_version   TEXT,
    notes_concluding_notes TEXT,

    -- Steam Deck
    battery_performance  TEXT,
    readability          TEXT,
    control_layout       TEXT,
    did_change_control_layout TEXT,
    control_layout_customization TEXT,
    frame_rate           TEXT,
    notes_control_layout TEXT,
    notes_control_layout_customization TEXT,
    notes_battery_performance TEXT,
    notes_readability    TEXT,

    -- Launcher
    launcher             TEXT,
    secondary_launcher   TEXT,
    notes_launcher       TEXT,
    notes_secondary_launcher TEXT,

    -- Multiplayer
    is_multiplayer_important TEXT,
    local_mp_attempted   TEXT,
    local_mp_played      TEXT,
    local_mp_appraisal   TEXT,
    notes_local_mp       TEXT,
    online_mp_attempted  TEXT,
    online_mp_played     TEXT,
    online_mp_appraisal  TEXT,
    notes_online_mp      TEXT,

    -- Anti-cheat
    is_impacted_by_anticheat TEXT,
    notes_anticheat      TEXT,

    -- Tinker override
    notes_tinker_override TEXT
);

CREATE INDEX IF NOT EXISTS idx_reports_app_id ON reports(app_id);
CREATE INDEX IF NOT EXISTS idx_reports_gpu ON reports(gpu);
CREATE INDEX IF NOT EXISTS idx_reports_timestamp ON reports(timestamp);
CREATE INDEX IF NOT EXISTS idx_reports_verdict ON reports(verdict);
CREATE INDEX IF NOT EXISTS idx_reports_launch_options ON reports(launch_options);

-- === Preprocessing: enrichment ===

CREATE TABLE IF NOT EXISTS game_metadata (
    app_id              INTEGER PRIMARY KEY REFERENCES games(app_id),
    developer           TEXT,
    publisher           TEXT,
    genres              TEXT,
    categories          TEXT,
    release_date        TEXT,
    has_linux_native    INTEGER,
    engine              TEXT,
    graphics_apis       TEXT,
    drm                 TEXT,
    anticheat           TEXT,
    anticheat_status    TEXT,
    deck_status         INTEGER,
    deck_tests_json     TEXT,
    protondb_tier       TEXT,
    protondb_score      REAL,
    protondb_confidence TEXT,
    protondb_trending   TEXT,
    github_issue_count       INTEGER,
    github_open_count        INTEGER,
    github_closed_completed  INTEGER,
    github_closed_not_planned INTEGER,
    github_has_regression    INTEGER,
    github_latest_issue_date TEXT,
    enriched_at         TEXT DEFAULT (datetime('now'))
);

-- === Preprocessing: GPU normalization ===

CREATE TABLE IF NOT EXISTS gpu_normalization (
    raw_string      TEXT PRIMARY KEY,
    vendor          TEXT NOT NULL,
    family          TEXT NOT NULL,
    model           TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    is_apu          INTEGER DEFAULT 0,
    is_igpu         INTEGER DEFAULT 0,
    is_virtual      INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_gpu_norm_family ON gpu_normalization(family);
CREATE INDEX IF NOT EXISTS idx_gpu_norm_vendor ON gpu_normalization(vendor);

CREATE TABLE IF NOT EXISTS cpu_normalization (
    raw_string      TEXT PRIMARY KEY,
    vendor          TEXT NOT NULL,
    family          TEXT NOT NULL,
    model           TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    generation      INTEGER,
    is_apu          INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cpu_norm_family ON cpu_normalization(family);
CREATE INDEX IF NOT EXISTS idx_cpu_norm_vendor ON cpu_normalization(vendor);

-- === Preprocessing: GPU/CPU heuristic normalization ===

CREATE TABLE IF NOT EXISTS gpu_normalization_heuristic (
    raw_string      TEXT PRIMARY KEY,
    vendor          TEXT NOT NULL,
    family          TEXT NOT NULL,
    model           TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    is_apu          INTEGER DEFAULT 0,
    is_igpu         INTEGER DEFAULT 0,
    is_mobile       INTEGER DEFAULT 0,
    is_virtual      INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_gpu_norm_h_family ON gpu_normalization_heuristic(family);
CREATE INDEX IF NOT EXISTS idx_gpu_norm_h_vendor ON gpu_normalization_heuristic(vendor);

CREATE TABLE IF NOT EXISTS cpu_normalization_heuristic (
    raw_string      TEXT PRIMARY KEY,
    vendor          TEXT NOT NULL,
    family          TEXT NOT NULL,
    model           TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    generation      INTEGER,
    is_apu          INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cpu_norm_h_family ON cpu_normalization_heuristic(family);
CREATE INDEX IF NOT EXISTS idx_cpu_norm_h_vendor ON cpu_normalization_heuristic(vendor);

CREATE TABLE IF NOT EXISTS gpu_driver_normalization (
    raw_string      TEXT PRIMARY KEY,
    driver_vendor   TEXT NOT NULL,
    driver_version  TEXT NOT NULL,
    driver_major    INTEGER,
    driver_minor    INTEGER,
    driver_patch    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_gpu_drv_vendor ON gpu_driver_normalization(driver_vendor);

-- === Preprocessing: launch options parsing ===

CREATE TABLE IF NOT EXISTS launch_options_parsed (
    raw_string       TEXT PRIMARY KEY,
    env_vars_json    TEXT,
    wrappers_json    TEXT,
    game_args_json   TEXT,
    unparsed         TEXT
);

-- === Preprocessing: text extraction ===

CREATE TABLE IF NOT EXISTS extracted_data (
    report_id         TEXT PRIMARY KEY REFERENCES reports(id),
    app_id            INTEGER NOT NULL REFERENCES games(app_id),
    actions_json      TEXT,
    observations_json TEXT,
    useful            INTEGER,
    processed_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_extracted_app_id ON extracted_data(app_id);
CREATE INDEX IF NOT EXISTS idx_extracted_useful ON extracted_data(useful);

-- === Preprocessing: report contributor data (from ProtonDB Reports API) ===

CREATE TABLE IF NOT EXISTS report_contributors (
    report_id       TEXT PRIMARY KEY REFERENCES reports(id),
    contributor_id  TEXT,
    report_tally    INTEGER,
    playtime        INTEGER,
    playtime_linux  INTEGER,
    fetched_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_contrib_contributor_id ON report_contributors(contributor_id);

-- === Enrichment source cache ===
-- Persists raw API responses so re-runs don't re-fetch

CREATE TABLE IF NOT EXISTS enrichment_cache (
    app_id     INTEGER NOT NULL,
    source     TEXT NOT NULL,
    data_json  TEXT,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (app_id, source)
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indices if they don't exist."""
    conn.executescript(_SCHEMA_SQL)

    # Incremental migrations for existing databases
    _add_column_if_missing(
        conn, "gpu_normalization_heuristic", "is_mobile", "INTEGER DEFAULT 0",
    )
    for col in (
        "github_issue_count", "github_open_count",
        "github_closed_completed", "github_closed_not_planned",
        "github_has_regression",
    ):
        _add_column_if_missing(conn, "game_metadata", col, "INTEGER")
    _add_column_if_missing(conn, "game_metadata", "github_latest_issue_date", "TEXT")


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, col_type: str,
) -> None:
    """Add a column to *table* if it doesn't already exist."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()
