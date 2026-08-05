-- schema.sql
-- ----------
-- Full database schema for ShuttleSmash AI.
-- All tables are created now (Phase 1) even though only `users` is used
-- until later phases, so the database never needs to be migrated as new
-- features are added -- new phases simply start writing to tables that
-- already exist.
--
-- This file is executed once via the `flask init-db` command
-- (see database/db_manager.py).

PRAGMA foreign_keys = ON;

-- Remove existing tables first so `flask init-db` can be re-run safely
-- during development without leaving stale tables behind.
DROP TABLE IF EXISTS reports;
DROP TABLE IF EXISTS recommendations;
DROP TABLE IF EXISTS injury_risks;
DROP TABLE IF EXISTS analyses;
DROP TABLE IF EXISTS videos;
DROP TABLE IF EXISTS users;

-- 1. Users -------------------------------------------------------------
CREATE TABLE users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name       TEXT NOT NULL,
    username        TEXT UNIQUE NOT NULL,
    email           TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Uploaded videos (Phase 2) ------------------------------------------
CREATE TABLE videos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    original_name   TEXT NOT NULL,
    stored_path     TEXT NOT NULL,
    video_type      TEXT CHECK(video_type IN
                      ('smash','footwork','jump','lunge','general')) NOT NULL,
    uploaded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status          TEXT CHECK(status IN
                      ('uploaded','processing','completed','failed')) DEFAULT 'uploaded',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 3. Analysis results (Phase 3-4) ---------------------------------------
CREATE TABLE analyses (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id            INTEGER NOT NULL,
    user_id             INTEGER NOT NULL,
    analysis_date       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    avg_joint_angles    TEXT,      -- JSON: {"left_elbow_angle": {"avg":..,"min":..,"max":..}, ...}
    movement_metrics_json TEXT,    -- JSON: flat metrics from the video-type-specific module
                                    -- (footwork_analysis.py / smash_analysis.py / etc.), ready
                                    -- to display directly on the results page.
    features_json       TEXT,      -- JSON: full flat ML-ready feature row (see
                                    -- modules/feature_builder.py) -- what a future CSV
                                    -- export for Random Forest training will read from.
    footwork_score      REAL,      -- reserved for a later scoring/injury-risk phase
    smash_score         REAL,      -- reserved for a later scoring/injury-risk phase
    jump_height_cm      REAL,      -- reserved: real-world cm needs camera calibration
    lunge_depth_score   REAL,      -- reserved for a later scoring/injury-risk phase
    overall_score        REAL,     -- reserved for a later scoring/injury-risk phase
    processed_preview   TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 4. Injury risk findings (Phase 4) --------------------------------------
CREATE TABLE injury_risks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     INTEGER NOT NULL,
    risk_type       TEXT NOT NULL,
    affected_joint  TEXT NOT NULL,
    risk_level      TEXT CHECK(risk_level IN ('Low','Moderate','High')) NOT NULL,
    confidence      REAL,
    description     TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);

-- 5. AI recommendations (Phase 4) ----------------------------------------
CREATE TABLE recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     INTEGER NOT NULL,
    category        TEXT NOT NULL,
    recommendation  TEXT NOT NULL,
    priority        TEXT CHECK(priority IN ('Low','Medium','High')) DEFAULT 'Medium',
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);

-- 6. Generated PDF reports (Phase 5) --------------------------------------
CREATE TABLE reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     INTEGER NOT NULL,
    user_id         INTEGER NOT NULL,
    file_path       TEXT NOT NULL,
    generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
