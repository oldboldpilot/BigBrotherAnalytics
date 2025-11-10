-- Database Performance Optimization Indexes
-- Run this script to add indexes for frequently queried columns
-- Usage: duckdb data/bigbrother.duckdb < scripts/setup_database_indexes.sql

-- Positions table indexes
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_is_bot ON positions(is_bot_managed);
CREATE INDEX IF NOT EXISTS idx_positions_pnl ON positions(unrealized_pnl DESC);

-- Positions history indexes
CREATE INDEX IF NOT EXISTS idx_positions_history_timestamp ON positions_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_positions_history_symbol ON positions_history(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_history_is_bot ON positions_history(is_bot_managed);

-- Sector employment indexes
CREATE INDEX IF NOT EXISTS idx_sector_employment_code_date ON sector_employment(sector_code, report_date DESC);
CREATE INDEX IF NOT EXISTS idx_sector_employment_date ON sector_employment(report_date DESC);

-- Sector employment raw indexes
CREATE INDEX IF NOT EXISTS idx_sector_employment_raw_series_date ON sector_employment_raw(series_id, report_date DESC);

-- Jobless claims indexes
CREATE INDEX IF NOT EXISTS idx_jobless_claims_date ON jobless_claims(report_date DESC);
CREATE INDEX IF NOT EXISTS idx_jobless_claims_spike ON jobless_claims(spike_detected, report_date DESC) WHERE spike_detected = TRUE;

-- Sectors index
CREATE INDEX IF NOT EXISTS idx_sectors_code ON sectors(sector_code);
