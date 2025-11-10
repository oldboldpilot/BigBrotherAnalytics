-- BigBrotherAnalytics: Sector Correlations Table
-- Stores time-lagged correlation analysis between GICS sectors
-- Author: Agent 6 - Correlation Discovery Agent
-- Date: 2025-11-10

CREATE TABLE IF NOT EXISTS sector_correlations (
    id INTEGER PRIMARY KEY,
    sector_code_1 INTEGER NOT NULL,
    sector_code_2 INTEGER NOT NULL,
    lag_days INTEGER NOT NULL,
    correlation_coefficient DOUBLE NOT NULL,
    correlation_type VARCHAR NOT NULL, -- 'pearson', 'spearman'
    p_value DOUBLE,
    sample_size INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sector_code_1) REFERENCES sectors(sector_code),
    FOREIGN KEY (sector_code_2) REFERENCES sectors(sector_code),
    -- Ensure unique combinations
    UNIQUE(sector_code_1, sector_code_2, lag_days, correlation_type)
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_sector_corr_codes ON sector_correlations(sector_code_1, sector_code_2);
CREATE INDEX IF NOT EXISTS idx_sector_corr_strength ON sector_correlations(correlation_coefficient);
CREATE INDEX IF NOT EXISTS idx_sector_corr_lag ON sector_correlations(lag_days);

-- View for strong correlations
CREATE OR REPLACE VIEW strong_sector_correlations AS
SELECT
    sc.*,
    s1.sector_name as sector_1_name,
    s1.sector_etf as sector_1_etf,
    s2.sector_name as sector_2_name,
    s2.sector_etf as sector_2_etf
FROM sector_correlations sc
JOIN sectors s1 ON sc.sector_code_1 = s1.sector_code
JOIN sectors s2 ON sc.sector_code_2 = s2.sector_code
WHERE ABS(sc.correlation_coefficient) > 0.5
  AND sc.p_value < 0.05
ORDER BY ABS(sc.correlation_coefficient) DESC;
