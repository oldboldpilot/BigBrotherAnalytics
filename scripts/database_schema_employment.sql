-- BigBrotherAnalytics: Employment & Sector Database Schema
-- Tier 1 Extension: Department of Labor API Integration
--
-- Author: Olumuyiwa Oluwasanmi
-- Created: 2025-11-08
-- Database: DuckDB

-- =============================================================================
-- SECTOR MASTER TABLES
-- =============================================================================

-- GICS Sector Classification (11 sectors)
CREATE TABLE IF NOT EXISTS sectors (
    sector_id INTEGER PRIMARY KEY,
    sector_code INTEGER NOT NULL,  -- GICS sector code (10, 15, 20, ...)
    sector_name VARCHAR NOT NULL,
    sector_category VARCHAR NOT NULL,  -- Defensive, Cyclical, Sensitive
    etf_ticker VARCHAR,  -- Primary sector ETF (XLE, XLF, XLK, etc.)
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Company to Sector Mapping
CREATE TABLE IF NOT EXISTS company_sectors (
    ticker VARCHAR PRIMARY KEY,
    company_name VARCHAR,
    sector_id INTEGER REFERENCES sectors(sector_id),
    industry VARCHAR,
    sub_industry VARCHAR,
    market_cap_category VARCHAR,  -- Large, Mid, Small cap
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- EMPLOYMENT DATA TABLES
-- =============================================================================

-- Bureau of Labor Statistics (BLS) Employment Data by Sector
CREATE TABLE IF NOT EXISTS sector_employment (
    id INTEGER PRIMARY KEY,
    sector_id INTEGER REFERENCES sectors(sector_id),
    bls_series_id VARCHAR NOT NULL,  -- BLS series identifier (e.g., CES1000000001)
    report_date DATE NOT NULL,
    employment_count INTEGER,  -- Total employees in thousands
    unemployment_rate DOUBLE,
    job_openings INTEGER,  -- JOLTS data
    hires INTEGER,  -- JOLTS hires
    separations INTEGER,  -- JOLTS separations
    quits INTEGER,  -- JOLTS voluntary quits
    layoffs_discharges INTEGER,  -- JOLTS layoffs
    data_source VARCHAR DEFAULT 'BLS',  -- BLS, JOLTS, WARN, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sector_id, bls_series_id, report_date)
);

-- Employment Events (Layoffs, Hiring, Freezes)
CREATE TABLE IF NOT EXISTS employment_events (
    id INTEGER PRIMARY KEY,
    event_date DATE NOT NULL,
    company_ticker VARCHAR,
    company_name VARCHAR,
    sector_id INTEGER REFERENCES sectors(sector_id),
    event_type VARCHAR NOT NULL,  -- layoff, hiring, freeze, expansion
    employee_count INTEGER,  -- Number of employees affected
    event_source VARCHAR,  -- Layoffs.fyi, WARN, Press Release, LinkedIn
    event_location VARCHAR,  -- City, State, Country
    impact_magnitude VARCHAR,  -- High, Medium, Low
    news_url VARCHAR,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Weekly Jobless Claims (Leading Economic Indicator)
CREATE TABLE IF NOT EXISTS jobless_claims (
    id INTEGER PRIMARY KEY,
    week_ending DATE NOT NULL UNIQUE,
    initial_claims INTEGER NOT NULL,  -- Weekly initial claims
    continued_claims INTEGER,  -- Continued/insured unemployment
    four_week_avg INTEGER,  -- 4-week moving average
    year_over_year_change DOUBLE,  -- % change YoY
    data_source VARCHAR DEFAULT 'BLS',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Monthly Jobs Report Summary (NFP - Nonfarm Payrolls)
CREATE TABLE IF NOT EXISTS monthly_jobs_report (
    id INTEGER PRIMARY KEY,
    report_date DATE NOT NULL UNIQUE,
    nonfarm_payrolls_change INTEGER NOT NULL,  -- Monthly change in thousands
    unemployment_rate DOUBLE NOT NULL,
    labor_force_participation_rate DOUBLE,
    average_hourly_earnings DOUBLE,  -- $ per hour
    avg_weekly_hours DOUBLE,  -- Hours worked
    private_sector_jobs INTEGER,  -- Private employment change
    government_jobs INTEGER,  -- Government employment change
    consensus_estimate INTEGER,  -- Market expectation
    surprise INTEGER,  -- Actual - Estimate
    market_impact VARCHAR,  -- Positive, Negative, Neutral
    data_source VARCHAR DEFAULT 'BLS',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- SECTOR ANALYSIS TABLES
-- =============================================================================

-- Sector News Sentiment
CREATE TABLE IF NOT EXISTS sector_news_sentiment (
    id INTEGER PRIMARY KEY,
    sector_id INTEGER REFERENCES sectors(sector_id),
    news_date TIMESTAMP NOT NULL,
    sentiment_score DOUBLE,  -- -1.0 (bearish) to 1.0 (bullish)
    news_count INTEGER,  -- Number of news articles analyzed
    major_events TEXT[],  -- Array of major event descriptions
    impact_magnitude VARCHAR,  -- High, Medium, Low
    bullish_mentions INTEGER,
    bearish_mentions INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sector Performance Metrics
CREATE TABLE IF NOT EXISTS sector_performance (
    id INTEGER PRIMARY KEY,
    sector_id INTEGER REFERENCES sectors(sector_id),
    date DATE NOT NULL,
    etf_close_price DOUBLE,
    daily_return DOUBLE,
    relative_strength DOUBLE,  -- vs S&P 500
    volume BIGINT,
    employment_trend VARCHAR,  -- Improving, Declining, Stable
    rotation_signal VARCHAR,  -- Rotate_In, Rotate_Out, Hold
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sector_id, date)
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_sector_employment_date ON sector_employment(sector_id, report_date);
CREATE INDEX IF NOT EXISTS idx_sector_employment_series ON sector_employment(bls_series_id, report_date);

CREATE INDEX IF NOT EXISTS idx_employment_events_date ON employment_events(event_date);
CREATE INDEX IF NOT EXISTS idx_employment_events_ticker ON employment_events(company_ticker);
CREATE INDEX IF NOT EXISTS idx_employment_events_sector ON employment_events(sector_id, event_date);
CREATE INDEX IF NOT EXISTS idx_employment_events_type ON employment_events(event_type);

CREATE INDEX IF NOT EXISTS idx_jobless_claims_date ON jobless_claims(week_ending);
CREATE INDEX IF NOT EXISTS idx_monthly_jobs_date ON monthly_jobs_report(report_date);

CREATE INDEX IF NOT EXISTS idx_sector_sentiment_date ON sector_news_sentiment(sector_id, news_date);
CREATE INDEX IF NOT EXISTS idx_sector_performance_date ON sector_performance(sector_id, date);

-- =============================================================================
-- SEED DATA: 11 GICS SECTORS
-- =============================================================================

INSERT INTO sectors (sector_id, sector_code, sector_name, sector_category, etf_ticker, description) VALUES
(1, 10, 'Energy', 'Cyclical', 'XLE', 'Oil, gas & consumable fuels, energy equipment & services'),
(2, 15, 'Materials', 'Cyclical', 'XLB', 'Chemicals, metals & mining, construction materials'),
(3, 20, 'Industrials', 'Sensitive', 'XLI', 'Aerospace & defense, machinery, transportation, construction'),
(4, 25, 'Consumer Discretionary', 'Sensitive', 'XLY', 'Automobiles, retail, hotels, restaurants, leisure'),
(5, 30, 'Consumer Staples', 'Defensive', 'XLP', 'Food, beverages, household products, tobacco'),
(6, 35, 'Health Care', 'Defensive', 'XLV', 'Pharmaceuticals, biotechnology, medical devices, providers'),
(7, 40, 'Financials', 'Sensitive', 'XLF', 'Banks, insurance, capital markets, consumer finance'),
(8, 45, 'Information Technology', 'Sensitive', 'XLK', 'Software, hardware, semiconductors, IT services'),
(9, 50, 'Communication Services', 'Sensitive', 'XLC', 'Media, entertainment, interactive media, telecom'),
(10, 55, 'Utilities', 'Defensive', 'XLU', 'Electric, gas, water utilities'),
(11, 60, 'Real Estate', 'Sensitive', 'XLRE', 'REITs (residential, commercial, healthcare, industrial)')
ON CONFLICT (sector_id) DO NOTHING;

-- =============================================================================
-- VIEWS FOR ANALYSIS
-- =============================================================================

-- Latest Employment by Sector
CREATE OR REPLACE VIEW latest_sector_employment AS
SELECT
    s.sector_name,
    se.report_date,
    se.employment_count,
    se.unemployment_rate,
    se.job_openings,
    se.layoffs_discharges,
    ROW_NUMBER() OVER (PARTITION BY s.sector_id ORDER BY se.report_date DESC) as rn
FROM sectors s
JOIN sector_employment se ON s.sector_id = se.sector_id
QUALIFY rn = 1;

-- Recent Employment Events by Sector
CREATE OR REPLACE VIEW recent_employment_events AS
SELECT
    s.sector_name,
    ee.event_date,
    ee.company_name,
    ee.event_type,
    ee.employee_count,
    ee.impact_magnitude,
    ee.event_source
FROM sectors s
JOIN employment_events ee ON s.sector_id = ee.sector_id
WHERE ee.event_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY ee.event_date DESC;

-- Sector Employment Trends (3-month change)
CREATE OR REPLACE VIEW sector_employment_trends AS
SELECT
    s.sector_name,
    current.report_date as current_date,
    current.employment_count as current_employment,
    past.employment_count as past_employment,
    ((current.employment_count - past.employment_count) * 100.0 / past.employment_count) as pct_change_3mo,
    CASE
        WHEN current.employment_count > past.employment_count THEN 'Improving'
        WHEN current.employment_count < past.employment_count THEN 'Declining'
        ELSE 'Stable'
    END as trend
FROM sectors s
JOIN sector_employment current ON s.sector_id = current.sector_id
JOIN sector_employment past ON s.sector_id = past.sector_id
WHERE current.report_date = (SELECT MAX(report_date) FROM sector_employment)
  AND past.report_date = (SELECT MAX(report_date) FROM sector_employment WHERE report_date <= CURRENT_DATE - INTERVAL '90 days');

-- =============================================================================
-- SAMPLE QUERIES
-- =============================================================================

-- Query 1: Sectors with recent layoff activity
-- SELECT sector_name, COUNT(*) as layoff_count, SUM(employee_count) as total_affected
-- FROM employment_events ee
-- JOIN sectors s ON ee.sector_id = s.sector_id
-- WHERE event_type = 'layoff' AND event_date >= CURRENT_DATE - INTERVAL '30 days'
-- GROUP BY sector_name
-- ORDER BY total_affected DESC;

-- Query 2: Employment trend by sector (last 12 months)
-- SELECT s.sector_name, se.report_date, se.employment_count
-- FROM sector_employment se
-- JOIN sectors s ON se.sector_id = s.sector_id
-- WHERE se.report_date >= CURRENT_DATE - INTERVAL '12 months'
-- ORDER BY s.sector_name, se.report_date;

-- Query 3: Jobless claims spike detection (>10% increase week-over-week)
-- SELECT week_ending, initial_claims,
--        LAG(initial_claims) OVER (ORDER BY week_ending) as prev_week,
--        ((initial_claims - LAG(initial_claims) OVER (ORDER BY week_ending)) * 100.0 /
--         LAG(initial_claims) OVER (ORDER BY week_ending)) as pct_change
-- FROM jobless_claims
-- WHERE week_ending >= CURRENT_DATE - INTERVAL '6 months'
-- ORDER BY week_ending DESC;
