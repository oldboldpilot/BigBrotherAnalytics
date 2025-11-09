-- ============================================================================
-- GICS Sectors Schema for BigBrotherAnalytics
-- Author: Olumuyiwa Oluwasanmi
-- Date: 2025-11-09
-- ============================================================================

-- 11 GICS Sectors (Global Industry Classification Standard)
CREATE TABLE IF NOT EXISTS sectors (
    sector_code INTEGER PRIMARY KEY,
    sector_name VARCHAR NOT NULL,
    sector_etf VARCHAR NOT NULL,     -- Primary ETF for sector
    category VARCHAR NOT NULL,       -- Cyclical, Sensitive, or Defensive
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Company to Sector Mapping
CREATE TABLE IF NOT EXISTS company_sectors (
    symbol VARCHAR PRIMARY KEY,
    company_name VARCHAR NOT NULL,
    sector_code INTEGER NOT NULL,
    industry VARCHAR,
    sub_industry VARCHAR,
    market_cap_category VARCHAR,     -- Large, Mid, Small
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sector_code) REFERENCES sectors(sector_code)
);

-- Sector Performance Tracking
CREATE TABLE IF NOT EXISTS sector_performance (
    performance_date DATE NOT NULL,
    sector_code INTEGER NOT NULL,
    etf_close DOUBLE,
    etf_volume BIGINT,
    daily_return DOUBLE,
    relative_strength DOUBLE,        -- vs S&P 500
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (performance_date, sector_code),
    FOREIGN KEY (sector_code) REFERENCES sectors(sector_code)
);

-- Populate 11 GICS Sectors
INSERT INTO sectors (sector_code, sector_name, sector_etf, category, description) VALUES
(10, 'Energy', 'XLE', 'Cyclical', 'Oil, gas, consumable fuels, energy equipment & services'),
(15, 'Materials', 'XLB', 'Cyclical', 'Chemicals, construction materials, metals & mining'),
(20, 'Industrials', 'XLI', 'Sensitive', 'Aerospace, defense, machinery, transportation'),
(25, 'Consumer Discretionary', 'XLY', 'Sensitive', 'Retail, hotels, restaurants, autos, consumer services'),
(30, 'Consumer Staples', 'XLP', 'Defensive', 'Food, beverages, tobacco, household products'),
(35, 'Health Care', 'XLV', 'Defensive', 'Pharma, biotech, healthcare equipment & services'),
(40, 'Financials', 'XLF', 'Sensitive', 'Banks, insurance, capital markets, financial services'),
(45, 'Information Technology', 'XLK', 'Sensitive', 'Software, hardware, semiconductors, IT services'),
(50, 'Communication Services', 'XLC', 'Sensitive', 'Telecom, media & entertainment, interactive media'),
(55, 'Utilities', 'XLU', 'Defensive', 'Electric, gas, water utilities'),
(60, 'Real Estate', 'XLRE', 'Sensitive', 'REITs, real estate management & development')
ON CONFLICT DO NOTHING;

-- Map current 24 stocks to sectors
INSERT INTO company_sectors (symbol, company_name, sector_code, industry, market_cap_category) VALUES
-- Technology (45)
('AAPL', 'Apple Inc', 45, 'Technology Hardware', 'Large'),
('MSFT', 'Microsoft Corp', 45, 'Software', 'Large'),
('GOOGL', 'Alphabet Inc', 50, 'Interactive Media', 'Large'),
('META', 'Meta Platforms', 50, 'Interactive Media', 'Large'),
('NVDA', 'NVIDIA Corp', 45, 'Semiconductors', 'Large'),
('AMZN', 'Amazon.com', 25, 'Internet Retail', 'Large'),
('TSLA', 'Tesla Inc', 25, 'Automobiles', 'Large'),

-- Healthcare (35)
('JNJ', 'Johnson & Johnson', 35, 'Pharmaceuticals', 'Large'),
('PFE', 'Pfizer Inc', 35, 'Pharmaceuticals', 'Large'),
('UNH', 'UnitedHealth Group', 35, 'Healthcare Services', 'Large'),

-- Financials (40)
('JPM', 'JPMorgan Chase', 40, 'Banks', 'Large'),
('BAC', 'Bank of America', 40, 'Banks', 'Large'),
('WFC', 'Wells Fargo', 40, 'Banks', 'Large'),
('GS', 'Goldman Sachs', 40, 'Capital Markets', 'Large'),

-- Consumer (25, 30)
('WMT', 'Walmart', 30, 'Consumer Staples', 'Large'),
('HD', 'Home Depot', 25, 'Specialty Retail', 'Large'),
('MCD', 'McDonalds', 25, 'Hotels/Restaurants', 'Large'),

-- Industrials (20)
('BA', 'Boeing', 20, 'Aerospace & Defense', 'Large'),
('CAT', 'Caterpillar', 20, 'Machinery', 'Large'),
('UPS', 'United Parcel Service', 20, 'Air Freight', 'Large'),

-- Energy (10)
('XOM', 'Exxon Mobil', 10, 'Oil & Gas', 'Large'),
('CVX', 'Chevron', 10, 'Oil & Gas', 'Large'),

-- Utilities (55)
('NEE', 'NextEra Energy', 55, 'Electric Utilities', 'Large'),

-- Materials (15)
('LIN', 'Linde plc', 15, 'Chemicals', 'Large')
ON CONFLICT DO NOTHING;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sector_performance_date ON sector_performance(performance_date);
CREATE INDEX IF NOT EXISTS idx_sector_performance_sector ON sector_performance(sector_code);
CREATE INDEX IF NOT EXISTS idx_company_sectors_sector ON company_sectors(sector_code);

-- Views for sector analysis
CREATE OR REPLACE VIEW sector_summary AS
SELECT 
    s.sector_code,
    s.sector_name,
    s.sector_etf,
    s.category,
    COUNT(DISTINCT cs.symbol) as company_count
FROM sectors s
LEFT JOIN company_sectors cs ON s.sector_code = cs.sector_code
GROUP BY s.sector_code, s.sector_name, s.sector_etf, s.category
ORDER BY s.sector_code;

CREATE OR REPLACE VIEW sector_diversification AS
SELECT 
    s.sector_name,
    s.category,
    COUNT(cs.symbol) as holdings,
    ROUND(COUNT(cs.symbol) * 100.0 / (SELECT COUNT(*) FROM company_sectors), 2) as portfolio_percentage
FROM sectors s
LEFT JOIN company_sectors cs ON s.sector_code = cs.sector_code
GROUP BY s.sector_name, s.category
ORDER BY holdings DESC;
