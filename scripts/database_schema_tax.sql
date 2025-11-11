-- Tax Tracking Schema for BigBrotherAnalytics
-- Author: Olumuyiwa Oluwasanmi
-- Date: 2025-11-10
-- Purpose: Track tax implications of trading activity with 3% fee structure

-- ============================================================================
-- Tax Records Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS tax_records (
    id INTEGER PRIMARY KEY,
    trade_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,

    -- Trade details
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    holding_period_days INTEGER NOT NULL,

    -- Financial data
    cost_basis DOUBLE NOT NULL,           -- Entry cost
    proceeds DOUBLE NOT NULL,             -- Exit proceeds
    gross_pnl DOUBLE NOT NULL,            -- Gross profit/loss

    -- Fees (3% of trade value)
    trading_fees DOUBLE NOT NULL DEFAULT 0.0,
    pnl_after_fees DOUBLE NOT NULL,       -- P&L after fees

    -- Tax classification
    is_long_term BOOLEAN NOT NULL,        -- > 365 days
    is_options BOOLEAN DEFAULT false,
    is_index_option BOOLEAN DEFAULT false, -- Section 1256 eligible

    -- Tax calculations
    short_term_gain DOUBLE DEFAULT 0.0,
    long_term_gain DOUBLE DEFAULT 0.0,
    short_term_loss DOUBLE DEFAULT 0.0,
    long_term_loss DOUBLE DEFAULT 0.0,

    -- Tax rates applied
    federal_tax_rate DOUBLE NOT NULL,     -- Short-term: 24%, Long-term: 15%
    state_tax_rate DOUBLE NOT NULL,       -- 5% default
    medicare_surtax DOUBLE NOT NULL,      -- 3.8% NIIT
    effective_tax_rate DOUBLE NOT NULL,   -- Combined rate

    -- Tax owed
    tax_owed DOUBLE NOT NULL,
    net_pnl_after_tax DOUBLE NOT NULL,    -- Final profit after fees and tax

    -- Wash sale tracking
    wash_sale_disallowed BOOLEAN DEFAULT false,
    wash_sale_amount DOUBLE DEFAULT 0.0,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_tax_records_symbol ON tax_records(symbol);
CREATE INDEX IF NOT EXISTS idx_tax_records_exit_time ON tax_records(exit_time);
CREATE INDEX IF NOT EXISTS idx_tax_records_trade_id ON tax_records(trade_id);

-- ============================================================================
-- Tax Summary Table (Daily/Monthly/Annual Aggregates)
-- ============================================================================
CREATE TABLE IF NOT EXISTS tax_summary (
    id INTEGER PRIMARY KEY,
    period_type VARCHAR NOT NULL,        -- 'daily', 'monthly', 'quarterly', 'annual'
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Gross P&L
    total_gross_pnl DOUBLE NOT NULL,
    total_trades INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL,
    losing_trades INTEGER NOT NULL,

    -- Fees
    total_trading_fees DOUBLE NOT NULL,  -- 3% of all trades
    pnl_after_fees DOUBLE NOT NULL,

    -- Tax breakdown
    short_term_gains DOUBLE NOT NULL,
    long_term_gains DOUBLE NOT NULL,
    short_term_losses DOUBLE NOT NULL,
    long_term_losses DOUBLE NOT NULL,

    taxable_short_term DOUBLE NOT NULL,  -- After offsetting losses
    taxable_long_term DOUBLE NOT NULL,   -- After offsetting losses

    -- Tax owed
    total_tax_owed DOUBLE NOT NULL,
    federal_tax DOUBLE NOT NULL,
    state_tax DOUBLE NOT NULL,
    medicare_surtax_amount DOUBLE NOT NULL,

    -- After-tax results
    net_pnl_after_tax DOUBLE NOT NULL,
    effective_tax_rate DOUBLE NOT NULL,
    tax_efficiency DOUBLE NOT NULL,      -- Net / Gross

    -- Wash sales
    wash_sales_count INTEGER NOT NULL DEFAULT 0,
    wash_sale_loss_disallowed DOUBLE NOT NULL DEFAULT 0.0,

    -- Carryforward
    capital_loss_carryforward DOUBLE DEFAULT 0.0,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast period lookups
CREATE INDEX IF NOT EXISTS idx_tax_summary_period ON tax_summary(period_type, period_start, period_end);

-- ============================================================================
-- Tax Configuration Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS tax_config (
    id INTEGER PRIMARY KEY,
    user_id VARCHAR DEFAULT 'default',

    -- Tax rates
    short_term_rate DOUBLE DEFAULT 0.24,  -- 24% federal
    long_term_rate DOUBLE DEFAULT 0.15,   -- 15% federal
    state_tax_rate DOUBLE DEFAULT 0.05,   -- 5% state
    medicare_surtax DOUBLE DEFAULT 0.038, -- 3.8% NIIT

    -- Trading fees
    trading_fee_percent DOUBLE DEFAULT 0.03, -- 3% per trade

    -- Trading status
    is_pattern_day_trader BOOLEAN DEFAULT true,
    is_section_1256_trader BOOLEAN DEFAULT false,

    -- Wash sale tracking
    track_wash_sales BOOLEAN DEFAULT true,
    wash_sale_window_days INTEGER DEFAULT 30,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default configuration (DuckDB-compatible)
INSERT INTO tax_config (id, user_id) VALUES (1, 'default')
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- Tax Events Table (Audit Trail)
-- ============================================================================
CREATE TABLE IF NOT EXISTS tax_events (
    id INTEGER PRIMARY KEY,
    event_type VARCHAR NOT NULL,         -- 'trade_closed', 'quarterly_tax', 'wash_sale_detected'
    event_date TIMESTAMP NOT NULL,

    -- Event details
    symbol VARCHAR,
    trade_id VARCHAR,
    amount DOUBLE,
    tax_impact DOUBLE,

    -- Description
    description TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tax_events_date ON tax_events(event_date);
CREATE INDEX IF NOT EXISTS idx_tax_events_type ON tax_events(event_type);

-- ============================================================================
-- Views for Dashboard
-- ============================================================================

-- Current YTD Tax Summary
CREATE OR REPLACE VIEW v_ytd_tax_summary AS
SELECT
    SUM(gross_pnl) as total_gross_pnl,
    SUM(trading_fees) as total_trading_fees,
    SUM(pnl_after_fees) as total_pnl_after_fees,
    SUM(tax_owed) as total_tax_owed,
    SUM(net_pnl_after_tax) as total_net_after_tax,
    AVG(effective_tax_rate) as avg_effective_tax_rate,
    COUNT(*) as total_trades,
    SUM(CASE WHEN wash_sale_disallowed THEN 1 ELSE 0 END) as wash_sales_count,
    SUM(wash_sale_amount) as total_wash_sale_loss
FROM tax_records
WHERE EXTRACT(YEAR FROM exit_time) = EXTRACT(YEAR FROM CURRENT_DATE);

-- Monthly Tax Summary
CREATE OR REPLACE VIEW v_monthly_tax_summary AS
SELECT
    EXTRACT(YEAR FROM exit_time) as year,
    EXTRACT(MONTH FROM exit_time) as month,
    SUM(gross_pnl) as gross_pnl,
    SUM(trading_fees) as trading_fees,
    SUM(pnl_after_fees) as pnl_after_fees,
    SUM(tax_owed) as tax_owed,
    SUM(net_pnl_after_tax) as net_after_tax,
    AVG(effective_tax_rate) as avg_tax_rate,
    COUNT(*) as trades
FROM tax_records
GROUP BY EXTRACT(YEAR FROM exit_time), EXTRACT(MONTH FROM exit_time)
ORDER BY year DESC, month DESC;

-- Tax Efficiency by Symbol
CREATE OR REPLACE VIEW v_tax_efficiency_by_symbol AS
SELECT
    symbol,
    COUNT(*) as trades,
    SUM(gross_pnl) as gross_pnl,
    SUM(trading_fees) as trading_fees,
    SUM(tax_owed) as tax_owed,
    SUM(net_pnl_after_tax) as net_after_tax,
    CASE
        WHEN SUM(gross_pnl) > 0 THEN SUM(net_pnl_after_tax) / SUM(gross_pnl)
        ELSE 0.0
    END as tax_efficiency,
    AVG(effective_tax_rate) as avg_tax_rate
FROM tax_records
GROUP BY symbol
HAVING COUNT(*) >= 3  -- Min 3 trades for meaningful stats
ORDER BY tax_efficiency DESC;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE tax_records IS 'Individual trade tax records with 3% fee and full tax breakdown';
COMMENT ON TABLE tax_summary IS 'Aggregated tax summaries by period (daily/monthly/quarterly/annual)';
COMMENT ON TABLE tax_config IS 'Tax configuration per user (rates, fees, wash sale tracking)';
COMMENT ON TABLE tax_events IS 'Audit trail of tax-related events';

COMMENT ON COLUMN tax_records.trading_fees IS '3% trading fee per trade (user requirement)';
COMMENT ON COLUMN tax_records.effective_tax_rate IS 'Combined federal + state + Medicare surtax';
COMMENT ON COLUMN tax_summary.tax_efficiency IS 'Net after-tax P&L / Gross P&L (higher is better)';
