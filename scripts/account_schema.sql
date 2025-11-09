-- ============================================================================
-- BigBrotherAnalytics - Schwab Account Schema
-- Position Tracking and Portfolio Analytics
--
-- Author: Olumuyiwa Oluwasanmi
-- Date: 2025-11-09
-- Database: DuckDB
-- ============================================================================

-- ============================================================================
-- ACCOUNT TABLES
-- ============================================================================

-- Account Information
CREATE TABLE IF NOT EXISTS accounts (
    account_id VARCHAR PRIMARY KEY,
    account_hash VARCHAR NOT NULL,
    account_type VARCHAR NOT NULL,           -- CASH, MARGIN, IRA, etc.
    account_nickname VARCHAR,
    is_day_trader BOOLEAN DEFAULT FALSE,
    is_closing_only BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Account Balances (time-series for tracking)
CREATE TABLE IF NOT EXISTS account_balances (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    total_equity DECIMAL(15,2) NOT NULL,
    cash DECIMAL(15,2) NOT NULL,
    cash_available DECIMAL(15,2) NOT NULL,
    buying_power DECIMAL(15,2) NOT NULL,
    day_trading_buying_power DECIMAL(15,2) NOT NULL,
    margin_balance DECIMAL(15,2) DEFAULT 0.0,
    margin_equity DECIMAL(15,2) DEFAULT 0.0,
    long_market_value DECIMAL(15,2) DEFAULT 0.0,
    short_market_value DECIMAL(15,2) DEFAULT 0.0,
    unsettled_cash DECIMAL(15,2) DEFAULT 0.0,
    maintenance_call DECIMAL(15,2) DEFAULT 0.0,
    reg_t_call DECIMAL(15,2) DEFAULT 0.0,
    equity_percentage DECIMAL(5,2) DEFAULT 100.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, timestamp)
);

-- ============================================================================
-- POSITION TABLES
-- ============================================================================

-- Current Positions (live snapshot)
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,         -- EQUITY, OPTION, BOND, etc.
    cusip VARCHAR(20),
    quantity BIGINT NOT NULL,
    long_quantity BIGINT DEFAULT 0,
    short_quantity BIGINT DEFAULT 0,
    average_cost DECIMAL(12,4) NOT NULL,
    current_price DECIMAL(12,4) NOT NULL,
    market_value DECIMAL(15,2) NOT NULL,
    cost_basis DECIMAL(15,2) NOT NULL,
    unrealized_pnl DECIMAL(15,2) NOT NULL,
    unrealized_pnl_percent DECIMAL(10,4) NOT NULL,
    day_pnl DECIMAL(15,2) DEFAULT 0.0,
    day_pnl_percent DECIMAL(10,4) DEFAULT 0.0,
    previous_close DECIMAL(12,4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, symbol)
);

-- Position History (for time-series analysis)
CREATE TABLE IF NOT EXISTS position_history (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,
    quantity BIGINT NOT NULL,
    average_cost DECIMAL(12,4) NOT NULL,
    current_price DECIMAL(12,4) NOT NULL,
    market_value DECIMAL(15,2) NOT NULL,
    cost_basis DECIMAL(15,2) NOT NULL,
    unrealized_pnl DECIMAL(15,2) NOT NULL,
    unrealized_pnl_percent DECIMAL(10,4) NOT NULL,
    day_pnl DECIMAL(15,2) DEFAULT 0.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Position Changes (for tracking entries/exits)
CREATE TABLE IF NOT EXISTS position_changes (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    change_type VARCHAR(20) NOT NULL,        -- OPENED, CLOSED, INCREASED, DECREASED
    quantity_before BIGINT,
    quantity_after BIGINT NOT NULL,
    quantity_change BIGINT NOT NULL,
    average_cost_before DECIMAL(12,4),
    average_cost_after DECIMAL(12,4) NOT NULL,
    price_at_change DECIMAL(12,4) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TRANSACTION TABLES
-- ============================================================================

-- Transaction Records
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    symbol VARCHAR(20),
    transaction_type VARCHAR(50) NOT NULL,   -- Trade, Dividend, etc.
    instruction VARCHAR(20),                  -- BUY, SELL, BUY_TO_COVER, SELL_SHORT
    description TEXT,
    transaction_date TIMESTAMP NOT NULL,
    settlement_date TIMESTAMP,
    net_amount DECIMAL(15,2) NOT NULL,
    gross_amount DECIMAL(15,2) NOT NULL,
    quantity BIGINT DEFAULT 0,
    price DECIMAL(12,4) DEFAULT 0.0,
    commission DECIMAL(10,2) DEFAULT 0.0,
    fees DECIMAL(10,2) DEFAULT 0.0,
    reg_fee DECIMAL(10,2) DEFAULT 0.0,
    sec_fee DECIMAL(10,2) DEFAULT 0.0,
    position_effect VARCHAR(20),              -- OPENING, CLOSING
    asset_type VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PORTFOLIO ANALYTICS TABLES
-- ============================================================================

-- Daily Portfolio Snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    snapshot_date DATE NOT NULL,
    total_equity DECIMAL(15,2) NOT NULL,
    total_cash DECIMAL(15,2) NOT NULL,
    total_market_value DECIMAL(15,2) NOT NULL,
    total_cost_basis DECIMAL(15,2) NOT NULL,
    total_unrealized_pnl DECIMAL(15,2) NOT NULL,
    total_unrealized_pnl_percent DECIMAL(10,4) NOT NULL,
    total_day_pnl DECIMAL(15,2) NOT NULL,
    total_day_pnl_percent DECIMAL(10,4) NOT NULL,
    position_count INTEGER NOT NULL,
    long_position_count INTEGER NOT NULL,
    short_position_count INTEGER NOT NULL,
    largest_position_percent DECIMAL(10,4) NOT NULL,
    portfolio_concentration DECIMAL(10,6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, snapshot_date)
);

-- Real-time Performance Tracking (intraday)
CREATE TABLE IF NOT EXISTS performance_tracking (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES accounts(account_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    equity DECIMAL(15,2) NOT NULL,
    day_pnl DECIMAL(15,2) NOT NULL,
    day_pnl_percent DECIMAL(10,4) NOT NULL,
    realized_pnl DECIMAL(15,2) DEFAULT 0.0,
    unrealized_pnl DECIMAL(15,2) NOT NULL,
    buying_power DECIMAL(15,2) NOT NULL,
    margin_usage_percent DECIMAL(5,2) DEFAULT 0.0
);

-- Position Risk Metrics
CREATE TABLE IF NOT EXISTS position_risks (
    id INTEGER PRIMARY KEY,
    account_id VARCHAR NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    calculation_date DATE NOT NULL,
    position_size_percent DECIMAL(10,4) NOT NULL,
    var_95 DECIMAL(15,2),                     -- Value at Risk (95%)
    expected_shortfall DECIMAL(15,2),         -- Conditional VaR
    beta DECIMAL(8,4),                        -- Market beta
    volatility DECIMAL(8,4),                  -- Historical volatility
    sharpe_ratio DECIMAL(8,4),                -- Risk-adjusted return
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, symbol, calculation_date)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Account indexes
CREATE INDEX IF NOT EXISTS idx_accounts_type ON accounts(account_type);

-- Balance indexes
CREATE INDEX IF NOT EXISTS idx_account_balances_account_time
    ON account_balances(account_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_account_balances_timestamp
    ON account_balances(timestamp DESC);

-- Position indexes
CREATE INDEX IF NOT EXISTS idx_positions_account ON positions(account_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_asset_type ON positions(asset_type);
CREATE INDEX IF NOT EXISTS idx_positions_updated ON positions(updated_at DESC);

-- Position history indexes
CREATE INDEX IF NOT EXISTS idx_position_history_account_symbol
    ON position_history(account_id, symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_position_history_timestamp
    ON position_history(timestamp DESC);

-- Position changes indexes
CREATE INDEX IF NOT EXISTS idx_position_changes_account_symbol
    ON position_changes(account_id, symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_position_changes_type
    ON position_changes(change_type, timestamp DESC);

-- Transaction indexes
CREATE INDEX IF NOT EXISTS idx_transactions_account_date
    ON transactions(account_id, transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_symbol
    ON transactions(symbol, transaction_date DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_type
    ON transactions(transaction_type, transaction_date DESC);

-- Portfolio snapshot indexes
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_account_date
    ON portfolio_snapshots(account_id, snapshot_date DESC);

-- Performance tracking indexes
CREATE INDEX IF NOT EXISTS idx_performance_tracking_account_time
    ON performance_tracking(account_id, timestamp DESC);

-- Risk metrics indexes
CREATE INDEX IF NOT EXISTS idx_position_risks_account_date
    ON position_risks(account_id, calculation_date DESC);
CREATE INDEX IF NOT EXISTS idx_position_risks_symbol
    ON position_risks(symbol, calculation_date DESC);

-- ============================================================================
-- VIEWS FOR ANALYSIS
-- ============================================================================

-- Latest Positions with P/L
CREATE OR REPLACE VIEW v_current_positions AS
SELECT
    p.account_id,
    p.symbol,
    p.asset_type,
    p.quantity,
    p.average_cost,
    p.current_price,
    p.market_value,
    p.cost_basis,
    p.unrealized_pnl,
    p.unrealized_pnl_percent,
    p.day_pnl,
    p.day_pnl_percent,
    p.updated_at,
    CASE
        WHEN p.quantity > 0 THEN 'LONG'
        WHEN p.quantity < 0 THEN 'SHORT'
        ELSE 'FLAT'
    END as position_side,
    ROUND((p.market_value / NULLIF(b.total_equity, 0)) * 100, 2) as portfolio_weight_percent
FROM positions p
LEFT JOIN (
    SELECT account_id, total_equity
    FROM account_balances
    QUALIFY ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY timestamp DESC) = 1
) b ON p.account_id = b.account_id
WHERE p.quantity != 0
ORDER BY p.market_value DESC;

-- Latest Account Balance
CREATE OR REPLACE VIEW v_latest_balance AS
SELECT
    account_id,
    total_equity,
    cash,
    buying_power,
    day_trading_buying_power,
    margin_balance,
    long_market_value,
    short_market_value,
    unsettled_cash,
    ROUND((margin_balance / NULLIF(total_equity, 0)) * 100, 2) as margin_usage_percent,
    CASE
        WHEN maintenance_call > 0 OR reg_t_call > 0 THEN 'MARGIN_CALL'
        WHEN (margin_balance / NULLIF(total_equity, 0)) > 0.5 THEN 'HIGH_MARGIN'
        ELSE 'NORMAL'
    END as account_status,
    timestamp
FROM account_balances
QUALIFY ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY timestamp DESC) = 1;

-- Top Gainers/Losers
CREATE OR REPLACE VIEW v_top_performers AS
SELECT
    symbol,
    asset_type,
    market_value,
    unrealized_pnl,
    unrealized_pnl_percent,
    day_pnl,
    day_pnl_percent,
    CASE
        WHEN unrealized_pnl >= 0 THEN 'WINNER'
        ELSE 'LOSER'
    END as performance_category
FROM positions
WHERE quantity != 0
ORDER BY unrealized_pnl DESC;

-- Position Concentration Risk
CREATE OR REPLACE VIEW v_concentration_risk AS
SELECT
    p.account_id,
    COUNT(*) as total_positions,
    COUNT(*) FILTER (WHERE portfolio_weight >= 10.0) as positions_over_10_percent,
    COUNT(*) FILTER (WHERE portfolio_weight >= 20.0) as positions_over_20_percent,
    MAX(portfolio_weight) as largest_position_percent,
    SUM(portfolio_weight * portfolio_weight) as herfindahl_index,
    1.0 - SUM(portfolio_weight * portfolio_weight) as diversification_score
FROM (
    SELECT
        p.account_id,
        p.symbol,
        (p.market_value / NULLIF(b.total_equity, 0)) * 100 as portfolio_weight
    FROM positions p
    LEFT JOIN (
        SELECT account_id, total_equity
        FROM account_balances
        QUALIFY ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY timestamp DESC) = 1
    ) b ON p.account_id = b.account_id
    WHERE p.quantity != 0
) sub
GROUP BY p.account_id;

-- Daily P/L Summary
CREATE OR REPLACE VIEW v_daily_pnl AS
SELECT
    account_id,
    snapshot_date,
    total_equity,
    total_day_pnl,
    total_day_pnl_percent,
    total_unrealized_pnl,
    LAG(total_equity) OVER (PARTITION BY account_id ORDER BY snapshot_date) as prev_day_equity,
    total_equity - LAG(total_equity) OVER (PARTITION BY account_id ORDER BY snapshot_date) as equity_change
FROM portfolio_snapshots
ORDER BY account_id, snapshot_date DESC;

-- Recent Transactions Summary
CREATE OR REPLACE VIEW v_recent_transactions AS
SELECT
    transaction_date::DATE as trade_date,
    symbol,
    instruction,
    quantity,
    price,
    gross_amount,
    commission + fees + reg_fee + sec_fee as total_costs,
    net_amount,
    asset_type
FROM transactions
WHERE transaction_type = 'Trade'
  AND transaction_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY transaction_date DESC;

-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================

-- Query 1: Current portfolio summary
-- SELECT * FROM v_current_positions ORDER BY market_value DESC;

-- Query 2: Account balance with risk metrics
-- SELECT * FROM v_latest_balance;

-- Query 3: P/L performance over time
-- SELECT * FROM v_daily_pnl WHERE snapshot_date >= CURRENT_DATE - INTERVAL '30 days';

-- Query 4: Position concentration analysis
-- SELECT * FROM v_concentration_risk;

-- Query 5: Top performers
-- SELECT * FROM v_top_performers LIMIT 10;

-- Query 6: Real-time position tracking (last hour)
-- SELECT * FROM performance_tracking
-- WHERE timestamp >= NOW() - INTERVAL '1 hour'
-- ORDER BY timestamp DESC;
