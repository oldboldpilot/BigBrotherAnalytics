-- BigBrotherAnalytics - Order Tracking Schema
-- DuckDB SQL Schema for Schwab API Order Management
--
-- Purpose: Track all orders for compliance, audit trail, and analysis
-- Features:
--   - Complete order lifecycle tracking
--   - Audit trail for all order modifications
--   - Performance analytics
--   - Regulatory compliance

-- ============================================================================
-- Orders Table (Main order tracking)
-- ============================================================================

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE NOT NULL,
    account_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(15) NOT NULL,  -- Buy, Sell, BuyToCover, SellShort
    quantity INTEGER NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    order_type VARCHAR(20) NOT NULL,  -- Market, Limit, Stop, StopLimit, TrailingStop
    order_class VARCHAR(15) DEFAULT 'Simple',  -- Simple, Bracket, OCO, OTO
    limit_price DECIMAL(10,2),
    stop_price DECIMAL(10,2),
    trail_amount DECIMAL(10,2),
    avg_fill_price DECIMAL(10,2),
    status VARCHAR(20) NOT NULL,  -- Pending, Working, Filled, PartiallyFilled, Canceled, Rejected
    duration VARCHAR(10) DEFAULT 'Day',  -- Day, GTC, GTD, FOK, IOC
    parent_order_id VARCHAR(50),  -- For bracket/OCO orders
    dry_run BOOLEAN DEFAULT FALSE,
    rejection_reason VARCHAR(255),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    filled_at TIMESTAMP,

    -- Constraints
    CHECK (quantity > 0),
    CHECK (filled_quantity >= 0),
    CHECK (filled_quantity <= quantity)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_orders_account ON orders(account_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_parent ON orders(parent_order_id);

-- ============================================================================
-- Order Updates Table (Audit trail)
-- ============================================================================

CREATE TABLE IF NOT EXISTS order_updates (
    id INTEGER PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL,
    field_name VARCHAR(50) NOT NULL,
    old_value VARCHAR(255),
    new_value VARCHAR(255) NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    updated_by VARCHAR(100) DEFAULT 'SYSTEM',

    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE INDEX IF NOT EXISTS idx_updates_order ON order_updates(order_id);
CREATE INDEX IF NOT EXISTS idx_updates_timestamp ON order_updates(updated_at);

-- ============================================================================
-- Order Fills Table (Execution details)
-- ============================================================================

CREATE TABLE IF NOT EXISTS order_fills (
    id INTEGER PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL,
    fill_id VARCHAR(50) UNIQUE NOT NULL,
    fill_quantity INTEGER NOT NULL,
    fill_price DECIMAL(10,2) NOT NULL,
    commission DECIMAL(10,2) DEFAULT 0.0,
    fill_timestamp TIMESTAMP NOT NULL,
    exchange VARCHAR(20),

    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    CHECK (fill_quantity > 0),
    CHECK (fill_price > 0)
);

CREATE INDEX IF NOT EXISTS idx_fills_order ON order_fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON order_fills(fill_timestamp);

-- ============================================================================
-- Order Rejections Table (Track all rejections for analysis)
-- ============================================================================

CREATE TABLE IF NOT EXISTS order_rejections (
    id INTEGER PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL,
    rejection_code VARCHAR(50),
    rejection_reason VARCHAR(255) NOT NULL,
    rejected_at TIMESTAMP NOT NULL,
    symbol VARCHAR(20),
    quantity INTEGER,
    attempted_price DECIMAL(10,2),
    account_balance DECIMAL(15,2),

    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE INDEX IF NOT EXISTS idx_rejections_timestamp ON order_rejections(rejected_at);
CREATE INDEX IF NOT EXISTS idx_rejections_code ON order_rejections(rejection_code);

-- ============================================================================
-- Order Performance Table (Analytics)
-- ============================================================================

CREATE TABLE IF NOT EXISTS order_performance (
    id INTEGER PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    entry_price DECIMAL(10,2) NOT NULL,
    exit_price DECIMAL(10,2),
    quantity INTEGER NOT NULL,
    pnl DECIMAL(15,2),
    pnl_percent DECIMAL(10,4),
    hold_duration_seconds INTEGER,
    slippage DECIMAL(10,2),  -- Difference between expected and actual fill
    entry_timestamp TIMESTAMP NOT NULL,
    exit_timestamp TIMESTAMP,

    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE INDEX IF NOT EXISTS idx_performance_symbol ON order_performance(symbol);
CREATE INDEX IF NOT EXISTS idx_performance_pnl ON order_performance(pnl);

-- ============================================================================
-- Daily Order Summary (Aggregated metrics)
-- ============================================================================

CREATE TABLE IF NOT EXISTS daily_order_summary (
    summary_date DATE PRIMARY KEY,
    total_orders INTEGER DEFAULT 0,
    filled_orders INTEGER DEFAULT 0,
    canceled_orders INTEGER DEFAULT 0,
    rejected_orders INTEGER DEFAULT 0,
    total_volume BIGINT DEFAULT 0,
    total_pnl DECIMAL(15,2) DEFAULT 0.0,
    win_rate DECIMAL(5,2),  -- Percentage
    avg_fill_time_seconds INTEGER,
    max_order_size DECIMAL(15,2),
    unique_symbols INTEGER,
    dry_run_orders INTEGER DEFAULT 0
);

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- Active orders view
CREATE OR REPLACE VIEW active_orders AS
SELECT
    order_id,
    account_id,
    symbol,
    side,
    quantity,
    filled_quantity,
    order_type,
    limit_price,
    stop_price,
    status,
    created_at,
    updated_at
FROM orders
WHERE status IN ('Pending', 'Working', 'Queued', 'Accepted', 'PartiallyFilled');

-- Today's orders view
CREATE OR REPLACE VIEW todays_orders AS
SELECT
    order_id,
    symbol,
    side,
    quantity,
    order_type,
    status,
    dry_run,
    created_at
FROM orders
WHERE DATE(created_at) = CURRENT_DATE
ORDER BY created_at DESC;

-- Filled orders with P&L view
CREATE OR REPLACE VIEW filled_orders_with_pnl AS
SELECT
    o.order_id,
    o.symbol,
    o.quantity,
    o.avg_fill_price,
    o.filled_at,
    p.pnl,
    p.pnl_percent,
    p.hold_duration_seconds
FROM orders o
LEFT JOIN order_performance p ON o.order_id = p.order_id
WHERE o.status = 'Filled'
ORDER BY o.filled_at DESC;

-- Order rejection summary view
CREATE OR REPLACE VIEW rejection_summary AS
SELECT
    rejection_code,
    COUNT(*) as rejection_count,
    COUNT(DISTINCT symbol) as affected_symbols,
    MIN(rejected_at) as first_occurrence,
    MAX(rejected_at) as last_occurrence
FROM order_rejections
GROUP BY rejection_code
ORDER BY rejection_count DESC;

-- ============================================================================
-- Compliance Functions
-- ============================================================================

-- Function to check daily order limit
CREATE OR REPLACE MACRO check_daily_order_limit(account VARCHAR, limit_count INTEGER) AS (
    SELECT COUNT(*) < limit_count as within_limit
    FROM orders
    WHERE account_id = account
    AND DATE(created_at) = CURRENT_DATE
    AND dry_run = FALSE
);

-- Function to get order count by status
CREATE OR REPLACE MACRO order_count_by_status(start_date TIMESTAMP, end_date TIMESTAMP) AS (
    SELECT
        status,
        COUNT(*) as count,
        COUNT(DISTINCT symbol) as unique_symbols,
        SUM(quantity) as total_quantity
    FROM orders
    WHERE created_at BETWEEN start_date AND end_date
    GROUP BY status
);

-- Function to calculate win rate
CREATE OR REPLACE MACRO calculate_win_rate(start_date DATE, end_date DATE) AS (
    SELECT
        COUNT(CASE WHEN pnl > 0 THEN 1 END)::FLOAT / COUNT(*)::FLOAT * 100 as win_rate_percent
    FROM order_performance
    WHERE DATE(entry_timestamp) BETWEEN start_date AND end_date
);

-- ============================================================================
-- Sample Queries for Analysis
-- ============================================================================

-- Most active symbols
-- SELECT symbol, COUNT(*) as order_count, SUM(quantity) as total_volume
-- FROM orders
-- WHERE created_at >= CURRENT_DATE - INTERVAL 30 DAY
-- GROUP BY symbol
-- ORDER BY order_count DESC
-- LIMIT 10;

-- Average fill times
-- SELECT
--     symbol,
--     AVG(EXTRACT(EPOCH FROM (filled_at - created_at))) as avg_fill_time_seconds,
--     COUNT(*) as fill_count
-- FROM orders
-- WHERE status = 'Filled'
--   AND filled_at IS NOT NULL
-- GROUP BY symbol
-- ORDER BY fill_count DESC;

-- Daily P&L summary
-- SELECT
--     DATE(entry_timestamp) as trade_date,
--     COUNT(*) as trades,
--     SUM(pnl) as total_pnl,
--     AVG(pnl) as avg_pnl,
--     COUNT(CASE WHEN pnl > 0 THEN 1 END) as winners,
--     COUNT(CASE WHEN pnl < 0 THEN 1 END) as losers
-- FROM order_performance
-- GROUP BY DATE(entry_timestamp)
-- ORDER BY trade_date DESC;

-- ============================================================================
-- Compliance Notes
-- ============================================================================

-- COMPLIANCE REQUIREMENTS:
-- 1. All orders must be logged before submission
-- 2. All modifications must be tracked in order_updates
-- 3. Retention period: Minimum 7 years for regulatory compliance
-- 4. Audit trail must be immutable (use append-only writes)
-- 5. Daily reconciliation against broker statements required

-- REGULATORY REFERENCES:
-- - SEC Rule 17a-4: Record retention requirements
-- - FINRA Rule 4511: General requirements for books and records
-- - Reg SHO: Short sale tracking requirements
