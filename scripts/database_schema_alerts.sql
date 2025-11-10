-- BigBrotherAnalytics: Alerts Database Schema
-- Stores alert history for system monitoring and notification tracking
--
-- Author: Olumuyiwa Oluwasanmi
-- Date: 2025-11-10
-- Phase 4, Week 3: Custom Alerts System

-- ============================================================================
-- ALERTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,

    -- Alert Classification
    alert_type VARCHAR NOT NULL,  -- trading, data, system, performance
    alert_subtype VARCHAR,         -- e.g., pnl_threshold, stop_loss, circuit_breaker
    severity VARCHAR NOT NULL,     -- INFO, WARNING, ERROR, CRITICAL

    -- Alert Content
    message VARCHAR NOT NULL,      -- Human-readable alert message
    context TEXT,                  -- JSON context data (symbol, amount, details)

    -- Alert Metadata
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR,                -- Source component (e.g., orders_manager, schwab_api)

    -- Delivery Status
    sent BOOLEAN DEFAULT false,
    sent_timestamp TIMESTAMP,
    email_sent BOOLEAN DEFAULT false,
    slack_sent BOOLEAN DEFAULT false,
    sms_sent BOOLEAN DEFAULT false,
    browser_sent BOOLEAN DEFAULT false,

    -- Alert Management
    acknowledged BOOLEAN DEFAULT false,
    acknowledged_by VARCHAR,
    acknowledged_at TIMESTAMP,

    -- Throttling
    throttle_key VARCHAR,          -- Key for throttling duplicate alerts

    -- Indexing
    INDEX idx_alerts_timestamp (timestamp DESC),
    INDEX idx_alerts_type (alert_type, alert_subtype),
    INDEX idx_alerts_severity (severity),
    INDEX idx_alerts_sent (sent),
    INDEX idx_alerts_throttle (throttle_key)
);

-- ============================================================================
-- ALERT THROTTLING TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS alert_throttle (
    throttle_key VARCHAR PRIMARY KEY,
    last_sent_timestamp TIMESTAMP NOT NULL,
    send_count INTEGER DEFAULT 1,

    INDEX idx_throttle_timestamp (last_sent_timestamp)
);

-- ============================================================================
-- ALERT DELIVERY LOG
-- ============================================================================

CREATE TABLE IF NOT EXISTS alert_delivery_log (
    id INTEGER PRIMARY KEY,
    alert_id INTEGER NOT NULL,
    channel VARCHAR NOT NULL,      -- email, slack, sms, browser
    status VARCHAR NOT NULL,        -- success, failed, throttled
    error_message VARCHAR,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (alert_id) REFERENCES alerts(id),
    INDEX idx_delivery_alert (alert_id),
    INDEX idx_delivery_timestamp (timestamp DESC)
);

-- ============================================================================
-- ALERT STATISTICS VIEW
-- ============================================================================

CREATE OR REPLACE VIEW alert_statistics AS
SELECT
    DATE_TRUNC('day', timestamp) as date,
    alert_type,
    severity,
    COUNT(*) as alert_count,
    SUM(CASE WHEN sent THEN 1 ELSE 0 END) as sent_count,
    SUM(CASE WHEN acknowledged THEN 1 ELSE 0 END) as acknowledged_count
FROM alerts
WHERE timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
GROUP BY DATE_TRUNC('day', timestamp), alert_type, severity
ORDER BY date DESC;

-- ============================================================================
-- RECENT ALERTS VIEW
-- ============================================================================

CREATE OR REPLACE VIEW recent_alerts AS
SELECT
    id,
    alert_type,
    alert_subtype,
    severity,
    message,
    timestamp,
    sent,
    acknowledged
FROM alerts
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 24 HOURS
ORDER BY timestamp DESC;

-- ============================================================================
-- CRITICAL ALERTS VIEW
-- ============================================================================

CREATE OR REPLACE VIEW critical_alerts AS
SELECT
    id,
    alert_type,
    alert_subtype,
    message,
    context,
    timestamp,
    acknowledged,
    acknowledged_by,
    acknowledged_at
FROM alerts
WHERE severity = 'CRITICAL'
  AND timestamp >= CURRENT_TIMESTAMP - INTERVAL 7 DAYS
ORDER BY timestamp DESC;

-- ============================================================================
-- UNACKNOWLEDGED ALERTS VIEW
-- ============================================================================

CREATE OR REPLACE VIEW unacknowledged_alerts AS
SELECT
    id,
    alert_type,
    alert_subtype,
    severity,
    message,
    timestamp
FROM alerts
WHERE acknowledged = false
  AND severity IN ('ERROR', 'CRITICAL')
  AND timestamp >= CURRENT_TIMESTAMP - INTERVAL 48 HOURS
ORDER BY
    CASE severity
        WHEN 'CRITICAL' THEN 1
        WHEN 'ERROR' THEN 2
    END,
    timestamp DESC;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Clean up old alerts (keep 90 days as per config)
CREATE OR REPLACE MACRO cleanup_old_alerts() AS (
    DELETE FROM alerts
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL 90 DAYS
);

-- Get alert count by type in last N hours
CREATE OR REPLACE MACRO get_alert_count(hours INTEGER) AS (
    SELECT
        alert_type,
        severity,
        COUNT(*) as count
    FROM alerts
    WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL (hours) HOURS
    GROUP BY alert_type, severity
    ORDER BY count DESC
);

-- Check if alert should be throttled
CREATE OR REPLACE MACRO should_throttle(key VARCHAR, window_seconds INTEGER) AS (
    SELECT
        CASE
            WHEN last_sent_timestamp >= CURRENT_TIMESTAMP - INTERVAL (window_seconds) SECONDS
            THEN true
            ELSE false
        END as is_throttled
    FROM alert_throttle
    WHERE throttle_key = key
);

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert a test alert to verify schema
INSERT INTO alerts (
    alert_type,
    alert_subtype,
    severity,
    message,
    context,
    source
) VALUES (
    'system',
    'system_startup',
    'INFO',
    'Alert system initialized successfully',
    '{"schema_version": "1.0", "timestamp": "' || CURRENT_TIMESTAMP || '"}',
    'database_schema'
);
