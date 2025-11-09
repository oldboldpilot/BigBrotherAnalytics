-- BigBrotherAnalytics: OAuth 2.0 Token Storage Schema
-- Schwab API Authentication Token Management
--
-- Author: Olumuyiwa Oluwasanmi
-- Created: 2025-11-09
-- Database: DuckDB
--
-- Purpose:
-- - Store OAuth 2.0 access tokens and refresh tokens
-- - Support PKCE (Proof Key for Code Exchange)
-- - Track token expiry for automatic refresh
-- - Support multiple client IDs (multi-account)

-- =============================================================================
-- OAUTH TOKENS TABLE
-- =============================================================================

-- OAuth 2.0 Token Storage
CREATE TABLE IF NOT EXISTS oauth_tokens (
    id INTEGER PRIMARY KEY,
    client_id VARCHAR(100) NOT NULL,         -- Schwab API Client ID (App Key)
    access_token TEXT,                        -- Access token (30-min lifetime)
    refresh_token TEXT,                       -- Refresh token (7-day lifetime)
    token_type VARCHAR(20),                   -- "Bearer"
    expires_at TIMESTAMP,                     -- Token expiry timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    code_verifier VARCHAR(128),               -- PKCE code verifier (128 chars)
    code_challenge VARCHAR(64)                -- PKCE code challenge (43 chars)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Index for fast client_id lookup
CREATE INDEX IF NOT EXISTS idx_oauth_tokens_client_id ON oauth_tokens(client_id);

-- Index for finding expired tokens
CREATE INDEX IF NOT EXISTS idx_oauth_tokens_expires_at ON oauth_tokens(expires_at);

-- Index for latest token per client
CREATE INDEX IF NOT EXISTS idx_oauth_tokens_created_at ON oauth_tokens(client_id, created_at DESC);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Latest active token per client
CREATE OR REPLACE VIEW latest_oauth_tokens AS
SELECT
    client_id,
    access_token,
    refresh_token,
    token_type,
    expires_at,
    created_at,
    CASE
        WHEN expires_at > CURRENT_TIMESTAMP THEN 'VALID'
        ELSE 'EXPIRED'
    END as token_status,
    EXTRACT(EPOCH FROM (expires_at - CURRENT_TIMESTAMP)) / 60 as minutes_until_expiry
FROM oauth_tokens
WHERE created_at IN (
    SELECT MAX(created_at)
    FROM oauth_tokens
    GROUP BY client_id
)
ORDER BY client_id;

-- Expired tokens (candidates for cleanup)
CREATE OR REPLACE VIEW expired_oauth_tokens AS
SELECT
    id,
    client_id,
    expires_at,
    created_at,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - expires_at)) / 3600 as hours_since_expiry
FROM oauth_tokens
WHERE expires_at < CURRENT_TIMESTAMP
ORDER BY expires_at;

-- Token refresh history (last 30 days)
CREATE OR REPLACE VIEW oauth_token_refresh_history AS
SELECT
    client_id,
    COUNT(*) as refresh_count,
    MIN(created_at) as first_refresh,
    MAX(created_at) as last_refresh,
    AVG(EXTRACT(EPOCH FROM (expires_at - created_at))) / 60 as avg_token_lifetime_minutes
FROM oauth_tokens
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY client_id
ORDER BY last_refresh DESC;

-- =============================================================================
-- CLEANUP PROCEDURES
-- =============================================================================

-- Note: Execute these queries manually for cleanup

-- Delete tokens older than 30 days
-- DELETE FROM oauth_tokens
-- WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days';

-- Delete all tokens for a specific client (for re-authentication)
-- DELETE FROM oauth_tokens
-- WHERE client_id = 'YOUR_CLIENT_ID';

-- Keep only the latest token per client
-- DELETE FROM oauth_tokens
-- WHERE id NOT IN (
--     SELECT MAX(id)
--     FROM oauth_tokens
--     GROUP BY client_id
-- );

-- =============================================================================
-- SAMPLE QUERIES
-- =============================================================================

-- Query 1: Get latest token for a client
-- SELECT access_token, refresh_token, expires_at
-- FROM oauth_tokens
-- WHERE client_id = 'YOUR_CLIENT_ID'
-- ORDER BY created_at DESC
-- LIMIT 1;

-- Query 2: Check if token needs refresh (< 5 minutes until expiry)
-- SELECT
--     client_id,
--     EXTRACT(EPOCH FROM (expires_at - CURRENT_TIMESTAMP)) / 60 as minutes_remaining,
--     CASE
--         WHEN expires_at < CURRENT_TIMESTAMP THEN 'EXPIRED'
--         WHEN expires_at < CURRENT_TIMESTAMP + INTERVAL '5 minutes' THEN 'NEEDS_REFRESH'
--         ELSE 'VALID'
--     END as status
-- FROM oauth_tokens
-- WHERE client_id = 'YOUR_CLIENT_ID'
-- ORDER BY created_at DESC
-- LIMIT 1;

-- Query 3: Count tokens by status
-- SELECT
--     CASE
--         WHEN expires_at > CURRENT_TIMESTAMP THEN 'VALID'
--         ELSE 'EXPIRED'
--     END as status,
--     COUNT(*) as count
-- FROM oauth_tokens
-- GROUP BY status;

-- Query 4: Token refresh rate (tokens per day, last 7 days)
-- SELECT
--     DATE_TRUNC('day', created_at) as date,
--     COUNT(*) as tokens_created
-- FROM oauth_tokens
-- WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
-- GROUP BY DATE_TRUNC('day', created_at)
-- ORDER BY date DESC;

-- =============================================================================
-- NOTES
-- =============================================================================

/*
Token Lifecycle:
1. Initial Authorization:
   - User authorizes via browser
   - Authorization code exchanged for tokens
   - Both access_token and refresh_token stored

2. Token Usage:
   - Access token used for API requests
   - Lifetime: 30 minutes
   - Automatically refreshed at 25 minutes

3. Token Refresh:
   - Refresh token used to get new access token
   - Refresh token lifetime: 7 days
   - Old tokens can be deleted after successful refresh

4. Token Expiry:
   - After 7 days, refresh token expires
   - User must re-authorize (full OAuth flow)

Security Considerations:
- Tokens are sensitive credentials
- Consider encrypting tokens at rest
- Limit database access to authorized applications only
- Regularly audit token access logs
- Implement token revocation on logout

PKCE (Proof Key for Code Exchange):
- code_verifier: Random string (43-128 chars)
- code_challenge: SHA256(code_verifier) -> base64url
- Used to prevent authorization code interception attacks
*/
