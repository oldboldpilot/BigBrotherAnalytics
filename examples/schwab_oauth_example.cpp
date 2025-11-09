/**
 * Schwab API OAuth 2.0 Authentication Example
 *
 * Demonstrates complete OAuth flow with PKCE and DuckDB persistence.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#include <iostream>
#include <string>

// Import Schwab API module
import bigbrother.schwab_api;
import bigbrother.utils.logger;

using namespace bigbrother::schwab;

/**
 * Example 1: Initial OAuth Authentication
 *
 * First-time authentication flow with PKCE.
 */
auto example_initial_authentication() -> void {
    std::cout << "\n=== Example 1: Initial OAuth Authentication ===\n" << std::endl;

    // 1. Create OAuth configuration
    OAuth2Config config;
    config.client_id = "YOUR_SCHWAB_CLIENT_ID";          // Get from Schwab Developer Portal
    config.client_secret = "YOUR_SCHWAB_CLIENT_SECRET";  // Get from Schwab Developer Portal
    config.redirect_uri = "https://localhost:8080/callback";

    // Validate configuration
    auto validation = config.validate();
    if (!validation) {
        std::cerr << "Configuration error: " << validation.error().message << std::endl;
        return;
    }

    // 2. Create TokenManager
    TokenManager token_mgr(std::move(config));

    // 3. Get authorization URL (generates PKCE challenge)
    auto auth_url_result = token_mgr.getAuthorizationUrl();
    if (!auth_url_result) {
        std::cerr << "Failed to generate auth URL: " << auth_url_result.error().message << std::endl;
        return;
    }

    std::cout << "Step 1: Open this URL in your browser:\n" << std::endl;
    std::cout << *auth_url_result << std::endl;
    std::cout << "\nStep 2: After authorizing, you'll be redirected to:" << std::endl;
    std::cout << "https://localhost:8080/callback?code=YOUR_AUTH_CODE" << std::endl;
    std::cout << "\nStep 3: Enter the authorization code: ";

    // 4. Get authorization code from user
    std::string auth_code;
    std::cin >> auth_code;

    // 5. Exchange authorization code for tokens
    std::cout << "\nExchanging authorization code for tokens..." << std::endl;

    auto exchange_result = token_mgr.exchangeAuthCode(auth_code);
    if (!exchange_result) {
        std::cerr << "Token exchange failed: " << exchange_result.error().message << std::endl;
        return;
    }

    std::cout << "✓ Successfully obtained access token and refresh token" << std::endl;
    std::cout << "✓ Tokens saved to DuckDB" << std::endl;
    std::cout << "✓ Automatic refresh enabled (refreshes at 25 minutes)" << std::endl;

    // 6. Get access token (now available)
    auto token_result = token_mgr.getAccessToken();
    if (token_result) {
        std::cout << "\nAccess Token (first 20 chars): "
                  << token_result->substr(0, 20) << "..." << std::endl;
    }
}

/**
 * Example 2: Load Existing Tokens from DuckDB
 *
 * Subsequent sessions - load previously saved tokens.
 */
auto example_load_tokens_from_database() -> void {
    std::cout << "\n=== Example 2: Load Tokens from Database ===\n" << std::endl;

    // 1. Create OAuth configuration (client_id required for lookup)
    OAuth2Config config;
    config.client_id = "YOUR_SCHWAB_CLIENT_ID";
    config.client_secret = "YOUR_SCHWAB_CLIENT_SECRET";

    // 2. Create TokenManager
    TokenManager token_mgr(std::move(config));

    // 3. Load tokens from DuckDB
    std::cout << "Loading tokens from DuckDB..." << std::endl;

    auto load_result = token_mgr.loadTokensFromDB();
    if (!load_result) {
        std::cerr << "Failed to load tokens: " << load_result.error().message << std::endl;
        std::cerr << "You may need to re-authenticate." << std::endl;
        return;
    }

    std::cout << "✓ Tokens loaded successfully" << std::endl;

    // 4. Get access token (auto-refreshes if expired)
    auto token_result = token_mgr.getAccessToken();
    if (token_result) {
        std::cout << "✓ Access token is valid" << std::endl;
        std::cout << "  Token (first 20 chars): " << token_result->substr(0, 20) << "..." << std::endl;
    } else {
        std::cerr << "Failed to get access token: " << token_result.error().message << std::endl;
    }
}

/**
 * Example 3: Manual Token Refresh
 *
 * Manually trigger token refresh (normally happens automatically).
 */
auto example_manual_token_refresh() -> void {
    std::cout << "\n=== Example 3: Manual Token Refresh ===\n" << std::endl;

    OAuth2Config config;
    config.client_id = "YOUR_SCHWAB_CLIENT_ID";
    config.client_secret = "YOUR_SCHWAB_CLIENT_SECRET";

    TokenManager token_mgr(std::move(config));

    // Load existing tokens
    auto load_result = token_mgr.loadTokensFromDB();
    if (!load_result) {
        std::cerr << "No tokens found. Run initial authentication first." << std::endl;
        return;
    }

    std::cout << "Manually refreshing access token..." << std::endl;

    auto refresh_result = token_mgr.refreshAccessToken();
    if (!refresh_result) {
        std::cerr << "Refresh failed: " << refresh_result.error().message << std::endl;
        return;
    }

    std::cout << "✓ Token refreshed successfully" << std::endl;
    std::cout << "✓ New token saved to DuckDB" << std::endl;
}

/**
 * Example 4: Save Tokens to JSON File (Backup)
 *
 * Save tokens to JSON file for backup/portability.
 */
auto example_save_tokens_to_file() -> void {
    std::cout << "\n=== Example 4: Save Tokens to JSON File ===\n" << std::endl;

    OAuth2Config config;
    config.client_id = "YOUR_SCHWAB_CLIENT_ID";
    config.client_secret = "YOUR_SCHWAB_CLIENT_SECRET";

    TokenManager token_mgr(std::move(config));

    // Load from database
    auto load_result = token_mgr.loadTokensFromDB();
    if (!load_result) {
        std::cerr << "No tokens to save." << std::endl;
        return;
    }

    // Save to JSON file
    std::string backup_file = "schwab_tokens_backup.json";
    auto save_result = token_mgr.saveTokens(backup_file);

    if (save_result) {
        std::cout << "✓ Tokens saved to: " << backup_file << std::endl;
    } else {
        std::cerr << "Failed to save tokens: " << save_result.error().message << std::endl;
    }
}

/**
 * Example 5: Load Tokens from JSON File
 *
 * Load tokens from JSON backup file.
 */
auto example_load_tokens_from_file() -> void {
    std::cout << "\n=== Example 5: Load Tokens from JSON File ===\n" << std::endl;

    std::string backup_file = "schwab_tokens_backup.json";

    auto config_result = TokenManager::loadTokens(backup_file);
    if (!config_result) {
        std::cerr << "Failed to load tokens from file: "
                  << config_result.error().message << std::endl;
        return;
    }

    std::cout << "✓ Tokens loaded from: " << backup_file << std::endl;
    std::cout << "  Client ID: " << config_result->client_id << std::endl;
    std::cout << "  Redirect URI: " << config_result->redirect_uri << std::endl;

    // Create TokenManager with loaded config
    TokenManager token_mgr(std::move(*config_result));

    // Verify token is valid
    auto token_result = token_mgr.getAccessToken();
    if (token_result) {
        std::cout << "✓ Access token is valid" << std::endl;
    }
}

/**
 * Example 6: Error Handling
 *
 * Demonstrates comprehensive error handling.
 */
auto example_error_handling() -> void {
    std::cout << "\n=== Example 6: Error Handling ===\n" << std::endl;

    OAuth2Config config;
    config.client_id = "INVALID_CLIENT_ID";
    config.client_secret = "INVALID_SECRET";
    config.redirect_uri = "https://localhost:8080/callback";

    TokenManager token_mgr(std::move(config));

    // Try to load tokens (will fail - no tokens for this client)
    auto load_result = token_mgr.loadTokensFromDB();
    if (!load_result) {
        std::cout << "Expected error: " << load_result.error().message << std::endl;

        // Error code
        switch (load_result.error().code) {
            case ErrorCode::DatabaseError:
                std::cout << "  Error type: Database Error" << std::endl;
                std::cout << "  Action: Check DuckDB connection" << std::endl;
                break;
            case ErrorCode::AuthenticationError:
                std::cout << "  Error type: Authentication Error" << std::endl;
                std::cout << "  Action: Re-authenticate" << std::endl;
                break;
            case ErrorCode::NetworkError:
                std::cout << "  Error type: Network Error" << std::endl;
                std::cout << "  Action: Check internet connection" << std::endl;
                break;
            default:
                std::cout << "  Error type: Unknown" << std::endl;
                break;
        }
    }
}

/**
 * Main Entry Point
 */
auto main() -> int {
    // Initialize logger
    Logger::getInstance().setLevel(LogLevel::Info);

    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Schwab API OAuth 2.0 Authentication Examples ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\nSelect example to run:" << std::endl;
    std::cout << "  1. Initial OAuth Authentication (first time)" << std::endl;
    std::cout << "  2. Load Tokens from Database" << std::endl;
    std::cout << "  3. Manual Token Refresh" << std::endl;
    std::cout << "  4. Save Tokens to JSON File" << std::endl;
    std::cout << "  5. Load Tokens from JSON File" << std::endl;
    std::cout << "  6. Error Handling Demo" << std::endl;
    std::cout << "\nEnter choice (1-6): ";

    int choice;
    std::cin >> choice;

    switch (choice) {
        case 1:
            example_initial_authentication();
            break;
        case 2:
            example_load_tokens_from_database();
            break;
        case 3:
            example_manual_token_refresh();
            break;
        case 4:
            example_save_tokens_to_file();
            break;
        case 5:
            example_load_tokens_from_file();
            break;
        case 6:
            example_error_handling();
            break;
        default:
            std::cout << "Invalid choice." << std::endl;
            break;
    }

    std::cout << "\n✓ Example completed" << std::endl;
    return 0;
}
