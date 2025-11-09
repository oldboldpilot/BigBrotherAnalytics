/**
 * Schwab API Token Manager Implementation
 * C++23 module implementation unit
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-08
 */

// Global module fragment
module;

#include <curl/curl.h>
#include <duckdb.hpp>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <atomic>
#include <random>
#include <iomanip>
#include <openssl/sha.h>
#include <openssl/evp.h>

// Module implementation unit declaration
module bigbrother.schwab_api;

import bigbrother.utils.logger;
import bigbrother.utils.timer;

// JSON parsing
#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

namespace bigbrother::schwab {

/**
 * OAuth 2.0 Token Manager Implementation
 *
 * Manages Schwab API authentication tokens with automatic refresh.
 * Tokens have 30-minute lifetime, we refresh proactively at 25 minutes.
 *
 * OAuth 2.0 Flow:
 * 1. Initial authorization code obtained via browser redirect
 * 2. Exchange auth code for access token + refresh token
 * 3. Use access token for API calls (30-min lifetime)
 * 4. Refresh access token using refresh token (7-day lifetime)
 *
 * Reference: https://developer.schwab.com/products/trader-api--individual/details/documentation/Retail%20Trader%20API%20Production
 */

namespace {

// Schwab API endpoints
constexpr char const* AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize";
constexpr char const* TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token";

// DuckDB schema for OAuth tokens
constexpr char const* OAUTH_TABLE_SCHEMA = R"(
    CREATE TABLE IF NOT EXISTS oauth_tokens (
        id INTEGER PRIMARY KEY,
        client_id VARCHAR(100) NOT NULL,
        access_token TEXT,
        refresh_token TEXT,
        token_type VARCHAR(20),
        expires_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        code_verifier VARCHAR(128),
        code_challenge VARCHAR(64)
    );
)";

// CURL callback for writing response data
auto curlWriteCallback(
    void* contents,
    size_t size,
    size_t nmemb,
    std::string* output
) -> size_t {
    size_t const total_size = size * nmemb;
    output->append(static_cast<char*>(contents), total_size);
    return total_size;
}

// URL encoding helper
auto urlEncode(std::string const& value) -> std::string {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (char c : value) {
        if (std::isalnum(static_cast<unsigned char>(c)) ||
            c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
        } else {
            escaped << std::uppercase;
            escaped << '%' << std::setw(2) << int(static_cast<unsigned char>(c));
            escaped << std::nouppercase;
        }
    }

    return escaped.str();
}

// Base64 URL-safe encoding (for PKCE)
auto base64UrlEncode(std::vector<unsigned char> const& data) -> std::string {
    static constexpr char const* base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    std::string encoded;
    int i = 0;
    int j = 0;
    unsigned char array_3[3];
    unsigned char array_4[4];

    for (unsigned char c : data) {
        array_3[i++] = c;
        if (i == 3) {
            array_4[0] = (array_3[0] & 0xfc) >> 2;
            array_4[1] = ((array_3[0] & 0x03) << 4) + ((array_3[1] & 0xf0) >> 4);
            array_4[2] = ((array_3[1] & 0x0f) << 2) + ((array_3[2] & 0xc0) >> 6);
            array_4[3] = array_3[2] & 0x3f;

            for (i = 0; i < 4; i++) {
                encoded += base64_chars[array_4[i]];
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++) {
            array_3[j] = '\0';
        }

        array_4[0] = (array_3[0] & 0xfc) >> 2;
        array_4[1] = ((array_3[0] & 0x03) << 4) + ((array_3[1] & 0xf0) >> 4);
        array_4[2] = ((array_3[1] & 0x0f) << 2) + ((array_3[2] & 0xc0) >> 6);

        for (j = 0; j < i + 1; j++) {
            encoded += base64_chars[array_4[j]];
        }
    }

    // Remove padding (URL-safe base64)
    while (!encoded.empty() && encoded.back() == '=') {
        encoded.pop_back();
    }

    return encoded;
}

// Generate cryptographically secure random string
auto generateRandomString(size_t length) -> std::string {
    static constexpr char const* charset =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 65);  // charset length - 1

    std::string result;
    result.reserve(length);

    for (size_t i = 0; i < length; ++i) {
        result += charset[dis(gen)];
    }

    return result;
}

// Generate PKCE code verifier (43-128 characters)
auto generateCodeVerifier() -> std::string {
    return generateRandomString(128);
}

// Generate PKCE code challenge (SHA256 hash of verifier, base64 URL-encoded)
auto generateCodeChallenge(std::string const& verifier) -> std::string {
    unsigned char hash[SHA256_DIGEST_LENGTH];

    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, verifier.c_str(), verifier.length());
    SHA256_Final(hash, &sha256);

    std::vector<unsigned char> hash_vec(hash, hash + SHA256_DIGEST_LENGTH);
    return base64UrlEncode(hash_vec);
}

} // anonymous namespace

[[nodiscard]] auto OAuth2Config::validate() const noexcept -> Result<void> {
    if (client_id.empty()) {
        return makeError<void>(
            ErrorCode::InvalidParameter,
            "Client ID is required"
        );
    }

    if (client_secret.empty()) {
        return makeError<void>(
            ErrorCode::InvalidParameter,
            "Client secret is required"
        );
    }

    if (redirect_uri.empty()) {
        return makeError<void>(
            ErrorCode::InvalidParameter,
            "Redirect URI is required"
        );
    }

    return {};
}

class TokenManager::Impl {
public:
    explicit Impl(OAuth2Config config)
        : config_{std::move(config)},
          refresh_thread_running_{false},
          db_path_{"data/bigbrother.duckdb"} {

        // Initialize CURL
        curl_global_init(CURL_GLOBAL_DEFAULT);

        // Initialize DuckDB connection and schema
        initializeDatabase();
    }

    ~Impl() {
        stopRefreshThread();
        curl_global_cleanup();
    }

    auto initializeDatabase() -> void {
        try {
            db_ = std::make_unique<duckdb::DuckDB>(db_path_);
            conn_ = std::make_unique<duckdb::Connection>(*db_);

            // Create OAuth tokens table
            auto result = conn_->Query(OAUTH_TABLE_SCHEMA);
            if (result->HasError()) {
                LOG_ERROR("Failed to create oauth_tokens table: {}", result->GetError());
            } else {
                LOG_INFO("DuckDB OAuth schema initialized successfully");
            }
        } catch (std::exception const& e) {
            LOG_ERROR("DuckDB initialization error: {}", e.what());
        }
    }

    [[nodiscard]] auto getAccessToken() -> Result<std::string> {
        std::lock_guard lock{mutex_};

        // Check if token needs refresh
        if (config_.isAccessTokenExpired()) {
            LOG_INFO("Access token expired, refreshing...");

            if (auto result = refreshAccessTokenInternal(); !result) {
                return std::unexpected(result.error());
            }
        }

        if (config_.access_token.empty()) {
            return makeError<std::string>(
                ErrorCode::AuthenticationError,
                "No access token available. Please authenticate first."
            );
        }

        return config_.access_token;
    }

    [[nodiscard]] auto refreshAccessToken() -> Result<void> {
        std::lock_guard lock{mutex_};
        return refreshAccessTokenInternal();
    }

    [[nodiscard]] auto refreshAccessTokenInternal() -> Result<void> {
        PROFILE_SCOPE("TokenManager::refreshAccessToken");

        if (config_.refresh_token.empty()) {
            return makeError<void>(
                ErrorCode::AuthenticationError,
                "No refresh token available"
            );
        }

        // Prepare refresh request
        auto* curl = curl_easy_init();
        if (!curl) {
            return makeError<void>(
                ErrorCode::NetworkError,
                "Failed to initialize CURL"
            );
        }

        std::string response;

        // Build POST data
        std::string post_data = "grant_type=refresh_token&refresh_token=" +
                               config_.refresh_token +
                               "&client_id=" + config_.client_id +
                               "&client_secret=" + config_.client_secret;

        // Set CURL options
        curl_easy_setopt(curl, CURLOPT_URL, TOKEN_URL);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        // Set headers
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Perform request
        CURLcode const res = curl_easy_perform(curl);

        // Clean up
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            return makeError<void>(
                ErrorCode::NetworkError,
                std::string("CURL error: ") + curl_easy_strerror(res)
            );
        }

#ifdef HAS_NLOHMANN_JSON
        // Parse response
        try {
            auto const json_response = json::parse(response);

            // Check for error
            if (json_response.contains("error")) {
                std::string const error = json_response["error"];
                std::string const error_desc = json_response.value("error_description", "");
                return makeError<void>(
                    ErrorCode::AuthenticationError,
                    "OAuth error: " + error + " - " + error_desc
                );
            }

            // Extract new tokens
            config_.access_token = json_response["access_token"];
            config_.refresh_token = json_response.value("refresh_token", config_.refresh_token);

            // Calculate expiry time
            int const expires_in = json_response.value("expires_in", 1800);  // Default 30 min
            config_.token_expiry = std::chrono::system_clock::now() +
                                  std::chrono::seconds(expires_in);

            LOG_INFO("Access token refreshed successfully (expires in {} seconds)", expires_in);

            // Save refreshed tokens to DuckDB
            auto save_result = saveTokensToDB();
            if (!save_result) {
                LOG_WARN("Failed to save refreshed tokens to database: {}", save_result.error().message);
            }

            return {};

        } catch (json::exception const& e) {
            return makeError<void>(
                ErrorCode::UnknownError,
                std::string("JSON parse error: ") + e.what()
            );
        }
#else
        LOG_ERROR("JSON library not available, cannot parse token response");
        return makeError<void>(
            ErrorCode::UnknownError,
            "JSON library not available"
        );
#endif
    }

    [[nodiscard]] auto getAuthorizationUrl() -> Result<std::string> {
        std::lock_guard lock{mutex_};

        PROFILE_SCOPE("TokenManager::getAuthorizationUrl");

        // Generate PKCE parameters
        code_verifier_ = generateCodeVerifier();
        code_challenge_ = generateCodeChallenge(code_verifier_);

        LOG_INFO("Generated PKCE code verifier (length: {})", code_verifier_.length());
        LOG_INFO("Generated PKCE code challenge: {}", code_challenge_);

        // Build authorization URL with PKCE
        std::string auth_url = std::string(AUTH_URL) +
                              "?client_id=" + urlEncode(config_.client_id) +
                              "&redirect_uri=" + urlEncode(config_.redirect_uri) +
                              "&response_type=code" +
                              "&scope=api" +
                              "&code_challenge=" + code_challenge_ +
                              "&code_challenge_method=S256";

        LOG_INFO("Authorization URL generated (with PKCE S256)");

        return auth_url;
    }

    [[nodiscard]] auto exchangeAuthCode(std::string const& auth_code) -> Result<void> {
        std::lock_guard lock{mutex_};

        PROFILE_SCOPE("TokenManager::exchangeAuthCode");

        if (code_verifier_.empty()) {
            return makeError<void>(
                ErrorCode::AuthenticationError,
                "PKCE code verifier not found. Call getAuthorizationUrl() first."
            );
        }

        auto* curl = curl_easy_init();
        if (!curl) {
            return makeError<void>(ErrorCode::NetworkError, "Failed to initialize CURL");
        }

        std::string response;

        // Build POST data for auth code exchange (with PKCE verifier)
        std::string post_data = "grant_type=authorization_code&code=" + urlEncode(auth_code) +
                               "&redirect_uri=" + urlEncode(config_.redirect_uri) +
                               "&client_id=" + urlEncode(config_.client_id) +
                               "&client_secret=" + urlEncode(config_.client_secret) +
                               "&code_verifier=" + code_verifier_;

        // Set CURL options
        curl_easy_setopt(curl, CURLOPT_URL, TOKEN_URL);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Enable verbose logging for debugging
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);

        CURLcode const res = curl_easy_perform(curl);

        // Get HTTP response code
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            return makeError<void>(
                ErrorCode::NetworkError,
                std::string("CURL error: ") + curl_easy_strerror(res)
            );
        }

        if (http_code != 200) {
            return makeError<void>(
                ErrorCode::AuthenticationError,
                "HTTP error " + std::to_string(http_code) + ": " + response
            );
        }

#ifdef HAS_NLOHMANN_JSON
        try {
            auto const json_response = json::parse(response);

            if (json_response.contains("error")) {
                return makeError<void>(
                    ErrorCode::AuthenticationError,
                    "OAuth error: " + json_response["error"].get<std::string>()
                );
            }

            config_.access_token = json_response["access_token"];
            config_.refresh_token = json_response["refresh_token"];

            int const expires_in = json_response.value("expires_in", 1800);
            config_.token_expiry = std::chrono::system_clock::now() +
                                  std::chrono::seconds(expires_in);

            LOG_INFO("Successfully exchanged auth code for tokens (expires in {} seconds)", expires_in);

            // Save tokens to DuckDB
            auto save_result = saveTokensToDB();
            if (!save_result) {
                LOG_WARN("Failed to save tokens to database: {}", save_result.error().message);
            }

            // Start automatic refresh thread
            startRefreshThread();

            return {};

        } catch (json::exception const& e) {
            return makeError<void>(
                ErrorCode::UnknownError,
                std::string("JSON parse error: ") + e.what()
            );
        }
#else
        return makeError<void>(ErrorCode::UnknownError, "JSON library not available");
#endif
    }

    auto startRefreshThread() -> void {
        if (refresh_thread_running_.exchange(true)) {
            return;  // Already running
        }

        refresh_thread_ = std::thread([this]() {
            LOG_INFO("Token refresh thread started");

            while (refresh_thread_running_) {
                // Sleep for 25 minutes (refresh before 30-min expiry)
                std::this_thread::sleep_for(std::chrono::minutes(25));

                if (!refresh_thread_running_) {
                    break;
                }

                LOG_INFO("Proactive token refresh (25-minute interval)");

                if (auto result = refreshAccessToken(); !result) {
                    LOG_ERROR("Failed to refresh token: {}", result.error().message);
                }
            }

            LOG_INFO("Token refresh thread stopped");
        });
    }

    auto stopRefreshThread() -> void {
        if (refresh_thread_running_.exchange(false)) {
            if (refresh_thread_.joinable()) {
                refresh_thread_.join();
            }
        }
    }

    [[nodiscard]] auto saveTokens(std::string const& file_path) const -> Result<void> {
        std::lock_guard lock{mutex_};

#ifdef HAS_NLOHMANN_JSON
        json j;
        j["client_id"] = config_.client_id;
        j["client_secret"] = config_.client_secret;
        j["redirect_uri"] = config_.redirect_uri;
        j["access_token"] = config_.access_token;
        j["refresh_token"] = config_.refresh_token;

        auto const expiry_time = std::chrono::system_clock::to_time_t(config_.token_expiry);
        j["token_expiry"] = expiry_time;

        std::ofstream file{file_path};
        if (!file.is_open()) {
            return makeError<void>(
                ErrorCode::DatabaseError,
                "Failed to open file for writing: " + file_path
            );
        }

        file << j.dump(4);

        LOG_INFO("Tokens saved to: {}", file_path);
        return {};
#else
        return makeError<void>(ErrorCode::UnknownError, "JSON library not available");
#endif
    }

    [[nodiscard]] auto saveTokensToDB() -> Result<void> {
        std::lock_guard lock{mutex_};

        if (!conn_) {
            return makeError<void>(
                ErrorCode::DatabaseError,
                "DuckDB connection not initialized"
            );
        }

        try {
            // Convert expiry time to timestamp string
            auto expiry_time = std::chrono::system_clock::to_time_t(config_.token_expiry);
            std::ostringstream expiry_oss;
            expiry_oss << std::put_time(std::gmtime(&expiry_time), "%Y-%m-%d %H:%M:%S");

            // Delete existing tokens for this client_id
            std::string delete_query = "DELETE FROM oauth_tokens WHERE client_id = '" +
                                      config_.client_id + "'";
            auto delete_result = conn_->Query(delete_query);
            if (delete_result->HasError()) {
                LOG_WARN("Failed to delete old tokens: {}", delete_result->GetError());
            }

            // Insert new tokens
            std::ostringstream insert_query;
            insert_query << "INSERT INTO oauth_tokens ("
                        << "client_id, access_token, refresh_token, token_type, "
                        << "expires_at, code_verifier, code_challenge) VALUES ("
                        << "'" << config_.client_id << "', "
                        << "'" << config_.access_token << "', "
                        << "'" << config_.refresh_token << "', "
                        << "'Bearer', "
                        << "TIMESTAMP '" << expiry_oss.str() << "', "
                        << "'" << code_verifier_ << "', "
                        << "'" << code_challenge_ << "')";

            auto insert_result = conn_->Query(insert_query.str());
            if (insert_result->HasError()) {
                return makeError<void>(
                    ErrorCode::DatabaseError,
                    "Failed to save tokens to DuckDB: " + insert_result->GetError()
                );
            }

            LOG_INFO("Tokens saved to DuckDB successfully");
            return {};

        } catch (std::exception const& e) {
            return makeError<void>(
                ErrorCode::DatabaseError,
                std::string("DuckDB save error: ") + e.what()
            );
        }
    }

    [[nodiscard]] auto loadTokensFromDB() -> Result<void> {
        std::lock_guard lock{mutex_};

        if (!conn_) {
            return makeError<void>(
                ErrorCode::DatabaseError,
                "DuckDB connection not initialized"
            );
        }

        try {
            std::string query = "SELECT access_token, refresh_token, expires_at, "
                              "code_verifier, code_challenge FROM oauth_tokens "
                              "WHERE client_id = '" + config_.client_id + "' "
                              "ORDER BY created_at DESC LIMIT 1";

            auto result = conn_->Query(query);

            if (result->HasError()) {
                return makeError<void>(
                    ErrorCode::DatabaseError,
                    "Failed to load tokens: " + result->GetError()
                );
            }

            if (result->RowCount() == 0) {
                return makeError<void>(
                    ErrorCode::DatabaseError,
                    "No tokens found for client_id: " + config_.client_id
                );
            }

            // Extract tokens from first row
            auto chunk = result->Fetch();
            if (chunk && chunk->size() > 0) {
                config_.access_token = chunk->GetValue(0, 0).ToString();
                config_.refresh_token = chunk->GetValue(1, 0).ToString();

                // Parse timestamp
                std::string expires_at_str = chunk->GetValue(2, 0).ToString();
                std::tm tm = {};
                std::istringstream ss(expires_at_str);
                ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

                if (!ss.fail()) {
                    config_.token_expiry = std::chrono::system_clock::from_time_t(std::mktime(&tm));
                }

                code_verifier_ = chunk->GetValue(3, 0).ToString();
                code_challenge_ = chunk->GetValue(4, 0).ToString();

                LOG_INFO("Tokens loaded from DuckDB successfully");

                // Check if token needs refresh
                if (config_.isAccessTokenExpired()) {
                    LOG_INFO("Loaded token is expired, will refresh on next access");
                }

                return {};
            }

            return makeError<void>(
                ErrorCode::DatabaseError,
                "Failed to parse token data"
            );

        } catch (std::exception const& e) {
            return makeError<void>(
                ErrorCode::DatabaseError,
                std::string("DuckDB load error: ") + e.what()
            );
        }
    }

    OAuth2Config config_;
    mutable std::mutex mutex_;
    std::thread refresh_thread_;
    std::atomic<bool> refresh_thread_running_;

    // DuckDB connection
    std::string db_path_;
    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;

    // PKCE parameters (stored for code exchange)
    std::string code_verifier_;
    std::string code_challenge_;
};

// TokenManager public interface
TokenManager::TokenManager(OAuth2Config config)
    : pImpl_{std::make_unique<Impl>(std::move(config))} {}

TokenManager::~TokenManager() = default;

TokenManager::TokenManager(TokenManager&&) noexcept = default;
auto TokenManager::operator=(TokenManager&&) noexcept -> TokenManager& = default;

[[nodiscard]] auto TokenManager::getAccessToken() -> Result<std::string> {
    return pImpl_->getAccessToken();
}

[[nodiscard]] auto TokenManager::refreshAccessToken() -> Result<void> {
    return pImpl_->refreshAccessToken();
}

[[nodiscard]] auto TokenManager::exchangeAuthCode(std::string const& auth_code)
    -> Result<void> {
    return pImpl_->exchangeAuthCode(auth_code);
}

[[nodiscard]] auto TokenManager::saveTokens(std::string const& file_path) const
    -> Result<void> {
    return pImpl_->saveTokens(file_path);
}

[[nodiscard]] auto TokenManager::loadTokens(std::string const& file_path)
    -> Result<OAuth2Config> {

#ifdef HAS_NLOHMANN_JSON
    std::ifstream file{file_path};
    if (!file.is_open()) {
        return makeError<OAuth2Config>(
            ErrorCode::DatabaseError,
            "Failed to open tokens file: " + file_path
        );
    }

    try {
        json j;
        file >> j;

        OAuth2Config config;
        config.client_id = j["client_id"];
        config.client_secret = j["client_secret"];
        config.redirect_uri = j["redirect_uri"];
        config.access_token = j["access_token"];
        config.refresh_token = j["refresh_token"];

        auto const expiry_time = j["token_expiry"].get<std::time_t>();
        config.token_expiry = std::chrono::system_clock::from_time_t(expiry_time);

        LOG_INFO("Tokens loaded from: {}", file_path);
        return config;

    } catch (json::exception const& e) {
        return makeError<OAuth2Config>(
            ErrorCode::UnknownError,
            std::string("JSON parse error: ") + e.what()
        );
    }
#else
    return makeError<OAuth2Config>(ErrorCode::UnknownError, "JSON library not available");
#endif
}

[[nodiscard]] auto TokenManager::getAuthorizationUrl() -> Result<std::string> {
    return pImpl_->getAuthorizationUrl();
}

[[nodiscard]] auto TokenManager::saveTokensToDB() -> Result<void> {
    return pImpl_->saveTokensToDB();
}

[[nodiscard]] auto TokenManager::loadTokensFromDB() -> Result<void> {
    return pImpl_->loadTokensFromDB();
}

[[nodiscard]] auto TokenManager::initialOAuthFlow(
    std::string const& client_id,
    std::string const& redirect_uri
) -> Result<OAuth2Config> {

    // Generate authorization URL
    std::string auth_url = std::string(AUTH_URL) +
                          "?client_id=" + client_id +
                          "&redirect_uri=" + redirect_uri +
                          "&response_type=code" +
                          "&scope=api";

    LOG_INFO("=================================================================");
    LOG_INFO("SCHWAB API INITIAL AUTHENTICATION");
    LOG_INFO("=================================================================");
    LOG_INFO("");
    LOG_INFO("Please open the following URL in your browser:");
    LOG_INFO("");
    LOG_INFO("{}", auth_url);
    LOG_INFO("");
    LOG_INFO("After authorizing, you will be redirected to:");
    LOG_INFO("{}?code=YOUR_AUTH_CODE", redirect_uri);
    LOG_INFO("");
    LOG_INFO("Copy the authorization code from the URL and paste it here:");
    LOG_INFO("=================================================================");

    // Read auth code from stdin
    std::string auth_code;
    std::cin >> auth_code;

    OAuth2Config config;
    config.client_id = client_id;
    config.redirect_uri = redirect_uri;
    config.auth_code = auth_code;

    return config;
}

} // namespace bigbrother::schwab
