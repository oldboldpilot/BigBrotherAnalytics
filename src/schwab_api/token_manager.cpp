#include "schwab_client.hpp"
#include "../utils/logger.hpp"
#include "../utils/timer.hpp"

#include <curl/curl.h>
#include <mutex>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>

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
          refresh_thread_running_{false} {

        // Initialize CURL
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }

    ~Impl() {
        stopRefreshThread();
        curl_global_cleanup();
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

    [[nodiscard]] auto exchangeAuthCode(std::string const& auth_code) -> Result<void> {
        std::lock_guard lock{mutex_};

        PROFILE_SCOPE("TokenManager::exchangeAuthCode");

        auto* curl = curl_easy_init();
        if (!curl) {
            return makeError<void>(ErrorCode::NetworkError, "Failed to initialize CURL");
        }

        std::string response;

        // Build POST data for auth code exchange
        std::string post_data = "grant_type=authorization_code&code=" + auth_code +
                               "&redirect_uri=" + config_.redirect_uri +
                               "&client_id=" + config_.client_id +
                               "&client_secret=" + config_.client_secret;

        // Set CURL options
        curl_easy_setopt(curl, CURLOPT_URL, TOKEN_URL);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        CURLcode const res = curl_easy_perform(curl);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            return makeError<void>(
                ErrorCode::NetworkError,
                std::string("CURL error: ") + curl_easy_strerror(res)
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

            LOG_INFO("Successfully exchanged auth code for tokens");

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

    OAuth2Config config_;
    mutable std::mutex mutex_;
    std::thread refresh_thread_;
    std::atomic<bool> refresh_thread_running_;
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
