/**
 * BigBrotherAnalytics - Token Receiver Module (C++23)
 *
 * Socket-based OAuth token refresh receiver for automated trading systems.
 * Listens on Unix domain socket or TCP socket for token updates from external
 * authentication service.
 *
 * Features:
 * - Thread-safe token storage with atomic operations
 * - Callback mechanism for token update notifications
 * - Graceful shutdown with std::jthread
 * - Unix domain socket (preferred) or TCP fallback
 * - Automatic socket cleanup and error recovery
 *
 * Following C++ Core Guidelines:
 * - R.1: RAII for socket management
 * - C.21: Rule of five for proper resource management
 * - F.6: noexcept where applicable
 * - F.20: Return values, not output parameters
 * - CP.2: Avoid data races with proper synchronization
 * - CP.8: Use std::jthread for automatic cleanup
 *
 * Performance: <10μs token update latency
 * Thread-Safety: Full thread-safe via std::mutex + atomic
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 13, 2025
 */

// ============================================================================
// 1. GLOBAL MODULE FRAGMENT (Platform Headers Only)
// ============================================================================
module;

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

// ============================================================================
// 2. MODULE DECLARATION
// ============================================================================
export module bigbrother.utils.token_receiver;

// Import logger for diagnostics
import bigbrother.utils.logger;

// ============================================================================
// 3. EXPORTED INTERFACE (Public API)
// ============================================================================
export namespace bigbrother::utils {

/**
 * Socket Configuration
 * C.1: Struct for passive configuration data
 */
struct TokenReceiverConfig {
    // Unix domain socket path (preferred for security and performance)
    std::string unix_socket_path{"/tmp/bigbrother_token.sock"};

    // TCP fallback configuration
    std::string tcp_host{"127.0.0.1"};
    uint16_t tcp_port{9999};

    // Socket behavior
    bool prefer_unix_socket{true}; // Try Unix socket first, fallback to TCP
    uint32_t max_token_size{8192}; // Maximum token size in bytes
    uint32_t listen_backlog{5};    // Socket listen backlog

    // Timeouts
    std::chrono::milliseconds accept_timeout{500};   // Accept timeout for graceful shutdown
    std::chrono::milliseconds receive_timeout{5000}; // Token receive timeout
};

/**
 * Token Receiver - Socket-based OAuth token refresh receiver
 *
 * Modern C++23 implementation with:
 * - std::jthread for automatic thread cleanup
 * - std::stop_token for graceful shutdown
 * - Callback mechanism for token updates
 * - Thread-safe token storage
 * - RAII socket management
 *
 * Usage:
 *   auto receiver = TokenReceiver([](std::string const& token) {
 *       std::cout << "New token received: " << token << std::endl;
 *   });
 *   receiver.start();
 *   // ... receiver runs in background ...
 *   std::string current_token = receiver.getLatestToken();
 *   receiver.stop();  // Automatic on destruction
 *
 * Design Philosophy:
 * - Callback-based: Client provides callback for token updates
 * - Non-blocking: Receiver runs on background thread
 * - Fail-safe: Automatic cleanup, no resource leaks
 * - Observable: Exposes running state and error diagnostics
 */
class TokenReceiver {
  public:
    /**
     * Token update callback type
     * Called on receiver thread when new token arrives
     * F.23: Use std::function for callbacks
     */
    using TokenCallback = std::function<void(std::string const&)>;

    /**
     * Constructor with callback
     * C.41: Constructor establishes invariants
     *
     * @param callback Function to call when new token received
     * @param config Socket configuration (optional)
     */
    explicit TokenReceiver(TokenCallback callback,
                           TokenReceiverConfig config = TokenReceiverConfig{});

    /**
     * Destructor - ensures graceful shutdown
     * R.1: RAII - automatically stops receiver thread
     */
    ~TokenReceiver();

    // C.21: Delete copy operations (socket ownership is unique)
    TokenReceiver(TokenReceiver const&) = delete;
    auto operator=(TokenReceiver const&) -> TokenReceiver& = delete;

    // Move operations allowed for unique ownership transfer
    TokenReceiver(TokenReceiver&&) noexcept = default;
    auto operator=(TokenReceiver&&) noexcept -> TokenReceiver& = default;

    /**
     * Start the token receiver thread
     *
     * Creates background thread that listens for incoming tokens.
     * Thread automatically stops on destruction or explicit stop().
     *
     * @return true if started successfully, false if already running
     * F.6: noexcept - cannot throw
     */
    [[nodiscard]] auto start() noexcept -> bool;

    /**
     * Stop the token receiver thread
     *
     * Requests thread shutdown via stop_token and waits for completion.
     * Safe to call multiple times. Automatically called by destructor.
     *
     * F.6: noexcept - guaranteed not to throw
     */
    auto stop() noexcept -> void;

    /**
     * Get the most recently received token
     *
     * Thread-safe access to latest token value.
     * Returns empty string if no token received yet.
     *
     * @return Copy of latest token (thread-safe)
     * F.20: Return by value (move semantics)
     */
    [[nodiscard]] auto getLatestToken() const -> std::string;

    /**
     * Check if receiver is currently running
     *
     * @return true if receiver thread is active
     * F.6: noexcept - cannot throw
     */
    [[nodiscard]] auto isRunning() const noexcept -> bool;

    /**
     * Get count of tokens received since start
     *
     * @return Total number of tokens received
     * F.6: noexcept - atomic read cannot throw
     */
    [[nodiscard]] auto getTokenCount() const noexcept -> uint64_t;

    /**
     * Get last error message (if any)
     *
     * @return Error message or empty string if no errors
     * F.20: Return by value (move semantics)
     */
    [[nodiscard]] auto getLastError() const -> std::string;

  private:
    // Forward declaration for pImpl
    class Impl;

    /**
     * Receiver thread main loop
     * CP.8: std::jthread with stop_token for cooperative cancellation
     *
     * @param stop_token Token for graceful shutdown signaling
     */
    auto receiverThreadMain(std::stop_token stop_token) -> void;

    /**
     * Create and bind Unix domain socket
     * R.1: Returns socket FD or -1 on failure (RAII via unique_ptr wrapper)
     */
    [[nodiscard]] auto createUnixSocket() -> int;

    /**
     * Create and bind TCP socket
     * R.1: Returns socket FD or -1 on failure (RAII via unique_ptr wrapper)
     */
    [[nodiscard]] auto createTcpSocket() -> int;

    /**
     * Accept incoming connection with timeout
     *
     * @param server_fd Server socket file descriptor
     * @param stop_token Stop token for cancellation
     * @return Client socket FD or -1 on error/timeout
     */
    [[nodiscard]] auto acceptConnection(int server_fd, std::stop_token const& stop_token) -> int;

    /**
     * Receive token from client connection
     *
     * @param client_fd Client socket file descriptor
     * @return Token string or empty on error
     * F.20: Return by value (move semantics)
     */
    [[nodiscard]] auto receiveToken(int client_fd) -> std::string;

    /**
     * Clean up socket resources
     * F.6: noexcept - cleanup must not throw
     */
    auto cleanupSocket() noexcept -> void;

    /**
     * Set last error message
     * CP.2: Thread-safe error recording
     */
    auto setError(std::string_view message) -> void;

    // Configuration
    TokenReceiverConfig config_;

    // Token callback
    TokenCallback callback_;

    // Thread-safe token storage
    mutable std::mutex token_mutex_;
    std::string latest_token_;

    // Statistics (atomic for lock-free access)
    std::atomic<uint64_t> token_count_{0};
    std::atomic<bool> running_{false};

    // Error tracking
    mutable std::mutex error_mutex_;
    std::string last_error_;

    // Receiver thread (std::jthread for automatic cleanup)
    std::unique_ptr<std::jthread> receiver_thread_;

    // Socket file descriptor (owned by this class)
    int server_fd_{-1};

    // Logger reference
    Logger& logger_;
};

} // namespace bigbrother::utils

// ============================================================================
// 4. PRIVATE IMPLEMENTATION
// ============================================================================
module :private;

namespace bigbrother::utils {

// ============================================================================
// Constructor / Destructor
// ============================================================================

TokenReceiver::TokenReceiver(TokenCallback callback, TokenReceiverConfig config)
    : config_{std::move(config)}, callback_{std::move(callback)}, logger_{Logger::getInstance()} {

    if (!callback_) {
        throw std::invalid_argument("TokenReceiver: callback must not be empty");
    }

    logger_.info("TokenReceiver initialized with config: unix_socket={}, tcp={}:{}",
                 config_.unix_socket_path, config_.tcp_host, config_.tcp_port);
}

TokenReceiver::~TokenReceiver() {
    stop();
}

// ============================================================================
// Public Interface
// ============================================================================

[[nodiscard]] auto TokenReceiver::start() noexcept -> bool {
    // Check if already running
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        logger_.warn("TokenReceiver::start() called but already running");
        return false;
    }

    try {
        // Create socket (try Unix domain first if preferred)
        if (config_.prefer_unix_socket) {
            server_fd_ = createUnixSocket();
            if (server_fd_ < 0) {
                logger_.warn("Unix domain socket creation failed, falling back to TCP");
                server_fd_ = createTcpSocket();
            }
        } else {
            server_fd_ = createTcpSocket();
        }

        if (server_fd_ < 0) {
            setError("Failed to create socket");
            running_ = false;
            return false;
        }

        // Start receiver thread
        receiver_thread_ = std::make_unique<std::jthread>(
            [this](std::stop_token stop_token) { receiverThreadMain(stop_token); });

        logger_.info("TokenReceiver started successfully on socket FD {}", server_fd_);
        return true;

    } catch (std::exception const& e) {
        setError(std::format("Exception in start(): {}", e.what()));
        logger_.error("TokenReceiver::start() exception: {}", e.what());
        running_ = false;
        cleanupSocket();
        return false;
    }
}

auto TokenReceiver::stop() noexcept -> void {
    if (!running_.exchange(false)) {
        return; // Already stopped
    }

    logger_.info("TokenReceiver stopping...");

    try {
        // Request thread stop (std::jthread automatically requests on destruction)
        if (receiver_thread_ && receiver_thread_->joinable()) {
            receiver_thread_->request_stop();
            // Wait for thread to finish (with timeout)
            // Note: std::jthread destructor will join automatically
        }

        // Clean up socket
        cleanupSocket();

        // Reset thread
        receiver_thread_.reset();

        logger_.info("TokenReceiver stopped successfully");

    } catch (std::exception const& e) {
        logger_.error("Exception during TokenReceiver::stop(): {}", e.what());
    }
}

[[nodiscard]] auto TokenReceiver::getLatestToken() const -> std::string {
    std::lock_guard<std::mutex> lock(token_mutex_);
    return latest_token_;
}

[[nodiscard]] auto TokenReceiver::isRunning() const noexcept -> bool {
    return running_.load();
}

[[nodiscard]] auto TokenReceiver::getTokenCount() const noexcept -> uint64_t {
    return token_count_.load();
}

[[nodiscard]] auto TokenReceiver::getLastError() const -> std::string {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

// ============================================================================
// Private Implementation - Thread Main Loop
// ============================================================================

auto TokenReceiver::receiverThreadMain(std::stop_token stop_token) -> void {
    logger_.info("TokenReceiver thread started");

    while (!stop_token.stop_requested()) {
        // Accept incoming connection (with timeout for graceful shutdown)
        int client_fd = acceptConnection(server_fd_, stop_token);

        if (client_fd < 0) {
            if (stop_token.stop_requested()) {
                break; // Graceful shutdown
            }
            // Timeout or error - continue listening
            continue;
        }

        logger_.debug("TokenReceiver accepted connection from client FD {}", client_fd);

        // Receive token from client
        std::string token = receiveToken(client_fd);

        // Close client connection
        ::close(client_fd);

        if (token.empty()) {
            logger_.warn("Received empty token or error, ignoring");
            continue;
        }

        logger_.info("TokenReceiver received new token ({} bytes)", token.size());

        // Update stored token (thread-safe)
        {
            std::lock_guard<std::mutex> lock(token_mutex_);
            latest_token_ = token;
        }

        // Increment counter
        token_count_.fetch_add(1, std::memory_order_relaxed);

        // Invoke callback
        try {
            callback_(token);
            logger_.debug("TokenReceiver callback invoked successfully");
        } catch (std::exception const& e) {
            logger_.error("Exception in token callback: {}", e.what());
            setError(std::format("Callback exception: {}", e.what()));
        }
    }

    logger_.info("TokenReceiver thread exiting");
}

// ============================================================================
// Private Implementation - Socket Operations
// ============================================================================

[[nodiscard]] auto TokenReceiver::createUnixSocket() -> int {
    // Create Unix domain socket
    int sock_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        setError(std::format("socket() failed: {}", std::strerror(errno)));
        logger_.error("Unix socket creation failed: {}", std::strerror(errno));
        return -1;
    }

    // Remove existing socket file if it exists
    ::unlink(config_.unix_socket_path.c_str());

    // Bind to socket path
    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, config_.unix_socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(sock_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        setError(std::format("bind() failed: {}", std::strerror(errno)));
        logger_.error("Unix socket bind failed: {}", std::strerror(errno));
        ::close(sock_fd);
        return -1;
    }

    // Listen for connections
    if (::listen(sock_fd, config_.listen_backlog) < 0) {
        setError(std::format("listen() failed: {}", std::strerror(errno)));
        logger_.error("Unix socket listen failed: {}", std::strerror(errno));
        ::close(sock_fd);
        return -1;
    }

    logger_.info("Unix domain socket created and listening on {}", config_.unix_socket_path);
    return sock_fd;
}

[[nodiscard]] auto TokenReceiver::createTcpSocket() -> int {
    // Create TCP socket
    int sock_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        setError(std::format("socket() failed: {}", std::strerror(errno)));
        logger_.error("TCP socket creation failed: {}", std::strerror(errno));
        return -1;
    }

    // Set SO_REUSEADDR to allow quick restarts
    int opt = 1;
    if (::setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        logger_.warn("setsockopt(SO_REUSEADDR) failed: {}", std::strerror(errno));
    }

    // Bind to TCP address
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config_.tcp_port);

    if (::inet_pton(AF_INET, config_.tcp_host.c_str(), &addr.sin_addr) <= 0) {
        setError(std::format("inet_pton() failed for {}", config_.tcp_host));
        logger_.error("Invalid TCP address: {}", config_.tcp_host);
        ::close(sock_fd);
        return -1;
    }

    if (::bind(sock_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        setError(std::format("bind() failed: {}", std::strerror(errno)));
        logger_.error("TCP socket bind failed: {}", std::strerror(errno));
        ::close(sock_fd);
        return -1;
    }

    // Listen for connections
    if (::listen(sock_fd, config_.listen_backlog) < 0) {
        setError(std::format("listen() failed: {}", std::strerror(errno)));
        logger_.error("TCP socket listen failed: {}", std::strerror(errno));
        ::close(sock_fd);
        return -1;
    }

    logger_.info("TCP socket created and listening on {}:{}", config_.tcp_host, config_.tcp_port);
    return sock_fd;
}

[[nodiscard]] auto TokenReceiver::acceptConnection(int server_fd, std::stop_token const& stop_token)
    -> int {
    // Set accept timeout using select() for interruptible wait
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(server_fd, &read_fds);

    struct timeval tv;
    auto timeout_ms = config_.accept_timeout.count();
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int select_result = ::select(server_fd + 1, &read_fds, nullptr, nullptr, &tv);

    if (select_result < 0) {
        if (errno == EINTR) {
            // Interrupted by signal - check stop token
            return -1;
        }
        setError(std::format("select() failed: {}", std::strerror(errno)));
        logger_.error("select() failed: {}", std::strerror(errno));
        return -1;
    }

    if (select_result == 0) {
        // Timeout - check if stop requested
        return -1;
    }

    // Accept connection
    int client_fd = ::accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) {
        setError(std::format("accept() failed: {}", std::strerror(errno)));
        logger_.error("accept() failed: {}", std::strerror(errno));
        return -1;
    }

    return client_fd;
}

[[nodiscard]] auto TokenReceiver::receiveToken(int client_fd) -> std::string {
    // Set receive timeout
    struct timeval tv;
    auto timeout_ms = config_.receive_timeout.count();
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    if (::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        logger_.warn("setsockopt(SO_RCVTIMEO) failed: {}", std::strerror(errno));
    }

    // Allocate buffer for token
    std::string buffer;
    buffer.resize(config_.max_token_size);

    // Read token data
    ssize_t bytes_read = ::recv(client_fd, buffer.data(), buffer.size(), 0);

    if (bytes_read < 0) {
        setError(std::format("recv() failed: {}", std::strerror(errno)));
        logger_.error("recv() failed: {}", std::strerror(errno));
        return "";
    }

    if (bytes_read == 0) {
        logger_.warn("Client closed connection before sending data");
        return "";
    }

    // Resize to actual data received
    buffer.resize(static_cast<size_t>(bytes_read));

    // Validate token is not empty
    if (buffer.empty()) {
        logger_.warn("Received empty token");
        return "";
    }

    return buffer;
}

auto TokenReceiver::cleanupSocket() noexcept -> void {
    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;

        // Clean up Unix socket file if it exists
        if (config_.prefer_unix_socket) {
            ::unlink(config_.unix_socket_path.c_str());
        }

        logger_.debug("Socket cleaned up");
    }
}

auto TokenReceiver::setError(std::string_view message) -> void {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = std::string(message);
}

} // namespace bigbrother::utils
