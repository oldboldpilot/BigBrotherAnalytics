# Token Receiver Module

## Overview

The Token Receiver module (`token_receiver.cppm`) provides a socket-based OAuth token refresh mechanism for automated trading systems. It enables external authentication services to push updated access tokens to the trading application without requiring manual intervention or system restarts.

## Features

- **Thread-Safe Token Storage**: Atomic operations and mutex protection for concurrent access
- **Callback Mechanism**: Notify application components when new tokens arrive
- **Dual Socket Support**: Unix domain socket (preferred) or TCP socket (fallback)
- **Graceful Shutdown**: C++23 `std::jthread` with cooperative cancellation
- **Error Recovery**: Automatic socket cleanup and error reporting
- **Zero-Copy Performance**: Direct socket-to-string operations
- **RAII Design**: Automatic resource cleanup on destruction

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  External Token Service                      │
│  (OAuth refresh server, authentication microservice, etc.)  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Send token via socket
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    TokenReceiver                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Receiver Thread (std::jthread)                       │  │
│  │  - Listen on Unix socket or TCP socket               │  │
│  │  - Accept connections                                 │  │
│  │  - Receive token data                                 │  │
│  │  - Validate and store token                           │  │
│  │  - Invoke callback                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         │ Token update                       │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Token Storage (mutex + atomic)                       │  │
│  │  - latest_token_: std::string                         │  │
│  │  - token_count_: std::atomic<uint64_t>               │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Callback invocation
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Application Components                          │
│  - Schwab API Client (update OAuth token)                   │
│  - Token Manager (persist to disk)                          │
│  - Trading Engine (restart connections)                     │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### TokenReceiverConfig

Configuration structure for socket and behavior settings.

```cpp
struct TokenReceiverConfig {
    // Socket paths
    std::string unix_socket_path{"/tmp/bigbrother_token.sock"};
    std::string tcp_host{"127.0.0.1"};
    uint16_t tcp_port{9999};

    // Behavior
    bool prefer_unix_socket{true};     // Try Unix first, fallback to TCP
    uint32_t max_token_size{8192};     // Maximum token size in bytes
    uint32_t listen_backlog{5};        // Socket listen queue size

    // Timeouts
    std::chrono::milliseconds accept_timeout{500};    // Connection accept timeout
    std::chrono::milliseconds receive_timeout{5000};  // Token receive timeout
};
```

### TokenReceiver Class

Main class for receiving and managing OAuth tokens.

#### Constructor

```cpp
explicit TokenReceiver(TokenCallback callback,
                      TokenReceiverConfig config = TokenReceiverConfig{});
```

**Parameters:**
- `callback`: Function called when new token received (signature: `void(std::string const&)`)
- `config`: Socket and behavior configuration (optional, uses defaults if omitted)

**Example:**
```cpp
TokenReceiver receiver(
    [](std::string const& token) {
        std::cout << "New token: " << token << std::endl;
        // Update OAuth client, persist to disk, etc.
    }
);
```

#### Public Methods

##### start()

Start the token receiver thread.

```cpp
[[nodiscard]] auto start() noexcept -> bool;
```

**Returns:** `true` if started successfully, `false` if already running or error

**Example:**
```cpp
if (!receiver.start()) {
    std::cerr << "Failed to start: " << receiver.getLastError() << std::endl;
}
```

##### stop()

Stop the token receiver thread gracefully.

```cpp
auto stop() noexcept -> void;
```

Automatically called by destructor. Safe to call multiple times.

**Example:**
```cpp
receiver.stop();  // Graceful shutdown
```

##### getLatestToken()

Get the most recently received token.

```cpp
[[nodiscard]] auto getLatestToken() const -> std::string;
```

**Returns:** Latest token string (empty if no token received yet)

**Thread-Safety:** Fully thread-safe, uses mutex protection

**Example:**
```cpp
std::string token = receiver.getLatestToken();
if (!token.empty()) {
    // Use token for API calls
}
```

##### isRunning()

Check if receiver thread is active.

```cpp
[[nodiscard]] auto isRunning() const noexcept -> bool;
```

**Returns:** `true` if receiver thread is running

##### getTokenCount()

Get total number of tokens received since start.

```cpp
[[nodiscard]] auto getTokenCount() const noexcept -> uint64_t;
```

**Returns:** Token count (atomic read, no locking)

##### getLastError()

Get last error message.

```cpp
[[nodiscard]] auto getLastError() const -> std::string;
```

**Returns:** Error message or empty string if no errors

## Usage Examples

### Basic Usage

```cpp
#include <iostream>
import bigbrother.utils.token_receiver;
import bigbrother.utils.logger;

using namespace bigbrother::utils;

int main() {
    // Initialize logger
    auto& logger = Logger::getInstance();
    logger.initialize("logs/app.log", LogLevel::INFO, true);

    // Create receiver with callback
    TokenReceiver receiver([&logger](std::string const& token) {
        logger.info("Token updated: {} bytes", token.size());
        // TODO: Update OAuth client with new token
    });

    // Start receiver
    if (!receiver.start()) {
        logger.error("Failed to start receiver: {}", receiver.getLastError());
        return 1;
    }

    // Application main loop
    while (running) {
        // ... trading logic ...

        // Get latest token when needed
        std::string token = receiver.getLatestToken();
        // Use token for API calls
    }

    // Graceful shutdown (automatic on destruction)
    receiver.stop();
    return 0;
}
```

### Integration with Schwab API

```cpp
import bigbrother.utils.token_receiver;
import bigbrother.schwab_api.schwab_api;

class TradingEngine {
private:
    SchwabAPIClient schwab_client_;
    TokenReceiver token_receiver_;

public:
    TradingEngine()
        : token_receiver_([this](std::string const& token) {
              onTokenUpdate(token);
          }) {
        token_receiver_.start();
    }

    void onTokenUpdate(std::string const& token) {
        // Update Schwab API client with new token
        schwab_client_.updateAccessToken(token);

        // Persist token to disk
        schwab_client_.saveTokens();

        logger_.info("OAuth token refreshed successfully");
    }

    void executeTradeWithLatestToken() {
        // Token is automatically kept up-to-date
        auto orders = schwab_client_.getOrders();
        // ... process orders ...
    }
};
```

### Custom Configuration

```cpp
// Configure for production environment
TokenReceiverConfig config;
config.unix_socket_path = "/var/run/bigbrother/token.sock";
config.tcp_host = "0.0.0.0";  // Listen on all interfaces
config.tcp_port = 8443;       // Custom port
config.prefer_unix_socket = true;
config.max_token_size = 16384;  // Support larger tokens
config.accept_timeout = std::chrono::milliseconds(1000);
config.receive_timeout = std::chrono::milliseconds(10000);

TokenReceiver receiver(my_callback, config);
receiver.start();
```

## Testing

### Manual Testing

1. **Build and run the test application:**
   ```bash
   cd build
   cmake --build .
   ./bin/test_token_receiver
   ```

2. **Send test tokens using Unix socket:**
   ```bash
   echo "test_access_token_12345" | nc -U /tmp/bigbrother_token.sock
   ```

3. **Send test tokens using TCP socket:**
   ```bash
   echo "test_access_token_67890" | nc localhost 9999
   ```

### Automated Testing

```cpp
#include <gtest/gtest.h>
import bigbrother.utils.token_receiver;

TEST(TokenReceiverTest, ReceiveToken) {
    std::string received_token;
    TokenReceiver receiver([&received_token](std::string const& token) {
        received_token = token;
    });

    ASSERT_TRUE(receiver.start());

    // Send test token via socket
    sendTestToken("test_token_123");

    // Wait for callback
    std::this_thread::sleep_for(std::chrono::seconds(1));

    EXPECT_EQ(receiver.getLatestToken(), "test_token_123");
    EXPECT_EQ(received_token, "test_token_123");
    EXPECT_EQ(receiver.getTokenCount(), 1);

    receiver.stop();
}
```

## Security Considerations

### Unix Domain Socket (Recommended)

- **Pros:**
  - Filesystem-based permissions (can restrict to specific users/groups)
  - Faster than TCP (no network stack overhead)
  - Cannot be accessed remotely
  - Better for local IPC

- **Security Setup:**
  ```bash
  # Set socket permissions after creation
  chmod 600 /tmp/bigbrother_token.sock
  chown bigbrother:bigbrother /tmp/bigbrother_token.sock
  ```

### TCP Socket

- **Pros:**
  - Can receive tokens from remote services
  - Easier to test and debug

- **Security Concerns:**
  - Should bind to `127.0.0.1` (localhost only) by default
  - Use firewall rules to restrict access
  - Consider TLS wrapper for production (stunnel, socat)

- **Production Hardening:**
  ```bash
  # Use stunnel for TLS
  stunnel /etc/stunnel/bigbrother.conf

  # Or use socat
  socat TCP-LISTEN:8443,reuseaddr,fork,cert=/path/to/cert.pem,verify=1 \
        UNIX-CONNECT:/tmp/bigbrother_token.sock
  ```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Token receive | <1ms | Unix socket, local network |
| Token update | <10μs | Mutex lock + string copy |
| Token retrieval | <5μs | Mutex lock + string copy |
| Callback invocation | Variable | Depends on callback implementation |
| Thread startup | ~1ms | Socket creation + thread spawn |
| Thread shutdown | <100ms | Graceful stop with timeout |

## Error Handling

The TokenReceiver handles errors gracefully:

1. **Socket Creation Failure:**
   - Logs error message
   - Returns `false` from `start()`
   - Error available via `getLastError()`

2. **Accept Timeout:**
   - Not an error - allows checking stop_token
   - Continues listening

3. **Receive Failure:**
   - Logs warning
   - Closes client connection
   - Continues listening for next connection

4. **Callback Exception:**
   - Catches exception
   - Logs error message
   - Continues operation (doesn't crash receiver)

## Integration with External Services

### OAuth Refresh Server

Example Python server to send tokens:

```python
#!/usr/bin/env python3
import socket
import time
import requests

SOCKET_PATH = "/tmp/bigbrother_token.sock"
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
REFRESH_TOKEN = "your_refresh_token"

def refresh_and_send_token():
    # Get new access token from OAuth provider
    response = requests.post("https://api.schwabapi.com/v1/oauth/token", data={
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    })

    access_token = response.json()["access_token"]

    # Send to BigBrotherAnalytics via socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)
    sock.sendall(access_token.encode())
    sock.close()

    print(f"Sent token: {access_token[:20]}...")

# Refresh every 25 minutes (tokens expire in 30 minutes)
while True:
    try:
        refresh_and_send_token()
        time.sleep(25 * 60)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
```

### Systemd Integration

Create a systemd service for the token refresh server:

```ini
[Unit]
Description=BigBrother OAuth Token Refresh Service
After=network.target

[Service]
Type=simple
User=bigbrother
ExecStart=/usr/local/bin/bigbrother-token-refresh.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### "Address already in use" Error

```bash
# Check if socket file exists
ls -la /tmp/bigbrother_token.sock

# Remove stale socket file
rm /tmp/bigbrother_token.sock

# Or change socket path in config
```

### Token Not Received

1. Check receiver is running:
   ```cpp
   if (!receiver.isRunning()) {
       std::cerr << "Receiver not running!" << std::endl;
   }
   ```

2. Test socket connectivity:
   ```bash
   echo "test" | nc -U /tmp/bigbrother_token.sock
   # or
   echo "test" | nc localhost 9999
   ```

3. Check logs for errors:
   ```bash
   tail -f logs/bigbrother.log | grep TokenReceiver
   ```

### Callback Not Invoked

- Verify callback is set (not null)
- Check for exceptions in callback (logged but not propagated)
- Ensure callback doesn't block (runs on receiver thread)

## Best Practices

1. **Callback Performance:**
   - Keep callback fast (<1ms)
   - Offload heavy work to separate thread
   - Don't block receiver thread

2. **Error Handling:**
   - Always check `start()` return value
   - Monitor `getLastError()` periodically
   - Log token reception statistics

3. **Security:**
   - Prefer Unix domain socket over TCP
   - Set restrictive filesystem permissions
   - Validate token format in callback

4. **Production:**
   - Use systemd for automatic restart
   - Monitor token refresh rate
   - Alert on stale tokens (no updates for >30 minutes)

## Future Enhancements

- [ ] TLS support for TCP sockets
- [ ] Token validation (JWT parsing, expiry check)
- [ ] Metrics export (Prometheus, StatsD)
- [ ] Multiple callback registration
- [ ] Token history/rotation tracking
- [ ] Automatic token persistence to disk

## References

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [Unix Domain Sockets](https://man7.org/linux/man-pages/man7/unix.7.html)
- [OAuth 2.0 Token Refresh](https://datatracker.ietf.org/doc/html/rfc6749#section-6)
- [Schwab API Documentation](https://developer.schwab.com/)
