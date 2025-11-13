# Token Receiver - Quick Start Guide

## Overview

The Token Receiver module provides automated OAuth token refresh for BigBrotherAnalytics trading system. It receives updated access tokens from external authentication services via socket communication.

## 5-Minute Setup

### 1. Build the Module

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics
cd build
cmake --build . --target test_token_receiver
```

### 2. Run the Test Application

```bash
./bin/test_token_receiver
```

You should see:
```
Token receiver started successfully!
Listening on:
  - Unix socket: /tmp/bigbrother_token.sock
  - TCP socket: 127.0.0.1:9999
```

### 3. Send a Test Token

Open a new terminal and run:

**Option A: Unix Socket (Recommended)**
```bash
echo "test_access_token_12345" | nc -U /tmp/bigbrother_token.sock
```

**Option B: TCP Socket**
```bash
echo "test_access_token_12345" | nc localhost 9999
```

**Option C: Python Script**
```bash
./scripts/send_test_token.py "test_access_token_12345"
```

### 4. Verify Token Received

In the first terminal, you should see:
```
[CALLBACK] New token received!
Token: test_access_token_12345
Token size: 26 bytes

[STATUS] Token count: 1
[STATUS] Latest token: test_access_token_12345
```

## Integration Example

### Basic Integration

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
        logger.info("New token received: {} bytes", token.size());
        // TODO: Update your OAuth client here
    });

    // Start receiving tokens
    receiver.start();

    // Your application logic here...
    while (running) {
        // Get latest token when needed
        std::string token = receiver.getLatestToken();
        // Use token for API calls...
    }

    return 0;
}
```

### Integration with Schwab API

```cpp
import bigbrother.utils.token_receiver;
import bigbrother.schwab_api.schwab_api;

class TradingApp {
private:
    SchwabAPIClient schwab_;
    TokenReceiver token_receiver_;

public:
    TradingApp()
        : token_receiver_([this](std::string const& token) {
              schwab_.updateAccessToken(token);
              schwab_.saveTokens();
              logger_.info("Token refreshed!");
          }) {
        token_receiver_.start();
    }
};
```

## Configuration Options

```cpp
TokenReceiverConfig config;
config.unix_socket_path = "/tmp/bigbrother_token.sock";
config.tcp_host = "127.0.0.1";
config.tcp_port = 9999;
config.prefer_unix_socket = true;  // Try Unix first
config.max_token_size = 8192;      // 8KB max token
config.accept_timeout = std::chrono::milliseconds(500);
config.receive_timeout = std::chrono::milliseconds(5000);

TokenReceiver receiver(my_callback, config);
```

## Testing Scripts

### Continuous Token Sending

```bash
# Send a new token every 30 seconds
./scripts/send_test_token.py --continuous --interval 30
```

### Send Tokens from File

```bash
# Create token file
cat > tokens.txt << EOF
token_1_abc123
token_2_def456
token_3_ghi789
EOF

# Send tokens from file
./scripts/send_test_token.py --file tokens.txt
```

### TCP vs Unix Socket

```bash
# Unix socket (faster, more secure)
./scripts/send_test_token.py "test_token"

# TCP socket (remote access)
./scripts/send_test_token.py --tcp --host 127.0.0.1 --port 9999 "test_token"
```

## Production Deployment

### 1. Create OAuth Refresh Service

Example Python service:

```python
#!/usr/bin/env python3
import socket
import time
import requests

def refresh_and_send_token():
    # Get new token from OAuth provider
    response = requests.post(
        "https://api.schwabapi.com/v1/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
    )

    token = response.json()["access_token"]

    # Send to BigBrother
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/tmp/bigbrother_token.sock")
    sock.sendall(token.encode())
    sock.close()

# Refresh every 25 minutes (tokens expire in 30)
while True:
    refresh_and_send_token()
    time.sleep(25 * 60)
```

### 2. Setup Systemd Service

```ini
# /etc/systemd/system/bigbrother-token-refresh.service
[Unit]
Description=BigBrother OAuth Token Refresh
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

Enable and start:
```bash
sudo systemctl enable bigbrother-token-refresh
sudo systemctl start bigbrother-token-refresh
```

### 3. Security Hardening

```bash
# Set socket permissions (Unix socket only)
chmod 600 /tmp/bigbrother_token.sock
chown bigbrother:bigbrother /tmp/bigbrother_token.sock

# For TCP: bind to localhost only (default)
# Or use firewall rules:
sudo ufw allow from 127.0.0.1 to any port 9999
```

## Troubleshooting

### "Address already in use"

```bash
# Remove stale socket file
rm /tmp/bigbrother_token.sock

# Or change path in config
config.unix_socket_path = "/tmp/bigbrother_token_new.sock";
```

### "Connection refused"

1. Check receiver is running:
   ```cpp
   if (!receiver.isRunning()) {
       std::cerr << "Not running!" << std::endl;
   }
   ```

2. Check socket exists:
   ```bash
   ls -la /tmp/bigbrother_token.sock
   ```

3. Test connectivity:
   ```bash
   echo "test" | nc -U /tmp/bigbrother_token.sock
   ```

### Token not received

Check logs:
```bash
tail -f logs/bigbrother.log | grep TokenReceiver
```

Verify callback:
```cpp
TokenReceiver receiver([](std::string const& token) {
    std::cout << "Token: " << token << std::endl;  // Add debug output
});
```

## API Reference

### Class: TokenReceiver

| Method | Description | Returns |
|--------|-------------|---------|
| `start()` | Start receiver thread | `bool` (true if success) |
| `stop()` | Stop receiver thread | `void` |
| `getLatestToken()` | Get current token | `std::string` |
| `isRunning()` | Check if running | `bool` |
| `getTokenCount()` | Get tokens received | `uint64_t` |
| `getLastError()` | Get error message | `std::string` |

### Callback Signature

```cpp
using TokenCallback = std::function<void(std::string const&)>;
```

**Important:** Callback runs on receiver thread, keep it fast (<1ms).

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Token receive | <1ms | Unix socket, local |
| Token update | <10μs | Mutex + copy |
| Token retrieval | <5μs | Mutex + copy |

## Next Steps

1. **Read Full Documentation**: See [TOKEN_RECEIVER.md](TOKEN_RECEIVER.md)
2. **Review Example**: Check [examples/test_token_receiver.cpp](../examples/test_token_receiver.cpp)
3. **Setup Production**: Deploy OAuth refresh service
4. **Monitor**: Track token refresh rate and errors

## Common Use Cases

### Case 1: Schwab API Integration

```cpp
TokenReceiver receiver([&schwab_client](std::string const& token) {
    schwab_client.updateAccessToken(token);
    schwab_client.saveTokens();
});
receiver.start();
```

### Case 2: Multiple Clients

```cpp
std::vector<std::reference_wrapper<APIClient>> clients;

TokenReceiver receiver([&clients](std::string const& token) {
    for (auto& client : clients) {
        client.get().updateToken(token);
    }
});
```

### Case 3: Token Persistence

```cpp
TokenReceiver receiver([](std::string const& token) {
    // Save to file
    std::ofstream file("tokens/current.txt");
    file << token;

    // Save to database
    db.execute("UPDATE tokens SET value = ? WHERE id = 1", token);
});
```

## Support

- Documentation: [docs/TOKEN_RECEIVER.md](TOKEN_RECEIVER.md)
- Examples: [examples/test_token_receiver.cpp](../examples/test_token_receiver.cpp)
- Test Script: [scripts/send_test_token.py](../scripts/send_test_token.py)

## License

Part of BigBrotherAnalytics - Automated Trading System
Author: Olumuyiwa Oluwasanmi
