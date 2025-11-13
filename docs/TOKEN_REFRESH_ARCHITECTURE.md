# Token Refresh Architecture

## System Overview

This document describes the complete token refresh architecture for the BigBrother Analytics trading platform.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Schwab API (OAuth 2.0)                       │
│                  https://api.schwabapi.com/v1/oauth                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ HTTPS
                                 │ (OAuth refresh_token grant)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Token Refresh Sender (Python)                    │
│                   scripts/token_refresh_sender.py                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Refresh Logic (Every 25 minutes)                           │  │
│  │  • Reads refresh_token from file                            │  │
│  │  • Calls Schwab API to get new access_token                 │  │
│  │  • Updates configs/schwab_tokens.json                       │  │
│  │  • Logs all operations                                      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└──────────────┬────────────────────────┬─────────────────────────────┘
               │                        │
               │ Writes                 │ Sends via socket
               │ (Atomic)               │ (JSON message)
               │                        │
               ▼                        ▼
┌──────────────────────────┐   ┌─────────────────────────────────────┐
│  Token File              │   │  Unix Domain Socket                 │
│  configs/                │   │  /tmp/bigbrother_token.sock         │
│  schwab_tokens.json      │   │                                     │
│                          │   │  Message Format:                    │
│  {                       │   │  {                                  │
│    "token": {            │   │    "access_token": "...",           │
│      "access_token": "…",│   │    "refresh_token": "...",          │
│      "refresh_token": "…"│   │    "expires_at": 1762984855         │
│      "expires_at": ...   │   │  }                                  │
│    }                     │   │                                     │
│  }                       │   │  Response: "OK"                     │
└──────────┬───────────────┘   └─────────────┬───────────────────────┘
           │                                 │
           │ Reads                           │ Receives
           │                                 │
           ▼                                 ▼
┌──────────────────────────┐   ┌─────────────────────────────────────┐
│  Dashboard (Python)      │   │  C++ Trading Bot                    │
│  dashboard/app.py        │   │  src/schwab_api/token_manager.cpp   │
│                          │   │                                     │
│  • Reads tokens for      │   │  ┌─────────────────────────────┐   │
│    API calls             │   │  │ Socket Server Thread        │   │
│  • Shows token status    │   │  │ • Listens on Unix socket    │   │
│  • No modification       │   │  │ • Receives token updates    │   │
│                          │   │  │ • Updates OAuth config      │   │
└──────────────────────────┘   │  │ • Sends "OK" acknowledgment │   │
                               │  └─────────────────────────────┘   │
                               │                                     │
                               │  ┌─────────────────────────────┐   │
                               │  │ File Reload Thread          │   │
                               │  │ • Fallback mechanism        │   │
                               │  │ • Reloads from file every   │   │
                               │  │   25 minutes                │   │
                               │  └─────────────────────────────┘   │
                               │                                     │
                               │  ┌─────────────────────────────┐   │
                               │  │ API Client                  │   │
                               │  │ • Uses current access_token │   │
                               │  │ • Makes Schwab API calls    │   │
                               │  └─────────────────────────────┘   │
                               └─────────────────────────────────────┘
```

## Component Responsibilities

### 1. Schwab API (External Service)

**Endpoint**: `https://api.schwabapi.com/v1/oauth/token`

**Responsibilities**:
- Issues initial OAuth tokens (access_token + refresh_token)
- Refreshes access_token when provided with valid refresh_token
- Enforces token lifetimes (access_token: 30 min, refresh_token: 7 days)

**OAuth Flow**:
```
POST /v1/oauth/token
Content-Type: application/x-www-form-urlencoded
Authorization: Basic <base64(client_id:client_secret)>

grant_type=refresh_token&refresh_token=<refresh_token>

Response:
{
  "access_token": "new_access_token",
  "refresh_token": "new_refresh_token",
  "expires_in": 1800,
  "token_type": "Bearer"
}
```

### 2. Token Refresh Sender (Python)

**File**: `scripts/token_refresh_sender.py`

**Responsibilities**:
- Refresh tokens proactively (every 25 minutes, before 30-minute expiry)
- Update token file atomically
- Send tokens to C++ bot via socket
- Handle network errors and retries
- Log all operations

**Timing**:
- Access tokens expire in 30 minutes
- Refreshes at 25 minutes (5-minute safety buffer)
- Non-blocking (continues even if bot is down)

**Error Handling**:
- Network errors: Log and retry next cycle
- Socket errors: 3 retries with 5-second delays
- Invalid tokens: Log error, requires manual OAuth re-authentication

### 3. Token File (Shared Storage)

**File**: `configs/schwab_tokens.json`

**Format**:
```json
{
    "creation_timestamp": 1762983055,
    "token": {
        "expires_in": 1800,
        "token_type": "Bearer",
        "scope": "api",
        "refresh_token": "...",
        "access_token": "...",
        "id_token": "...",
        "expires_at": 1762984855
    }
}
```

**Access Pattern**:
- **Writer**: Token Refresh Sender (atomic writes)
- **Readers**: Dashboard, C++ Bot (fallback)
- **Locking**: Not required (atomic writes prevent corruption)

### 4. Unix Domain Socket (IPC Mechanism)

**Path**: `/tmp/bigbrother_token.sock`

**Protocol**: SOCK_STREAM (connection-oriented)

**Message Flow**:
```
Client (Python)              Server (C++)
     │                            │
     │─── Connect ───────────────>│
     │                            │
     │─── JSON token data ───────>│
     │                            │
     │                            │── Parse JSON
     │                            │── Update config
     │                            │
     │<────── "OK" ──────────────│
     │                            │
     │─── Close ─────────────────>│
```

**Error Scenarios**:
- Socket doesn't exist → Python logs warning, continues
- Connection refused → Python retries 3 times
- Timeout → Python retries with 5-second delay
- Invalid JSON → C++ sends "ERROR", Python logs and retries

### 5. Dashboard (Python)

**File**: `dashboard/app.py`

**Responsibilities**:
- Read tokens from file for API calls
- Display token status to user
- Make Schwab API calls using current access_token
- No token modification (read-only)

**Token Usage**:
```python
# Load token from file
with open('configs/schwab_tokens.json') as f:
    data = json.load(f)
    access_token = data['token']['access_token']

# Use in API calls
headers = {
    'Authorization': f'Bearer {access_token}'
}
response = requests.get(schwab_api_url, headers=headers)
```

### 6. C++ Trading Bot

**File**: `src/schwab_api/token_manager.cpp`

**Components**:

#### Socket Server Thread
```cpp
// Listens for token updates from Python
socket_thread_ = std::thread([this]() {
    // Create Unix socket
    socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    bind(socket_fd_, "/tmp/bigbrother_token.sock");
    listen(socket_fd_, 5);

    while (running) {
        // Accept connections
        int client_fd = accept(socket_fd_);

        // Receive JSON token data
        recv(client_fd, buffer);

        // Parse and update config
        json token_json = json::parse(buffer);
        config_.access_token = token_json["access_token"];
        config_.refresh_token = token_json["refresh_token"];

        // Send acknowledgment
        send(client_fd, "OK");
        close(client_fd);
    }
});
```

#### File Reload Thread (Fallback)
```cpp
// Fallback: reload from file every 25 minutes
refresh_thread_ = std::thread([this]() {
    while (running) {
        sleep(25 * 60); // 25 minutes

        // Reload from file
        std::ifstream file("configs/schwab_tokens.json");
        json j;
        file >> j;

        config_.access_token = j["token"]["access_token"];
        config_.refresh_token = j["token"]["refresh_token"];
    }
});
```

#### API Client
```cpp
// Use current access_token for API calls
auto getAccessToken() -> Result<std::string> {
    std::lock_guard lock{mutex_};

    // Check if expired
    if (config_.isAccessTokenExpired()) {
        // Refresh (or wait for socket update)
        refreshAccessTokenInternal();
    }

    return config_.access_token;
}
```

## Sequence Diagrams

### Normal Token Refresh Flow

```
Time: T+0 (Initial state, token expires at T+30)

Token Sender          Token File          Socket          C++ Bot
     │                     │                 │                │
T+25 │─── Read ──────────>│                 │                │
     │<── refresh_token ──│                 │                │
     │                     │                 │                │
     │─── Schwab API (HTTPS) ──────────────────────────────> │
     │<──── new tokens ──────────────────────────────────────│
     │                     │                 │                │
     │─── Write ─────────>│                 │                │
     │                     │                 │                │
     │─────────────── Connect ────────────>│                │
     │─────────────── JSON ────────────────>│─── Receive ──>│
     │                     │                 │                │
     │                     │                 │<── Update ────│
     │                     │                 │    config      │
     │<──────────────── "OK" ──────────────│                │
     │                     │                 │                │
     │─── Log success ───>│                 │                │
     │                     │                 │                │
     │─── Sleep 25min ───>│                 │                │
     │                     │                 │                │
```

### Bot Not Running Scenario

```
Token Sender          Token File          Socket          C++ Bot
     │                     │                 │                │
T+25 │─── Read ──────────>│                 │                │
     │<── refresh_token ──│                 │                │
     │                     │                 │                │
     │─── Schwab API ────────────────────────────────────────> │
     │<──── new tokens ──────────────────────────────────────│
     │                     │                 │                │
     │─── Write ─────────>│                 │                │
     │                     │                 │                │
     │─────────────── Connect ────────────>│                │
     │                     │                 X (not running) │
     │<──────── FileNotFoundError ─────────│                │
     │                     │                 │                │
     │─── Log warning ───>│                 │                │
     │    "Bot not running"│                 │                │
     │                     │                 │                │
     │─── Sleep 25min ───>│                 │                │
     │                     │                 │                │
     │                     │                 │                │
     │                     │<──────────────────────── Start ──│
     │                     │                 │                │
     │                     │─── Read on startup ────────────>│
     │                     │    (has latest tokens)          │
```

### Network Error Scenario

```
Token Sender          Schwab API          Token File
     │                     │                 │
T+25 │─── Refresh request >│                 │
     │                     X (timeout)       │
     │<──── Timeout ──────│                 │
     │                     │                 │
     │─── Log error ─────────────────────────>
     │    "Network timeout"                  │
     │                     │                 │
     │─── Sleep 25min ────────────────────────>
     │                     │                 │
T+50 │─── Retry ──────────>│                 │
     │<──── Success ──────│                 │
     │                     │                 │
     │─── Write ──────────────────────────────>
```

## Configuration Files

### 1. `configs/schwab_tokens.json`
```json
{
    "creation_timestamp": 1762983055,
    "token": {
        "expires_in": 1800,
        "token_type": "Bearer",
        "scope": "api",
        "refresh_token": "-ObzVyoqmneuwhZkH4KUFr2F7Dxayi6L...",
        "access_token": "I0.b2F1dGgyLmNkYy5zY2h3YWIuY29t...",
        "id_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
        "expires_at": 1762984855
    }
}
```

### 2. `configs/schwab_app_config.yaml`
```yaml
app_name: DataTradingApp
app_key: 8fOTwEsi51wbdmEsn5dSYxk6Y38ZJKH09etLXf3uJgyUXcIa
app_secret: PKy7ILBYmEnxEkm0BQNt28AHtHoYs3c4y09cG51LqMsVkdZkmwuOqBBUDJWzMamT
callback_url: https://127.0.0.1:8182
```

## Deployment Scenarios

### Scenario 1: Dashboard Only (No C++ Bot)

```
┌──────────────────┐
│ Token Sender     │─── Refreshes tokens every 25min
│ (Python)         │─── Updates file
└────────┬─────────┘    No socket (logs warning)
         │
         │ Writes
         ▼
┌──────────────────┐
│ Token File       │
└────────┬─────────┘
         │
         │ Reads
         ▼
┌──────────────────┐
│ Dashboard        │─── Uses tokens for API calls
│ (Python)         │
└──────────────────┘
```

### Scenario 2: Full System (Dashboard + C++ Bot)

```
┌──────────────────┐      ┌──────────────────┐
│ Token Sender     │─────>│ Token File       │
│ (Python)         │      └──────────────────┘
└────────┬─────────┘               │
         │                         │ Reads
         │ Socket                  │
         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│ C++ Bot          │      │ Dashboard        │
│ (Primary)        │      │ (Secondary)      │
└──────────────────┘      └──────────────────┘
```

### Scenario 3: C++ Bot Only (No Dashboard)

```
┌──────────────────┐      ┌──────────────────┐
│ Token Sender     │─────>│ Token File       │
│ (Python)         │      └──────────────────┘
└────────┬─────────┘               │
         │                         │ Fallback
         │ Socket (Primary)        │ (if socket fails)
         ▼                         ▼
┌──────────────────────────────────────────┐
│            C++ Bot                       │
│  ┌──────────────┐   ┌─────────────────┐ │
│  │ Socket Thread│   │ File Reload     │ │
│  │ (Primary)    │   │ (Fallback)      │ │
│  └──────────────┘   └─────────────────┘ │
└──────────────────────────────────────────┘
```

## Failure Modes and Recovery

| Failure | Detection | Recovery | Impact |
|---------|-----------|----------|--------|
| Network timeout | Schwab API timeout | Retry next cycle (25 min) | Tokens may expire if persistent |
| Invalid refresh_token | HTTP 401 from Schwab | Manual re-auth required | Service stops working |
| Socket connection failed | Connection refused | 3 retries, then continue | Bot uses file fallback |
| Token file corrupted | JSON parse error | Log error, wait for next refresh | Services use cached tokens |
| Bot not running | FileNotFoundError | Continue, log warning | Tokens still updated in file |
| Disk full | Write error | Log error, retry next cycle | Tokens not updated |

## Monitoring and Observability

### Key Metrics to Monitor

1. **Token Refresh Success Rate**
   - Monitor: `logs/token_refresh_sender.log`
   - Pattern: `"Token refreshed successfully"`
   - Alert: If < 95% over 24 hours

2. **Socket Connection Success Rate**
   - Monitor: `logs/token_refresh_sender.log`
   - Pattern: `"Token successfully sent to trading bot"`
   - Alert: If < 90% and bot is running

3. **Token Age**
   - Monitor: `configs/schwab_tokens.json`
   - Field: `creation_timestamp`
   - Alert: If > 35 minutes old (should be ~25 min)

4. **Service Uptime**
   - Check: Process running
   - Command: `ps aux | grep token_refresh_sender`
   - Alert: If not running

### Log Patterns

**Success**:
```
[2025-11-13 08:44:24] INFO: Token refreshed successfully (expires in 30 minutes)
[2025-11-13 08:44:24] INFO: Token successfully sent to trading bot
```

**Warning (Bot down)**:
```
[2025-11-13 08:44:24] WARNING: Bot socket not found - trading bot may not be running
```

**Error (Network)**:
```
[2025-11-13 08:44:24] ERROR: Token refresh failed: HTTP 401
[2025-11-13 08:44:24] ERROR: Response: {"error": "invalid_grant"}
```

## Security Architecture

### Authentication Flow

```
App Key + App Secret
        │
        │ (Base64 encoded)
        ▼
  Authorization: Basic <base64>
        │
        │ (HTTPS)
        ▼
   Schwab API
        │
        │ (Validates credentials)
        ▼
 Refresh Token (7-day lifetime)
        │
        │ (Stored in file)
        ▼
 Access Token (30-min lifetime)
```

### Data Protection

1. **In Transit**:
   - Schwab API: HTTPS (TLS 1.2+)
   - Socket: Unix domain (no network exposure)

2. **At Rest**:
   - Token file: OS file permissions
   - App config: OS file permissions
   - Logs: No full tokens (only truncated)

3. **In Memory**:
   - Python: Cleared on shutdown
   - C++: Locked with mutex, cleared on shutdown

## Conclusion

This architecture provides:
- ✅ Automatic token refresh (every 25 minutes)
- ✅ Dual delivery mechanism (socket + file)
- ✅ Graceful degradation (continues if bot is down)
- ✅ Comprehensive logging and monitoring
- ✅ Security through HTTPS and file permissions
- ✅ Compatibility with multiple deployment scenarios

The system is resilient, observable, and production-ready.
