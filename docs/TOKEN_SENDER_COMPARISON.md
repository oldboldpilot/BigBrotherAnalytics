# Token Sender Script Comparison

## Overview

This document compares the newly created `token_refresh_sender.py` with the existing `token_refresh_service.py`, explaining improvements and why both exist.

## File Locations

| Script | Path | Purpose |
|--------|------|---------|
| **New** | `/scripts/token_refresh_sender.py` | Production-ready token sender with enhanced features |
| **Original** | `/scripts/token_refresh_service.py` | Initial implementation of token refresh service |

## Key Improvements in `token_refresh_sender.py`

### 1. Enhanced Error Handling

**Original (`token_refresh_service.py`)**:
```python
def send_token_to_bot(self, token_data):
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        # ... send data
    except FileNotFoundError:
        self.log("⚠️  Bot socket not found")
        return False
    except Exception as e:
        self.log(f"❌ Failed to send token: {e}")
        return False
```

**New (`token_refresh_sender.py`)**:
```python
def send_token_to_bot(self, token_data: Dict[str, Any]) -> bool:
    for attempt in range(1, MAX_SOCKET_RETRIES + 1):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(self.socket_path)
            # ... send data
        except FileNotFoundError:
            self.log_warn("Bot socket not found - bot may not be running")
            return False
        except ConnectionRefusedError:
            self.log_warn(f"Connection refused (attempt {attempt}/{MAX_SOCKET_RETRIES})")
            if attempt < MAX_SOCKET_RETRIES:
                time.sleep(SOCKET_RETRY_DELAY)
                continue
            return False
        except socket.timeout:
            self.log_error(f"Socket timeout (attempt {attempt}/{MAX_SOCKET_RETRIES})")
            if attempt < MAX_SOCKET_RETRIES:
                time.sleep(SOCKET_RETRY_DELAY)
                continue
            return False
```

**Improvements**:
- Retry logic with configurable attempts (3 retries by default)
- Delays between retries (5 seconds)
- Specific exception handling for different failure modes
- Socket timeouts to prevent hanging
- Better logging with log levels (info, warn, error)

### 2. Professional Logging

**Original**:
```python
def log(self, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}\n"
    print(log_msg.strip())
    with open(LOG_FILE, "a") as f:
        f.write(log_msg)
```

**New**:
```python
def _setup_logging(self):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    self.logger = logging.getLogger(__name__)

def log_info(self, message: str):
    self.logger.info(message)

def log_warn(self, message: str):
    self.logger.warning(message)

def log_error(self, message: str):
    self.logger.error(message)
```

**Improvements**:
- Uses Python's standard `logging` module
- Log levels (INFO, WARNING, ERROR) for better filtering
- Atomic file writes (logging module handles file locking)
- Better formatting and structure
- Can easily change log levels for debugging

### 3. Type Hints

**Original**:
```python
def refresh_token(self):
    # ...

def send_token_to_bot(self, token_data):
    # ...
```

**New**:
```python
def refresh_token(self) -> Optional[Dict[str, Any]]:
    # ...

def send_token_to_bot(self, token_data: Dict[str, Any]) -> bool:
    # ...
```

**Improvements**:
- Type hints for better code clarity and IDE support
- Easier to understand function contracts
- Catches type errors during development
- Better documentation

### 4. Validation and Error Messages

**Original**:
```python
def __init__(self):
    with open(APP_CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    # Continues even if file doesn't exist (crashes later)
```

**New**:
```python
def __init__(self):
    self.config_path = Path(APP_CONFIG_FILE)
    if not self.config_path.exists():
        self.log_error(f"App config file not found: {APP_CONFIG_FILE}")
        raise FileNotFoundError(f"Missing config file: {APP_CONFIG_FILE}")

    with open(self.config_path) as f:
        config = yaml.safe_load(f)

    # Verify token file exists
    self.token_path = Path(TOKEN_FILE)
    if not self.token_path.exists():
        self.log_error(f"Token file not found: {TOKEN_FILE}")
        raise FileNotFoundError(f"Missing token file: {TOKEN_FILE}")
```

**Improvements**:
- Validates configuration files exist before starting
- Provides clear error messages for missing files
- Fails fast with actionable error information
- Uses `pathlib.Path` for better path handling

### 5. Documentation

**Original**:
```python
"""
Token Refresh Service - Refreshes Schwab OAuth tokens
"""
```

**New**:
```python
"""
Token Refresh Sender - Refreshes Schwab OAuth tokens and sends to C++ trading bot

This service runs in the background and:
1. Reads Schwab OAuth tokens from configs/schwab_tokens.json
2. Refreshes OAuth tokens every 25 minutes using Schwab's refresh_token grant
3. Updates the token file with new tokens
4. Sends updated tokens to the C++ trading bot via Unix domain socket
5. Handles socket disconnections gracefully with retry logic
6. Logs all refresh operations

Compatible with running alongside the dashboard.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-13
"""
```

**Improvements**:
- Comprehensive docstring explaining all functionality
- Step-by-step description of what the script does
- Clear compatibility notes
- Proper attribution and dating

### 6. Better Token File Handling

**Original**:
```python
with open(TOKEN_FILE) as f:
    data = json.load(f)

refresh_token = data['token']['refresh_token']
# May crash if structure is wrong
```

**New**:
```python
with open(self.token_path) as f:
    data = json.load(f)

if 'token' not in data or 'refresh_token' not in data['token']:
    self.log_error("Invalid token file structure - missing refresh_token")
    return None

refresh_token = data['token']['refresh_token']
```

**Improvements**:
- Validates token file structure
- Graceful handling of malformed JSON
- Clear error messages for debugging
- Doesn't crash on invalid data

### 7. Enhanced Token Refresh Logic

**Original**:
```python
if response.status_code == 200:
    token_data = response.json()
    data['token']['access_token'] = token_data['access_token']
    data['token']['refresh_token'] = token_data.get('refresh_token', refresh_token)
```

**New**:
```python
if response.status_code == 200:
    token_data = response.json()

    # Update token file
    data['creation_timestamp'] = int(time.time())
    data['token']['access_token'] = token_data['access_token']
    data['token']['refresh_token'] = token_data.get('refresh_token', refresh_token)
    data['token']['expires_in'] = token_data['expires_in']
    data['token']['expires_at'] = int(time.time()) + token_data['expires_in']

    # Preserve other fields if they exist
    if 'token_type' in token_data:
        data['token']['token_type'] = token_data['token_type']
    if 'scope' in token_data:
        data['token']['scope'] = token_data['scope']
```

**Improvements**:
- Updates creation timestamp
- Preserves all token metadata
- Calculates expiration time
- Maintains token file integrity

## Feature Comparison Table

| Feature | Original | New | Notes |
|---------|----------|-----|-------|
| Socket retry logic | ❌ No | ✅ Yes (3 retries) | Handles bot startup delays |
| Socket timeout | ❌ No | ✅ Yes (5 seconds) | Prevents hanging |
| Structured logging | ❌ Basic | ✅ Professional | Uses logging module |
| Log levels | ❌ No | ✅ Yes (INFO/WARN/ERROR) | Better filtering |
| Type hints | ❌ No | ✅ Yes | Better IDE support |
| Input validation | ❌ Minimal | ✅ Comprehensive | Fails fast with clear errors |
| Error messages | ⚠️ Generic | ✅ Specific | Actionable debugging info |
| Documentation | ⚠️ Basic | ✅ Extensive | Includes README |
| Exception handling | ⚠️ Catch-all | ✅ Granular | Specific error types |
| Path handling | ⚠️ Strings | ✅ pathlib.Path | Better cross-platform |
| Token validation | ❌ No | ✅ Yes | Checks structure |
| Graceful shutdown | ✅ Yes | ✅ Yes | Both handle signals |
| Non-blocking socket | ✅ Yes | ✅ Yes | Both continue if bot down |

## When to Use Each Script

### Use `token_refresh_sender.py` (NEW) when:

- Running in production
- Need robust error handling
- Want detailed logging for monitoring
- Running with systemd or other service managers
- Need retry logic for reliability
- Want clear error diagnostics

### Use `token_refresh_service.py` (ORIGINAL) when:

- Quick testing or development
- Don't need retry logic
- Prefer simpler code
- Have stable bot connection

## Migration Guide

To migrate from `token_refresh_service.py` to `token_refresh_sender.py`:

### 1. Stop Old Service

```bash
# If running manually
pkill -f token_refresh_service.py

# If using systemd
sudo systemctl stop token-refresh-service
```

### 2. Update Systemd Service (if applicable)

Edit `/etc/systemd/system/token-refresh-sender.service`:

```ini
[Service]
# OLD:
# ExecStart=/path/to/scripts/token_refresh_service.py

# NEW:
ExecStart=/path/to/scripts/run_token_sender.sh
```

### 3. Start New Service

```bash
# Manual
./scripts/run_token_sender.sh

# Systemd
sudo systemctl daemon-reload
sudo systemctl start token-refresh-sender
```

### 4. Monitor Logs

```bash
# Old log file
tail -f logs/token_refresh.log

# New log file
tail -f logs/token_refresh_sender.log
```

## Running with UV

The new script includes a wrapper for uv:

```bash
# Direct invocation
uv run python scripts/token_refresh_sender.py

# Using wrapper script
./scripts/run_token_sender.sh
```

## Socket Communication Protocol

Both scripts use the same socket protocol, so they're compatible with the C++ bot:

**Socket Path**: `/tmp/bigbrother_token.sock`

**Message Format**:
```json
{
    "access_token": "...",
    "refresh_token": "...",
    "expires_at": 1762984855
}
```

**Response**: `"OK"` on success

## Conclusion

The new `token_refresh_sender.py` is a production-ready version with:
- Better error handling and retry logic
- Professional logging with levels
- Type hints for code quality
- Comprehensive validation
- Extensive documentation

The original `token_refresh_service.py` remains as a simpler reference implementation.

Both are compatible with the C++ token manager's socket receiver (`src/schwab_api/token_manager.cpp`).
