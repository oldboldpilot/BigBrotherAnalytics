# Token Refresh Sender

## Overview

The `token_refresh_sender.py` script implements the token sender side of the socket-based token refresh solution for the BigBrother Analytics trading bot.

## Features

- Reads Schwab API OAuth tokens from `configs/schwab_tokens.json`
- Automatically refreshes tokens every 25 minutes using Schwab's OAuth refresh_token grant
- Updates the token file with newly refreshed tokens
- Sends refreshed tokens to the C++ trading bot via Unix domain socket (`/tmp/bigbrother_token.sock`)
- Graceful error handling with retry logic for socket connections
- Compatible with running alongside the dashboard
- Comprehensive logging to both console and file

## Requirements

- Python 3.8+
- Required packages: `requests`, `pyyaml` (included in project dependencies)
- Valid Schwab OAuth tokens in `configs/schwab_tokens.json`
- Schwab app credentials in `configs/schwab_app_config.yaml`

## Usage

### Basic Usage

```bash
# Run the token refresh sender
./scripts/token_refresh_sender.py
```

### Running as Background Service

```bash
# Run in background
nohup ./scripts/token_refresh_sender.py > /dev/null 2>&1 &

# Or use systemd (recommended for production)
sudo systemctl start token-refresh-sender
```

### Stopping the Service

```bash
# If running in foreground
Ctrl+C

# If running in background
pkill -f token_refresh_sender.py

# If using systemd
sudo systemctl stop token-refresh-sender
```

## Configuration

### Token File Structure (`configs/schwab_tokens.json`)

```json
{
    "creation_timestamp": 1762983055,
    "token": {
        "expires_in": 1800,
        "token_type": "Bearer",
        "scope": "api",
        "refresh_token": "your_refresh_token",
        "access_token": "your_access_token",
        "expires_at": 1762984855
    }
}
```

### App Config Structure (`configs/schwab_app_config.yaml`)

```yaml
app_name: DataTradingApp
app_key: your_app_key
app_secret: your_app_secret
callback_url: https://127.0.0.1:8182
```

## Architecture

### Token Refresh Flow

1. **Initial Load**: Script loads current tokens from `configs/schwab_tokens.json`
2. **Refresh Request**: Every 25 minutes, sends refresh_token grant request to Schwab API
3. **Update File**: Writes new access_token and refresh_token to JSON file
4. **Socket Send**: Sends updated tokens to C++ bot via Unix domain socket
5. **Retry Logic**: If bot is not running, logs warning and continues (non-blocking)

### Socket Communication

- **Socket Path**: `/tmp/bigbrother_token.sock`
- **Protocol**: Unix domain socket (SOCK_STREAM)
- **Message Format**: JSON with `access_token`, `refresh_token`, `expires_at`
- **Acknowledgment**: Bot responds with "OK" on success
- **Retries**: 3 attempts with 5-second delays between retries

### C++ Bot Integration

The C++ trading bot (`token_manager.cpp`) runs a socket server that:
- Listens on `/tmp/bigbrother_token.sock`
- Receives token updates from this Python script
- Updates its in-memory OAuth config
- Sends "OK" acknowledgment

## Logging

### Log File Location

```
logs/token_refresh_sender.log
```

### Log Format

```
[2025-11-13 08:44:23] INFO: Token Refresh Sender Started
[2025-11-13 08:44:23] INFO: Refreshing OAuth token...
[2025-11-13 08:44:24] INFO: Token refreshed successfully (expires in 30 minutes)
[2025-11-13 08:44:24] INFO: Token successfully sent to trading bot
[2025-11-13 08:44:24] INFO: Next refresh at 2025-11-13 09:09:24 (25 minutes)
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Token file not found` | Missing `configs/schwab_tokens.json` | Run OAuth flow to generate initial tokens |
| `App config file not found` | Missing `configs/schwab_app_config.yaml` | Create config file with Schwab credentials |
| `Bot socket not found` | Trading bot not running | Start bot or ignore (script continues) |
| `Token refresh failed: HTTP 401` | Refresh token expired | Re-authenticate via OAuth flow |
| `Connection refused` | Bot starting up | Script auto-retries 3 times with delay |

### Graceful Degradation

- If the C++ bot is not running, the script logs a warning and continues
- Token file is still updated, so bot can reload tokens when it starts
- Non-blocking design ensures compatibility with dashboard-only deployments

## Integration with Dashboard

This script is designed to run alongside the dashboard without conflicts:

```bash
# Terminal 1: Run dashboard
python dashboard/app.py

# Terminal 2: Run token refresh sender
./scripts/token_refresh_sender.py

# Terminal 3: Run C++ trading bot (optional)
./build/bigbrother_trading_bot
```

## Systemd Service (Production Deployment)

Create `/etc/systemd/system/token-refresh-sender.service`:

```ini
[Unit]
Description=BigBrother Token Refresh Sender
After=network.target

[Service]
Type=simple
User=muyiwa
WorkingDirectory=/home/muyiwa/Development/BigBrotherAnalytics
ExecStart=/home/muyiwa/Development/BigBrotherAnalytics/scripts/token_refresh_sender.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable token-refresh-sender
sudo systemctl start token-refresh-sender
sudo systemctl status token-refresh-sender
```

## Monitoring

### Check Service Status

```bash
# View recent logs
tail -f logs/token_refresh_sender.log

# Check if running
ps aux | grep token_refresh_sender

# If using systemd
sudo systemctl status token-refresh-sender
journalctl -u token-refresh-sender -f
```

### Health Checks

The script logs every refresh cycle, so monitor for:
- Regular refresh cycles (every 25 minutes)
- Successful token refreshes
- Socket connection status

## Security Considerations

1. **File Permissions**: Ensure `configs/schwab_tokens.json` and `configs/schwab_app_config.yaml` are readable only by the service user
2. **Socket Permissions**: Socket is created in `/tmp` with default permissions
3. **Secrets**: Never commit token files or app config to version control
4. **Logging**: Logs do not contain sensitive token data (only truncated previews in debug mode)

## Troubleshooting

### Debug Mode

For verbose logging, modify the script's logging level:

```python
# In _setup_logging() method
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    ...
)
```

### Manual Token Refresh Test

```python
# Test token refresh without waiting 25 minutes
python3 << EOF
import sys
sys.path.insert(0, 'scripts')
from token_refresh_sender import TokenRefreshSender

service = TokenRefreshSender()
token_data = service.refresh_token()
print(f"Token refresh result: {token_data is not None}")
EOF
```

### Socket Communication Test

```bash
# Check if socket exists
ls -la /tmp/bigbrother_token.sock

# Test socket connection
python3 -c "import socket; s=socket.socket(socket.AF_UNIX); s.connect('/tmp/bigbrother_token.sock'); print('Connected')"
```

## Differences from Original `token_refresh_service.py`

This script (`token_refresh_sender.py`) is an improved version with:

1. Better error handling and retry logic
2. Enhanced logging with log levels (INFO, WARN, ERROR)
3. More detailed documentation and comments
4. Graceful shutdown with signal handlers
5. Non-blocking socket operations (continues even if bot isn't running)
6. Comprehensive type hints for better code quality
7. Structured logging to both file and console

## See Also

- `src/schwab_api/token_manager.cpp` - C++ bot's socket receiver implementation
- `scripts/schwab_oauth_server.py` - Initial OAuth flow to obtain tokens
- `docs/OAUTH_AUTO_REFRESH_FEATURE.md` - OAuth architecture documentation
