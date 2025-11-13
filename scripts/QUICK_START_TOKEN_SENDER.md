# Token Refresh Sender - Quick Start Guide

## Prerequisites

1. Valid Schwab OAuth tokens exist in `configs/schwab_tokens.json`
2. Schwab app credentials exist in `configs/schwab_app_config.yaml`
3. Python dependencies installed via `uv sync`

## Quick Start

### Option 1: Using the Wrapper Script (Recommended)

```bash
./scripts/run_token_sender.sh
```

### Option 2: Direct Execution with UV

```bash
uv run python scripts/token_refresh_sender.py
```

### Option 3: Direct Execution (requires proper environment)

```bash
./scripts/token_refresh_sender.py
```

## What Happens When You Run It

1. Script validates config files exist
2. Loads Schwab app credentials from `configs/schwab_app_config.yaml`
3. Loads current tokens from `configs/schwab_tokens.json`
4. Immediately refreshes the token
5. Updates the token file with new tokens
6. Attempts to send tokens to C++ bot (if running)
7. Sleeps for 25 minutes
8. Repeats steps 4-7 indefinitely

## Expected Output

```
[2025-11-13 08:44:23] INFO: ======================================================================
[2025-11-13 08:44:23] INFO: Token Refresh Sender Initialized
[2025-11-13 08:44:23] INFO: ======================================================================
[2025-11-13 08:44:23] INFO: ======================================================================
[2025-11-13 08:44:23] INFO: Token Refresh Sender Started
[2025-11-13 08:44:23] INFO: ======================================================================
[2025-11-13 08:44:23] INFO: Refresh interval: 25 minutes
[2025-11-13 08:44:23] INFO: Socket path: /tmp/bigbrother_token.sock
[2025-11-13 08:44:23] INFO: Token file: configs/schwab_tokens.json
[2025-11-13 08:44:23] INFO: App config: configs/schwab_app_config.yaml
[2025-11-13 08:44:23] INFO:
[2025-11-13 08:44:23] INFO: --- Refresh Cycle 1 ---
[2025-11-13 08:44:23] INFO: Refreshing OAuth token...
[2025-11-13 08:44:24] INFO: Token refreshed successfully (expires in 30 minutes)
[2025-11-13 08:44:24] INFO: Token successfully sent to trading bot
[2025-11-13 08:44:24] INFO: Next refresh at 2025-11-13 09:09:24 (25 minutes)
```

## If C++ Bot Is Not Running

The script will continue running and log a warning:

```
[2025-11-13 08:44:24] WARNING: Bot socket not found - trading bot may not be running
[2025-11-13 08:44:24] INFO: Next refresh at 2025-11-13 09:09:24 (25 minutes)
```

This is normal and expected. The script will:
- Continue refreshing tokens every 25 minutes
- Update the token file so the bot can read it when it starts
- Work perfectly with dashboard-only deployments

## Running in Background

### Method 1: Using nohup

```bash
nohup ./scripts/run_token_sender.sh > /dev/null 2>&1 &
```

Check it's running:
```bash
ps aux | grep token_refresh_sender
```

Stop it:
```bash
pkill -f token_refresh_sender.py
```

### Method 2: Using screen

```bash
screen -S token-sender
./scripts/run_token_sender.sh

# Detach with: Ctrl+A, then D
# Re-attach with: screen -r token-sender
```

### Method 3: Using systemd (Production)

See `scripts/README_TOKEN_SENDER.md` for full systemd setup.

## Stopping the Service

### Graceful Shutdown

Press `Ctrl+C` in the terminal running the script.

Expected output:
```
^C
[2025-11-13 09:15:32] INFO:
[2025-11-13 09:15:32] INFO: Received shutdown signal (2)
[2025-11-13 09:15:32] INFO: ======================================================================
[2025-11-13 09:15:32] INFO: Token Refresh Sender Stopped
[2025-11-13 09:15:32] INFO: ======================================================================
```

### Force Kill (if needed)

```bash
pkill -9 -f token_refresh_sender.py
```

## Monitoring

### View Live Logs

```bash
tail -f logs/token_refresh_sender.log
```

### Check Log File

```bash
cat logs/token_refresh_sender.log
```

### Verify Tokens Are Being Refreshed

```bash
# Check token file timestamp
ls -lh configs/schwab_tokens.json

# View token file (shows creation_timestamp)
cat configs/schwab_tokens.json | jq '.creation_timestamp'
```

## Common Issues

### Issue: "Token file not found"

**Solution**: Run OAuth flow to generate initial tokens:
```bash
./scripts/schwab_oauth_server.py
```

### Issue: "App config file not found"

**Solution**: Create `configs/schwab_app_config.yaml` with your Schwab credentials:
```yaml
app_name: DataTradingApp
app_key: your_app_key
app_secret: your_app_secret
callback_url: https://127.0.0.1:8182
```

### Issue: "Token refresh failed: HTTP 401"

**Cause**: Refresh token has expired (7-day lifetime)

**Solution**: Re-authenticate via OAuth flow:
```bash
./scripts/schwab_oauth_server.py
```

### Issue: "Bot socket not found"

This is **not an error** if you're running dashboard-only. The script will:
- Continue running
- Keep refreshing tokens
- Update the token file
- The C++ bot can read the file when it starts

## Integration with Other Services

### Running with Dashboard

```bash
# Terminal 1: Dashboard
python dashboard/app.py

# Terminal 2: Token Sender
./scripts/run_token_sender.sh

# Terminal 3 (optional): C++ Trading Bot
./build/bigbrother_trading_bot
```

### Running All Services Together

```bash
# Start token sender in background
./scripts/run_token_sender.sh &

# Start dashboard
python dashboard/app.py
```

## Testing

### Test Token Refresh Only (No Loop)

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, 'scripts')
from token_refresh_sender import TokenRefreshSender

# Initialize service
service = TokenRefreshSender()

# Refresh token once
print("Refreshing token...")
token_data = service.refresh_token()

if token_data:
    print("✅ Token refresh successful!")
    print(f"   Access token: {token_data['access_token'][:30]}...")
    print(f"   Expires at: {token_data['expires_at']}")
else:
    print("❌ Token refresh failed")
    sys.exit(1)
```

Save as `test_token_refresh.py` and run:
```bash
uv run python test_token_refresh.py
```

### Test Socket Communication

First, start the token sender in one terminal:
```bash
./scripts/run_token_sender.sh
```

Then verify the socket exists:
```bash
ls -la /tmp/bigbrother_token.sock
```

## Performance

- **Memory**: ~20-30 MB (Python process)
- **CPU**: Negligible (only active during 5-second refresh window)
- **Network**: One HTTPS request every 25 minutes to Schwab API
- **Disk I/O**: Minimal (writes token file every 25 minutes)

## Security

- Token file is updated atomically
- Logs do not contain full token values
- Socket permissions inherit from /tmp directory
- Uses HTTPS for Schwab API communication

## Next Steps

1. **Production Deployment**: Set up systemd service (see `scripts/README_TOKEN_SENDER.md`)
2. **Monitoring**: Set up log rotation and monitoring alerts
3. **Integration**: Connect C++ trading bot to receive tokens
4. **Automation**: Add to startup scripts

## Additional Documentation

- **Full Documentation**: `scripts/README_TOKEN_SENDER.md`
- **Comparison with Original**: `docs/TOKEN_SENDER_COMPARISON.md`
- **OAuth Architecture**: `docs/OAUTH_AUTO_REFRESH_FEATURE.md`
- **C++ Bot Integration**: `src/schwab_api/token_manager.cpp`
