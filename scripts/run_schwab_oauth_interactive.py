
"""
Schwab API OAuth Token Creation - THE ONLY WORKING METHOD

========================================================================
IMPORTANT: This is the ONLY reliable method to create OAuth tokens!
========================================================================

WHY OTHER METHODS FAIL:
- Manual copy-paste methods fail because Schwab authorization codes
  expire in 5-10 seconds (server-controlled, cannot be extended)
- By the time you copy the URL and paste it, the code has expired
- This script runs a local callback server that catches the OAuth
  redirect IMMEDIATELY (< 1 second), before expiration

REQUIREMENTS:
- Must be run in an interactive terminal (SSH session is fine)
- You need a web browser on the same machine or accessible via X11/VNC
- Port 8182 must be available

USAGE:
    uv run python run_oauth_interactive.py

The script will:
1. Start a local HTTPS callback server on port 8182
2. Open your browser to Schwab's login page
3. Wait for you to login and authorize (up to 500 seconds)
4. Capture the OAuth callback immediately
5. Save the token to configs/schwab_tokens.json
6. Test the API to verify it works

After successful authentication, you can use:
    uv run python test_schwab.py          # Basic test
    uv run python tests/test_real_data.py # Comprehensive test
"""

from schwab import auth
from pathlib import Path
import yaml
import sys

def load_config():
    config_path = Path('configs/schwab_app_config.yaml')
    if not config_path.exists():
        print(f"? Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate config
    if not isinstance(config, dict):
        print(f"? Invalid YAML format. Got type: {type(config)}")
        print(f"Content: {config}")
        sys.exit(1)
    
    required_keys = ['app_key', 'app_secret']
    for key in required_keys:
        if key not in config:
            print(f"? Missing required config key: {key}")
            sys.exit(1)
    
    return config

def main():
    print("=" * 80)
    print("SCHWAB API - INTERACTIVE OAUTH")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Start a local HTTPS server on port 8182")
    print("  2. Open your browser to Schwab login")
    print("  3. Capture the OAuth callback IMMEDIATELY (no expiration)")
    print("  4. Save the token")
    print()
    print("IMPORTANT: You must run this in an interactive terminal where you can")
    print("           see the output and press keys.")
    print()

    # Load config
    config = load_config()
    token_path = Path('configs/schwab_tokens.json')

    # Delete old token if exists
    if token_path.exists():
        print(f"Deleting old token: {token_path}")
        token_path.unlink()

    print()
    print("Starting OAuth flow with extended timeout (500 seconds)...")
    print(f"Using config: {config}" )

    try:
        # Use client_from_login_flow which runs the callback server
        client = auth.client_from_login_flow(
            config["app_key"],
            config["app_secret"],
            config.get("callback_url", "https://127.0.0.1:8182"),
            str(token_path),
            callback_timeout=500.0  # 8+ minutes
        )

        print()
        print("=" * 80)
        print("? SUCCESS!")
        print("=" * 80)
        print(f"Token saved to: {token_path}")
        print()

        # Test it immediately
        print("Testing API...")
        resp = client.get_quote('SPY')
        if resp.status_code == 200:
            data = resp.json()
            price = data['SPY']['quote']['lastPrice']
            print(f"? API works! SPY = ${price:.2f}")
            print()
            print("You're ready to use the Schwab API!")
            print("Test with: uv run python test_schwab.py")
            return 0
        else:
            print(f"?  API returned status: {resp.status_code}")
            return 1

    except KeyboardInterrupt:
        print()
        print("? Cancelled by user")
        return 1
    except Exception as e:
        print()
        print(f"? Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
