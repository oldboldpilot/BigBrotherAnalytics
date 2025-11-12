#!/usr/bin/env python3
"""
Initialize FRED Rate Provider from api_keys.yaml

Loads FRED API key and initializes C++ FRED rate provider singleton.
Used by trading engine and dashboard for live risk-free rates.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-11
"""

import sys
import yaml
from pathlib import Path

# Add build directory to path for C++ bindings
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))

try:
    from fred_rates_py import FREDRatesFetcher, FREDConfig, RateSeries
    FRED_AVAILABLE = True
except ImportError:
    print("⚠️  FRED C++ bindings not available. Build fred_rates_py first.")
    FRED_AVAILABLE = False


def load_fred_api_key():
    """Load FRED API key from api_keys.yaml"""
    api_keys_path = Path(__file__).parent.parent / 'api_keys.yaml'

    if not api_keys_path.exists():
        print(f"❌ API keys file not found: {api_keys_path}")
        return None

    with open(api_keys_path) as f:
        keys = yaml.safe_load(f)

    fred_key = keys.get('fred_api_key')

    if not fred_key:
        print("❌ FRED API key not found in api_keys.yaml")
        return None

    return fred_key


def initialize_fred_provider():
    """Initialize FRED rate provider singleton"""
    if not FRED_AVAILABLE:
        return False

    api_key = load_fred_api_key()
    if not api_key:
        return False

    try:
        # Create configuration
        config = FREDConfig()
        config.api_key = api_key
        config.timeout_seconds = 10
        config.max_observations = 5

        # Create fetcher
        fetcher = FREDRatesFetcher(config)

        # Test fetch
        rate_data = fetcher.fetch_latest_rate(RateSeries.ThreeMonthTreasury)

        print("✅ FRED Rate Provider initialized successfully")
        print(f"   3-Month Treasury: {rate_data.rate_value * 100:.3f}%")
        print(f"   Series ID: {rate_data.series_id}")
        print(f"   Last Updated: {rate_data.last_updated}")

        return True

    except Exception as e:
        print(f"❌ Failed to initialize FRED provider: {e}")
        return False


def fetch_all_rates():
    """Fetch and display all available rates"""
    if not FRED_AVAILABLE:
        return

    api_key = load_fred_api_key()
    if not api_key:
        return

    try:
        config = FREDConfig()
        config.api_key = api_key

        fetcher = FREDRatesFetcher(config)
        rates = fetcher.fetch_all_rates()

        print("\n" + "=" * 60)
        print("FRED Risk-Free Rates (Live)")
        print("=" * 60)

        for series, data in rates.items():
            print(f"{data.series_name:30s}: {data.rate_value * 100:6.3f}%")

        print("=" * 60)

    except Exception as e:
        print(f"❌ Failed to fetch rates: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("FRED Rate Provider Initialization")
    print("=" * 60)
    print()

    # Initialize provider
    if initialize_fred_provider():
        # Fetch all rates
        fetch_all_rates()
    else:
        print("\n⚠️  FRED provider not initialized")
        print("   Make sure fred_rates_py is built and API key is configured")
