#!/usr/bin/env python3
"""
Get Current Risk-Free Rate from FRED

Uses 10-Year Treasury Constant Maturity Rate (DGS10) as the risk-free rate.
Alternative: 3-month T-Bill (DTB3) for short-term options.

Data Source: Federal Reserve Economic Data (FRED) - FREE
https://fred.stlouisfed.org/series/DGS10
https://fred.stlouisfed.org/series/DTB3

Usage:
    uv run python scripts/get_risk_free_rate.py
    uv run python scripts/get_risk_free_rate.py --tenor 3m
    uv run python scripts/get_risk_free_rate.py --date 2024-01-01
"""

import argparse
from fredapi import Fred
import yaml
from pathlib import Path
from datetime import datetime

def load_fred_api():
    """Load FRED API key from config."""
    api_keys_file = Path("configs/api_keys.yaml")

    with open(api_keys_file) as f:
        keys = yaml.safe_load(f)
        api_key = keys.get('fred', {}).get('api_key')

    return Fred(api_key=api_key)

def get_risk_free_rate(tenor='10y', date=None):
    """
    Get risk-free rate for given tenor.

    Tenors:
    - 3m: 3-Month Treasury Bill (DTB3)
    - 6m: 6-Month Treasury Bill (DTB6)
    - 1y: 1-Year Treasury (DGS1)
    - 2y: 2-Year Treasury (DGS2)
    - 5y: 5-Year Treasury (DGS5)
    - 10y: 10-Year Treasury (DGS10) - DEFAULT
    - 30y: 30-Year Treasury (DGS30)

    Returns: Risk-free rate as decimal (e.g., 0.0450 for 4.50%)
    """
    fred = load_fred_api()

    # Map tenor to FRED series
    series_map = {
        '3m': 'DTB3',
        '6m': 'DTB6',
        '1y': 'DGS1',
        '2y': 'DGS2',
        '5y': 'DGS5',
        '10y': 'DGS10',
        '30y': 'DGS30'
    }

    series_id = series_map.get(tenor, 'DGS10')

    print(f"📊 Fetching {tenor.upper()} Treasury Rate ({series_id})...")

    if date:
        # Get rate for specific date
        data = fred.get_series(series_id, observation_start=date, observation_end=date)
        if data.empty:
            print(f"⚠️  No data for {date}, using latest...")
            data = fred.get_series(series_id)
    else:
        # Get latest rate
        data = fred.get_series(series_id)

    latest_rate = data.iloc[-1]
    latest_date = data.index[-1]

    # Convert to decimal (FRED gives percentages)
    rate_decimal = latest_rate / 100.0

    print(f"✅ {tenor.upper()} Treasury Rate as of {latest_date.date()}:")
    print(f"   Rate: {latest_rate:.3f}% ({rate_decimal:.6f} decimal)")
    print()

    return rate_decimal, latest_date

def get_term_structure():
    """
    Get full term structure of interest rates.

    Returns: Dictionary with all tenors
    """
    fred = load_fred_api()

    print("📊 Fetching Full Term Structure...")
    print()

    series = {
        '3M': 'DTB3',
        '6M': 'DTB6',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',
        '30Y': 'DGS30'
    }

    rates = {}

    for tenor, series_id in series.items():
        try:
            data = fred.get_series(series_id)
            rate = data.iloc[-1]
            date = data.index[-1]

            rates[tenor] = {
                'rate_pct': rate,
                'rate_decimal': rate / 100.0,
                'date': date
            }

            print(f"  {tenor:4s}: {rate:6.3f}%  ({date.date()})")
        except Exception as e:
            print(f"  {tenor:4s}: ⚠️  {e}")

    print()

    # Calculate yield curve slope (10Y - 2Y)
    if '10Y' in rates and '2Y' in rates:
        slope = rates['10Y']['rate_pct'] - rates['2Y']['rate_pct']
        print(f"📈 Yield Curve Slope (10Y - 2Y): {slope:+.3f}%")

        if slope < 0:
            print(f"   ⚠️  INVERTED YIELD CURVE - Recession signal!")
        elif slope < 0.50:
            print(f"   ⚠️  FLAT yield curve - Economic uncertainty")
        else:
            print(f"   ✅ Normal upward sloping curve")

    print()

    return rates

def export_to_cpp():
    """Generate C++ header with current risk-free rate."""
    rate, date = get_risk_free_rate('10y')

    cpp_content = f"""// Auto-generated: {datetime.now()}
// Risk-free rate from FRED as of {date.date()}
// Data source: https://fred.stlouisfed.org/series/DGS10

#pragma once

namespace bigbrother::constants {{

// 10-Year Treasury Rate (updated {date.date()})
constexpr double RISK_FREE_RATE = {rate:.6f};  // {rate * 100:.3f}%

// For different option expirations, use appropriate tenor:
constexpr double RISK_FREE_RATE_3M = {rate:.6f};   // Update with DTB3
constexpr double RISK_FREE_RATE_1Y = {rate:.6f};   // Update with DGS1
constexpr double RISK_FREE_RATE_10Y = {rate:.6f};  // DGS10

// Last updated
constexpr const char* RATE_DATE = "{date.date()}";

}} // namespace bigbrother::constants
"""

    output_path = Path("src/utils/risk_free_rate.hpp")
    output_path.write_text(cpp_content)

    print(f"✅ Generated C++ header: {output_path}")
    print(f"   Use: #include \"utils/risk_free_rate.hpp\"")
    print(f"   Then: bigbrother::constants::RISK_FREE_RATE")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get risk-free rate from FRED for options pricing"
    )
    parser.add_argument(
        '--tenor',
        choices=['3m', '6m', '1y', '2y', '5y', '10y', '30y'],
        default='10y',
        help='Treasury tenor to use as risk-free rate'
    )
    parser.add_argument(
        '--date',
        help='Get rate for specific date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--term-structure',
        action='store_true',
        help='Show full yield curve'
    )
    parser.add_argument(
        '--export-cpp',
        action='store_true',
        help='Export to C++ header file'
    )

    args = parser.parse_args()

    try:
        if args.term_structure:
            rates = get_term_structure()
        else:
            rate, date = get_risk_free_rate(args.tenor, args.date)

        if args.export_cpp:
            export_to_cpp()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
