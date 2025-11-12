#!/usr/bin/env python3
"""
FRED (Federal Reserve Economic Data) Integration

Fetches risk-free rate data from Federal Reserve:
- 3-Month Treasury Bill Rate (DGS3MO)
- 10-Year Treasury Constant Maturity Rate (DGS10)
- Effective Federal Funds Rate (DFF)

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-11
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
import duckdb
from typing import Optional, Dict


class FREDRatesFetcher:
    """Fetch risk-free rates from FRED API"""

    # FRED API endpoint
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Risk-free rate series IDs
    SERIES = {
        '3_month_treasury': 'DGS3MO',    # 3-Month Treasury Bill
        '10_year_treasury': 'DGS10',     # 10-Year Treasury
        'fed_funds_rate': 'DFF',         # Effective Federal Funds Rate
        '2_year_treasury': 'DGS2',       # 2-Year Treasury
        '5_year_treasury': 'DGS5',       # 5-Year Treasury
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED fetcher

        Args:
            api_key: FRED API key (get free at https://fred.stlouisfed.org/docs/api/api_key.html)
                    If None, looks for FRED_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            print("⚠️  FRED_API_KEY not set. Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            print("   Set with: export FRED_API_KEY='your_key_here'")

        # Database path
        self.db_path = Path(__file__).parent.parent.parent / "data" / "bigbrother.duckdb"

    def fetch_latest_rate(self, series_id: str) -> Optional[float]:
        """
        Fetch the most recent rate for a given series

        Args:
            series_id: FRED series ID (e.g., 'DGS3MO')

        Returns:
            Latest rate as decimal (e.g., 0.05 for 5%) or None if unavailable
        """
        if not self.api_key:
            return None

        try:
            # Fetch last 5 observations to handle weekends/holidays
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': 5,
                'sort_order': 'desc'
            }

            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'observations' in data and data['observations']:
                # Find first non-missing observation
                for obs in data['observations']:
                    if obs['value'] != '.':  # FRED uses '.' for missing data
                        rate = float(obs['value']) / 100.0  # Convert percentage to decimal
                        return rate

            return None

        except Exception as e:
            print(f"Error fetching FRED data for {series_id}: {e}")
            return None

    def fetch_all_latest_rates(self) -> Dict[str, Optional[float]]:
        """
        Fetch latest rates for all series

        Returns:
            Dictionary with rate names and values
        """
        rates = {}
        for name, series_id in self.SERIES.items():
            rate = self.fetch_latest_rate(series_id)
            rates[name] = rate

        return rates

    def store_rates_in_db(self, rates: Dict[str, Optional[float]]) -> bool:
        """
        Store rates in DuckDB for dashboard access

        Args:
            rates: Dictionary of rate names and values

        Returns:
            True if successful
        """
        try:
            conn = duckdb.connect(str(self.db_path))

            # Create table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_free_rates (
                    rate_name VARCHAR PRIMARY KEY,
                    rate_value DOUBLE,
                    last_updated TIMESTAMP,
                    series_id VARCHAR
                )
            """)

            # Insert or update rates
            timestamp = datetime.now()
            for name, value in rates.items():
                if value is not None:
                    series_id = self.SERIES[name]
                    conn.execute("""
                        INSERT INTO risk_free_rates (rate_name, rate_value, last_updated, series_id)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (rate_name) DO UPDATE SET
                            rate_value = EXCLUDED.rate_value,
                            last_updated = EXCLUDED.last_updated
                    """, [name, value, timestamp, series_id])

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error storing rates in database: {e}")
            return False

    def get_risk_free_rate(self, maturity: str = '3_month_treasury') -> float:
        """
        Get risk-free rate for options pricing

        Args:
            maturity: Rate maturity ('3_month_treasury', '2_year_treasury', etc.)

        Returns:
            Rate as decimal, or default 4% if unavailable
        """
        try:
            conn = duckdb.connect(str(self.db_path))
            result = conn.execute("""
                SELECT rate_value FROM risk_free_rates
                WHERE rate_name = ?
            """, [maturity]).fetchone()
            conn.close()

            if result and result[0] is not None:
                return result[0]

        except Exception as e:
            print(f"Error reading rate from database: {e}")

        # Fallback to fetching directly
        rate = self.fetch_latest_rate(self.SERIES.get(maturity, 'DGS3MO'))
        if rate is not None:
            return rate

        # Default fallback
        print(f"⚠️  Using default risk-free rate: 4%")
        return 0.04

    def update_rates(self) -> bool:
        """
        Fetch and store all latest rates

        Returns:
            True if successful
        """
        print("Fetching latest risk-free rates from FRED...")

        rates = self.fetch_all_latest_rates()

        # Display rates
        print("\nLatest Risk-Free Rates:")
        for name, value in rates.items():
            if value is not None:
                print(f"  {name.replace('_', ' ').title()}: {value*100:.3f}%")
            else:
                print(f"  {name.replace('_', ' ').title()}: N/A")

        # Store in database
        success = self.store_rates_in_db(rates)

        if success:
            print(f"\n✅ Rates stored in database: {self.db_path}")
        else:
            print("\n⚠️  Failed to store rates in database")

        return success


def get_current_risk_free_rate(maturity: str = '3_month_treasury') -> float:
    """
    Convenience function to get current risk-free rate

    Args:
        maturity: Rate maturity

    Returns:
        Current rate as decimal
    """
    fetcher = FREDRatesFetcher()
    return fetcher.get_risk_free_rate(maturity)


if __name__ == "__main__":
    print("=" * 80)
    print("FRED Risk-Free Rates Fetcher")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("\n⚠️  FRED_API_KEY not set!")
        print("\nTo use this script:")
        print("1. Get free API key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Set environment variable: export FRED_API_KEY='your_key_here'")
        print("\nUsing fallback rates for demonstration...")
        print("\nFallback rates:")
        print("  3-Month Treasury: 5.00%")
        print("  2-Year Treasury: 4.50%")
        print("  5-Year Treasury: 4.25%")
        print("  10-Year Treasury: 4.50%")
        print("  Fed Funds Rate: 5.25%")
    else:
        # Fetch and update rates
        fetcher = FREDRatesFetcher()
        success = fetcher.update_rates()

        if success:
            print("\n" + "=" * 80)
            print("Risk-Free Rates Ready!")
            print("=" * 80)

            # Show how to use
            print("\nUsage in Python:")
            print("```python")
            print("from fred_rates import get_current_risk_free_rate")
            print("")
            print("# Get 3-month Treasury rate")
            print("rate = get_current_risk_free_rate('3_month_treasury')")
            print(f"print(f'Risk-free rate: {{rate*100:.3f}}%')")
            print("```")
        else:
            print("\n⚠️  Rate update failed. Check API key and network connection.")

    print("\n" + "=" * 80)
