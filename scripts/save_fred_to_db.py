#!/usr/bin/env python3
"""
Save FRED Economic Data to DuckDB

Downloads and stores economic indicators for trading analysis.
"""

from fredapi import Fred
import yaml
from pathlib import Path
import duckdb
import pandas as pd

def save_fred_data():
    """Download and save FRED data to DuckDB."""

    # Load API key
    with open('configs/api_keys.yaml') as f:
        keys = yaml.safe_load(f)
        api_key = keys['fred']['api_key']

    fred = Fred(api_key=api_key)

    # Connect to database
    conn = duckdb.connect('data/bigbrother.duckdb')

    # Create economic_data table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS economic_data (
            series_id VARCHAR,
            date DATE,
            value DOUBLE,
            series_name VARCHAR,
            PRIMARY KEY (series_id, date)
        )
    """)

    indicators = {
        'FEDFUNDS': 'Federal Funds Rate',
        'DGS10': '10-Year Treasury Yield',
        'DGS2': '2-Year Treasury Yield',
        'DGS3MO': '3-Month Treasury',
        'DGS1': '1-Year Treasury',
        'DGS5': '5-Year Treasury',
        'DGS30': '30-Year Treasury',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'CPI Inflation',
        'GDP': 'GDP',
        'VIXCLS': 'VIX (Volatility Index)'
    }

    print('📊 Downloading and storing FRED economic data...\n')

    for series_id, name in indicators.items():
        print(f'  {name:30s}', end=' ')
        try:
            # Get data (5 years)
            df = fred.get_series(series_id, observation_start='2020-01-01')

            if df.empty:
                print('❌ No data')
                continue

            # Convert to DataFrame for DuckDB
            df = df.reset_index()
            df.columns = ['date', 'value']
            df['series_id'] = series_id
            df['series_name'] = name

            # Insert into database
            conn.execute("""
                INSERT OR REPLACE INTO economic_data
                SELECT series_id, date::DATE, value, series_name
                FROM df
            """)

            print(f'✅ {len(df):,} observations')

        except Exception as e:
            print(f'❌ {e}')

    # Show summary
    result = conn.execute("""
        SELECT
            series_name,
            COUNT(*) as observations,
            MIN(date) as start_date,
            MAX(date) as end_date,
            ROUND(AVG(value), 3) as avg_value,
            ROUND(value, 3) as latest_value
        FROM economic_data
        GROUP BY series_name, value
        ORDER BY series_name
    """).df()

    print('\n' + '='*80)
    print('📊 Economic Data Summary:')
    print('='*80)
    print(result.to_string(index=False))
    print()

    conn.close()
    print('✅ FRED data saved to: data/bigbrother.duckdb')
    print()

if __name__ == "__main__":
    save_fred_data()
