#!/usr/bin/env python3
"""
BigBrotherAnalytics: BLS Employment Data Collection
Bureau of Labor Statistics (BLS) API Integration

This module fetches employment data from the BLS API and stores it in DuckDB.

Data Sources:
- BLS Current Employment Statistics (CES)
- JOLTS (Job Openings and Labor Turnover Survey)
- Weekly Unemployment Insurance Claims
- Monthly Employment Situation Report (Nonfarm Payrolls)

API Documentation: https://www.bls.gov/developers/home.htm
Free API Key: https://data.bls.gov/registrationEngine/
Rate Limits: 500 queries per day (registered), 25 per day (unregistered)

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-08
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import duckdb
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BLSEmploymentCollector:
    """Collects employment data from Bureau of Labor Statistics API."""

    # BLS API Configuration
    BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    BLS_UNAUTH_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

    # CES Series IDs (Current Employment Statistics) - in thousands
    SECTOR_SERIES = {
        'total_nonfarm': 'CES0000000001',      # Total nonfarm employment
        'total_private': 'CES0500000001',      # Total private employment
        'mining_logging': 'CES1000000001',     # Energy/Materials sector
        'construction': 'CES2000000001',       # Industrials sector
        'manufacturing': 'CES3000000001',      # Industrials/Materials sector
        'durable_goods': 'CES3100000001',      # Manufacturing - durable
        'nondurable_goods': 'CES3200000001',   # Manufacturing - nondurable
        'trade_transport_utilities': 'CES4000000001',  # Multiple sectors
        'wholesale_trade': 'CES4142000001',    # Materials distribution
        'retail_trade': 'CES4200000001',       # Consumer Discretionary
        'transport_warehousing': 'CES4300000001',  # Industrials
        'utilities': 'CES4422000001',          # Utilities sector
        'information': 'CES5000000001',        # Technology/Communication
        'financial_activities': 'CES5500000001',   # Financials sector
        'professional_business': 'CES6000000001',  # IT Services
        'education_health': 'CES6500000001',   # Health Care sector
        'leisure_hospitality': 'CES7000000001',    # Consumer Discretionary
        'other_services': 'CES8000000001',     # Various
        'government': 'CES9000000001',         # Government employment
    }

    # Unemployment Insurance Claims Series
    CLAIMS_SERIES = {
        'initial_claims': 'ICSA',          # Initial jobless claims (weekly)
        'continued_claims': 'CCSA',        # Continued claims (weekly)
        'insured_unemployment': 'IURNSA',  # Insured unemployment rate
    }

    # JOLTS Series (Job Openings and Labor Turnover Survey)
    JOLTS_SERIES = {
        'job_openings': 'JTS00000000JOL',      # Total job openings
        'hires': 'JTS00000000HIL',             # Total hires
        'total_separations': 'JTS00000000TSL', # Total separations
        'quits': 'JTS00000000QUL',             # Voluntary quits
        'layoffs': 'JTS00000000LDL',           # Layoffs & discharges
    }

    # Sector ID mapping (BLS series -> GICS sector_id)
    SECTOR_MAPPING = {
        'mining_logging': [1, 2],          # Energy, Materials
        'construction': [3],                # Industrials
        'manufacturing': [2, 3],            # Materials, Industrials
        'retail_trade': [4],                # Consumer Discretionary
        'utilities': [10],                  # Utilities
        'information': [8, 9],              # Technology, Communication
        'financial_activities': [7],        # Financials
        'professional_business': [8],       # Information Technology
        'education_health': [6],            # Health Care
        'leisure_hospitality': [4],         # Consumer Discretionary
    }

    def __init__(self, api_key: Optional[str] = None, db_path: str = "data/bigbrother.duckdb"):
        """
        Initialize BLS Employment Collector.

        Args:
            api_key: BLS API key (optional, reads from api_keys.yaml or env BLS_API_KEY)
            db_path: Path to DuckDB database
        """
        # Try to load from api_keys.yaml first, then env variable
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._load_api_key_from_config() or os.getenv('BLS_API_KEY')
        self.db_path = Path(db_path)

        # Use v2 API if we have a key, otherwise v1
        if self.api_key:
            self.base_url = self.BLS_BASE_URL
            logger.info("Using BLS API v2 (authenticated)")
        else:
            self.base_url = self.BLS_UNAUTH_URL
            logger.warning("Using BLS API v1 (unauthenticated) - limited to 25 queries/day")

        # Ensure database exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True)

    def _load_api_key_from_config(self) -> Optional[str]:
        """Load BLS API key from api_keys.yaml configuration file."""
        try:
            import yaml
            config_path = Path(__file__).parent.parent.parent / 'api_keys.yaml'

            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('bls_api_key')
        except Exception as e:
            logger.debug(f"Could not load API key from config: {e}")

        return None

    def fetch_series(
        self,
        series_ids: List[str],
        start_year: int,
        end_year: int,
        catalog: bool = False
    ) -> Dict:
        """
        Fetch time series data from BLS API.

        Args:
            series_ids: List of BLS series IDs
            start_year: Start year (YYYY)
            end_year: End year (YYYY)
            catalog: Include catalog metadata

        Returns:
            JSON response from BLS API
        """
        payload = {
            'seriesid': series_ids,
            'startyear': str(start_year),
            'endyear': str(end_year),
            'catalog': catalog,
        }

        if self.api_key:
            payload['registrationkey'] = self.api_key

        try:
            response = requests.post(self.base_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'REQUEST_SUCCEEDED':
                logger.error(f"BLS API request failed: {data.get('message')}")
                return {}

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching BLS data: {e}")
            return {}

    def parse_employment_data(self, series_data: Dict) -> List[Tuple]:
        """
        Parse BLS series data into database rows.

        Args:
            series_data: BLS API response for a series

        Returns:
            List of tuples (date, value, series_id)
        """
        rows = []
        series_id = series_data.get('seriesID')

        for item in series_data.get('data', []):
            year = item.get('year')
            period = item.get('period')  # M01, M02, ... M12 for monthly
            value = item.get('value')

            # Parse period (M01 = January, M13 = Annual average, skip M13)
            if not period.startswith('M') or period == 'M13':
                continue

            month = int(period[1:])
            date = f"{year}-{month:02d}-01"

            try:
                employment = int(float(value))  # BLS reports in thousands
                rows.append((date, employment, series_id))
            except (ValueError, TypeError):
                logger.warning(f"Could not parse value '{value}' for {series_id} on {date}")
                continue

        return rows

    def collect_sector_employment(self, years: int = 5) -> None:
        """
        Collect employment data for all sectors.

        Args:
            years: Number of years of historical data to fetch
        """
        end_year = datetime.now().year
        start_year = end_year - years + 1

        logger.info(f"Collecting sector employment data from {start_year} to {end_year}")

        # Fetch data in batches (BLS allows up to 50 series per request)
        batch_size = 50
        series_list = list(self.SECTOR_SERIES.items())

        all_data = []

        for i in range(0, len(series_list), batch_size):
            batch = series_list[i:i+batch_size]
            series_ids = [series_id for _, series_id in batch]

            logger.info(f"Fetching batch {i//batch_size + 1}: {len(series_ids)} series")

            data = self.fetch_series(series_ids, start_year, end_year)

            if not data or 'Results' not in data:
                logger.error(f"No data returned for batch {i//batch_size + 1}")
                continue

            # Parse each series
            for series_data in data['Results'].get('series', []):
                rows = self.parse_employment_data(series_data)
                all_data.extend(rows)

        # Save to database
        if all_data:
            self._save_sector_employment(all_data)
            logger.info(f"Saved {len(all_data)} employment records to database")
        else:
            logger.warning("No employment data collected")

    def collect_jobless_claims(self, weeks: int = 52) -> None:
        """
        Collect weekly jobless claims data.

        Args:
            weeks: Number of weeks of data to fetch
        """
        # BLS reports weekly data, fetch last 2 years to be safe
        end_year = datetime.now().year
        start_year = end_year - 2

        logger.info(f"Collecting jobless claims data")

        series_ids = list(self.CLAIMS_SERIES.values())
        data = self.fetch_series(series_ids, start_year, end_year)

        if not data or 'Results' not in data:
            logger.error("No jobless claims data returned")
            return

        # Parse and save
        all_rows = []
        for series_data in data['Results'].get('series', []):
            rows = self._parse_claims_data(series_data)
            all_rows.extend(rows)

        if all_rows:
            self._save_jobless_claims(all_rows)
            logger.info(f"Saved {len(all_rows)} jobless claims records")

    def _parse_claims_data(self, series_data: Dict) -> List[Tuple]:
        """Parse weekly claims data."""
        rows = []
        series_id = series_data.get('seriesID')

        for item in series_data.get('data', []):
            year = item.get('year')
            period = item.get('period')
            value = item.get('value')

            # Weekly data has periods like 'M01', 'M02', etc. for weeks
            # We need to calculate the actual date
            try:
                claims = int(float(value)) * 1000  # Convert from thousands
                # Simplified: use first day of month for now
                # TODO: Calculate exact week-ending date
                month = int(period[1:]) if period.startswith('M') else 1
                date = f"{year}-{month:02d}-01"
                rows.append((date, claims, series_id))
            except (ValueError, TypeError):
                continue

        return rows

    def _save_sector_employment(self, data: List[Tuple]) -> None:
        """Save sector employment data to database."""
        conn = duckdb.connect(str(self.db_path))

        # Create table if not exists (from schema file)
        # For now, simplified insert
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sector_employment_raw (
                    report_date DATE,
                    employment_count INTEGER,
                    series_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert data
            conn.executemany(
                "INSERT INTO sector_employment_raw VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                data
            )

            conn.commit()
            logger.info(f"Inserted {len(data)} records into sector_employment_raw")

        except Exception as e:
            logger.error(f"Error saving employment data: {e}")
        finally:
            conn.close()

    def _save_jobless_claims(self, data: List[Tuple]) -> None:
        """Save jobless claims data to database."""
        conn = duckdb.connect(str(self.db_path))

        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobless_claims_raw (
                    week_ending DATE,
                    claims INTEGER,
                    series_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.executemany(
                "INSERT INTO jobless_claims_raw VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                data
            )

            conn.commit()
            logger.info(f"Inserted {len(data)} records into jobless_claims_raw")

        except Exception as e:
            logger.error(f"Error saving claims data: {e}")
        finally:
            conn.close()

    def get_latest_employment_by_sector(self) -> Dict[str, int]:
        """
        Get the latest employment figures for each sector.

        Returns:
            Dictionary mapping sector name to employment count
        """
        conn = duckdb.connect(str(self.db_path))

        try:
            result = conn.execute("""
                SELECT series_id, report_date, employment_count
                FROM sector_employment_raw
                WHERE report_date = (SELECT MAX(report_date) FROM sector_employment_raw)
                ORDER BY series_id
            """).fetchall()

            return {row[0]: row[2] for row in result}

        except Exception as e:
            logger.error(f"Error querying employment data: {e}")
            return {}
        finally:
            conn.close()


def main():
    """Main entry point for BLS data collection."""
    # Get API key from environment
    api_key = os.getenv('BLS_API_KEY')

    if not api_key:
        logger.warning("BLS_API_KEY not found. Using unauthenticated API (25 queries/day limit)")
        logger.info("Get a free API key at: https://data.bls.gov/registrationEngine/")

    # Initialize collector
    collector = BLSEmploymentCollector(api_key=api_key)

    # Collect data
    logger.info("=" * 80)
    logger.info("Starting BLS Employment Data Collection")
    logger.info("=" * 80)

    # Collect 5 years of sector employment data
    collector.collect_sector_employment(years=5)

    # Collect 1 year of weekly jobless claims
    collector.collect_jobless_claims(weeks=52)

    # Display summary
    latest = collector.get_latest_employment_by_sector()
    if latest:
        logger.info("\n" + "=" * 80)
        logger.info("Latest Employment Figures (in thousands):")
        logger.info("=" * 80)
        for series_id, count in sorted(latest.items()):
            logger.info(f"  {series_id}: {count:,}")

    logger.info("\n" + "=" * 80)
    logger.info("BLS Data Collection Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
