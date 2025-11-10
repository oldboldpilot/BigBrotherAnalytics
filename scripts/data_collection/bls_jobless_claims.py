#!/usr/bin/env python3
"""
BigBrotherAnalytics: FRED Jobless Claims Data Collection
Federal Reserve Economic Data (FRED) API Integration for Jobless Claims

This module fetches weekly jobless claims data from the FRED API for recession detection.
Implements spike detection logic to identify potential recession warning signals.

Data Sources:
- ICSA: Initial Claims (Seasonally Adjusted)
- CCSA: Continued Claims (Seasonally Adjusted)

API Documentation: https://fred.stlouisfed.org/docs/api/fred/
Free API Key: https://fred.stlouisfed.org/docs/api/api_key.html

Author: BigBrotherAnalytics - Agent 3
Date: 2025-11-10
"""

import os
import sys
import json
import logging
import yaml
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


class FREDJoblessClaimsCollector:
    """Collects weekly jobless claims data from Federal Reserve Economic Data API."""

    # FRED API Configuration
    FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Jobless Claims Series IDs
    CLAIMS_SERIES = {
        'initial_claims': 'ICSA',          # Initial jobless claims (weekly, seasonally adjusted)
        'continued_claims': 'CCSA',        # Continued claims (weekly, seasonally adjusted)
    }

    # Spike detection threshold (10% increase from 4-week average)
    SPIKE_THRESHOLD = 0.10

    def __init__(self, api_key: Optional[str] = None, db_path: str = "data/bigbrother.duckdb"):
        """
        Initialize FRED Jobless Claims Collector.

        Args:
            api_key: FRED API key (optional, reads from api_keys.yaml or env FRED_API_KEY)
            db_path: Path to DuckDB database
        """
        # Try to get API key from config file or environment
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._load_api_key_from_config() or os.getenv('FRED_API_KEY')

        self.db_path = Path(db_path)

        if not self.api_key:
            logger.error("FRED API key required! Get one at: https://fred.stlouisfed.org/docs/api/api_key.html")
            raise ValueError("FRED API key not found")

        logger.info("Using FRED API for jobless claims data")

        # Ensure database exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True)

    def _load_api_key_from_config(self) -> Optional[str]:
        """Load FRED API key from api_keys.yaml configuration file."""
        try:
            # Look in configs directory
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'api_keys.yaml'

            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('fred', {}).get('api_key')
        except Exception as e:
            logger.debug(f"Could not load API key from config: {e}")

        return None

    def fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Fetch time series data from FRED API.

        Args:
            series_id: FRED series ID (ICSA or CCSA)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of observations from FRED API
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'sort_order': 'asc'
        }

        try:
            response = requests.get(self.FRED_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'observations' not in data:
                logger.error(f"FRED API request failed for {series_id}")
                return []

            return data['observations']

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching FRED data: {e}")
            return []

    def parse_fred_observations(self, observations: List[Dict], series_id: str) -> Dict[str, int]:
        """
        Parse FRED observations into date->value dictionary.

        Args:
            observations: List of FRED observations
            series_id: Series ID (ICSA or CCSA)

        Returns:
            Dictionary mapping date to claims value
        """
        data = {}

        for obs in observations:
            date = obs.get('date')
            value = obs.get('value')

            # Skip if missing data (marked as '.')
            if not date or value == '.':
                continue

            try:
                # FRED values are already in actual counts
                claims_value = int(float(value))
                data[date] = claims_value
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse value '{value}' for {series_id} on {date}: {e}")
                continue

        return data

    def create_table(self) -> None:
        """Create jobless_claims table with proper schema."""
        conn = duckdb.connect(str(self.db_path))

        try:
            # Create sequence for IDs
            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS jobless_claims_seq START 1;
            """)

            # Create table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobless_claims (
                    id INTEGER PRIMARY KEY DEFAULT nextval('jobless_claims_seq'),
                    report_date DATE NOT NULL,
                    initial_claims INTEGER NOT NULL,
                    continued_claims INTEGER,
                    four_week_avg INTEGER,
                    spike_detected BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create index for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobless_claims_date
                ON jobless_claims(report_date DESC);
            """)

            conn.commit()
            logger.info("Created jobless_claims table successfully")

        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise
        finally:
            conn.close()

    def collect_claims_data(self, weeks: int = 52) -> None:
        """
        Collect weekly jobless claims data.

        Args:
            weeks: Number of weeks of data to fetch (default: 52 for 1 year)
        """
        # Calculate date range (52 weeks = ~365 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks)

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        logger.info(f"Collecting {weeks} weeks of jobless claims data ({start_date_str} to {end_date_str})")

        # Fetch initial claims (ICSA)
        logger.info("Fetching Initial Claims (ICSA)...")
        icsa_obs = self.fetch_series('ICSA', start_date_str, end_date_str)
        initial_claims_data = self.parse_fred_observations(icsa_obs, 'ICSA')

        # Fetch continued claims (CCSA)
        logger.info("Fetching Continued Claims (CCSA)...")
        ccsa_obs = self.fetch_series('CCSA', start_date_str, end_date_str)
        continued_claims_data = self.parse_fred_observations(ccsa_obs, 'CCSA')

        # Merge data by date
        all_dates = sorted(set(initial_claims_data.keys()) | set(continued_claims_data.keys()))

        logger.info(f"Retrieved {len(all_dates)} observations")

        # Take most recent weeks
        recent_dates = all_dates[-weeks:] if len(all_dates) > weeks else all_dates

        merged_records = []
        for date in recent_dates:
            initial = initial_claims_data.get(date)
            continued = continued_claims_data.get(date)

            if initial:  # Must have initial claims at minimum
                merged_records.append({
                    'report_date': date,
                    'initial_claims': initial,
                    'continued_claims': continued
                })

        if merged_records:
            self._save_claims_data(merged_records)
            logger.info(f"Saved {len(merged_records)} jobless claims records to database")
        else:
            logger.warning("No jobless claims data to save")

    def _save_claims_data(self, records: List[Dict]) -> None:
        """
        Save jobless claims data to database.

        Args:
            records: List of claims records with report_date, initial_claims, continued_claims
        """
        conn = duckdb.connect(str(self.db_path))

        try:
            # Clear existing data to avoid duplicates (for this prototype)
            conn.execute("DELETE FROM jobless_claims")

            # Insert new data
            for record in records:
                conn.execute("""
                    INSERT INTO jobless_claims (report_date, initial_claims, continued_claims)
                    VALUES (?, ?, ?)
                """, [
                    record['report_date'],
                    record['initial_claims'],
                    record.get('continued_claims')
                ])

            conn.commit()
            logger.info(f"Inserted {len(records)} records into jobless_claims table")

        except Exception as e:
            logger.error(f"Error saving claims data: {e}")
            raise
        finally:
            conn.close()

    def calculate_spike_detection(self) -> None:
        """
        Calculate 4-week moving average and detect spikes.

        A spike is detected when initial claims increase by more than 10%
        from the 4-week moving average.
        """
        conn = duckdb.connect(str(self.db_path))

        try:
            # Calculate 4-week moving average
            logger.info("Calculating 4-week moving averages...")

            conn.execute("""
                UPDATE jobless_claims
                SET four_week_avg = (
                    SELECT AVG(initial_claims)::INTEGER
                    FROM (
                        SELECT initial_claims
                        FROM jobless_claims AS jc2
                        WHERE jc2.report_date <= jobless_claims.report_date
                        ORDER BY jc2.report_date DESC
                        LIMIT 4
                    )
                )
            """)

            # Detect spikes (>10% increase from 4-week average)
            logger.info("Detecting spikes...")

            conn.execute(f"""
                UPDATE jobless_claims
                SET spike_detected = (
                    initial_claims > four_week_avg * (1 + {self.SPIKE_THRESHOLD})
                )
                WHERE four_week_avg IS NOT NULL
            """)

            conn.commit()

            # Count spikes detected
            spike_count = conn.execute("""
                SELECT COUNT(*) FROM jobless_claims WHERE spike_detected = TRUE
            """).fetchone()[0]

            logger.info(f"Spike detection complete. Found {spike_count} spikes.")

        except Exception as e:
            logger.error(f"Error calculating spike detection: {e}")
            raise
        finally:
            conn.close()

    def get_latest_claims(self, limit: int = 10) -> List[Tuple]:
        """
        Get latest jobless claims data.

        Args:
            limit: Number of recent records to return

        Returns:
            List of tuples with claims data
        """
        conn = duckdb.connect(str(self.db_path))

        try:
            result = conn.execute(f"""
                SELECT
                    report_date,
                    initial_claims,
                    continued_claims,
                    four_week_avg,
                    spike_detected
                FROM jobless_claims
                ORDER BY report_date DESC
                LIMIT {limit}
            """).fetchall()

            return result

        except Exception as e:
            logger.error(f"Error querying claims data: {e}")
            return []
        finally:
            conn.close()

    def get_spike_summary(self) -> Dict:
        """
        Get summary statistics about spike detection.

        Returns:
            Dictionary with spike statistics
        """
        conn = duckdb.connect(str(self.db_path))

        try:
            # Get overall stats
            stats = conn.execute("""
                SELECT
                    COUNT(*) as total_weeks,
                    SUM(CASE WHEN spike_detected THEN 1 ELSE 0 END) as spike_count,
                    AVG(initial_claims)::INTEGER as avg_initial_claims,
                    MAX(initial_claims) as max_initial_claims,
                    MIN(initial_claims) as min_initial_claims
                FROM jobless_claims
            """).fetchone()

            # Get most recent spike
            recent_spike = conn.execute("""
                SELECT report_date, initial_claims, four_week_avg
                FROM jobless_claims
                WHERE spike_detected = TRUE
                ORDER BY report_date DESC
                LIMIT 1
            """).fetchone()

            return {
                'total_weeks': stats[0],
                'spike_count': stats[1],
                'avg_initial_claims': stats[2],
                'max_initial_claims': stats[3],
                'min_initial_claims': stats[4],
                'most_recent_spike': recent_spike
            }

        except Exception as e:
            logger.error(f"Error getting spike summary: {e}")
            return {}
        finally:
            conn.close()


def main():
    """Main entry point for FRED jobless claims data collection."""

    logger.info("=" * 80)
    logger.info("BigBrotherAnalytics: FRED Jobless Claims Data Collection")
    logger.info("=" * 80)

    # Initialize collector (will load API key from config or environment)
    try:
        collector = FREDJoblessClaimsCollector()
    except ValueError as e:
        logger.error(f"Failed to initialize collector: {e}")
        logger.error("Please set FRED_API_KEY in environment or configs/api_keys.yaml")
        logger.error("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    # Step 1: Create table
    logger.info("\nStep 1: Creating jobless_claims table...")
    collector.create_table()

    # Step 2: Collect 52 weeks of data
    logger.info("\nStep 2: Fetching 52 weeks of jobless claims data...")
    collector.collect_claims_data(weeks=52)

    # Step 3: Calculate spike detection
    logger.info("\nStep 3: Calculating 4-week averages and detecting spikes...")
    collector.calculate_spike_detection()

    # Step 4: Display results
    logger.info("\n" + "=" * 80)
    logger.info("Latest Jobless Claims (Most Recent 10 Weeks):")
    logger.info("=" * 80)

    latest = collector.get_latest_claims(limit=10)

    print("\n{:<15} {:>15} {:>15} {:>15} {:>10}".format(
        "Date", "Initial", "Continued", "4-Week Avg", "Spike?"
    ))
    print("-" * 80)

    for row in latest:
        date, initial, continued, four_week, spike = row
        continued_str = f"{continued:,}" if continued else "N/A"
        four_week_str = f"{four_week:,}" if four_week else "N/A"
        spike_str = "YES" if spike else "No"

        print("{:<15} {:>15,} {:>15} {:>15} {:>10}".format(
            str(date), initial, continued_str, four_week_str, spike_str
        ))

    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("Spike Detection Summary:")
    logger.info("=" * 80)

    summary = collector.get_spike_summary()

    if summary and summary.get('total_weeks', 0) > 0:
        print(f"\nTotal Weeks Analyzed: {summary['total_weeks']}")
        print(f"Spikes Detected: {summary['spike_count'] or 0}")
        if summary.get('avg_initial_claims'):
            print(f"Average Initial Claims: {summary['avg_initial_claims']:,}")
        if summary.get('max_initial_claims'):
            print(f"Max Initial Claims: {summary['max_initial_claims']:,}")
        if summary.get('min_initial_claims'):
            print(f"Min Initial Claims: {summary['min_initial_claims']:,}")

        if summary.get('most_recent_spike'):
            date, claims, avg = summary['most_recent_spike']
            print(f"\nMost Recent Spike:")
            print(f"  Date: {date}")
            print(f"  Initial Claims: {claims:,}")
            print(f"  4-Week Average: {avg:,}")
            print(f"  Increase: {((claims - avg) / avg * 100):.1f}%")
    else:
        print("\nNo data available in database yet.")

    logger.info("\n" + "=" * 80)
    logger.info("FRED Jobless Claims Data Collection Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
