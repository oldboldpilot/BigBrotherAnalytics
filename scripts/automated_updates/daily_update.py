#!/usr/bin/env python3
"""
BigBrotherAnalytics: Daily Data Update Script
Automated daily updates for BLS employment and jobless claims data

Schedule:
- BLS Employment: First Friday after month end
- Jobless Claims: Every Thursday

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'data_collection'))

try:
    from data_collection.bls_employment import BLSEmploymentCollector
except ImportError:
    # Fallback to direct import if module structure is different
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bls_employment",
        Path(__file__).parent.parent / 'data_collection' / 'bls_employment.py'
    )
    bls_employment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bls_employment)
    BLSEmploymentCollector = bls_employment.BLSEmploymentCollector

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
LOG_DIR = BASE_DIR / 'logs' / 'automated_updates'
DB_PATH = BASE_DIR / 'data' / 'bigbrother.duckdb'
CONFIG_PATH = BASE_DIR / 'configs' / 'config.yaml'

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / f'daily_update_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DailyDataUpdater:
    """Handles daily automated data updates for BLS employment and jobless claims."""

    def __init__(self, db_path: Path = DB_PATH, test_mode: bool = False):
        """
        Initialize daily data updater.

        Args:
            db_path: Path to DuckDB database
            test_mode: If True, run in test mode (fetch data but report only)
        """
        self.db_path = db_path
        self.test_mode = test_mode
        self.update_summary = {
            'timestamp': datetime.now().isoformat(),
            'updates_performed': [],
            'errors': [],
            'data_available': {}
        }

        logger.info("=" * 80)
        logger.info("BigBrotherAnalytics Daily Data Update")
        logger.info(f"Mode: {'TEST' if test_mode else 'LIVE'}")
        logger.info(f"Database: {db_path}")
        logger.info("=" * 80)

    def is_first_friday_after_month_end(self, check_date: Optional[datetime] = None) -> bool:
        """
        Check if today (or check_date) is the first Friday after month end.

        BLS releases employment data on the first Friday after the month ends.

        Args:
            check_date: Date to check (default: today)

        Returns:
            True if it's the first Friday after month end
        """
        if check_date is None:
            check_date = datetime.now()

        # Get first day of current month
        first_of_month = check_date.replace(day=1)

        # If we're still in first few days of month and it's Friday
        if check_date.day <= 7 and check_date.weekday() == 4:  # Friday = 4
            # This is likely the first Friday
            # Check if any earlier Friday exists this month
            for day in range(1, check_date.day):
                test_date = check_date.replace(day=day)
                if test_date.weekday() == 4:
                    return False  # Earlier Friday exists
            return True

        return False

    def is_thursday(self, check_date: Optional[datetime] = None) -> bool:
        """
        Check if today (or check_date) is Thursday.

        Jobless claims are released every Thursday.

        Args:
            check_date: Date to check (default: today)

        Returns:
            True if it's Thursday
        """
        if check_date is None:
            check_date = datetime.now()

        return check_date.weekday() == 3  # Thursday = 3

    def should_update_employment_data(self) -> Tuple[bool, str]:
        """
        Determine if employment data should be updated today.

        Returns:
            (should_update, reason)
        """
        today = datetime.now()

        # Always update on first Friday after month end
        if self.is_first_friday_after_month_end(today):
            return True, "First Friday after month end - BLS employment data release day"

        # Also check if we're in first week of month (in case we missed Friday)
        if today.day <= 7:
            return True, "First week of month - checking for BLS data availability"

        return False, "Not a BLS employment data release window"

    def should_update_jobless_claims(self) -> Tuple[bool, str]:
        """
        Determine if jobless claims data should be updated today.

        Returns:
            (should_update, reason)
        """
        if self.is_thursday():
            return True, "Thursday - weekly jobless claims release day"

        return False, "Not a jobless claims release day"

    def update_employment_data(self) -> Dict:
        """
        Update BLS employment data.

        Returns:
            Dictionary with update results
        """
        logger.info("Checking for BLS employment data updates...")

        result = {
            'success': False,
            'records_updated': 0,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }

        try:
            collector = BLSEmploymentCollector(db_path=str(self.db_path))

            # Fetch last 2 years to catch any revisions
            if not self.test_mode:
                collector.collect_sector_employment(years=2)
                result['success'] = True
                result['records_updated'] = 'Updated'  # BLS collector logs actual count
                logger.info("Employment data update completed successfully")
            else:
                logger.info("TEST MODE: Would update employment data (2 years)")
                result['success'] = True
                result['records_updated'] = 'TEST MODE - No actual update'

        except Exception as e:
            logger.error(f"Error updating employment data: {e}", exc_info=True)
            result['error'] = str(e)

        return result

    def update_jobless_claims(self) -> Dict:
        """
        Update weekly jobless claims data.

        Returns:
            Dictionary with update results
        """
        logger.info("Checking for jobless claims updates...")

        result = {
            'success': False,
            'records_updated': 0,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }

        try:
            collector = BLSEmploymentCollector(db_path=str(self.db_path))

            # Fetch last 52 weeks
            if not self.test_mode:
                collector.collect_jobless_claims(weeks=52)
                result['success'] = True
                result['records_updated'] = 'Updated'
                logger.info("Jobless claims update completed successfully")
            else:
                logger.info("TEST MODE: Would update jobless claims (52 weeks)")
                result['success'] = True
                result['records_updated'] = 'TEST MODE - No actual update'

        except Exception as e:
            logger.error(f"Error updating jobless claims: {e}", exc_info=True)
            result['error'] = str(e)

        return result

    def run_daily_update(self) -> Dict:
        """
        Run the daily update process.

        Returns:
            Summary dictionary of all updates
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting Daily Update Process")
        logger.info("=" * 80 + "\n")

        # Check if employment data update is needed
        should_update_employment, employment_reason = self.should_update_employment_data()
        logger.info(f"Employment Data: {employment_reason}")
        self.update_summary['data_available']['employment'] = should_update_employment

        if should_update_employment:
            employment_result = self.update_employment_data()
            self.update_summary['updates_performed'].append({
                'type': 'employment_data',
                'result': employment_result
            })
            if not employment_result['success']:
                self.update_summary['errors'].append({
                    'type': 'employment_data',
                    'error': employment_result['error']
                })

        # Check if jobless claims update is needed
        should_update_claims, claims_reason = self.should_update_jobless_claims()
        logger.info(f"Jobless Claims: {claims_reason}")
        self.update_summary['data_available']['jobless_claims'] = should_update_claims

        if should_update_claims:
            claims_result = self.update_jobless_claims()
            self.update_summary['updates_performed'].append({
                'type': 'jobless_claims',
                'result': claims_result
            })
            if not claims_result['success']:
                self.update_summary['errors'].append({
                    'type': 'jobless_claims',
                    'error': claims_result['error']
                })

        # Save update summary
        self._save_update_summary()

        logger.info("\n" + "=" * 80)
        logger.info("Daily Update Process Complete")
        logger.info(f"Updates performed: {len(self.update_summary['updates_performed'])}")
        logger.info(f"Errors: {len(self.update_summary['errors'])}")
        logger.info("=" * 80 + "\n")

        return self.update_summary

    def _save_update_summary(self) -> None:
        """Save update summary to JSON file."""
        summary_file = LOG_DIR / f'update_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        try:
            with open(summary_file, 'w') as f:
                json.dump(self.update_summary, f, indent=2)
            logger.info(f"Update summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Error saving update summary: {e}")


def main():
    """Main entry point for daily update script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BigBrotherAnalytics Daily Data Update Script'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (no actual updates)'
    )
    parser.add_argument(
        '--force-employment',
        action='store_true',
        help='Force employment data update regardless of schedule'
    )
    parser.add_argument(
        '--force-claims',
        action='store_true',
        help='Force jobless claims update regardless of schedule'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=str(DB_PATH),
        help='Path to DuckDB database'
    )

    args = parser.parse_args()

    updater = DailyDataUpdater(
        db_path=Path(args.db_path),
        test_mode=args.test
    )

    # Handle forced updates
    if args.force_employment:
        logger.info("FORCED UPDATE: Employment data")
        result = updater.update_employment_data()
        logger.info(f"Result: {result}")

    if args.force_claims:
        logger.info("FORCED UPDATE: Jobless claims")
        result = updater.update_jobless_claims()
        logger.info(f"Result: {result}")

    # Run normal daily update if no forced updates
    if not args.force_employment and not args.force_claims:
        summary = updater.run_daily_update()

        # Exit with error code if there were errors
        if summary['errors']:
            logger.error("Daily update completed with errors")
            sys.exit(1)
        else:
            logger.info("Daily update completed successfully")
            sys.exit(0)


if __name__ == '__main__':
    main()
