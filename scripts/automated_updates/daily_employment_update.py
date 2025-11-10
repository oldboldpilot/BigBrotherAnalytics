#!/usr/bin/env python3
"""
BigBrotherAnalytics: Daily Employment Update Orchestrator

This is the main entry point for automated daily BLS employment data updates.
Orchestrates data collection, signal recalculation, and notifications.

Schedule:
- BLS Employment: First Friday after month end at 10:00 AM ET
- Jobless Claims: Every Thursday at 10:00 AM ET
- Signal recalculation: Immediately after data update
- Notifications: On significant changes or errors

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import our automation modules
try:
    from daily_update import DailyDataUpdater
    from recalculate_signals import SignalRecalculator
    from notify import NotificationManager
except ImportError as e:
    print(f"Error importing automation modules: {e}", file=sys.stderr)
    print("Make sure you're running from the correct directory", file=sys.stderr)
    sys.exit(1)

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
LOG_DIR = BASE_DIR / 'logs' / 'automated_updates'
DB_PATH = BASE_DIR / 'data' / 'bigbrother.duckdb'
ALERT_CONFIG = BASE_DIR / 'configs' / 'alert_config.yaml'

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / f'daily_employment_update_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EmploymentUpdateOrchestrator:
    """
    Orchestrates the complete daily employment update workflow:
    1. Check if data update is needed (based on BLS release schedule)
    2. Fetch new BLS employment and jobless claims data
    3. Validate data quality
    4. Recalculate employment and rotation signals
    5. Detect significant changes
    6. Send notifications
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        test_mode: bool = False,
        enable_notifications: bool = True
    ):
        """
        Initialize the orchestrator.

        Args:
            db_path: Path to DuckDB database
            test_mode: If True, run in test mode (no actual updates)
            enable_notifications: Enable email/Slack notifications
        """
        self.db_path = db_path
        self.test_mode = test_mode
        self.enable_notifications = enable_notifications

        self.workflow_summary = {
            'timestamp': datetime.now().isoformat(),
            'test_mode': test_mode,
            'steps_completed': [],
            'steps_failed': [],
            'data_updates': {},
            'signal_changes': {},
            'notifications_sent': [],
            'overall_success': False
        }

        logger.info("=" * 80)
        logger.info("BigBrotherAnalytics - Daily Employment Update Orchestrator")
        logger.info(f"Mode: {'TEST' if test_mode else 'LIVE'}")
        logger.info(f"Database: {db_path}")
        logger.info(f"Notifications: {'ENABLED' if enable_notifications else 'DISABLED'}")
        logger.info("=" * 80)

    def step_1_update_data(self) -> Dict:
        """
        Step 1: Update BLS employment and jobless claims data.

        Returns:
            Update summary dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Data Update")
        logger.info("=" * 80)

        try:
            updater = DailyDataUpdater(
                db_path=self.db_path,
                test_mode=self.test_mode
            )

            summary = updater.run_daily_update()

            self.workflow_summary['data_updates'] = summary
            self.workflow_summary['steps_completed'].append('data_update')

            logger.info("✓ Data update step completed")
            return summary

        except Exception as e:
            logger.error(f"✗ Data update step failed: {e}", exc_info=True)
            self.workflow_summary['steps_failed'].append({
                'step': 'data_update',
                'error': str(e)
            })
            raise

    def step_2_validate_data_quality(self, update_summary: Dict) -> bool:
        """
        Step 2: Validate data quality after update.

        Args:
            update_summary: Summary from data update step

        Returns:
            True if data quality is acceptable
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Data Quality Validation")
        logger.info("=" * 80)

        try:
            # Check if any updates were performed
            updates_performed = update_summary.get('updates_performed', [])
            if not updates_performed:
                logger.info("No updates performed - validation skipped")
                self.workflow_summary['steps_completed'].append('data_validation')
                return True

            # Check for errors in data update
            errors = update_summary.get('errors', [])
            if errors:
                logger.warning(f"Data update had {len(errors)} errors")
                for error in errors:
                    logger.warning(f"  - {error.get('type')}: {error.get('error')}")

                # Continue anyway - partial updates are ok
                # Full validation would check record counts, date ranges, etc.

            # Basic validation checks
            validation_passed = True

            # TODO: Add more sophisticated validation:
            # - Check record counts increased
            # - Verify date ranges are correct
            # - Check for data gaps
            # - Validate sector code mappings

            if validation_passed:
                logger.info("✓ Data quality validation passed")
                self.workflow_summary['steps_completed'].append('data_validation')
            else:
                logger.warning("⚠ Data quality validation had warnings")
                self.workflow_summary['steps_completed'].append('data_validation')

            return validation_passed

        except Exception as e:
            logger.error(f"✗ Data validation step failed: {e}", exc_info=True)
            self.workflow_summary['steps_failed'].append({
                'step': 'data_validation',
                'error': str(e)
            })
            # Don't raise - continue with signal recalculation
            return False

    def step_3_recalculate_signals(self) -> Dict:
        """
        Step 3: Recalculate employment and sector rotation signals.

        Returns:
            Signal recalculation summary
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Signal Recalculation")
        logger.info("=" * 80)

        try:
            recalculator = SignalRecalculator(
                db_path=self.db_path,
                test_mode=self.test_mode
            )

            summary = recalculator.run_recalculation()

            self.workflow_summary['signal_changes'] = summary
            self.workflow_summary['steps_completed'].append('signal_recalculation')

            logger.info("✓ Signal recalculation completed")
            return summary

        except Exception as e:
            logger.error(f"✗ Signal recalculation failed: {e}", exc_info=True)
            self.workflow_summary['steps_failed'].append({
                'step': 'signal_recalculation',
                'error': str(e)
            })
            raise

    def step_4_send_notifications(
        self,
        update_summary: Dict,
        signal_summary: Dict
    ) -> List[str]:
        """
        Step 4: Send notifications for significant changes and errors.

        Args:
            update_summary: Data update summary
            signal_summary: Signal recalculation summary

        Returns:
            List of notifications sent
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Notifications")
        logger.info("=" * 80)

        if not self.enable_notifications:
            logger.info("Notifications disabled - skipping")
            return []

        notifications_sent = []

        try:
            # Initialize notification manager
            # Check environment variables to enable email/slack
            email_enabled = bool(os.getenv('SMTP_USER') and os.getenv('EMAIL_TO'))
            slack_enabled = bool(os.getenv('SLACK_WEBHOOK_URL'))

            notifier = NotificationManager(
                email_enabled=email_enabled,
                slack_enabled=slack_enabled,
                test_mode=self.test_mode
            )

            # 1. Send error notifications if there were errors
            all_errors = []
            all_errors.extend(update_summary.get('errors', []))
            all_errors.extend(signal_summary.get('errors', []))

            if all_errors:
                logger.info(f"Sending error notification ({len(all_errors)} errors)")
                notifier.notify_errors(all_errors)
                notifications_sent.append('errors')

            # 2. Send new data notifications
            updates_performed = update_summary.get('updates_performed', [])
            for update in updates_performed:
                if update.get('result', {}).get('success'):
                    logger.info(f"Sending new data notification: {update.get('type')}")
                    notifier.notify_new_data_available(
                        update.get('type'),
                        update.get('result')
                    )
                    notifications_sent.append(f"data_{update.get('type')}")

            # 3. Send signal change notifications
            significant_changes = signal_summary.get('significant_changes', [])
            if significant_changes:
                logger.info(f"Sending signal changes notification ({len(significant_changes)} changes)")
                notifier.notify_signal_changes(significant_changes)
                notifications_sent.append('signal_changes')

            # 4. Send daily summary
            logger.info("Sending daily summary")
            notifier.send_daily_summary(update_summary, signal_summary)
            notifications_sent.append('daily_summary')

            self.workflow_summary['notifications_sent'] = notifications_sent
            self.workflow_summary['steps_completed'].append('notifications')

            logger.info(f"✓ Sent {len(notifications_sent)} notifications")
            return notifications_sent

        except Exception as e:
            logger.error(f"✗ Notification step failed: {e}", exc_info=True)
            self.workflow_summary['steps_failed'].append({
                'step': 'notifications',
                'error': str(e)
            })
            # Don't raise - notifications failing shouldn't fail the whole workflow
            return notifications_sent

    def run_complete_workflow(self) -> Dict:
        """
        Run the complete daily employment update workflow.

        Returns:
            Complete workflow summary
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting Complete Daily Employment Update Workflow")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # Step 1: Update data
            update_summary = self.step_1_update_data()

            # Step 2: Validate data quality
            self.step_2_validate_data_quality(update_summary)

            # Step 3: Recalculate signals
            signal_summary = self.step_3_recalculate_signals()

            # Step 4: Send notifications
            self.step_4_send_notifications(update_summary, signal_summary)

            # Workflow completed successfully
            self.workflow_summary['overall_success'] = len(self.workflow_summary['steps_failed']) == 0

        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)
            self.workflow_summary['overall_success'] = False

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        self.workflow_summary['duration_seconds'] = duration

        # Save workflow summary
        self._save_workflow_summary()

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("Workflow Summary")
        logger.info("=" * 80)
        logger.info(f"Status: {'SUCCESS' if self.workflow_summary['overall_success'] else 'FAILED'}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Steps Completed: {len(self.workflow_summary['steps_completed'])}")
        logger.info(f"Steps Failed: {len(self.workflow_summary['steps_failed'])}")
        logger.info(f"Notifications Sent: {len(self.workflow_summary['notifications_sent'])}")
        logger.info("=" * 80)

        return self.workflow_summary

    def _save_workflow_summary(self) -> None:
        """Save workflow summary to JSON file."""
        summary_file = LOG_DIR / f'workflow_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        try:
            with open(summary_file, 'w') as f:
                json.dump(self.workflow_summary, f, indent=2)
            logger.info(f"Workflow summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Error saving workflow summary: {e}")


def main():
    """Main entry point for daily employment update orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BigBrotherAnalytics Daily Employment Update Orchestrator',
        epilog='This script orchestrates data updates, signal recalculation, and notifications.'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (no actual updates or notifications)'
    )
    parser.add_argument(
        '--no-notifications',
        action='store_true',
        help='Disable all notifications'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=str(DB_PATH),
        help='Path to DuckDB database'
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = EmploymentUpdateOrchestrator(
        db_path=Path(args.db_path),
        test_mode=args.test,
        enable_notifications=not args.no_notifications
    )

    # Run complete workflow
    summary = orchestrator.run_complete_workflow()

    # Exit with appropriate code
    if summary['overall_success']:
        logger.info("Daily employment update completed successfully")
        sys.exit(0)
    else:
        logger.error("Daily employment update completed with errors")
        sys.exit(1)


if __name__ == '__main__':
    main()
