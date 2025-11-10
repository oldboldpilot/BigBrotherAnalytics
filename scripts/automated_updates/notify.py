#!/usr/bin/env python3
"""
BigBrotherAnalytics: Notification System
Send notifications via Email and/or Slack for automated updates

Notifications sent for:
- New data available
- Significant signal changes (>10% confidence change)
- Errors or failures
- Daily summary reports

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import os
import sys
import json
import logging
import smtplib
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
LOG_DIR = BASE_DIR / 'logs' / 'automated_updates'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages email and Slack notifications for automated updates."""

    def __init__(
        self,
        email_enabled: bool = False,
        slack_enabled: bool = False,
        test_mode: bool = False
    ):
        """
        Initialize notification manager.

        Args:
            email_enabled: Enable email notifications
            slack_enabled: Enable Slack notifications
            test_mode: If True, log notifications but don't send
        """
        self.email_enabled = email_enabled
        self.slack_enabled = slack_enabled
        self.test_mode = test_mode

        # Email configuration from environment variables
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.email_from = os.getenv('EMAIL_FROM', self.smtp_user)
        self.email_to = os.getenv('EMAIL_TO', '').split(',')

        # Slack configuration from environment variables
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')

        # Validate configuration
        if self.email_enabled and not all([self.smtp_user, self.smtp_password, self.email_to]):
            logger.warning("Email notifications enabled but configuration incomplete")
            self.email_enabled = False

        if self.slack_enabled and not self.slack_webhook_url:
            logger.warning("Slack notifications enabled but webhook URL not configured")
            self.slack_enabled = False

        logger.info(f"Notification Manager initialized (Email: {self.email_enabled}, "
                   f"Slack: {self.slack_enabled}, Test: {self.test_mode})")

    def send_email(self, subject: str, body: str, html: bool = False) -> bool:
        """
        Send email notification.

        Args:
            subject: Email subject
            body: Email body (plain text or HTML)
            html: If True, body is HTML

        Returns:
            True if sent successfully
        """
        if not self.email_enabled:
            logger.debug("Email notifications disabled")
            return False

        if self.test_mode:
            logger.info(f"TEST MODE - Would send email:")
            logger.info(f"  To: {', '.join(self.email_to)}")
            logger.info(f"  Subject: {subject}")
            logger.info(f"  Body preview: {body[:200]}...")
            return True

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_from
            msg['To'] = ', '.join(self.email_to)
            msg['Subject'] = subject

            # Add body
            mime_type = 'html' if html else 'plain'
            msg.attach(MIMEText(body, mime_type))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {', '.join(self.email_to)}")
            return True

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def send_slack(self, message: str, blocks: Optional[List[Dict]] = None) -> bool:
        """
        Send Slack notification.

        Args:
            message: Message text (fallback text if blocks used)
            blocks: Optional Slack blocks for rich formatting

        Returns:
            True if sent successfully
        """
        if not self.slack_enabled:
            logger.debug("Slack notifications disabled")
            return False

        if self.test_mode:
            logger.info(f"TEST MODE - Would send Slack message:")
            logger.info(f"  Message: {message}")
            if blocks:
                logger.info(f"  Blocks: {json.dumps(blocks, indent=2)}")
            return True

        try:
            payload = {'text': message}
            if blocks:
                payload['blocks'] = blocks

            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            logger.info("Slack notification sent successfully")
            return True

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False

    def notify_new_data_available(self, data_type: str, details: Dict) -> None:
        """
        Notify that new data is available.

        Args:
            data_type: Type of data (e.g., 'employment', 'jobless_claims')
            details: Dictionary with update details
        """
        subject = f"BigBrotherAnalytics: New {data_type.replace('_', ' ').title()} Data Available"

        # Email body
        email_body = f"""
BigBrotherAnalytics Data Update Notification

New {data_type.replace('_', ' ')} data has been successfully collected and stored.

Update Details:
- Timestamp: {details.get('timestamp', 'N/A')}
- Records Updated: {details.get('records_updated', 'Unknown')}
- Status: {'Success' if details.get('success') else 'Failed'}

This is an automated notification from BigBrotherAnalytics.
"""

        # Slack message
        slack_message = f":chart_with_upwards_trend: New {data_type.replace('_', ' ')} data available"

        slack_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"New {data_type.replace('_', ' ').title()} Data"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Status:*\n{'✅ Success' if details.get('success') else '❌ Failed'}"},
                    {"type": "mrkdwn", "text": f"*Records:*\n{details.get('records_updated', 'Unknown')}"},
                ]
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Timestamp: {details.get('timestamp', 'N/A')}"}
                ]
            }
        ]

        # Send notifications
        self.send_email(subject, email_body)
        self.send_slack(slack_message, slack_blocks)

    def notify_signal_changes(self, changes: List[Dict]) -> None:
        """
        Notify about significant signal changes.

        Args:
            changes: List of significant change dictionaries
        """
        if not changes:
            logger.debug("No significant signal changes to notify")
            return

        subject = f"BigBrotherAnalytics: {len(changes)} Significant Signal Changes Detected"

        # Email body
        email_body = f"""
BigBrotherAnalytics Signal Change Notification

{len(changes)} significant signal changes have been detected:

"""
        for i, change in enumerate(changes[:10], 1):  # Limit to top 10
            email_body += f"\n{i}. {change.get('description', 'Unknown change')}"

        if len(changes) > 10:
            email_body += f"\n\n... and {len(changes) - 10} more changes."

        email_body += f"""

Review the full report in the automated updates logs.

This is an automated notification from BigBrotherAnalytics.
"""

        # Slack message
        slack_message = f":rotating_light: {len(changes)} significant signal changes detected"

        # Build Slack blocks for top changes
        slack_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{len(changes)} Signal Changes Detected"
                }
            }
        ]

        # Add top 5 changes
        for change in changes[:5]:
            change_type = change.get('type', 'unknown')
            description = change.get('description', 'Unknown change')

            emoji = ":arrow_up:" if change_type == 'employment' else ":arrows_counterclockwise:"

            slack_blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{emoji} {description}"
                }
            })

        if len(changes) > 5:
            slack_blocks.append({
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"_... and {len(changes) - 5} more changes_"}
                ]
            })

        # Send notifications
        self.send_email(subject, email_body)
        self.send_slack(slack_message, slack_blocks)

    def notify_errors(self, errors: List[Dict]) -> None:
        """
        Notify about errors during automated updates.

        Args:
            errors: List of error dictionaries
        """
        if not errors:
            return

        subject = f"BigBrotherAnalytics: ERROR - {len(errors)} Update Failures"

        # Email body
        email_body = f"""
BigBrotherAnalytics Error Notification

{len(errors)} errors occurred during automated updates:

"""
        for i, error in enumerate(errors, 1):
            email_body += f"\n{i}. {error.get('type', 'Unknown')}: {error.get('error', 'Unknown error')}"

        email_body += f"""

Please review the logs and take corrective action.

This is an automated notification from BigBrotherAnalytics.
"""

        # Slack message
        slack_message = f":x: {len(errors)} errors during automated updates"

        slack_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"⚠️ {len(errors)} Update Errors"
                }
            }
        ]

        for error in errors[:5]:
            slack_blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{error.get('type', 'Unknown')}*\n```{error.get('error', 'Unknown error')}```"
                }
            })

        # Send notifications
        self.send_email(subject, email_body)
        self.send_slack(slack_message, slack_blocks)

    def send_daily_summary(
        self,
        update_summary: Optional[Dict] = None,
        signal_summary: Optional[Dict] = None
    ) -> None:
        """
        Send daily summary report.

        Args:
            update_summary: Data update summary dictionary
            signal_summary: Signal recalculation summary dictionary
        """
        subject = f"BigBrotherAnalytics Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"

        # Build email body
        email_body = f"""
BigBrotherAnalytics Daily Summary Report
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

        # Data updates section
        if update_summary:
            email_body += "=" * 60 + "\n"
            email_body += "DATA UPDATES\n"
            email_body += "=" * 60 + "\n"

            updates = update_summary.get('updates_performed', [])
            email_body += f"Updates performed: {len(updates)}\n"

            for update in updates:
                update_type = update.get('type', 'Unknown')
                result = update.get('result', {})
                success = result.get('success', False)
                email_body += f"\n- {update_type}: {'✓ Success' if success else '✗ Failed'}"
                if not success:
                    email_body += f"\n  Error: {result.get('error', 'Unknown')}"

            email_body += "\n"

        # Signal changes section
        if signal_summary:
            email_body += "\n" + "=" * 60 + "\n"
            email_body += "SIGNAL RECALCULATION\n"
            email_body += "=" * 60 + "\n"

            signals_recalc = signal_summary.get('signals_recalculated', {})
            email_body += f"Employment signals: {signals_recalc.get('employment', {}).get('count', 0)}\n"
            email_body += f"Rotation signals: {signals_recalc.get('rotation', {}).get('count', 0)}\n"

            changes = signal_summary.get('significant_changes', [])
            email_body += f"\nSignificant changes: {len(changes)}\n"

            if changes:
                email_body += "\nTop changes:\n"
                for change in changes[:5]:
                    email_body += f"- {change.get('description', 'Unknown')}\n"

        email_body += "\n" + "=" * 60 + "\n"
        email_body += "This is an automated daily summary from BigBrotherAnalytics.\n"

        # Build Slack message
        slack_message = f":memo: Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"

        slack_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"
                }
            }
        ]

        # Add update stats
        if update_summary:
            updates = update_summary.get('updates_performed', [])
            errors = update_summary.get('errors', [])

            slack_blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Updates:*\n{len(updates)}"},
                    {"type": "mrkdwn", "text": f"*Errors:*\n{len(errors)}"}
                ]
            })

        # Add signal stats
        if signal_summary:
            signals_recalc = signal_summary.get('signals_recalculated', {})
            changes = signal_summary.get('significant_changes', [])

            slack_blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Signals:*\n{signals_recalc.get('employment', {}).get('count', 0)} + {signals_recalc.get('rotation', {}).get('count', 0)}"},
                    {"type": "mrkdwn", "text": f"*Changes:*\n{len(changes)}"}
                ]
            })

        # Send notifications
        self.send_email(subject, email_body)
        self.send_slack(slack_message, slack_blocks)


def load_latest_summary(summary_type: str) -> Optional[Dict]:
    """
    Load the latest summary file.

    Args:
        summary_type: 'update' or 'recalc'

    Returns:
        Summary dictionary or None
    """
    pattern = f'{summary_type}_summary_*.json'
    summary_files = sorted(LOG_DIR.glob(pattern))

    if not summary_files:
        return None

    latest_file = summary_files[-1]
    logger.info(f"Loading {summary_type} summary from: {latest_file}")

    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading summary: {e}")
        return None


def main():
    """Main entry point for notification script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BigBrotherAnalytics Notification Script'
    )
    parser.add_argument(
        '--email',
        action='store_true',
        help='Enable email notifications'
    )
    parser.add_argument(
        '--slack',
        action='store_true',
        help='Enable Slack notifications'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (log but don\'t send)'
    )
    parser.add_argument(
        '--type',
        choices=['data', 'signals', 'errors', 'summary'],
        default='summary',
        help='Notification type to send'
    )

    args = parser.parse_args()

    # Initialize notification manager
    notifier = NotificationManager(
        email_enabled=args.email,
        slack_enabled=args.slack,
        test_mode=args.test
    )

    # Load summaries
    update_summary = load_latest_summary('update')
    signal_summary = load_latest_summary('recalc')

    # Send appropriate notification
    if args.type == 'data' and update_summary:
        for update in update_summary.get('updates_performed', []):
            notifier.notify_new_data_available(
                update.get('type', 'unknown'),
                update.get('result', {})
            )

    elif args.type == 'signals' and signal_summary:
        changes = signal_summary.get('significant_changes', [])
        if changes:
            notifier.notify_signal_changes(changes)

    elif args.type == 'errors':
        errors = []
        if update_summary:
            errors.extend(update_summary.get('errors', []))
        if signal_summary:
            errors.extend(signal_summary.get('errors', []))
        if errors:
            notifier.notify_errors(errors)

    elif args.type == 'summary':
        notifier.send_daily_summary(update_summary, signal_summary)

    logger.info("Notification script completed")


if __name__ == '__main__':
    main()
