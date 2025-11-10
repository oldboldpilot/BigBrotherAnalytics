#!/usr/bin/env python3
"""
BigBrotherAnalytics: Alert Processor
Continuously polls alerts database and delivers notifications via configured channels

This script runs as a background daemon to:
1. Poll alerts table every 5 seconds for unsent alerts
2. Send alerts via email, Slack, SMS, etc. based on configuration
3. Mark alerts as sent
4. Handle throttling and batching

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 4, Week 3: Custom Alerts System
"""

import os
import sys
import time
import json
import signal
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import duckdb
import yaml

# Import notification manager from existing script
sys.path.insert(0, str(BASE_DIR / 'scripts' / 'automated_updates'))
from notify import NotificationManager
from alert_templates import render_alert_email


# Setup logging
LOG_DIR = BASE_DIR / 'logs' / 'monitoring'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'alert_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlertProcessor:
    """
    Processes alerts from database and delivers via configured channels.
    """

    def __init__(self, db_path: str, config_path: str, test_mode: bool = False):
        """
        Initialize alert processor.

        Args:
            db_path: Path to DuckDB database
            config_path: Path to alert configuration file
            test_mode: If True, log alerts but don't actually send
        """
        self.db_path = db_path
        self.config_path = config_path
        self.test_mode = test_mode
        self.running = True

        # Load configuration
        self.config = self._load_config()

        # Initialize notification manager
        email_enabled = self.config.get('email', {}).get('enabled', False) and not test_mode
        slack_enabled = self.config.get('slack', {}).get('enabled', False) and not test_mode

        self.notifier = NotificationManager(
            email_enabled=email_enabled,
            slack_enabled=slack_enabled,
            test_mode=test_mode
        )

        # Polling interval (seconds)
        self.poll_interval = 5

        logger.info(f"AlertProcessor initialized (test_mode={test_mode})")
        logger.info(f"Email: {email_enabled}, Slack: {slack_enabled}")

    def _load_config(self) -> Dict:
        """Load alert configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}

    def run(self):
        """Main processing loop."""
        logger.info("Alert processor started - polling for alerts...")

        while self.running:
            try:
                # Process pending alerts
                self._process_pending_alerts()

                # Sleep for poll interval
                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(self.poll_interval)

        logger.info("Alert processor stopped")

    def _process_pending_alerts(self):
        """Process all pending unsent alerts."""
        try:
            conn = duckdb.connect(self.db_path, read_only=False)

            # Get unsent alerts
            alerts = conn.execute("""
                SELECT
                    id,
                    alert_type,
                    alert_subtype,
                    severity,
                    message,
                    context,
                    timestamp
                FROM alerts
                WHERE sent = false
                ORDER BY
                    CASE severity
                        WHEN 'CRITICAL' THEN 1
                        WHEN 'ERROR' THEN 2
                        WHEN 'WARNING' THEN 3
                        WHEN 'INFO' THEN 4
                    END,
                    timestamp ASC
                LIMIT 100
            """).fetchall()

            if not alerts:
                return

            logger.info(f"Processing {len(alerts)} pending alerts")

            # Process each alert
            for alert in alerts:
                self._deliver_alert(conn, alert)

            conn.close()

        except Exception as e:
            logger.error(f"Error processing pending alerts: {e}", exc_info=True)

    def _deliver_alert(self, conn, alert: tuple):
        """
        Deliver a single alert via configured channels.

        Args:
            conn: Database connection
            alert: Alert tuple (id, type, subtype, severity, message, context, timestamp)
        """
        alert_id, alert_type, alert_subtype, severity, message, context, timestamp = alert

        # Check if alert should be delivered based on severity configuration
        severity_config = self.config.get('severity', {}).get(severity, {})

        # Parse context JSON
        try:
            context_data = json.loads(context) if context else {}
        except:
            context_data = {}

        # Build notification
        subject = f"[{severity}] BigBrotherAnalytics: {alert_type.title()} Alert"

        # Build plain text message
        full_message = f"{message}\n\nType: {alert_type}\nSubtype: {alert_subtype}\nTime: {timestamp}"
        if context_data:
            full_message += f"\n\nContext:\n{json.dumps(context_data, indent=2)}"

        # Build HTML email
        alert_dict = {
            'alert_type': alert_type,
            'alert_subtype': alert_subtype,
            'severity': severity,
            'message': message,
            'context': context_data,
            'timestamp': str(timestamp)
        }
        html_message = render_alert_email(alert_dict)

        # Slack blocks for rich formatting
        slack_blocks = self._build_slack_blocks(alert_type, alert_subtype, severity, message,
                                                 context_data, timestamp)

        # Track delivery status
        email_sent = False
        slack_sent = False
        sms_sent = False

        # Send via email (HTML if enabled)
        if severity_config.get('email', False):
            use_html = self.config.get('format', {}).get('html_emails', True)
            if use_html:
                email_sent = self._send_email_html(subject, html_message)
            else:
                email_sent = self._send_email(subject, full_message)

        # Send via Slack
        if severity_config.get('slack', False):
            slack_sent = self._send_slack(message, slack_blocks)

        # Send via SMS (if configured and CRITICAL)
        if severity_config.get('sms', False) and severity == 'CRITICAL':
            sms_sent = self._send_sms(message)

        # Update alert as sent in database
        try:
            conn.execute(f"""
                UPDATE alerts
                SET
                    sent = true,
                    sent_timestamp = CURRENT_TIMESTAMP,
                    email_sent = {email_sent},
                    slack_sent = {slack_sent},
                    sms_sent = {sms_sent}
                WHERE id = {alert_id}
            """)

            # Log delivery
            conn.execute(f"""
                INSERT INTO alert_delivery_log (alert_id, channel, status, timestamp)
                VALUES
                    ({alert_id}, 'email', '{'success' if email_sent else 'skipped'}', CURRENT_TIMESTAMP),
                    ({alert_id}, 'slack', '{'success' if slack_sent else 'skipped'}', CURRENT_TIMESTAMP),
                    ({alert_id}, 'sms', '{'success' if sms_sent else 'skipped'}', CURRENT_TIMESTAMP)
            """)

            conn.commit()

            logger.info(f"Alert {alert_id} delivered: email={email_sent}, slack={slack_sent}, sms={sms_sent}")

        except Exception as e:
            logger.error(f"Error updating alert {alert_id}: {e}")

    def _send_email(self, subject: str, message: str) -> bool:
        """Send plain text email notification."""
        try:
            return self.notifier.send_email(subject, message, html=False)
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def _send_email_html(self, subject: str, html_message: str) -> bool:
        """Send HTML email notification."""
        try:
            return self.notifier.send_email(subject, html_message, html=True)
        except Exception as e:
            logger.error(f"Error sending HTML email: {e}")
            return False

    def _send_slack(self, message: str, blocks: Optional[List[Dict]] = None) -> bool:
        """Send Slack notification."""
        try:
            return self.notifier.send_slack(message, blocks)
        except Exception as e:
            logger.error(f"Error sending Slack: {e}")
            return False

    def _send_sms(self, message: str) -> bool:
        """Send SMS notification (Twilio)."""
        # SMS implementation would go here
        # For now, just log
        logger.info(f"SMS would be sent: {message[:100]}")
        return False

    def _build_slack_blocks(self, alert_type: str, alert_subtype: str, severity: str,
                            message: str, context: Dict, timestamp: str) -> List[Dict]:
        """Build Slack block formatting for rich notifications."""
        # Get color based on severity
        color_map = self.config.get('format', {}).get('slack_colors', {})
        color = color_map.get(severity, 'warning')

        # Emoji based on alert type
        emoji_map = {
            'trading': ':chart_with_upwards_trend:',
            'data': ':bar_chart:',
            'system': ':warning:',
            'performance': ':zap:'
        }
        emoji = emoji_map.get(alert_type, ':bell:')

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {severity} Alert: {alert_type.title()}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{message}*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Type:*\n{alert_type}"},
                    {"type": "mrkdwn", "text": f"*Subtype:*\n{alert_subtype}"},
                    {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{timestamp}"}
                ]
            }
        ]

        # Add context if available
        if context:
            context_text = "\n".join([f"• *{k}:* {v}" for k, v in context.items()])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Context:*\n{context_text}"
                }
            })

        blocks.append({
            "type": "divider"
        })

        return blocks

    def stop(self):
        """Stop the processor."""
        self.running = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='BigBrotherAnalytics Alert Processor')
    parser.add_argument(
        '--db',
        default=str(BASE_DIR / 'data' / 'bigbrother.duckdb'),
        help='Path to DuckDB database'
    )
    parser.add_argument(
        '--config',
        default=str(BASE_DIR / 'configs' / 'alert_config.yaml'),
        help='Path to alert configuration'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (log but don\'t send)'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as daemon (background process)'
    )

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create processor
    processor = AlertProcessor(
        db_path=args.db,
        config_path=args.config,
        test_mode=args.test
    )

    # Run processor
    try:
        processor.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
