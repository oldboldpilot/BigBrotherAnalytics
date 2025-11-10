#!/usr/bin/env python3
"""
BigBrotherAnalytics: Automated Health Monitoring
Continuously monitor system health and send alerts on issues

Runs health checks every 5 minutes
Logs health status to file
Sends alerts when components are unhealthy

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import health check and notification modules
from monitoring.health_check import check_system_health, STATUS_HEALTHY, STATUS_DEGRADED, STATUS_DOWN, STATUS_STALE, STATUS_WARNING, STATUS_NO_DATA
from automated_updates.notify import NotificationManager

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
LOG_DIR = BASE_DIR / 'logs' / 'monitoring'
CHECK_INTERVAL = 300  # 5 minutes

# Setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'monitor_health_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors system health and sends alerts."""

    def __init__(self, alert_enabled: bool = False, test_mode: bool = False):
        """
        Initialize health monitor.

        Args:
            alert_enabled: Enable alert notifications
            test_mode: Run in test mode
        """
        self.alert_enabled = alert_enabled
        self.test_mode = test_mode

        # Initialize notification manager
        self.notifier = NotificationManager(
            email_enabled=alert_enabled,
            slack_enabled=alert_enabled,
            test_mode=test_mode
        )

        # Track alert state to avoid spam
        self.last_alerts = {}
        self.alert_cooldown = 3600  # 1 hour cooldown between same alerts

        logger.info(f"Health Monitor initialized (Alerts: {alert_enabled}, Test: {test_mode})")

    def should_send_alert(self, component: str, status: str) -> bool:
        """
        Check if alert should be sent based on cooldown.

        Args:
            component: Component name
            status: Component status

        Returns:
            True if alert should be sent
        """
        alert_key = f"{component}:{status}"
        last_alert_time = self.last_alerts.get(alert_key)

        if not last_alert_time:
            return True

        # Check if cooldown period has passed
        time_since_alert = time.time() - last_alert_time
        return time_since_alert > self.alert_cooldown

    def record_alert(self, component: str, status: str) -> None:
        """
        Record that an alert was sent.

        Args:
            component: Component name
            status: Component status
        """
        alert_key = f"{component}:{status}"
        self.last_alerts[alert_key] = time.time()

    def check_and_alert(self) -> Dict[str, Any]:
        """
        Run health check and send alerts if needed.

        Returns:
            Health check results
        """
        logger.info("Running health check...")

        # Run health check
        health = check_system_health()

        # Log health status
        self.log_health(health)

        # Check overall status
        if health['overall_status'] != STATUS_HEALTHY:
            if self.should_send_alert('system', health['overall_status']):
                self.send_system_alert(health)
                self.record_alert('system', health['overall_status'])

        # Check individual components
        components = health['components']

        # Critical components that should trigger immediate alerts
        critical_components = {
            'database': STATUS_DOWN,
            'disk_space': STATUS_WARNING,
            'memory': STATUS_WARNING
        }

        for component, critical_status in critical_components.items():
            if component in components:
                component_status = components[component].get('status')

                # Alert if status is at or worse than critical status
                if self.is_status_critical(component_status, critical_status):
                    if self.should_send_alert(component, component_status):
                        self.send_component_alert(component, components[component])
                        self.record_alert(component, component_status)

        # Warning components (less urgent)
        warning_components = ['schwab_api', 'signal_generation', 'data_freshness']

        for component in warning_components:
            if component in components:
                component_status = components[component].get('status')

                # Alert if status is problematic
                if component_status in [STATUS_DOWN, STATUS_STALE, STATUS_WARNING, STATUS_NO_DATA]:
                    if self.should_send_alert(component, component_status):
                        self.send_component_alert(component, components[component], severity='WARNING')
                        self.record_alert(component, component_status)

        logger.info(f"Health check complete. Overall status: {health['overall_status']}")

        return health

    def is_status_critical(self, status: str, critical_threshold: str) -> bool:
        """
        Check if status is at or above critical threshold.

        Args:
            status: Current status
            critical_threshold: Critical status threshold

        Returns:
            True if status is critical
        """
        # Status severity order (higher = worse)
        severity_order = {
            STATUS_HEALTHY: 0,
            STATUS_DEGRADED: 1,
            STATUS_STALE: 2,
            STATUS_WARNING: 3,
            STATUS_NO_DATA: 3,
            STATUS_DOWN: 4
        }

        current_severity = severity_order.get(status, 0)
        threshold_severity = severity_order.get(critical_threshold, 0)

        return current_severity >= threshold_severity

    def send_system_alert(self, health: Dict[str, Any]) -> None:
        """
        Send alert for system-wide health issue.

        Args:
            health: Health check results
        """
        if not self.alert_enabled:
            logger.info(f"Would send system alert: {health['overall_status']}")
            return

        overall_status = health['overall_status']

        # Build alert message
        subject = f"BigBrotherAnalytics: System Health {overall_status}"

        # Count unhealthy components
        unhealthy = []
        for component, status in health['components'].items():
            comp_status = status.get('status', 'UNKNOWN')
            if comp_status != STATUS_HEALTHY:
                unhealthy.append(f"{component}: {comp_status}")

        email_body = f"""
BigBrotherAnalytics Health Alert

System health status: {overall_status}
Timestamp: {health['timestamp']}

Unhealthy Components ({len(unhealthy)}):
"""
        for item in unhealthy:
            email_body += f"\n- {item}"

        email_body += """

Please review the system and take corrective action.

This is an automated alert from BigBrotherAnalytics.
"""

        # Send via notification manager
        try:
            self.notifier.send_email(subject, email_body)
            self.notifier.send_slack(
                f":warning: System health: {overall_status}",
                [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"System Health Alert: {overall_status}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{len(unhealthy)} unhealthy components*\n" + "\n".join([f"• {u}" for u in unhealthy[:10]])
                        }
                    }
                ]
            )
            logger.info(f"System alert sent: {overall_status}")
        except Exception as e:
            logger.error(f"Error sending system alert: {e}")

    def send_component_alert(self, component: str, component_data: Dict[str, Any], severity: str = 'ERROR') -> None:
        """
        Send alert for individual component issue.

        Args:
            component: Component name
            component_data: Component health data
            severity: Alert severity (ERROR or WARNING)
        """
        if not self.alert_enabled:
            logger.info(f"Would send {severity} alert for {component}: {component_data.get('status')}")
            return

        status = component_data.get('status', 'UNKNOWN')

        # Build alert message
        subject = f"BigBrotherAnalytics: {component.upper()} {status}"

        email_body = f"""
BigBrotherAnalytics Component Alert

Component: {component.replace('_', ' ').title()}
Status: {status}
Severity: {severity}

Details:
"""
        # Add relevant details
        for key, value in component_data.items():
            if key != 'status':
                email_body += f"\n- {key}: {value}"

        email_body += """

Please review this component and take corrective action if needed.

This is an automated alert from BigBrotherAnalytics.
"""

        # Emoji mapping
        emoji_map = {
            'schwab_api': ':chart_with_upwards_trend:',
            'database': ':floppy_disk:',
            'signal_generation': ':arrows_counterclockwise:',
            'data_freshness': ':calendar:',
            'disk_space': ':minidisc:',
            'memory': ':brain:',
            'cpu': ':zap:',
            'process': ':gear:',
            'logs': ':page_facing_up:'
        }

        emoji = emoji_map.get(component, ':warning:')

        # Send via notification manager
        try:
            self.notifier.send_email(subject, email_body)
            self.notifier.send_slack(
                f"{emoji} {component.replace('_', ' ').title()}: {status}",
                [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{component.replace('_', ' ').title()} Alert"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Status:*\n{status}"},
                            {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"}
                        ]
                    }
                ]
            )
            logger.info(f"Component alert sent: {component} - {status}")
        except Exception as e:
            logger.error(f"Error sending component alert: {e}")

    def log_health(self, health: Dict[str, Any]) -> None:
        """
        Log health check results to file.

        Args:
            health: Health check results
        """
        # Save JSON log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"health_{timestamp}.json"

        try:
            with open(log_file, 'w') as f:
                json.dump(health, f, indent=2)
            logger.debug(f"Health log saved to: {log_file}")
        except Exception as e:
            logger.error(f"Error saving health log: {e}")

    def run(self, continuous: bool = True, check_interval: int = CHECK_INTERVAL) -> None:
        """
        Run health monitoring.

        Args:
            continuous: If True, run continuously. If False, run once.
            check_interval: Seconds between checks (for continuous mode)
        """
        logger.info(f"Starting health monitoring (Continuous: {continuous}, Interval: {check_interval}s)")

        if not continuous:
            # Run once
            self.check_and_alert()
            return

        # Run continuously
        check_count = 0

        try:
            while True:
                check_count += 1
                logger.info(f"=== Health Check #{check_count} ===")

                try:
                    self.check_and_alert()
                except Exception as e:
                    logger.error(f"Error during health check: {e}", exc_info=True)

                logger.info(f"Next check in {check_interval} seconds...")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("Health monitoring stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in health monitoring: {e}", exc_info=True)
            raise

    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Clean up old health check logs.

        Args:
            days_to_keep: Number of days of logs to keep
        """
        logger.info(f"Cleaning up logs older than {days_to_keep} days...")

        cutoff_time = time.time() - (days_to_keep * 86400)
        removed_count = 0

        for log_file in LOG_DIR.glob("health_*.json"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.error(f"Error removing log file {log_file}: {e}")

        logger.info(f"Removed {removed_count} old log files")


def main():
    """Main entry point for monitoring script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BigBrotherAnalytics Automated Health Monitoring'
    )
    parser.add_argument(
        '--alerts',
        action='store_true',
        help='Enable alert notifications (email/Slack)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (log alerts but don\'t send)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (default is continuous)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=CHECK_INTERVAL,
        help=f'Check interval in seconds (default: {CHECK_INTERVAL})'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old logs before starting'
    )

    args = parser.parse_args()

    # Initialize monitor
    monitor = HealthMonitor(
        alert_enabled=args.alerts,
        test_mode=args.test
    )

    # Clean up old logs if requested
    if args.cleanup:
        monitor.cleanup_old_logs()

    # Run monitoring
    monitor.run(
        continuous=not args.once,
        check_interval=args.interval
    )


if __name__ == '__main__':
    main()
