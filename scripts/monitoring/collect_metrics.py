#!/usr/bin/env python3
"""
BigBrotherAnalytics: Performance Metrics Collection
Track and store performance metrics over time

Metrics tracked:
- Signal generation time
- Database query times
- API response times
- Order placement time

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import time
import duckdb
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = BASE_DIR / "data" / "bigbrother.duckdb"
LOG_DIR = BASE_DIR / "logs" / "monitoring"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and stores performance metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics_buffer = []
        self.ensure_metrics_table()

    def ensure_metrics_table(self) -> None:
        """Ensure metrics table exists in database."""
        try:
            # Try to connect and create table if needed
            db = duckdb.connect(str(DB_PATH))

            create_table_sql = """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY,
                metric_name VARCHAR NOT NULL,
                value DOUBLE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                context VARCHAR,
                tags VARCHAR
            );
            """

            db.execute(create_table_sql)
            db.close()

            logger.info("Metrics table verified/created")

        except Exception as e:
            logger.warning(f"Could not create metrics table (may be locked): {e}")
            logger.info("Will buffer metrics to file instead")

    def record_metric(
        self,
        metric_name: str,
        value: float,
        context: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Optional context information
            tags: Optional tags dictionary
        """
        timestamp = datetime.now()

        metric_data = {
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp.isoformat(),
            'context': context,
            'tags': json.dumps(tags) if tags else None
        }

        # Try to write to database
        try:
            db = duckdb.connect(str(DB_PATH))

            insert_sql = """
            INSERT INTO metrics (metric_name, value, timestamp, context, tags)
            VALUES (?, ?, ?, ?, ?)
            """

            db.execute(
                insert_sql,
                [metric_name, value, timestamp, context, json.dumps(tags) if tags else None]
            )

            db.close()

            logger.debug(f"Metric recorded: {metric_name} = {value}")

        except Exception as e:
            # If database is locked, buffer to file
            logger.warning(f"Could not write metric to database: {e}")
            self.metrics_buffer.append(metric_data)

            # Periodically flush buffer to file
            if len(self.metrics_buffer) >= 100:
                self.flush_buffer()

    def flush_buffer(self) -> None:
        """Flush metrics buffer to file."""
        if not self.metrics_buffer:
            return

        LOG_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        buffer_file = LOG_DIR / f"metrics_buffer_{timestamp}.json"

        try:
            with open(buffer_file, 'w') as f:
                json.dump(self.metrics_buffer, f, indent=2)

            logger.info(f"Flushed {len(self.metrics_buffer)} metrics to {buffer_file}")
            self.metrics_buffer = []

        except Exception as e:
            logger.error(f"Error flushing metrics buffer: {e}")

    def record_timing(self, operation: str, duration_ms: float, **kwargs) -> None:
        """
        Record a timing metric.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            **kwargs: Additional context as keyword arguments
        """
        self.record_metric(
            f"timing.{operation}",
            duration_ms,
            context=kwargs.get('context'),
            tags=kwargs
        )

    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        hours: int = 24
    ) -> list:
        """
        Retrieve metrics from database.

        Args:
            metric_name: Filter by metric name (optional)
            hours: Hours of history to retrieve

        Returns:
            List of metric records
        """
        try:
            db = duckdb.connect(str(DB_PATH), read_only=True)

            if metric_name:
                sql = """
                SELECT metric_name, value, timestamp, context, tags
                FROM metrics
                WHERE metric_name = ?
                  AND timestamp > NOW() - INTERVAL ? HOUR
                ORDER BY timestamp DESC
                """
                result = db.execute(sql, [metric_name, hours]).fetchall()
            else:
                sql = """
                SELECT metric_name, value, timestamp, context, tags
                FROM metrics
                WHERE timestamp > NOW() - INTERVAL ? HOUR
                ORDER BY timestamp DESC
                LIMIT 1000
                """
                result = db.execute(sql, [hours]).fetchall()

            db.close()

            return result

        except Exception as e:
            logger.error(f"Error retrieving metrics: {e}")
            return []

    def get_metric_stats(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics for a specific metric.

        Args:
            metric_name: Metric name
            hours: Hours of history

        Returns:
            Dictionary with statistics
        """
        try:
            db = duckdb.connect(str(DB_PATH), read_only=True)

            sql = """
            SELECT
                COUNT(*) as count,
                AVG(value) as avg,
                MIN(value) as min,
                MAX(value) as max,
                STDDEV(value) as stddev
            FROM metrics
            WHERE metric_name = ?
              AND timestamp > NOW() - INTERVAL ? HOUR
            """

            result = db.execute(sql, [metric_name, hours]).fetchone()
            db.close()

            if result:
                return {
                    'metric_name': metric_name,
                    'count': result[0],
                    'average': round(result[1], 2) if result[1] else 0,
                    'min': round(result[2], 2) if result[2] else 0,
                    'max': round(result[3], 2) if result[3] else 0,
                    'stddev': round(result[4], 2) if result[4] else 0
                }
            else:
                return {
                    'metric_name': metric_name,
                    'count': 0
                }

        except Exception as e:
            logger.error(f"Error getting metric stats: {e}")
            return {
                'metric_name': metric_name,
                'error': str(e)
            }


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience functions
def record_metric(metric_name: str, value: float, **kwargs) -> None:
    """Record a metric (convenience function)."""
    get_metrics_collector().record_metric(metric_name, value, **kwargs)


def record_timing(operation: str, duration_ms: float, **kwargs) -> None:
    """Record a timing metric (convenience function)."""
    get_metrics_collector().record_timing(operation, duration_ms, **kwargs)


class timer:
    """Context manager for timing operations."""

    def __init__(self, operation: str, **kwargs):
        """
        Initialize timer.

        Args:
            operation: Operation name
            **kwargs: Additional context
        """
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record metric."""
        duration_ms = (time.time() - self.start_time) * 1000
        record_timing(self.operation, duration_ms, **self.kwargs)
        return False


def main():
    """Main entry point for testing metrics collection."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BigBrotherAnalytics Metrics Collection'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test metrics collection'
    )
    parser.add_argument(
        '--stats',
        type=str,
        help='Show statistics for metric name'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List recent metrics'
    )

    args = parser.parse_args()

    collector = get_metrics_collector()

    if args.test:
        print("Recording test metrics...")

        # Test timing metrics
        with timer('database_query', query='SELECT * FROM positions'):
            time.sleep(0.1)  # Simulate query

        with timer('signal_generation', strategy='sector_rotation'):
            time.sleep(0.5)  # Simulate signal generation

        # Test direct metric recording
        collector.record_metric('api_response_time', 123.45, context='GET /quotes')
        collector.record_metric('order_placement_time', 234.56, context='BUY XLF 100')

        print("✓ Test metrics recorded")

    elif args.stats:
        print(f"\nStatistics for metric: {args.stats}")
        print("-" * 60)

        stats = collector.get_metric_stats(args.stats)

        for key, value in stats.items():
            print(f"{key}: {value}")

    elif args.list:
        print("\nRecent metrics (last 24 hours):")
        print("-" * 80)

        metrics = collector.get_metrics()

        if metrics:
            for metric in metrics[:20]:
                name, value, timestamp, context, tags = metric
                print(f"{timestamp} | {name} = {value:.2f}")
                if context:
                    print(f"  Context: {context}")
        else:
            print("No metrics found")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
