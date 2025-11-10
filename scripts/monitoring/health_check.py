#!/usr/bin/env python3
"""
BigBrotherAnalytics: Comprehensive System Health Check
Monitor all system components and report health status

Checks:
- Schwab API connectivity
- Database health and integrity
- Signal freshness
- Data freshness
- System resources (disk, memory, CPU)
- Process status

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import os
import sys
import json
import duckdb
import requests
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = BASE_DIR / "data" / "bigbrother.duckdb"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Health status constants
STATUS_HEALTHY = "HEALTHY"
STATUS_DEGRADED = "DEGRADED"
STATUS_DOWN = "DOWN"
STATUS_STALE = "STALE"
STATUS_WARNING = "WARNING"
STATUS_NO_DATA = "NO_DATA"
STATUS_RUNNING = "RUNNING"
STATUS_NOT_RUNNING = "NOT_RUNNING"
STATUS_LOW = "LOW"
STATUS_HIGH = "HIGH"


def check_schwab_api() -> Dict[str, Any]:
    """
    Test Schwab API connectivity.

    Returns:
        Dictionary with API health status
    """
    try:
        # Check if API keys file exists
        api_keys_file = BASE_DIR / "api_keys.yaml"

        if not api_keys_file.exists():
            return {
                "status": STATUS_WARNING,
                "message": "API keys file not found",
                "last_check": datetime.now().isoformat()
            }

        # Note: We can't actually test the API without implementing the full OAuth flow
        # So we check if the keys file exists and has been recently modified
        import yaml

        with open(api_keys_file, 'r') as f:
            keys = yaml.safe_load(f)

        schwab_config = keys.get('schwab', {})
        has_keys = all([
            schwab_config.get('client_id'),
            schwab_config.get('client_secret')
        ])

        if not has_keys:
            return {
                "status": STATUS_WARNING,
                "message": "Schwab API keys not configured",
                "last_check": datetime.now().isoformat()
            }

        # Check if there are recent orders in the database (indicates API is working)
        if DB_PATH.exists():
            try:
                db = duckdb.connect(str(DB_PATH), read_only=True)
                recent_positions = db.execute(
                    "SELECT COUNT(*) FROM positions WHERE updated_at > NOW() - INTERVAL '24 hours'"
                ).fetchone()[0]
                db.close()

                if recent_positions > 0:
                    return {
                        "status": STATUS_HEALTHY,
                        "message": "Recent positions found (API likely operational)",
                        "recent_positions": recent_positions,
                        "last_check": datetime.now().isoformat()
                    }
            except Exception as e:
                pass

        return {
            "status": STATUS_HEALTHY,
            "message": "API keys configured (cannot test connectivity without full OAuth)",
            "last_check": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "status": STATUS_DOWN,
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }


def check_database() -> Dict[str, Any]:
    """
    Check DuckDB connectivity and integrity.

    Returns:
        Dictionary with database health status
    """
    try:
        if not DB_PATH.exists():
            return {
                "status": STATUS_DOWN,
                "error": "Database file not found",
                "path": str(DB_PATH)
            }

        db = duckdb.connect(str(DB_PATH), read_only=True)

        # Check tables exist
        tables_result = db.execute("SHOW TABLES").fetchall()
        tables = [row[0] for row in tables_result]

        required_tables = ["positions", "sector_employment", "jobless_claims", "stock_prices", "sectors"]
        missing_tables = [t for t in required_tables if t not in tables]

        if missing_tables:
            db.close()
            return {
                "status": STATUS_DEGRADED,
                "message": f"Missing tables: {', '.join(missing_tables)}",
                "tables_found": len(tables),
                "tables_missing": missing_tables
            }

        # Check record counts
        counts = {}
        try:
            counts['positions'] = db.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        except:
            counts['positions'] = 0

        try:
            counts['sector_employment'] = db.execute("SELECT COUNT(*) FROM sector_employment").fetchone()[0]
        except:
            counts['sector_employment'] = 0

        try:
            counts['jobless_claims'] = db.execute("SELECT COUNT(*) FROM jobless_claims").fetchone()[0]
        except:
            counts['jobless_claims'] = 0

        try:
            counts['stock_prices'] = db.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
        except:
            counts['stock_prices'] = 0

        # Check database size
        size_mb = os.path.getsize(DB_PATH) / (1024**2)

        db.close()

        return {
            "status": STATUS_HEALTHY,
            "size_mb": round(size_mb, 2),
            "tables": len(tables),
            "record_counts": counts,
            "last_check": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "status": STATUS_DOWN,
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }


def check_signal_freshness() -> Dict[str, Any]:
    """
    Check when signals were last generated.

    Returns:
        Dictionary with signal freshness status
    """
    try:
        if not DB_PATH.exists():
            return {
                "status": STATUS_NO_DATA,
                "message": "Database not found"
            }

        db = duckdb.connect(str(DB_PATH), read_only=True)

        # Check positions_history for last signal time
        try:
            last_signal = db.execute(
                "SELECT MAX(timestamp) FROM positions_history"
            ).fetchone()[0]
        except:
            last_signal = None

        # If no positions_history, check positions table
        if not last_signal:
            try:
                last_signal = db.execute(
                    "SELECT MAX(updated_at) FROM positions"
                ).fetchone()[0]
            except:
                last_signal = None

        db.close()

        if last_signal:
            age = datetime.now() - last_signal
            age_hours = age.total_seconds() / 3600

            # Signals should be updated at least daily
            if age_hours < 24:
                status = STATUS_HEALTHY
            elif age_hours < 72:  # 3 days
                status = STATUS_STALE
            else:
                status = STATUS_WARNING

            return {
                "status": status,
                "last_signal_time": last_signal.isoformat(),
                "age_hours": round(age_hours, 1),
                "age_days": round(age_hours / 24, 1)
            }
        else:
            return {
                "status": STATUS_NO_DATA,
                "message": "No signals found in database",
                "last_signal_time": None
            }

    except Exception as e:
        return {
            "status": STATUS_DOWN,
            "error": str(e)
        }


def check_data_freshness() -> Dict[str, Any]:
    """
    Check age of employment and market data.

    Returns:
        Dictionary with data freshness status
    """
    try:
        if not DB_PATH.exists():
            return {
                "status": STATUS_NO_DATA,
                "message": "Database not found"
            }

        db = duckdb.connect(str(DB_PATH), read_only=True)

        # Employment data age
        try:
            last_employment = db.execute(
                "SELECT MAX(report_date) FROM sector_employment"
            ).fetchone()[0]
        except:
            last_employment = None

        # Jobless claims age
        try:
            last_jobless = db.execute(
                "SELECT MAX(date) FROM jobless_claims"
            ).fetchone()[0]
        except:
            last_jobless = None

        # Stock prices age
        try:
            last_prices = db.execute(
                "SELECT MAX(date) FROM stock_prices"
            ).fetchone()[0]
        except:
            last_prices = None

        db.close()

        result = {}

        # Employment data (updated monthly, allow 45 days)
        if last_employment:
            age_days = (datetime.now().date() - last_employment).days
            status = STATUS_HEALTHY if age_days < 45 else STATUS_STALE
            result['employment_data'] = {
                "status": status,
                "last_update": last_employment.isoformat(),
                "age_days": age_days
            }
        else:
            result['employment_data'] = {
                "status": STATUS_NO_DATA,
                "last_update": None,
                "age_days": None
            }

        # Jobless claims (updated weekly, allow 14 days)
        if last_jobless:
            age_days = (datetime.now().date() - last_jobless).days
            status = STATUS_HEALTHY if age_days < 14 else STATUS_STALE
            result['jobless_claims'] = {
                "status": status,
                "last_update": last_jobless.isoformat(),
                "age_days": age_days
            }
        else:
            result['jobless_claims'] = {
                "status": STATUS_NO_DATA,
                "last_update": None,
                "age_days": None
            }

        # Stock prices (updated daily for trading days, allow 5 days)
        if last_prices:
            age_days = (datetime.now().date() - last_prices).days
            status = STATUS_HEALTHY if age_days < 5 else STATUS_STALE
            result['stock_prices'] = {
                "status": status,
                "last_update": last_prices.isoformat(),
                "age_days": age_days
            }
        else:
            result['stock_prices'] = {
                "status": STATUS_NO_DATA,
                "last_update": None,
                "age_days": None
            }

        # Overall data freshness status
        statuses = [v['status'] for v in result.values()]
        if STATUS_STALE in statuses:
            overall_status = STATUS_STALE
        elif STATUS_NO_DATA in statuses:
            overall_status = STATUS_NO_DATA
        else:
            overall_status = STATUS_HEALTHY

        result['overall_status'] = overall_status

        return result

    except Exception as e:
        return {
            "status": STATUS_DOWN,
            "error": str(e)
        }


def check_disk_space() -> Dict[str, Any]:
    """
    Check available disk space.

    Returns:
        Dictionary with disk space status
    """
    try:
        usage = psutil.disk_usage(str(BASE_DIR))

        percent_used = usage.percent
        status = STATUS_HEALTHY if percent_used < 80 else (STATUS_WARNING if percent_used < 90 else STATUS_LOW)

        return {
            "status": status,
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent_used": round(percent_used, 1)
        }

    except Exception as e:
        return {
            "status": STATUS_DOWN,
            "error": str(e)
        }


def check_memory() -> Dict[str, Any]:
    """
    Check system memory usage.

    Returns:
        Dictionary with memory status
    """
    try:
        mem = psutil.virtual_memory()

        percent_used = mem.percent
        status = STATUS_HEALTHY if percent_used < 80 else (STATUS_WARNING if percent_used < 90 else STATUS_HIGH)

        return {
            "status": status,
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "percent_used": round(percent_used, 1)
        }

    except Exception as e:
        return {
            "status": STATUS_DOWN,
            "error": str(e)
        }


def check_cpu() -> Dict[str, Any]:
    """
    Check CPU usage.

    Returns:
        Dictionary with CPU status
    """
    try:
        # Get CPU percentage over 1 second
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        status = STATUS_HEALTHY if cpu_percent < 80 else (STATUS_WARNING if cpu_percent < 95 else STATUS_HIGH)

        return {
            "status": status,
            "percent_used": round(cpu_percent, 1),
            "cpu_count": cpu_count
        }

    except Exception as e:
        return {
            "status": STATUS_DOWN,
            "error": str(e)
        }


def check_process_status() -> Dict[str, Any]:
    """
    Check if BigBrother process is running.

    Returns:
        Dictionary with process status
    """
    try:
        bigbrother_processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                proc_info = proc.info
                name = proc_info.get('name', '').lower()
                cmdline = ' '.join(proc_info.get('cmdline', [])).lower()

                # Look for bigbrother-related processes
                if 'bigbrother' in name or 'bigbrother' in cmdline or 'daily_update' in cmdline:
                    memory_mb = proc_info['memory_info'].rss / (1024**2) if proc_info.get('memory_info') else 0

                    bigbrother_processes.append({
                        "pid": proc_info['pid'],
                        "name": proc_info.get('name', 'unknown'),
                        "cpu_percent": round(proc_info.get('cpu_percent', 0), 1),
                        "memory_mb": round(memory_mb, 2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if bigbrother_processes:
            return {
                "status": STATUS_RUNNING,
                "process_count": len(bigbrother_processes),
                "processes": bigbrother_processes
            }
        else:
            return {
                "status": STATUS_NOT_RUNNING,
                "message": "No BigBrother processes detected"
            }

    except Exception as e:
        return {
            "status": STATUS_DOWN,
            "error": str(e)
        }


def check_log_files() -> Dict[str, Any]:
    """
    Check log files for recent errors.

    Returns:
        Dictionary with log file status
    """
    try:
        if not LOG_DIR.exists():
            return {
                "status": STATUS_WARNING,
                "message": "Log directory not found"
            }

        # Find recent log files
        recent_errors = 0
        log_files = list(LOG_DIR.rglob("*.log"))

        # Check for recent error lines in log files
        for log_file in log_files[-10:]:  # Check last 10 log files
            try:
                # Check if file was modified in last 24 hours
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if datetime.now() - mtime > timedelta(hours=24):
                    continue

                # Count error lines
                with open(log_file, 'r') as f:
                    for line in f:
                        if 'ERROR' in line or 'CRITICAL' in line:
                            recent_errors += 1
            except:
                continue

        if recent_errors > 10:
            status = STATUS_WARNING
        elif recent_errors > 0:
            status = STATUS_DEGRADED
        else:
            status = STATUS_HEALTHY

        return {
            "status": status,
            "log_files": len(log_files),
            "recent_errors": recent_errors,
            "message": f"Found {recent_errors} recent error(s) in logs"
        }

    except Exception as e:
        return {
            "status": STATUS_DOWN,
            "error": str(e)
        }


def check_system_health() -> Dict[str, Any]:
    """
    Check all system components and return comprehensive health status.

    Returns:
        Dictionary with overall health status and component details
    """
    print("Running comprehensive system health check...")

    components = {
        "schwab_api": check_schwab_api(),
        "database": check_database(),
        "signal_generation": check_signal_freshness(),
        "data_freshness": check_data_freshness(),
        "disk_space": check_disk_space(),
        "memory": check_memory(),
        "cpu": check_cpu(),
        "process": check_process_status(),
        "logs": check_log_files()
    }

    # Determine overall health status
    critical_components = ["database", "disk_space", "memory"]
    important_components = ["data_freshness", "signal_generation"]

    # Check for critical failures
    critical_down = any(
        components[c].get('status') == STATUS_DOWN
        for c in critical_components
        if c in components
    )

    critical_degraded = any(
        components[c].get('status') in [STATUS_DEGRADED, STATUS_LOW, STATUS_HIGH, STATUS_WARNING]
        for c in critical_components
        if c in components
    )

    important_issues = any(
        components[c].get('status') in [STATUS_STALE, STATUS_NO_DATA, STATUS_WARNING]
        for c in important_components
        if c in components
    )

    # Determine overall status
    if critical_down:
        overall_status = STATUS_DOWN
    elif critical_degraded:
        overall_status = STATUS_DEGRADED
    elif important_issues:
        overall_status = STATUS_WARNING
    else:
        overall_status = STATUS_HEALTHY

    return {
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall_status,
        "components": components
    }


def format_health_report(health: Dict[str, Any]) -> str:
    """
    Format health check results as a readable report.

    Args:
        health: Health check results dictionary

    Returns:
        Formatted health report string
    """
    report = []
    report.append("=" * 80)
    report.append("BIGBROTHER ANALYTICS - SYSTEM HEALTH REPORT")
    report.append("=" * 80)
    report.append(f"Timestamp: {health['timestamp']}")
    report.append(f"Overall Status: {health['overall_status']}")
    report.append("=" * 80)
    report.append("")

    # Status emoji mapping
    status_emoji = {
        STATUS_HEALTHY: "✅",
        STATUS_DEGRADED: "⚠️",
        STATUS_DOWN: "❌",
        STATUS_STALE: "🕒",
        STATUS_WARNING: "⚠️",
        STATUS_NO_DATA: "❓",
        STATUS_RUNNING: "✅",
        STATUS_NOT_RUNNING: "⭕",
        STATUS_LOW: "⚠️",
        STATUS_HIGH: "⚠️"
    }

    components = health['components']

    # Schwab API
    report.append("1. SCHWAB API")
    report.append("-" * 80)
    api = components.get('schwab_api', {})
    status = api.get('status', 'UNKNOWN')
    report.append(f"   Status: {status_emoji.get(status, '❓')} {status}")
    if 'message' in api:
        report.append(f"   Message: {api['message']}")
    if 'error' in api:
        report.append(f"   Error: {api['error']}")
    report.append("")

    # Database
    report.append("2. DATABASE")
    report.append("-" * 80)
    db = components.get('database', {})
    status = db.get('status', 'UNKNOWN')
    report.append(f"   Status: {status_emoji.get(status, '❓')} {status}")
    if 'size_mb' in db:
        report.append(f"   Size: {db['size_mb']} MB")
        report.append(f"   Tables: {db.get('tables', 0)}")
        if 'record_counts' in db:
            report.append(f"   Record Counts:")
            for table, count in db['record_counts'].items():
                report.append(f"      - {table}: {count:,}")
    if 'error' in db:
        report.append(f"   Error: {db['error']}")
    report.append("")

    # Signal Generation
    report.append("3. SIGNAL GENERATION")
    report.append("-" * 80)
    signals = components.get('signal_generation', {})
    status = signals.get('status', 'UNKNOWN')
    report.append(f"   Status: {status_emoji.get(status, '❓')} {status}")
    if 'last_signal_time' in signals and signals['last_signal_time']:
        report.append(f"   Last Signal: {signals['last_signal_time']}")
        report.append(f"   Age: {signals.get('age_hours', 0):.1f} hours ({signals.get('age_days', 0):.1f} days)")
    if 'message' in signals:
        report.append(f"   Message: {signals['message']}")
    report.append("")

    # Data Freshness
    report.append("4. DATA FRESHNESS")
    report.append("-" * 80)
    data = components.get('data_freshness', {})
    overall_data_status = data.get('overall_status', 'UNKNOWN')
    report.append(f"   Overall Status: {status_emoji.get(overall_data_status, '❓')} {overall_data_status}")

    for data_type in ['employment_data', 'jobless_claims', 'stock_prices']:
        if data_type in data:
            data_info = data[data_type]
            status = data_info.get('status', 'UNKNOWN')
            report.append(f"   {data_type.replace('_', ' ').title()}:")
            report.append(f"      Status: {status_emoji.get(status, '❓')} {status}")
            if data_info.get('last_update'):
                report.append(f"      Last Update: {data_info['last_update']}")
                report.append(f"      Age: {data_info.get('age_days', 0)} days")
    report.append("")

    # System Resources
    report.append("5. SYSTEM RESOURCES")
    report.append("-" * 80)

    # Disk
    disk = components.get('disk_space', {})
    status = disk.get('status', 'UNKNOWN')
    report.append(f"   Disk Space: {status_emoji.get(status, '❓')} {status}")
    if 'total_gb' in disk:
        report.append(f"      Total: {disk['total_gb']} GB")
        report.append(f"      Used: {disk['used_gb']} GB ({disk['percent_used']}%)")
        report.append(f"      Free: {disk['free_gb']} GB")

    # Memory
    mem = components.get('memory', {})
    status = mem.get('status', 'UNKNOWN')
    report.append(f"   Memory: {status_emoji.get(status, '❓')} {status}")
    if 'total_gb' in mem:
        report.append(f"      Total: {mem['total_gb']} GB")
        report.append(f"      Used: {mem['used_gb']} GB ({mem['percent_used']}%)")
        report.append(f"      Available: {mem['available_gb']} GB")

    # CPU
    cpu = components.get('cpu', {})
    status = cpu.get('status', 'UNKNOWN')
    report.append(f"   CPU: {status_emoji.get(status, '❓')} {status}")
    if 'percent_used' in cpu:
        report.append(f"      Usage: {cpu['percent_used']}%")
        report.append(f"      CPU Count: {cpu.get('cpu_count', 0)}")
    report.append("")

    # Process Status
    report.append("6. PROCESS STATUS")
    report.append("-" * 80)
    proc = components.get('process', {})
    status = proc.get('status', 'UNKNOWN')
    report.append(f"   Status: {status_emoji.get(status, '❓')} {status}")
    if 'processes' in proc:
        report.append(f"   Running Processes: {proc['process_count']}")
        for p in proc['processes']:
            report.append(f"      - PID {p['pid']}: {p['name']} (CPU: {p['cpu_percent']}%, Mem: {p['memory_mb']} MB)")
    if 'message' in proc:
        report.append(f"   Message: {proc['message']}")
    report.append("")

    # Logs
    report.append("7. LOG FILES")
    report.append("-" * 80)
    logs = components.get('logs', {})
    status = logs.get('status', 'UNKNOWN')
    report.append(f"   Status: {status_emoji.get(status, '❓')} {status}")
    if 'log_files' in logs:
        report.append(f"   Total Log Files: {logs['log_files']}")
        report.append(f"   Recent Errors: {logs.get('recent_errors', 0)}")
    if 'message' in logs:
        report.append(f"   Message: {logs['message']}")
    report.append("")

    report.append("=" * 80)
    report.append("END OF HEALTH REPORT")
    report.append("=" * 80)

    return "\n".join(report)


def save_health_check(health: Dict[str, Any], output_dir: Path) -> None:
    """
    Save health check results to file.

    Args:
        health: Health check results
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f"health_check_{timestamp}.json"

    with open(json_file, 'w') as f:
        json.dump(health, f, indent=2)

    print(f"Health check saved to: {json_file}")

    # Save formatted report
    report_file = output_dir / f"health_report_{timestamp}.txt"
    report = format_health_report(health)

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"Health report saved to: {report_file}")


def main():
    """Main entry point for health check script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BigBrotherAnalytics System Health Check'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=LOG_DIR / 'monitoring',
        help='Output directory for saved results'
    )

    args = parser.parse_args()

    # Run health check
    health = check_system_health()

    # Output results
    if args.json:
        print(json.dumps(health, indent=2))
    else:
        report = format_health_report(health)
        print(report)

    # Save if requested
    if args.save:
        save_health_check(health, args.output_dir)

    # Exit with appropriate code
    if health['overall_status'] == STATUS_DOWN:
        sys.exit(2)
    elif health['overall_status'] in [STATUS_DEGRADED, STATUS_WARNING]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
