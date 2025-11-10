# BigBrother Analytics - Monitoring & Health Checks

Comprehensive system health monitoring and performance metrics tracking.

## Overview

The monitoring system provides:

1. **Health Checks** - Monitor all system components
2. **Automated Monitoring** - Continuous health monitoring with alerts
3. **Performance Metrics** - Track and analyze performance over time
4. **Dashboard Integration** - Real-time health view in dashboard

## Components

### 1. Health Check Script (`health_check.py`)

Comprehensive health checker for all system components.

**Checks:**
- Schwab API connectivity
- Database health and integrity
- Signal generation freshness
- Data freshness (employment, jobless claims, stock prices)
- System resources (disk, memory, CPU)
- Process status
- Log file errors

**Usage:**

```bash
# Run health check once
uv run python scripts/monitoring/health_check.py

# Output as JSON
uv run python scripts/monitoring/health_check.py --json

# Save results to file
uv run python scripts/monitoring/health_check.py --save

# Custom output directory
uv run python scripts/monitoring/health_check.py --save --output-dir /path/to/logs
```

**Exit Codes:**
- 0: All systems healthy
- 1: Warnings or degraded components
- 2: Critical failures

### 2. Automated Health Monitor (`monitor_health.py`)

Continuous health monitoring with alert notifications.

**Features:**
- Runs health checks every 5 minutes (configurable)
- Sends alerts on component failures
- Alert cooldown to prevent spam (1 hour)
- Logs all health checks
- Automatic log cleanup

**Usage:**

```bash
# Run once (single check)
uv run python scripts/monitoring/monitor_health.py --once

# Run continuously (every 5 minutes)
uv run python scripts/monitoring/monitor_health.py

# Enable alerts (requires email/Slack configuration)
uv run python scripts/monitoring/monitor_health.py --alerts

# Test mode (log alerts but don't send)
uv run python scripts/monitoring/monitor_health.py --alerts --test

# Custom interval (seconds)
uv run python scripts/monitoring/monitor_health.py --interval 600

# Clean up old logs before starting
uv run python scripts/monitoring/monitor_health.py --cleanup
```

**Alert Configuration:**

Set environment variables in `.env` or system environment:

```bash
# Email configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=bigbrother@yourdomain.com
EMAIL_TO=alerts@yourdomain.com

# Slack configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### 3. Performance Metrics (`collect_metrics.py`)

Track and store performance metrics over time.

**Metrics Tracked:**
- Signal generation time
- Database query times
- API response times
- Order placement time
- Custom metrics

**Usage:**

```bash
# Record test metrics
uv run python scripts/monitoring/collect_metrics.py --test

# Show statistics for a metric
uv run python scripts/monitoring/collect_metrics.py --stats timing.signal_generation

# List recent metrics
uv run python scripts/monitoring/collect_metrics.py --list
```

**Programmatic Usage:**

```python
from scripts.monitoring.collect_metrics import timer, record_metric

# Time an operation
with timer('signal_generation', strategy='sector_rotation'):
    # Your code here
    generate_signals()

# Record a metric directly
record_metric('api_response_time', 123.45, context='GET /quotes')
```

### 4. Dashboard Integration

Health monitoring view integrated into the Streamlit dashboard.

**Access:**
1. Start dashboard: `cd dashboard && uv run streamlit run app.py`
2. Navigate to "System Health" view
3. Click "Refresh" button to update

**Features:**
- Overall system status
- Component-by-component health
- System resource usage (disk, memory, CPU)
- Process monitoring
- Visual alerts for issues
- Auto-refresh capability

## Running as a Service

### Using systemd (Linux)

1. **Copy service file:**
   ```bash
   sudo cp scripts/monitoring/bigbrother-monitor.service /etc/systemd/system/
   ```

2. **Update paths in service file if needed:**
   ```bash
   sudo nano /etc/systemd/system/bigbrother-monitor.service
   ```

3. **Enable and start service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable bigbrother-monitor
   sudo systemctl start bigbrother-monitor
   ```

4. **Check status:**
   ```bash
   sudo systemctl status bigbrother-monitor
   ```

5. **View logs:**
   ```bash
   sudo journalctl -u bigbrother-monitor -f
   ```

### Using nohup (Manual)

```bash
# Start in background
nohup uv run python scripts/monitoring/monitor_health.py --alerts > logs/monitoring/monitor.log 2>&1 &

# Check if running
ps aux | grep monitor_health

# Stop
pkill -f monitor_health.py
```

## Health Status Levels

- **HEALTHY**: Component functioning normally
- **DEGRADED**: Component operational but with issues
- **STALE**: Data is outdated
- **WARNING**: Potential issue detected
- **NO_DATA**: No data available
- **DOWN**: Component not functioning
- **LOW**: Resource running low (disk/memory)
- **HIGH**: Resource usage high (CPU/memory)

## Thresholds

### Data Freshness
- Employment data: Stale after 45 days (monthly updates)
- Jobless claims: Stale after 14 days (weekly updates)
- Stock prices: Stale after 5 days (daily updates)
- Signals: Stale after 24 hours

### System Resources
- Disk space: Warning at 80%, Critical at 90%
- Memory: Warning at 80%, Critical at 90%
- CPU: Warning at 80%, Critical at 95%

### Alerts
- System-level alerts: Overall status not HEALTHY
- Component alerts: Individual component failures
- Cooldown: 1 hour between same alerts

## Log Management

### Health Check Logs

Location: `logs/monitoring/health_*.json`

Retention: 30 days (automatic cleanup)

Format: JSON with full health check results

### Monitor Logs

Location: `logs/monitoring/monitor_health_YYYYMMDD.log`

Retention: 30 days (automatic cleanup)

Format: Standard log format with timestamps

### Metrics Buffer

Location: `logs/monitoring/metrics_buffer_*.json`

Created when database is locked

Should be imported to database when available

## Troubleshooting

### Database Locked Errors

If you see "Conflicting lock" errors, the database is open elsewhere (e.g., dashboard).

**Solutions:**
1. Metrics will buffer to file automatically
2. Use read-only mode for queries
3. Stop other processes temporarily
4. Import buffered metrics later

### No Alerts Received

Check:
1. Alert configuration in environment variables
2. SMTP credentials are correct
3. Slack webhook URL is valid
4. Test mode is disabled (`--test` flag)
5. Monitor is running with `--alerts` flag

### Service Not Starting

Check:
1. Service file paths are correct
2. User has permissions
3. Virtual environment exists
4. Dependencies installed
5. View logs: `journalctl -u bigbrother-monitor -n 50`

## Monitoring Best Practices

1. **Regular Health Checks**: Run automated monitoring continuously
2. **Review Logs**: Check health logs weekly
3. **Set Up Alerts**: Configure email/Slack for critical issues
4. **Monitor Trends**: Use metrics to identify performance degradation
5. **Clean Up Logs**: Run cleanup periodically (or use `--cleanup`)
6. **Test Alerts**: Use `--test` mode to verify alert configuration
7. **Dashboard Review**: Check System Health view daily

## Integration with Automated Updates

The health monitoring system integrates with the automated updates system:

- Uses same notification framework (`notify.py`)
- Monitors data freshness from automated updates
- Tracks signal generation from recalculation
- Can detect update failures

## Performance Impact

- Health checks: ~1-2 seconds per check
- CPU usage: Minimal (<1%)
- Memory: ~50-100 MB
- Disk: ~1 MB per day of logs
- Network: Only for alerts (minimal)

## Future Enhancements

- [ ] HTTP health endpoint for external monitoring
- [ ] Prometheus metrics export
- [ ] Grafana dashboard
- [ ] Predictive alerting (ML-based)
- [ ] Mobile notifications
- [ ] Alert aggregation
- [ ] Custom alert rules
- [ ] Performance baselines
- [ ] Anomaly detection

## Support

For issues or questions about the monitoring system:

1. Check logs in `logs/monitoring/`
2. Review health check output
3. Test individual components
4. Verify configuration
5. Check documentation

## Related Documentation

- [Automated Updates README](../automated_updates/README.md)
- [Dashboard README](../../dashboard/README.md)
- [Main Project README](../../README.md)
