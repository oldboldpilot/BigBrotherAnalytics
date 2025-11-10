# BigBrotherAnalytics - Automated Data Update System

Automated daily data updates for BLS employment and jobless claims with signal recalculation and notifications.

## Overview

This automated system handles:
- **Daily Data Updates**: Fetches BLS employment data (first Friday of month) and jobless claims (every Thursday)
- **Signal Recalculation**: Updates sector rotation signals based on new data
- **Notifications**: Sends alerts via Email/Slack for significant changes
- **Error Handling**: Robust error handling with detailed logging

## Components

### 1. `daily_update.py`
**Purpose**: Fetches and updates BLS employment and jobless claims data

**Features**:
- Smart scheduling: Checks if today is a data release day
- BLS Employment Data: First Friday after month end
- Jobless Claims: Every Thursday
- Test mode for safe testing
- Detailed logging and update summaries

**Usage**:
```bash
# Normal daily update (automatic scheduling)
python3 daily_update.py

# Test mode (no actual updates)
python3 daily_update.py --test

# Force employment data update
python3 daily_update.py --force-employment

# Force jobless claims update
python3 daily_update.py --force-claims

# Custom database path
python3 daily_update.py --db-path /path/to/database.duckdb
```

**Exit Codes**:
- `0`: Success
- `1`: Errors occurred during update

### 2. `recalculate_signals.py`
**Purpose**: Recalculates employment and sector rotation signals after data updates

**Features**:
- Recalculates employment signals (improving/declining trends)
- Recalculates sector rotation signals (overweight/underweight)
- Detects significant changes (>5% confidence or >10% allocation)
- Compares with previous signals to track changes
- Saves signal history for trend analysis

**Usage**:
```bash
# Normal recalculation
python3 recalculate_signals.py

# Test mode (calculate but don't save)
python3 recalculate_signals.py --test

# Custom database path
python3 recalculate_signals.py --db-path /path/to/database.duckdb
```

**Exit Codes**:
- `0`: Success
- `1`: Errors occurred during recalculation

### 3. `notify.py`
**Purpose**: Sends notifications via Email and/or Slack

**Features**:
- Email notifications (SMTP)
- Slack notifications (webhook)
- Multiple notification types
- Test mode for safe testing
- Rich formatting (HTML email, Slack blocks)

**Usage**:
```bash
# Send daily summary (default)
python3 notify.py --slack --type summary

# Send signal change notifications
python3 notify.py --email --slack --type signals

# Send error notifications
python3 notify.py --email --type errors

# Send new data notifications
python3 notify.py --slack --type data

# Test mode (log but don't send)
python3 notify.py --email --slack --test --type summary
```

**Notification Types**:
- `data`: New data available
- `signals`: Significant signal changes
- `errors`: Error notifications
- `summary`: Daily summary report (default)

### 4. `setup_cron.sh`
**Purpose**: Sets up automated cron jobs

**Features**:
- Creates cron job entries
- Backs up existing crontab
- Interactive installation
- Configuration instructions
- WSL detection and warnings

**Usage**:
```bash
# Run setup script (interactive)
./setup_cron.sh
```

**Cron Schedule** (Eastern Time):
- `6:00 AM ET`: Daily data update
- `6:30 AM ET`: Signal recalculation
- `7:00 AM ET`: Daily summary notification
- `2:00 AM Sunday`: Weekly log cleanup

## Installation & Setup

### Step 1: Install Dependencies

Ensure you have Python 3.8+ and required packages:

```bash
pip install duckdb requests pyyaml
```

### Step 2: Configure API Keys

#### BLS API Key (Optional but Recommended)

Get a free API key: https://data.bls.gov/registrationEngine/

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):
```bash
export BLS_API_KEY='your_bls_api_key_here'
```

Or add to `configs/api_keys.yaml`:
```yaml
bls:
  api_key: 'your_bls_api_key_here'
```

### Step 3: Configure Notifications (Optional)

#### Email Notifications

Set environment variables:
```bash
export SMTP_HOST='smtp.gmail.com'
export SMTP_PORT='587'
export SMTP_USER='your_email@gmail.com'
export SMTP_PASSWORD='your_app_password'  # Use app-specific password for Gmail
export EMAIL_FROM='your_email@gmail.com'
export EMAIL_TO='recipient@example.com'
```

**Gmail Users**: Generate an app-specific password:
1. Go to Google Account settings
2. Security → 2-Step Verification → App passwords
3. Generate password for "Mail"

#### Slack Notifications

1. Create a Slack webhook: https://api.slack.com/messaging/webhooks
2. Set environment variable:
```bash
export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
```

### Step 4: Test Scripts

Test each script in test mode before scheduling:

```bash
# Test data update
cd /home/muyiwa/Development/BigBrotherAnalytics
python3 scripts/automated_updates/daily_update.py --test

# Test signal recalculation
python3 scripts/automated_updates/recalculate_signals.py --test

# Test notifications
python3 scripts/automated_updates/notify.py --slack --test --type summary
```

### Step 5: Setup Cron Jobs

Run the setup script:

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics/scripts/automated_updates
./setup_cron.sh
```

Follow the interactive prompts to install cron jobs.

## Manual Cron Setup

If you prefer manual setup, add these entries to your crontab (`crontab -e`):

```bash
# BigBrotherAnalytics Automated Updates

# Daily data update at 6:00 AM ET
0 6 * * * cd /home/muyiwa/Development/BigBrotherAnalytics && python3 scripts/automated_updates/daily_update.py >> logs/automated_updates/cron_daily_update.log 2>&1

# Signal recalculation at 6:30 AM ET
30 6 * * * cd /home/muyiwa/Development/BigBrotherAnalytics && python3 scripts/automated_updates/recalculate_signals.py >> logs/automated_updates/cron_signal_recalc.log 2>&1

# Daily summary notification at 7:00 AM ET
0 7 * * * cd /home/muyiwa/Development/BigBrotherAnalytics && python3 scripts/automated_updates/notify.py --slack --type summary >> logs/automated_updates/cron_notify.log 2>&1

# Weekly log cleanup (every Sunday at 2:00 AM)
0 2 * * 0 find /home/muyiwa/Development/BigBrotherAnalytics/logs/automated_updates -name "*.log" -mtime +30 -delete
```

## Directory Structure

```
scripts/automated_updates/
├── daily_update.py           # Daily data update script
├── recalculate_signals.py    # Signal recalculation script
├── notify.py                  # Notification script
├── setup_cron.sh             # Cron job setup script
└── README.md                 # This file

logs/automated_updates/
├── daily_update_YYYYMMDD.log      # Daily update logs
├── signal_recalc_YYYYMMDD.log     # Signal recalc logs
├── cron_daily_update.log          # Cron job logs (data)
├── cron_signal_recalc.log         # Cron job logs (signals)
├── cron_notify.log                # Cron job logs (notify)
├── update_summary_*.json          # Update summaries
├── recalc_summary_*.json          # Recalc summaries
└── signals/
    ├── employment_signals_*.json  # Employment signal history
    └── rotation_signals_*.json    # Rotation signal history
```

## Monitoring & Logs

### Check Cron Job Status

```bash
# View current crontab
crontab -l

# Check cron logs (Ubuntu/Debian)
grep CRON /var/log/syslog

# Check script logs
tail -f logs/automated_updates/cron_daily_update.log
tail -f logs/automated_updates/cron_signal_recalc.log
```

### Review Update Summaries

```bash
# Latest update summary
cat logs/automated_updates/update_summary_*.json | jq .

# Latest signal recalc summary
cat logs/automated_updates/recalc_summary_*.json | jq .

# Latest signals
cat logs/automated_updates/signals/rotation_signals_*.json | jq .
```

### Check for Errors

```bash
# Check for errors in logs
grep -i error logs/automated_updates/*.log

# Check script exit codes
echo $?  # After running a script manually
```

## Troubleshooting

### Issue: Cron jobs not running

**Solution**:
1. Check if cron service is running: `sudo service cron status`
2. Start cron if needed: `sudo service cron start`
3. Check cron logs: `grep CRON /var/log/syslog`
4. Verify crontab: `crontab -l`

### Issue: Scripts can't find modules

**Solution**:
- Ensure you're using absolute paths in cron jobs
- Add `cd /path/to/project` before running scripts
- Check Python path: `which python3`

### Issue: BLS API rate limit exceeded

**Solution**:
- Get a free BLS API key (increases limit to 500/day)
- Reduce update frequency if needed
- Check logs for API error messages

### Issue: Email notifications not working

**Solution**:
- Verify SMTP credentials are correct
- For Gmail: Use app-specific password, not account password
- Check firewall/network allows SMTP traffic
- Test manually: `python3 notify.py --email --test`

### Issue: Slack notifications not working

**Solution**:
- Verify webhook URL is correct
- Check webhook is active in Slack settings
- Test manually: `python3 notify.py --slack --test`

### Issue: WSL cron not working

**Solution**:
WSL doesn't auto-start cron. Options:
1. Manually start cron: `sudo service cron start`
2. Add to Windows Task Scheduler instead
3. Use WSL startup script to auto-start cron

## WSL Alternative: Windows Task Scheduler

For WSL users, Windows Task Scheduler may be more reliable:

1. Open Task Scheduler (Windows)
2. Create Basic Task
3. Trigger: Daily at 6:00 AM
4. Action: Start a program
5. Program: `wsl`
6. Arguments: `cd /home/muyiwa/Development/BigBrotherAnalytics && python3 scripts/automated_updates/daily_update.py`

Repeat for other scripts (signal recalc, notify).

## Data Release Schedule

### BLS Employment Data
- **Release**: First Friday after month end
- **Data**: Previous month's employment statistics
- **Example**: November 2024 data released on December 6, 2024 (first Friday)

### Jobless Claims
- **Release**: Every Thursday at 8:30 AM ET
- **Data**: Previous week's initial claims
- **Example**: Data for week ending Nov 2 released on Nov 9

## Signal Change Thresholds

### Employment Signals
- **Significant Change**: >5% confidence change OR >0.15 signal strength change
- **Signal Types**: EmploymentImproving, EmploymentDeclining
- **Confidence Range**: 60% - 95%

### Rotation Signals
- **Significant Change**: >10% allocation change OR action change (Overweight ↔ Neutral ↔ Underweight)
- **Allocation Range**: 2% - 18% per sector
- **Composite Score**: -1.0 to +1.0

## Advanced Usage

### Running Individual Components

```bash
# Just fetch employment data (no schedule check)
python3 scripts/automated_updates/daily_update.py --force-employment

# Just recalculate employment signals
python3 scripts/employment_signals.py generate_signals

# Just recalculate rotation signals
python3 scripts/employment_signals.py rotation_signals

# Send custom notification
python3 scripts/automated_updates/notify.py --slack --type signals
```

### Integration with Trading Engine

The automated signals can be consumed by the C++ trading engine:

```cpp
// Employment signals are saved to JSON files
auto signals = loadSignalsFromFile("logs/automated_updates/signals/rotation_signals_latest.json");

// Process rotation signals for portfolio allocation
for (const auto& signal : signals) {
    if (signal.action == "Overweight") {
        increaseAllocation(signal.sector_etf, signal.target_allocation);
    }
}
```

## Support

For issues or questions:
1. Check logs in `logs/automated_updates/`
2. Review this README
3. Test scripts manually with `--test` flag
4. Check BLS API status: https://www.bls.gov/

## License

Part of BigBrotherAnalytics project by Olumuyiwa Oluwasanmi.
