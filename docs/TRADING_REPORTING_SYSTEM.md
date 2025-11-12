# Trading Reporting System

**Status:** Production Ready (Phase 5+)
**Last Updated:** November 11, 2025
**Version:** 1.0

## Overview

The Trading Reporting System provides comprehensive, automated reporting of trading activity, signal analysis, and risk metrics. It consists of two primary report generators that create both JSON and HTML outputs for analysis and stakeholder communication.

## Core Components

### 1. Daily Report Generator
**File:** `scripts/reporting/generate_daily_report.py`

Generates comprehensive daily trading reports with:

**Executive Summary**
- Account value and equity metrics
- Total signals generated, executed, and rejected
- Execution rate and average confidence
- Expected return from executed trades

**Trade Execution Details**
- List of all executed trades with metadata
- Strategy breakdown
- Greeks at signal generation (Delta, Theta, Vega)
- Risk metrics per trade

**Signal Analysis**
- Signal flow breakdown by status (EXECUTED, FILTERED_*, REJECTED_*)
- Rejection reason distribution
- Performance by strategy
- Acceptance rates

**Risk Compliance**
- Risk management status and rejection counts
- Budget compliance metrics
- Position sizing analysis

**Market Conditions**
- IV metrics (percentile, range)
- Days to expiration (DTE) analysis
- Market environment assessment

#### Usage

```bash
# Generate today's daily report
python scripts/reporting/generate_daily_report.py

# Reports saved to:
# - reports/daily_report_YYYYMMDD.json
# - reports/daily_report_YYYYMMDD.html
```

#### Output Structure

```json
{
  "metadata": {
    "report_type": "Daily Trading Report",
    "date": "2025-11-11",
    "generated_at": "2025-11-11T14:30:00",
    "version": "1.0"
  },
  "executive_summary": {
    "report_date": "2025-11-11",
    "account": {
      "liquidation_value": 50000.00,
      "equity": 50000.00,
      "buying_power": 100000.00
    },
    "signals": {
      "total_signals": 15,
      "executed": 12,
      "rejected": 3,
      "execution_rate": 80.0,
      "avg_confidence": 0.725,
      "executed_return": 1250.00
    }
  },
  "trade_execution": {
    "summary": {
      "total_executed": 12,
      "strategies": ["IVR", "Earnings"],
      "avg_confidence": 0.725,
      "total_expected_return": 1250.00,
      "total_estimated_cost": 4800.00,
      "total_max_risk": 600.00
    },
    "trades": [...]
  },
  "signal_analysis": {...},
  "risk_compliance": {...},
  "market_conditions": {...}
}
```

### 2. Weekly Report Generator
**File:** `scripts/reporting/generate_weekly_report.py`

Generates comprehensive weekly performance reports with:

**Performance Summary**
- Trading days active
- Total signals and execution metrics
- Expected return and risk metrics
- Risk/Reward ratio
- Sharpe ratio

**Strategy Comparison Table**
- Per-strategy signal counts
- Execution rates by strategy
- Return metrics and risk analysis
- Symbols traded and trading days per strategy

**Signal Acceptance Rates**
- Daily breakdown by strategy
- Rejection reason distribution
- Acceptance rate trends

**Risk Analysis**
- Risk rejections and budget rejections
- Max risk per position
- High-risk signal tracking
- Overall compliance status

**Recommendations**
- Automated suggestions based on metrics
- Execution rate optimization
- Budget constraint analysis
- Strategy performance guidance

#### Usage

```bash
# Generate this week's report (week starting Monday)
python scripts/reporting/generate_weekly_report.py

# Generate last week's report
python scripts/reporting/generate_weekly_report.py 1

# Generate report from 3 weeks ago
python scripts/reporting/generate_weekly_report.py 3

# Reports saved to:
# - reports/weekly_report_YYYYMMDD_to_YYYYMMDD.json
# - reports/weekly_report_YYYYMMDD_to_YYYYMMDD.html
```

## Database Schema

### trading_signals Table

Core table for all signal tracking:

```sql
CREATE TABLE trading_signals (
    signal_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    strategy VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    signal_type VARCHAR,              -- BUY, SELL, HOLD, CLOSE
    confidence DOUBLE,                -- 0.0-1.0
    expected_return DOUBLE,           -- Expected profit in dollars
    win_probability DOUBLE,           -- 0.0-1.0
    estimated_cost DOUBLE,            -- Position cost in dollars
    max_risk DOUBLE,                  -- Maximum loss in dollars
    status VARCHAR NOT NULL,          -- EXECUTED, FILTERED_*, REJECTED_*
    rejection_reason VARCHAR,         -- Why it was rejected
    order_ids VARCHAR,                -- Comma-separated order IDs

    -- Greeks at signal generation
    greeks_delta DOUBLE,
    greeks_gamma DOUBLE,
    greeks_theta DOUBLE,
    greeks_vega DOUBLE,
    greeks_rho DOUBLE,

    -- Options metrics
    iv_percentile DOUBLE,             -- 0-100
    days_to_expiration INTEGER,
    underlying_price DOUBLE,
    strike_price DOUBLE,

    market_conditions JSON            -- Market state snapshot
);
```

### Indexes
- `idx_signals_timestamp` - For time-range queries
- `idx_signals_status` - For filtering by status
- `idx_signals_strategy` - For strategy analysis
- `idx_signals_symbol` - For symbol tracking
- `idx_signals_date` - For daily reports

### Database Views

#### v_daily_signal_summary
Daily signal breakdown by status:

```sql
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN status = 'EXECUTED' THEN 1 END) as executed,
    COUNT(CASE WHEN status = 'FILTERED_BUDGET' THEN 1 END) as budget_rejected,
    COUNT(CASE WHEN status = 'FILTERED_CONFIDENCE' THEN 1 END) as confidence_rejected,
    COUNT(CASE WHEN status = 'FILTERED_RETURN' THEN 1 END) as return_rejected,
    COUNT(CASE WHEN status = 'REJECTED_RISK' THEN 1 END) as risk_rejected,
    AVG(expected_return) as avg_expected_return,
    SUM(CASE WHEN status = 'EXECUTED' THEN expected_return ELSE 0 END) as executed_return_potential,
    SUM(CASE WHEN status != 'EXECUTED' THEN expected_return ELSE 0 END) as missed_return_potential
FROM trading_signals
GROUP BY DATE(timestamp)
ORDER BY date DESC
```

#### v_signal_acceptance_by_strategy
Per-strategy acceptance metrics:

```sql
SELECT
    strategy,
    DATE(timestamp) as date,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN status = 'EXECUTED' THEN 1 END) as executed,
    COUNT(CASE WHEN status != 'EXECUTED' THEN 1 END) as rejected,
    CAST(COUNT(CASE WHEN status = 'EXECUTED' THEN 1 END) AS DOUBLE) / COUNT(*) * 100 as acceptance_rate
FROM trading_signals
GROUP BY strategy, DATE(timestamp)
ORDER BY date DESC, acceptance_rate DESC
```

#### v_budget_rejected_signals
Analysis of budget-constrained rejections:

```sql
SELECT
    DATE(timestamp) as date,
    strategy,
    symbol,
    estimated_cost,
    expected_return,
    confidence,
    win_probability
FROM trading_signals
WHERE status = 'FILTERED_BUDGET'
ORDER BY timestamp DESC
```

## Signal Status Values

| Status | Meaning | Analysis |
|--------|---------|----------|
| EXECUTED | Signal passed all filters and was traded | Success metric |
| FILTERED_CONFIDENCE | Rejected due to low confidence (<60%) | Refine signal generation |
| FILTERED_RETURN | Rejected due to low expected return | Check profitability thresholds |
| FILTERED_WIN_PROB | Rejected due to low win probability (<60%) | Improve model accuracy |
| FILTERED_BUDGET | Rejected due to position size limit ($500) | Consider increasing budget |
| REJECTED_RISK | Rejected by risk manager | Risk parameters too strict |
| GENERATED | Signal created but not yet processed | Initial state |
| FAILED | Signal generated but trade execution failed | Debug order issues |

## Dashboard Integration

### Live Trading Activity View
**File:** `dashboard/views/live_trading_activity.py`

Real-time monitoring of signal generation and execution:

**Features:**
- Signal flow Sankey diagram showing filter stages
- Rejection reason breakdown (pie chart)
- Strategy performance table
- Recent signals timeline with Greeks
- Key metrics and alerts

**Data Source:** `trading_signals` table filtered to today's date

**Refresh:** Auto-refresh support (30s, 60s, 120s intervals)

### Rejection Analysis View
**File:** `dashboard/views/rejection_analysis.py`

Deep dive into rejected signals:

**Features:**
- Rejection distribution by reason
- Daily rejection trends
- Strategy-specific rejection analysis
- Cost distribution for budget rejections
- Budget optimization analysis with impact modeling
- Export to CSV functionality

**Data Source:** `trading_signals` table for configurable time range (1, 7, 30 days)

## Configuration

### Report Generation Settings

No configuration file required. Reports auto-detect:

1. **Database Path:** `data/bigbrother.duckdb`
2. **Reports Directory:** `reports/` (auto-created)
3. **Date Handling:** Automatic based on system date

### Signal Generation Settings (in C++ code)

```yaml
# Thresholds affecting signal filtering
signal_filters:
  confidence_threshold: 0.60        # 60% minimum confidence
  return_threshold: 50.00            # $50 minimum expected return
  win_probability_threshold: 0.60   # 60% minimum win probability
  max_position_size: 500.00          # $500 position limit
  max_single_position_risk: 300.00  # Max loss per position
```

## API Reference

### DailyReportGenerator

```python
from scripts.reporting import DailyReportGenerator

# Create generator
gen = DailyReportGenerator(db_path="data/bigbrother.duckdb")

# Generate report
report = gen.generate_report()

# Save as JSON and HTML
json_path = gen.save_report(report, format='json')
html_path = gen.save_report(report, format='html')

# Print to console
gen.print_summary(report)

# Access components
exec_summary = gen.get_executive_summary()
trades = gen.get_trade_execution_details()
signals = gen.get_signal_analysis()
risk = gen.get_risk_compliance()
market = gen.get_market_conditions()
```

### WeeklyReportGenerator

```python
from scripts.reporting import WeeklyReportGenerator

# Current week
gen = WeeklyReportGenerator(week_offset=0)

# Last week
gen = WeeklyReportGenerator(week_offset=1)

# Generate report
report = gen.generate_report()

# Save as JSON and HTML
json_path = gen.save_report(report, format='json')
html_path = gen.save_report(report, format='html')

# Access components
perf = gen.get_performance_summary()
strat = gen.get_strategy_comparison()
acceptance = gen.get_signal_acceptance_rates()
risk = gen.get_risk_analysis()
recs = gen.get_recommendations(report)
```

## Usage Examples

### Daily Report via Command Line

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics

# Generate daily report
python scripts/reporting/generate_daily_report.py

# Output:
# ✅ JSON report saved: reports/daily_report_20251111.json
# ✅ HTML report saved: reports/daily_report_20251111.html
# DAILY TRADING REPORT - 2025-11-11
# ...
```

### Weekly Report via Command Line

```bash
# Current week
python scripts/reporting/generate_weekly_report.py

# Last week
python scripts/reporting/generate_weekly_report.py 1

# Last month's first week
python scripts/reporting/generate_weekly_report.py 4
```

### Python Integration

```python
from datetime import date
import json
from scripts.reporting import DailyReportGenerator, WeeklyReportGenerator

# Daily report programmatically
daily = DailyReportGenerator()
report = daily.generate_report()

# Access specific metrics
execution_rate = report['executive_summary']['signals']['execution_rate']
executed_trades = report['trade_execution']['summary']['total_executed']

# Weekly analysis
weekly = WeeklyReportGenerator(week_offset=0)
weekly_report = weekly.generate_report()

# Get recommendations
recs = weekly_report['recommendations']
for rec in recs:
    print(f"[{rec['severity']}] {rec['category']}: {rec['message']}")
```

### Automated Scheduling

```bash
# Add to crontab for daily reports
0 16 * * * cd /home/muyiwa/Development/BigBrotherAnalytics && python scripts/reporting/generate_daily_report.py >> reports/daily_reports.log 2>&1

# Add for weekly reports (Sunday at 4 PM)
0 16 * * 0 cd /home/muyiwa/Development/BigBrotherAnalytics && python scripts/reporting/generate_weekly_report.py 0 >> reports/weekly_reports.log 2>&1
```

## Report Outputs

### JSON Format

Structured data suitable for:
- Further analysis in Python/R
- Integration with other systems
- Data warehouse ingestion
- API consumption

**Size:** Typically 10-50KB per report

### HTML Format

Browser-viewable reports with:
- Formatted tables and metrics
- Color-coded status indicators
- Mobile-responsive layout
- Print-friendly styling

**Size:** Typically 20-100KB per report

## Key Metrics Explained

### Execution Rate
Percentage of generated signals that were actually traded:
```
Execution Rate = (Executed Signals / Total Signals) × 100
```

Target: 60-80% (too high = insufficient filters, too low = overly conservative)

### Risk/Reward Ratio
Risk efficiency of executed trades:
```
Risk/Reward = Expected Return / Total Risk
```

Higher is better. Target: > 1.0

### Sharpe Ratio (Simplified)
Return adjusted for risk:
```
Sharpe = Expected Return / Total Risk
```

Measures excess return per unit of risk.

### Signal Acceptance Rate by Strategy
Per-strategy execution metrics:
```
Acceptance = (Executed / Total) × 100
```

Identifies which strategies have the best signal quality.

### Budget Constraint Impact
Percentage of signals rejected due to $500 position limit:
```
Budget Rejection % = (Budget Rejected / Total) × 100
```

High values indicate budget constraint is limiting opportunity.

## Troubleshooting

### Reports Not Generated
1. Verify database exists: `ls -la data/bigbrother.duckdb`
2. Check permissions: Ensure read access to database
3. Verify trading_signals table: `python scripts/setup_trading_signals_table.py`

### No Signals Showing
1. Verify signals are being logged: Check `data/bigbrother.duckdb`
2. Confirm trading_signals table has data: Use DuckDB CLI
3. Check date in report matches trading date

### Empty Strategy List
Ensure strategies are logging signals with the `strategy` field populated in the C++ code.

### HTML Not Rendering
- Verify reports directory created: `mkdir -p reports`
- Check file permissions
- Open HTML in modern browser (Chrome, Firefox, Safari)

## Performance Characteristics

- **Daily Report Generation:** < 1 second (DuckDB is very fast)
- **Weekly Report Generation:** < 2 seconds
- **Database Query Time:** < 100ms for most queries
- **HTML Rendering:** < 500ms in browser

## Future Enhancements

Potential features for future versions:

1. **Email Distribution** - Automated report delivery
2. **Dashboard Widgets** - Real-time metric cards
3. **Slack Integration** - Alerts and summaries
4. **PDF Export** - Professional PDF reports
5. **Excel Export** - Spreadsheet format with formulas
6. **Multi-Period Comparison** - Period-over-period analysis
7. **Anomaly Detection** - Alert on unusual patterns
8. **Performance Attribution** - Break down returns by factor
9. **Backtesting Integration** - Compare live vs backtest metrics
10. **Custom Metrics** - User-defined metric calculation

## File Locations

| File | Purpose |
|------|---------|
| `scripts/reporting/generate_daily_report.py` | Daily report generator |
| `scripts/reporting/generate_weekly_report.py` | Weekly report generator |
| `scripts/reporting/__init__.py` | Package initialization |
| `scripts/database_schema_trading_signals.sql` | Schema definition |
| `scripts/setup_trading_signals_table.py` | Schema setup script |
| `dashboard/views/live_trading_activity.py` | Real-time dashboard view |
| `dashboard/views/rejection_analysis.py` | Rejection analysis view |
| `reports/` | Output directory for generated reports |
| `data/bigbrother.duckdb` | DuckDB database |

## Integration with Existing Systems

### Trading Engine (C++)
Logs signals to `trading_signals` table with:
- Signal generation timestamp
- Strategy identifier
- Confidence and win probability
- Greeks and IV metrics

### Dashboard (Streamlit)
Reads from `trading_signals` for:
- Live Activity view (real-time signals)
- Rejection Analysis view (rejection patterns)
- Performance graphs and metrics

### Risk Management
Uses risk metrics from reports to:
- Monitor per-position risk
- Track budget utilization
- Identify constraint impact

## References

- **Related Docs:** `docs/PRD.md`, `docs/PHASE5_SETUP_GUIDE.md`
- **Database Schema:** `scripts/database_schema_trading_signals.sql`
- **Dashboard:** `dashboard/app.py`
- **Configuration:** `config.yaml` (signal generation thresholds)

---

**Last Updated:** November 11, 2025
**Author:** Claude (BigBrother Analytics)
