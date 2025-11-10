# BigBrother Dashboard - Quick Start

## Launch Dashboard

```bash
cd /home/muyiwa/Development/BigBrotherAnalytics/dashboard
./run_dashboard.sh
```

Then open: **http://localhost:8501**

## Dashboard Views

1. **Overview** - Quick summary of all components
2. **Positions** - Real-time position monitoring
3. **P&L Analysis** - Performance charts
4. **Employment Signals** - Sector rotation signals
5. **Trade History** - Historical activity

## Components

### ✅ Component 1: Real-Time Positions
- 25 active positions
- Color-coded P&L (Green/Red/Gray)
- Summary metrics and charts

### ✅ Component 2: P&L Charts
- Total P&L: $5,796.55
- Win/Loss ratio charts
- Daily and cumulative trends

### ✅ Component 3: Employment Signals
- 11 GICS sectors with ETF symbols
- 3-month growth rates
- OVERWEIGHT/MARKET WEIGHT/AVOID signals

### ✅ Component 4: Trade History
- 275 historical records
- Advanced filtering
- Activity charts

## Testing

```bash
# Validate components
uv run python dashboard/test_dashboard.py

# Generate sample data
uv run python dashboard/generate_sample_data.py
```

## Files

- `app.py` (721 lines) - Main dashboard
- `README.md` - Full documentation
- `DEPLOYMENT_REPORT.md` - Complete deployment guide
- `generate_sample_data.py` - Sample data generator
- `test_dashboard.py` - Component tests
- `run_dashboard.sh` - Launch script

## Status

✅ All 4 components implemented
✅ All tests passing
✅ Sample data generated
✅ Production ready

**Location:** `/home/muyiwa/Development/BigBrotherAnalytics/dashboard/`
