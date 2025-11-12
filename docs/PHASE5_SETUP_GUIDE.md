# Phase 5 Setup Guide - Unified Script

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-10
**Status:** Production Ready

---

## One Command Setup

Instead of running multiple setup commands, use the unified Phase 5 setup script:

```bash
uv run python scripts/phase5_setup.py
```

This single script handles **all setup tasks**:
- ✅ **Automatic process cleanup** (stops old dashboard/trading engine)
- ✅ OAuth token management
- ✅ Tax configuration verification (married filing jointly, $300K base)
- ✅ Database initialization
- ✅ **Schwab portfolio sync** (fetches real positions ~$210K)
- ✅ Paper trading configuration check
- ✅ Schwab API connectivity test
- ✅ System component verification
- ✅ Comprehensive status report

---

## Usage Options

### Full Setup (First Time)
```bash
uv run python scripts/phase5_setup.py
```
**Time:** 2-3 minutes
**Does:** Complete initialization + database setup

### Quick Check + Auto-Start (Daily - Recommended)
```bash
uv run python scripts/phase5_setup.py --quick --start-all
```
**Time:** 10-15 seconds
**Does:** Verifies everything working + auto-starts dashboard & trading engine

### Quick Check Only (Verification)
```bash
uv run python scripts/phase5_setup.py --quick
```
**Time:** 10-15 seconds
**Does:** Verifies everything working, skips initialization, no auto-start

### Skip OAuth (If Offline)
```bash
uv run python scripts/phase5_setup.py --skip-oauth
```
**Time:** 30 seconds
**Does:** Skips OAuth and API tests

---

## What It Does

### 0. Process Cleanup ✅ (NEW)
- **Automatic cleanup of old processes** - No manual killing needed!
- Finds and stops old dashboard (streamlit) processes
- Finds and stops old trading engine (bigbrother) processes
- Finds and stops old news ingestion processes
- Graceful SIGTERM first, force kill if timeout
- Shows clear status for each stopped process

### 1. OAuth Token Management ✅
- **Automatic token refresh** - No manual intervention needed!
- Checks access token expiry (valid for 30 min)
- Checks refresh token expiry (valid for 7 days)
- Refreshes expired tokens automatically using Schwab API
- Shows clear status with color coding
- Only requires full re-auth if refresh token expired (every 7 days)

### 2. Tax Configuration ✅
- Verifies married filing jointly status
- Confirms $300K base income
- Validates tax rates: 24% ST / 15% LT federal
- Calculates effective rates: 37.1% ST / 28.1% LT

### 3. Database Initialization ✅
- Creates tax tables if missing
- Verifies tax_config table
- Confirms database structure

### 4. Paper Trading Configuration ✅
- Verifies paper trading enabled
- Shows trading limits ($2,000 max position)
- Confirms conservative settings

### 4.5. Schwab Portfolio Sync ✅ (NEW)
- **Automatic portfolio sync before dashboard starts**
- Fetches real positions from Schwab API (account 69398875)
- Syncs 10 positions totaling ~$210K:
  - QS: 6,078 shares @ $8.08 = $95,303
  - SNSXX (cash): 35,524 @ $1.00 = $35,524
  - QS Options, GOOGL, MSFT, INTC Options, VONG, NVDA, NFLX, ARM
- All positions marked as `is_bot_managed=false` (manual positions protected)
- Dashboard shows fresh data immediately on startup
- Timeout: 60 seconds (graceful failure if slow)

### 5. Schwab API Connectivity ✅
- Tests live market data API
- Gets real-time SPY quote
- Verifies authentication working

### 6. System Components ✅
- Dashboard (app.py)
- Orders Manager (orders_manager.cppm)
- Risk Manager (risk_management.cppm)
- Tax Calculator (calculate_taxes.py)
- Config files

---

## Output Example

```
======================================================================
                 BigBrotherAnalytics - Phase 5 Setup
======================================================================

   Date: 2025-11-10 16:49:54
   Directory: /home/muyiwa/Development/BigBrotherAnalytics
   Mode: Quick Check

======================================================================
                    Step 1: OAuth Token Management
======================================================================

   Access Token Status:
      Expires: 2025-11-11 01:06:48+00:00
      Valid for: 0.3 hours
✅ Access token valid for 0.3 hours

======================================================================
                Step 2: Tax Configuration Verification
======================================================================

   Tax Configuration:
      Filing Status: married_joint
      Tax Year: 2025
      Base Income: $300,000
      Short-term rate: 24.0% federal
      Long-term rate: 15.0% federal
      State tax: 5.0%
      Medicare surtax: 3.8%

   Effective Rates:
      Short-term: 32.8%
      Long-term: 23.8%
✅ Tax configuration correct (Married Filing Jointly)

======================================================================
                 Step 4: Paper Trading Configuration
======================================================================

✅ Paper trading enabled in config.yaml
✅ Paper trading config found: paper_trading.yaml
ℹ️  Paper trading limits:
      max_position_size: 100.0
      max_daily_loss: 100.0
      max_concurrent_positions: 2

======================================================================
                 Step 5: Schwab API Connectivity Test
======================================================================

ℹ️  Testing market data API...
✅ API connectivity verified
      SPY Quote: $681.71
      Volume: 75,806,554

======================================================================
                   Step 6: System Components Check
======================================================================

✅ Dashboard found
✅ Orders Manager found
✅ Risk Manager found
✅ Tax Calculator found
✅ Config found

======================================================================
                         Phase 5 Setup Report
======================================================================

Summary:
   Total Checks: 6
   Passed: 6
   Failed: 0
   Warnings: 0
   Success Rate: 100%

✅ Checks Passed:
   • OAuth access token valid
   • Tax rates configured correctly
   • Paper trading enabled
   • Paper trading config verified
   • Schwab API connectivity verified
   • All system components present

Overall Status:
✅ READY FOR PHASE 5 - All systems go! 🚀
ℹ️  Next steps:
   1. Start dashboard: uv run streamlit run dashboard/app.py
   2. Review execution plan: cat /tmp/phase5_execution_plan.md
   3. Begin Day 0 setup checklist
```

---

## Status Indicators

**Color Coding:**
- 🟢 **Green (✅)** - Check passed, all good
- 🟡 **Yellow (⚠️)** - Warning, review recommended
- 🔴 **Red (❌)** - Failed, must fix before proceeding
- 🔵 **Blue (ℹ️)** - Information, no action needed

**Overall Status:**
- **READY FOR PHASE 5** - All checks passed, proceed
- **MOSTLY READY** - Some warnings, review before proceeding
- **NOT READY** - Failed checks, fix issues first

---

## Typical Workflow

### Morning Pre-Market (Every Trading Day)
```bash
# Single command - verifies everything + auto-starts services (10-15 seconds)
uv run python scripts/phase5_setup.py --quick --start-all
```
**Expected:** All green checks, 100% success rate, dashboard + trading engine running

**Token refresh:** Happens automatically! No manual intervention needed.
- Expired access token? → Auto-refreshed in 2-3 seconds
- Expired refresh token (every 7 days)? → Prompts for full re-auth

### First Time Setup
```bash
# Full initialization
uv run python scripts/phase5_setup.py
```

### After System Changes
```bash
# Verify everything still working
uv run python scripts/phase5_setup.py
```

---

## Troubleshooting

### OAuth Token Expired
**Symptom:** Red ❌ on OAuth check (rarely happens now - auto-refresh!)

**Fix (Automatic):**
```bash
# Script auto-refreshes expired tokens - just run it!
uv run python scripts/phase5_setup.py --quick
```

**Manual Refresh (Only if automatic refresh fails):**
```bash
uv run python scripts/run_schwab_oauth_interactive.py
```

### Tax Configuration Wrong
**Symptom:** Warning on tax rates

**Fix:**
```bash
# Update to married filing jointly
uv run python scripts/monitoring/update_tax_rates_married.py

# Verify
uv run python scripts/phase5_setup.py --quick
```

### API Test Failed
**Symptom:** Red ❌ on API connectivity

**Possible Causes:**
1. Token expired → Run full setup
2. Network issue → Check internet
3. Schwab API down → Wait and retry

**Fix:**
```bash
# Refresh token
uv run python scripts/phase5_setup.py

# Skip API test if needed
uv run python scripts/phase5_setup.py --skip-oauth
```

### Component Missing
**Symptom:** Red ❌ on component check

**Fix:** Verify file exists:
```bash
ls -la src/schwab_api/orders_manager.cppm
ls -la src/risk_management/risk_management.cppm
ls -la dashboard/app.py
```

---

## Daily Checklist

**Every morning before market open:**

1. **Single command - verifies and starts everything**
   ```bash
   uv run python scripts/phase5_setup.py --quick --start-all
   ```

2. **Verify 100% success rate**
   - All 6 checks passed (including automatic token refresh if needed)
   - No warnings
   - Status: READY FOR PHASE 5
   - Dashboard running at http://localhost:8501
   - Trading engine running in background

3. **If any issues:**
   - Fix immediately
   - Re-run check
   - Don't trade until 100% pass

**Every evening at market close:**

6. **Run end-of-day shutdown**
   ```bash
   uv run python scripts/phase5_shutdown.py
   ```
   - Stops all trading processes gracefully
   - Generates EOD report with today's activity
   - Calculates taxes for closed trades
   - Backs up database (keeps last 7 days)
   - Cleans up temp files

**Force shutdown (skip confirmations):**
```bash
uv run python scripts/phase5_shutdown.py --force
```

**Skip database backup:**
```bash
uv run python scripts/phase5_shutdown.py --no-backup
```

---

## Integration with Phase 5

**Phase 5 Timeline:**
- **Day 0 (Today):** Run full setup → Verify 100% → Prepare
- **Day 1 (Monday):** Run quick check → Start dry-run
- **Day 2-3:** Run quick check daily → Monitor dry-run
- **Day 4+:** Run quick check daily → Execute paper trades

**The setup script is your "go/no-go" decision maker every day.**

---

## Advanced Usage

### Check Specific Component Only
The script runs all checks sequentially. To focus on specific issues, use flags:

```bash
# Skip OAuth (for offline testing)
uv run python scripts/phase5_setup.py --skip-oauth

# Quick mode (skip database init)
uv run python scripts/phase5_setup.py --quick

# Combine flags
uv run python scripts/phase5_setup.py --quick --skip-oauth
```

### Get Help
```bash
uv run python scripts/phase5_setup.py --help
```

### Exit Codes
- **0** - Success (all checks passed or warnings only)
- **1** - Failure (critical checks failed)

**Use in scripts:**
```bash
if uv run python scripts/phase5_setup.py --quick; then
    echo "System ready - starting trading"
    ./build/bigbrother
else
    echo "System not ready - fix issues first"
    exit 1
fi
```

---

## Key Benefits

✅ **One Command** - No more running 5-10 separate scripts
✅ **Fast** - Quick mode takes 10-15 seconds
✅ **Comprehensive** - Checks all critical systems
✅ **Clear Output** - Color-coded, easy to understand
✅ **Automated** - Auto-refresh OAuth when possible
✅ **Go/No-Go** - Clear decision on readiness
✅ **Daily Use** - Perfect for pre-market checks

---

## Summary

**Replace all these:**
```bash
# OLD WAY - Morning Setup (multiple commands, manual token refresh)
uv run python /tmp/refresh_schwab_token.py
uv run python scripts/monitoring/update_tax_config_ytd.py
uv run python scripts/monitoring/setup_tax_database.py
uv run python scripts/test_schwab_api_live.py
uv run streamlit run dashboard/app.py
./build/bigbrother
# ... and 5 more commands

# OLD WAY - Evening Shutdown (manual process)
killall bigbrother
killall streamlit
# ... manually stop each process
```

**With these:**
```bash
# NEW WAY - Morning Setup (one command with automatic token refresh!)
uv run python scripts/phase5_setup.py --quick --start-all

# NEW WAY - Evening Shutdown (one command)
uv run python scripts/phase5_shutdown.py
```

**Phase 5 paper trading is now 10x easier:**
- ✅ Morning: 1 command verifies all systems + auto-refreshes token + starts services (10-15 seconds)
- ✅ Evening: 1 command stops everything gracefully + reports + backup
- ✅ Token management: 100% automatic (no manual intervention for 7 days)
- ✅ Health monitoring: Real-time token validation and system checks

---

**Created:** 2025-11-10
**Scripts:**
- Setup: [scripts/phase5_setup.py](../scripts/phase5_setup.py)
- Shutdown: [scripts/phase5_shutdown.py](../scripts/phase5_shutdown.py)
**Documentation:** This file

**Next Step:** Run `uv run python scripts/phase5_setup.py` and begin Phase 5!
