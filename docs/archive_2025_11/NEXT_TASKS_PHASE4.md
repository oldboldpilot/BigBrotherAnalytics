# Next Tasks - BigBrotherAnalytics (Phase 4)

**Date:** November 10, 2025
**Author:** Olumuyiwa Oluwasanmi
**Previous Phase:** Phase 3 Complete (98% Production Ready)
**Current Status:** Ready for Paper Trading & Production Hardening
**Dashboard:** 🟢 LIVE at http://localhost:8501

---

## 🎯 Phase 4 Goals

**Primary Objective:** Validate trading system with paper trading and harden for production

**Success Criteria:**
- Paper trading operational with real Schwab account
- $150+/day profit target validated
- All safety systems tested in live conditions
- Dashboard monitoring confirmed operational
- 80%+ winning days achieved

**Timeline:** 2-3 weeks (Nov 11 - Nov 30, 2025)

---

## 🔴 CRITICAL PRIORITY - Paper Trading Validation

### **TASK 1: Paper Trading Setup & Testing (Week 1: Nov 11-15)**

**Goal:** Begin small-scale paper trading with full monitoring

#### 1.1 Schwab API Live Connection (Day 1-2)
**Priority:** CRITICAL
**Time:** 4-6 hours

**Steps:**
1. Verify Schwab OAuth tokens are current
2. Test live API connection (not mock)
3. Validate account access (read-only first)
4. Test market data fetching (real-time quotes)
5. Verify position retrieval works

**Validation:**
```bash
# Test Schwab connection
./bin/test_schwab_e2e_workflow

# Should connect to real API (not mock)
# Verify:
# - OAuth token refresh works
# - Account balance retrieved
# - Current positions listed
# - Market data streams
```

**Deliverables:**
- [ ] Live Schwab API connection verified
- [ ] OAuth token refresh tested
- [ ] Account data retrieved successfully
- [ ] Market data streaming works

---

#### 1.2 Paper Trading Dry-Run (Day 3-4)
**Priority:** CRITICAL
**Time:** 8-10 hours

**Steps:**
1. Run bigbrother in dry-run mode
   ```bash
   LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/bigbrother --dry-run
   ```
2. Monitor signal generation via dashboard
3. Verify no orders placed (dry-run mode)
4. Check employment signals appear correctly
5. Validate sector rotation logic
6. Test stop-loss calculation
7. Verify manual position protection

**Validation Checklist:**
- [ ] Signals generated for all 11 sectors
- [ ] Dashboard shows signals in real-time
- [ ] No actual orders placed
- [ ] Employment data influences signals
- [ ] Manual positions are skipped
- [ ] Position sizing follows Kelly Criterion
- [ ] Stop-losses calculated at 10%

**Expected Output:**
- Signal logs showing sector recommendations
- Dashboard displaying OVERWEIGHT/UNDERWEIGHT sectors
- Zero orders in Schwab account (dry-run)

---

#### 1.3 Small Position Paper Trading (Day 5-7)
**Priority:** HIGH
**Time:** Daily monitoring (30 min/day)

**Steps:**
1. Enable paper trading (remove --dry-run flag)
2. Start with $50-100 position sizes
3. Monitor dashboard continuously
4. Verify orders execute correctly
5. Track P&L in real-time
6. Test stop-loss triggers
7. Monitor for 3 consecutive days

**Starting Configuration:**
```yaml
# configs/paper_trading.yaml
account:
  max_position_size: $100
  max_daily_loss: $50
  paper_trading: true
```

**Daily Monitoring:**
- Morning: Check overnight positions
- Midday: Monitor signal changes
- Close: Review P&L and positions

**Validation:**
- [ ] First order executes successfully
- [ ] Dashboard shows position immediately
- [ ] P&L updates in real-time
- [ ] Stop-loss triggers correctly
- [ ] Manual positions remain untouched
- [ ] 3 consecutive days of successful trading

**Success Metrics:**
- Win rate: >50% (realistic for first week)
- No manual position violations
- All orders logged correctly
- Dashboard accuracy: 100%

---

### **TASK 2: Production Hardening (Week 2: Nov 16-22)**

**Goal:** Make system robust for continuous operation

#### 2.1 Error Handling & Retry Logic
**Priority:** HIGH
**Time:** 8-10 hours

**Components to Harden:**

1. **Schwab API Calls**
   ```cpp
   // Add exponential backoff retry
   auto fetchQuote(std::string symbol) -> Result<Quote> {
       int retries = 3;
       for (int i = 0; i < retries; i++) {
           auto result = schwab_api.getQuote(symbol);
           if (result.has_value()) return result;
           std::this_thread::sleep_for(std::chrono::seconds(std::pow(2, i)));
       }
       return makeError("Max retries exceeded");
   }
   ```

2. **Database Operations**
   - Add transaction retry logic
   - Handle database lock errors
   - Implement graceful degradation

3. **Network Failures**
   - Detect connection drops
   - Auto-reconnect to Schwab API
   - Continue from last known state

**Deliverables:**
- [ ] Retry logic in all API calls
- [ ] Database transaction handling
- [ ] Network failure recovery
- [ ] Graceful degradation paths

---

#### 2.2 Circuit Breaker Pattern
**Priority:** HIGH
**Time:** 6-8 hours

**Implementation:**
```cpp
class CircuitBreaker {
    enum State { CLOSED, OPEN, HALF_OPEN };
    State state = CLOSED;
    int failure_count = 0;
    int failure_threshold = 5;

    auto call(auto func) -> Result<auto> {
        if (state == OPEN) {
            if (should_attempt_reset()) {
                state = HALF_OPEN;
            } else {
                return makeError("Circuit breaker open");
            }
        }

        auto result = func();
        if (result.has_value()) {
            on_success();
        } else {
            on_failure();
        }
        return result;
    }
};
```

**Apply To:**
- Schwab API calls
- Database queries
- External data sources (BLS, FRED)

**Deliverables:**
- [ ] Circuit breaker implemented
- [ ] Applied to all critical paths
- [ ] Dashboard shows circuit state
- [ ] Alerts on circuit open

---

#### 2.3 Performance Optimization
**Priority:** MEDIUM
**Time:** 8-10 hours

**Optimization Targets:**

1. **Signal Generation**
   - Cache employment data (refresh daily)
   - Parallelize sector calculations
   - Use correlation cache

2. **Database Queries**
   - Add indexes on frequently queried columns
   - Use prepared statements
   - Batch insert position history

3. **Dashboard Loading**
   - Lazy load chart data
   - Cache static queries
   - Optimize position refresh rate

**Target Metrics:**
- Signal generation: <500ms
- Order placement: <200ms
- Dashboard refresh: <1s
- Database queries: <50ms

**Deliverables:**
- [ ] Signal generation optimized
- [ ] Database indexes added
- [ ] Dashboard performance improved
- [ ] All metrics achieved

---

#### 2.4 Stress Testing
**Priority:** MEDIUM
**Time:** 4-6 hours

**Test Scenarios:**

1. **High Volatility**
   - Simulate rapid price changes
   - Test stop-loss triggers
   - Verify order execution under load

2. **Market Open/Close**
   - Test 9:30 AM ET spike
   - Verify 4:00 PM ET cleanup
   - Check after-hours handling

3. **Multiple Positions**
   - Open 10 positions simultaneously
   - Monitor dashboard performance
   - Test P&L calculation accuracy

4. **Network Issues**
   - Disconnect during order
   - Reconnect and verify state
   - Check order completion

**Deliverables:**
- [ ] Stress test scenarios defined
- [ ] All tests passed
- [ ] Issues documented and fixed
- [ ] System remains stable

---

### **TASK 3: Dashboard Enhancements (Week 2-3: Nov 18-29)**

**Goal:** Improve dashboard usability and add features

#### 3.1 Mobile Responsiveness
**Priority:** MEDIUM
**Time:** 4-6 hours

**Updates:**
- Responsive layouts for all 5 views
- Touch-friendly controls
- Mobile-optimized charts
- Collapsible sidebar

**Testing:**
- [ ] iPhone Safari
- [ ] Android Chrome
- [ ] iPad
- [ ] Desktop browsers

---

#### 3.2 Additional Chart Types
**Priority:** LOW
**Time:** 4-6 hours

**New Charts:**
1. Candlestick chart for price history
2. Volume profile chart
3. Drawdown chart (max loss over time)
4. Sharpe ratio chart (rolling 30-day)
5. Correlation heatmap (live)

**Deliverables:**
- [ ] 5 new chart types added
- [ ] All charts interactive
- [ ] Data updates in real-time

---

#### 3.3 Custom Alerts
**Priority:** MEDIUM
**Time:** 6-8 hours

**Alert Types:**
1. P&L threshold reached ($X profit/loss)
2. Stop-loss triggered
3. New signal generated (sector rotation)
4. Employment data updated
5. Circuit breaker opened

**Delivery Methods:**
- Browser notification
- Email (via SMTP)
- Slack (via webhook)
- SMS (via Twilio - optional)

**Deliverables:**
- [ ] Alert system implemented
- [ ] All alert types working
- [ ] Email/Slack configured
- [ ] Alert history in dashboard

---

### **TASK 4: Automated Operations (Week 3: Nov 23-29)**

**Goal:** Ensure system runs autonomously

#### 4.1 Install Cron Jobs
**Priority:** HIGH
**Time:** 2-3 hours

**Jobs to Install:**

1. **Daily BLS Data Update** (10:00 AM ET)
   ```bash
   0 10 * * * cd /home/muyiwa/Development/BigBrotherAnalytics && uv run python scripts/automated_updates/daily_employment_update.py
   ```

2. **Weekly Jobless Claims** (10:30 AM ET Thursday)
   ```bash
   30 10 * * 4 cd /home/muyiwa/Development/BigBrotherAnalytics && uv run python scripts/data_collection/bls_jobless_claims.py
   ```

3. **Daily Signal Recalculation** (6:00 AM ET)
   ```bash
   0 6 * * * cd /home/muyiwa/Development/BigBrotherAnalytics && uv run python scripts/automated_updates/recalculate_signals.py
   ```

4. **Weekly Correlation Update** (Sunday 2:00 AM ET)
   ```bash
   0 2 * * 0 cd /home/muyiwa/Development/BigBrotherAnalytics && uv run python scripts/analysis/discover_correlations.py
   ```

**Setup:**
```bash
cd scripts/automated_updates
./setup_cron.sh
```

**Deliverables:**
- [ ] All cron jobs installed
- [ ] Tested once manually
- [ ] Logs directory created
- [ ] Log rotation configured

---

#### 4.2 Monitoring & Health Checks
**Priority:** HIGH
**Time:** 6-8 hours

**Health Check Endpoints:**

1. **System Health**
   - CPU/Memory usage
   - Disk space
   - Database size
   - Process status

2. **Trading Health**
   - Schwab API status
   - Last signal time
   - Last order time
   - Position count

3. **Data Health**
   - Last BLS update
   - Employment data age
   - Database integrity
   - Correlation freshness

**Implementation:**
```python
# scripts/monitoring/health_check.py
def check_system_health():
    return {
        "schwab_api": check_schwab_connection(),
        "database": check_database_connection(),
        "last_signal": get_last_signal_time(),
        "disk_space": get_disk_space(),
        "active_positions": count_positions()
    }
```

**Alerting:**
- Email if any check fails
- Slack notification on critical issues
- Dashboard displays health status

**Deliverables:**
- [ ] Health check script created
- [ ] Endpoint accessible
- [ ] Alerts configured
- [ ] Dashboard health view added

---

#### 4.3 Log Management
**Priority:** MEDIUM
**Time:** 3-4 hours

**Log Structure:**
```
logs/
├── trading/
│   ├── bigbrother_2025-11-10.log
│   ├── orders_2025-11-10.log
│   └── signals_2025-11-10.log
├── automated_updates/
│   ├── daily_employment_update_20251110.log
│   └── bls_data_load.log
├── dashboard/
│   └── streamlit_2025-11-10.log
└── monitoring/
    └── health_check_2025-11-10.log
```

**Log Rotation:**
- Daily rotation
- Keep 30 days
- Compress old logs
- Max 10GB total

**Setup:**
```bash
# /etc/logrotate.d/bigbrother
/home/muyiwa/Development/BigBrotherAnalytics/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
}
```

**Deliverables:**
- [ ] Log structure created
- [ ] Rotation configured
- [ ] Old logs cleaned
- [ ] Monitoring dashboard added

---

## 🟡 MEDIUM PRIORITY - Feature Enhancements

### **TASK 5: Advanced Trading Features**

**Goal:** Add sophisticated trading strategies

#### 5.1 Time-Based Strategy Filters
**Priority:** MEDIUM
**Time:** 6-8 hours

**Filters:**
1. **Market Hours Only** - No orders before 9:30 AM or after 4:00 PM ET
2. **Avoid First 15 Min** - Skip 9:30-9:45 AM volatility
3. **Power Hour** - Increase position size 3:00-4:00 PM
4. **Friday Close** - Reduce exposure before weekend

**Implementation:**
```cpp
auto shouldTrade() -> bool {
    auto now = std::chrono::system_clock::now();
    auto hour = getHour(now);
    auto minute = getMinute(now);
    auto day = getDayOfWeek(now);

    // Market hours only
    if (hour < 9 || hour >= 16) return false;

    // Skip first 15 minutes
    if (hour == 9 && minute < 45) return false;

    // Reduce Friday exposure
    if (day == Friday && hour >= 15) return false;

    return true;
}
```

**Deliverables:**
- [ ] Time filters implemented
- [ ] Configurable in YAML
- [ ] Tested in paper trading
- [ ] Dashboard shows active filters

---

#### 5.2 Volatility-Based Position Sizing
**Priority:** MEDIUM
**Time:** 8-10 hours

**Strategy:**
- Calculate sector volatility (rolling 30-day)
- Reduce position size in high volatility
- Increase size in low volatility

**Formula:**
```
position_size = base_size * (target_vol / current_vol)
```

**Implementation:**
```cpp
auto calculateVolatilityAdjustedSize(
    double base_size,
    std::string sector_etf) -> double {

    auto volatility = calculateRollingVolatility(sector_etf, 30);
    double target_vol = 0.15; // 15% target volatility
    double adjustment = std::min(2.0, target_vol / volatility);

    return base_size * adjustment;
}
```

**Deliverables:**
- [ ] Volatility calculation implemented
- [ ] Position sizing adjusted
- [ ] Tested with historical data
- [ ] Dashboard shows volatility metrics

---

#### 5.3 Multi-Strategy Blending
**Priority:** LOW
**Time:** 10-12 hours

**Strategies to Blend:**
1. **Employment Rotation** (current, 60% weight)
2. **Momentum** (new, 25% weight)
3. **Mean Reversion** (new, 15% weight)

**Blending Logic:**
```cpp
auto blendSignals(
    Signal employment,
    Signal momentum,
    Signal mean_reversion) -> Signal {

    double composite =
        0.60 * employment.score +
        0.25 * momentum.score +
        0.15 * mean_reversion.score;

    return Signal{composite, employment.sector};
}
```

**Deliverables:**
- [ ] Momentum strategy implemented
- [ ] Mean reversion strategy implemented
- [ ] Blending logic added
- [ ] Backtested on historical data

---

## 🟢 LOW PRIORITY - Nice to Have

### **TASK 6: Documentation & Training**

#### 6.1 Video Walkthrough
- Record dashboard demo (10 min)
- System architecture overview (15 min)
- Paper trading tutorial (20 min)

#### 6.2 API Documentation
- Generate C++ API docs (Doxygen)
- Python API docs (Sphinx)
- REST API docs (OpenAPI)

#### 6.3 Deployment Guide
- Docker containerization
- Kubernetes deployment
- Cloud migration guide

---

## 📊 Success Metrics (Phase 4)

### Week 1 (Paper Trading Setup)
- [ ] Schwab API connected
- [ ] Dry-run successful (3 days)
- [ ] First paper trade executed
- [ ] 3 consecutive trading days completed
- [ ] Dashboard monitoring confirmed

### Week 2 (Production Hardening)
- [ ] Retry logic implemented
- [ ] Circuit breaker deployed
- [ ] Performance optimized
- [ ] Stress tests passed
- [ ] Error rate < 1%

### Week 3 (Automation & Enhancements)
- [ ] Cron jobs installed
- [ ] Health checks operational
- [ ] Logs managed
- [ ] Alerts configured
- [ ] System runs autonomously

### Overall Phase 4 Goals
- [ ] Paper trading operational (1 week)
- [ ] 80% winning days achieved
- [ ] $150+/day target validated
- [ ] Zero manual position violations
- [ ] Dashboard accuracy 100%
- [ ] System uptime >99%

---

## 🚨 Risk Mitigation

### High Risk Areas

1. **Schwab API Changes**
   - **Risk:** API endpoints change without notice
   - **Mitigation:** Monitor API version, implement graceful degradation
   - **Backup:** Manual trading via Schwab website

2. **Data Feed Interruptions**
   - **Risk:** BLS data not available on release day
   - **Mitigation:** Cache last known data, alert on staleness
   - **Backup:** Use previous month's data temporarily

3. **System Downtime**
   - **Risk:** Server crash during market hours
   - **Mitigation:** Health checks, auto-restart, alerts
   - **Backup:** Cloud failover (future)

4. **Erroneous Signals**
   - **Risk:** Bad data causes wrong trading signals
   - **Mitigation:** Data validation, sanity checks, circuit breaker
   - **Backup:** Manual override capability

---

## 📅 Timeline Summary

| Week | Focus | Key Deliverables |
|------|-------|-----------------|
| Week 1 (Nov 11-15) | Paper Trading Setup | Schwab API live, dry-run, first trades |
| Week 2 (Nov 16-22) | Production Hardening | Retry logic, circuit breaker, optimization |
| Week 3 (Nov 23-29) | Automation & Enhancement | Cron jobs, health checks, dashboard features |

**Target Completion:** November 30, 2025
**Next Phase:** Phase 5 - Scale to Production (Target: $150+/day)

---

## 🔗 Resources

**Documentation:**
- Phase 3 Summary: [PROJECT_STATUS_2025-11-10_PHASE3.md](PROJECT_STATUS_2025-11-10_PHASE3.md)
- Agent Reports: `/tmp/agent*_report.md`
- Current Status: [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md)
- Trading Constraints: [docs/TRADING_CONSTRAINTS.md](docs/TRADING_CONSTRAINTS.md)

**Key Files:**
- Dashboard: [dashboard/app.py](dashboard/app.py)
- Main Trading: [src/main.cpp](src/main.cpp)
- Employment Signals: [scripts/employment_signals.py](scripts/employment_signals.py)
- Automation: [scripts/automated_updates/](scripts/automated_updates/)

**Quick Commands:**
```bash
# Dashboard
cd dashboard && ./run_dashboard.sh

# Paper Trading (dry-run)
cd build
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/bigbrother --dry-run

# Paper Trading (live)
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/bigbrother

# Run tests
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./bin/test_schwab_e2e_workflow

# Check health
uv run python scripts/monitoring/health_check.py
```

---

**Status:** 🟢 Ready to Begin Phase 4
**Last Updated:** November 10, 2025
**Next Review:** November 15, 2025 (After Week 1)
