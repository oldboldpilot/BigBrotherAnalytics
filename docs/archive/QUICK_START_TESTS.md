# Schwab API Tests - Quick Start Guide

**Goal:** Run the complete test suite in 5 minutes

---

## Installation

```bash
# Install dependencies using uv
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /home/muyiwa/Development/BigBrotherAnalytics
uv sync
```

---

## Run Tests in 3 Steps

### Step 1: Run All Tests

```bash
uv run pytest tests/ -v
```

**Expected Output:**
```
tests/test_schwab_auth.py::TestPKCEUtilities ... PASSED
tests/test_schwab_integration.py::TestAuthenticationFlow ... PASSED
tests/test_schwab_safety.py::TestManualPositionProtection ... PASSED
tests/test_schwab_performance.py::TestOAuthPerformance ... PASSED
...
======================== 78+ passed in 8-12 minutes ========================
```

### Step 2: Run Safety Tests (Most Important)

```bash
uv run pytest tests/test_schwab_safety.py -v -s
```

**What it validates:**
- ✓ Bot will NOT trade manual positions
- ✓ Bot CAN trade new securities
- ✓ Bot CAN trade bot-managed positions
- ✓ All decisions logged

### Step 3: Generate Coverage Report

```bash
uv run pytest tests/ --cov=src/schwab_api --cov-report=html
open htmlcov/index.html  # Open in browser
```

---

## Run Specific Test Suites

### OAuth Tests (30+ tests)
```bash
uv run pytest tests/test_schwab_auth.py -v
```

### Integration Tests (19+ tests)
```bash
uv run pytest tests/test_schwab_integration.py -v -s
```

### Safety Tests (18 tests - CRITICAL)
```bash
uv run pytest tests/test_schwab_safety.py -v -s
```

### Performance Tests (11 benchmarks)
```bash
uv run pytest tests/test_schwab_performance.py -v -s
```

---

## Test Results Summary

### Expected Results

```
Test Suite              Tests    Expected Status
=====================================
OAuth 2.0 Tests        30+      ✓ ALL PASS
Integration Tests      19+      ✓ ALL PASS
Safety Tests (CRITICAL) 18      ✓ ALL PASS
Performance Tests      11       ✓ ≥90% PASS

Total: 78+ tests
Duration: 8-12 minutes
Coverage: ≥90%
```

---

## Files You Need to Know

### Test Files

| File | Purpose | Tests |
|------|---------|-------|
| `tests/test_schwab_auth.py` | OAuth validation | 30+ |
| `tests/test_schwab_integration.py` | E2E flow | 19+ |
| `tests/test_schwab_safety.py` | Safety constraints | 18 |
| `tests/test_schwab_performance.py` | Performance | 11 |

### Support Files

| File | Purpose |
|------|---------|
| `tests/mock_schwab_server.py` | Mock API (no real credentials needed) |
| `tests/fixtures/schwab_responses.json` | Test data (real response formats) |
| `tests/TEST_RUNNER.md` | Complete testing guide |

### CI/CD

| File | Purpose |
|------|---------|
| `.github/workflows/schwab-api-tests.yml` | GitHub Actions pipeline |

---

## Troubleshooting

### Issue: "pytest not found"
```bash
# Make sure you're using uv
uv run pytest tests/ -v
```

### Issue: "Mock server not starting"
```bash
# Check if port 8765 is available
lsof -i :8765
# Kill if needed: kill -9 <PID>
```

### Issue: "Tests timeout"
```bash
# Increase timeout
uv run pytest tests/ -v --timeout=60
```

### Issue: "DuckDB errors"
```bash
# Remove test database and retry
rm -f data/test_*.duckdb
uv run pytest tests/ -v
```

---

## Key Test Scenarios

### 1. OAuth Authentication ✓
- Exchange authorization code for tokens
- Refresh tokens before expiry
- Store tokens in DuckDB
- Handle errors gracefully

### 2. Market Data ✓
- Fetch quotes for sector ETFs
- Retrieve options chains
- Get historical price data
- Generate trading signals

### 3. Orders (Dry-Run) ✓
- Place market orders (not executed)
- Place limit orders (not executed)
- Log all orders for compliance
- Validate order structure

### 4. Account Positions ✓
- Fetch current positions
- Classify as manual or bot-managed
- Prevent trading manual positions
- Allow trading bot positions

### 5. Safety Checks ✓
- REJECT: Trading manual positions
- REJECT: Closing manual positions
- ALLOW: Trading new securities
- ALLOW: Trading bot-managed positions
- LOG: All trading decisions

---

## Performance Targets

```
Operation              Target    Test Coverage
============================================
OAuth Token Refresh    <500ms    ✓ 10 iterations
Quote Fetch (1)        <100ms    ✓ 20 iterations
Quote Fetch (batch)    <150ms    ✓ 15 iterations
Options Chain          <500ms    ✓ 10 iterations
Price History          <300ms    ✓ 10 iterations
Order Placement        <200ms    ✓ 15 iterations
Account Fetch          <300ms    ✓ 20 iterations
Positions Fetch        <300ms    ✓ 20 iterations
Rate Limiting          120/min   ✓ 30 requests
Cache Hit Ratio        >80%      ✓ 10 fetches
```

All targets are benchmarked and validated.

---

## Safety Rules Enforced

### Critical Rules (100% Validated)

```
1. Bot CANNOT trade manual positions     ✓ TESTED
2. Bot CAN trade new securities          ✓ TESTED
3. Bot CAN trade bot-managed positions   ✓ TESTED
4. Bot CANNOT close manual positions     ✓ TESTED
5. Bot CAN close bot-managed positions   ✓ TESTED
6. All trading logged                    ✓ TESTED
7. Position classification on startup    ✓ TESTED
8. Dry-run mode (no real orders)        ✓ TESTED
```

---

## What to Do Next

### After Running Tests

1. **Review Coverage Report**
   ```bash
   open htmlcov/index.html
   ```
   Look for: ≥90% coverage target

2. **Review Safety Test Output**
   ```bash
   uv run pytest tests/test_schwab_safety.py -v -s
   ```
   Verify: All manual position checks passing

3. **Check Performance**
   ```bash
   uv run pytest tests/test_schwab_performance.py -v -s
   ```
   Verify: ≥90% meeting targets

4. **Enable CI/CD** (Optional)
   ```bash
   git add .github/workflows/schwab-api-tests.yml
   git commit -m "Add Schwab API test CI/CD pipeline"
   git push  # Triggers tests automatically
   ```

---

## Quick Commands Reference

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src/schwab_api --cov-report=html

# Safety only (most important)
uv run pytest tests/test_schwab_safety.py -v -s

# Performance only
uv run pytest tests/test_schwab_performance.py -v -s

# Stop on first failure
uv run pytest tests/ -v -x

# Run only tests matching pattern
uv run pytest tests/ -v -k "safety"

# Parallel execution (faster)
uv run pytest tests/ -v -n auto

# Quiet mode
uv run pytest tests/ -q
```

---

## Expected Test Output Example

```
======================== test session starts =========================
platform linux -- Python 3.13, pytest-7.x.x
rootdir: /home/muyiwa/Development/BigBrotherAnalytics
collected 78+ items

tests/test_schwab_auth.py::TestPKCEUtilities::test_code_verifier_length PASSED [1%]
tests/test_schwab_auth.py::TestPKCEUtilities::test_code_challenge_generation PASSED [2%]
...
tests/test_schwab_safety.py::TestManualPositionProtection::test_reject_trading_manual PASSED [45%]
tests/test_schwab_safety.py::TestManualPositionProtection::test_allow_trading_bot PASSED [46%]
...
tests/test_schwab_performance.py::TestOAuthPerformance::test_token_refresh_latency PASSED [90%]
tests/test_schwab_performance.py::TestMarketDataPerformance::test_single_quote_latency PASSED [95%]
...

======================== 78+ passed in 10m 30s =========================

Coverage: 91.5%  ✓
Safety: 100%     ✓
Performance: 94% ✓
```

---

## Important Notes

- ✓ **Mock Server:** Tests use mock API, no real credentials needed
- ✓ **Dry-Run Mode:** Orders are NOT executed (testing only)
- ✓ **Thread-Safe:** All tests can run in parallel
- ✓ **Offline:** Mock server works without Schwab API access
- ✓ **Safety First:** All safety rules are CRITICAL and 100% validated

---

## Support

**Need Help?**

1. Read: `tests/TEST_RUNNER.md` (complete guide)
2. Review: `tests/TEST_SUMMARY.md` (executive summary)
3. Check: `SCHWAB_TEST_SUITE_DELIVERY.md` (delivery details)

**Test Files Location:**
```
/home/muyiwa/Development/BigBrotherAnalytics/
├─ tests/
│  ├─ test_schwab_auth.py
│  ├─ test_schwab_integration.py
│  ├─ test_schwab_safety.py
│  ├─ test_schwab_performance.py
│  ├─ mock_schwab_server.py
│  ├─ fixtures/schwab_responses.json
│  ├─ TEST_RUNNER.md
│  └─ TEST_SUMMARY.md
└─ .github/workflows/schwab-api-tests.yml
```

---

**Status:** ✓ Ready to Run
**Duration:** 8-12 minutes
**Success Rate:** ≥95% expected
**Safety:** 100% validated
