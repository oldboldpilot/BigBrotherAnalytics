# Python Tests Directory

This directory contains all Python test scripts for BigBrotherAnalytics components.

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-11
**Purpose:** Centralized testing suite for system validation

---

## Test Scripts

### 1. Account & Portfolio Tests

#### `test_account.py`
**Purpose:** Test Schwab account integration and portfolio management
**Tests:**
- Account connection and authentication
- Portfolio data retrieval
- Position tracking
- Account balance queries

**Run:**
```bash
uv run python python_tests/test_account.py
```

---

### 2. Risk Management Tests

#### `test_risk_bindings.py`
**Purpose:** Test C++ risk management Python bindings
**Tests:**
- Position sizing calculations
- Risk limits enforcement
- Portfolio risk metrics
- Stop-loss calculations

**Run:**
```bash
uv run python python_tests/test_risk_bindings.py
```

**Dependencies:** Requires C++ build (`ninja -C build`)

---

### 3. Market Data Tests

#### `test_market_data.py`
**Purpose:** Test market data retrieval and processing
**Tests:**
- Real-time quote fetching
- Historical data retrieval
- Options chain queries
- Market data validation

**Run:**
```bash
uv run python python_tests/test_market_data.py
```

---

### 4. Order Management Tests

#### `test_orders.py`
**Purpose:** Test order creation and execution workflow
**Tests:**
- Order validation
- Paper trading orders
- Order status tracking
- Order rejection handling

**Run:**
```bash
uv run python python_tests/test_orders.py
```

---

### 5. Database Tests

#### `test_duckdb_bindings.py`
**Purpose:** Test DuckDB Python bindings and database operations
**Tests:**
- Database connection
- Schema validation
- Query execution
- Data persistence

**Run:**
```bash
uv run python python_tests/test_duckdb_bindings.py
```

---

### 6. Sentiment Analysis Tests

#### `test_sentiment_keywords.py`
**Purpose:** Test keyword-based sentiment analysis engine
**Tests:**
- Positive/negative keyword detection
- Sentiment scoring (-1.0 to +1.0)
- Negation handling
- Intensifier detection

**Run:**
```bash
uv run python python_tests/test_sentiment_keywords.py
```

#### `test_sentiment_manual.py`
**Purpose:** Manual testing of sentiment analysis on sample articles
**Tests:**
- Real-world article analysis
- Sentiment label assignment
- Score accuracy validation

**Run:**
```bash
uv run python python_tests/test_sentiment_manual.py
```

---

### 7. News & Data Collection Tests

#### `test_source_filtering.py`
**Purpose:** Test news source filtering and prioritization
**Tests:**
- Source quality scoring
- Duplicate detection
- Source blacklist/whitelist
- Article relevance filtering

**Run:**
```bash
uv run python python_tests/test_source_filtering.py
```

---

### 8. Signal Generation Tests

#### `test_signal_generation.py`
**Purpose:** Test trading signal generation logic
**Tests:**
- Entry/exit signal generation
- Confidence scoring
- Signal validation
- Signal persistence

**Run:**
```bash
uv run python python_tests/test_signal_generation.py
```

---

### 9. Employment Data Tests

#### `test_employment_pipeline.py`
**Purpose:** Test BLS employment data pipeline
**Tests:**
- BLS API data retrieval
- Employment data parsing
- Sector classification
- Growth rate calculations

**Run:**
```bash
uv run python python_tests/test_employment_pipeline.py
```

---

### 10. Strategy Tests

#### `test_sector_rotation_end_to_end.py`
**Purpose:** End-to-end test of sector rotation strategy
**Tests:**
- Employment data → signal generation
- Sector scoring
- Portfolio rebalancing
- Risk-adjusted allocation

**Run:**
```bash
uv run python python_tests/test_sector_rotation_end_to_end.py
```

**Type:** Integration test (requires full system)

---

### 11. Dashboard Tests

#### `test_dashboard.py`
**Purpose:** Test Streamlit dashboard functionality
**Tests:**
- Dashboard rendering
- Data loading
- Interactive components
- Performance metrics

**Run:**
```bash
uv run python python_tests/test_dashboard.py
```

**Note:** May require dashboard to be running

---

## Running All Tests

### Individual Test
```bash
uv run python python_tests/<test_name>.py
```

### Run All Tests (Sequential)
```bash
for test in python_tests/test_*.py; do
    echo "Running $test..."
    uv run python "$test"
done
```

### With pytest (if available)
```bash
uv run pytest python_tests/
```

---

## Test Organization

### By Category

**Integration Tests** (require full system):
- `test_sector_rotation_end_to_end.py`
- `test_dashboard.py`

**Unit Tests** (isolated components):
- `test_sentiment_keywords.py`
- `test_signal_generation.py`
- `test_source_filtering.py`

**System Tests** (require external services):
- `test_account.py` (requires Schwab API)
- `test_market_data.py` (requires Schwab API)
- `test_orders.py` (requires Schwab API)
- `test_employment_pipeline.py` (requires BLS API)

**C++ Binding Tests** (require build):
- `test_risk_bindings.py`
- `test_duckdb_bindings.py`

---

## Adding New Tests

### Template

```python
#!/usr/bin/env python3
"""
Test: <Component Name>

Purpose: Test <specific functionality>
Author: <Your Name>
Date: <Date>
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_<functionality>():
    """Test <specific case>"""
    # Test implementation
    pass

if __name__ == "__main__":
    print("Testing <Component>...")
    test_<functionality>()
    print("✅ All tests passed!")
```

### Best Practices

1. **Clear naming:** `test_<component>_<functionality>.py`
2. **Documentation:** Include purpose and test cases in docstring
3. **Isolation:** Tests should not depend on each other
4. **Cleanup:** Restore state after tests
5. **Assertions:** Use clear assertion messages
6. **Coverage:** Test happy path, edge cases, and error cases

---

## CI/CD Integration

### GitHub Actions (example)

```yaml
name: Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Run tests
        run: |
          for test in python_tests/test_*.py; do
            uv run python "$test"
          done
```

---

## Troubleshooting

### Import Errors
**Problem:** `ModuleNotFoundError`

**Solution:** Ensure project root is in Python path:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Missing Dependencies
**Problem:** `ImportError: cannot import 'module_name'`

**Solution:** Install dependencies:
```bash
uv pip install <missing-package>
```

### C++ Binding Errors
**Problem:** `ImportError: No module named 'risk_py'`

**Solution:** Build C++ modules:
```bash
cmake -G Ninja -B build
ninja -C build
```

### API Errors
**Problem:** `401 Unauthorized` or `403 Forbidden`

**Solution:** Check API credentials:
```bash
# Verify tokens exist
ls -la data/tokens/

# Refresh tokens
uv run python scripts/refresh_schwab_token.py
```

---

## Test Coverage

**Current Coverage:** 11 test files
- Account & Portfolio: 1
- Risk Management: 1
- Market Data: 1
- Orders: 1
- Database: 1
- Sentiment: 2
- News: 1
- Signals: 1
- Employment: 1
- Strategy: 1
- Dashboard: 1

**Coverage Goals:**
- Unit test coverage: >80%
- Integration test coverage: >60%
- Critical path coverage: 100%

---

## Maintenance

### Regular Tasks

1. **Monthly:** Review and update test data
2. **Quarterly:** Add tests for new features
3. **Annually:** Refactor outdated tests

### Test Hygiene

- Remove obsolete tests
- Update test data
- Fix flaky tests
- Improve test documentation

---

## Related Documentation

- [Testing Guide](../docs/TESTING_GUIDE.md)
- [CI/CD Pipeline](../docs/CICD.md)
- [Development Workflow](../docs/DEVELOPMENT.md)

---

**Last Updated:** 2025-11-11
**Maintained By:** BigBrotherAnalytics Team
