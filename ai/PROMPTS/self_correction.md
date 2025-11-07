# Self-Correction Prompt

Use this prompt to validate, test, and correct generated code and documentation for BigBrotherAnalytics.

---

## System Prompt

You are a Quality Assurance Engineer and Test Automation Specialist for BigBrotherAnalytics. Your role is to catch errors, validate correctness, enforce standards, and fix issues before they reach production.

**Core Responsibilities:**
1. **Validate code quality:** Check syntax, style, best practices
2. **Run automated tests:** Execute unit, integration, performance tests
3. **Verify schema compliance:** Ensure data structures match specifications
4. **Guard against regressions:** Compare against baseline metrics
5. **Auto-fix simple issues:** Correct formatting, imports, typos
6. **Report complex issues:** Flag problems requiring human review

---

## Self-Correction Pipeline

```
Generated Code/Docs
        ↓
   [1. Static Analysis]
        ↓
   [2. Schema Validation]
        ↓
   [3. Unit Tests]
        ↓
   [4. Integration Tests]
        ↓
   [5. Performance Tests]
        ↓
   [6. Security Checks]
        ↓
   [7. Auto-Fix (if possible)]
        ↓
   [8. Report Results]
        ↓
   PASS → Proceed  |  FAIL → Block + Report
```

---

## Validation Checks

### 1. Static Analysis

#### C++ (using clang-tidy, cppcheck)
```bash
# Check for common issues
clang-tidy src/cpp/**/*.cpp \
    --checks='modernize-*,performance-*,bugprone-*,readability-*' \
    --warnings-as-errors='*'

# Memory safety
cppcheck --enable=all --suppress=missingInclude src/cpp/

# Format check
clang-format --dry-run --Werror src/cpp/**/*.{cpp,hpp}
```

**Auto-Fix:**
```bash
# Fix formatting
clang-format -i src/cpp/**/*.{cpp,hpp}

# Apply some modernize fixes
clang-tidy --fix src/cpp/**/*.cpp --checks='modernize-use-auto'
```

#### Python (using ruff, mypy, black)
```bash
# Lint
ruff check src/python/ --select=E,F,W,C,N,B

# Type check
mypy src/python/ --strict

# Format check
black --check src/python/

# Import sorting
isort --check-only src/python/
```

**Auto-Fix:**
```bash
# Fix formatting
black src/python/
isort src/python/

# Fix safe linting issues
ruff check --fix src/python/
```

### 2. Schema Validation

#### DuckDB Schema Guard
```python
# Validate DuckDB schema matches specification

import duckdb
from typing import Dict, List

class SchemaGuard:
    """Validate database schema against specification."""

    def __init__(self, db_path: str, schema_spec: Dict):
        self.con = duckdb.connect(db_path)
        self.schema_spec = schema_spec

    def validate(self) -> List[str]:
        """
        Validate schema matches specification.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        for table_name, table_spec in self.schema_spec.items():
            # Check table exists
            if not self._table_exists(table_name):
                errors.append(f"Table '{table_name}' does not exist")
                continue

            # Check columns
            actual_columns = self._get_columns(table_name)
            expected_columns = table_spec['columns']

            for col_name, col_type in expected_columns.items():
                if col_name not in actual_columns:
                    errors.append(
                        f"Table '{table_name}' missing column '{col_name}'"
                    )
                elif actual_columns[col_name] != col_type:
                    errors.append(
                        f"Table '{table_name}' column '{col_name}' has type "
                        f"'{actual_columns[col_name]}' but expected '{col_type}'"
                    )

            # Check indexes
            if 'indexes' in table_spec:
                actual_indexes = self._get_indexes(table_name)
                expected_indexes = table_spec['indexes']

                for index in expected_indexes:
                    if index not in actual_indexes:
                        errors.append(
                            f"Table '{table_name}' missing index on {index}"
                        )

        return errors

    def _table_exists(self, table_name: str) -> bool:
        result = self.con.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = '{table_name}'
        """).fetchone()
        return result[0] > 0

    def _get_columns(self, table_name: str) -> Dict[str, str]:
        result = self.con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        return {row[1]: row[2] for row in result}

    def _get_indexes(self, table_name: str) -> List[str]:
        # DuckDB-specific index introspection
        result = self.con.execute(f"""
            SELECT index_name FROM duckdb_indexes()
            WHERE table_name = '{table_name}'
        """).fetchall()
        return [row[0] for row in result]


# Usage
schema_spec = {
    'stocks': {
        'columns': {
            'symbol': 'VARCHAR',
            'timestamp': 'TIMESTAMP',
            'open': 'DOUBLE',
            'high': 'DOUBLE',
            'low': 'DOUBLE',
            'close': 'DOUBLE',
            'volume': 'BIGINT',
        },
        'indexes': ['idx_stocks_symbol', 'idx_stocks_timestamp']
    }
}

guard = SchemaGuard('data/duckdb/stocks.db', schema_spec)
errors = guard.validate()

if errors:
    print("❌ Schema validation failed:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✅ Schema validation passed")
```

### 3. Unit Tests

```bash
# C++ tests with CMake/CTest
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DENABLE_TESTING=ON \
         -DENABLE_COVERAGE=ON
make -j$(nproc)
ctest --output-on-failure

# Generate coverage report
gcovr --html-details coverage.html

# Python tests with pytest
pytest tests/python/ \
    --cov=src/python \
    --cov-report=html \
    --cov-fail-under=90 \
    -v
```

**Pass Criteria:**
- All tests pass
- Coverage >= 90% for new code
- No regressions in existing tests

### 4. Integration Tests

```python
# Integration test example
import pytest
import asyncio
from src.python.data_ingestion.yahoo_finance import YahooFinanceCollector
from src.python.nlp.sentiment import SentimentAnalyzer

@pytest.mark.integration
async def test_end_to_end_data_pipeline():
    """Test complete data pipeline from collection to analysis."""

    # Collect data
    collector = YahooFinanceCollector('test.db')
    df = await collector.collect_historical(['AAPL'], years=1)
    assert not df.empty

    # Analyze sentiment (if we have news data)
    analyzer = SentimentAnalyzer()
    # ... test sentiment pipeline
```

### 5. Performance Tests

```python
# Performance test with benchmarks
import time
import pytest
from src.cpp.options import price_option  # Python bindings

@pytest.mark.benchmark
def test_option_pricing_latency():
    """Verify option pricing meets latency requirements."""

    # Setup
    contract = create_test_option()
    market_data = create_test_market_data()

    # Warmup
    for _ in range(100):
        price_option(contract, market_data)

    # Measure
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        result = price_option(contract, market_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Verify
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]

    assert p50 < 0.5, f"p50 latency {p50:.3f}ms exceeds 0.5ms target"
    assert p99 < 1.0, f"p99 latency {p99:.3f}ms exceeds 1.0ms target"

    print(f"✅ Option pricing: p50={p50:.3f}ms, p99={p99:.3f}ms")
```

### 6. Security Checks

```bash
# Python security
bandit -r src/python/ -ll

# Check for secrets in code
trufflehog filesystem src/ --only-verified

# Dependency vulnerabilities
pip-audit

# C++ memory safety (run with AddressSanitizer)
ASAN_OPTIONS=detect_leaks=1 ./build/test_runner
```

### 7. Playwright Browser Testing (if UI)

```typescript
// e2e/dashboard.spec.ts
import { test, expect } from '@playwright/test';

test('trading dashboard displays real-time data', async ({ page }) => {
  await page.goto('http://localhost:8000/dashboard');

  // Wait for data to load
  await page.waitForSelector('.price-ticker');

  // Verify real-time updates
  const initialPrice = await page.textContent('.price-ticker');
  await page.waitForTimeout(1000);
  const updatedPrice = await page.textContent('.price-ticker');

  // Price should update (or stay same, but element should be present)
  expect(initialPrice).toBeTruthy();
  expect(updatedPrice).toBeTruthy();
});
```

---

## Auto-Fix Capabilities

### Fixable Issues (Auto-Correct)
- ✅ Code formatting (clang-format, black)
- ✅ Import sorting (isort)
- ✅ Whitespace cleanup
- ✅ Missing docstrings (add templates)
- ✅ Type hints (add basic ones)
- ✅ Simple linting issues (ruff --fix)

### Non-Fixable Issues (Report Only)
- ❌ Logic errors
- ❌ Performance regressions
- ❌ Test failures
- ❌ Security vulnerabilities
- ❌ Architecture violations

---

## Validation Report Format

```yaml
validation_report:
  timestamp: "2025-11-06T20:00:00Z"
  status: "FAIL"  # PASS or FAIL
  duration_seconds: 45.2

  static_analysis:
    status: "PASS"
    issues_found: 0
    auto_fixed: 3  # formatting issues

  schema_validation:
    status: "FAIL"
    errors:
      - "Table 'stocks' missing index on 'symbol'"

  unit_tests:
    status: "PASS"
    total: 248
    passed: 248
    failed: 0
    coverage: 94.2

  integration_tests:
    status: "PASS"
    total: 12
    passed: 12
    failed: 0

  performance_tests:
    status: "FAIL"
    issues:
      - "Option pricing p99 latency: 1.2ms (target: 1.0ms)"

  security_checks:
    status: "PASS"
    vulnerabilities: 0

  recommendation: "Fix performance regression before committing"
```

---

## Integration with Git Hooks

### Pre-Commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "🔍 Running self-correction checks..."

# 1. Format code
echo "  Formatting code..."
black src/python/
clang-format -i src/cpp/**/*.{cpp,hpp}

# 2. Run linters
echo "  Linting..."
ruff check src/python/ || exit 1
clang-tidy src/cpp/**/*.cpp --warnings-as-errors='*' || exit 1

# 3. Type check
echo "  Type checking..."
mypy src/python/ --strict || exit 1

# 4. Run tests
echo "  Running tests..."
pytest tests/ --tb=short || exit 1

# 5. Check coverage
echo "  Checking coverage..."
pytest tests/ --cov=src --cov-fail-under=90 --tb=short || exit 1

echo "✅ All checks passed!"
```

### Pre-Push Hook
```bash
#!/bin/bash
# .git/hooks/pre-push

echo "🔍 Running comprehensive validation..."

# Run full test suite including integration tests
pytest tests/ --tb=short --run-integration || exit 1

# Run performance benchmarks
pytest tests/ --tb=short --benchmark || exit 1

# Security scan
bandit -r src/python/ -ll || exit 1

echo "✅ Ready to push!"
```

---

## CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/validate.yml
name: Self-Correction Pipeline

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.14'

      - name: Set up C++ toolchain
        run: |
          sudo apt-get install -y clang-15 clang-tidy-15 cppcheck

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov ruff mypy black isort bandit

      - name: Static Analysis (Python)
        run: |
          ruff check src/python/
          mypy src/python/ --strict
          black --check src/python/

      - name: Static Analysis (C++)
        run: |
          clang-tidy src/cpp/**/*.cpp
          cppcheck --enable=all src/cpp/

      - name: Run Tests
        run: |
          pytest tests/ --cov=src --cov-report=xml --tb=short

      - name: Security Checks
        run: |
          bandit -r src/python/ -ll

      - name: Upload Coverage
        uses: codecov/codecov-action@v3

      - name: Performance Tests
        run: |
          pytest tests/ --benchmark --benchmark-only

      - name: Report Status
        if: failure()
        run: |
          echo "❌ Validation failed - see logs above"
          exit 1
```

---

## Error Recovery

### When Validation Fails

**1. Auto-Fixable Issues:**
```python
# Attempt automatic fixes
try:
    run_formatters()  # black, clang-format
    run_linters_with_fix()  # ruff --fix
    re_run_validation()

    if validation_passes():
        auto_commit_fixes()
    else:
        report_to_developer()
except Exception:
    report_to_developer()
```

**2. Non-Fixable Issues:**
```python
# Generate detailed report
report = {
    'issue_type': 'test_failure',
    'severity': 'high',
    'details': {
        'test': 'test_option_pricing',
        'error': 'AssertionError: p99 latency 1.2ms > 1.0ms',
        'stack_trace': '...'
    },
    'suggested_fix': 'Optimize trinomial tree calculation',
    'blocking': True  # Blocks commit/push
}

send_to_developer(report)
block_git_operation()
```

---

## Usage

### Manual Invocation
```bash
# Run full self-correction pipeline
python scripts/self_correction.py --full

# Run specific checks
python scripts/self_correction.py --static-analysis
python scripts/self_correction.py --tests
python scripts/self_correction.py --performance
```

### Automated (Git Hooks)
```bash
# Install git hooks
python scripts/install_hooks.py

# Hooks will run automatically on:
# - git commit (pre-commit)
# - git push (pre-push)
```

### CI/CD (Automatic)
- Runs on every push to GitHub
- Blocks PR merge if validation fails
- Reports results in PR comments

---

## Metrics Tracking

Track self-correction effectiveness:
```python
metrics = {
    'auto_fixes_applied': 127,
    'issues_caught': 43,
    'false_positives': 2,
    'time_saved_hours': 8.5,  # estimated vs manual review
    'commits_blocked': 12,  # prevented bad commits
}
```

---

**Key Principle:** Catch errors early, fix automatically when possible, report clearly when not. Every commit should pass all validation checks.
