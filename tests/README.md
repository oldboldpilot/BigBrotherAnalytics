# Tests Directory

Comprehensive test suite for BigBrotherAnalytics.

## Structure

### unit/
Unit tests for individual components
- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution (<5 minutes for full suite)

### integration/
Integration tests for component interactions
- Test data flow between modules
- Test API integrations (Schwab, data sources)
- Test database operations
- Medium execution time (10-15 minutes)

### e2e/
End-to-end tests for complete workflows
- Test full trading workflows (signal → decision → execution)
- Test backtesting pipeline
- Test paper trading system
- Slower execution (20-30 minutes)

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run unit tests only
uv run pytest tests/unit/

# Run integration tests
uv run pytest tests/integration/

# Run e2e tests
uv run pytest tests/e2e/

# Run with coverage
uv run pytest --cov=src tests/
```

## Test Requirements

- All new code must have unit tests (minimum 80% coverage)
- Integration tests required for API integrations
- E2E tests required for trading strategies
- Tests must pass before committing
