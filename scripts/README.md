# Scripts Directory

Utility scripts for data collection, backtesting, and deployment.

## Structure

### data_collection/
Scripts for collecting and updating market data
- Historical data download (Yahoo Finance, FRED)
- Real-time data collection
- News and sentiment data aggregation
- Data validation and quality checks

### backtesting/
Backtesting and validation scripts
- Run historical simulations
- Performance analysis
- Walk-forward validation
- Strategy comparison

### deployment/
Deployment and operations scripts
- Environment setup
- Database initialization
- Service deployment
- Monitoring setup

## Usage

All scripts should be run using `uv run` to ensure correct Python environment:

```bash
# Example: Collect historical data
uv run python scripts/data_collection/download_historical.py

# Example: Run backtest
uv run python scripts/backtesting/run_backtest.py --strategy straddle --start 2020-01-01
```
