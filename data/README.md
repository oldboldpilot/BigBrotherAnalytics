# Data Directory

Storage for all data files used by the trading platform.

## Structure

### raw/
Raw data files from external sources
- Market data (OHLCV, options chains)
- News and sentiment data
- Economic indicators
- SEC filings
- **DO NOT COMMIT TO GIT** (add to .gitignore)

### processed/
Cleaned and processed data ready for analysis
- Normalized market data
- Computed features
- Correlation matrices
- Sentiment scores
- **DO NOT COMMIT TO GIT** (add to .gitignore)

### models/
Trained ML models and model artifacts
- Saved PyTorch models
- XGBoost models
- FinBERT fine-tuned models
- Model metadata and versioning
- **DO NOT COMMIT LARGE FILES** (use Git LFS or exclude)

### backtest_results/
Results from backtesting runs
- Performance metrics
- Trade logs
- P&L curves
- Strategy comparisons
- **Selective commit** (summary only, not full logs)

## Database Files

DuckDB database files will be stored in `data/` root:
- `bigbrother.duckdb` - Main operational database
- `backtest.duckdb` - Backtesting database
- `*.parquet` - Parquet data files for efficient storage

**Note:** Add large data files to `.gitignore` to avoid bloating the repository.
