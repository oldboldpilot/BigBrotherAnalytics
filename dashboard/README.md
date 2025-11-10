# BigBrother Trading Dashboard

A comprehensive web dashboard for real-time monitoring of trades, performance metrics, and employment signals using Streamlit.

## Features

### Component 1: Real-Time Positions View
- Displays all active positions from DuckDB
- Shows: Symbol, Quantity, Entry Price, Current Price, Market Value, Unrealized P&L
- Color-coded status: Green (profit), Red (loss), Gray (break-even)
- Position breakdown charts and visualizations
- Summary metrics: Total positions, profitable count, losing count, total market value

### Component 2: P&L Charts
- Current P&L summary with key metrics
- Win/Loss ratio pie chart
- P&L by symbol bar chart
- Daily P&L trend over last 30 days (when historical data available)
- Cumulative P&L chart
- Performance distribution analysis

### Component 3: Employment Signals
- Displays all 11 GICS sectors with latest employment data
- Shows 3-month employment growth rates
- Color-coded signals:
  - 🟢 Green (OVERWEIGHT): Growth rate >= 2.0%
  - 🟡 Yellow (MARKET WEIGHT): Growth rate >= 0%
  - 🔴 Red (AVOID): Growth rate < 0%
- Displays corresponding ETF symbols (XLE, XLB, XLK, etc.)
- Category-based analysis (Cyclical, Defensive, Sensitive)
- Interactive visualizations and growth trends

### Component 4: Trade History
- Recent trades from positions_history table
- Shows: Timestamp, Symbol, Quantity, Average Price, Current Price, P&L
- Filterable by: Symbol, Strategy, Bot-managed status
- Trading activity timeline
- Symbol and strategy distribution charts
- Comprehensive trade analytics

## Technology Stack

- **Frontend**: Streamlit 1.50.0+
- **Database**: DuckDB (data/bigbrother.duckdb)
- **Visualization**: Plotly Express & Plotly Graph Objects
- **Data Processing**: Pandas
- **Package Manager**: uv

## Installation

All dependencies are already included in the main project's `pyproject.toml`:

```bash
# From project root
cd /home/muyiwa/Development/BigBrotherAnalytics
uv sync
```

Key dependencies:
- streamlit >= 1.50.0
- duckdb >= 1.4.1
- plotly >= 6.4.0
- pandas >= 2.3.3

## Usage

### Running the Dashboard

From the project root directory:

```bash
# Method 1: Using uv (recommended)
cd dashboard
uv run streamlit run app.py

# Method 2: Using virtual environment
source .venv/bin/activate
cd dashboard
streamlit run app.py
```

The dashboard will be available at: **http://localhost:8501**

### Navigation

The dashboard provides five main views accessible from the sidebar:

1. **Overview**: Quick summary of all components
2. **Positions**: Detailed real-time position monitoring
3. **P&L Analysis**: Performance charts and metrics
4. **Employment Signals**: Sector rotation strategy signals
5. **Trade History**: Historical trading activity

### Auto-Refresh

Enable auto-refresh in the sidebar to update the dashboard every 30 seconds automatically.

## Database Schema

The dashboard connects to `data/bigbrother.duckdb` and uses the following tables:

### positions
- Active trading positions
- Fields: symbol, quantity, avg_cost, current_price, market_value, unrealized_pnl, etc.

### positions_history
- Historical position snapshots
- Fields: timestamp, symbol, quantity, average_price, current_price, unrealized_pnl, strategy

### sectors
- GICS sector information
- Fields: sector_code, sector_name, sector_etf, category, description

### sector_employment
- Employment data by sector from BLS
- Fields: sector_code, bls_series_id, report_date, employment_count, unemployment_rate

## Sample Data

If the database is empty, you can populate it with sample data:

```bash
# Run the employment pipeline
uv run python test_employment_pipeline.py

# Or use the sample data generator (if available)
uv run python dashboard/generate_sample_data.py
```

## Development

### Project Structure

```
dashboard/
├── app.py              # Main Streamlit application
├── README.md           # This file
└── generate_sample_data.py  # Sample data generator (optional)
```

### Adding New Features

1. Add new functions to `app.py`
2. Create new views in the sidebar navigation
3. Use the `@st.cache_resource` decorator for database connections
4. Use `@st.cache_data` for data loading functions (if needed)

### Customization

- **Theme**: Modify Streamlit theme in `.streamlit/config.toml`
- **Layout**: Adjust column widths and component placement in `app.py`
- **Colors**: Update color schemes in the plotting functions
- **Metrics**: Add custom calculations in the data loading functions

## Troubleshooting

### Database Not Found

```
Error: Database not found at data/bigbrother.duckdb
```

**Solution**: Ensure the database file exists at the correct path:
```bash
ls -la data/bigbrother.duckdb
```

### No Data Displayed

If tables are empty:
1. Run the employment pipeline: `uv run python test_employment_pipeline.py`
2. Execute some trades to populate positions
3. Check data availability: `uv run python test_duckdb_bindings.py`

### Port Already in Use

```
Error: Port 8501 is already in use
```

**Solution**: Use a different port:
```bash
streamlit run app.py --server.port 8502
```

### Module Import Errors

```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution**: Ensure dependencies are installed:
```bash
uv sync
```

## Performance Notes

- Database connections are cached for performance
- Read-only mode prevents accidental data modifications
- Queries are optimized for fast loading
- Auto-refresh can be disabled for slower systems

## Future Enhancements

Potential improvements for future versions:

1. **Real-time Updates**: WebSocket integration for live data
2. **Advanced Filtering**: More granular filtering options
3. **Export Features**: Download reports as CSV/PDF
4. **Alerts**: Configurable alerts for P&L thresholds
5. **Multi-Account**: Support for multiple trading accounts
6. **Backtesting**: Historical strategy performance analysis
7. **Mobile View**: Responsive design optimization
8. **User Authentication**: Secure login system

## Support

For issues or questions:
1. Check the main project documentation
2. Review the database schema
3. Verify all dependencies are installed
4. Check Streamlit logs for error messages

## License

Part of the BigBrother Analytics trading system.
