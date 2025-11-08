/**
 * DuckDB Fluent API Usage Examples
 *
 * Demonstrates the modern C++23 fluent interface for database operations.
 * All using trailing return syntax and C++ Core Guidelines.
 */

// When DuckDB module import works, use:
// import bigbrother.database.api;

// For now:
// #include "utils/database_api.cppm"

using namespace bigbrother::database;

auto example_basic_query() -> void {
    // Example 1: Simple query
    auto results = Database::connect("data/bigbrother.duckdb")
        .query("SELECT * FROM stock_prices WHERE symbol = 'SPY'")
        .execute();

    std::cout << "Found " << results.rowCount() << " rows\n";
}

auto example_parametrized_query() -> void {
    // Example 2: Parametrized query (SQL injection safe)
    auto results = Database::connect("data/bigbrother.duckdb")
        .query("SELECT close, volume FROM stock_prices WHERE symbol = ? AND date >= ?")
        .bind("SPY")
        .bind("2024-01-01")
        .execute();

    for (auto const& row : results.rows()) {
        auto close = row.get("close");
        auto volume = row.get("volume");

        if (close && volume) {
            std::cout << "Close: $" << *close << ", Volume: " << *volume << "\n";
        }
    }
}

auto example_fluent_transaction() -> void {
    // Example 3: Fluent transaction
    auto conn = Database::connect("data/bigbrother.duckdb");

    conn.beginTransaction()
        .executeUpdate("INSERT INTO trades VALUES (?, ?, ?)")
        .executeUpdate("UPDATE portfolio SET pnl = pnl + 100")
        .commit();

    std::cout << "Transaction committed\n";
}

auto example_query_builder() -> void {
    // Example 4: Query builder for complex queries
    auto query = QueryBuilder{}
        .select("symbol, AVG(close) as avg_price, COUNT(*) as days")
        .from("stock_prices")
        .where("date >= '2024-01-01'")
        .where("volume > 1000000")
        .orderBy("avg_price DESC")
        .limit(10)
        .build();

    auto results = Database::connect("data/bigbrother.duckdb")
        .execute(query);

    std::cout << "Top 10 stocks by average price:\n";
    // Process results...
}

auto example_parquet_operations() -> void {
    // Example 5: Parquet import/export (fluent)
    Database::connect("data/bigbrother.duckdb")
        .loadParquet("data/historical/stocks/SPY_daily.parquet", "spy_prices")
        .exportParquet("spy_prices", "data/exports/spy_analysis.parquet");

    std::cout << "Parquet operations complete\n";
}

auto example_aggregations() -> void {
    // Example 6: Complex aggregations
    auto conn = Database::connect("data/bigbrother.duckdb");

    // Calculate realized volatility
    auto vol_query = R"(
        SELECT
            symbol,
            STDDEV(log_return) * SQRT(252) as realized_vol
        FROM (
            SELECT
                symbol,
                LN(close / LAG(close) OVER (PARTITION BY symbol ORDER BY date)) as log_return
            FROM stock_prices
            WHERE date >= '2024-01-01'
        )
        GROUP BY symbol
        ORDER BY realized_vol DESC
    )";

    auto results = conn.execute(vol_query);

    std::cout << "Stock volatilities calculated\n";
}

auto example_time_series_analysis() -> void {
    // Example 7: Time series operations
    auto conn = Database::connect("data/bigbrother.duckdb");

    // Get moving averages
    auto ma_query = R"(
        SELECT
            date,
            close,
            AVG(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20,
            AVG(close) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as sma_50
        FROM stock_prices
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT 100
    )";

    auto results = conn.query(ma_query)
        .bind("SPY")
        .execute();

    // Use results for technical analysis...
}

// Usage in real trading code:
auto get_latest_price(std::string const& symbol) -> std::optional<double> {
    auto result = Database::connect("data/bigbrother.duckdb")
        .query("SELECT close FROM stock_prices WHERE symbol = ? ORDER BY date DESC LIMIT 1")
        .bind(symbol)
        .first();

    if (result) {
        auto close = result->get("close");
        if (close) {
            return std::stod(*close);
        }
    }

    return std::nullopt;
}

// Fluent API for backtesting:
auto run_backtest_query() -> ResultSet {
    return Database::connect("data/bigbrother.duckdb")
        .query(R"(
            SELECT
                date,
                close,
                volume,
                (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) as daily_return
            FROM stock_prices
            WHERE symbol = ?
              AND date BETWEEN ? AND ?
            ORDER BY date
        )")
        .bind("SPY")
        .bind("2024-01-01")
        .bind("2025-01-01")
        .execute();
}
