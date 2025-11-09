/**
 * DuckDB Fluent API - C++23 Module
 *
 * Modern fluent interface for DuckDB database operations.
 * Following C++ Core Guidelines and builder pattern.
 *
 * Example Usage:
 *   auto results = Database::connect("data/trading.duckdb")
 *       .query("SELECT * FROM stock_prices WHERE symbol = ?")
 *       .bind("SPY")
 *       .execute()
 *       .toDataFrame();
 *
 * Following Guidelines:
 * - F.20: Return values for chaining
 * - F.51: Prefer default arguments over overloading
 * - C.2: Use class when invariants exist
 * - R.1: RAII for connection management
 */

// Global module fragment
module;

#include <expected>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <vector>

// Module declaration
export module bigbrother.database.api;

export namespace bigbrother::database {

/**
 * Database Row (using std::vector for efficiency)
 *
 * C.1: struct for passive data
 */
struct Row {
    std::vector<std::string> columns;
    std::vector<std::string> values;

    /**
     * Get value by column name
     * F.20: Return std::optional for possibly-missing values
     */
    [[nodiscard]] auto get(std::string const& column) const -> std::optional<std::string> {
        for (size_t i = 0; i < columns.size(); ++i) {
            if (columns[i] == column) {
                return values[i];
            }
        }
        return std::nullopt;
    }
};

/**
 * Query Result Set
 *
 * C.2: Private data with public interface
 * R.1: RAII - manages result lifecycle
 */
class ResultSet {
  public:
    /**
     * Get all rows
     * F.20: Return by const reference (avoid copy)
     */
    [[nodiscard]] auto rows() const noexcept -> std::vector<Row> const& { return rows_; }

    /**
     * Get row count
     * F.4: constexpr
     * F.6: noexcept
     */
    [[nodiscard]] constexpr auto rowCount() const noexcept -> size_t { return rows_.size(); }

    /**
     * Get column names
     */
    [[nodiscard]] auto columns() const noexcept -> std::vector<std::string> const& {
        return column_names_;
    }

    /**
     * Convert to Pandas-style structure for Python integration
     */
    [[nodiscard]] auto toDataFrame() const -> std::string {
        // Returns CSV-style string for Python pandas
        std::string result;

        // Header
        for (size_t i = 0; i < column_names_.size(); ++i) {
            if (i > 0)
                result += ",";
            result += column_names_[i];
        }
        result += "\n";

        // Rows
        for (auto const& row : rows_) {
            for (size_t i = 0; i < row.values.size(); ++i) {
                if (i > 0)
                    result += ",";
                result += row.values[i];
            }
            result += "\n";
        }

        return result;
    }

    // C.21: Rule of five
    ResultSet() = default;
    ResultSet(ResultSet const&) = default;
    auto operator=(ResultSet const&) -> ResultSet& = default;
    ResultSet(ResultSet&&) noexcept = default;
    auto operator=(ResultSet&&) noexcept -> ResultSet& = default;
    ~ResultSet() = default;

    // Allow construction from data
    auto addRow(Row row) -> void { rows_.push_back(std::move(row)); }

    auto setColumns(std::vector<std::string> cols) -> void { column_names_ = std::move(cols); }

  private:
    std::vector<Row> rows_;
    std::vector<std::string> column_names_;
};

/**
 * Prepared Statement (Fluent API)
 *
 * Allows parameter binding and execution.
 * Following builder pattern for fluent interface.
 */
class PreparedStatement {
  public:
    /**
     * Bind parameter (fluent)
     *
     * F.18: Perfect forwarding for efficiency
     * F.20: Return *this for chaining
     */
    template <typename T>
    auto bind(T&& value) -> PreparedStatement& {
        // Convert to string for now (will use DuckDB native types later)
        bindings_.push_back(std::to_string(std::forward<T>(value)));
        return *this;
    }

    /**
     * Bind string parameter
     */
    auto bind(std::string value) -> PreparedStatement& {
        bindings_.push_back(std::move(value));
        return *this;
    }

    /**
     * Execute prepared statement
     *
     * F.20: Return ResultSet by value (move semantics)
     */
    [[nodiscard]] auto execute() -> ResultSet {
        // Stub implementation
        return ResultSet{};
    }

    /**
     * Execute and get first row
     */
    [[nodiscard]] auto first() -> std::optional<Row> {
        // Stub implementation
        return std::nullopt;
    }

    /**
     * Execute and get scalar value
     */
    template <typename T>
    [[nodiscard]] auto scalar() -> std::optional<T> {
        // Stub implementation
        return std::nullopt;
    }

  private:
    friend class DatabaseConnection;

    PreparedStatement(std::string query) : query_{std::move(query)} {}

    std::string query_;
    std::vector<std::string> bindings_;
};

/**
 * Database Connection (Fluent API)
 *
 * Following C++ Core Guidelines:
 * - C.2: Encapsulation with private data
 * - C.21: Non-copyable (owns resource)
 * - R.1: RAII for connection lifecycle
 * - F.20: Return values for chaining
 *
 * Fluent interface for database operations.
 */
class DatabaseConnection {
  public:
    /**
     * Begin fluent query
     *
     * F.20: Return PreparedStatement for chaining
     */
    [[nodiscard]] auto query(std::string sql) -> PreparedStatement {
        return PreparedStatement{std::move(sql)};
    }

    /**
     * Execute immediate query (no parameters)
     */
    [[nodiscard]] auto execute(std::string const& sql) -> ResultSet {
        // Stub implementation - returns empty result
        return ResultSet{};
    }

    /**
     * Execute update/insert (returns affected rows)
     */
    [[nodiscard]] auto executeUpdate(std::string const& sql) -> int64_t {
        // Stub implementation
        return 0;
    }

    /**
     * Begin transaction (fluent)
     */
    auto beginTransaction() -> DatabaseConnection& {
        // Execute BEGIN TRANSACTION
        return *this;
    }

    /**
     * Commit transaction (fluent)
     */
    auto commit() -> DatabaseConnection& {
        // Execute COMMIT
        return *this;
    }

    /**
     * Rollback transaction (fluent)
     */
    auto rollback() -> DatabaseConnection& {
        // Execute ROLLBACK
        return *this;
    }

    /**
     * Load Parquet file (fluent)
     *
     * F.16: Pass string by value (will be moved)
     */
    auto loadParquet(std::string file_path, std::string table_name) -> DatabaseConnection& {
        // CREATE TABLE table_name AS SELECT * FROM read_parquet('file.parquet')
        return *this;
    }

    /**
     * Export to Parquet (fluent)
     */
    auto exportParquet(std::string const& table_name, std::string const& file_path)
        -> DatabaseConnection& {
        // COPY table_name TO 'file.parquet' (FORMAT PARQUET)
        return *this;
    }

    /**
     * Close connection
     * F.6: noexcept
     */
    auto close() noexcept -> void {
        // Stub implementation
    }

    // C.21: Non-copyable, movable
    DatabaseConnection(DatabaseConnection const&) = delete;
    auto operator=(DatabaseConnection const&) -> DatabaseConnection& = delete;
    DatabaseConnection(DatabaseConnection&&) noexcept = default;
    auto operator=(DatabaseConnection&&) noexcept -> DatabaseConnection& = default;

    ~DatabaseConnection() = default;

  private:
    friend class Database;

    // Constructor (inline stub implementation for now - DuckDB integration via Python)
    DatabaseConnection(std::string db_path) : db_path_{std::move(db_path)} {}

    std::string db_path_;
    // Note: pImpl removed - using Python DuckDB integration via pybind11
};

/**
 * Database Factory (Static Interface)
 *
 * F.1: Meaningfully named static methods
 * F.20: Return connection by value (move semantics)
 */
class Database {
  public:
    /**
     * Connect to database (fluent entry point)
     *
     * F.16: Pass by value (will be moved)
     * F.20: Return by value
     *
     * Usage:
     *   auto conn = Database::connect("data.duckdb");
     *   auto results = conn.query("SELECT * FROM stocks").execute();
     */
    [[nodiscard]] static auto connect(std::string db_path) -> DatabaseConnection {
        return DatabaseConnection{std::move(db_path)};
    }

    /**
     * Connect to in-memory database
     */
    [[nodiscard]] static auto memory() -> DatabaseConnection {
        return DatabaseConnection{":memory:"};
    }

    /**
     * Create database if not exists
     */
    [[nodiscard]] static auto create(std::string db_path) -> DatabaseConnection {
        // Creates file if doesn't exist
        return DatabaseConnection{std::move(db_path)};
    }
};

/**
 * Query Builder (Fluent API for complex queries)
 *
 * F.20: Each method returns *this for chaining
 *
 * Usage:
 *   auto query = QueryBuilder{}
 *       .select("symbol, close, volume")
 *       .from("stock_prices")
 *       .where("symbol = 'SPY'")
 *       .where("date >= '2024-01-01'")
 *       .orderBy("date DESC")
 *       .limit(100)
 *       .build();
 */
class QueryBuilder {
  public:
    auto select(std::string columns) -> QueryBuilder& {
        select_ = std::move(columns);
        return *this;
    }

    auto from(std::string table) -> QueryBuilder& {
        from_ = std::move(table);
        return *this;
    }

    auto where(std::string condition) -> QueryBuilder& {
        where_clauses_.push_back(std::move(condition));
        return *this;
    }

    auto orderBy(std::string order) -> QueryBuilder& {
        order_by_ = std::move(order);
        return *this;
    }

    auto limit(int count) -> QueryBuilder& {
        limit_ = count;
        return *this;
    }

    /**
     * Build final SQL query
     * F.20: Return by value
     */
    [[nodiscard]] auto build() const -> std::string {
        std::string sql = "SELECT " + select_ + " FROM " + from_;

        if (!where_clauses_.empty()) {
            sql += " WHERE ";
            for (size_t i = 0; i < where_clauses_.size(); ++i) {
                if (i > 0)
                    sql += " AND ";
                sql += where_clauses_[i];
            }
        }

        if (!order_by_.empty()) {
            sql += " ORDER BY " + order_by_;
        }

        if (limit_ > 0) {
            sql += " LIMIT " + std::to_string(limit_);
        }

        return sql;
    }

  private:
    std::string select_{"*"};
    std::string from_;
    std::vector<std::string> where_clauses_;
    std::string order_by_;
    int limit_{0};
};

} // namespace bigbrother::database
