#pragma once

#include <string>
#include <memory>
#include <vector>
#include <optional>
#include <variant>
#include <unordered_map>

namespace bigbrother {
namespace utils {

/**
 * Database Value Type
 *
 * Represents a value that can be stored in or retrieved from the database.
 * Supports common SQL types.
 */
using DBValue = std::variant<
    std::monostate,      // NULL
    int64_t,             // INTEGER
    double,              // REAL/DOUBLE
    std::string,         // TEXT/VARCHAR
    bool,                // BOOLEAN
    std::vector<uint8_t> // BLOB
>;

/**
 * Database Row
 *
 * Represents a single row from a query result.
 * Maps column names to their values.
 */
using DBRow = std::unordered_map<std::string, DBValue>;

/**
 * Database Result Set
 *
 * Represents the result of a query.
 * Contains multiple rows and metadata.
 */
class DBResultSet {
public:
    DBResultSet() = default;

    void addRow(DBRow row) { rows.push_back(std::move(row)); }
    void setColumnNames(std::vector<std::string> names) { column_names = std::move(names); }

    size_t rowCount() const { return rows.size(); }
    size_t columnCount() const { return column_names.size(); }

    const std::vector<DBRow>& getRows() const { return rows; }
    const std::vector<std::string>& getColumnNames() const { return column_names; }

    const DBRow& operator[](size_t index) const { return rows[index]; }

    // Iterator support
    auto begin() const { return rows.begin(); }
    auto end() const { return rows.end(); }

private:
    std::vector<DBRow> rows;
    std::vector<std::string> column_names;
};

/**
 * Database Connection
 *
 * Thread-safe DuckDB database connection wrapper.
 * Provides simple API for executing queries and managing transactions.
 *
 * Features:
 * - ACID transactions
 * - Prepared statements
 * - Connection pooling (future)
 * - Query parameterization
 * - Efficient bulk inserts
 */
class Database {
public:
    /**
     * Constructor
     * @param db_path Path to DuckDB database file (or ":memory:" for in-memory)
     * @param read_only Open database in read-only mode
     */
    explicit Database(const std::string& db_path = "data/bigbrother.duckdb",
                     bool read_only = false);

    ~Database();

    // Delete copy constructor and assignment
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    // Move constructor and assignment
    Database(Database&&) noexcept;
    Database& operator=(Database&&) noexcept;

    /**
     * Open database connection
     * @return true if successful
     */
    bool open();

    /**
     * Close database connection
     */
    void close();

    /**
     * Check if database is open
     */
    bool isOpen() const;

    /**
     * Execute SQL query
     * @param query SQL query string
     * @return Result set (empty if no results)
     */
    DBResultSet execute(const std::string& query);

    /**
     * Execute SQL query with parameters
     * @param query SQL query with ? placeholders
     * @param params Parameter values
     * @return Result set (empty if no results)
     */
    template<typename... Args>
    DBResultSet execute(const std::string& query, Args&&... params);

    /**
     * Execute SQL statement (no results expected)
     * @param statement SQL statement
     * @return Number of rows affected
     */
    int64_t executeUpdate(const std::string& statement);

    /**
     * Begin transaction
     */
    void beginTransaction();

    /**
     * Commit transaction
     */
    void commit();

    /**
     * Rollback transaction
     */
    void rollback();

    /**
     * RAII Transaction Guard
     *
     * Automatically rolls back if not committed.
     *
     * Usage:
     *   {
     *     auto txn = db.transaction();
     *     db.executeUpdate("INSERT ...");
     *     db.executeUpdate("UPDATE ...");
     *     txn.commit();
     *   } // Auto rollback if commit() not called
     */
    class Transaction {
    public:
        explicit Transaction(Database& db);
        ~Transaction();

        Transaction(const Transaction&) = delete;
        Transaction& operator=(const Transaction&) = delete;

        void commit();
        void rollback();

    private:
        Database& db_;
        bool committed_;
        bool rolled_back_;
    };

    Transaction transaction() {
        return Transaction(*this);
    }

    /**
     * Bulk insert data
     * @param table_name Target table
     * @param rows Data rows to insert
     * @return Number of rows inserted
     */
    int64_t bulkInsert(const std::string& table_name,
                       const std::vector<DBRow>& rows);

    /**
     * Load data from Parquet file
     * @param parquet_path Path to Parquet file
     * @param table_name Target table name (optional, uses filename)
     * @return true if successful
     */
    bool loadParquet(const std::string& parquet_path,
                     const std::string& table_name = "");

    /**
     * Export table to Parquet
     * @param table_name Source table
     * @param parquet_path Output Parquet file
     * @return true if successful
     */
    bool exportParquet(const std::string& table_name,
                       const std::string& parquet_path);

    /**
     * Get last error message
     */
    std::string getLastError() const;

    /**
     * Create table if not exists
     * @param create_statement CREATE TABLE statement
     * @return true if successful
     */
    bool createTable(const std::string& create_statement);

    /**
     * Drop table if exists
     * @param table_name Table to drop
     * @return true if successful
     */
    bool dropTable(const std::string& table_name);

    /**
     * Check if table exists
     * @param table_name Table name
     * @return true if exists
     */
    bool tableExists(const std::string& table_name);

    /**
     * Get table schema
     * @param table_name Table name
     * @return Schema information
     */
    std::vector<std::string> getTableSchema(const std::string& table_name);

    /**
     * Vacuum database (cleanup and optimize)
     */
    void vacuum();

    /**
     * Checkpoint database (flush WAL to disk)
     */
    void checkpoint();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Database Connection Pool
 *
 * Manages a pool of database connections for concurrent access.
 * Thread-safe.
 */
class DatabasePool {
public:
    /**
     * Constructor
     * @param db_path Path to database
     * @param pool_size Number of connections in pool
     */
    explicit DatabasePool(const std::string& db_path, size_t pool_size = 10);
    ~DatabasePool();

    /**
     * Get a connection from the pool
     * Blocks if no connections available.
     * @return Database connection (automatically returned to pool when destroyed)
     */
    std::shared_ptr<Database> acquire();

    /**
     * Get pool statistics
     */
    struct Stats {
        size_t total_connections;
        size_t active_connections;
        size_t idle_connections;
    };

    Stats getStats() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace utils
} // namespace bigbrother
