/**
 * BigBrotherAnalytics - DuckDB Bridge Library
 *
 * Isolates DuckDB incomplete types from C++23 modules by providing
 * a clean C++ interface. This bridge library:
 * - Hides DuckDB's problematic forward declarations (e.g., QueryNode)
 * - Provides opaque handle types for modules
 * - Implements all DuckDB operations in .cpp file
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace bigbrother::duckdb_bridge {

/**
 * Opaque handle types - actual DuckDB types hidden in implementation
 */
class DatabaseHandle {
  public:
    DatabaseHandle();
    ~DatabaseHandle();

    // Non-copyable, non-movable
    DatabaseHandle(DatabaseHandle const&) = delete;
    auto operator=(DatabaseHandle const&) -> DatabaseHandle& = delete;
    DatabaseHandle(DatabaseHandle&&) noexcept = delete;
    auto operator=(DatabaseHandle&&) noexcept -> DatabaseHandle& = delete;

    [[nodiscard]] auto getImpl() -> void*;
    [[nodiscard]] auto getImpl() const -> void const*;

  private:
    friend auto openDatabase(std::string const& path) -> std::unique_ptr<DatabaseHandle>;

    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

class ConnectionHandle {
  public:
    explicit ConnectionHandle(DatabaseHandle& db);
    ~ConnectionHandle();

    // Non-copyable, non-movable
    ConnectionHandle(ConnectionHandle const&) = delete;
    auto operator=(ConnectionHandle const&) -> ConnectionHandle& = delete;
    ConnectionHandle(ConnectionHandle&&) noexcept = delete;
    auto operator=(ConnectionHandle&&) noexcept -> ConnectionHandle& = delete;

    [[nodiscard]] auto getImpl() -> void*;
    [[nodiscard]] auto getImpl() const -> void const*;

  private:
    friend auto createConnection(DatabaseHandle& db) -> std::unique_ptr<ConnectionHandle>;

    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

class PreparedStatementHandle {
  public:
    PreparedStatementHandle() = default;
    ~PreparedStatementHandle();

    PreparedStatementHandle(PreparedStatementHandle const&) = delete;
    auto operator=(PreparedStatementHandle const&) -> PreparedStatementHandle& = delete;
    PreparedStatementHandle(PreparedStatementHandle&&) noexcept;
    auto operator=(PreparedStatementHandle&&) noexcept -> PreparedStatementHandle&;

    [[nodiscard]] auto getImpl() -> void*;
    [[nodiscard]] auto getImpl() const -> void const*;
    auto setImpl(void* impl) -> void;

  private:
    void* pImpl_{nullptr};
};

/**
 * Bridge functions - all DuckDB operations go through these
 */

// Database operations
auto openDatabase(std::string const& path) -> std::unique_ptr<DatabaseHandle>;

// Connection operations
auto createConnection(DatabaseHandle& db) -> std::unique_ptr<ConnectionHandle>;

// Query operations
auto executeQuery(ConnectionHandle& conn, std::string const& query) -> bool;
auto prepareStatement(ConnectionHandle& conn, std::string const& query)
    -> std::unique_ptr<PreparedStatementHandle>;

// Prepared statement operations
auto bindString(PreparedStatementHandle& stmt, int index, std::string const& value) -> bool;
auto bindInt(PreparedStatementHandle& stmt, int index, int value) -> bool;
auto bindDouble(PreparedStatementHandle& stmt, int index, double value) -> bool;
auto bindInt64(PreparedStatementHandle& stmt, int index, int64_t value) -> bool;
auto executeStatement(PreparedStatementHandle& stmt) -> bool;

} // namespace bigbrother::duckdb_bridge
