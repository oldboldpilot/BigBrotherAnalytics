/**
 * BigBrotherAnalytics - DuckDB Bridge Implementation
 *
 * Uses DuckDB's C API to avoid C++ template instantiation issues with
 * incomplete types (QueryNode). The C API is simpler and more stable.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-11
 */

#include "duckdb_bridge.hpp"

#include <cstring>
#include <duckdb.h> // Use C API instead of C++ API
#include <stdexcept>
#include <string>

namespace bigbrother::duckdb_bridge {

// ============================================================================
// DatabaseHandle Implementation (C API)
// ============================================================================

struct DatabaseHandle::Impl {
    duckdb_database db{nullptr};

    explicit Impl(std::string const& path) {
        if (duckdb_open(path.c_str(), &db) == DuckDBError) {
            throw std::runtime_error("Failed to open database: " + path);
        }
    }

    ~Impl() {
        if (db != nullptr) {
            duckdb_close(&db);
        }
    }

    // Rule of Five
    Impl(Impl const&) = delete;
    auto operator=(Impl const&) -> Impl& = delete;
    Impl(Impl&&) noexcept = delete;
    auto operator=(Impl&&) noexcept -> Impl& = delete;
};

DatabaseHandle::DatabaseHandle() = default;

DatabaseHandle::~DatabaseHandle() = default;

auto DatabaseHandle::getImpl() -> void* {
    return &pImpl_->db;
}

auto DatabaseHandle::getImpl() const -> void const* {
    return &pImpl_->db;
}

// ============================================================================
// ConnectionHandle Implementation (C API)
// ============================================================================

struct ConnectionHandle::Impl {
    duckdb_connection conn{nullptr};

    explicit Impl(duckdb_database db) {
        if (duckdb_connect(db, &conn) == DuckDBError) {
            throw std::runtime_error("Failed to create connection");
        }
    }

    ~Impl() {
        if (conn != nullptr) {
            duckdb_disconnect(&conn);
        }
    }

    // Rule of Five
    Impl(Impl const&) = delete;
    auto operator=(Impl const&) -> Impl& = delete;
    Impl(Impl&&) noexcept = delete;
    auto operator=(Impl&&) noexcept -> Impl& = delete;
};

ConnectionHandle::ConnectionHandle(DatabaseHandle& db)
    : pImpl_(std::make_unique<Impl>(*static_cast<duckdb_database*>(db.getImpl()))) {}

ConnectionHandle::~ConnectionHandle() = default;

auto ConnectionHandle::getImpl() -> void* {
    return &pImpl_->conn;
}

auto ConnectionHandle::getImpl() const -> void const* {
    return &pImpl_->conn;
}

// ============================================================================
// QueryResultHandle Implementation (C API)
// ============================================================================

QueryResultHandle::~QueryResultHandle() {
    if (pImpl_ != nullptr) {
        auto* result = static_cast<duckdb_result*>(pImpl_);
        duckdb_destroy_result(result);
        delete result;
        pImpl_ = nullptr;
    }
}

QueryResultHandle::QueryResultHandle(QueryResultHandle&& other) noexcept : pImpl_(other.pImpl_) {
    other.pImpl_ = nullptr;
}

auto QueryResultHandle::operator=(QueryResultHandle&& other) noexcept -> QueryResultHandle& {
    if (this != &other) {
        if (pImpl_ != nullptr) {
            auto* result = static_cast<duckdb_result*>(pImpl_);
            duckdb_destroy_result(result);
            delete result;
        }
        pImpl_ = other.pImpl_;
        other.pImpl_ = nullptr;
    }
    return *this;
}

auto QueryResultHandle::getImpl() -> void* {
    return pImpl_;
}

auto QueryResultHandle::getImpl() const -> void const* {
    return pImpl_;
}

auto QueryResultHandle::setImpl(void* impl) -> void {
    pImpl_ = impl;
}

// ============================================================================
// PreparedStatementHandle Implementation (C API)
// ============================================================================

PreparedStatementHandle::~PreparedStatementHandle() {
    if (pImpl_ != nullptr) {
        auto* stmt = static_cast<duckdb_prepared_statement*>(pImpl_);
        duckdb_destroy_prepare(stmt);
        delete stmt;
        pImpl_ = nullptr;
    }
}

PreparedStatementHandle::PreparedStatementHandle(PreparedStatementHandle&& other) noexcept
    : pImpl_(other.pImpl_) {
    other.pImpl_ = nullptr;
}

auto PreparedStatementHandle::operator=(PreparedStatementHandle&& other) noexcept
    -> PreparedStatementHandle& {
    if (this != &other) {
        if (pImpl_ != nullptr) {
            auto* stmt = static_cast<duckdb_prepared_statement*>(pImpl_);
            duckdb_destroy_prepare(stmt);
            delete stmt;
        }
        pImpl_ = other.pImpl_;
        other.pImpl_ = nullptr;
    }
    return *this;
}

auto PreparedStatementHandle::getImpl() -> void* {
    return pImpl_;
}

auto PreparedStatementHandle::getImpl() const -> void const* {
    return pImpl_;
}

auto PreparedStatementHandle::setImpl(void* impl) -> void {
    pImpl_ = impl;
}

// ============================================================================
// Bridge Functions (C API)
// ============================================================================

auto openDatabase(std::string const& path) -> std::unique_ptr<DatabaseHandle> {
    auto handle = std::make_unique<DatabaseHandle>();
    handle->pImpl_ = std::make_unique<DatabaseHandle::Impl>(path);
    return handle;
}

auto createConnection(DatabaseHandle& db) -> std::unique_ptr<ConnectionHandle> {
    return std::make_unique<ConnectionHandle>(db);
}

auto executeQuery(ConnectionHandle& conn, std::string const& query) -> bool {
    try {
        auto* duckdb_conn = static_cast<duckdb_connection*>(conn.getImpl());

        duckdb_result result;
        auto state = duckdb_query(*duckdb_conn, query.c_str(), &result);

        bool success = (state == DuckDBSuccess);
        duckdb_destroy_result(&result);

        return success;
    } catch (...) {
        return false;
    }
}

auto prepareStatement(ConnectionHandle& conn, std::string const& query)
    -> std::unique_ptr<PreparedStatementHandle> {
    try {
        auto* duckdb_conn = static_cast<duckdb_connection*>(conn.getImpl());

        auto* stmt = new duckdb_prepared_statement;
        if (duckdb_prepare(*duckdb_conn, query.c_str(), stmt) == DuckDBError) {
            delete stmt;
            return nullptr;
        }

        auto handle = std::make_unique<PreparedStatementHandle>();
        handle->setImpl(stmt);
        return handle;
    } catch (...) {
        return nullptr;
    }
}

auto bindString(PreparedStatementHandle& stmt, int index, std::string const& value) -> bool {
    try {
        auto* duckdb_stmt = static_cast<duckdb_prepared_statement*>(stmt.getImpl());
        if (duckdb_stmt == nullptr)
            return false;

        return duckdb_bind_varchar(*duckdb_stmt, index, value.c_str()) == DuckDBSuccess;
    } catch (...) {
        return false;
    }
}

auto bindInt(PreparedStatementHandle& stmt, int index, int value) -> bool {
    try {
        auto* duckdb_stmt = static_cast<duckdb_prepared_statement*>(stmt.getImpl());
        if (duckdb_stmt == nullptr)
            return false;

        return duckdb_bind_int32(*duckdb_stmt, index, value) == DuckDBSuccess;
    } catch (...) {
        return false;
    }
}

auto bindDouble(PreparedStatementHandle& stmt, int index, double value) -> bool {
    try {
        auto* duckdb_stmt = static_cast<duckdb_prepared_statement*>(stmt.getImpl());
        if (duckdb_stmt == nullptr)
            return false;

        return duckdb_bind_double(*duckdb_stmt, index, value) == DuckDBSuccess;
    } catch (...) {
        return false;
    }
}

auto bindInt64(PreparedStatementHandle& stmt, int index, int64_t value) -> bool {
    try {
        auto* duckdb_stmt = static_cast<duckdb_prepared_statement*>(stmt.getImpl());
        if (duckdb_stmt == nullptr)
            return false;

        return duckdb_bind_int64(*duckdb_stmt, index, value) == DuckDBSuccess;
    } catch (...) {
        return false;
    }
}

auto executeStatement(PreparedStatementHandle& stmt) -> bool {
    try {
        auto* duckdb_stmt = static_cast<duckdb_prepared_statement*>(stmt.getImpl());
        if (duckdb_stmt == nullptr)
            return false;

        duckdb_result result;
        auto state = duckdb_execute_prepared(*duckdb_stmt, &result);

        bool success = (state == DuckDBSuccess);
        duckdb_destroy_result(&result);

        return success;
    } catch (...) {
        return false;
    }
}

auto executeQueryWithResults(ConnectionHandle& conn, std::string const& query)
    -> std::unique_ptr<QueryResultHandle> {
    try {
        auto* duckdb_conn = static_cast<duckdb_connection*>(conn.getImpl());

        auto* result = new duckdb_result;
        auto state = duckdb_query(*duckdb_conn, query.c_str(), result);

        if (state == DuckDBError) {
            duckdb_destroy_result(result);
            delete result;
            return nullptr;
        }

        auto handle = std::make_unique<QueryResultHandle>();
        handle->setImpl(result);
        return handle;
    } catch (...) {
        return nullptr;
    }
}

auto getRowCount(QueryResultHandle const& result) -> size_t {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return 0;
    return duckdb_row_count(duckdb_res);
}

auto getColumnCount(QueryResultHandle const& result) -> size_t {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return 0;
    return duckdb_column_count(duckdb_res);
}

auto getColumnName(QueryResultHandle const& result, size_t col_idx) -> std::string {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return "";

    const char* name = duckdb_column_name(duckdb_res, col_idx);
    return name ? std::string(name) : "";
}

auto hasError(QueryResultHandle const& result) -> bool {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return true;

    const char* error = duckdb_result_error(duckdb_res);
    return (error != nullptr);
}

auto getErrorMessage(QueryResultHandle const& result) -> std::string {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return "Invalid result handle";

    const char* error = duckdb_result_error(duckdb_res);
    return error ? std::string(error) : "";
}

auto getValueAsString(QueryResultHandle const& result, size_t col_idx, size_t row_idx)
    -> std::string {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return "";

    auto value = duckdb_value_varchar(duckdb_res, col_idx, row_idx);
    std::string str = value ? std::string(value) : "";
    duckdb_free(value);
    return str;
}

auto getValueAsInt(QueryResultHandle const& result, size_t col_idx, size_t row_idx) -> int32_t {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return 0;

    return duckdb_value_int32(duckdb_res, col_idx, row_idx);
}

auto getValueAsInt64(QueryResultHandle const& result, size_t col_idx, size_t row_idx) -> int64_t {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return 0;

    return duckdb_value_int64(duckdb_res, col_idx, row_idx);
}

auto getValueAsDouble(QueryResultHandle const& result, size_t col_idx, size_t row_idx) -> double {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return 0.0;

    return duckdb_value_double(duckdb_res, col_idx, row_idx);
}

auto getValueAsBool(QueryResultHandle const& result, size_t col_idx, size_t row_idx) -> bool {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return false;

    return duckdb_value_boolean(duckdb_res, col_idx, row_idx);
}

auto isValueNull(QueryResultHandle const& result, size_t col_idx, size_t row_idx) -> bool {
    auto* duckdb_res = static_cast<duckdb_result*>(const_cast<void*>(result.getImpl()));
    if (duckdb_res == nullptr)
        return true;

    return duckdb_value_is_null(duckdb_res, col_idx, row_idx);
}

} // namespace bigbrother::duckdb_bridge
