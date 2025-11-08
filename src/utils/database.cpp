/**
 * Database Module Implementation
 * C++23 module implementation file
 */

import bigbrother.utils.database;
import bigbrother.utils.logger;

#include <mutex>
#include <condition_variable>
#include <queue>
#include <stdexcept>

// DuckDB C++ headers have compatibility issues with Clang 21 + C++23
// Using stub implementation for now - will use Python DuckDB via uv/pybind11
// TODO: Upgrade to newer DuckDB version or use Python bindings
#if 0  // Temporarily disabled
#include <duckdb.hpp>
#endif

namespace bigbrother {
namespace utils {

#if 0  // DuckDB C++ disabled - using stub until Python integration ready

class Database::Impl {
public:
    Impl(const std::string& db_path, bool read_only)
        : db_path_(db_path), read_only_(read_only), is_open_(false) {}

    ~Impl() {
        close();
    }

    bool open() {
        try {
            duckdb::DBConfig config;
            if (read_only_) {
                config.options.access_mode = duckdb::AccessMode::READ_ONLY;
            }

            db_ = std::make_unique<duckdb::DuckDB>(db_path_, &config);
            conn_ = std::make_unique<duckdb::Connection>(*db_);
            is_open_ = true;

            LOG_INFO("DuckDB connection opened: {}", db_path_);
            return true;
        } catch (const std::exception& e) {
            last_error_ = e.what();
            LOG_ERROR("Failed to open database {}: {}", db_path_, e.what());
            return false;
        }
    }

    void close() {
        if (is_open_) {
            conn_.reset();
            db_.reset();
            is_open_ = false;
            LOG_INFO("DuckDB connection closed: {}", db_path_);
        }
    }

    bool isOpen() const {
        return is_open_;
    }

    DBResultSet execute(const std::string& query) {
        if (!is_open_) {
            throw std::runtime_error("Database is not open");
        }

        try {
            auto result = conn_->Query(query);

            if (result->HasError()) {
                last_error_ = result->GetError();
                LOG_ERROR("Query failed: {}", last_error_);
                return DBResultSet();
            }

            return convertResult(*result);
        } catch (const std::exception& e) {
            last_error_ = e.what();
            LOG_ERROR("Query execution failed: {}", e.what());
            return DBResultSet();
        }
    }

    int64_t executeUpdate(const std::string& statement) {
        if (!is_open_) {
            throw std::runtime_error("Database is not open");
        }

        try {
            auto result = conn_->Query(statement);

            if (result->HasError()) {
                last_error_ = result->GetError();
                LOG_ERROR("Statement failed: {}", last_error_);
                return -1;
            }

            // Return number of affected rows
            return result->RowCount();
        } catch (const std::exception& e) {
            last_error_ = e.what();
            LOG_ERROR("Statement execution failed: {}", e.what());
            return -1;
        }
    }

    void beginTransaction() {
        executeUpdate("BEGIN TRANSACTION");
        LOG_DEBUG("Transaction started");
    }

    void commit() {
        executeUpdate("COMMIT");
        LOG_DEBUG("Transaction committed");
    }

    void rollback() {
        executeUpdate("ROLLBACK");
        LOG_DEBUG("Transaction rolled back");
    }

    int64_t bulkInsert(const std::string& table_name, const std::vector<DBRow>& rows) {
        if (rows.empty()) {
            return 0;
        }

        // Use COPY statement for efficient bulk insert
        // For now, use batched INSERT statements
        beginTransaction();

        try {
            int64_t inserted = 0;

            for (const auto& row : rows) {
                // Build INSERT statement
                std::string columns;
                std::string values;

                for (const auto& [col, val] : row) {
                    if (!columns.empty()) {
                        columns += ", ";
                        values += ", ";
                    }
                    columns += col;
                    values += valueToString(val);
                }

                std::string insert_stmt = "INSERT INTO " + table_name +
                                         " (" + columns + ") VALUES (" + values + ")";

                if (executeUpdate(insert_stmt) >= 0) {
                    inserted++;
                }
            }

            commit();
            LOG_INFO("Bulk inserted {} rows into {}", inserted, table_name);
            return inserted;

        } catch (const std::exception& e) {
            rollback();
            LOG_ERROR("Bulk insert failed: {}", e.what());
            return -1;
        }
    }

    bool loadParquet(const std::string& parquet_path, const std::string& table_name) {
        std::string table = table_name;
        if (table.empty()) {
            // Use filename as table name
            size_t last_slash = parquet_path.find_last_of("/\\");
            size_t last_dot = parquet_path.find_last_of(".");
            table = parquet_path.substr(last_slash + 1, last_dot - last_slash - 1);
        }

        std::string query = "CREATE TABLE " + table + " AS SELECT * FROM '" + parquet_path + "'";

        try {
            executeUpdate(query);
            LOG_INFO("Loaded Parquet file {} into table {}", parquet_path, table);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load Parquet: {}", e.what());
            return false;
        }
    }

    bool exportParquet(const std::string& table_name, const std::string& parquet_path) {
        std::string query = "COPY " + table_name + " TO '" + parquet_path + "' (FORMAT PARQUET)";

        try {
            executeUpdate(query);
            LOG_INFO("Exported table {} to Parquet: {}", table_name, parquet_path);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to export Parquet: {}", e.what());
            return false;
        }
    }

    std::string getLastError() const {
        return last_error_;
    }

    bool createTable(const std::string& create_statement) {
        return executeUpdate(create_statement) >= 0;
    }

    bool dropTable(const std::string& table_name) {
        std::string stmt = "DROP TABLE IF EXISTS " + table_name;
        return executeUpdate(stmt) >= 0;
    }

    bool tableExists(const std::string& table_name) {
        std::string query = "SELECT name FROM sqlite_master WHERE type='table' AND name='" + table_name + "'";
        auto result = execute(query);
        return result.rowCount() > 0;
    }

    std::vector<std::string> getTableSchema(const std::string& table_name) {
        std::string query = "PRAGMA table_info(" + table_name + ")";
        auto result = execute(query);

        std::vector<std::string> schema;
        for (const auto& row : result) {
            // Each row contains: cid, name, type, notnull, dflt_value, pk
            if (row.count("name") > 0) {
                auto name_val = row.at("name");
                if (std::holds_alternative<std::string>(name_val)) {
                    schema.push_back(std::get<std::string>(name_val));
                }
            }
        }

        return schema;
    }

    void vacuum() {
        executeUpdate("VACUUM");
        LOG_INFO("Database vacuumed");
    }

    void checkpoint() {
        executeUpdate("CHECKPOINT");
        LOG_DEBUG("Database checkpointed");
    }

private:
    DBResultSet convertResult(duckdb::MaterializedQueryResult& result) {
        DBResultSet resultSet;

        // Get column names
        std::vector<std::string> col_names;
        for (size_t i = 0; i < result.ColumnCount(); i++) {
            col_names.push_back(result.names[i]);
        }
        resultSet.setColumnNames(col_names);

        // Convert rows
        for (size_t row_idx = 0; row_idx < result.RowCount(); row_idx++) {
            DBRow row;

            for (size_t col_idx = 0; col_idx < result.ColumnCount(); col_idx++) {
                auto value = result.GetValue(col_idx, row_idx);
                row[col_names[col_idx]] = convertValue(value);
            }

            resultSet.addRow(std::move(row));
        }

        return resultSet;
    }

    DBValue convertValue(const duckdb::Value& value) {
        if (value.IsNull()) {
            return std::monostate{};
        }

        switch (value.type().id()) {
            case duckdb::LogicalTypeId::BOOLEAN:
                return value.GetValue<bool>();
            case duckdb::LogicalTypeId::TINYINT:
            case duckdb::LogicalTypeId::SMALLINT:
            case duckdb::LogicalTypeId::INTEGER:
            case duckdb::LogicalTypeId::BIGINT:
                return value.GetValue<int64_t>();
            case duckdb::LogicalTypeId::FLOAT:
            case duckdb::LogicalTypeId::DOUBLE:
                return value.GetValue<double>();
            case duckdb::LogicalTypeId::VARCHAR:
                return value.GetValue<std::string>();
            case duckdb::LogicalTypeId::BLOB:
                return std::vector<uint8_t>(); // TODO: implement blob conversion
            default:
                return value.ToString();
        }
    }

    std::string valueToString(const DBValue& value) {
        return std::visit([](auto&& arg) -> std::string {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return "NULL";
            } else if constexpr (std::is_same_v<T, bool>) {
                return arg ? "TRUE" : "FALSE";
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return std::to_string(arg);
            } else if constexpr (std::is_same_v<T, double>) {
                return std::to_string(arg);
            } else if constexpr (std::is_same_v<T, std::string>) {
                return "'" + arg + "'";
            } else {
                return "NULL";
            }
        }, value);
    }

    std::string db_path_;
    bool read_only_;
    bool is_open_;
    std::string last_error_;

    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;
};

#else // No DuckDB - Stub implementation

class Database::Impl {
public:
    Impl(const std::string& db_path, bool read_only)
        : db_path_(db_path), read_only_(read_only), is_open_(false) {
        LOG_ERROR("DuckDB support not compiled. Database operations will fail.");
    }

    bool open() {
        LOG_ERROR("DuckDB not available");
        return false;
    }

    void close() {}
    bool isOpen() const { return false; }

    DBResultSet execute(const std::string& query) {
        LOG_ERROR("DuckDB not available");
        return DBResultSet();
    }

    int64_t executeUpdate(const std::string& statement) {
        LOG_ERROR("DuckDB not available");
        return -1;
    }

    void beginTransaction() {}
    void commit() {}
    void rollback() {}

    int64_t bulkInsert(const std::string&, const std::vector<DBRow>&) { return -1; }
    bool loadParquet(const std::string&, const std::string&) { return false; }
    bool exportParquet(const std::string&, const std::string&) { return false; }

    std::string getLastError() const { return "DuckDB not available"; }
    bool createTable(const std::string&) { return false; }
    bool dropTable(const std::string&) { return false; }
    bool tableExists(const std::string&) { return false; }
    std::vector<std::string> getTableSchema(const std::string&) { return {}; }
    void vacuum() {}
    void checkpoint() {}

private:
    std::string db_path_;
    bool read_only_;
    bool is_open_;
};

#endif // HAS_DUCKDB

// Database public interface implementation
Database::Database(const std::string& db_path, bool read_only)
    : pImpl(std::make_unique<Impl>(db_path, read_only)) {}

Database::~Database() = default;

Database::Database(Database&&) noexcept = default;
Database& Database::operator=(Database&&) noexcept = default;

bool Database::open() { return pImpl->open(); }
void Database::close() { pImpl->close(); }
bool Database::isOpen() const { return pImpl->isOpen(); }

DBResultSet Database::execute(const std::string& query) {
    return pImpl->execute(query);
}

int64_t Database::executeUpdate(const std::string& statement) {
    return pImpl->executeUpdate(statement);
}

void Database::beginTransaction() { pImpl->beginTransaction(); }
void Database::commit() { pImpl->commit(); }
void Database::rollback() { pImpl->rollback(); }

int64_t Database::bulkInsert(const std::string& table_name, const std::vector<DBRow>& rows) {
    return pImpl->bulkInsert(table_name, rows);
}

bool Database::loadParquet(const std::string& parquet_path, const std::string& table_name) {
    return pImpl->loadParquet(parquet_path, table_name);
}

bool Database::exportParquet(const std::string& table_name, const std::string& parquet_path) {
    return pImpl->exportParquet(table_name, parquet_path);
}

std::string Database::getLastError() const {
    return pImpl->getLastError();
}

bool Database::createTable(const std::string& create_statement) {
    return pImpl->createTable(create_statement);
}

bool Database::dropTable(const std::string& table_name) {
    return pImpl->dropTable(table_name);
}

bool Database::tableExists(const std::string& table_name) {
    return pImpl->tableExists(table_name);
}

std::vector<std::string> Database::getTableSchema(const std::string& table_name) {
    return pImpl->getTableSchema(table_name);
}

void Database::vacuum() { pImpl->vacuum(); }
void Database::checkpoint() { pImpl->checkpoint(); }

// Transaction RAII implementation
Database::Transaction::Transaction(Database& db)
    : db_(db), committed_(false), rolled_back_(false) {
    db_.beginTransaction();
}

Database::Transaction::~Transaction() {
    if (!committed_ && !rolled_back_) {
        db_.rollback();
        LOG_WARN("Transaction auto-rolled back (not committed)");
    }
}

void Database::Transaction::commit() {
    if (!committed_ && !rolled_back_) {
        db_.commit();
        committed_ = true;
    }
}

void Database::Transaction::rollback() {
    if (!committed_ && !rolled_back_) {
        db_.rollback();
        rolled_back_ = true;
    }
}

// DatabasePool implementation
class DatabasePool::Impl {
public:
    Impl(const std::string& db_path, size_t pool_size)
        : db_path_(db_path), pool_size_(pool_size) {

        for (size_t i = 0; i < pool_size; i++) {
            auto db = std::make_shared<Database>(db_path, false);
            if (db->open()) {
                pool_.push(db);
            } else {
                LOG_ERROR("Failed to create database connection {}/{}", i + 1, pool_size);
            }
        }

        LOG_INFO("DatabasePool initialized with {} connections", pool_.size());
    }

    ~Impl() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!pool_.empty()) {
            auto db = pool_.front();
            pool_.pop();
            db->close();
        }
    }

    std::shared_ptr<Database> acquire() {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this] { return !pool_.empty(); });

        auto db = pool_.front();
        pool_.pop();

        return std::shared_ptr<Database>(db.get(), [this, db](Database*) {
            // Custom deleter: return connection to pool
            std::unique_lock<std::mutex> lock(mutex_);
            pool_.push(db);
            cv_.notify_one();
        });
    }

    DatabasePool::Stats getStats() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return {
            pool_size_,
            pool_size_ - pool_.size(),
            pool_.size()
        };
    }

private:
    std::string db_path_;
    size_t pool_size_;
    std::queue<std::shared_ptr<Database>> pool_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

DatabasePool::DatabasePool(const std::string& db_path, size_t pool_size)
    : pImpl(std::make_unique<Impl>(db_path, pool_size)) {}

DatabasePool::~DatabasePool() = default;

std::shared_ptr<Database> DatabasePool::acquire() {
    return pImpl->acquire();
}

DatabasePool::Stats DatabasePool::getStats() const {
    return pImpl->getStats();
}

} // namespace utils
} // namespace bigbrother
