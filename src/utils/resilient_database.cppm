/**
 * BigBrotherAnalytics - Resilient Database Wrapper (C++23)
 *
 * Enhanced database operations with:
 * - Automatic retry on lock contention
 * - Transaction management with rollback
 * - Graceful degradation to in-memory mode
 * - Connection pooling and health monitoring
 * - State preservation for recovery
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 * Phase: 4 (Week 2 - Error Handling & Retry Logic)
 */

// Global module fragment
module;

#include <atomic>
#include <chrono>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

// DuckDB bridge - isolates DuckDB incomplete types from C++23 modules
#include "schwab_api/duckdb_bridge.hpp"

// Module declaration
export module bigbrother.utils.resilient_database;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.utils.retry;

export namespace bigbrother::utils {

using namespace bigbrother::types;

// ============================================================================
// Database Operation Modes
// ============================================================================

/**
 * Operating mode for database
 */
enum class DatabaseMode {
    Normal,         // Full database functionality
    Degraded,       // Database available but experiencing issues
    InMemory,       // Fallback to in-memory cache (no persistence)
    Unavailable     // Database completely unavailable
};

/**
 * Transaction isolation levels
 */
enum class IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable
};

// ============================================================================
// Database Configuration
// ============================================================================

/**
 * Configuration for resilient database
 */
struct ResilientDatabaseConfig {
    std::string db_path{"data/bigbrother.duckdb"};
    bool enable_in_memory_fallback{true};
    int max_retry_attempts{3};
    std::chrono::milliseconds lock_timeout{5000};
    std::chrono::milliseconds retry_delay{100};
    int connection_pool_size{4};
    bool auto_checkpoint{true};
    std::chrono::seconds checkpoint_interval{300};  // 5 minutes

    [[nodiscard]] static auto defaultConfig() noexcept -> ResilientDatabaseConfig {
        return ResilientDatabaseConfig{};
    }
};

// ============================================================================
// In-Memory Operation Cache
// ============================================================================

/**
 * Cache for database operations when database is unavailable
 */
struct CachedOperation {
    std::string sql;
    std::chrono::system_clock::time_point timestamp;
    int retry_count{0};
};

class OperationCache {
public:
    auto add(std::string sql) -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        cached_operations_.push({std::move(sql),
                                std::chrono::system_clock::now(),
                                0});
        Logger::getInstance().warn("Cached database operation (queue size: {})",
                                  cached_operations_.size());
    }

    [[nodiscard]] auto getNext() -> std::optional<CachedOperation> {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cached_operations_.empty()) {
            return std::nullopt;
        }
        auto op = cached_operations_.front();
        cached_operations_.pop();
        return op;
    }

    [[nodiscard]] auto size() const -> size_t {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_operations_.size();
    }

    auto clear() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!cached_operations_.empty()) {
            cached_operations_.pop();
        }
    }

private:
    mutable std::mutex mutex_;
    std::queue<CachedOperation> cached_operations_;
};

// ============================================================================
// Database Health Tracker
// ============================================================================

/**
 * Track database health metrics
 */
struct DatabaseHealth {
    DatabaseMode mode{DatabaseMode::Normal};
    std::atomic<int> total_operations{0};
    std::atomic<int> successful_operations{0};
    std::atomic<int> failed_operations{0};
    std::atomic<int> retried_operations{0};
    std::atomic<int> cached_operations{0};
    std::chrono::steady_clock::time_point last_successful_operation;
    std::chrono::steady_clock::time_point last_failure;
    std::string last_error_message;

    [[nodiscard]] auto getSuccessRate() const noexcept -> double {
        int total = total_operations.load();
        if (total == 0) return 1.0;
        return static_cast<double>(successful_operations.load()) / total;
    }

    [[nodiscard]] auto isHealthy() const noexcept -> bool {
        return mode == DatabaseMode::Normal &&
               getSuccessRate() >= 0.95;
    }
};

// ============================================================================
// Resilient Database
// ============================================================================

/**
 * Database wrapper with automatic retry and graceful degradation
 */
class ResilientDatabase {
public:
    explicit ResilientDatabase(ResilientDatabaseConfig config = ResilientDatabaseConfig::defaultConfig())
        : config_{std::move(config)},
          retry_engine_{getRetryEngine()},
          health_{} {

        health_.mode = DatabaseMode::Unavailable;
    }

    // Rule of Five - deleted due to mutex member
    ResilientDatabase(ResilientDatabase const&) = delete;
    auto operator=(ResilientDatabase const&) -> ResilientDatabase& = delete;
    ResilientDatabase(ResilientDatabase&&) noexcept = delete;
    auto operator=(ResilientDatabase&&) noexcept -> ResilientDatabase& = delete;

    ~ResilientDatabase() {
        disconnect();
    }

    /**
     * Connect to database with retry
     */
    [[nodiscard]] auto connect() -> Result<void> {
        std::lock_guard<std::mutex> lock(mutex_);

        Logger::getInstance().info("Connecting to database: {}", config_.db_path);

        auto connect_operation = [this]() -> Result<void> {
            try {
                // Use DuckDB bridge to avoid incomplete type issues with C++23 modules
                db_ = duckdb_bridge::openDatabase(config_.db_path);
                conn_ = duckdb_bridge::createConnection(*db_);

                health_.mode = DatabaseMode::Normal;
                health_.last_successful_operation = std::chrono::steady_clock::now();

                Logger::getInstance().info("Database connected successfully");
                return {};

            } catch (duckdb::Exception const& e) {
                return makeError<void>(ErrorCode::DatabaseError,
                                     std::string("DuckDB connection failed: ") + e.what());
            } catch (std::exception const& e) {
                return makeError<void>(ErrorCode::DatabaseError,
                                     std::string("Database connection failed: ") + e.what());
            }
        };

        // Retry connection with conservative policy
        auto result = retry_engine_.execute<void>(
            connect_operation,
            "database_connect",
            RetryPolicy::conservativePolicy()
        );

        if (!result) {
            health_.mode = DatabaseMode::Unavailable;
            health_.last_error_message = result.error().message;
            health_.last_failure = std::chrono::steady_clock::now();

            if (config_.enable_in_memory_fallback) {
                Logger::getInstance().error(
                    "Database connection failed, entering IN-MEMORY mode: {}",
                    result.error().message);
                health_.mode = DatabaseMode::InMemory;
            }
        }

        return result;
    }

    /**
     * Disconnect from database
     */
    auto disconnect() -> void {
        std::lock_guard<std::mutex> lock(mutex_);

        if (conn_) {
            conn_.reset();
            Logger::getInstance().info("Database connection closed");
        }

        if (db_) {
            db_.reset();
        }

        health_.mode = DatabaseMode::Unavailable;
    }

    /**
     * Execute SQL query with automatic retry on lock contention
     */
    [[nodiscard]] auto execute(std::string const& sql) -> Result<void> {
        health_.total_operations++;

        // Check if database is available
        if (health_.mode == DatabaseMode::Unavailable) {
            if (config_.enable_in_memory_fallback) {
                // Cache operation for later execution
                operation_cache_.add(sql);
                health_.cached_operations++;
                return {};
            }

            return makeError<void>(ErrorCode::DatabaseError,
                                 "Database unavailable and in-memory fallback disabled");
        }

        // In-memory mode - just cache
        if (health_.mode == DatabaseMode::InMemory) {
            operation_cache_.add(sql);
            health_.cached_operations++;
            return {};
        }

        // Execute with retry logic
        auto execute_operation = [this, &sql]() -> Result<void> {
            std::lock_guard<std::mutex> lock(mutex_);

            if (!conn_) {
                return makeError<void>(ErrorCode::DatabaseError, "No active connection");
            }

            try {
                conn_->Query(sql);
                health_.successful_operations++;
                health_.last_successful_operation = std::chrono::steady_clock::now();
                return {};

            } catch (duckdb::Exception const& e) {
                std::string error_msg = e.what();

                // Check if this is a lock contention error (transient)
                if (error_msg.find("locked") != std::string::npos ||
                    error_msg.find("busy") != std::string::npos) {
                    return makeError<void>(ErrorCode::DatabaseLocked,
                                         "Database locked: " + error_msg);
                }

                // Other database errors (potentially permanent)
                return makeError<void>(ErrorCode::DatabaseError,
                                     "Database error: " + error_msg);

            } catch (std::exception const& e) {
                return makeError<void>(ErrorCode::DatabaseError,
                                     std::string("Unexpected error: ") + e.what());
            }
        };

        // Use database-specific retry policy
        auto result = retry_engine_.execute<void>(
            execute_operation,
            "database_execute",
            RetryPolicy{
                .max_attempts = config_.max_retry_attempts,
                .initial_backoff = config_.retry_delay,
                .max_backoff = config_.lock_timeout,
                .backoff_multiplier = 1.5,
                .add_jitter = true
            }
        );

        if (!result) {
            health_.failed_operations++;
            health_.last_error_message = result.error().message;
            health_.last_failure = std::chrono::steady_clock::now();

            // Check if we should enter degraded mode
            if (health_.getSuccessRate() < 0.80 && health_.mode == DatabaseMode::Normal) {
                Logger::getInstance().warn("Database entering DEGRADED mode (success rate: {:.1f}%)",
                                         health_.getSuccessRate() * 100.0);
                health_.mode = DatabaseMode::Degraded;
            }

            // Fallback to in-memory if enabled
            if (config_.enable_in_memory_fallback) {
                operation_cache_.add(sql);
                health_.cached_operations++;
                Logger::getInstance().warn("Operation cached due to database error: {}",
                                         result.error().message);
                // Return success since we cached it
                return {};
            }
        }

        return result;
    }

    /**
     * Execute query with result set
     */
    [[nodiscard]] auto query(std::string const& sql)
        -> Result<std::unique_ptr<duckdb::MaterializedQueryResult>> {

        health_.total_operations++;

        if (health_.mode == DatabaseMode::Unavailable || health_.mode == DatabaseMode::InMemory) {
            return makeError<std::unique_ptr<duckdb::MaterializedQueryResult>>(
                ErrorCode::DatabaseError, "Database unavailable - cannot execute queries");
        }

        auto query_operation = [this, &sql]()
            -> Result<std::unique_ptr<duckdb::MaterializedQueryResult>> {
            std::lock_guard<std::mutex> lock(mutex_);

            if (!conn_) {
                return makeError<std::unique_ptr<duckdb::MaterializedQueryResult>>(
                    ErrorCode::DatabaseError, "No active connection");
            }

            try {
                auto result = conn_->Query(sql);

                if (result->HasError()) {
                    return makeError<std::unique_ptr<duckdb::MaterializedQueryResult>>(
                        ErrorCode::DatabaseError, result->GetError());
                }

                health_.successful_operations++;
                health_.last_successful_operation = std::chrono::steady_clock::now();

                return std::unique_ptr<duckdb::MaterializedQueryResult>(
                    static_cast<duckdb::MaterializedQueryResult*>(result.release()));

            } catch (duckdb::Exception const& e) {
                std::string error_msg = e.what();

                if (error_msg.find("locked") != std::string::npos ||
                    error_msg.find("busy") != std::string::npos) {
                    return makeError<std::unique_ptr<duckdb::MaterializedQueryResult>>(
                        ErrorCode::DatabaseLocked, "Database locked: " + error_msg);
                }

                return makeError<std::unique_ptr<duckdb::MaterializedQueryResult>>(
                    ErrorCode::DatabaseError, "Database error: " + error_msg);

            } catch (std::exception const& e) {
                return makeError<std::unique_ptr<duckdb::MaterializedQueryResult>>(
                    ErrorCode::DatabaseError, std::string("Unexpected error: ") + e.what());
            }
        };

        auto result = retry_engine_.execute<std::unique_ptr<duckdb::MaterializedQueryResult>>(
            query_operation,
            "database_query",
            RetryPolicy{
                .max_attempts = config_.max_retry_attempts,
                .initial_backoff = config_.retry_delay,
                .max_backoff = config_.lock_timeout,
                .backoff_multiplier = 1.5
            }
        );

        if (!result) {
            health_.failed_operations++;
            health_.last_error_message = result.error().message;
            health_.last_failure = std::chrono::steady_clock::now();
        }

        return result;
    }

    /**
     * Begin transaction with automatic retry
     */
    [[nodiscard]] auto beginTransaction() -> Result<void> {
        return execute("BEGIN TRANSACTION");
    }

    /**
     * Commit transaction
     */
    [[nodiscard]] auto commit() -> Result<void> {
        return execute("COMMIT");
    }

    /**
     * Rollback transaction
     */
    [[nodiscard]] auto rollback() -> Result<void> {
        return execute("ROLLBACK");
    }

    /**
     * Execute operation within transaction (with automatic rollback on failure)
     */
    template<typename Func>
    [[nodiscard]] auto executeInTransaction(Func&& operation) -> Result<void> {
        auto begin_result = beginTransaction();
        if (!begin_result) {
            return begin_result;
        }

        try {
            auto op_result = operation();

            if (!op_result) {
                // Operation failed, rollback
                Logger::getInstance().warn("Transaction operation failed, rolling back");
                rollback();
                return op_result;
            }

            // Operation succeeded, commit
            auto commit_result = commit();
            if (!commit_result) {
                Logger::getInstance().error("Transaction commit failed, attempting rollback");
                rollback();
                return commit_result;
            }

            return {};

        } catch (std::exception const& e) {
            Logger::getInstance().error("Transaction exception: {}, rolling back", e.what());
            rollback();
            return makeError<void>(ErrorCode::DatabaseError,
                                 std::string("Transaction failed: ") + e.what());
        }
    }

    /**
     * Flush cached operations when database becomes available
     */
    [[nodiscard]] auto flushCache() -> Result<int> {
        if (health_.mode == DatabaseMode::Unavailable ||
            health_.mode == DatabaseMode::InMemory) {
            return makeError<int>(ErrorCode::DatabaseError,
                                "Cannot flush cache - database unavailable");
        }

        int flushed_count = 0;
        int failed_count = 0;

        Logger::getInstance().info("Flushing {} cached operations",
                                 operation_cache_.size());

        while (auto op = operation_cache_.getNext()) {
            auto result = execute(op->sql);
            if (result) {
                flushed_count++;
            } else {
                failed_count++;
                Logger::getInstance().error("Failed to flush cached operation: {}",
                                          result.error().message);

                // Re-cache if retry count is low
                if (op->retry_count < 3) {
                    op->retry_count++;
                    operation_cache_.add(op->sql);
                }
            }
        }

        Logger::getInstance().info("Flushed {} operations ({} failed)",
                                 flushed_count, failed_count);

        return flushed_count;
    }

    /**
     * Get current database mode
     */
    [[nodiscard]] auto getMode() const -> DatabaseMode {
        return health_.mode;
    }

    /**
     * Get database health metrics
     */
    [[nodiscard]] auto getHealth() const -> DatabaseHealth {
        return health_;
    }

    /**
     * Check if database is available
     */
    [[nodiscard]] auto isAvailable() const -> bool {
        return health_.mode == DatabaseMode::Normal ||
               health_.mode == DatabaseMode::Degraded;
    }

    /**
     * Get number of cached operations
     */
    [[nodiscard]] auto getCachedOperationCount() const -> size_t {
        return operation_cache_.size();
    }

private:
    ResilientDatabaseConfig config_;
    RetryEngine& retry_engine_;
    // TODO: Full migration to bridge API requires converting all conn_->Query() calls
    std::unique_ptr<duckdb_bridge::DatabaseHandle> db_;
    std::unique_ptr<duckdb_bridge::ConnectionHandle> conn_;
    DatabaseHealth health_;
    OperationCache operation_cache_;
    mutable std::mutex mutex_;
};

} // namespace bigbrother::utils
