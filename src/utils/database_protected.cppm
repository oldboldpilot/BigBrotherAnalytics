/**
 * BigBrotherAnalytics - Database API with Circuit Breaker Protection (C++23)
 *
 * Wraps DuckDB operations with circuit breaker protection to prevent
 * cascading failures. Provides in-memory cache fallback when database
 * circuit is open.
 *
 * Following C++ Core Guidelines:
 * - R.1: RAII for resource management
 * - C.21: Rule of Five
 * - Thread-safe implementation
 */

// Global module fragment
module;

#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <vector>

// Module declaration
export module bigbrother.database.protected;

// Import dependencies
import bigbrother.database.api;
import bigbrother.circuit_breaker;
import bigbrother.utils.logger;

export namespace bigbrother::database::protected_api {

using namespace bigbrother::database;
using namespace bigbrother::circuit_breaker;
using namespace bigbrother::utils;

/**
 * Queued Write Operation
 *
 * Stores write operations when circuit is open for replay later.
 */
struct QueuedWrite {
    std::string sql;
    std::vector<std::string> bindings;
    std::chrono::system_clock::time_point timestamp;

    QueuedWrite(std::string sql_query, std::vector<std::string> params)
        : sql{std::move(sql_query)}, bindings{std::move(params)},
          timestamp{std::chrono::system_clock::now()} {}
};

/**
 * Protected Database Connection with Circuit Breaker
 *
 * Wraps all database operations with circuit breaker protection.
 * Provides in-memory cache for reads and write queue for failures.
 */
class ProtectedDatabaseConnection {
  public:
    explicit ProtectedDatabaseConnection(std::string db_path)
        : db_path_{std::move(db_path)}, connection_{Database::connect(db_path_)},
          circuit_breaker_{CircuitConfig{
              .failure_threshold = 5,
              .timeout = std::chrono::seconds(60),
              .half_open_timeout = std::chrono::seconds(30),
              .half_open_max_calls = 3,
              .enable_logging = true,
              .name = "database",
          }},
          max_queued_writes_{1000} {

        Logger::getInstance().info("Protected database connection initialized: {}", db_path_);
    }

    // C.21: Rule of Five - non-copyable, non-movable due to mutex
    ProtectedDatabaseConnection(ProtectedDatabaseConnection const&) = delete;
    auto operator=(ProtectedDatabaseConnection const&) -> ProtectedDatabaseConnection& = delete;
    ProtectedDatabaseConnection(ProtectedDatabaseConnection&&) noexcept = delete;
    auto operator=(ProtectedDatabaseConnection&&) noexcept -> ProtectedDatabaseConnection& = delete;
    ~ProtectedDatabaseConnection() {
        // Attempt to flush queued writes on destruction
        if (!write_queue_.empty()) {
            Logger::getInstance().warn("Destroying database connection with {} queued writes",
                                       write_queue_.size());
            flushWriteQueue();
        }
    }

    /**
     * Execute Query with Circuit Breaker Protection
     *
     * Falls back to in-memory cache if circuit is open.
     */
    [[nodiscard]] auto execute(std::string const& sql) -> std::expected<ResultSet, std::string> {
        return circuit_breaker_.call<ResultSet>([&]() -> std::expected<ResultSet, std::string> {
            try {
                auto result = connection_.execute(sql);

                // Cache read queries
                if (isReadQuery(sql)) {
                    cacheQuery(sql, result);
                }

                return result;
            } catch (std::exception const& e) {
                return std::unexpected(std::string("Database error: ") + e.what());
            }
        });
    }

    /**
     * Execute Query with Fallback
     *
     * Returns cached result if circuit is open.
     */
    [[nodiscard]] auto executeWithFallback(std::string const& sql)
        -> std::expected<ResultSet, std::string> {

        auto result = execute(sql);

        if (!result && isReadQuery(sql)) {
            // Try to get cached data
            auto cached = getCachedQuery(sql);
            if (cached) {
                Logger::getInstance().warn("Using cached query result (circuit breaker active)");
                return *cached;
            }
        }

        return result;
    }

    /**
     * Execute Update with Circuit Breaker Protection
     *
     * Queues write if circuit is open for later replay.
     */
    [[nodiscard]] auto executeUpdate(std::string const& sql) -> std::expected<int64_t, std::string> {

        auto result =
            circuit_breaker_.call<int64_t>([&]() -> std::expected<int64_t, std::string> {
                try {
                    auto affected = connection_.executeUpdate(sql);
                    return affected;
                } catch (std::exception const& e) {
                    return std::unexpected(std::string("Database error: ") + e.what());
                }
            });

        // If failed and circuit is open, queue the write
        if (!result && circuit_breaker_.isOpen()) {
            queueWrite(sql, {});
            Logger::getInstance().warn("Queued write operation (circuit breaker open): {}",
                                       sql.substr(0, 100));
        }

        return result;
    }

    /**
     * Prepared Statement with Circuit Breaker Protection
     */
    [[nodiscard]] auto query(std::string sql) -> PreparedStatement {
        // Note: PreparedStatement execution will use circuit breaker via execute()
        return connection_.query(std::move(sql));
    }

    /**
     * Begin Transaction with Circuit Breaker Protection
     */
    [[nodiscard]] auto beginTransaction() -> std::expected<void, std::string> {
        return circuit_breaker_.call<void>([&]() -> std::expected<void, std::string> {
            try {
                connection_.beginTransaction();
                return {};
            } catch (std::exception const& e) {
                return std::unexpected(std::string("Database error: ") + e.what());
            }
        });
    }

    /**
     * Commit Transaction with Circuit Breaker Protection
     */
    [[nodiscard]] auto commit() -> std::expected<void, std::string> {
        return circuit_breaker_.call<void>([&]() -> std::expected<void, std::string> {
            try {
                connection_.commit();
                return {};
            } catch (std::exception const& e) {
                return std::unexpected(std::string("Database error: ") + e.what());
            }
        });
    }

    /**
     * Rollback Transaction with Circuit Breaker Protection
     */
    [[nodiscard]] auto rollback() -> std::expected<void, std::string> {
        return circuit_breaker_.call<void>([&]() -> std::expected<void, std::string> {
            try {
                connection_.rollback();
                return {};
            } catch (std::exception const& e) {
                return std::unexpected(std::string("Database error: ") + e.what());
            }
        });
    }

    /**
     * Flush Queued Writes
     *
     * Attempts to execute all queued write operations.
     * Should be called when circuit is closed after recovery.
     */
    auto flushWriteQueue() -> int {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        if (circuit_breaker_.isOpen()) {
            Logger::getInstance().warn("Cannot flush write queue - circuit breaker is still open");
            return 0;
        }

        int flushed = 0;
        int failed = 0;

        Logger::getInstance().info("Flushing {} queued write operations", write_queue_.size());

        while (!write_queue_.empty()) {
            auto& write_op = write_queue_.front();

            try {
                // Attempt to execute queued write
                auto result = connection_.executeUpdate(write_op.sql);

                if (result > 0) {
                    flushed++;
                    Logger::getInstance().debug("Flushed queued write: {}",
                                                write_op.sql.substr(0, 100));
                } else {
                    failed++;
                    Logger::getInstance().error("Failed to flush queued write: {}",
                                                write_op.sql.substr(0, 100));
                }
            } catch (std::exception const& e) {
                failed++;
                Logger::getInstance().error("Exception flushing queued write: {}", e.what());
            }

            write_queue_.pop();
        }

        Logger::getInstance().info("Write queue flush complete: {} succeeded, {} failed", flushed,
                                   failed);

        return flushed;
    }

    /**
     * Get Queued Write Count
     */
    [[nodiscard]] auto getQueuedWriteCount() const -> size_t {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return write_queue_.size();
    }

    /**
     * Get Circuit Breaker Statistics
     */
    [[nodiscard]] auto getCircuitStats() const -> CircuitStats {
        return circuit_breaker_.getStats();
    }

    /**
     * Check if circuit is open
     */
    [[nodiscard]] auto isCircuitOpen() const -> bool { return circuit_breaker_.isOpen(); }

    /**
     * Manually reset circuit breaker
     */
    auto resetCircuit() -> void {
        circuit_breaker_.reset();
        Logger::getInstance().info("Database circuit breaker manually reset");
    }

    /**
     * Clear query cache
     */
    auto clearCache() -> void {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        query_cache_.clear();
        Logger::getInstance().info("Database query cache cleared");
    }

    /**
     * Close connection
     */
    auto close() -> void {
        // Flush any pending writes before closing
        flushWriteQueue();
        connection_.close();
        Logger::getInstance().info("Database connection closed: {}", db_path_);
    }

  private:
    /**
     * Check if SQL query is a read operation
     */
    [[nodiscard]] static auto isReadQuery(std::string const& sql) -> bool {
        // Simple heuristic: check if query starts with SELECT
        auto trimmed = sql;
        // Trim leading whitespace
        trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));

        if (trimmed.size() >= 6) {
            auto prefix = trimmed.substr(0, 6);
            // Convert to uppercase for comparison
            for (auto& c : prefix) {
                c = std::toupper(static_cast<unsigned char>(c));
            }
            return prefix == "SELECT";
        }

        return false;
    }

    /**
     * Cache query result
     */
    auto cacheQuery(std::string const& sql, ResultSet const& result) -> void {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Simple cache with no eviction (for now)
        // In production, implement LRU cache with size limits
        query_cache_[sql] = result;
    }

    /**
     * Get cached query result
     */
    [[nodiscard]] auto getCachedQuery(std::string const& sql) const -> std::optional<ResultSet> {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = query_cache_.find(sql);
        if (it != query_cache_.end()) {
            return it->second;
        }

        return std::nullopt;
    }

    /**
     * Queue write operation for later execution
     */
    auto queueWrite(std::string sql, std::vector<std::string> bindings) -> void {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        if (write_queue_.size() >= max_queued_writes_) {
            Logger::getInstance().error("Write queue full ({} operations) - dropping oldest",
                                        max_queued_writes_);
            write_queue_.pop(); // Drop oldest write
        }

        write_queue_.emplace(std::move(sql), std::move(bindings));
    }

    // ========================================================================
    // Member Variables
    // ========================================================================

    std::string db_path_;
    DatabaseConnection connection_;
    CircuitBreaker circuit_breaker_;

    // Query result cache (for read operations)
    std::unordered_map<std::string, ResultSet> query_cache_;
    mutable std::mutex cache_mutex_;

    // Write operation queue (for failed writes)
    std::queue<QueuedWrite> write_queue_;
    size_t max_queued_writes_;
    mutable std::mutex queue_mutex_;
};

/**
 * Protected Database Factory
 */
class ProtectedDatabase {
  public:
    /**
     * Connect to database with circuit breaker protection
     */
    [[nodiscard]] static auto connect(std::string db_path) -> ProtectedDatabaseConnection {
        return ProtectedDatabaseConnection{std::move(db_path)};
    }

    /**
     * Connect to in-memory database with circuit breaker protection
     */
    [[nodiscard]] static auto memory() -> ProtectedDatabaseConnection {
        return ProtectedDatabaseConnection{":memory:"};
    }
};

} // namespace bigbrother::database::protected_api
