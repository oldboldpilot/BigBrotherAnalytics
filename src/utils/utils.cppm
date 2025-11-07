/**
 * BigBrother Analytics - Utils Module
 *
 * C++23 module for utility functions and classes.
 * Using modules significantly speeds up compilation by:
 * - Eliminating redundant header parsing
 * - Providing better encapsulation
 * - Enabling better optimization opportunities
 *
 * Usage:
 *   import bigbrother.utils;
 *
 * Instead of:
 *   #include "utils/logger.hpp"
 *   #include "utils/config.hpp"
 *   // etc.
 */

export module bigbrother.utils;

// Export all public interfaces from utils
export import :logger;
export import :config;
export import :database;
export import :timer;
export import :types;
export import :math;

/**
 * Module partitions for each component
 * Each partition is defined in a separate .cppm file
 */

// Logger partition
export module bigbrother.utils:logger;

export namespace bigbrother::utils {
    enum class LogLevel {
        TRACE,
        DEBUG,
        INFO,
        WARN,
        ERROR,
        CRITICAL
    };

    class Logger {
    public:
        static auto getInstance() -> Logger&;

        auto initialize(std::string const& log_file_path = "logs/bigbrother.log",
                       LogLevel level = LogLevel::INFO,
                       bool console_output = true) -> void;

        auto setLevel(LogLevel level) -> void;
        [[nodiscard]] auto getLevel() const -> LogLevel;

        template<typename... Args>
        auto trace(std::string const& msg, Args&&... args,
                  std::source_location const& loc = std::source_location::current()) -> void;

        template<typename... Args>
        auto debug(std::string const& msg, Args&&... args,
                  std::source_location const& loc = std::source_location::current()) -> void;

        template<typename... Args>
        auto info(std::string const& msg, Args&&... args,
                 std::source_location const& loc = std::source_location::current()) -> void;

        template<typename... Args>
        auto warn(std::string const& msg, Args&&... args,
                 std::source_location const& loc = std::source_location::current()) -> void;

        template<typename... Args>
        auto error(std::string const& msg, Args&&... args,
                  std::source_location const& loc = std::source_location::current()) -> void;

        template<typename... Args>
        auto critical(std::string const& msg, Args&&... args,
                     std::source_location const& loc = std::source_location::current()) -> void;

        auto flush() -> void;

    private:
        Logger();
        ~Logger();

        class Impl;
        std::unique_ptr<Impl> pImpl;
    };
} // namespace bigbrother::utils

// Config partition
export module bigbrother.utils:config;

export namespace bigbrother::utils {
    class Config {
    public:
        [[nodiscard]] static auto getInstance() -> Config&;

        [[nodiscard]] auto load(std::string const& config_file_path) -> bool;
        [[nodiscard]] auto reload() -> bool;

        template<typename T>
        [[nodiscard]] auto get(std::string const& key) const -> std::optional<T>;

        template<typename T>
        [[nodiscard]] auto get(std::string const& key, T const& default_value) const -> T;

        [[nodiscard]] auto has(std::string const& key) const -> bool;

        [[nodiscard]] auto keys(std::string const& section = "") const
            -> std::vector<std::string>;

        template<typename T>
        auto set(std::string const& key, T const& value) -> void;

        [[nodiscard]] auto save(std::string const& config_file_path) const -> bool;

        auto clear() -> void;

    private:
        Config();
        ~Config();

        class Impl;
        std::unique_ptr<Impl> pImpl;
    };
} // namespace bigbrother::utils

// Database partition
export module bigbrother.utils:database;

export namespace bigbrother::utils {
    // Database value type
    using DBValue = std::variant<
        std::monostate,
        int64_t,
        double,
        std::string,
        bool,
        std::vector<uint8_t>
    >;

    using DBRow = std::map<std::string, DBValue>;

    class DBResultSet {
    public:
        DBResultSet() = default;

        auto addRow(DBRow row) -> void;
        auto setColumnNames(std::vector<std::string> names) -> void;

        [[nodiscard]] auto rowCount() const -> size_t;
        [[nodiscard]] auto columnCount() const -> size_t;

        [[nodiscard]] auto getRows() const -> std::vector<DBRow> const&;
        [[nodiscard]] auto getColumnNames() const -> std::vector<std::string> const&;

        [[nodiscard]] auto operator[](size_t index) const -> DBRow const&;

        [[nodiscard]] auto begin() const;
        [[nodiscard]] auto end() const;
    };

    class Database {
    public:
        explicit Database(std::string const& db_path = "data/bigbrother.duckdb",
                         bool read_only = false);
        ~Database();

        Database(Database&&) noexcept;
        auto operator=(Database&&) noexcept -> Database&;

        [[nodiscard]] auto open() -> bool;
        auto close() -> void;
        [[nodiscard]] auto isOpen() const -> bool;

        [[nodiscard]] auto execute(std::string const& query) -> DBResultSet;

        [[nodiscard]] auto executeUpdate(std::string const& statement) -> int64_t;

        auto beginTransaction() -> void;
        auto commit() -> void;
        auto rollback() -> void;

        class Transaction {
        public:
            explicit Transaction(Database& db);
            ~Transaction();

            auto commit() -> void;
            auto rollback() -> void;

        private:
            Database& db_;
            bool committed_;
            bool rolled_back_;
        };

        [[nodiscard]] auto transaction() -> Transaction;

        [[nodiscard]] auto bulkInsert(std::string const& table_name,
                                       std::vector<DBRow> const& rows) -> int64_t;

        [[nodiscard]] auto loadParquet(std::string const& parquet_path,
                                        std::string const& table_name = "") -> bool;

        [[nodiscard]] auto exportParquet(std::string const& table_name,
                                          std::string const& parquet_path) -> bool;

        [[nodiscard]] auto getLastError() const -> std::string;

        [[nodiscard]] auto createTable(std::string const& create_statement) -> bool;
        [[nodiscard]] auto dropTable(std::string const& table_name) -> bool;
        [[nodiscard]] auto tableExists(std::string const& table_name) -> bool;

        [[nodiscard]] auto getTableSchema(std::string const& table_name)
            -> std::vector<std::string>;

        auto vacuum() -> void;
        auto checkpoint() -> void;

    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;
    };

    class DatabasePool {
    public:
        explicit DatabasePool(std::string const& db_path, size_t pool_size = 10);
        ~DatabasePool();

        [[nodiscard]] auto acquire() -> std::shared_ptr<Database>;

        struct Stats {
            size_t total_connections;
            size_t active_connections;
            size_t idle_connections;
        };

        [[nodiscard]] auto getStats() const -> Stats;

    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;
    };
} // namespace bigbrother::utils

// Timer partition
export module bigbrother.utils:timer;

export namespace bigbrother::utils {
    class Timer {
    public:
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;
        using Duration = std::chrono::duration<double, std::micro>;

        Timer();

        auto start() -> void;
        [[nodiscard]] auto stop() -> double;
        [[nodiscard]] auto elapsed() const -> double;
        [[nodiscard]] auto elapsedMillis() const -> double;
        [[nodiscard]] auto elapsedSeconds() const -> double;
        auto reset() -> void;
        [[nodiscard]] auto isRunning() const -> bool;

        [[nodiscard]] static auto now() -> int64_t;
        [[nodiscard]] static auto timepoint() -> TimePoint;
    };

    class ScopedTimer {
    public:
        explicit ScopedTimer(std::string const& name);
        ~ScopedTimer();

        auto stop() -> void;
    };

    class Profiler {
    public:
        struct Stats {
            std::string name;
            size_t count;
            double total_us;
            double mean_us;
            double min_us;
            double max_us;
            double stddev_us;
            double median_us;
            double p95_us;
            double p99_us;
        };

        [[nodiscard]] static auto getInstance() -> Profiler&;

        auto record(std::string const& name, double elapsed_us) -> void;

        [[nodiscard]] auto getStats(std::string const& name) const -> Stats;
        [[nodiscard]] auto getAllStats() const -> std::vector<Stats>;

        auto printStats() const -> void;
        auto clear() -> void;
        auto clear(std::string const& name) -> void;

        [[nodiscard]] auto saveToFile(std::string const& filename) const -> bool;

        class ProfileGuard {
        public:
            explicit ProfileGuard(std::string const& name);
            ~ProfileGuard();
        };

        [[nodiscard]] auto profile(std::string const& name) -> ProfileGuard;
    };

    class RateLimiter {
    public:
        explicit RateLimiter(double max_rate);

        auto acquire() -> void;
        [[nodiscard]] auto tryAcquire() -> bool;

        [[nodiscard]] auto getRate() const -> double;
        auto setRate(double max_rate) -> void;
        auto reset() -> void;
    };

    class LatencyMonitor {
    public:
        struct Alert {
            std::string name;
            double latency_us;
            double threshold_us;
            Timer::TimePoint timestamp;
        };

        using AlertCallback = std::function<void(Alert const&)>;

        LatencyMonitor(std::string const& name, double threshold_us);

        auto record(double latency_us) -> void;
        auto setAlertCallback(AlertCallback callback) -> void;

        [[nodiscard]] auto getStats() const -> Profiler::Stats;
        auto clear() -> void;
    };
} // namespace bigbrother::utils

// Types partition
export module bigbrother.utils:types;

export namespace bigbrother::types {
    // All type definitions exported...
    // (Full definitions omitted for brevity - would include all from types.hpp)
} // namespace bigbrother::types

// Math partition
export module bigbrother.utils:math;

export namespace bigbrother::utils::math {
    // All math functions exported...
    // (Full definitions omitted for brevity - would include all from math.hpp)
} // namespace bigbrother::utils::math
