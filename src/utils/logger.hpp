#pragma once

#include <string>
#include <memory>
#include <source_location>
#include <format>

namespace bigbrother {
namespace utils {

enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    CRITICAL
};

/**
 * Logger class - Thread-safe logging with multiple backends
 *
 * Features:
 * - Multiple log levels (trace, debug, info, warn, error, critical)
 * - Thread-safe operation
 * - File and console output
 * - Automatic timestamp and source location
 * - Lazy evaluation for performance
 * - Integration with spdlog if available
 */
class Logger {
public:
    [[nodiscard]] static auto getInstance() -> Logger&;

    // Delete copy and move constructors
    Logger(Logger const&) = delete;
    auto operator=(Logger const&) -> Logger& = delete;
    Logger(Logger&&) = delete;
    auto operator=(Logger&&) -> Logger& = delete;

    // Initialize logger with configuration
    auto initialize(
        std::string const& log_file_path = "logs/bigbrother.log",
        LogLevel level = LogLevel::INFO,
        bool console_output = true
    ) -> void;

    // Set log level
    auto setLevel(LogLevel level) -> void;
    [[nodiscard]] auto getLevel() const -> LogLevel;

    // Logging methods with automatic source location
    template<typename... Args>
    void trace(const std::string& msg, Args&&... args) {
        logFormatted(LogLevel::TRACE, msg, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void debug(const std::string& msg, Args&&... args) {
        logFormatted(LogLevel::DEBUG, msg, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(const std::string& msg, Args&&... args) {
        logFormatted(LogLevel::INFO, msg, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warn(const std::string& msg, Args&&... args) {
        logFormatted(LogLevel::WARN, msg, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(const std::string& msg, Args&&... args) {
        logFormatted(LogLevel::ERROR, msg, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void critical(const std::string& msg, Args&&... args) {
        logFormatted(LogLevel::CRITICAL, msg, std::forward<Args>(args)...);
    }

    // Flush logs to disk
    void flush();

private:
    Logger();
    ~Logger();

    // Internal logging method
    auto logMessage(LogLevel level, std::string const& msg) -> void;

    template<typename... Args>
    void logFormatted(LogLevel level, const std::string& fmt, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            logMessage(level, fmt);
        } else {
            try {
                std::string formatted = std::vformat(fmt, std::make_format_args(args...));
                logMessage(level, formatted);
            } catch (...) {
                logMessage(level, fmt);  // Fall back to unformatted
            }
        }
    }

    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Convenience macros for logging
#define LOG_TRACE(msg, ...) \
    ::bigbrother::utils::Logger::getInstance().trace(msg, ##__VA_ARGS__)

#define LOG_DEBUG(msg, ...) \
    ::bigbrother::utils::Logger::getInstance().debug(msg, ##__VA_ARGS__)

#define LOG_INFO(msg, ...) \
    ::bigbrother::utils::Logger::getInstance().info(msg, ##__VA_ARGS__)

#define LOG_WARN(msg, ...) \
    ::bigbrother::utils::Logger::getInstance().warn(msg, ##__VA_ARGS__)

#define LOG_ERROR(msg, ...) \
    ::bigbrother::utils::Logger::getInstance().error(msg, ##__VA_ARGS__)

#define LOG_CRITICAL(msg, ...) \
    ::bigbrother::utils::Logger::getInstance().critical(msg, ##__VA_ARGS__)

} // namespace utils
} // namespace bigbrother
