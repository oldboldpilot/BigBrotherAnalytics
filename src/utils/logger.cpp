#include "logger.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>

namespace bigbrother::utils {

/**
 * Thread-Safe Logger Implementation
 *
 * Now that build is working, added thread safety for:
 * - WebSocket streaming (separate thread)
 * - Concurrent strategy execution
 * - Parallel market data processing
 * - Real-time order execution
 */

class Logger::Impl {
public:
    Impl() : current_level(LogLevel::INFO), console_enabled(true), initialized(false) {}

    void initialize(const std::string& log_file_path, LogLevel level, bool console) {
        current_level = level;
        console_enabled = console;

        // Create log directory if it doesn't exist
        std::filesystem::path log_path(log_file_path);
        std::filesystem::create_directories(log_path.parent_path());

        // Open log file
        log_file.open(log_file_path, std::ios::app);
        if (!log_file.is_open()) {
            std::cerr << "Failed to open log file: " << log_file_path << std::endl;
        } else {
            initialized = true;
            std::cout << "Logger initialized: " << log_file_path << std::endl;
        }
    }

    void setLevel(LogLevel level) {
        current_level = level;
    }

    LogLevel getLevel() const {
        return current_level;
    }

    void log(LogLevel level, const std::string& msg, const std::source_location& loc) {
        if (level < current_level) {
            return;  // Skip if level is below threshold
        }

        std::lock_guard<std::mutex> lock(mutex_);  // Thread-safe logging

        // Get current time
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        // Format log entry
        std::ostringstream ss;
        ss << std::put_time(std::localtime(&time), "[%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms.count() << "] "
           << "[" << levelToString(level) << "] "
           << "[" << loc.file_name() << ":" << loc.line() << "] "
           << msg << std::endl;

        std::string log_entry = ss.str();

        // Output to console
        if (console_enabled) {
            std::cout << log_entry;
        }

        // Output to file
        if (log_file.is_open()) {
            log_file << log_entry;
            log_file.flush();  // Ensure immediate write
        }
    }

    void flush() {
        if (log_file.is_open()) {
            log_file.flush();
        }
    }

private:
    static std::string levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARN: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }

    LogLevel current_level;
    bool console_enabled;
    bool initialized;
    std::ofstream log_file;
    std::mutex mutex_;  // Thread-safety
};

// Logger singleton implementation
Logger::Logger() : pImpl(std::make_unique<Impl>()) {}

Logger::~Logger() {
    flush();
}

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::initialize(const std::string& log_file_path, LogLevel level, bool console_output) {
    pImpl->initialize(log_file_path, level, console_output);
}

void Logger::setLevel(LogLevel level) {
    pImpl->setLevel(level);
}

LogLevel Logger::getLevel() const {
    return pImpl->getLevel();
}

void Logger::flush() {
    pImpl->flush();
}

void Logger::logMessage(LogLevel level, const std::string& msg) {
    pImpl->log(level, msg, std::source_location::current());
}

} // namespace bigbrother::utils
