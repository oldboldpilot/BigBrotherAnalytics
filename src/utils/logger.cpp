#include "logger.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <format>

#ifdef HAS_SPDLOG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#endif

namespace bigbrother {
namespace utils {

class Logger::Impl {
public:
    Impl() : current_level(LogLevel::INFO), console_enabled(true), initialized(false) {}

    void initialize(const std::string& log_file_path, LogLevel level, bool console) {
        std::lock_guard<std::mutex> lock(mutex);

        current_level = level;
        console_enabled = console;

        // Create log directory if it doesn't exist
        std::filesystem::path log_path(log_file_path);
        std::filesystem::create_directories(log_path.parent_path());

#ifdef HAS_SPDLOG
        try {
            // Create spdlog logger with console and file sinks
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                log_file_path, 1024 * 1024 * 10, 3  // 10MB, 3 files
            );

            std::vector<spdlog::sink_ptr> sinks{file_sink};
            if (console) {
                sinks.push_back(console_sink);
            }

            spdlogger = std::make_shared<spdlog::logger>("bigbrother", sinks.begin(), sinks.end());
            spdlogger->set_level(toSpdlogLevel(level));
            spdlogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
            spdlog::register_logger(spdlogger);

            initialized = true;
            std::cout << "Logger initialized with spdlog" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize spdlog: " << e.what() << std::endl;
            fallbackInit(log_file_path);
        }
#else
        fallbackInit(log_file_path);
#endif
    }

    void setLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex);
        current_level = level;
#ifdef HAS_SPDLOG
        if (spdlogger) {
            spdlogger->set_level(toSpdlogLevel(level));
        }
#endif
    }

    LogLevel getLevel() const {
        return current_level;
    }

    void log(LogLevel level, const std::string& msg, const std::source_location& loc) {
        if (level < current_level) {
            return;  // Skip if level is below threshold
        }

        std::lock_guard<std::mutex> lock(mutex);

#ifdef HAS_SPDLOG
        if (spdlogger) {
            std::string formatted_msg = std::format("[{}:{}] {}",
                                                   loc.file_name(),
                                                   loc.line(),
                                                   msg);
            switch (level) {
                case LogLevel::TRACE:
                    spdlogger->trace(formatted_msg);
                    break;
                case LogLevel::DEBUG:
                    spdlogger->debug(formatted_msg);
                    break;
                case LogLevel::INFO:
                    spdlogger->info(formatted_msg);
                    break;
                case LogLevel::WARN:
                    spdlogger->warn(formatted_msg);
                    break;
                case LogLevel::ERROR:
                    spdlogger->error(formatted_msg);
                    break;
                case LogLevel::CRITICAL:
                    spdlogger->critical(formatted_msg);
                    break;
            }
            return;
        }
#endif

        // Fallback logging
        fallbackLog(level, msg, loc);
    }

    void flush() {
#ifdef HAS_SPDLOG
        if (spdlogger) {
            spdlogger->flush();
        }
#endif
        if (log_file.is_open()) {
            log_file.flush();
        }
    }

private:
    void fallbackInit(const std::string& log_file_path) {
        log_file.open(log_file_path, std::ios::app);
        if (!log_file.is_open()) {
            std::cerr << "Failed to open log file: " << log_file_path << std::endl;
        }
        initialized = true;
        std::cout << "Logger initialized with fallback implementation" << std::endl;
    }

    void fallbackLog(LogLevel level, const std::string& msg, const std::source_location& loc) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "[%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms.count() << "] "
           << "[" << levelToString(level) << "] "
           << "[" << loc.file_name() << ":" << loc.line() << "] "
           << msg << std::endl;

        std::string log_entry = ss.str();

        if (console_enabled) {
            std::cout << log_entry;
        }

        if (log_file.is_open()) {
            log_file << log_entry;
        }
    }

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

#ifdef HAS_SPDLOG
    static spdlog::level::level_enum toSpdlogLevel(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return spdlog::level::trace;
            case LogLevel::DEBUG: return spdlog::level::debug;
            case LogLevel::INFO: return spdlog::level::info;
            case LogLevel::WARN: return spdlog::level::warn;
            case LogLevel::ERROR: return spdlog::level::err;
            case LogLevel::CRITICAL: return spdlog::level::critical;
            default: return spdlog::level::info;
        }
    }

    std::shared_ptr<spdlog::logger> spdlogger;
#endif

    LogLevel current_level;
    bool console_enabled;
    bool initialized;
    std::mutex mutex;
    std::ofstream log_file;
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

} // namespace utils
} // namespace bigbrother
