#pragma once

#include <string>
#include <memory>
#include <optional>
#include <vector>
#include <map>

namespace bigbrother {
namespace utils {

/**
 * Configuration Manager
 *
 * Loads and manages application configuration from YAML files.
 * Supports nested configuration, environment variable substitution,
 * and type-safe access to configuration values.
 *
 * Example YAML:
 *   database:
 *     path: "data/bigbrother.duckdb"
 *     readonly: false
 *   schwab:
 *     client_id: "${SCHWAB_CLIENT_ID}"
 *     redirect_uri: "https://localhost:8080/callback"
 *   trading:
 *     max_daily_loss: 900.0
 *     max_position_size: 1500.0
 */
class Config {
public:
    Config();
    ~Config();

    // Load configuration from file
    bool load(const std::string& config_file_path);

    // Reload configuration
    bool reload();

    // Get configuration values with type safety
    template<typename T>
    std::optional<T> get(const std::string& key) const;

    // Get configuration value with default
    template<typename T>
    T get(const std::string& key, const T& default_value) const;

    // Check if key exists
    bool has(const std::string& key) const;

    // Get all keys in a section
    std::vector<std::string> keys(const std::string& section = "") const;

    // Set configuration value (for runtime updates)
    template<typename T>
    void set(const std::string& key, const T& value);

    // Save current configuration to file
    bool save(const std::string& config_file_path) const;

    // Clear all configuration
    void clear();

    // Singleton access
    static Config& getInstance();

    // Delete copy and move constructors
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&) = delete;
    Config& operator=(Config&&) = delete;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Specializations for common types
template<>
std::optional<std::string> Config::get<std::string>(const std::string& key) const;

template<>
std::optional<int> Config::get<int>(const std::string& key) const;

template<>
std::optional<long> Config::get<long>(const std::string& key) const;

template<>
std::optional<double> Config::get<double>(const std::string& key) const;

template<>
std::optional<bool> Config::get<bool>(const std::string& key) const;

template<>
std::optional<std::vector<std::string>> Config::get<std::vector<std::string>>(const std::string& key) const;

} // namespace utils
} // namespace bigbrother
