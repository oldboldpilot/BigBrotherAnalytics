/**
 * BigBrotherAnalytics - Configuration Module (C++23)
 *
 * YAML-based configuration management following C++ Core Guidelines:
 * - C.2: Private data with public interface
 * - C.41: Constructor establishes invariants
 * - F.6: noexcept where applicable
 * - F.20: Return values, not output parameters
 * - R.1: RAII for resource management
 * - I.10: Use exceptions for errors (in this case, std::optional for missing values)
 */

// Global module fragment
module;

#include <expected>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <unordered_map>
#include <vector>

// Module declaration
export module bigbrother.utils.config;

export namespace bigbrother::utils {

/**
 * Configuration Manager (Singleton)
 *
 * Following Core Guidelines:
 * - C.2: Encapsulation with private data
 * - C.21: Rule of five (delete copy, default move)
 * - F.4: constexpr for getters
 * - R.1: RAII for file handling
 *
 * Loads configuration from YAML files or environment variables.
 * Thread-safe read access after initialization.
 */
class Config {
  public:
    /**
     * Get singleton instance
     * F.1: Meaningfully named
     * F.6: noexcept - cannot throw
     */
    [[nodiscard]] static auto getInstance() noexcept -> Config&;

    // C.21: Non-copyable singleton
    Config(Config const&) = delete;
    auto operator=(Config const&) -> Config& = delete;

    /**
     * Load configuration from YAML file
     *
     * F.20: Return bool for success/failure
     * Following C++ Core Guidelines for error handling
     */
    [[nodiscard]] auto load(std::string const& config_file_path) -> bool;

    /**
     * Reload configuration from last loaded file
     */
    [[nodiscard]] auto reload() -> bool;

    /**
     * Get configuration value with type safety
     *
     * F.20: Return std::optional for possibly-missing values
     * F.16: Template for generic value retrieval
     *
     * Example:
     *   auto port = config.get<int>("server.port");
     *   if (port) { use(*port); }
     */
    template <typename T>
    [[nodiscard]] auto get(std::string const& key) const -> std::optional<T>;

    /**
     * Get configuration value with default
     *
     * F.16: Pass default by const& (might be expensive to copy)
     * F.20: Return by value
     */
    template <typename T>
    [[nodiscard]] auto get(std::string const& key, T const& default_value) const -> T;

    /**
     * Check if key exists
     * F.6: noexcept - cannot throw
     */
    [[nodiscard]] auto has(std::string const& key) const noexcept -> bool;

    /**
     * Get all keys in a section
     * F.20: Return vector by value (move semantics)
     */
    [[nodiscard]] auto keys(std::string const& section = "") const -> std::vector<std::string>;

    /**
     * Set configuration value
     *
     * F.18: Pass by forwarding reference for efficiency
     */
    template <typename T>
    auto set(std::string const& key, T&& value) -> void;

    /**
     * Save current configuration to file
     */
    [[nodiscard]] auto save(std::string const& config_file_path) const -> bool;

    /**
     * Clear all configuration
     * F.6: noexcept - cannot throw
     */
    auto clear() noexcept -> void;

    /**
     * Get configuration as string (for debugging)
     * F.20: Return by value
     */
    [[nodiscard]] auto toString() const -> std::string;

  private:
    // C.41: Private constructor for singleton
    Config();
    ~Config();

    // C.21: Allow move for implementation flexibility
    Config(Config&&) noexcept = default;
    auto operator=(Config&&) noexcept -> Config& = default;

    // R.1: RAII - Implementation handles resource management
    // NOTE: Using inline stub implementation (pImpl removed for now)
    std::unordered_map<std::string, std::string> config_map_;
};

// Inline stub implementations (temporary until proper YAML config implemented)
inline Config::Config() = default;
inline Config::~Config() = default;

inline auto Config::getInstance() noexcept -> Config& {
    static Config instance;
    return instance;
}

inline auto Config::load(std::string const& config_file_path) -> bool {
    // Stub - returns true (config loading via Python/YAML later)
    return true;
}

inline auto Config::reload() -> bool {
    return true;
}

template <typename T>
inline auto Config::get(std::string const& key, T const& default_value) const -> T {
    // Stub - always returns default value for now
    return default_value;
}

template <typename T>
inline auto Config::get(std::string const& key) const -> std::optional<T> {
    // Stub - always returns nullopt for now
    return std::nullopt;
}

inline auto Config::has(std::string const& key) const noexcept -> bool {
    return config_map_.find(key) != config_map_.end();
}

inline auto Config::keys(std::string const& section) const -> std::vector<std::string> {
    return {};
}

template <typename T>
inline auto Config::set(std::string const& key, T&& value) -> void {
    // Stub implementation
}

inline auto Config::save(std::string const& config_file_path) const -> bool {
    return true;
}

inline auto Config::clear() noexcept -> void {
    config_map_.clear();
}

inline auto Config::toString() const -> std::string {
    return "{}";
}

/**
 * Configuration namespace helpers
 *
 * Following F.1: Meaningful names for operations
 */

/**
 * Load configuration with validation
 *
 * F.20: Return Result<Config&> for error handling
 */
[[nodiscard]] inline auto
loadConfig(std::string const& path, std::source_location location = std::source_location::current())
    -> bool {
    auto& config = Config::getInstance();
    if (!config.load(path)) {
        // Log error with source location
        return false;
    }
    return true;
}

/**
 * Get required configuration value
 *
 * F.20: Return std::expected for required values
 * Throws if key missing (fail-fast for required config)
 */
template <typename T>
[[nodiscard]] inline auto getRequired(std::string const& key) -> T {
    auto& config = Config::getInstance();
    auto value = config.get<T>(key);

    if (!value) {
        throw std::runtime_error("Required configuration key missing: " + key);
    }

    return *value;
}

} // namespace bigbrother::utils
