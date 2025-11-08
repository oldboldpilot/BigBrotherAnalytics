/**
 * Config Module Implementation
 * C++23 module implementation unit
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-08
 */

// Global module fragment - standard library headers MUST go here
module;

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <regex>
#include <unordered_map>

#ifdef HAS_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

// Module implementation unit declaration
module bigbrother.utils.config;

import bigbrother.utils.logger;

namespace bigbrother {
namespace utils {

class Config::Impl {
public:
    Impl() : loaded(false) {}

    bool load(const std::string& config_file_path) {
        current_file = config_file_path;

#ifdef HAS_YAML_CPP
        try {
            YAML::Node config = YAML::LoadFile(config_file_path);
            root = config;
            flattenNode(config, "");
            expandEnvironmentVariables();
            loaded = true;
            LOG_INFO("Configuration loaded from: {}", config_file_path);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load config from {}: {}", config_file_path, e.what());
            return false;
        }
#else
        // Fallback: simple key=value parser
        return loadSimple(config_file_path);
#endif
    }

    bool reload() {
        if (current_file.empty()) {
            LOG_ERROR("No configuration file to reload");
            return false;
        }
        values.clear();
        return load(current_file);
    }

    template<typename T>
    std::optional<T> get(const std::string& key) const {
        auto it = values.find(key);
        if (it != values.end()) {
            try {
                return convertTo<T>(it->second);
            } catch (...) {
                LOG_WARN("Failed to convert config value for key: {}", key);
            }
        }
        return std::nullopt;
    }

    template<typename T>
    T get(const std::string& key, const T& default_value) const {
        return get<T>(key).value_or(default_value);
    }

    bool has(const std::string& key) const {
        return values.find(key) != values.end();
    }

    std::vector<std::string> keys(const std::string& section) const {
        std::vector<std::string> result;
        for (const auto& [key, value] : values) {
            if (section.empty() || key.starts_with(section + ".")) {
                result.push_back(key);
            }
        }
        return result;
    }

    template<typename T>
    void set(const std::string& key, const T& value) {
        std::ostringstream oss;
        oss << value;
        values[key] = oss.str();
    }

    bool save(const std::string& config_file_path) const {
        // Simple key=value format for now
        std::ofstream file(config_file_path);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open config file for writing: {}", config_file_path);
            return false;
        }

        for (const auto& [key, value] : values) {
            file << key << "=" << value << std::endl;
        }

        LOG_INFO("Configuration saved to: {}", config_file_path);
        return true;
    }

    void clear() {
        values.clear();
        loaded = false;
    }

private:
#ifdef HAS_YAML_CPP
    void flattenNode(const YAML::Node& node, const std::string& prefix) {
        if (node.IsScalar()) {
            values[prefix] = node.as<std::string>();
        } else if (node.IsSequence()) {
            for (size_t i = 0; i < node.size(); ++i) {
                flattenNode(node[i], prefix + "[" + std::to_string(i) + "]");
            }
        } else if (node.IsMap()) {
            for (const auto& it : node) {
                std::string key = it.first.as<std::string>();
                std::string fullKey = prefix.empty() ? key : prefix + "." + key;
                flattenNode(it.second, fullKey);
            }
        }
    }

    YAML::Node root;
#endif

    bool loadSimple(const std::string& config_file_path) {
        std::ifstream file(config_file_path);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open config file: {}", config_file_path);
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // Parse key=value
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                values[key] = value;
            }
        }

        expandEnvironmentVariables();
        loaded = true;
        LOG_INFO("Configuration loaded from: {} (simple format)", config_file_path);
        return true;
    }

    void expandEnvironmentVariables() {
        std::regex env_var_pattern(R"(\$\{([A-Za-z0-9_]+)\})");

        for (auto& [key, value] : values) {
            std::smatch match;
            std::string result = value;

            while (std::regex_search(result, match, env_var_pattern)) {
                std::string var_name = match[1].str();
                const char* env_value = std::getenv(var_name.c_str());

                if (env_value) {
                    result.replace(match.position(), match.length(), env_value);
                } else {
                    LOG_WARN("Environment variable not found: {}", var_name);
                    // Leave as-is if not found
                    break;
                }
            }

            value = result;
        }
    }

    template<typename T>
    T convertTo(const std::string& value) const {
        std::istringstream iss(value);
        T result;
        iss >> result;
        return result;
    }

    std::unordered_map<std::string, std::string> values;
    std::string current_file;
    bool loaded;
};

// Config implementation
Config::Config() : pImpl(std::make_unique<Impl>()) {}

Config::~Config() = default;

bool Config::load(const std::string& config_file_path) {
    return pImpl->load(config_file_path);
}

bool Config::reload() {
    return pImpl->reload();
}

bool Config::has(const std::string& key) const {
    return pImpl->has(key);
}

std::vector<std::string> Config::keys(const std::string& section) const {
    return pImpl->keys(section);
}

bool Config::save(const std::string& config_file_path) const {
    return pImpl->save(config_file_path);
}

void Config::clear() {
    pImpl->clear();
}

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

// Template specializations
template<>
std::optional<std::string> Config::get<std::string>(const std::string& key) const {
    return pImpl->get<std::string>(key);
}

template<>
std::optional<int> Config::get<int>(const std::string& key) const {
    return pImpl->get<int>(key);
}

template<>
std::optional<long> Config::get<long>(const std::string& key) const {
    return pImpl->get<long>(key);
}

template<>
std::optional<double> Config::get<double>(const std::string& key) const {
    return pImpl->get<double>(key);
}

template<>
std::optional<bool> Config::get<bool>(const std::string& key) const {
    auto value = pImpl->get<std::string>(key);
    if (!value) {
        return std::nullopt;
    }

    std::string v = *value;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);

    if (v == "true" || v == "1" || v == "yes" || v == "on") {
        return true;
    } else if (v == "false" || v == "0" || v == "no" || v == "off") {
        return false;
    }

    return std::nullopt;
}

template<>
std::optional<std::vector<std::string>> Config::get<std::vector<std::string>>(const std::string& key) const {
    auto value = pImpl->get<std::string>(key);
    if (!value) {
        return std::nullopt;
    }

    std::vector<std::string> result;
    std::istringstream iss(*value);
    std::string item;

    // Split by comma
    while (std::getline(iss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            result.push_back(item);
        }
    }

    return result;
}

template<>
std::string Config::get<std::string>(const std::string& key, const std::string& default_value) const {
    return pImpl->get<std::string>(key, default_value);
}

template<>
int Config::get<int>(const std::string& key, const int& default_value) const {
    return pImpl->get<int>(key, default_value);
}

template<>
long Config::get<long>(const std::string& key, const long& default_value) const {
    return pImpl->get<long>(key, default_value);
}

template<>
double Config::get<double>(const std::string& key, const double& default_value) const {
    return pImpl->get<double>(key, default_value);
}

template<>
bool Config::get<bool>(const std::string& key, const bool& default_value) const {
    return pImpl->get<bool>(key, default_value);
}

template<>
void Config::set<std::string>(const std::string& key, const std::string& value) {
    pImpl->set<std::string>(key, value);
}

template<>
void Config::set<int>(const std::string& key, const int& value) {
    pImpl->set<int>(key, value);
}

template<>
void Config::set<double>(const std::string& key, const double& value) {
    pImpl->set<double>(key, value);
}

template<>
void Config::set<bool>(const std::string& key, const bool& value) {
    pImpl->set<std::string>(key, value ? "true" : "false");
}

} // namespace utils
} // namespace bigbrother
