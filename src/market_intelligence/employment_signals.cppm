/**
 * BigBrotherAnalytics - Employment Signals Module (C++23)
 *
 * Generates trading signals from BLS employment data and sector analysis.
 * Integrates with Trading Decision Engine for sector rotation strategies.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

// Global module fragment
module;

#include <array>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Module declaration
export module bigbrother.employment.signals;

// Import dependencies
import bigbrother.utils.types;

export namespace bigbrother::employment {

using namespace bigbrother::types;

/**
 * Employment Signal Types
 */
enum class EmploymentSignalType {
    JoblessClaimsSpike,  // Weekly claims >10% increase
    SectorLayoffs,       // Major layoffs in sector
    SectorHiring,        // Expansion in sector
    EmploymentImproving, // Sector employment trending up
    EmploymentDeclining, // Sector employment trending down
    RecessionWarning     // Multiple negative indicators
};

/**
 * Employment Signal
 *
 * Generated from BLS data and sector analysis
 */
struct EmploymentSignal {
    EmploymentSignalType type;
    int sector_code{0}; // GICS sector code
    std::string sector_name;
    double confidence{0.0};        // 0.0 to 1.0
    double employment_change{0.0}; // % change
    std::string rationale;
    Timestamp timestamp{0};

    // Trading implications
    bool bullish{false};         // Positive for sector
    bool bearish{false};         // Negative for sector
    double signal_strength{0.0}; // -1.0 (very bearish) to +1.0 (very bullish)

    [[nodiscard]] auto isActionable() const noexcept -> bool {
        return confidence > 0.60 && std::abs(signal_strength) > 0.50;
    }
};

/**
 * Sector Rotation Signal
 *
 * Indicates which sectors to overweight/underweight
 */
struct SectorRotationSignal {
    int sector_code{0};
    std::string sector_name;
    std::string sector_etf;

    // Signal components
    double employment_score{0.0}; // From BLS data
    double sentiment_score{0.0};  // From news (future)
    double technical_score{0.0};  // From price action (future)
    double composite_score{0.0};  // Weighted average

    // Recommendation
    enum class Action { Overweight, Neutral, Underweight };
    Action action{Action::Neutral};
    double target_allocation{0.0}; // % of portfolio

    [[nodiscard]] auto isStrongSignal() const noexcept -> bool {
        return std::abs(composite_score) > 0.70;
    }
};

/**
 * Employment Signal Generator
 *
 * Analyzes BLS data and generates trading signals
 */
class EmploymentSignalGenerator {
  public:
    EmploymentSignalGenerator(const std::string& scripts_path = "scripts",
                              const std::string& db_path = "data/bigbrother.duckdb")
        : scripts_path_(scripts_path), db_path_(db_path) {}

    /**
     * Generate employment signals for all sectors
     *
     * Calls Python backend to analyze DuckDB employment data
     * and generates signals for employment trends by sector.
     */
    [[nodiscard]] auto generateSignals() -> std::vector<EmploymentSignal> {
        std::vector<EmploymentSignal> signals;

        // Execute Python signal generator
        const std::string cmd = "uv run python " + scripts_path_ +
                                "/employment_signals.py generate_signals " + db_path_;
        const auto json_output = executeCommand(cmd);

        if (json_output.empty()) {
            return signals;
        }

        // Parse JSON output (simple parsing - assumes valid JSON array)
        signals = parseEmploymentSignalsJSON(json_output);

        return signals;
    }

    /**
     * Generate sector rotation signals
     *
     * Analyzes employment trends by sector and generates
     * Overweight/Underweight recommendations.
     */
    [[nodiscard]] auto generateRotationSignals() -> std::vector<SectorRotationSignal> {
        std::vector<SectorRotationSignal> signals;

        // Execute Python rotation signal generator
        const std::string cmd = "uv run python " + scripts_path_ +
                                "/employment_signals.py rotation_signals " + db_path_;
        const auto json_output = executeCommand(cmd);

        if (json_output.empty()) {
            return signals;
        }

        // Parse JSON output
        signals = parseRotationSignalsJSON(json_output);

        return signals;
    }

    /**
     * Check for jobless claims spike (recession warning)
     *
     * Queries jobless claims data and checks for >10% spike.
     */
    [[nodiscard]] auto checkJoblessClaimsSpike() -> std::optional<EmploymentSignal> {
        // Execute Python jobless claims checker
        const std::string cmd = "uv run python " + scripts_path_ +
                                "/employment_signals.py check_jobless_claims " + db_path_;
        const auto json_output = executeCommand(cmd);

        if (json_output.empty() || json_output.find("no_spike") != std::string::npos) {
            return std::nullopt;
        }

        // Parse single signal JSON
        auto signals = parseEmploymentSignalsJSON("[" + json_output + "]");
        if (!signals.empty()) {
            return signals[0];
        }

        return std::nullopt;
    }

  private:
    std::string scripts_path_;
    std::string db_path_;

    /**
     * Execute shell command and capture output
     */
    [[nodiscard]] auto executeCommand(const std::string& cmd) -> std::string {
        std::array<char, 128> buffer{};
        std::string result;

        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            return "";
        }

        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }

        return result;
    }

    /**
     * Parse JSON string to extract EmploymentSignal values
     *
     * Simple JSON parser - assumes well-formed JSON from Python script
     */
    [[nodiscard]] auto parseEmploymentSignalsJSON(const std::string& json)
        -> std::vector<EmploymentSignal> {
        std::vector<EmploymentSignal> signals;

        // Simple JSON parsing - look for signal objects between braces
        size_t pos = 0;
        while ((pos = json.find("{", pos)) != std::string::npos) {
            size_t end_pos = json.find("}", pos);
            if (end_pos == std::string::npos)
                break;

            std::string obj = json.substr(pos, end_pos - pos + 1);

            EmploymentSignal signal;
            signal.type = extractSignalType(obj);
            signal.sector_code = extractInt(obj, "sector_code");
            signal.sector_name = extractString(obj, "sector_name");
            signal.confidence = extractDouble(obj, "confidence");
            signal.employment_change = extractDouble(obj, "employment_change");
            signal.rationale = extractString(obj, "rationale");
            signal.timestamp = static_cast<Timestamp>(extractInt(obj, "timestamp"));
            signal.bullish = extractBool(obj, "bullish");
            signal.bearish = extractBool(obj, "bearish");
            signal.signal_strength = extractDouble(obj, "signal_strength");

            signals.push_back(signal);
            pos = end_pos + 1;
        }

        return signals;
    }

    /**
     * Parse JSON string to extract SectorRotationSignal values
     */
    [[nodiscard]] auto parseRotationSignalsJSON(const std::string& json)
        -> std::vector<SectorRotationSignal> {
        std::vector<SectorRotationSignal> signals;

        // Simple JSON parsing - look for signal objects between braces
        size_t pos = 0;
        while ((pos = json.find("{", pos)) != std::string::npos) {
            size_t end_pos = json.find("}", pos);
            if (end_pos == std::string::npos)
                break;

            std::string obj = json.substr(pos, end_pos - pos + 1);

            SectorRotationSignal signal;
            signal.sector_code = extractInt(obj, "sector_code");
            signal.sector_name = extractString(obj, "sector_name");
            signal.sector_etf = extractString(obj, "sector_etf");
            signal.employment_score = extractDouble(obj, "employment_score");
            signal.sentiment_score = extractDouble(obj, "sentiment_score");
            signal.technical_score = extractDouble(obj, "technical_score");
            signal.composite_score = extractDouble(obj, "composite_score");

            const auto action_str = extractString(obj, "action");
            if (action_str == "Overweight") {
                signal.action = SectorRotationSignal::Action::Overweight;
            } else if (action_str == "Underweight") {
                signal.action = SectorRotationSignal::Action::Underweight;
            } else {
                signal.action = SectorRotationSignal::Action::Neutral;
            }

            signal.target_allocation = extractDouble(obj, "target_allocation");

            signals.push_back(signal);
            pos = end_pos + 1;
        }

        return signals;
    }

    // JSON parsing helper functions

    [[nodiscard]] auto extractString(const std::string& json, const std::string& key)
        -> std::string {
        const std::string search = "\"" + key + "\": \"";
        const size_t start = json.find(search);
        if (start == std::string::npos)
            return "";

        const size_t value_start = start + search.length();
        const size_t value_end = json.find("\"", value_start);
        if (value_end == std::string::npos)
            return "";

        return json.substr(value_start, value_end - value_start);
    }

    [[nodiscard]] auto extractInt(const std::string& json, const std::string& key) -> int {
        const std::string search = "\"" + key + "\": ";
        const size_t start = json.find(search);
        if (start == std::string::npos)
            return 0;

        const size_t value_start = start + search.length();
        const size_t value_end = json.find_first_of(",}", value_start);
        if (value_end == std::string::npos)
            return 0;

        const std::string value = json.substr(value_start, value_end - value_start);
        return std::atoi(value.c_str());
    }

    [[nodiscard]] auto extractDouble(const std::string& json, const std::string& key) -> double {
        const std::string search = "\"" + key + "\": ";
        const size_t start = json.find(search);
        if (start == std::string::npos)
            return 0.0;

        const size_t value_start = start + search.length();
        const size_t value_end = json.find_first_of(",}", value_start);
        if (value_end == std::string::npos)
            return 0.0;

        const std::string value = json.substr(value_start, value_end - value_start);
        return std::atof(value.c_str());
    }

    [[nodiscard]] auto extractBool(const std::string& json, const std::string& key) -> bool {
        const std::string search = "\"" + key + "\": ";
        const size_t start = json.find(search);
        if (start == std::string::npos)
            return false;

        const size_t value_start = start + search.length();
        return json.substr(value_start, 4) == "true";
    }

    [[nodiscard]] auto extractSignalType(const std::string& json) -> EmploymentSignalType {
        const auto type_str = extractString(json, "type");

        if (type_str == "JoblessClaimsSpike")
            return EmploymentSignalType::JoblessClaimsSpike;
        if (type_str == "SectorLayoffs")
            return EmploymentSignalType::SectorLayoffs;
        if (type_str == "SectorHiring")
            return EmploymentSignalType::SectorHiring;
        if (type_str == "EmploymentImproving")
            return EmploymentSignalType::EmploymentImproving;
        if (type_str == "EmploymentDeclining")
            return EmploymentSignalType::EmploymentDeclining;
        if (type_str == "RecessionWarning")
            return EmploymentSignalType::RecessionWarning;

        return EmploymentSignalType::EmploymentImproving; // Default
    }
};

} // namespace bigbrother::employment
