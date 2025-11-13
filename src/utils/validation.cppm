/**
 * BigBrother Analytics - Validation Module
 *
 * C++23 module for input validation utilities.
 * Provides validation functions for stock symbols and other trading data.
 */

// Global module fragment
module;

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

// Module declaration
export module bigbrother.utils.validation;

export namespace bigbrother::utils {

/**
 * Validates if a string is a valid stock symbol
 *
 * A valid stock symbol must:
 * - Not be empty
 * - Be at most 10 characters long (most exchanges limit symbols to 5-6 chars)
 * - Contain only uppercase letters, digits, dots, or hyphens
 * - Not look like JSON field names (e.g., "percentChange", "bidPrice", etc.)
 *
 * @param symbol The symbol string to validate
 * @return true if the symbol is valid, false otherwise
 *
 * Examples:
 *   isValidStockSymbol("SPY")           -> true
 *   isValidStockSymbol("AAPL")          -> true
 *   isValidStockSymbol("BRK.B")         -> true
 *   isValidStockSymbol("percentChange") -> false (JSON field name)
 *   isValidStockSymbol("bidPrice")      -> false (JSON field name)
 *   isValidStockSymbol("")              -> false (empty)
 *   isValidStockSymbol("TOOLONGSYMBOL") -> false (too long)
 */
[[nodiscard]] inline constexpr auto isValidStockSymbol(std::string_view symbol) -> bool {
    // Empty or too long
    if (symbol.empty() || symbol.length() > 10) {
        return false;
    }

    // Check if it looks like a JSON field name (contains lowercase after first char)
    // Stock symbols are typically all uppercase
    bool has_lowercase = false;
    for (size_t i = 1; i < symbol.length(); ++i) {
        if (std::islower(static_cast<unsigned char>(symbol[i]))) {
            has_lowercase = true;
            break;
        }
    }

    // Reject common JSON field patterns
    if (has_lowercase) {
        // Common Schwab API field names that should NEVER be symbols
        if (symbol.find("Change") != std::string_view::npos ||
            symbol.find("Price") != std::string_view::npos ||
            symbol.find("Volume") != std::string_view::npos ||
            symbol.find("Time") != std::string_view::npos ||
            symbol.find("percent") != std::string_view::npos ||
            symbol.find("mark") != std::string_view::npos ||
            symbol.find("net") != std::string_view::npos) {
            return false;
        }
    }

    // Stock symbols should contain only uppercase letters, digits, dots, or hyphens
    // Allow lowercase only for special index symbols or specific tickers
    for (char c : symbol) {
        if (!std::isupper(static_cast<unsigned char>(c)) &&
            !std::isdigit(static_cast<unsigned char>(c)) && c != '.' && c != '-' && c != '^' &&
            c != '=' &&
            !std::islower(static_cast<unsigned char>(c))) { // Some brokers use lowercase
            return false;
        }
    }

    return true;
}

/**
 * Validates if a symbol looks like a JSON field name (false positive check)
 *
 * @param symbol The symbol string to check
 * @return true if it looks like a JSON field, false if it looks like a stock symbol
 */
[[nodiscard]] inline constexpr auto looksLikeJsonField(std::string_view symbol) -> bool {
    if (symbol.empty()) {
        return true; // Empty strings are definitely not stock symbols
    }

    // Check for camelCase pattern (lowercase followed by uppercase)
    for (size_t i = 0; i < symbol.length() - 1; ++i) {
        if (std::islower(static_cast<unsigned char>(symbol[i])) &&
            std::isupper(static_cast<unsigned char>(symbol[i + 1]))) {
            return true; // camelCase pattern detected
        }
    }

    // Check for common API field suffixes
    return symbol.ends_with("Change") || symbol.ends_with("Price") || symbol.ends_with("Volume") ||
           symbol.ends_with("Time") || symbol.ends_with("Rate") || symbol.ends_with("Percent");
}

} // namespace bigbrother::utils
