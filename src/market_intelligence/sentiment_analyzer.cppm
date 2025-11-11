/**
 * BigBrotherAnalytics - Sentiment Analyzer Module (C++23)
 *
 * Keyword-based sentiment analysis for financial news.
 * Fast, lightweight sentiment scoring without ML dependencies.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 * Phase 5+: News Ingestion System
 *
 * Following C++ Core Guidelines:
 * - I.2: Avoid non-const global variables
 * - F.51: Prefer default arguments over overloading
 * - Trailing return type syntax
 */

// Global module fragment
module;

#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <cmath>

// Module declaration
export module bigbrother.market_intelligence.sentiment;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::market_intelligence {

using namespace bigbrother::types;
using namespace bigbrother::utils;

// ============================================================================
// Sentiment Analysis Data Types
// ============================================================================

/**
 * Sentiment analysis result
 * C.1: Use struct for passive data
 */
struct SentimentResult {
    double score{0.0};              // -1.0 to 1.0
    std::string label;              // "positive", "negative", "neutral"
    std::vector<std::string> positive_keywords;
    std::vector<std::string> negative_keywords;
    double positive_score{0.0};
    double negative_score{0.0};
    size_t total_words{0};
    double keyword_density{0.0};
};

// ============================================================================
// Sentiment Analyzer
// ============================================================================

/**
 * Keyword-based sentiment analyzer for financial news
 *
 * Fast sentiment analysis using predefined keyword lists.
 * No ML dependencies required.
 *
 * C.2: Use class when invariants exist
 */
class SentimentAnalyzer {
public:
    /**
     * Constructor
     */
    SentimentAnalyzer() {
        initializeKeywords();
        Logger::getInstance().info("Sentiment analyzer initialized");
        Logger::getInstance().info("  Positive keywords: {}", positive_keywords_.size());
        Logger::getInstance().info("  Negative keywords: {}", negative_keywords_.size());
    }

    /**
     * Analyze sentiment of text
     *
     * @param text Text to analyze (title + description)
     * @return Sentiment analysis result
     *
     * F.20: Return by value
     */
    [[nodiscard]] auto analyze(std::string const& text) const -> SentimentResult {
        SentimentResult result;

        if (text.empty()) {
            result.label = "neutral";
            return result;
        }

        // Tokenize text
        auto words = tokenize(text);
        result.total_words = words.size();

        if (words.empty()) {
            result.label = "neutral";
            return result;
        }

        // Analyze keywords
        double positive_score = 0.0;
        double negative_score = 0.0;

        for (size_t i = 0; i < words.size(); ++i) {
            auto const& word = words[i];

            // Check for intensifier
            double intensifier = 1.0;
            if (i > 0 && intensifiers_.contains(words[i-1])) {
                intensifier = 1.5;
            }

            // Check for negation (within 3 words before)
            bool negated = false;
            for (size_t j = (i >= 3 ? i - 3 : 0); j < i; ++j) {
                if (negations_.contains(words[j])) {
                    negated = true;
                    break;
                }
            }

            // Check positive keywords
            if (positive_keywords_.contains(word)) {
                if (negated) {
                    negative_score += 1.0 * intensifier;
                    result.negative_keywords.push_back(word);
                } else {
                    positive_score += 1.0 * intensifier;
                    result.positive_keywords.push_back(word);
                }
            }
            // Check negative keywords
            else if (negative_keywords_.contains(word)) {
                if (negated) {
                    positive_score += 1.0 * intensifier;
                    result.positive_keywords.push_back(word);
                } else {
                    negative_score += 1.0 * intensifier;
                    result.negative_keywords.push_back(word);
                }
            }
        }

        // Calculate final score
        result.positive_score = positive_score;
        result.negative_score = negative_score;

        double total_keywords = positive_score + negative_score;
        if (total_keywords == 0.0) {
            result.score = 0.0;
        } else {
            result.score = (positive_score - negative_score) / total_keywords;
        }

        // Clamp to [-1, 1]
        result.score = std::clamp(result.score, -1.0, 1.0);

        // Calculate keyword density
        result.keyword_density = total_keywords / static_cast<double>(result.total_words);

        // Determine label (thresholds: ±0.1)
        if (result.score > 0.1) {
            result.label = "positive";
        } else if (result.score < -0.1) {
            result.label = "negative";
        } else {
            result.label = "neutral";
        }

        return result;
    }

    /**
     * Analyze batch of texts
     *
     * @param texts Vector of texts to analyze
     * @return Vector of sentiment results
     */
    [[nodiscard]] auto analyzeBatch(std::vector<std::string> const& texts) const -> std::vector<SentimentResult> {
        std::vector<SentimentResult> results;
        results.reserve(texts.size());

        for (auto const& text : texts) {
            results.push_back(analyze(text));
        }

        return results;
    }

private:
    /**
     * Initialize keyword sets
     */
    auto initializeKeywords() -> void {
        // Positive financial keywords
        positive_keywords_ = {
            "profit", "profits", "profitable", "gain", "gains", "growth", "grow", "growing",
            "surge", "surged", "surges", "surging", "bull", "bullish", "rally", "rallied",
            "upgrade", "upgraded", "upgrades", "beat", "beats", "beating", "exceed", "exceeded",
            "exceeds", "outperform", "outperformed", "outperforming", "strong", "stronger",
            "strength", "success", "successful", "positive", "optimistic", "optimism",
            "improve", "improved", "improving", "improvement", "rise", "rises", "rising",
            "rose", "increase", "increased", "increasing", "up", "higher", "high", "record",
            "advance", "advanced", "advancing", "expansion", "expand", "expanding", "boom",
            "breakthrough", "win", "wins", "winning", "won", "leader", "leading", "innovation",
            "innovative", "opportunity", "opportunities", "recovery", "recover", "recovering",
        };

        // Negative financial keywords
        negative_keywords_ = {
            "loss", "losses", "lose", "losing", "lost", "decline", "declined", "declining",
            "fall", "falls", "falling", "fell", "drop", "dropped", "dropping", "drops",
            "bear", "bearish", "downgrade", "downgrades", "downgraded", "miss", "missed",
            "misses", "missing", "underperform", "underperformed", "underperforming",
            "weak", "weaker", "weakness", "failure", "fail", "failed", "failing", "fails",
            "negative", "pessimistic", "pessimism", "worsen", "worsened", "worsening",
            "worse", "decrease", "decreased", "decreasing", "down", "lower", "low",
            "plunge", "plunged", "plunging", "crash", "crashed", "crashing", "slump",
            "slumped", "slumping", "risk", "risks", "risky", "concern", "concerns",
            "concerned", "concerning", "warning", "warnings", "warn", "warned", "trouble",
            "troubled", "crisis", "recession", "bankruptcy", "bankrupt", "deficit",
        };

        // Intensifiers
        intensifiers_ = {
            "very", "extremely", "highly", "significantly", "substantially",
            "dramatically", "sharply", "rapidly", "strongly", "massively",
        };

        // Negations
        negations_ = {
            "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
            "hardly", "scarcely", "barely",
        };
    }

    /**
     * Tokenize text into lowercase words
     *
     * @param text Text to tokenize
     * @return Vector of lowercase words
     */
    [[nodiscard]] auto tokenize(std::string const& text) const -> std::vector<std::string> {
        std::vector<std::string> words;
        std::string word;

        for (char c : text) {
            if (std::isalnum(c) || c == '\'') {
                word += std::tolower(c);
            } else if (!word.empty()) {
                if (word.length() >= 2) {  // Filter very short words
                    words.push_back(word);
                }
                word.clear();
            }
        }

        // Don't forget last word
        if (!word.empty() && word.length() >= 2) {
            words.push_back(word);
        }

        return words;
    }

    std::set<std::string> positive_keywords_;
    std::set<std::string> negative_keywords_;
    std::set<std::string> intensifiers_;
    std::set<std::string> negations_;
};

} // export namespace bigbrother::market_intelligence
