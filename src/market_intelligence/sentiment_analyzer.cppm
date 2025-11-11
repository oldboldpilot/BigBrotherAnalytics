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

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

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
    double score{0.0}; // -1.0 to 1.0
    std::string label; // "positive", "negative", "neutral"
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
            if (i > 0 && intensifiers_.contains(words[i - 1])) {
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
    [[nodiscard]] auto analyzeBatch(std::vector<std::string> const& texts) const
        -> std::vector<SentimentResult> {
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
        // Positive financial keywords (150+ total with negative)
        positive_keywords_ = {
            // Core positive terms
            "profit",
            "profits",
            "profitable",
            "gain",
            "gains",
            "growth",
            "grow",
            "growing",
            "surge",
            "surged",
            "surges",
            "surging",
            "bull",
            "bullish",
            "rally",
            "rallied",
            "upgrade",
            "upgraded",
            "upgrades",
            "beat",
            "beats",
            "beating",
            "exceed",
            "exceeded",
            "exceeds",
            "outperform",
            "outperformed",
            "outperforming",
            "strong",
            "stronger",
            "strength",
            "success",
            "successful",
            "positive",
            "optimistic",
            "optimism",
            "improve",
            "improved",
            "improving",
            "improvement",
            "rise",
            "rises",
            "rising",
            "rose",
            "increase",
            "increased",
            "increasing",
            "up",
            "higher",
            "high",
            "record",
            "advance",
            "advanced",
            "advancing",
            "expansion",
            "expand",
            "expanding",
            "boom",
            "breakthrough",
            "win",
            "wins",
            "winning",
            "won",
            "leader",
            "leading",
            "innovation",
            "innovative",
            "opportunity",
            "opportunities",
            "recovery",
            "recover",
            "recovering",

            // Financial performance terms
            "revenue",
            "revenues",
            "earnings",
            "margin",
            "margins",
            "dividend",
            "dividends",
            "buyback",
            "buybacks",
            "acquisition",
            "acquisitions",
            "merge",
            "merger",
            "synergy",
            "synergies",
            "accretive",
            "cashflow",
            "ebitda",
            "guidance",
            "reaffirm",
            "reaffirmed",
            "raise",
            "raised",
            "raising",
            "boost",
            "boosted",
            "boosting",
            "robust",
            "solid",
            "healthy",
            "impressive",
            "stellar",
            "outstanding",
            "excellent",
            "exceptional",
            "resilient",
            "resilience",
            "capitalize",
            "capitalizing",
            "monetize",
            "monetizing",

            // Market sentiment terms
            "breakout",
            "uptrend",
            "upturn",
            "momentum",
            "support",
            "accumulation",
            "accumulate",
            "soar",
            "soared",
            "soaring",
            "skyrocket",
            "skyrocketed",
            "climb",
            "climbed",
            "climbing",
            "spike",
            "spiked",
            "spiking",
            "jump",
            "jumped",
            "jumping",
            "leap",
            "leaped",
            "leaping",
            "thrive",
            "thriving",
            "thrived",
            "flourish",
            "flourishing",
            "prosper",
            "prospering",
            "prosperity",
            "confidence",
            "confident",
            "promising",
            "favorable",
            "attractive",

            // Company performance terms
            "overweight",
            "buy",
            "accelerate",
            "accelerated",
            "accelerating",
            "scale",
            "scaling",
            "penetrate",
            "penetration",
            "diversify",
            "diversification",
            "streamline",
            "streamlined",
            "efficient",
            "efficiency",
            "productive",
            "productivity",
            "competitive",
            "competitiveness",
            "advantage",
            "advantages",
            "edge",
            "dominate",
            "dominated",
            "dominating",
            "leadership",
            "strategic",
            "synergistic",
            "transformative",
            "pioneering",
            "disruptive",
            "revolutionary",

            // Economic indicators
            "uptick",
            "expansion",
            "expansionary",
            "stimulus",
            "tailwind",
            "tailwinds",
            "rebound",
            "rebounding",
            "revive",
            "revival",
            "upside",
            "appreciate",
            "appreciation",
            "strengthen",
            "strengthening",
            "optimized",
            "optimize",
            "maximized",
            "maximize",
        };

        // Negative financial keywords (150+ total with positive)
        negative_keywords_ = {
            // Core negative terms
            "loss",
            "losses",
            "lose",
            "losing",
            "lost",
            "decline",
            "declined",
            "declining",
            "fall",
            "falls",
            "falling",
            "fell",
            "drop",
            "dropped",
            "dropping",
            "drops",
            "bear",
            "bearish",
            "downgrade",
            "downgrades",
            "downgraded",
            "miss",
            "missed",
            "misses",
            "missing",
            "underperform",
            "underperformed",
            "underperforming",
            "weak",
            "weaker",
            "weakness",
            "failure",
            "fail",
            "failed",
            "failing",
            "fails",
            "negative",
            "pessimistic",
            "pessimism",
            "worsen",
            "worsened",
            "worsening",
            "worse",
            "decrease",
            "decreased",
            "decreasing",
            "down",
            "lower",
            "low",
            "plunge",
            "plunged",
            "plunging",
            "crash",
            "crashed",
            "crashing",
            "slump",
            "slumped",
            "slumping",
            "risk",
            "risks",
            "risky",
            "concern",
            "concerns",
            "concerned",
            "concerning",
            "warning",
            "warnings",
            "warn",
            "warned",
            "trouble",
            "troubled",
            "crisis",
            "recession",
            "bankruptcy",
            "bankrupt",
            "deficit",

            // Financial performance terms
            "shortfall",
            "shortfalls",
            "writedown",
            "writedowns",
            "impairment",
            "impairments",
            "charge",
            "charges",
            "restructure",
            "restructuring",
            "layoff",
            "layoffs",
            "cut",
            "cuts",
            "cutting",
            "eliminate",
            "eliminated",
            "eliminating",
            "reduction",
            "reductions",
            "reduce",
            "reduced",
            "reducing",
            "dilute",
            "diluted",
            "dilution",
            "erode",
            "eroded",
            "eroding",
            "erosion",
            "shrink",
            "shrinking",
            "contraction",
            "disappointing",
            "disappoints",
            "disappointed",
            "lowered",
            "lowering",
            "slash",
            "slashed",
            "slashing",
            "drag",
            "dragged",
            "dragging",

            // Market sentiment terms
            "breakdown",
            "downtrend",
            "downturn",
            "resistance",
            "selloff",
            "sell-off",
            "dump",
            "dumped",
            "dumping",
            "tumble",
            "tumbled",
            "tumbling",
            "sink",
            "sinking",
            "sank",
            "plummet",
            "plummeted",
            "plummeting",
            "crater",
            "cratered",
            "cratering",
            "collapse",
            "collapsed",
            "collapsing",
            "vulnerable",
            "vulnerability",
            "volatile",
            "volatility",
            "unstable",
            "instability",
            "oversold",
            "overbought",
            "distribution",
            "capitulation",
            "panic",
            "fear",
            "uncertainty",

            // Company performance terms
            "underweight",
            "sell",
            "avoid",
            "decelerate",
            "decelerated",
            "decelerating",
            "struggle",
            "struggled",
            "struggling",
            "deteriorate",
            "deteriorated",
            "deteriorating",
            "deterioration",
            "impair",
            "impaired",
            "impairing",
            "obsolete",
            "obsolescence",
            "stagnant",
            "stagnation",
            "stagnate",
            "stagnating",
            "uncompetitive",
            "disadvantage",
            "disadvantages",
            "challenged",
            "challenges",
            "challenging",
            "headwind",
            "headwinds",
            "obstacle",
            "obstacles",
            "friction",

            // Economic indicators
            "downtick",
            "contraction",
            "contractionary",
            "recessionary",
            "slowdown",
            "slowing",
            "inflation",
            "inflationary",
            "stagflation",
            "deflation",
            "deflationary",
            "overhang",
            "debt",
            "debts",
            "leverage",
            "leveraged",
            "overleveraged",
            "insolvent",
            "insolvency",
            "default",
            "defaulted",
            "defaulting",
            "distress",
            "distressed",
            "stressed",
            "fragile",
            "fragility",
            "contagion",
            "exposure",
            "exposed",
            "downside",
            "depreciate",
            "depreciation",
            "weaken",
            "weakening",
            "undermine",
            "undermined",
            "undermining",
            "disruptive",
            "disruption",
        };

        // Intensifiers
        intensifiers_ = {
            "very",         "extremely", "highly",  "significantly", "substantially",
            "dramatically", "sharply",   "rapidly", "strongly",      "massively",
        };

        // Negations
        negations_ = {
            "not",     "no",      "never",  "neither",  "nobody",
            "nothing", "nowhere", "hardly", "scarcely", "barely",
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
                if (word.length() >= 2) { // Filter very short words
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

} // namespace bigbrother::market_intelligence
