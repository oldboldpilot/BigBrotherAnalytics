/**
 * BigBrotherAnalytics - Sentiment Analysis Unit Tests
 *
 * Unit tests for the C++23 sentiment analyzer module.
 * Tests keyword detection, scoring, and edge cases.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 */

#include <catch2/catch_all.hpp>
#include <string>
#include <vector>

// Import C++23 modules
import bigbrother.market_intelligence.sentiment;

using namespace bigbrother::market_intelligence;

// ============================================================================
// Basic Sentiment Detection Tests
// ============================================================================

TEST_CASE("Sentiment analyzer detects positive keywords", "[sentiment][positive]") {
    SentimentAnalyzer analyzer;

    SECTION("Strong positive headline") {
        auto result = analyzer.analyze("Apple stock surges on strong earnings beat");

        REQUIRE(result.score > 0.0);
        REQUIRE(result.label == "positive");
        REQUIRE(result.positive_keywords.size() >= 2); // "surges", "strong", "beat"
    }

    SECTION("Moderate positive headline") {
        auto result = analyzer.analyze("Company announces revenue growth for quarter");

        REQUIRE(result.score > 0.0);
        REQUIRE(result.label == "positive");
        REQUIRE(result.positive_keywords.size() >= 1);
    }

    SECTION("Multiple positive keywords") {
        auto result = analyzer.analyze("Record profits, strong guidance, accelerating growth");

        REQUIRE(result.score > 0.5); // Multiple keywords should boost score
        REQUIRE(result.label == "positive");
        REQUIRE(result.positive_keywords.size() >= 3);
    }
}

TEST_CASE("Sentiment analyzer detects negative keywords", "[sentiment][negative]") {
    SentimentAnalyzer analyzer;

    SECTION("Strong negative headline") {
        auto result = analyzer.analyze("Stock plunges on weak earnings miss");

        REQUIRE(result.score < 0.0);
        REQUIRE(result.label == "negative");
        REQUIRE(result.negative_keywords.size() >= 2); // "plunges", "weak", "miss"
    }

    SECTION("Regulatory concerns") {
        auto result = analyzer.analyze("Company faces antitrust probe and regulatory concerns");

        REQUIRE(result.score < 0.0);
        REQUIRE(result.label == "negative");
        REQUIRE(result.negative_keywords.size() >= 2);
    }

    SECTION("Multiple negative keywords") {
        auto result = analyzer.analyze("Losses deepen, guidance cut, crisis worsens");

        REQUIRE(result.score < -0.5); // Multiple keywords should amplify
        REQUIRE(result.label == "negative");
        REQUIRE(result.negative_keywords.size() >= 3);
    }
}

TEST_CASE("Sentiment analyzer handles neutral text", "[sentiment][neutral]") {
    SentimentAnalyzer analyzer;

    SECTION("Factual announcement") {
        auto result = analyzer.analyze("Company announces product launch event next month");

        REQUIRE(std::abs(result.score) < 0.3); // Should be near neutral
        REQUIRE(result.label == "neutral");
    }

    SECTION("No sentiment keywords") {
        auto result = analyzer.analyze("Meeting scheduled for Tuesday at headquarters");

        REQUIRE(result.label == "neutral");
        REQUIRE(result.positive_keywords.empty());
        REQUIRE(result.negative_keywords.empty());
    }
}

// ============================================================================
// Financial Domain Keywords
// ============================================================================

TEST_CASE("Sentiment analyzer recognizes financial keywords", "[sentiment][financial]") {
    SentimentAnalyzer analyzer;

    SECTION("Earnings-related keywords") {
        auto result = analyzer.analyze("Earnings beat estimates with strong margins");

        REQUIRE(result.score > 0.0);
        REQUIRE(result.label == "positive");
    }

    SECTION("Revenue keywords") {
        auto result = analyzer.analyze("Revenue miss disappoints investors");

        REQUIRE(result.score < 0.0);
        REQUIRE(result.label == "negative");
    }

    SECTION("Market movement keywords") {
        auto result = analyzer.analyze("Stock rallies on bullish analyst upgrade");

        REQUIRE(result.score > 0.0);
        REQUIRE(result.label == "positive");
    }

    SECTION("Downside market keywords") {
        auto result = analyzer.analyze("Selloff accelerates on bearish outlook");

        REQUIRE(result.score < 0.0);
        REQUIRE(result.label == "negative");
    }
}

// ============================================================================
// Scoring Logic Tests
// ============================================================================

TEST_CASE("Sentiment scores are in valid range", "[sentiment][scoring]") {
    SentimentAnalyzer analyzer;

    std::vector<std::string> test_cases = {
        "Stock surges dramatically on record earnings",
        "Company faces severe crisis and bankruptcy concerns",
        "Routine quarterly update with stable results", "Mixed results with both gains and losses",
        "" // Empty string
    };

    for (auto const& text : test_cases) {
        auto result = analyzer.analyze(text);

        REQUIRE(result.score >= -1.0);
        REQUIRE(result.score <= 1.0);
    }
}

TEST_CASE("Sentiment label matches score", "[sentiment][consistency]") {
    SentimentAnalyzer analyzer;

    SECTION("Positive score should have positive label") {
        auto result = analyzer.analyze("Excellent results exceed expectations");

        if (result.score > 0.1) {
            REQUIRE(result.label == "positive");
        }
    }

    SECTION("Negative score should have negative label") {
        auto result = analyzer.analyze("Disastrous results below expectations");

        if (result.score < -0.1) {
            REQUIRE(result.label == "negative");
        }
    }

    SECTION("Near-zero score should have neutral label") {
        auto result = analyzer.analyze("Company provides routine update");

        if (std::abs(result.score) <= 0.1) {
            REQUIRE(result.label == "neutral");
        }
    }
}

TEST_CASE("Keyword density calculation", "[sentiment][density]") {
    SentimentAnalyzer analyzer;

    SECTION("High keyword density") {
        auto result = analyzer.analyze("Surge rally growth beat exceed strong");

        // 6 positive keywords out of 6 words = 100% density
        REQUIRE(result.keyword_density >= 0.8);
    }

    SECTION("Low keyword density") {
        auto result =
            analyzer.analyze("The company held a meeting yesterday to discuss the upcoming "
                             "product launch and marketing strategy for next quarter");

        // Few keywords in many words = low density
        REQUIRE(result.keyword_density < 0.3);
    }
}

// ============================================================================
// Edge Cases and Robustness
// ============================================================================

TEST_CASE("Sentiment analyzer handles edge cases", "[sentiment][edge-cases]") {
    SentimentAnalyzer analyzer;

    SECTION("Empty string") {
        auto result = analyzer.analyze("");

        REQUIRE(result.score == 0.0);
        REQUIRE(result.label == "neutral");
        REQUIRE(result.total_words == 0);
    }

    SECTION("Single word - positive") {
        auto result = analyzer.analyze("Excellent");

        REQUIRE(result.score > 0.0);
        REQUIRE(result.label == "positive");
    }

    SECTION("Single word - negative") {
        auto result = analyzer.analyze("Terrible");

        REQUIRE(result.score < 0.0);
        REQUIRE(result.label == "negative");
    }

    SECTION("Very long text") {
        std::string long_text;
        for (int i = 0; i < 100; i++) {
            long_text += "The company continues to operate normally. ";
        }

        auto result = analyzer.analyze(long_text);

        // Should not crash or produce invalid scores
        REQUIRE(result.score >= -1.0);
        REQUIRE(result.score <= 1.0);
    }

    SECTION("Special characters and punctuation") {
        auto result = analyzer.analyze("Stock!!! Surges!!! 📈 Amazing!!!");

        REQUIRE(result.score > 0.0); // Should still detect "surges" and "amazing"
    }

    SECTION("Case insensitivity") {
        auto result1 = analyzer.analyze("STOCK SURGES ON EARNINGS");
        auto result2 = analyzer.analyze("stock surges on earnings");
        auto result3 = analyzer.analyze("Stock Surges On Earnings");

        // All should have same label
        REQUIRE(result1.label == result2.label);
        REQUIRE(result2.label == result3.label);

        // Scores should be similar (within 10%)
        REQUIRE(std::abs(result1.score - result2.score) < 0.1);
    }
}

TEST_CASE("Mixed sentiment handling", "[sentiment][mixed]") {
    SentimentAnalyzer analyzer;

    SECTION("Balanced positive and negative") {
        auto result = analyzer.analyze("Strong revenue growth offset by weak margin pressure");

        // Should have both types of keywords
        REQUIRE(!result.positive_keywords.empty());
        REQUIRE(!result.negative_keywords.empty());

        // Overall sentiment should be moderate
        REQUIRE(std::abs(result.score) < 0.7);
    }

    SECTION("Positive outweighs negative") {
        auto result = analyzer.analyze("Record earnings, strong guidance, and accelerating growth, "
                                       "despite minor concerns about supply chain");

        REQUIRE(result.score > 0.0); // Should be net positive
        REQUIRE(result.label == "positive");
    }

    SECTION("Negative outweighs positive") {
        auto result = analyzer.analyze("Severe losses, guidance cut, and layoffs announced, "
                                       "though some bright spots remain");

        REQUIRE(result.score < 0.0); // Should be net negative
        REQUIRE(result.label == "negative");
    }
}

// ============================================================================
// Batch Analysis Tests
// ============================================================================

TEST_CASE("Batch sentiment analysis", "[sentiment][batch]") {
    SentimentAnalyzer analyzer;

    std::vector<std::string> headlines = {
        "Stock surges on earnings beat", "Company faces regulatory probe",
        "Quarterly update shows stable results", "Record profits announced today",
        "Sales decline amid weak demand"};

    auto results = analyzer.analyzeBatch(headlines);

    REQUIRE(results.size() == headlines.size());

    // Check each result is valid
    for (size_t i = 0; i < results.size(); i++) {
        REQUIRE(results[i].score >= -1.0);
        REQUIRE(results[i].score <= 1.0);
        REQUIRE(!results[i].label.empty());
    }

    // Specific checks
    REQUIRE(results[0].label == "positive"); // "surges", "beat"
    REQUIRE(results[1].label == "negative"); // "probe"
    REQUIRE(results[3].label == "positive"); // "record", "profits"
    REQUIRE(results[4].label == "negative"); // "decline", "weak"
}

// ============================================================================
// Performance and Keyword Coverage
// ============================================================================

TEST_CASE("Keyword expansion improves coverage", "[sentiment][coverage]") {
    SentimentAnalyzer analyzer;

    // Test financial domain terms that should be recognized after expansion
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"Guidance raised on strong demand", "positive"},
        {"Margins compressed by inflation", "negative"},
        {"Breakout above resistance levels", "positive"},
        {"Selloff intensifies on recession fears", "negative"},
        {"Dividend increased for shareholders", "positive"},
        {"Bankruptcy concerns mount rapidly", "negative"},
        {"Bullish outlook from analysts", "positive"},
        {"Bearish sentiment prevails", "negative"},
    };

    int correct = 0;
    for (auto const& [text, expected] : test_cases) {
        auto result = analyzer.analyze(text);
        if (result.label == expected) {
            correct++;
        }
    }

    // After keyword expansion, should correctly classify at least 75%
    double accuracy = static_cast<double>(correct) / test_cases.size();
    REQUIRE(accuracy >= 0.75);
}

// ============================================================================
// Regression Tests
// ============================================================================

TEST_CASE("Sentiment regression tests", "[sentiment][regression]") {
    SentimentAnalyzer analyzer;

    // These specific examples should maintain consistent behavior
    SECTION("Known positive examples") {
        struct TestCase {
            std::string text;
            double min_score;
        };

        std::vector<TestCase> cases = {{"Apple stock surges on earnings beat", 0.3},
                                       {"Record profits exceed all expectations", 0.4},
                                       {"Strong revenue growth accelerates", 0.3}};

        for (auto const& tc : cases) {
            auto result = analyzer.analyze(tc.text);
            REQUIRE(result.label == "positive");
            REQUIRE(result.score >= tc.min_score);
        }
    }

    SECTION("Known negative examples") {
        struct TestCase {
            std::string text;
            double max_score;
        };

        std::vector<TestCase> cases = {{"Stock plunges on earnings miss", -0.3},
                                       {"Severe losses force layoffs", -0.4},
                                       {"Weak demand causes revenue decline", -0.3}};

        for (auto const& tc : cases) {
            auto result = analyzer.analyze(tc.text);
            REQUIRE(result.label == "negative");
            REQUIRE(result.score <= tc.max_score);
        }
    }
}
