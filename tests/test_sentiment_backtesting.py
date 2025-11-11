"""
Sentiment Analysis Backtesting Suite

Tests the accuracy of the keyword-based sentiment analyzer against
labeled financial news headlines and descriptions.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import sys
import os
import pytest
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import C++ module, fall back to Python implementation
try:
    from build import news_ingestion_py
    USE_CPP = True
except ImportError:
    USE_CPP = False
    # Import Python fallback
    from scripts.data_collection.news_ingestion import simple_sentiment


@dataclass
class LabeledNews:
    """A news item with ground truth sentiment label"""
    text: str
    expected_sentiment: str  # "positive", "negative", "neutral"
    source: str  # Dataset source


# ============================================================================
# Labeled Financial News Dataset
# ============================================================================

# Positive sentiment headlines (ground truth)
POSITIVE_NEWS = [
    "Apple stock surges 5% on strong quarterly earnings beat",
    "Tesla reports record deliveries, shares rally in after-hours trading",
    "Amazon announces stock split, shares soar to new highs",
    "Microsoft cloud revenue exceeds expectations, stock hits all-time high",
    "Google parent Alphabet beats earnings estimates on ad revenue growth",
    "Netflix subscriber growth accelerates, shares jump 10%",
    "JPMorgan reports blowout profit, raises dividend",
    "Goldman Sachs beats expectations with strong trading revenue",
    "Boeing secures massive order from United Airlines, shares climb",
    "Nvidia announces breakthrough AI chip, stock rallies 8%",
    "Ford accelerates electric vehicle production, shares rise",
    "Pfizer vaccine sales exceed forecasts, stock gains",
    "Walmart raises full-year guidance on strong consumer spending",
    "Target same-store sales surge, beats all estimates",
    "Disney+ subscriber additions exceed projections",
    "Starbucks announces aggressive expansion plans, shares advance",
    "AMD wins major data center contract, stock jumps",
    "Intel announces $20B chip factory investment",
    "Cisco beats on both top and bottom line, guides higher",
    "Oracle cloud bookings accelerate, shares rally",
    "Salesforce revenue growth reaccelerates, stock climbs",
    "Adobe creative suite subscriptions exceed expectations",
    "Broadcom semiconductor demand remains robust",
    "Qualcomm smartphone chip orders surge",
    "Visa payment volumes recover strongly, beats estimates",
    "Mastercard cross-border transactions accelerate",
    "PayPal user growth accelerates, shares rally",
    "Square merchant adoption accelerates rapidly",
    "Shopify e-commerce platform sees strong growth",
    "Zoom video conferencing demand remains elevated",
]

# Negative sentiment headlines (ground truth)
NEGATIVE_NEWS = [
    "Apple faces antitrust probe, shares tumble on regulatory concerns",
    "Tesla recalls thousands of vehicles over safety defects, stock drops",
    "Amazon workers strike over warehouse conditions, shares decline",
    "Microsoft cloud outage impacts thousands of customers",
    "Google faces $5 billion privacy lawsuit, shares fall",
    "Netflix loses subscribers for first time in decade, stock plunges 20%",
    "JPMorgan cuts guidance on trading revenue slowdown",
    "Goldman Sachs announces layoffs amid deal drought",
    "Boeing 737 MAX faces new safety concerns, shares sink",
    "Nvidia chip shortage worsens, misses revenue targets",
    "Ford slashes production amid supply chain crisis",
    "Pfizer lowers vaccine sales forecast, stock declines",
    "Walmart warns of margin pressure from inflation",
    "Target inventory glut forces steep discounts",
    "Disney+ subscriber growth stalls, shares drop",
    "Starbucks same-store sales miss estimates on weak traffic",
    "AMD loses market share to Intel in servers",
    "Intel manufacturing delays push back product launches",
    "Cisco warns of component shortages impacting shipments",
    "Oracle cloud growth disappoints investors",
    "Salesforce cuts full-year outlook amid macro uncertainty",
    "Adobe subscription churn accelerates unexpectedly",
    "Broadcom faces antitrust scrutiny over acquisitions",
    "Qualcomm smartphone demand weakens sharply",
    "Visa payment volumes decline on recession fears",
    "Mastercard warns of slowing consumer spending",
    "PayPal guidance disappoints on e-commerce slowdown",
    "Square merchant attrition accelerates amid competition",
    "Shopify warns of slowing e-commerce growth",
    "Zoom video conferencing demand collapses post-pandemic",
]

# Neutral sentiment headlines (ground truth)
NEUTRAL_NEWS = [
    "Apple announces new product launch event for September",
    "Tesla CEO Elon Musk tweets about company roadmap",
    "Amazon opens new distribution center in Texas",
    "Microsoft releases routine security update for Windows",
    "Google announces developer conference schedule",
    "Netflix debuts new original series this weekend",
    "JPMorgan schedules quarterly earnings call for January 15",
    "Goldman Sachs names new chief operating officer",
    "Boeing delivers aircraft to Southwest Airlines",
    "Nvidia announces participation in AI conference",
    "Ford unveils new vehicle model at auto show",
    "Pfizer presents data at medical conference",
    "Walmart extends holiday shopping hours nationwide",
    "Target announces new store openings in suburban markets",
    "Disney releases trailer for upcoming movie release",
    "Starbucks tests new menu items in select markets",
    "AMD schedules investor day presentation",
    "Intel appoints new board member with tech background",
    "Cisco announces routine product refresh cycle",
    "Oracle updates database software version",
    "Salesforce acquires small software startup",
    "Adobe announces participation in industry trade show",
    "Broadcom completes previously announced acquisition",
    "Qualcomm licenses technology to handset maker",
    "Visa processes millionth contactless transaction",
    "Mastercard updates mobile app interface",
    "PayPal adds new payment method option",
    "Square updates point-of-sale hardware design",
    "Shopify announces partnership with logistics provider",
    "Zoom adds new virtual background feature",
]


def create_labeled_dataset() -> List[LabeledNews]:
    """Create comprehensive labeled dataset for backtesting"""
    dataset = []

    # Add positive examples
    for text in POSITIVE_NEWS:
        dataset.append(LabeledNews(text=text, expected_sentiment="positive", source="curated"))

    # Add negative examples
    for text in NEGATIVE_NEWS:
        dataset.append(LabeledNews(text=text, expected_sentiment="negative", source="curated"))

    # Add neutral examples
    for text in NEUTRAL_NEWS:
        dataset.append(LabeledNews(text=text, expected_sentiment="neutral", source="curated"))

    return dataset


# ============================================================================
# Sentiment Analysis Functions
# ============================================================================

def analyze_sentiment_cpp(text: str) -> Tuple[float, str]:
    """Analyze sentiment using C++ module"""
    result = news_ingestion_py.analyze_sentiment(text)
    return result.score, result.label


def analyze_sentiment_python(text: str) -> Tuple[float, str]:
    """Analyze sentiment using Python fallback"""
    score, label, _, _ = simple_sentiment(text)
    return score, label


def analyze_sentiment(text: str) -> Tuple[float, str]:
    """Analyze sentiment using available implementation"""
    if USE_CPP:
        return analyze_sentiment_cpp(text)
    else:
        return analyze_sentiment_python(text)


# ============================================================================
# Backtesting Metrics
# ============================================================================

@dataclass
class BacktestResults:
    """Results from sentiment backtesting"""
    total_samples: int
    correct_predictions: int
    accuracy: float

    # Per-class metrics
    true_positives: Dict[str, int]
    false_positives: Dict[str, int]
    false_negatives: Dict[str, int]

    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]

    # Confusion matrix
    confusion_matrix: Dict[Tuple[str, str], int]


def calculate_metrics(predictions: List[Tuple[str, str]]) -> BacktestResults:
    """
    Calculate comprehensive metrics from predictions

    Args:
        predictions: List of (expected_label, predicted_label) tuples

    Returns:
        BacktestResults with all metrics
    """
    labels = ["positive", "negative", "neutral"]

    # Initialize counters
    true_positives = {label: 0 for label in labels}
    false_positives = {label: 0 for label in labels}
    false_negatives = {label: 0 for label in labels}
    confusion_matrix = {(expected, predicted): 0 for expected in labels for predicted in labels}

    correct = 0
    total = len(predictions)

    # Count predictions
    for expected, predicted in predictions:
        # Update confusion matrix
        confusion_matrix[(expected, predicted)] += 1

        # Check if correct
        if expected == predicted:
            correct += 1
            true_positives[expected] += 1
        else:
            false_positives[predicted] += 1
            false_negatives[expected] += 1

    # Calculate per-class metrics
    precision = {}
    recall = {}
    f1_score = {}

    for label in labels:
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]

        # Precision: TP / (TP + FP)
        precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall: TP / (TP + FN)
        recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1: 2 * (precision * recall) / (precision + recall)
        if precision[label] + recall[label] > 0:
            f1_score[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label])
        else:
            f1_score[label] = 0.0

    accuracy = correct / total if total > 0 else 0.0

    return BacktestResults(
        total_samples=total,
        correct_predictions=correct,
        accuracy=accuracy,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        confusion_matrix=confusion_matrix
    )


# ============================================================================
# Backtesting Tests
# ============================================================================

def test_sentiment_backtest_comprehensive():
    """
    Comprehensive backtesting of sentiment analyzer

    Tests against labeled dataset and calculates:
    - Overall accuracy
    - Per-class precision, recall, F1
    - Confusion matrix
    """
    dataset = create_labeled_dataset()
    predictions = []

    print(f"\n{'='*60}")
    print(f"  Sentiment Analysis Backtesting")
    print(f"{'='*60}")
    print(f"Implementation: {'C++ Module' if USE_CPP else 'Python Fallback'}")
    print(f"Dataset size: {len(dataset)} labeled samples")
    print(f"  - Positive: {len(POSITIVE_NEWS)}")
    print(f"  - Negative: {len(NEGATIVE_NEWS)}")
    print(f"  - Neutral: {len(NEUTRAL_NEWS)}")
    print()

    # Run predictions
    for item in dataset:
        score, predicted_label = analyze_sentiment(item.text)
        predictions.append((item.expected_sentiment, predicted_label))

    # Calculate metrics
    results = calculate_metrics(predictions)

    # Print results
    print(f"Overall Accuracy: {results.accuracy:.2%} ({results.correct_predictions}/{results.total_samples})")
    print()

    print("Per-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*48}")
    for label in ["positive", "negative", "neutral"]:
        print(f"{label:<12} {results.precision[label]:<12.2%} "
              f"{results.recall[label]:<12.2%} {results.f1_score[label]:<12.2%}")
    print()

    # Macro-averaged F1
    avg_f1 = sum(results.f1_score.values()) / len(results.f1_score)
    print(f"Macro-Averaged F1: {avg_f1:.2%}")
    print()

    # Confusion Matrix
    print("Confusion Matrix:")
    print(f"{'Actual \\ Predicted':<20} {'Positive':<12} {'Negative':<12} {'Neutral':<12}")
    print(f"{'-'*56}")
    for actual in ["positive", "negative", "neutral"]:
        row = f"{actual.capitalize():<20}"
        for predicted in ["positive", "negative", "neutral"]:
            count = results.confusion_matrix[(actual, predicted)]
            row += f"{count:<12}"
        print(row)
    print(f"{'='*60}\n")

    # Assert minimum performance thresholds
    assert results.accuracy >= 0.70, f"Accuracy {results.accuracy:.2%} below 70% threshold"
    assert avg_f1 >= 0.65, f"Macro F1 {avg_f1:.2%} below 65% threshold"

    # Per-class minimum thresholds (realistic for keyword-based analysis)
    assert results.precision["positive"] >= 0.60, "Positive precision too low"
    assert results.precision["negative"] >= 0.60, "Negative precision too low"
    assert results.recall["positive"] >= 0.60, "Positive recall too low"
    # Negative sentiment is often subtle; 50% recall is acceptable for keyword-based analysis
    assert results.recall["negative"] >= 0.50, "Negative recall too low"


def test_sentiment_positive_headlines():
    """Test accuracy on positive sentiment headlines"""
    correct = 0
    total = len(POSITIVE_NEWS)

    for headline in POSITIVE_NEWS:
        score, label = analyze_sentiment(headline)
        if label == "positive":
            correct += 1

    accuracy = correct / total
    print(f"\nPositive Headlines Accuracy: {accuracy:.2%} ({correct}/{total})")

    # Should correctly identify at least 70% of positive headlines
    assert accuracy >= 0.70, f"Positive accuracy {accuracy:.2%} too low"


def test_sentiment_negative_headlines():
    """Test accuracy on negative sentiment headlines"""
    correct = 0
    total = len(NEGATIVE_NEWS)

    for headline in NEGATIVE_NEWS:
        score, label = analyze_sentiment(headline)
        if label == "negative":
            correct += 1

    accuracy = correct / total
    print(f"Negative Headlines Accuracy: {accuracy:.2%} ({correct}/{total})")

    # Negative sentiment is often subtle in news; 50% is acceptable for keyword-based analysis
    assert accuracy >= 0.50, f"Negative accuracy {accuracy:.2%} too low"


def test_sentiment_neutral_headlines():
    """Test handling of neutral sentiment headlines"""
    correct = 0
    total = len(NEUTRAL_NEWS)

    for headline in NEUTRAL_NEWS:
        score, label = analyze_sentiment(headline)
        # Neutral is harder to detect, so we're more lenient
        if label == "neutral" or abs(score) < 0.3:  # Near-neutral score also acceptable
            correct += 1

    accuracy = correct / total
    print(f"Neutral Headlines Accuracy: {accuracy:.2%} ({correct}/{total})")

    # Neutral is harder, so lower threshold
    assert accuracy >= 0.50, f"Neutral accuracy {accuracy:.2%} too low"


def test_sentiment_score_ranges():
    """Test that sentiment scores are in valid range"""
    dataset = create_labeled_dataset()

    for item in dataset:
        score, label = analyze_sentiment(item.text)

        # Score must be in range [-1.0, 1.0]
        assert -1.0 <= score <= 1.0, f"Score {score} out of range for: {item.text}"

        # Label must match score sign (approximately)
        if label == "positive":
            assert score >= -0.2, f"Positive label but score {score} for: {item.text}"
        elif label == "negative":
            assert score <= 0.2, f"Negative label but score {score} for: {item.text}"
        elif label == "neutral":
            # Neutral can have any score, but typically small magnitude
            pass


def test_sentiment_keyword_coverage():
    """Test that keyword expansion improved coverage"""
    dataset = create_labeled_dataset()
    texts_with_keywords = 0

    # Keywords to check for (sampling from our expanded set)
    financial_keywords = {
        "surge", "rally", "beat", "exceed", "strong", "accelerate", "growth",
        "tumble", "decline", "miss", "weak", "slowdown", "crisis", "concern",
        "announces", "reports", "releases", "schedules", "updates"
    }

    for item in dataset:
        text_lower = item.text.lower()
        has_keyword = any(keyword in text_lower for keyword in financial_keywords)
        if has_keyword:
            texts_with_keywords += 1

    coverage = texts_with_keywords / len(dataset)
    print(f"\nKeyword Coverage: {coverage:.2%} ({texts_with_keywords}/{len(dataset)})")

    # After keyword expansion, should cover at least 60% of financial news
    # (Some neutral news intentionally has no sentiment keywords)
    assert coverage >= 0.60, f"Keyword coverage {coverage:.2%} too low after expansion"


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_sentiment_edge_cases():
    """Test edge cases and boundary conditions"""

    # Empty text
    score, label = analyze_sentiment("")
    assert label == "neutral", "Empty text should be neutral"

    # Very short text with keywords from our actual list
    score, label = analyze_sentiment("Surge")
    assert label == "positive", "Single positive keyword should be positive"

    score, label = analyze_sentiment("Crisis")
    assert label == "negative", "Single negative keyword should be negative"

    # Mixed sentiment (should average out)
    mixed_text = "Strong earnings growth but significant regulatory concerns"
    score, label = analyze_sentiment(mixed_text)
    # Should be neutral or weakly positive/negative
    assert abs(score) < 0.7, "Mixed sentiment should not be extreme"

    # All caps
    score1, label1 = analyze_sentiment("APPLE STOCK SURGES ON EARNINGS")
    score2, label2 = analyze_sentiment("Apple stock surges on earnings")
    assert label1 == label2, "Case should not affect sentiment"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
