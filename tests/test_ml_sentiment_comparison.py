"""
ML vs Keyword Sentiment Comparison and Backtesting

Compares ML-based sentiment (DistilRoBERTa) against keyword-based
sentiment analyzer on labeled financial news dataset.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import sys
import os
import pytest
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import both sentiment analyzers
try:
    from scripts.ml.sentiment_analyzer_ml import (
        MLSentimentAnalyzer,
        HybridSentimentAnalyzer
    )
    HAS_ML_SENTIMENT = True
except ImportError:
    HAS_ML_SENTIMENT = False

# Import keyword-based analyzer
try:
    from build import news_ingestion_py
    USE_CPP = True
except ImportError:
    USE_CPP = False
    from scripts.data_collection.news_ingestion import simple_sentiment

# Import backtesting utilities from existing test
from tests.test_sentiment_backtesting import (
    POSITIVE_NEWS,
    NEGATIVE_NEWS,
    NEUTRAL_NEWS,
    BacktestResults,
    calculate_metrics
)


@dataclass
class ComparisonResult:
    """Comparison between ML and keyword sentiment"""
    text: str
    expected_label: str
    ml_score: float
    ml_label: str
    ml_confidence: float
    keyword_score: float
    keyword_label: str
    ml_correct: bool
    keyword_correct: bool
    ml_latency_ms: float


# ============================================================================
# Sentiment Analysis Wrapper Functions
# ============================================================================

def analyze_ml_sentiment(analyzer, text: str) -> Tuple[float, str, float, float]:
    """Analyze with ML sentiment"""
    result = analyzer.analyze(text)
    return (result.score, result.label, result.confidence, result.latency_ms)


def analyze_keyword_sentiment(text: str) -> Tuple[float, str]:
    """Analyze with keyword sentiment"""
    if USE_CPP:
        result = news_ingestion_py.analyze_sentiment(text)
        return (result.score, result.label)
    else:
        score, label, _, _ = simple_sentiment(text)
        return (score, label)


# ============================================================================
# Comparison Tests
# ============================================================================

@pytest.mark.skipif(not HAS_ML_SENTIMENT, reason="ML sentiment not available")
def test_ml_sentiment_accuracy():
    """Test ML sentiment accuracy on labeled dataset"""
    print("\n" + "=" * 70)
    print("ML Sentiment Accuracy Test")
    print("=" * 70)

    # Initialize ML analyzer
    analyzer = MLSentimentAnalyzer(use_gpu=True)

    if not analyzer.is_available():
        pytest.skip("ML model not loaded")

    # Combine all labeled data
    dataset = []
    for text in POSITIVE_NEWS:
        dataset.append((text, "positive"))
    for text in NEGATIVE_NEWS:
        dataset.append((text, "negative"))
    for text in NEUTRAL_NEWS:
        dataset.append((text, "neutral"))

    print(f"\nDataset size: {len(dataset)} samples")
    print(f"  Positive: {len(POSITIVE_NEWS)}")
    print(f"  Negative: {len(NEGATIVE_NEWS)}")
    print(f"  Neutral: {len(NEUTRAL_NEWS)}")

    # Analyze all texts (use batching for efficiency)
    texts = [item[0] for item in dataset]
    expected_labels = [item[1] for item in dataset]

    print("\nRunning ML sentiment analysis...")
    start_time = time.time()
    results = analyzer.analyze_batch(texts, batch_size=16)
    total_time = time.time() - start_time

    # Calculate predictions
    predictions = []
    for expected, result in zip(expected_labels, results):
        predictions.append((expected, result.label))

    # Calculate metrics
    metrics = calculate_metrics(predictions)

    # Print results
    print(f"\nML Sentiment Performance:")
    print(f"  Overall Accuracy: {metrics.accuracy:.2%} ({metrics.correct_predictions}/{metrics.total_samples})")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time per text: {total_time / len(dataset) * 1000:.2f}ms")
    print()

    print("Per-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 48)
    for label in ["positive", "negative", "neutral"]:
        print(f"{label:<12} {metrics.precision[label]:<12.2%} "
              f"{metrics.recall[label]:<12.2%} {metrics.f1_score[label]:<12.2%}")

    avg_f1 = sum(metrics.f1_score.values()) / len(metrics.f1_score)
    print(f"\nMacro-Averaged F1: {avg_f1:.2%}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(f"{'Actual \\ Predicted':<20} {'Positive':<12} {'Negative':<12} {'Neutral':<12}")
    print("-" * 56)
    for actual in ["positive", "negative", "neutral"]:
        row = f"{actual.capitalize():<20}"
        for predicted in ["positive", "negative", "neutral"]:
            count = metrics.confusion_matrix[(actual, predicted)]
            row += f"{count:<12}"
        print(row)
    print("=" * 70 + "\n")

    # Assert ML performance (should be much better than keyword-based 74%)
    assert metrics.accuracy >= 0.85, f"ML accuracy {metrics.accuracy:.2%} below 85% threshold"
    assert avg_f1 >= 0.80, f"ML macro F1 {avg_f1:.2%} below 80% threshold"


@pytest.mark.skipif(not HAS_ML_SENTIMENT, reason="ML sentiment not available")
def test_ml_vs_keyword_comparison():
    """Head-to-head comparison of ML vs keyword sentiment"""
    print("\n" + "=" * 70)
    print("ML vs Keyword Sentiment Comparison")
    print("=" * 70)

    # Initialize ML analyzer
    ml_analyzer = MLSentimentAnalyzer(use_gpu=True)

    if not ml_analyzer.is_available():
        pytest.skip("ML model not loaded")

    # Use smaller subset for detailed comparison
    test_samples = [
        # Clear positive (both should get correct)
        ("Apple stock surges 5% on strong quarterly earnings beat", "positive"),
        ("Tesla reports record deliveries, shares rally", "positive"),
        ("Amazon announces stock split, shares soar to new highs", "positive"),

        # Clear negative (both should get correct)
        ("Netflix loses subscribers, stock plunges 20%", "negative"),
        ("Boeing faces safety concerns, shares sink", "negative"),
        ("Goldman Sachs announces layoffs amid deal drought", "negative"),

        # Subtle positive (ML should outperform)
        ("Company reaffirms guidance with slight upward revision", "positive"),
        ("Margins expand modestly despite headwinds", "positive"),

        # Subtle negative (ML should outperform)
        ("Revenue growth decelerates to single digits", "negative"),
        ("Management commentary suggests caution ahead", "negative"),

        # Neutral (both should get correct)
        ("Apple announces new product launch event", "neutral"),
        ("Microsoft releases routine security update", "neutral"),
    ]

    print(f"\nTest set: {len(test_samples)} samples")

    # Run comparison
    comparisons = []
    for text, expected in test_samples:
        # ML sentiment
        ml_score, ml_label, ml_conf, ml_latency = analyze_ml_sentiment(ml_analyzer, text)

        # Keyword sentiment
        kw_score, kw_label = analyze_keyword_sentiment(text)

        comparison = ComparisonResult(
            text=text,
            expected_label=expected,
            ml_score=ml_score,
            ml_label=ml_label,
            ml_confidence=ml_conf,
            keyword_score=kw_score,
            keyword_label=kw_label,
            ml_correct=(ml_label == expected),
            keyword_correct=(kw_label == expected),
            ml_latency_ms=ml_latency
        )
        comparisons.append(comparison)

    # Calculate accuracy
    ml_correct = sum(1 for c in comparisons if c.ml_correct)
    kw_correct = sum(1 for c in comparisons if c.keyword_correct)

    ml_accuracy = ml_correct / len(comparisons)
    kw_accuracy = kw_correct / len(comparisons)

    print(f"\nOverall Accuracy:")
    print(f"  ML Sentiment:      {ml_accuracy:.2%} ({ml_correct}/{len(comparisons)})")
    print(f"  Keyword Sentiment: {kw_accuracy:.2%} ({kw_correct}/{len(comparisons)})")
    print(f"  Improvement:       {(ml_accuracy - kw_accuracy):.2%}")

    # Average latency
    avg_ml_latency = sum(c.ml_latency_ms for c in comparisons) / len(comparisons)
    print(f"\nAverage Latency:")
    print(f"  ML Sentiment:      {avg_ml_latency:.2f}ms")

    # Show cases where ML outperforms keyword
    print("\n" + "-" * 70)
    print("Cases where ML outperforms Keyword:")
    print("-" * 70)
    for comp in comparisons:
        if comp.ml_correct and not comp.keyword_correct:
            print(f"\nText: {comp.text[:60]}...")
            print(f"  Expected: {comp.expected_label}")
            print(f"  ML:       {comp.ml_label} (confidence: {comp.ml_confidence:.2%}) ✓")
            print(f"  Keyword:  {comp.keyword_label} ✗")

    # Show cases where keyword outperforms ML (if any)
    kw_better = [c for c in comparisons if c.keyword_correct and not c.ml_correct]
    if kw_better:
        print("\n" + "-" * 70)
        print("Cases where Keyword outperforms ML:")
        print("-" * 70)
        for comp in kw_better:
            print(f"\nText: {comp.text[:60]}...")
            print(f"  Expected: {comp.expected_label}")
            print(f"  ML:       {comp.ml_label} (confidence: {comp.ml_confidence:.2%}) ✗")
            print(f"  Keyword:  {comp.keyword_label} ✓")

    print("=" * 70 + "\n")

    # ML should outperform keyword-based
    assert ml_accuracy >= kw_accuracy, "ML should be at least as accurate as keyword-based"


@pytest.mark.skipif(not HAS_ML_SENTIMENT, reason="ML sentiment not available")
def test_ml_sentiment_performance():
    """Test ML sentiment performance and latency"""
    print("\n" + "=" * 70)
    print("ML Sentiment Performance Test")
    print("=" * 70)

    analyzer = MLSentimentAnalyzer(use_gpu=True)

    if not analyzer.is_available():
        pytest.skip("ML model not loaded")

    # Run benchmark
    print("\nRunning performance benchmark...")
    results = analyzer.benchmark(num_samples=100)

    print("\nPerformance Results:")
    print(f"  Device: {results['device']}")
    print(f"  GPU enabled: {results['is_gpu']}")
    print()
    print(f"Single Inference:")
    print(f"  Average:  {results['single_avg_ms']:.2f}ms")
    print(f"  P50:      {results['single_p50_ms']:.2f}ms")
    print(f"  P95:      {results['single_p95_ms']:.2f}ms")
    print()
    print(f"Batch Inference ({results['batch_size']} texts):")
    print(f"  Total:    {results['batch_total_ms']:.2f}ms")
    print(f"  Per-text: {results['batch_avg_ms']:.2f}ms")

    # Assert performance targets
    # With batching, should be well under 100ms per article
    assert results['batch_avg_ms'] < 100, \
        f"Batch avg {results['batch_avg_ms']:.2f}ms exceeds 100ms target"

    print("\n✓ Performance target met: <100ms per article (batched)")
    print("=" * 70 + "\n")


@pytest.mark.skipif(not HAS_ML_SENTIMENT, reason="ML sentiment not available")
def test_hybrid_fallback_strategy():
    """Test hybrid analyzer with fallback to keyword-based"""
    print("\n" + "=" * 70)
    print("Hybrid Sentiment Fallback Test")
    print("=" * 70)

    # Initialize hybrid analyzer
    hybrid = HybridSentimentAnalyzer(use_gpu=True)

    test_texts = [
        "Apple stock surges on earnings beat",
        "Tesla faces production delays",
        "Microsoft announces new product"
    ]

    print("\nTesting hybrid analyzer (ML + keyword fallback)...")
    for text in test_texts:
        score, label, source, confidence = hybrid.analyze(text)
        print(f"\nText: {text}")
        print(f"  Score: {score:+.3f}")
        print(f"  Label: {label}")
        print(f"  Source: {source}")
        print(f"  Confidence: {confidence:.2%}")

    # Batch analysis
    print("\n" + "-" * 70)
    print("Batch analysis with hybrid:")
    results = hybrid.analyze_batch(test_texts)
    for text, (score, label, source, conf) in zip(test_texts, results):
        print(f"{text[:40]:40} | {label:8} | {source:8} | {conf:.2%}")

    print("=" * 70 + "\n")


@pytest.mark.skipif(not HAS_ML_SENTIMENT, reason="ML sentiment not available")
def test_full_dataset_comparison():
    """
    Full dataset comparison between ML and keyword sentiment

    This is the comprehensive test that runs all labeled data through
    both analyzers and provides detailed comparison metrics.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ML vs KEYWORD COMPARISON")
    print("=" * 70)

    # Initialize analyzers
    ml_analyzer = MLSentimentAnalyzer(use_gpu=True)

    if not ml_analyzer.is_available():
        pytest.skip("ML model not loaded")

    # Prepare full dataset
    dataset = []
    for text in POSITIVE_NEWS:
        dataset.append((text, "positive"))
    for text in NEGATIVE_NEWS:
        dataset.append((text, "negative"))
    for text in NEUTRAL_NEWS:
        dataset.append((text, "neutral"))

    print(f"\nFull Dataset: {len(dataset)} samples")
    print(f"  Positive: {len(POSITIVE_NEWS)}")
    print(f"  Negative: {len(NEGATIVE_NEWS)}")
    print(f"  Neutral: {len(NEUTRAL_NEWS)}")

    # Analyze with ML (batched)
    print("\n[1/2] Running ML sentiment analysis (batched)...")
    texts = [item[0] for item in dataset]
    expected_labels = [item[1] for item in dataset]

    start_time = time.time()
    ml_results = ml_analyzer.analyze_batch(texts, batch_size=16)
    ml_time = time.time() - start_time

    # Analyze with keyword
    print("[2/2] Running keyword sentiment analysis...")
    start_time = time.time()
    kw_results = [analyze_keyword_sentiment(text) for text in texts]
    kw_time = time.time() - start_time

    # Calculate ML metrics
    ml_predictions = [(expected, result.label) for expected, result in zip(expected_labels, ml_results)]
    ml_metrics = calculate_metrics(ml_predictions)

    # Calculate keyword metrics
    kw_predictions = [(expected, kw_label) for expected, (_, kw_label) in zip(expected_labels, kw_results)]
    kw_metrics = calculate_metrics(kw_predictions)

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'ML Sentiment':<20} {'Keyword Sentiment':<20} {'Improvement':<15}")
    print("-" * 85)

    print(f"{'Overall Accuracy':<30} "
          f"{ml_metrics.accuracy:<20.2%} "
          f"{kw_metrics.accuracy:<20.2%} "
          f"{(ml_metrics.accuracy - kw_metrics.accuracy):+.2%}")

    print(f"{'Positive Precision':<30} "
          f"{ml_metrics.precision['positive']:<20.2%} "
          f"{kw_metrics.precision['positive']:<20.2%} "
          f"{(ml_metrics.precision['positive'] - kw_metrics.precision['positive']):+.2%}")

    print(f"{'Positive Recall':<30} "
          f"{ml_metrics.recall['positive']:<20.2%} "
          f"{kw_metrics.recall['positive']:<20.2%} "
          f"{(ml_metrics.recall['positive'] - kw_metrics.recall['positive']):+.2%}")

    print(f"{'Negative Precision':<30} "
          f"{ml_metrics.precision['negative']:<20.2%} "
          f"{kw_metrics.precision['negative']:<20.2%} "
          f"{(ml_metrics.precision['negative'] - kw_metrics.precision['negative']):+.2%}")

    print(f"{'Negative Recall':<30} "
          f"{ml_metrics.recall['negative']:<20.2%} "
          f"{kw_metrics.recall['negative']:<20.2%} "
          f"{(ml_metrics.recall['negative'] - kw_metrics.recall['negative']):+.2%}")

    ml_avg_f1 = sum(ml_metrics.f1_score.values()) / len(ml_metrics.f1_score)
    kw_avg_f1 = sum(kw_metrics.f1_score.values()) / len(kw_metrics.f1_score)

    print(f"{'Macro F1-Score':<30} "
          f"{ml_avg_f1:<20.2%} "
          f"{kw_avg_f1:<20.2%} "
          f"{(ml_avg_f1 - kw_avg_f1):+.2%}")

    print(f"{'Total Processing Time':<30} "
          f"{ml_time:<20.2f}s "
          f"{kw_time:<20.2f}s "
          f"{(ml_time / kw_time):+.2f}x")

    print(f"{'Avg Time Per Text':<30} "
          f"{ml_time / len(dataset) * 1000:<20.2f}ms "
          f"{kw_time / len(dataset) * 1000:<20.2f}ms "
          f"{(ml_time / kw_time):+.2f}x")

    print("=" * 70)

    # Print per-class F1 scores
    print("\nPer-Class F1-Scores:")
    print(f"{'Class':<15} {'ML F1':<15} {'Keyword F1':<15} {'Improvement':<15}")
    print("-" * 60)
    for label in ["positive", "negative", "neutral"]:
        print(f"{label:<15} "
              f"{ml_metrics.f1_score[label]:<15.2%} "
              f"{kw_metrics.f1_score[label]:<15.2%} "
              f"{(ml_metrics.f1_score[label] - kw_metrics.f1_score[label]):+.2%}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    improvement = ml_metrics.accuracy - kw_metrics.accuracy
    print(f"\nML sentiment provides {improvement:+.2%} accuracy improvement over keyword-based")
    print(f"ML accuracy: {ml_metrics.accuracy:.2%} vs Keyword accuracy: {kw_metrics.accuracy:.2%}")

    if ml_metrics.accuracy >= 0.85:
        print("✓ ML sentiment meets 85% accuracy target")
    else:
        print("✗ ML sentiment below 85% accuracy target")

    if ml_time / len(dataset) * 1000 < 100:
        print("✓ ML sentiment meets <100ms per-article performance target (batched)")
    else:
        print("✗ ML sentiment exceeds 100ms per-article target")

    print("=" * 70 + "\n")

    # Assert ML outperforms keyword
    assert ml_metrics.accuracy > kw_metrics.accuracy, \
        "ML sentiment should outperform keyword-based sentiment"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
