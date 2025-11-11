#!/usr/bin/env python3
"""
BigBrotherAnalytics - Comprehensive News & ML Sentiment Regression Testing

Extensive regression testing for news sources with ML sentiment analysis
Implements power-of-2 batch testing for performance validation
Includes backtesting capabilities for historical data analysis

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-11
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts" / "data_collection"))

from ml_sentiment_analyzer_py import create_ml_sentiment_analyzer, MLSentimentResult


class NewsMLRegressionTest:
    """
    Comprehensive regression testing for news sources with ML sentiment

    Features:
    - Power-of-2 batch testing (1, 2, 4, 8, 16, 32, 64, 128)
    - NewsAPI testing with real articles
    - ML vs Keyword sentiment comparison
    - Performance metrics and throughput analysis
    - Backtesting support for historical data
    - Statistical validation
    """

    def __init__(self, db_path: str = "data/bigbrother.duckdb"):
        """
        Initialize regression tester

        Args:
            db_path: Path to DuckDB database
        """
        self.db_path = Path(project_root) / db_path
        self.conn = duckdb.connect(str(self.db_path))

        # Initialize ML sentiment analyzer
        print("Initializing ML sentiment analyzer...")
        self.ml_analyzer = create_ml_sentiment_analyzer(use_cuda=True)
        print()

        # Power-of-2 batch sizes for testing
        self.power_of_2_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

        # Results storage
        self.test_results = []

    def fetch_news_articles(
        self, limit: int = 128, source: str = "newsapi"
    ) -> pd.DataFrame:
        """
        Fetch news articles from database

        Args:
            limit: Maximum number of articles to fetch
            source: News source (newsapi, alphavantage)

        Returns:
            DataFrame with articles
        """
        query = f"""
            SELECT
                article_id,
                title,
                description,
                content,
                published_at,
                source_name,
                sentiment_score as keyword_sentiment_score,
                sentiment_label as keyword_sentiment_label
            FROM news_articles
            WHERE source_name LIKE '%{source}%'
            ORDER BY published_at DESC
            LIMIT {limit}
        """

        df = self.conn.execute(query).fetchdf()
        print(f"✓ Fetched {len(df)} articles from {source}")
        return df

    def analyze_with_ml_sentiment(
        self, texts: List[str], batch_size: int = 8
    ) -> List[MLSentimentResult]:
        """
        Analyze texts with ML sentiment in batches

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing

        Returns:
            List of ML sentiment results
        """
        return self.ml_analyzer.analyze_batch(texts, batch_size=batch_size)

    def compare_sentiment_methods(
        self, articles: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compare ML vs keyword-based sentiment

        Args:
            articles: DataFrame with articles and keyword sentiment

        Returns:
            Comparison metrics
        """
        # Combine title and description for analysis
        texts = [
            f"{row['title']}. {row['description'] or ''}"
            for _, row in articles.iterrows()
        ]

        # Analyze with ML
        start_time = time.perf_counter()
        ml_results = self.ml_analyzer.analyze_batch(texts)
        end_time = time.perf_counter()

        ml_time = end_time - start_time
        avg_ml_time = (ml_time / len(texts)) * 1000  # ms per article

        # Calculate agreement
        agreements = 0
        for i, ml_result in enumerate(ml_results):
            keyword_label = articles.iloc[i]["keyword_sentiment_label"]
            if ml_result.label == keyword_label:
                agreements += 1

        agreement_rate = (agreements / len(ml_results)) * 100

        # Calculate sentiment distribution
        ml_positive = sum(1 for r in ml_results if r.label == "positive")
        ml_negative = sum(1 for r in ml_results if r.label == "negative")
        ml_neutral = sum(1 for r in ml_results if r.label == "neutral")

        keyword_positive = sum(
            1
            for label in articles["keyword_sentiment_label"]
            if label == "positive"
        )
        keyword_negative = sum(
            1
            for label in articles["keyword_sentiment_label"]
            if label == "negative"
        )
        keyword_neutral = sum(
            1 for label in articles["keyword_sentiment_label"] if label == "neutral"
        )

        return {
            "total_articles": len(texts),
            "ml_avg_time_ms": avg_ml_time,
            "ml_throughput": 1000 / avg_ml_time,
            "agreement_rate": agreement_rate,
            "ml_positive": ml_positive,
            "ml_negative": ml_negative,
            "ml_neutral": ml_neutral,
            "keyword_positive": keyword_positive,
            "keyword_negative": keyword_negative,
            "keyword_neutral": keyword_neutral,
            "ml_avg_confidence": sum(r.confidence for r in ml_results)
            / len(ml_results),
        }

    def power_of_2_batch_test(self) -> pd.DataFrame:
        """
        Test ML sentiment with power-of-2 batch sizes

        Returns:
            DataFrame with performance metrics for each batch size
        """
        print("=" * 70)
        print("Power-of-2 Batch Size Testing")
        print("=" * 70)
        print()

        # Fetch maximum articles needed
        max_articles = max(self.power_of_2_sizes)
        articles = self.fetch_news_articles(limit=max_articles)

        if len(articles) == 0:
            print("✗ No articles found in database")
            print("  Run: uv run python scripts/data_collection/news_ingestion.py")
            return pd.DataFrame()

        results = []

        for batch_size in self.power_of_2_sizes:
            if batch_size > len(articles):
                print(
                    f"⚠ Skipping batch size {batch_size} (only {len(articles)} articles available)"
                )
                continue

            print(f"\nTesting batch size: {batch_size}")
            print("-" * 70)

            # Take first N articles
            test_articles = articles.head(batch_size)
            texts = [
                f"{row['title']}. {row['description'] or ''}"
                for _, row in test_articles.iterrows()
            ]

            # Warm-up run
            _ = self.ml_analyzer.analyze_batch(texts[:min(2, len(texts))])

            # Actual test run
            start_time = time.perf_counter()
            ml_results = self.ml_analyzer.analyze_batch(texts)
            end_time = time.perf_counter()

            total_time = end_time - start_time
            avg_time_ms = (total_time / batch_size) * 1000
            throughput = 1000 / avg_time_ms

            # Calculate sentiment distribution
            positive = sum(1 for r in ml_results if r.label == "positive")
            negative = sum(1 for r in ml_results if r.label == "negative")
            neutral = sum(1 for r in ml_results if r.label == "neutral")
            avg_confidence = sum(r.confidence for r in ml_results) / len(ml_results)

            result = {
                "batch_size": batch_size,
                "total_time_s": total_time,
                "avg_time_ms": avg_time_ms,
                "throughput_per_sec": throughput,
                "positive_count": positive,
                "negative_count": negative,
                "neutral_count": neutral,
                "avg_confidence": avg_confidence,
            }

            results.append(result)

            print(f"  Total Time: {total_time:.4f}s")
            print(f"  Avg Time: {avg_time_ms:.2f}ms per article")
            print(f"  Throughput: {throughput:.2f} articles/sec")
            print(f"  Sentiment: +{positive} -{negative} ={neutral}")
            print(f"  Avg Confidence: {avg_confidence:.4f}")

        results_df = pd.DataFrame(results)
        return results_df

    def backtest_sentiment_signals(
        self, days_back: int = 30
    ) -> Dict[str, any]:
        """
        Backtest sentiment signals for trading decisions

        Args:
            days_back: Number of days to backtest

        Returns:
            Backtesting metrics
        """
        print("\n" + "=" * 70)
        print("Backtesting Sentiment Signals")
        print("=" * 70)
        print()

        # Fetch historical articles
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        query = f"""
            SELECT
                article_id,
                title,
                description,
                published_at,
                symbol,
                sentiment_score as keyword_sentiment_score
            FROM news_articles
            WHERE published_at >= '{start_date.isoformat()}'
              AND published_at <= '{end_date.isoformat()}'
              AND symbol IS NOT NULL
            ORDER BY published_at ASC
        """

        articles = self.conn.execute(query).fetchdf()

        if len(articles) == 0:
            print(f"✗ No articles found in last {days_back} days")
            return {}

        print(f"✓ Fetched {len(articles)} articles for backtesting")
        print(f"  Date Range: {start_date.date()} to {end_date.date()}")
        print()

        # Analyze with ML sentiment
        texts = [
            f"{row['title']}. {row['description'] or ''}"
            for _, row in articles.iterrows()
        ]
        ml_results = self.ml_analyzer.analyze_batch(texts)

        # Add ML sentiment to dataframe
        articles["ml_sentiment_score"] = [r.score for r in ml_results]
        articles["ml_sentiment_label"] = [r.label for r in ml_results]
        articles["ml_confidence"] = [r.confidence for r in ml_results]

        # Group by symbol and calculate aggregate sentiment
        symbol_sentiment = (
            articles.groupby("symbol")
            .agg(
                {
                    "article_id": "count",
                    "ml_sentiment_score": ["mean", "std"],
                    "ml_confidence": "mean",
                    "keyword_sentiment_score": "mean",
                }
            )
            .reset_index()
        )

        symbol_sentiment.columns = [
            "_".join(col).strip("_") for col in symbol_sentiment.columns.values
        ]
        symbol_sentiment = symbol_sentiment.rename(
            columns={
                "article_id_count": "article_count",
                "ml_sentiment_score_mean": "ml_avg_sentiment",
                "ml_sentiment_score_std": "ml_sentiment_volatility",
                "ml_confidence_mean": "ml_avg_confidence",
                "keyword_sentiment_score_mean": "keyword_avg_sentiment",
            }
        )

        # Find strong signals (high confidence, clear sentiment)
        strong_signals = symbol_sentiment[
            (symbol_sentiment["ml_avg_confidence"] > 0.90)
            & (abs(symbol_sentiment["ml_avg_sentiment"]) > 0.5)
            & (symbol_sentiment["article_count"] >= 3)
        ].sort_values("ml_avg_sentiment", ascending=False)

        print("Strong Sentiment Signals (High Confidence):")
        print("-" * 70)
        for _, row in strong_signals.iterrows():
            sentiment_direction = (
                "POSITIVE" if row["ml_avg_sentiment"] > 0 else "NEGATIVE"
            )
            print(f"  {row['symbol']}: {sentiment_direction}")
            print(f"    ML Sentiment: {row['ml_avg_sentiment']:.4f}")
            print(
                f"    Confidence: {row['ml_avg_confidence']:.4f} ({row['article_count']} articles)"
            )
            print(
                f"    vs Keyword: {row['keyword_avg_sentiment']:.4f} (Δ={abs(row['ml_avg_sentiment'] - row['keyword_avg_sentiment']):.4f})"
            )
            print()

        return {
            "backtest_days": days_back,
            "total_articles": len(articles),
            "unique_symbols": len(symbol_sentiment),
            "strong_signals": len(strong_signals),
            "symbol_sentiment": symbol_sentiment,
            "strong_signals_df": strong_signals,
        }

    def run_comprehensive_test(self) -> Dict[str, any]:
        """
        Run comprehensive regression test suite

        Returns:
            Complete test results
        """
        print("=" * 70)
        print("BigBrotherAnalytics - Comprehensive News & ML Regression Test")
        print("=" * 70)
        print()
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Database: {self.db_path}")
        print(f"ML Device: {self.ml_analyzer.get_device()}")
        print()

        # Test 1: Power-of-2 batch testing
        power_of_2_results = self.power_of_2_batch_test()

        # Test 2: ML vs Keyword comparison
        print("\n" + "=" * 70)
        print("ML vs Keyword Sentiment Comparison")
        print("=" * 70)
        print()

        articles = self.fetch_news_articles(limit=100)
        if len(articles) > 0:
            comparison = self.compare_sentiment_methods(articles)

            print(f"Total Articles: {comparison['total_articles']}")
            print(f"ML Avg Time: {comparison['ml_avg_time_ms']:.2f}ms per article")
            print(f"ML Throughput: {comparison['ml_throughput']:.2f} articles/sec")
            print(f"Agreement Rate: {comparison['agreement_rate']:.2f}%")
            print(f"ML Avg Confidence: {comparison['ml_avg_confidence']:.4f}")
            print()
            print("Sentiment Distribution:")
            print(
                f"  ML:      +{comparison['ml_positive']} -{comparison['ml_negative']} ={comparison['ml_neutral']}"
            )
            print(
                f"  Keyword: +{comparison['keyword_positive']} -{comparison['keyword_negative']} ={comparison['keyword_neutral']}"
            )
        else:
            comparison = {}

        # Test 3: Backtesting
        backtest_results = self.backtest_sentiment_signals(days_back=30)

        # Summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print()
        print("✓ Power-of-2 batch testing: COMPLETE")
        print("✓ ML vs Keyword comparison: COMPLETE")
        print("✓ Sentiment backtesting: COMPLETE")
        print()

        if not power_of_2_results.empty:
            best_throughput = power_of_2_results["throughput_per_sec"].max()
            best_batch = power_of_2_results.loc[
                power_of_2_results["throughput_per_sec"].idxmax(), "batch_size"
            ]
            print(f"Best Throughput: {best_throughput:.2f} articles/sec (batch={best_batch})")

        print()
        print(f"Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        return {
            "power_of_2_results": power_of_2_results,
            "comparison": comparison,
            "backtest": backtest_results,
        }

    def save_results(self, results: Dict[str, any], output_dir: str = "test_results"):
        """
        Save test results to files

        Args:
            results: Test results dictionary
            output_dir: Output directory for results
        """
        output_path = project_root / output_dir
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save power-of-2 results
        if not results["power_of_2_results"].empty:
            power_of_2_file = output_path / f"power_of_2_results_{timestamp}.csv"
            results["power_of_2_results"].to_csv(power_of_2_file, index=False)
            print(f"✓ Saved power-of-2 results: {power_of_2_file}")

        # Save comparison results
        if results["comparison"]:
            comparison_file = output_path / f"ml_vs_keyword_{timestamp}.json"
            import json

            with open(comparison_file, "w") as f:
                json.dump(results["comparison"], f, indent=2)
            print(f"✓ Saved comparison results: {comparison_file}")

        # Save backtest results
        if results["backtest"] and "strong_signals_df" in results["backtest"]:
            backtest_file = output_path / f"backtest_signals_{timestamp}.csv"
            results["backtest"]["strong_signals_df"].to_csv(backtest_file, index=False)
            print(f"✓ Saved backtest results: {backtest_file}")


def main():
    """Run comprehensive regression tests"""
    tester = NewsMLRegressionTest()
    results = tester.run_comprehensive_test()
    tester.save_results(results)


if __name__ == "__main__":
    main()
