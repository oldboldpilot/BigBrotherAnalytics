#!/usr/bin/env python3
"""
ML-Based Sentiment Analyzer for Financial News

Uses DistilRoBERTa fine-tuned on financial news for high-accuracy
sentiment analysis. Provides graceful fallback to keyword-based
sentiment if model is unavailable.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 5+: ML Sentiment Integration

Model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
Accuracy: 98.23% on Financial PhraseBank dataset
"""

import os
import sys
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Try to import ML dependencies
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline
    )
    HAS_TRANSFORMERS = True
except (ImportError, AttributeError) as e:
    HAS_TRANSFORMERS = False
    logger.warning(f"transformers library not available - ML sentiment disabled: {e}")


@dataclass
class SentimentResult:
    """Sentiment analysis result with metadata"""
    score: float  # -1.0 to 1.0
    label: str    # "positive", "negative", "neutral"
    confidence: float  # 0.0 to 1.0
    source: str   # "ml", "keyword", or "neutral"
    latency_ms: float  # Processing time in milliseconds
    raw_logits: Optional[List[float]] = None


class MLSentimentAnalyzer:
    """
    ML-based sentiment analyzer using DistilRoBERTa

    Features:
    - High accuracy (98.23% on financial news)
    - Batch processing for efficiency
    - GPU acceleration when available
    - Graceful fallback to CPU
    - Caching for repeated texts
    """

    # Model configuration
    DEFAULT_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

    # Label mapping (model outputs positive/negative/neutral)
    LABEL_TO_SCORE = {
        "positive": 1.0,
        "negative": -1.0,
        "neutral": 0.0
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_gpu: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize ML sentiment analyzer

        Args:
            model_name: Hugging Face model ID (default: DistilRoBERTa-financial)
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            use_gpu: Whether to try using GPU if available
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_dir = cache_dir
        self.device = self._detect_device(device, use_gpu)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._is_available = False

        # Try to load model
        self._load_model()

    def _detect_device(self, device: Optional[str], use_gpu: bool) -> str:
        """Detect best available device"""
        if device:
            return device

        if not HAS_TRANSFORMERS:
            return "cpu"

        if use_gpu and torch.cuda.is_available():
            logger.info("GPU detected - using CUDA for inference")
            return "cuda"

        logger.info("Using CPU for inference")
        return "cpu"

    def _load_model(self) -> None:
        """Load model and tokenizer"""
        if not HAS_TRANSFORMERS:
            logger.error("transformers library not available")
            return

        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            start_time = time.time()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            # Move to device
            try:
                self.model.to(self.device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device == "cuda":
                    logger.warning("GPU OOM - falling back to CPU")
                    self.device = "cpu"
                    self.model.to(self.device)
                else:
                    raise

            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                top_k=None  # Return all classes with scores
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Model: {self.model_name}")

            self._is_available = True

        except Exception as e:
            logger.error(f"Failed to load ML sentiment model: {e}")
            self._is_available = False

    def is_available(self) -> bool:
        """Check if ML model is loaded and ready"""
        return self._is_available

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of single text

        Args:
            text: Text to analyze (headline + description)

        Returns:
            SentimentResult with score, label, confidence
        """
        if not self._is_available:
            return SentimentResult(
                score=0.0,
                label="neutral",
                confidence=0.0,
                source="unavailable",
                latency_ms=0.0
            )

        if not text or not text.strip():
            return SentimentResult(
                score=0.0,
                label="neutral",
                confidence=1.0,
                source="ml",
                latency_ms=0.0
            )

        try:
            start_time = time.time()

            # Run inference
            results = self.pipeline(text[:512])  # Truncate to max length

            # Parse results (returns list of dicts with 'label' and 'score')
            # Find the prediction with highest score
            best_pred = max(results[0], key=lambda x: x['score'])
            pred_label = best_pred['label'].lower()
            confidence = best_pred['score']

            # Map label to score (-1 to 1)
            score = self.LABEL_TO_SCORE.get(pred_label, 0.0)

            latency_ms = (time.time() - start_time) * 1000

            return SentimentResult(
                score=score,
                label=pred_label,
                confidence=confidence,
                source="ml",
                latency_ms=latency_ms,
                raw_logits=[r['score'] for r in results[0]]
            )

        except Exception as e:
            logger.error(f"ML sentiment analysis failed: {e}")
            return SentimentResult(
                score=0.0,
                label="neutral",
                confidence=0.0,
                source="error",
                latency_ms=0.0
            )

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts (efficient batching)

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for inference (default: 16)

        Returns:
            List of SentimentResult in same order as input
        """
        if not self._is_available:
            return [
                SentimentResult(
                    score=0.0,
                    label="neutral",
                    confidence=0.0,
                    source="unavailable",
                    latency_ms=0.0
                )
                for _ in texts
            ]

        if not texts:
            return []

        try:
            start_time = time.time()

            # Truncate texts to max length
            truncated_texts = [t[:512] if t else "" for t in texts]

            # Run batch inference
            all_results = self.pipeline(
                truncated_texts,
                batch_size=batch_size,
                truncation=True,
                max_length=512
            )

            total_latency = (time.time() - start_time) * 1000
            avg_latency = total_latency / len(texts)

            # Parse results
            parsed_results = []
            for results in all_results:
                # Find prediction with highest score
                best_pred = max(results, key=lambda x: x['score'])
                pred_label = best_pred['label'].lower()
                confidence = best_pred['score']
                score = self.LABEL_TO_SCORE.get(pred_label, 0.0)

                parsed_results.append(
                    SentimentResult(
                        score=score,
                        label=pred_label,
                        confidence=confidence,
                        source="ml",
                        latency_ms=avg_latency,
                        raw_logits=[r['score'] for r in results]
                    )
                )

            logger.info(f"Batch analysis: {len(texts)} texts in {total_latency:.1f}ms "
                       f"({avg_latency:.1f}ms/text)")

            return parsed_results

        except Exception as e:
            logger.error(f"Batch ML sentiment analysis failed: {e}")
            return [
                SentimentResult(
                    score=0.0,
                    label="neutral",
                    confidence=0.0,
                    source="error",
                    latency_ms=0.0
                )
                for _ in texts
            ]

    def benchmark(self, num_samples: int = 100) -> dict:
        """
        Benchmark model performance

        Args:
            num_samples: Number of samples to test

        Returns:
            Dictionary with benchmark metrics
        """
        if not self._is_available:
            return {"error": "Model not available"}

        # Sample texts from different sentiments
        test_texts = [
            "Company reports strong quarterly earnings, stock surges",
            "Weak guidance disappoints investors, shares tumble",
            "Company announces new product launch next month",
        ] * (num_samples // 3)

        # Warmup
        self.analyze(test_texts[0])

        # Single inference benchmark
        single_times = []
        for text in test_texts[:10]:
            result = self.analyze(text)
            single_times.append(result.latency_ms)

        # Batch inference benchmark
        batch_start = time.time()
        batch_results = self.analyze_batch(test_texts)
        batch_time = (time.time() - batch_start) * 1000

        return {
            "device": self.device,
            "model": self.model_name,
            "single_avg_ms": sum(single_times) / len(single_times),
            "single_p50_ms": sorted(single_times)[len(single_times) // 2],
            "single_p95_ms": sorted(single_times)[int(len(single_times) * 0.95)],
            "batch_total_ms": batch_time,
            "batch_size": len(test_texts),
            "batch_avg_ms": batch_time / len(test_texts),
            "is_gpu": self.device == "cuda"
        }


# ============================================================================
# Hybrid Sentiment Analyzer (ML + Keyword Fallback)
# ============================================================================

class HybridSentimentAnalyzer:
    """
    Hybrid sentiment analyzer with ML-first and keyword fallback

    Fallback chain:
    1. ML sentiment (DistilRoBERTa) - highest accuracy
    2. Keyword-based sentiment - fast fallback
    3. Neutral - ultimate fallback
    """

    def __init__(
        self,
        ml_model_name: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize hybrid analyzer

        Args:
            ml_model_name: Hugging Face model ID for ML analyzer
            use_gpu: Whether to use GPU for ML analyzer
        """
        # Try to initialize ML analyzer
        self.ml_analyzer = None
        try:
            self.ml_analyzer = MLSentimentAnalyzer(
                model_name=ml_model_name,
                use_gpu=use_gpu
            )
            if self.ml_analyzer.is_available():
                logger.info("Hybrid analyzer: ML sentiment enabled")
            else:
                logger.warning("Hybrid analyzer: ML sentiment unavailable, using keyword-only")
        except Exception as e:
            logger.error(f"Failed to initialize ML analyzer: {e}")

        # Import keyword-based fallback
        try:
            from scripts.data_collection.news_ingestion import simple_sentiment
            self.keyword_analyzer = simple_sentiment
            self.has_keyword = True
        except ImportError:
            logger.warning("Keyword sentiment not available")
            self.has_keyword = False

    def analyze(self, text: str) -> Tuple[float, str, str, float]:
        """
        Analyze sentiment with fallback strategy

        Args:
            text: Text to analyze

        Returns:
            Tuple of (score, label, source, confidence)
        """
        # Try ML first
        if self.ml_analyzer and self.ml_analyzer.is_available():
            result = self.ml_analyzer.analyze(text)
            if result.source == "ml":
                return (result.score, result.label, "ml", result.confidence)

        # Fallback to keyword-based
        if self.has_keyword:
            score, label, _, _ = self.keyword_analyzer(text)
            return (score, label, "keyword", 0.0)

        # Ultimate fallback: neutral
        return (0.0, "neutral", "fallback", 0.0)

    def analyze_batch(self, texts: List[str]) -> List[Tuple[float, str, str, float]]:
        """
        Analyze batch with fallback strategy

        Args:
            texts: List of texts to analyze

        Returns:
            List of (score, label, source, confidence) tuples
        """
        # Try ML batch first
        if self.ml_analyzer and self.ml_analyzer.is_available():
            results = self.ml_analyzer.analyze_batch(texts)
            return [
                (r.score, r.label, r.source, r.confidence)
                for r in results
            ]

        # Fallback to keyword-based
        if self.has_keyword:
            return [
                (*self.keyword_analyzer(text)[:2], "keyword", 0.0)
                for text in texts
            ]

        # Ultimate fallback: neutral
        return [(0.0, "neutral", "fallback", 0.0) for _ in texts]


# ============================================================================
# CLI for Testing
# ============================================================================

def main():
    """Test ML sentiment analyzer"""
    import argparse

    parser = argparse.ArgumentParser(description="ML Sentiment Analysis for Financial News")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--batch", type=str, nargs="+", help="Batch of texts to analyze")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid analyzer with fallback")

    args = parser.parse_args()

    if args.hybrid:
        print("\n" + "=" * 70)
        print("Hybrid Sentiment Analyzer (ML + Keyword Fallback)")
        print("=" * 70)

        analyzer = HybridSentimentAnalyzer(use_gpu=not args.no_gpu)

        if args.text:
            score, label, source, confidence = analyzer.analyze(args.text)
            print(f"\nText: {args.text}")
            print(f"Score: {score:.3f}")
            print(f"Label: {label}")
            print(f"Source: {source}")
            print(f"Confidence: {confidence:.3f}")

        return

    # Standard ML analyzer
    print("\n" + "=" * 70)
    print("ML Sentiment Analyzer - DistilRoBERTa Financial")
    print("=" * 70)

    analyzer = MLSentimentAnalyzer(use_gpu=not args.no_gpu)

    if not analyzer.is_available():
        print("\nERROR: ML model not available!")
        print("Install dependencies: pip install transformers torch accelerate")
        return

    if args.benchmark:
        print("\nRunning benchmark...")
        results = analyzer.benchmark(num_samples=100)
        print("\nBenchmark Results:")
        print(f"  Device: {results['device']}")
        print(f"  Model: {results['model']}")
        print(f"  Single inference (avg): {results['single_avg_ms']:.2f}ms")
        print(f"  Single inference (p50): {results['single_p50_ms']:.2f}ms")
        print(f"  Single inference (p95): {results['single_p95_ms']:.2f}ms")
        print(f"  Batch inference (total): {results['batch_total_ms']:.2f}ms")
        print(f"  Batch size: {results['batch_size']}")
        print(f"  Batch inference (avg/text): {results['batch_avg_ms']:.2f}ms")
        print(f"  GPU enabled: {results['is_gpu']}")
        return

    if args.text:
        result = analyzer.analyze(args.text)
        print(f"\nText: {args.text}")
        print(f"Score: {result.score:.3f}")
        print(f"Label: {result.label}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Source: {result.source}")
        print(f"Latency: {result.latency_ms:.2f}ms")
        return

    if args.batch:
        results = analyzer.analyze_batch(args.batch)
        print(f"\nBatch Analysis ({len(results)} texts):")
        for i, result in enumerate(results):
            print(f"\n  Text {i+1}: {args.batch[i][:60]}...")
            print(f"  Score: {result.score:.3f} | Label: {result.label} | "
                  f"Confidence: {result.confidence:.3f}")
        return

    # Default: analyze sample financial headlines
    print("\nAnalyzing sample financial headlines...\n")

    samples = [
        "Apple stock surges 5% on strong quarterly earnings beat",
        "Tesla reports record deliveries, shares rally in after-hours trading",
        "Netflix loses subscribers for first time in decade, stock plunges 20%",
        "Amazon faces antitrust probe, shares tumble on regulatory concerns",
        "Microsoft announces new product launch event for September",
    ]

    results = analyzer.analyze_batch(samples)

    for text, result in zip(samples, results):
        print(f"Text: {text}")
        print(f"  Score: {result.score:+.3f} | Label: {result.label:<8} | "
              f"Confidence: {result.confidence:.3f} | Latency: {result.latency_ms:.1f}ms")
        print()


if __name__ == "__main__":
    main()
