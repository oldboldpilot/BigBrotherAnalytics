#!/usr/bin/env python3
"""
ML Sentiment Setup Verification Script

Verifies that all components for ML sentiment analysis are correctly
installed and configured. Provides detailed diagnostics and next steps.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10

Usage:
    python3 scripts/ml/verify_ml_sentiment_setup.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


def check_python_version():
    """Check Python version"""
    print("=" * 70)
    print("PYTHON VERSION CHECK")
    print("=" * 70)

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python version: {version_str}")

    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version OK (requires 3.8+)")
        return True
    else:
        print("✗ Python version too old (requires 3.8+)")
        return False


def check_dependencies():
    """Check if ML dependencies are installed"""
    print("\n" + "=" * 70)
    print("DEPENDENCY CHECK")
    print("=" * 70)

    deps = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'accelerate': 'Accelerate (GPU support)',
    }

    results = {}
    for module_name, display_name in deps.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name}: {version}")
            results[module_name] = True
        except (ImportError, AttributeError) as e:
            print(f"✗ {display_name}: Not installed ({e})")
            results[module_name] = False

    return all(results.values())


def check_ml_modules():
    """Check if ML sentiment modules can be imported"""
    print("\n" + "=" * 70)
    print("ML SENTIMENT MODULES CHECK")
    print("=" * 70)

    try:
        from scripts.ml.sentiment_analyzer_ml import (
            MLSentimentAnalyzer,
            HybridSentimentAnalyzer,
            SentimentResult
        )
        print("✓ MLSentimentAnalyzer import successful")
        print("✓ HybridSentimentAnalyzer import successful")
        print("✓ SentimentResult import successful")
        return True
    except Exception as e:
        print(f"✗ ML sentiment modules failed to import: {e}")
        return False


def check_model_availability():
    """Check if ML model can be loaded"""
    print("\n" + "=" * 70)
    print("ML MODEL AVAILABILITY CHECK")
    print("=" * 70)

    try:
        from scripts.ml.sentiment_analyzer_ml import MLSentimentAnalyzer

        print("Initializing ML sentiment analyzer (may take 30-60s on first run)...")
        analyzer = MLSentimentAnalyzer(use_gpu=False)  # Force CPU for compatibility

        if analyzer.is_available():
            print("✓ ML sentiment model loaded successfully")
            print(f"  Device: {analyzer.device}")
            print(f"  Model: {analyzer.model_name}")

            # Test inference
            print("\nTesting inference...")
            result = analyzer.analyze("Apple reports strong quarterly earnings")
            print(f"✓ Inference successful")
            print(f"  Score: {result.score:.3f}")
            print(f"  Label: {result.label}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Latency: {result.latency_ms:.2f}ms")

            return True
        else:
            print("✗ ML sentiment model failed to load")
            return False

    except Exception as e:
        print(f"✗ ML model check failed: {e}")
        return False


def check_hybrid_analyzer():
    """Check if hybrid analyzer works with fallback"""
    print("\n" + "=" * 70)
    print("HYBRID ANALYZER CHECK (WITH FALLBACK)")
    print("=" * 70)

    try:
        from scripts.ml.sentiment_analyzer_ml import HybridSentimentAnalyzer

        analyzer = HybridSentimentAnalyzer(use_gpu=False)
        print("✓ HybridSentimentAnalyzer initialized")

        # Test analysis
        score, label, source, confidence = analyzer.analyze(
            "Tesla reports record deliveries"
        )
        print(f"\nTest analysis:")
        print(f"  Text: 'Tesla reports record deliveries'")
        print(f"  Score: {score:.3f}")
        print(f"  Label: {label}")
        print(f"  Source: {source}")
        print(f"  Confidence: {confidence:.3f}")

        if source == "ml":
            print("✓ Using ML sentiment")
        elif source == "keyword":
            print("! Using keyword fallback (ML not available)")
        else:
            print("! Using neutral fallback")

        return True

    except Exception as e:
        print(f"✗ Hybrid analyzer check failed: {e}")
        return False


def check_test_suite():
    """Check if test suite can be imported"""
    print("\n" + "=" * 70)
    print("TEST SUITE CHECK")
    print("=" * 70)

    try:
        from tests.test_ml_sentiment_comparison import (
            test_ml_sentiment_accuracy,
            test_ml_vs_keyword_comparison
        )
        print("✓ Test suite imports successfully")
        print("  Run with: pytest tests/test_ml_sentiment_comparison.py -v -s")
        return True
    except Exception as e:
        print(f"✗ Test suite import failed: {e}")
        return False


def print_next_steps(all_checks_passed):
    """Print recommended next steps"""
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    if all_checks_passed:
        print("\n✓ All checks passed! ML sentiment analysis is ready to use.\n")
        print("Recommended next steps:")
        print()
        print("1. Run performance benchmark:")
        print("   python3 scripts/ml/sentiment_analyzer_ml.py --benchmark")
        print()
        print("2. Test ML vs keyword comparison:")
        print("   pytest tests/test_ml_sentiment_comparison.py::test_ml_vs_keyword_comparison -v -s")
        print()
        print("3. Run full backtesting:")
        print("   pytest tests/test_ml_sentiment_comparison.py::test_full_dataset_comparison -v -s")
        print()
        print("4. Integrate into news ingestion:")
        print("   See docs/ML_SENTIMENT_INTEGRATION_GUIDE.md")
        print()
    else:
        print("\n✗ Some checks failed. Follow the steps below to fix:\n")

        # Check which dependencies are missing
        try:
            import torch
        except:
            print("1. Install PyTorch:")
            print("   pip install torch")
            print()

        try:
            import transformers
        except:
            print("2. Install Transformers:")
            print("   pip install transformers")
            print()

        try:
            import accelerate
        except:
            print("3. Install Accelerate:")
            print("   pip install accelerate")
            print()

        print("Or install all dependencies at once:")
        print("   pip install transformers torch accelerate")
        print()
        print("Then re-run this verification script.")
        print()


def main():
    """Main verification flow"""
    print("\n" + "=" * 70)
    print("ML SENTIMENT ANALYSIS SETUP VERIFICATION")
    print("=" * 70)
    print()

    results = {}

    # Run all checks
    results['python'] = check_python_version()
    results['deps'] = check_dependencies()
    results['modules'] = check_ml_modules()

    # Only check model if dependencies are available
    if results['deps'] and results['modules']:
        results['model'] = check_model_availability()
        results['hybrid'] = check_hybrid_analyzer()
        results['tests'] = check_test_suite()
    else:
        results['model'] = False
        results['hybrid'] = False
        results['tests'] = False
        print("\n⚠ Skipping model checks due to missing dependencies")

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    print(f"\nChecks passed: {passed}/{total}")
    for check, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")

    all_passed = all(results.values())

    # Next steps
    print_next_steps(all_passed)

    print("=" * 70)
    print()

    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
