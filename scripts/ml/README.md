# ML Sentiment Analysis for Financial News

Machine learning-based sentiment analysis system for BigBrotherAnalytics using DistilRoBERTa fine-tuned on financial news.

## Quick Start

### 1. Verify Setup

```bash
python3 scripts/ml/verify_ml_sentiment_setup.py
```

This will check:
- Python version (3.8+)
- Required dependencies (torch, transformers, accelerate)
- ML modules can be imported
- Model can be loaded
- Inference works correctly

### 2. Install Dependencies (if needed)

```bash
# Install all ML dependencies
pip install transformers torch accelerate

# Or use the project's pyproject.toml
pip install -e .
```

### 3. Test the Analyzer

```bash
# Test single text
python3 scripts/ml/sentiment_analyzer_ml.py \
  --text "Apple stock surges on strong earnings"

# Test batch processing
python3 scripts/ml/sentiment_analyzer_ml.py --batch \
  "Tesla reports record deliveries" \
  "Amazon faces regulatory concerns" \
  "Microsoft announces new product"

# Run performance benchmark
python3 scripts/ml/sentiment_analyzer_ml.py --benchmark

# Test hybrid analyzer (ML + keyword fallback)
python3 scripts/ml/sentiment_analyzer_ml.py --hybrid \
  --text "Strong quarterly results exceed expectations"
```

### 4. Run Backtesting

```bash
# Full comparison (ML vs keyword on 90 labeled samples)
pytest tests/test_ml_sentiment_comparison.py::test_full_dataset_comparison -v -s

# Accuracy test
pytest tests/test_ml_sentiment_comparison.py::test_ml_sentiment_accuracy -v -s

# Performance test
pytest tests/test_ml_sentiment_comparison.py::test_ml_sentiment_performance -v -s

# Hybrid fallback test
pytest tests/test_ml_sentiment_comparison.py::test_hybrid_fallback_strategy -v -s
```

## Files Overview

### Implementation (Production Code)

| File | Description | Lines | Purpose |
|------|-------------|-------|---------|
| `sentiment_analyzer_ml.py` | Main ML analyzer | 600+ | Core ML sentiment implementation |
| `export_sentiment_model.py` | ONNX export | 300+ | Export model for C++ (optional) |
| `verify_ml_sentiment_setup.py` | Setup verification | 250+ | Diagnostic and verification tool |

### Documentation

| File | Description | Size | Purpose |
|------|-------------|------|---------|
| `docs/ML_SENTIMENT_ANALYSIS_PROPOSAL.md` | Full proposal | 19KB | Comprehensive implementation plan |
| `docs/ML_SENTIMENT_INTEGRATION_GUIDE.md` | Quick start | 7KB | Integration instructions |
| `docs/ML_SENTIMENT_RESEARCH_SUMMARY.md` | Research report | 11KB | Research findings and recommendations |
| `scripts/ml/README.md` | This file | 6KB | Quick reference guide |

### Testing

| File | Description | Lines | Purpose |
|------|-------------|-------|---------|
| `tests/test_ml_sentiment_comparison.py` | Backtesting suite | 500+ | ML vs keyword comparison |

## Architecture

### MLSentimentAnalyzer

Core ML sentiment analyzer using DistilRoBERTa:

```python
from scripts.ml.sentiment_analyzer_ml import MLSentimentAnalyzer

# Initialize (downloads model on first run ~328MB)
analyzer = MLSentimentAnalyzer(use_gpu=True)

# Analyze single text
result = analyzer.analyze("Apple stock surges on earnings beat")
print(f"Score: {result.score:.3f}")      # -1.0 to 1.0
print(f"Label: {result.label}")          # positive/negative/neutral
print(f"Confidence: {result.confidence:.3f}")  # 0.0 to 1.0

# Analyze batch (efficient)
texts = ["text1", "text2", "text3"]
results = analyzer.analyze_batch(texts, batch_size=16)

# Benchmark performance
metrics = analyzer.benchmark(num_samples=100)
print(f"Avg latency: {metrics['batch_avg_ms']:.2f}ms")
```

### HybridSentimentAnalyzer

Hybrid analyzer with automatic fallback:

```python
from scripts.ml.sentiment_analyzer_ml import HybridSentimentAnalyzer

# Initialize (tries ML, falls back to keyword if unavailable)
hybrid = HybridSentimentAnalyzer(use_gpu=True)

# Analyze (uses best available method)
score, label, source, confidence = hybrid.analyze("Strong earnings")
print(f"Source: {source}")  # "ml", "keyword", or "fallback"

# Batch analysis
results = hybrid.analyze_batch(["text1", "text2"])
```

## Performance Characteristics

### Accuracy

| Approach | Expected Accuracy | Precision | Recall | F1-Score |
|----------|------------------|-----------|--------|----------|
| **ML (DistilRoBERTa)** | 95-98% | 95-98% | 95-98% | 95-98% |
| Keyword-based (current) | 74.44% | 70-75% | 70-75% | 70-75% |
| **Improvement** | **+20-24%** | **+20-25%** | **+20-25%** | **+20-25%** |

### Latency

| Mode | Single Article | Batch (20 articles) | Per-Article (batched) |
|------|---------------|---------------------|----------------------|
| CPU | 60-80ms | 800ms | **40ms** |
| GPU | 50-60ms | 400ms | **20ms** |

**Target:** <100ms per article ✅

### Memory

| Component | Memory Usage |
|-----------|-------------|
| Model loaded | ~500MB |
| ONNX Runtime (C++) | ~200MB |
| Keyword-based | ~1KB |

## Integration Examples

### Example 1: Simple Integration

```python
from scripts.ml.sentiment_analyzer_ml import HybridSentimentAnalyzer

# Initialize once at startup
analyzer = HybridSentimentAnalyzer(use_gpu=True)

# Use in news ingestion
def process_article(title, description):
    text = f"{title} {description}"
    score, label, source, confidence = analyzer.analyze(text)

    return {
        'sentiment_score': score,
        'sentiment_label': label,
        'sentiment_source': source,
        'ml_confidence': confidence
    }
```

### Example 2: Batch Processing

```python
from scripts.ml.sentiment_analyzer_ml import MLSentimentAnalyzer

analyzer = MLSentimentAnalyzer(use_gpu=True)

# Process articles in batches for efficiency
articles = fetch_news_articles()  # List of articles
texts = [f"{a['title']} {a['description']}" for a in articles]

# Batch process (much faster than one-by-one)
results = analyzer.analyze_batch(texts, batch_size=16)

# Store results
for article, result in zip(articles, results):
    article['sentiment_score'] = result.score
    article['sentiment_label'] = result.label
    article['ml_confidence'] = result.confidence
```

### Example 3: With Fallback

```python
from scripts.ml.sentiment_analyzer_ml import MLSentimentAnalyzer
from scripts.data_collection.news_ingestion import simple_sentiment

try:
    # Try ML first
    ml_analyzer = MLSentimentAnalyzer(use_gpu=True)
    if ml_analyzer.is_available():
        result = ml_analyzer.analyze(text)
        score = result.score
        source = "ml"
    else:
        raise RuntimeError("ML not available")
except Exception as e:
    # Fallback to keyword-based
    score, label, _, _ = simple_sentiment(text)
    source = "keyword"
```

## Model Information

### Selected Model

**Name:** `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`

**Details:**
- Architecture: DistilRoBERTa (6 layers, 768 dim, 12 heads)
- Parameters: 82M
- Training data: Financial PhraseBank (4,840 financial news sentences)
- Accuracy: 98.23% on Financial PhraseBank
- Speed: 2x faster than RoBERTa-base
- Size: 328MB

**Download:** Automatic on first use (cached in `~/.cache/huggingface/`)

**Hugging Face:** https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis

## Advanced Usage

### Export to ONNX (for C++ deployment)

```bash
# Export model to ONNX format
python3 scripts/ml/export_sentiment_model.py

# Output: models/sentiment_distilroberta.onnx (~328MB)
# Tokenizer: models/tokenizer/
```

### Custom Model

```python
# Use different model
analyzer = MLSentimentAnalyzer(
    model_name="ProsusAI/finbert",  # Alternative model
    use_gpu=True
)
```

### Force CPU

```python
# Disable GPU (useful for testing)
analyzer = MLSentimentAnalyzer(use_gpu=False)
```

### Custom Cache Directory

```python
# Use custom model cache
analyzer = MLSentimentAnalyzer(
    cache_dir="/path/to/cache",
    use_gpu=True
)
```

## Troubleshooting

### Model Download Fails

```bash
# Manually download model
python3 -c "
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
)
print('Model downloaded successfully')
"
```

### GPU Out of Memory

```python
# Force CPU usage
analyzer = MLSentimentAnalyzer(use_gpu=False)
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade transformers torch accelerate

# Verify installation
python3 scripts/ml/verify_ml_sentiment_setup.py
```

### NumPy Compatibility Issues

```bash
# Update NumPy
pip install --upgrade numpy

# Or use specific version
pip install numpy==1.26.0
```

## Testing

### Unit Tests

```bash
# All tests
pytest tests/test_ml_sentiment_comparison.py -v -s

# Specific test
pytest tests/test_ml_sentiment_comparison.py::test_ml_sentiment_accuracy -v -s
```

### Manual Testing

```bash
# Test analyzer directly
python3 scripts/ml/sentiment_analyzer_ml.py --text "Strong earnings beat estimates"

# Run benchmark
python3 scripts/ml/sentiment_analyzer_ml.py --benchmark
```

### Verification

```bash
# Full verification
python3 scripts/ml/verify_ml_sentiment_setup.py
```

## Production Deployment

### Phase 1: Python-Only (Recommended)

1. Install dependencies in production environment
2. Integrate `HybridSentimentAnalyzer` into news ingestion
3. Update database schema (add `sentiment_source`, `ml_confidence`)
4. Deploy and monitor

### Phase 2: ONNX C++ (Optional)

1. Export model: `python3 scripts/ml/export_sentiment_model.py`
2. Install ONNX Runtime C++ library
3. Implement C++ integration in `src/market_intelligence/ml_sentiment_analyzer.cppm`
4. Link against ONNX Runtime in CMakeLists.txt
5. Deploy and benchmark

## Resources

### Documentation
- Full Proposal: `docs/ML_SENTIMENT_ANALYSIS_PROPOSAL.md`
- Integration Guide: `docs/ML_SENTIMENT_INTEGRATION_GUIDE.md`
- Research Summary: `docs/ML_SENTIMENT_RESEARCH_SUMMARY.md`

### External Links
- Model: https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
- Transformers: https://huggingface.co/docs/transformers/
- ONNX Runtime: https://onnxruntime.ai/docs/

### Papers
- FinBERT: arXiv:1908.10063
- DistilBERT: arXiv:1910.01108
- RoBERTa: arXiv:1907.11692

## Support

For issues or questions:
1. Check the verification script: `python3 scripts/ml/verify_ml_sentiment_setup.py`
2. Review the integration guide: `docs/ML_SENTIMENT_INTEGRATION_GUIDE.md`
3. See troubleshooting section above

---

**Version:** 1.0
**Last Updated:** 2025-11-10
**Status:** Ready for Production
