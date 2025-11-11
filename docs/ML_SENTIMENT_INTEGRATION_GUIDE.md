# ML Sentiment Analysis Integration Guide

Quick start guide for integrating ML-based sentiment analysis into BigBrotherAnalytics.

## Quick Start (5 minutes)

### 1. Install Dependencies (if not already installed)

```bash
# All dependencies already in pyproject.toml
pip install transformers torch accelerate
```

### 2. Test ML Sentiment Analyzer

```bash
# Test single text
python3 scripts/ml/sentiment_analyzer_ml.py --text "Apple stock surges on strong earnings"

# Test batch
python3 scripts/ml/sentiment_analyzer_ml.py --batch \
  "Tesla reports record deliveries" \
  "Amazon faces regulatory concerns" \
  "Microsoft announces new product"

# Run benchmark
python3 scripts/ml/sentiment_analyzer_ml.py --benchmark

# Test hybrid analyzer with fallback
python3 scripts/ml/sentiment_analyzer_ml.py --hybrid --text "Strong quarterly results"
```

### 3. Run Backtesting Comparison

```bash
# Compare ML vs keyword sentiment on labeled dataset
cd /home/muyiwa/Development/BigBrotherAnalytics
python3 -m pytest tests/test_ml_sentiment_comparison.py -v -s

# Run specific tests
python3 -m pytest tests/test_ml_sentiment_comparison.py::test_ml_sentiment_accuracy -v -s
python3 -m pytest tests/test_ml_sentiment_comparison.py::test_ml_vs_keyword_comparison -v -s
python3 -m pytest tests/test_ml_sentiment_comparison.py::test_full_dataset_comparison -v -s
```

### 4. Integrate into News Ingestion

Update `scripts/data_collection/news_ingestion.py`:

```python
# Add at top of file
try:
    from scripts.ml.sentiment_analyzer_ml import HybridSentimentAnalyzer
    ml_analyzer = HybridSentimentAnalyzer(use_gpu=True)
    HAS_ML_SENTIMENT = True
except Exception as e:
    logger.warning(f"ML sentiment not available: {e}")
    HAS_ML_SENTIMENT = False

# Replace simple_sentiment calls with:
if HAS_ML_SENTIMENT:
    score, label, source, confidence = ml_analyzer.analyze(text_to_analyze)
else:
    # Fallback to keyword-based
    score, label, _, _ = simple_sentiment(text_to_analyze)
    source = "keyword"
    confidence = 0.0

# Store in database with metadata
conn.execute("""
    INSERT INTO news_articles (
        ...,
        sentiment_score,
        sentiment_label,
        sentiment_source,
        ml_confidence
    ) VALUES (?, ?, ?, ?, ?, ...)
""", [..., score, label, source, confidence, ...])
```

## Integration Options

### Option 1: Python-Only (Recommended - Phase 1)

**Pros:**
- Simple integration
- Easy to update models
- GPU acceleration available
- No C++ changes needed

**Implementation:**
1. Use `HybridSentimentAnalyzer` in Python news ingestion
2. Pre-compute sentiment scores and store in DuckDB
3. C++ trading engine reads pre-computed scores

**Files to modify:**
- `scripts/data_collection/news_ingestion.py` (add ML analyzer)
- Database schema (add `sentiment_source` and `ml_confidence` columns)

### Option 2: ONNX C++ (Optional - Phase 2)

**Pros:**
- No Python runtime in production
- Faster inference (20-30ms vs 40ms)
- Single binary deployment

**Implementation:**
1. Export model to ONNX: `python3 scripts/ml/export_sentiment_model.py`
2. Implement C++ ONNX Runtime integration in `src/market_intelligence/ml_sentiment_analyzer.cppm`
3. Link against ONNX Runtime library

**Requirements:**
- ONNX Runtime C++ library (~200MB)
- WordPiece tokenizer implementation in C++
- CMake integration

## Performance Expectations

### Python ML Sentiment

| Metric | Single Article | Batch (20 articles) | Per-Article (batched) |
|--------|---------------|---------------------|----------------------|
| CPU | 60-80ms | 800ms | 40ms |
| GPU | 50-60ms | 400ms | 20ms |

### Keyword Sentiment (baseline)

| Metric | Single Article | Batch (20 articles) | Per-Article (batched) |
|--------|---------------|---------------------|----------------------|
| CPU | <1ms | ~10ms | <1ms |

### Accuracy Expectations

| Approach | Expected Accuracy |
|----------|------------------|
| Keyword-based (current) | 74.44% |
| ML (DistilRoBERTa) | 90-95% |
| Hybrid (ML + keyword fallback) | 85-90% |

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
ml_analyzer = MLSentimentAnalyzer(use_gpu=False)
```

### Model Not Available (Offline)

The hybrid analyzer will automatically fallback to keyword-based sentiment:

```python
hybrid = HybridSentimentAnalyzer()
# Will use ML if available, keyword-based otherwise
```

## Database Schema Updates

Add these columns to `news_articles` table:

```sql
ALTER TABLE news_articles
ADD COLUMN sentiment_source VARCHAR DEFAULT 'keyword',
ADD COLUMN ml_confidence DOUBLE DEFAULT 0.0,
ADD COLUMN sentiment_score_keyword DOUBLE DEFAULT 0.0;

-- Create index for querying by sentiment source
CREATE INDEX idx_sentiment_source ON news_articles(sentiment_source);
```

## Monitoring and Metrics

Track these metrics in production:

1. **Sentiment Source Distribution:**
   ```sql
   SELECT sentiment_source, COUNT(*) as count
   FROM news_articles
   GROUP BY sentiment_source;
   ```

2. **ML Confidence Distribution:**
   ```sql
   SELECT
       CASE
           WHEN ml_confidence >= 0.9 THEN 'high'
           WHEN ml_confidence >= 0.7 THEN 'medium'
           ELSE 'low'
       END as confidence_level,
       COUNT(*) as count
   FROM news_articles
   WHERE sentiment_source = 'ml'
   GROUP BY confidence_level;
   ```

3. **Sentiment Score Comparison (ML vs Keyword):**
   ```sql
   SELECT
       sentiment_score as ml_score,
       sentiment_score_keyword as keyword_score,
       ABS(sentiment_score - sentiment_score_keyword) as score_diff,
       title
   FROM news_articles
   WHERE sentiment_source = 'ml'
   ORDER BY score_diff DESC
   LIMIT 10;
   ```

## Next Steps

1. **Phase 1 (Week 1-2):**
   - [ ] Test ML analyzer: `python3 scripts/ml/sentiment_analyzer_ml.py --benchmark`
   - [ ] Run backtesting: `pytest tests/test_ml_sentiment_comparison.py -v -s`
   - [ ] Integrate into news ingestion
   - [ ] Update database schema
   - [ ] Deploy and monitor

2. **Phase 2 (Optional - Week 3+):**
   - [ ] Export model to ONNX: `python3 scripts/ml/export_sentiment_model.py`
   - [ ] Implement C++ ONNX Runtime integration
   - [ ] Performance testing and optimization
   - [ ] Gradual rollout to production

## Resources

- Model on Hugging Face: https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
- ONNX Runtime Documentation: https://onnxruntime.ai/docs/
- Full Implementation Proposal: `docs/ML_SENTIMENT_ANALYSIS_PROPOSAL.md`
