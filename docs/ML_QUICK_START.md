# Neural Network Quick Start Guide

**5-Minute Guide to Using BigBrotherAnalytics ML Libraries**

---

## TL;DR

```cpp
import bigbrother.ml.weight_loader;
import bigbrother.ml.neural_net_mkl;

// Load weights (one-time)
auto weights = bigbrother::ml::PricePredictorConfig::createLoader().load();

// Initialize predictor
bigbrother::ml::NeuralNetMKL predictor(weights);

// Prepare 60 features
std::array<float, 60> features = { /* your features */ };

// Get prediction
auto output = predictor.predict(features);  // [day_1%, day_5%, day_20%]
```

---

## 3 Engines Available

| Engine | Speed | Use When |
|--------|-------|----------|
| `NeuralNetMKL` | 227M/sec | Production (most stable) |
| `NeuralNetSIMD` (AVX-512) | **233M/sec** | Newest CPUs (fastest) |
| `NeuralNetSIMD` (AVX-2) | 200M/sec | Older CPUs |

**Recommendation:** Use `NeuralNetMKL` for production, it's battle-tested.

---

## Feature Extraction (60 Features Required)

### Quick Template

```cpp
bigbrother::market_intelligence::PriceFeatures features;

// 1. Identification (3)
features.symbol_encoded = 0.0f;  // 0-19 for top 20 symbols
features.sector_encoded = 0.0f;  // Sector ID
features.is_option = 0.0f;       // 0=stock, 1=option

// 2. Time (8) - Use std::chrono::system_clock::now()
features.hour_of_day = 14.0f;    // 0-23
features.day_of_week = 2.0f;     // 0=Mon, 6=Sun
// ... (see full docs for remaining)

// 3. Treasury Rates (7) - Use current FRED rates as DECIMALS
features.fed_funds_rate = 0.0525f;   // 5.25% as 0.0525
features.treasury_10yr = 0.0430f;    // 4.30% as 0.0430
// ...

// 4. Greeks (6) - Estimate or use Black-Scholes
features.delta = 0.5f;               // ATM delta for stocks
features.implied_volatility = 0.25f; // ~25% IV
// ...

// 5. Sentiment (2) - From news database or default to 0
features.avg_sentiment = 0.0f;   // -1 to 1
features.news_count = 0.0f;      // Count

// 6. Price (5) - From quote
features.close = quote.last;
features.volume = quote.volume;
// ...

// 7. Momentum (7) - Calculate from price history
features.return_1d = (price[0] - price[1]) / price[1];
features.rsi_14 = calculateRSI(price_history);
// ...

// 8. Volatility (4) - Technical indicators
features.atr_14 = calculateATR(price_history);
features.bb_position = (price - bb_lower) / (bb_upper - bb_lower);
// ...

// 9. Interaction (10) - Computed from base features
features.sentiment_momentum = features.avg_sentiment * features.return_5d;
features.rate_return = features.fed_funds_rate * features.return_20d;
// ...

// 10. Directionality (8) - From price patterns
features.price_direction = features.return_1d > 0 ? 1.0f : 0.0f;
features.volume_trend = features.volume_ratio > 1.0f ? 1.0f : 0.0f;
// ...

// Convert to array
auto input = features.toArray();
```

---

## Common Patterns

### Pattern 1: Initialize Once, Predict Many Times

```cpp
class MyStrategy {
    bigbrother::ml::NeuralNetMKL predictor_;

public:
    MyStrategy() {
        auto weights = bigbrother::ml::PricePredictorConfig::createLoader().load();
        predictor_ = bigbrother::ml::NeuralNetMKL(weights);
    }

    auto evaluateSymbol(Quote const& quote) -> Signal {
        auto features = extractFeatures(quote);
        auto prediction = predictor_.predict(features.toArray());

        if (prediction[0] > 0.02f) return Signal::BUY;   // >2% gain
        if (prediction[0] < -0.02f) return Signal::SELL; // >2% loss
        return Signal::HOLD;
    }
};
```

### Pattern 2: Compare Engines

```cpp
auto weights = bigbrother::ml::PricePredictorConfig::createLoader().load();
bigbrother::ml::NeuralNetMKL mkl(weights);
bigbrother::ml::NeuralNetSIMD simd(weights);

auto mkl_pred = mkl.predict(features);
auto simd_pred = simd.predict(features);

// Should be virtually identical (<0.000001% difference)
```

### Pattern 3: Custom Architecture

```cpp
auto weights = bigbrother::ml::WeightLoader::fromDirectory("my_model")
    .withArchitecture(60, {128, 64}, 3)  // 60→128→64→3
    .withNamingScheme("layer_{}_w.bin", "layer_{}_b.bin")
    .load();
```

---

## Build & Run

```bash
# Build
SKIP_CLANG_TIDY=1 cmake -G Ninja -B build && ninja -C build

# Run benchmark
./build/bin/benchmark_all_ml_engines

# Run tests
./build/bin/test_weight_loader
```

---

## File Locations

```
src/ml/weight_loader.cppm        - Fluent API loader
src/ml/neural_net_mkl.cppm       - MKL engine
src/ml/neural_net_simd.cppm      - SIMD engine
models/weights/*.bin             - Weight files (10 files, 230 KB)
```

---

## Key Numbers to Remember

- **60 features** required for input
- **3 outputs** (1-day%, 5-day%, 20-day% change)
- **58,947 parameters** (60→256→128→64→32→3)
- **230 KB** weight file size
- **233M predictions/sec** (AVX-512 SIMD)
- **53% accuracy** (current model - needs improvement)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | `rm -rf build && cmake -G Ninja -B build` |
| Weight files missing | Run `uv run python scripts/ml/export_trained_weights.py` |
| NaN predictions | Check for NaN/Inf in input features |
| Slow inference | Use Release build: `-DCMAKE_BUILD_TYPE=Release` |
| Low accuracy (<50%) | Retrain with proper feature normalization |

---

## Next Steps

1. **Read Full Docs:** [NEURAL_NETWORK_ARCHITECTURE.md](NEURAL_NETWORK_ARCHITECTURE.md)
2. **Improve Accuracy:** [TRAINING_REPORT.md](../TRAINING_REPORT.md)
3. **See Examples:** [benchmark_all_ml_engines.cpp](../benchmarks/benchmark_all_ml_engines.cpp)

---

**Questions?** Check the full documentation or see [examples/](../examples/).
