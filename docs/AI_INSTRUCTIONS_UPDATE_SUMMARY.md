# AI Instructions & Configuration Update Summary

**Date**: 2025-11-10
**Author**: Olumuyiwa Oluwasanmi
**Purpose**: Document updates to AI assistant configurations for news ingestion system

---

## Overview

Updated AI assistant configurations (GitHub Copilot, Claude Code, etc.) to include comprehensive documentation about the news ingestion system, C++23 module architecture, build system requirements, and error handling patterns.

---

## Files Created/Modified

### 1. NEW: docs/AI_CONTEXT.md (Created)

**Purpose**: Comprehensive context document for AI assistants
**Size**: ~50KB
**Location**: `/home/muyiwa/Development/BigBrotherAnalytics/docs/AI_CONTEXT.md`

**Contents**:
1. **Project Overview** - Technology stack, core technologies
2. **News Ingestion System** - Complete architecture and implementation details
   - C++ sentiment analyzer (60+ keywords, 100K articles/sec)
   - NewsAPI collector with circuit breaker
   - Python bindings via pybind11
   - Database schema and integration
3. **Module Architecture** - Dependency graph and module system
4. **Build System** - CMake with Ninja, clang-tidy enforcement
5. **Error Handling Patterns** - Result<T> type, std::unexpected usage
6. **Coding Standards** - Trailing return types, [[nodiscard]], Rule of Five
7. **Python Bindings** - pybind11 patterns and library path setup
8. **Integration Points** - News → Dashboard, Daily automation
9. **Testing Strategy** - C++ and Python test patterns
10. **Common Workflows** - Adding modules, troubleshooting

**Key Sections for AI Assistants**:
- Module import syntax: `import bigbrother.market_intelligence.sentiment;`
- Build order: `ninja utils → market_intelligence → news_ingestion_py`
- Error handling: Always use `std::unexpected(Error::make(...))`
- Python usage: `uv run python` (NOT bare python or pip)

---

### 2. UPDATED: .github/copilot-instructions.md (Modified)

**Purpose**: GitHub Copilot configuration and quick reference
**Location**: `/home/muyiwa/Development/BigBrotherAnalytics/.github/copilot-instructions.md`

**Changes Made**:

#### A. Essential Documentation Section (Line 66-75)
**Added**:
- Link to `AI_CONTEXT.md` (NEW - comprehensive AI context)
- Link to `NEWS_INGESTION_SYSTEM.md` (news architecture)
- Link to `NEWS_INGESTION_QUICKSTART.md` (quick start guide)

#### B. Market Intelligence Engine Section (Line 89-97)
**Added**:
- `sentiment_analyzer.cppm` - Keyword-based sentiment analysis
- `news_ingestion.cppm` - NewsAPI client with circuit breaker
- Performance metrics: 100K articles/sec sentiment analysis

#### C. Data Sources Section (Line 160-172)
**Added**:
- **NewsAPI** integration details:
  - 100 requests/day (free tier)
  - 7-day lookback period
  - 20 articles per request
  - Keyword-based sentiment (no ML dependencies)
  - Circuit breaker protection (5 failures → 60s timeout)
- Updated DuckDB storage to include news articles

#### D. Coding Standards Section (Line 171-263)
**Major Expansion**:
- **C++23 Module System** documentation:
  - Compiler: Clang 21.1.5
  - Build tool: Ninja (REQUIRED)
  - Module import syntax examples
  - Module structure template
- **Error Handling Pattern**:
  - ALWAYS use `std::unexpected(Error::make(...))`
  - NEVER use raw `Error{}` construction
  - NEVER throw exceptions for expected failures
- **Style Requirements**:
  - Trailing return types (MANDATORY, ERROR level)
  - [[nodiscard]] on getters/queries
  - Rule of Five enforcement
  - const correctness rules

#### E. Build System Section (Line 492-527)
**Added**:
- Note that `SKIP_CLANG_TIDY=1` no longer works
- `ninja news_ingestion_py` build target
- **clang-tidy Enforcement** subsection:
  - Status: MANDATORY, runs automatically
  - Cannot be skipped
  - Key checks: trailing return types, Rule of Five (ERROR level)
  - System headers excluded

#### F. Module Dependencies Section (Line 518-586)
**Major Addition**:
- Complete module dependency graph (ASCII art)
- **News Ingestion System** subsection:
  - Architecture diagram (NewsAPI → C++ → DuckDB → Dashboard)
  - Key modules listing
  - Build commands (sequential order)
  - Python usage examples
  - Database schema

#### G. Troubleshooting Section (Line 665-676)
**Added**:
- **clang-tidy errors blocking build** problem/solution:
  - Error example: "trailing return types are required"
  - Solution: Fix all errors (cannot be skipped)
  - Code example showing old vs new syntax

#### H. Key Files Map Section (Line 773-798)
**Updated**:
- **Phase 5 Scripts**:
  - Added `scripts/data_collection/news_ingestion.py`
  - Added `scripts/monitoring/setup_news_database.py`
- **Documentation**:
  - Added `AI_CONTEXT.md` (with NEWS! marker)
  - Added `NEWS_INGESTION_SYSTEM.md` (with NEWS! marker)
  - Added `NEWS_INGESTION_QUICKSTART.md` (with NEWS! marker)

---

### 3. VERIFIED: .claude/settings.local.json (No Changes Needed)

**Purpose**: Claude Code permissions configuration
**Location**: `/home/muyiwa/Development/BigBrotherAnalytics/.claude/settings.local.json`

**Status**: Already configured correctly

**Existing Permissions** (relevant to news system):
- `Bash(ninja market_intelligence news_ingestion_py:*)` - Build news modules
- `Bash(uv run python:*)` - Run Python scripts
- `Bash(duckdb:*)` - Database operations
- `Bash(ninja:*)` - General build commands

**No changes required** - All necessary permissions already in place.

---

## Key Information Added for AI Assistants

### 1. News Ingestion System Architecture

**Data Flow**:
```
NewsAPI → C++ News Collector → C++ Sentiment Analyzer → DuckDB → Dashboard
  (fetch)    (circuit breaker)     (60+ keywords)      (store)   (visualize)
```

**Components**:
- `sentiment_analyzer.cppm` - Keyword-based sentiment scoring (-1.0 to +1.0)
- `news_ingestion.cppm` - HTTP client with rate limiting and circuit breaker
- `news_bindings.cpp` - Python bindings via pybind11
- Dashboard integration - Streamlit news feed view

### 2. C++23 Module System

**Module Imports** (CORRECT):
```cpp
import bigbrother.utils.types;
import bigbrother.utils.logger;
import bigbrother.market_intelligence.sentiment;
import bigbrother.market_intelligence.news;
```

**Old Headers** (WRONG):
```cpp
#include "sentiment_analyzer.h"  // Don't use for modules
```

### 3. Build System Requirements

**Compiler**: Clang 21.1.5 (C++23 modules support)
**Build Tool**: Ninja (REQUIRED for modules)
**Linter**: clang-tidy (MANDATORY, cannot be skipped)

**Build Order**:
```bash
ninja utils                    # Core utilities first
ninja market_intelligence      # Market modules (includes news/sentiment)
ninja news_ingestion_py       # Python bindings last
```

### 4. Error Handling Pattern

**ALWAYS use `std::unexpected`**:
```cpp
// ✅ CORRECT
return std::unexpected(Error::make(ErrorCode::NetworkError, "message"));

// ❌ WRONG
return Error{"message"};  // Compile error!
```

### 5. Coding Standards

**Trailing Return Types** (MANDATORY):
```cpp
// ✅ CORRECT (clang-tidy passes)
auto calculate(int x) -> double { return x * 2.0; }

// ❌ WRONG (clang-tidy ERROR)
double calculate(int x) { return x * 2.0; }
```

**[[nodiscard]] Attribute**:
```cpp
// ✅ CORRECT - All getters/queries
[[nodiscard]] auto getName() const -> std::string;
[[nodiscard]] auto calculatePrice() const -> double;
```

### 6. Python Usage

**ALWAYS use `uv run python`** (NOT bare `python` or `pip`):
```bash
# ✅ CORRECT
uv run python script.py
uv run streamlit run app.py
uv add pandas

# ❌ WRONG
python script.py
pip install pandas
```

### 7. Sentiment Analysis Algorithm

**Features**:
- 60+ positive keywords (profit, gain, surge, bull, upgrade...)
- 60+ negative keywords (loss, decline, fall, bear, downgrade...)
- Intensifiers (very, extremely → 1.5x multiplier)
- Negation handling ("not good" → negative)
- Scoring: (positive - negative) / total, normalized [-1.0, +1.0]

**Performance**:
- Speed: ~0.01ms per article (100,000 articles/second)
- Memory: <1MB (keyword sets cached)
- Accuracy: 70-75% on financial news

### 8. Database Schema

**news_articles Table**:
```sql
CREATE TABLE news_articles (
    article_id VARCHAR PRIMARY KEY,    -- MD5 hash of URL (deduplication)
    symbol VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    sentiment_score DOUBLE,            -- -1.0 to 1.0
    sentiment_label VARCHAR,           -- 'positive', 'negative', 'neutral'
    positive_keywords TEXT[],
    negative_keywords TEXT[],
    published_at TIMESTAMP NOT NULL
);
```

**Indexes**:
- `idx_news_symbol` - Fast symbol lookups
- `idx_news_published` - Time-based queries
- `idx_news_sentiment` - Sentiment filtering

### 9. clang-tidy Enforcement

**Status**: MANDATORY - Runs before every build
**Cannot be disabled**: `SKIP_CLANG_TIDY=1` no longer works

**Key Checks (ERROR level)**:
- Trailing return types (`modernize-use-trailing-return-type`)
- Rule of Five (`cppcoreguidelines-special-member-functions`)
- [[nodiscard]] on query methods
- const correctness

**System Headers**: Excluded via `.clang-tidy` config

---

## Integration Points Documented

### 1. News → Dashboard
```bash
# Fetch news
uv run python scripts/data_collection/news_ingestion.py

# View in dashboard
uv run streamlit run dashboard/app.py
# Navigate to: "News Feed"
```

### 2. News → Trading Strategy (Future)
```cpp
// Planned: Use sentiment in sector rotation
auto score = 0.6 * employment + 0.3 * sentiment + 0.1 * momentum;
```

### 3. Daily Automation
```bash
# Morning setup
uv run python scripts/phase5_setup.py --quick

# Evening shutdown
uv run python scripts/phase5_shutdown.py
```

---

## Documentation Cross-References

### Primary Documents (Now Linked)
1. **AI_CONTEXT.md** - Comprehensive AI assistant context (NEW!)
2. **NEWS_INGESTION_SYSTEM.md** - Full architecture documentation
3. **NEWS_INGESTION_QUICKSTART.md** - 3-step quick start guide
4. **CODING_STANDARDS.md** - C++23 coding standards
5. **PYTHON_BINDINGS_GUIDE.md** - pybind11 patterns

### Referenced in Copilot Instructions
- All essential docs now listed in "Essential Documentation" section
- Quick reference section updated with news system commands
- Troubleshooting section expanded with clang-tidy issues

---

## Benefits for AI Assistants

### 1. Context Awareness
- Full understanding of news ingestion system architecture
- Knowledge of module dependencies and build order
- Awareness of mandatory coding standards (trailing return types, etc.)

### 2. Code Generation
- Can generate correct module import syntax
- Uses proper error handling patterns (std::unexpected)
- Follows clang-tidy requirements automatically
- Generates correct Python binding code

### 3. Build System Understanding
- Knows to use Ninja generator (not Make)
- Understands build order: utils → market_intelligence → bindings
- Aware of clang-tidy enforcement (cannot be skipped)

### 4. Error Handling Guidance
- Suggests Result<T> for error-prone operations
- Uses std::unexpected(Error::make(...)) pattern
- Avoids throwing exceptions for expected failures

### 5. Python Integration
- Always suggests `uv run python` (not bare python)
- Sets correct PYTHONPATH for bindings
- Understands pybind11 patterns for STL types

---

## Testing Validation

All updated documentation has been validated against:
- Existing news ingestion implementation (src/market_intelligence/)
- Python bindings (src/python_bindings/news_bindings.cpp)
- Build system (CMakeLists.txt)
- Dashboard integration (dashboard/app.py)
- Coding standards enforcement (.clang-tidy)

---

## Next Steps for Development

### Immediate (Phase 5+)
1. Build C++ modules: `ninja market_intelligence news_ingestion_py`
2. Test Python bindings: `uv run python test_news_bindings.py`
3. Fetch sample news: `uv run python scripts/data_collection/news_ingestion.py`
4. Verify dashboard: `uv run streamlit run dashboard/app.py`

### Future Integration
1. Add news sentiment to trading strategy scoring (30% weight)
2. Schedule daily news updates (cron job)
3. Aggregate news by sector (populate sector_news_sentiment table)
4. Add more news sources (AlphaVantage, Finnhub, Yahoo Finance)

---

## Files Summary

### Created
- **docs/AI_CONTEXT.md** (50KB) - Comprehensive AI assistant context
- **docs/AI_INSTRUCTIONS_UPDATE_SUMMARY.md** (this file) - Update summary

### Modified
- **.github/copilot-instructions.md** - 8 major sections updated with news system details

### Verified (No Changes)
- **.claude/settings.local.json** - Already has necessary permissions

---

## Conclusion

AI assistant configurations have been comprehensively updated with:
1. Complete news ingestion system documentation
2. C++23 module architecture and dependency graph
3. Build system requirements (Ninja, clang-tidy enforcement)
4. Error handling patterns (Result<T>, std::unexpected)
5. Coding standards (trailing return types, [[nodiscard]], Rule of Five)
6. Python bindings patterns (pybind11)
7. Integration points and workflows

All documentation is cross-referenced and validated against actual implementation. AI assistants now have full context for working with the news ingestion system and the broader C++23 module architecture.

---

**Status**: Complete ✅
**Last Updated**: 2025-11-10
**Author**: Olumuyiwa Oluwasanmi
