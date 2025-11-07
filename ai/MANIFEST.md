# BigBrotherAnalytics - Project Manifest

**Last Updated:** 2025-11-06
**Phase:** Tier 1 POC - DuckDB-First Validation
**Timeline:** Weeks 1-12 (3 months)
**Success Metric:** $150+/day profit with $30k Schwab account

---

## Primary Goal

**Prove profitability of algorithmic options day trading** using AI-powered market intelligence and correlation analysis before investing in infrastructure scaling.

**Success Criteria:**
- Daily profit > $150 (80% of trading days)
- Win rate > 60% on options trades
- Sharpe ratio > 2.0
- Max drawdown < 15%
- Consistent performance across 3+ months

---

## Project Phases

### Phase 1: CURRENT - Planning & Design (Weeks 1-2)
**Status:** ✅ COMPLETE (as of 2025-11-06)
- ✅ PRD finalized with DuckDB-first strategy
- ✅ Architecture documents created for all 3 subsystems
- ✅ Database strategy analysis (DuckDB-first decision)
- ✅ Ansible playbook updated for Tier 1 (DuckDB only)
- ✅ AI documentation structure initialized

### Phase 2: Environment Setup (Week 2)
**Status:** 🔄 READY TO START
- [ ] Run ansible playbook for Tier 1 setup
- [ ] Verify C++23, Python 3.14+, CUDA installation
- [ ] Install DuckDB and test basic operations
- [ ] Set up development environment (IDE, tooling)

### Phase 3: Core Implementation (Weeks 3-8)
**Status:** ⏸️ PENDING
- [ ] **Market Intelligence Engine** (Weeks 3-5)
  - Data ingestion from free sources (FRED, SEC, Yahoo Finance)
  - Basic NLP pipeline (sentiment, entity recognition)
  - Impact prediction with ML models
  - Store results in DuckDB
- [ ] **Correlation Analysis Tool** (Weeks 4-6)
  - Historical data loading (10 years)
  - C++23 correlation calculations (MPI, OpenMP)
  - Time-lagged correlation discovery
  - Store correlations in DuckDB
- [ ] **Trading Decision Engine** (Weeks 6-8)
  - Options pricing (trinomial trees, Black-Scholes)
  - Strategy implementation (delta-neutral, volatility)
  - Schwab API integration (paper trading)
  - Explainability layer (SHAP, LIME)

### Phase 4: Backtesting & Validation (Weeks 9-10)
**Status:** ⏸️ PENDING
- [ ] Load 10 years of free historical data
- [ ] Run comprehensive backtests
- [ ] Walk-forward validation
- [ ] Stress testing and scenario analysis
- [ ] Performance metrics calculation

### Phase 5: Paper Trading (Weeks 11-12)
**Status:** ⏸️ PENDING
- [ ] Connect to Schwab paper trading
- [ ] Real-time signal generation
- [ ] Live trade execution (paper)
- [ ] Performance monitoring
- [ ] Daily profitability tracking

### Phase 6: DECISION POINT (Month 4)
**Status:** ⏸️ PENDING
- [ ] Evaluate profitability over 3 months
- [ ] If profitable ($150+/day): Proceed to Tier 2
- [ ] If not profitable: Pivot or stop (no sunk cost)

### Phase 7: Tier 2 Scaling (Month 5+, CONDITIONAL)
**Status:** ⏸️ DEFERRED
- [ ] Add PostgreSQL for operational data (1-2 days migration)
- [ ] Subscribe to paid data feeds (Polygon.io, NewsAPI)
- [ ] Scale to real money trading
- [ ] Expand to stock trading strategies

---

## Active Agents / Components

### Agent 1: Data Ingestion Agent
**Status:** Not implemented
**Purpose:** Collect data from free sources (FRED, Yahoo Finance, SEC)
**Technology:** Python 3.14+, aiohttp, scrapy
**Storage:** DuckDB + Parquet files
**Priority:** HIGH (Week 3)

### Agent 2: NLP Processing Agent
**Status:** Not implemented
**Purpose:** Sentiment analysis, entity recognition, event extraction
**Technology:** PyTorch, Transformers (BERT, FinBERT)
**Storage:** DuckDB
**Priority:** HIGH (Week 3-4)

### Agent 3: Impact Prediction Agent
**Status:** Not implemented
**Purpose:** Predict market impacts from news/events
**Technology:** XGBoost, PyTorch, SHAP
**Storage:** DuckDB
**Priority:** HIGH (Week 4-5)

### Agent 4: Correlation Engine
**Status:** Not implemented
**Purpose:** Calculate correlations across thousands of securities
**Technology:** C++23, MPI, OpenMP, Intel MKL
**Storage:** DuckDB
**Priority:** HIGH (Week 4-6)

### Agent 5: Options Pricing Engine
**Status:** Not implemented
**Purpose:** Real-time options valuation
**Technology:** C++23, CUDA (optional)
**Storage:** DuckDB
**Priority:** CRITICAL (Week 6-7)

### Agent 6: Trading Decision Engine
**Status:** Not implemented
**Purpose:** Generate and execute trading decisions
**Technology:** C++23, Python 3.14+, Schwab API
**Storage:** DuckDB
**Priority:** CRITICAL (Week 7-8)

### Agent 7: Explainability Agent
**Status:** Not implemented
**Purpose:** Explain every trading decision
**Technology:** SHAP, LIME, custom visualization
**Storage:** DuckDB
**Priority:** HIGH (Week 8)

### Agent 8: Backtesting Engine
**Status:** Not implemented
**Purpose:** Historical performance validation
**Technology:** C++23, DuckDB (fast analytics)
**Storage:** DuckDB
**Priority:** HIGH (Week 9-10)

---

## Technology Decisions

### Database Strategy: DuckDB-First ✅
**Decision Date:** 2025-11-06
**Rationale:**
- Zero setup time (30 seconds vs 4-12 hours)
- Perfect for POC validation phase
- 5-10x faster iteration
- Full ACID compliance for financial data
- Add PostgreSQL only after proving profitability

**Reference:** `docs/architecture/database-strategy-analysis.md`

### Options Trading First ✅
**Decision Date:** 2025-11-05
**Rationale:**
- Higher profit potential per trade
- Exploit volatility and time decay
- Delta-neutral strategies reduce directional risk
- Rapid feedback loop for validation

### Free Data Validation ✅
**Decision Date:** 2025-11-05
**Rationale:**
- Yahoo Finance (free, unlimited historical)
- FRED API (free, 800,000+ economic series)
- SEC EDGAR (free, all filings)
- Only pay for data after strategies proven

---

## Risk Register

### Risk 1: POC May Not Be Profitable
**Probability:** Medium
**Impact:** High (project stop)
**Mitigation:** DuckDB-first = zero infrastructure waste, can pivot quickly

### Risk 2: Free Data Insufficient
**Probability:** Low
**Impact:** Medium
**Mitigation:** 10 years of historical data available for validation

### Risk 3: Options Market Too Complex
**Probability:** Medium
**Impact:** Medium
**Mitigation:** Start with simple strategies (straddles, strangles), expand gradually

### Risk 4: Schwab API Limitations
**Probability:** Low
**Impact:** Medium
**Mitigation:** Well-documented API, backup brokers available

---

## Key Metrics Tracking

### Development Metrics (Current Phase)
- Lines of code written: 0 (design phase complete)
- Tests written: 0 (not started)
- Documentation coverage: 100% (planning phase)

### Financial Metrics (Future)
- Daily profit: TBD (paper trading not started)
- Win rate: TBD
- Sharpe ratio: TBD
- Max drawdown: TBD

### Performance Metrics (Future)
- Signal generation latency: TBD
- ML inference throughput: TBD
- Database query latency: TBD

---

## Next Immediate Actions (Week 2)

1. **Run ansible playbook** - Set up Tier 1 environment
2. **Create project structure** - Set up src/, tests/, scripts/ directories
3. **Start Agent 1** - Implement data ingestion from Yahoo Finance
4. **Start Agent 4** - Begin C++23 correlation engine implementation

---

## AI Orchestration System

**For structured development, BigBrotherAnalytics uses a multi-agent orchestration system:**

See `ai/README.md` for complete documentation.

**Quick Reference:**

```
Orchestrator → PRD Writer → System Architect → File Creator → Self-Correction
```

**Available Agents:**
1. **Orchestrator** - Coordinates complex multi-agent workflows
2. **PRD Writer** - Creates/updates requirements documentation
3. **System Architect** - Designs system architecture
4. **File Creator** - Generates implementation code
5. **Self-Correction** - Validates and auto-fixes code
6. **Code Reviewer** - Reviews code quality
7. **Debugger** - Systematic debugging and fixes

**Workflows:**
- Feature Implementation: `WORKFLOWS/feature_implementation.md`
- Bug Fix: `WORKFLOWS/bug_fix.md`

---

## Notes for AI Assistants

- **Always reference this manifest** before starting new work
- **For complex tasks, use the Orchestrator** (`PROMPTS/orchestrator.md`)
- **Update status** as tasks complete
- **Document decisions** with rationale and date
- **Track risks** and mitigations
- **Focus on Tier 1 POC** - don't prematurely optimize for scale
