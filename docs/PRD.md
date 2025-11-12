# Product Requirements Document: BigBrotherAnalytics

**Version:** 1.5.0
**Date:** November 12, 2025
**Status:** Phase 5+ Delivery Ready - ML Price Predictor v3.0 Integrated
**Author:** Olumuyiwa Oluwasanmi

> **Update (2025-11-12):** ML Price Predictor v3.0 now integrated with 60-feature neural network (56.3% 5-day, 56.6% 20-day accuracy). ONNX Runtime inference with AVX2 SIMD optimization (8x speedup). C++23 integration complete. See `ai/CLAUDE.md` for technical details.
>
> **Update (2025-11-11):** Trading Reporting System now complete with daily and weekly report generators, comprehensive signal analysis, and HTML/JSON output formats. See `docs/TRADING_REPORTING_SYSTEM.md` for details.

---

## Executive Summary

BigBrotherAnalytics is a **high-performance**, AI-powered trading intelligence platform built for **microsecond-level latency**. The system combines advanced machine learning with ultra-low latency execution in C++23, Rust, and CUDA to identify and exploit market opportunities with unprecedented speed. The platform consists of three interconnected subsystems:

1. **Market Intelligence & Impact Analysis Engine** - Processes multi-source data to predict market impacts
2. **Trading Correlation Analysis Tool** - Discovers temporal and causal relationships between securities
3. **Intelligent Trading Decision Engine** - Executes trading strategies with initial focus on options day trading

**Speed is of the essence.** The platform is architected for lightning-fast analysis and execution, with core components written in **C++23** (leveraging latest language features) and Rust, AI inference accelerated by CUDA and vLLM, and massive parallelization using MPI, OpenMP, and UPC++. Machine learning components use **Python 3.14+ in GIL-free mode** to exploit true multi-threaded parallelism for CPU-bound ML workloads. Initial deployment targets private servers (32+ cores) to maximize performance and minimize security concerns, with cloud deployment deferred until after validation.

**Technology Highlights:**
- **C++23:** Cache-friendly containers (`std::flat_map`), better error handling (`std::expected`), multi-dimensional arrays (`std::mdspan`)
- **Python 3.13:** Production-stable ML ecosystem with pybind11 for C++ acceleration (GIL-bypass)
- **Performance Target:** Near-linear scaling with core count across both C++ and Python components

**Initial Focus:** Algorithmic options day trading to exploit rapid market movements and volatility patterns, with specialized strategies for:
- **Recession Detection & Exploitation:** Identify macroeconomic turning points using Fed data, yield curves, and leading indicators
- **Defensive Positioning:** Options strategies (protective puts, spreads) during recession signals
- **Counter-Cyclical Plays:** Identify recession-resistant sectors and anti-correlation opportunities

---

## Key Architecture Highlights: Affordability & Unification

**🎯 Startup-Friendly Cost Structure:**
- **Monthly Operational Cost: $250-1,000** (vs. $25,000+ with traditional enterprise solutions)
- **Zero licensing fees** - 95%+ open-source technology stack
- **Own hardware deployment** - No cloud bills during development
- **Free government data** - FRED, SEC, Congress, FDA, EPA, HHS APIs (all free)
- **Affordable market data** - Polygon.io ($200-500/month) instead of Bloomberg Terminal ($24,000/year)

**🔧 DuckDB-First Database Strategy:**
- **Tier 1 POC (Months 1-4): DuckDB ONLY**
  - **Zero setup** - Embedded database, no server configuration
  - **Instant start** - 30 seconds to working database
  - **Perfect for validation** - Fast backtesting, rapid iteration
  - **Full ACID compliance** - Safe for financial data
  - **Production-ready** - Handles real-time trading at POC scale
  - **See detailed analysis:** [docs/architecture/database-strategy-analysis.md](./architecture/database-strategy-analysis.md)

- **Tier 2 Production (Month 5+, after proving profitability): Dual Database**
  - **DuckDB** continues for analytics, backtesting, model training
  - **PostgreSQL 16+** added for operational data with extensions:
    - **TimescaleDB** for high-frequency time-series
    - **pgvector** for semantic search
    - **Apache AGE** for impact graphs
  - **Migration time:** 1-2 days (well-tested path)
  - **Best of both worlds** - DuckDB for analytics, PostgreSQL for operations

**Why DuckDB-First:**
- Reduces POC complexity by 90%
- 12+ hours saved on database setup
- 5-10x faster iteration during validation
- Zero risk - add PostgreSQL only after proving profitability
- Focus on algorithms, not database administration

**📊 Comprehensive Data Coverage:**
- **15+ government agencies** - Congress, Treasury, USDA, FDA, EPA, HHS, and more
- **Global intelligence** - Fed, ECB, World Bank, IMF, OECD (all free APIs)
- **Legal & regulatory** - SEC filings, court opinions, patent data
- **Corporate intelligence** - News aggregators, press releases, social media
- **Industrial data** - Manufacturing indices, trade data, supply chain signals
- **Cost-effective news** - NewsAPI, MarketAux, RSS feeds (vs. Bloomberg, Refinitiv)

**⚡ Performance Without Compromise:**
- **C++23** for ultra-low latency critical paths (< 1ms execution)
- **Python 3.14+ GIL-free** for true multi-threaded ML workloads
- **GPU acceleration** with CUDA for inference (vLLM for 10K+ predictions/sec)
- **32+ core parallelization** with MPI, OpenMP, UPC++
- **Private servers** - Direct hardware access, no virtualization overhead

**💰 Cost Comparison (Three Deployment Tiers):**

| Component | Traditional Enterprise | Production Tier | **Quick Start (Zero-Fee)** |
|-----------|----------------------|----------------|------------------------|
| Market Data | Bloomberg Terminal ($2,000/mo) | Polygon.io ($200-500/mo) | **Free APIs ($0/mo)** |
| Database | InfluxDB + Neo4j + Pinecone ($1,100/mo) | PostgreSQL + Extensions ($0/mo) | **DuckDB ($0/mo)** |
| News Feed | Refinitiv/Factiva ($1,000+/mo) | NewsAPI + RSS ($50-450/mo) | **RSS + Free APIs ($0/mo)** |
| Cloud Infra | AWS/GCP ($2,000+/mo) | Own Hardware ($0/mo) | **Own Hardware ($0/mo)** |
| Other Services | Various ($500+/mo) | Sentry + misc ($0-50/mo) | **All Open-Source ($0/mo)** |
| **Total** | **~$25,000+/month** | **$250-1,000/month** | **$0/month** |
| **Annual Cost** | **~$300,000/year** | **~$3,000-12,000/year** | **$0/year (validate first!)** |

**Deployment Strategy:**
1. **Months 1-2:** Zero-fee tier for validation and prototyping
2. **Months 3-4:** Add Production tier ($250-1K/mo) when strategies proven
3. **Never:** Enterprise tier (unnecessary with modern open-source)

**Total Possible Savings: $300,000/year** while maintaining institutional-grade performance.

---

## Quick Start: Zero-Fee Rapid Deployment (Own Hardware)

**🚀 Goal:** Get a minimal viable system running in days, not months, with ZERO subscription fees.

### Minimal Infrastructure Setup (Week 1)

**Hardware Requirements (Existing Computer):**
- **CPU:** 8+ cores (any modern desktop/laptop)
- **RAM:** 16GB+ (32GB recommended)
- **Storage:** 500GB+ SSD
- **OS:** Ubuntu 22.04 LTS, macOS, or Windows with WSL2
- **GPU:** Optional (any NVIDIA GPU for ML acceleration later)

**Software Installation (1-2 hours):**
```bash
# 1. Install Python 3.11+ (3.14 when available)
sudo apt install python3.11 python3.11-pip

# 2. Install DuckDB (instant)
pip install duckdb

# 3. Install data processing libraries
pip install pandas polars pyarrow requests aiohttp scrapy

# 4. Install NLP libraries (optional, for later)
pip install spacy transformers

# That's it! No database server, no configuration, ready to go.
```

### Phase 1: Free Data Collection (Weeks 1-2)

**Zero-Cost Data Sources (Start Immediately):**

1. **Market Data (Free Tiers):**
   - **Alpha Vantage:** 500 calls/day (free forever)
   - **Finnhub:** 60 calls/minute (free tier)
   - **Yahoo Finance:** Unlimited via yfinance library
   - Download historical data once, store in Parquet files

2. **Government Data (100% Free, Unlimited):**
   - **FRED API:** 800,000+ economic time series (free, unlimited)
   - **SEC EDGAR:** All company filings (free, unlimited)
   - **Congress.gov API:** Legislative data (free)
   - **FDA OpenFDA APIs:** Drug/device data (free)
   - **EPA APIs:** Environmental data (free)
   - **Treasury APIs:** Treasury data (free)

3. **News (Free Tiers):**
   - **NewsAPI:** 100 requests/day free tier
   - **RSS Feeds:** Unlimited from major publications
   - **Reddit API:** Free via PRAW library
   - **Twitter/X API:** Free tier (with limits)

**Simple Data Collection Script:**
```python
import duckdb
import yfinance as yf
import pandas as pd
from fredapi import Fred

# 1. Collect stock data (free)
ticker = yf.Ticker("AAPL")
hist = ticker.history(period="5y")

# 2. Store in DuckDB
con = duckdb.connect('trading.db')
con.execute("CREATE TABLE IF NOT EXISTS prices AS SELECT * FROM hist")

# 3. Add economic data (free)
fred = Fred(api_key='YOUR_FREE_KEY')
gdp = fred.get_series('GDP')
con.execute("CREATE TABLE IF NOT EXISTS economic_data AS SELECT * FROM gdp")

# Query instantly
result = con.execute("""
    SELECT * FROM prices
    WHERE Date > '2024-01-01'
    ORDER BY Date DESC
""").df()

print(f"Collected {len(result)} days of data in seconds!")
```

### Phase 2: Basic Analytics (Weeks 2-4)

**Build Core Features with DuckDB:**

1. **Historical Analysis:**
   - Store 10+ years of data in Parquet files
   - Query directly without loading to database
   - Compute correlations across thousands of securities
   - Run backtests on historical data

2. **Simple Strategies:**
   - Moving average crossovers
   - RSI/MACD signals
   - Volume analysis
   - Correlation-based pairs trading

3. **Performance Testing:**
   - Backtest strategies on historical data
   - Calculate Sharpe ratios, drawdowns
   - Validate approach before spending money

**Example: Correlation Analysis**
```python
import duckdb
import pandas as pd

# Load historical data from Parquet files
con = duckdb.connect()

# Calculate rolling correlations across 1000 stocks
result = con.execute("""
    SELECT
        a.symbol as symbol_a,
        b.symbol as symbol_b,
        corr(a.close, b.close) as correlation
    FROM 'data/prices/*.parquet' a
    JOIN 'data/prices/*.parquet' b
        ON a.date = b.date
    WHERE a.date >= '2020-01-01'
    GROUP BY a.symbol, b.symbol
    HAVING correlation > 0.8
    ORDER BY correlation DESC
""").df()

print(f"Found {len(result)} highly correlated pairs")
```

### Phase 3: Scale When Validated (Month 2+)

**Only pay for data when your strategies show promise:**

1. **Month 1-2:** Free data only, prove concepts with DuckDB
2. **Month 3:** Add Polygon.io ($200/month) if backtests are positive
3. **Month 4:** Add NewsAPI business tier ($449/month) if news strategies work
4. **Month 5+:** Optionally add PostgreSQL for enhanced real-time operations (DuckDB sufficient for most cases)

**Progressive Cost Structure:**
- **Weeks 1-4:** $0/month (free data only)
- **Months 1-2:** $0/month (validation phase)
- **Month 3:** $200/month (if validated)
- **Month 4+:** $250-1,000/month (if profitable)

### Quick Start Technology Stack (Zero Setup)

**Database & Storage (Tier 1 POC):**
- ✅ **DuckDB** - PRIMARY DATABASE (embedded, zero setup, ACID compliant)
- ✅ **Parquet files** - Efficient columnar storage
- ✅ **pandas/polars** - Data manipulation
- ⏸️  PostgreSQL - DEFERRED to Tier 2 (only after proving profitability)
- ⏸️  Redis - OPTIONAL (DuckDB handles caching needs for POC)

**Data Collection:**
- ✅ **yfinance** - Free Yahoo Finance data
- ✅ **Alpha Vantage** - Free tier (500 calls/day)
- ✅ **requests/aiohttp** - Free government APIs
- ✅ **Scrapy** - Web scraping (free)
- ⏸️  Paid APIs - Add when validated

**Analytics:**
- ✅ **pandas** - Data analysis
- ✅ **numpy** - Numerical computing
- ✅ **scipy** - Statistical functions
- ✅ **scikit-learn** - ML models (free)
- ⏸️  CUDA/GPU - Add for speed later

**Visualization:**
- ✅ **matplotlib** - Basic plots
- ✅ **seaborn** - Statistical visualization
- ✅ **plotly** - Interactive dashboards
- ⏸️  Grafana - Add for production monitoring

### Deployment Timeline (Zero Fees)

**Week 1: Setup**
- Install Python + DuckDB (2 hours)
- Download historical data from free sources (1 day)
- Build basic data pipeline (3 days)

**Week 2-4: Prototype**
- Implement correlation analysis
- Build simple trading strategies
- Backtest on historical data
- Validate performance metrics

**Month 2-3: Validate**
- Paper trading with free data
- Refine strategies
- Measure actual vs predicted performance
- Decision point: proceed or pivot?

**Month 4+: Scale**
- Add paid data sources incrementally
- Continue with DuckDB (sufficient for most trading at this scale)
- Optionally add PostgreSQL for Tier 2 features (migration: 1-2 days)
- Real money trading (small amounts)
- Scale up as profitability proven

### Success Metrics Before Spending Money

**Validation Criteria (Using Free Data):**
- ✅ Backtest Sharpe ratio > 1.5
- ✅ Win rate > 55%
- ✅ Consistent performance across 3+ years
- ✅ Strategy works on out-of-sample data
- ✅ Max drawdown < 20%

**Only proceed with paid subscriptions if all criteria met!**

---

## 1. Product Vision and Goals

### 1.1 Vision Statement

To create the **fastest and most intelligent** automated trading platform that combines ultra-low latency execution, real-time data analysis, and predictive modeling to consistently outperform market benchmarks, with initial focus on exploiting options market opportunities through algorithmic day trading.

### 1.2 Primary Goals

1. **Extreme Performance** - Achieve microsecond-level latency for critical execution paths
2. **Superior Returns** - Achieve above-market returns through intelligent analysis and timing
3. **Options Mastery** - Develop sophisticated strategies for options trading (initial priority)
4. **Risk Management** - Minimize downside risk through diversification and predictive analysis
5. **Automation** - Eliminate human intervention in trading decisions (especially day trading)
6. **Massive Parallelization** - Leverage 32+ cores for simultaneous analysis
7. **Adaptability** - Learn and adapt to changing market conditions continuously

### 1.3 Success Criteria

**Performance Metrics:**
- Signal-to-execution latency < 1 millisecond for critical path
- Market data processing latency < 100 microseconds
- ML inference throughput > 10,000 predictions/second (with vLLM)
- Parallel correlation calculations across all cores with near-linear scaling

**Financial Metrics:**
- Achieve consistent positive alpha (outperformance vs. market benchmarks)
- Options day trading: Sharpe ratio > 2.0
- Overall portfolio: Sharpe ratio > 1.5
- Minimize maximum drawdown to < 15% annually
- Demonstrate statistically significant predictive power (p < 0.01)

**Initial Focus Metrics (Options Day Trading):**
- Win rate > 60%
- Average profit per options trade > 10%
- Daily profitability > 80% of trading days

---

## 2. Target Users and Use Cases

### 2.1 Primary Users

- **Quantitative Traders** - Seeking data-driven trading signals with ultra-low latency
- **Options Traders** - Requiring sophisticated options strategy identification (initial priority)
- **Investment Analysts** - Requiring comprehensive market intelligence
- **Portfolio Managers** - Looking for diversified strategy execution
- **Algorithmic Trading Firms** - Needing high-performance automated execution

### 2.2 Use Cases (in Priority Order)

**Phase 1: Options Day Trading (Initial Focus)**

**⚠️ CRITICAL UNDERSTANDING: Day Trading is Driven by Sentiment, Not Just Logic**

**Behavioral Finance Principles for Day Trading:**
- **"Animal Spirits" (Keynes):** Markets driven by emotion, instinct, and crowd psychology
  - Fear and greed dominate short-term price movements
  - Rational analysis only part of the equation
  - Momentum can override fundamental value for hours or days
  - Herd behavior creates exploitable patterns

- **Sentiment Indicators MUST Be Tracked:**
  - Market-wide sentiment (VIX, put/call ratios, advance/decline)
  - Stock-specific sentiment (social media, news tone, analyst upgrades/downgrades)
  - Options sentiment (implied volatility skew, unusual option activity)
  - Institutional vs retail sentiment (order flow, dark pool activity)

- **Momentum Over Fundamentals (Intraday):**
  - Day trades resolved in minutes to hours
  - Fundamental value matters less than directional momentum
  - Technical patterns (support/resistance) are self-fulfilling prophecies
  - Volume and liquidity create momentum continuation

- **Psychology Patterns to Exploit:**
  - Overreaction to news (buy panic, sell euphoria)
  - FOMO (Fear of Missing Out) - chasing momentum
  - Panic selling - capitulation patterns
  - "Dead cat bounce" - false reversals
  - "Catching falling knives" - premature bottom picking

**Day Trading Use Cases (Sentiment-Driven):**
1. Exploiting rapid implied volatility changes based on news events **AND market overreaction**
2. Identifying mispriced options using correlation analysis, impact predictions, **AND sentiment divergence**
3. Executing delta-neutral strategies (straddles, strangles) around major events **capturing sentiment volatility**
4. Capitalizing on options Greeks arbitrage opportunities **driven by irrational fear/greed**
5. Leveraging time decay (theta) in high-probability trades **while monitoring sentiment shifts**
6. **NEW: Trading momentum breakouts driven by social media and retail trader activity**
7. **NEW: Fading extreme sentiment (contrarian plays when fear/greed reaches extremes)**
8. **NEW: Capturing volatility spikes from panic or euphoria**

**Phase 2: Stock Trading (Lower Initial Priority, Ultimate Goal)**
6. Identifying emerging market trends before mainstream awareness
7. Predicting company-specific impacts from geopolitical events
8. Executing high-frequency stock trades based on micro-movements
9. Building long-term stock portfolios with superior risk-adjusted returns
10. Hedging positions using correlation analysis

---

## 3. Sub-Project 1: Market Intelligence & Impact Analysis Engine

### 3.1 Overview

A sophisticated ML system that ingests, processes, and analyzes diverse data sources to predict market impacts on specific securities and generate impact graphs showing causal chains and relationship strengths.

**📋 Complete Architecture:** For detailed system architecture, component designs, database schemas, API specifications, and implementation guidelines, see the **[Market Intelligence & Impact Analysis Engine - Architecture Design Document](./architecture/market-intelligence-engine.md)**.

The architecture document includes:
- High-level system architecture with Mermaid diagrams
- Component breakdown with C++23, Python, CUDA, MPI/OpenMP/UPC++ implementations
- Data flow architecture and sequence diagrams
- Database schema design (PostgreSQL + TimescaleDB/AGE/pgvector, DuckDB)
- Complete REST/WebSocket/GraphQL API specifications (OpenAPI 3.0)
- C++23 code examples with std::expected, std::flat_map, MPI, OpenMP, UPC++
- CUDA/PyTorch GPU acceleration examples with vLLM and TensorRT
- Python 3.14+ GIL-free parallel processing examples
- Performance optimization strategies (caching, indexing, async patterns)
- Deployment architectures (single-node to multi-node clusters)
- Phase 1-12 week implementation checklist

### 3.2 Data Sources & Extraction Technologies

**Note:** This section details the comprehensive data sources and specific technologies for extracting intelligence from each sector. All technologies selected prioritize affordability and open-source solutions suitable for a startup.

#### 3.2.1 Real-Time News Analysis
- **Corporate Announcements** - Earnings reports, guidance updates, management changes
- **Breaking News** - Financial news aggregators
- **Media Sentiment** - Social media trends (Twitter/X, Reddit, StockTwits)
- **Press Releases** - Company PR wires
- **Analyst Reports** - Upgrades, downgrades, price target changes

#### 3.2.2 Market Data

**Options Data (Initial Priority):**
- **Options Chain** - Complete options chains with all strikes and expirations
- **Implied Volatility** - IV for all options, IV surface modeling
- **Greeks** - Delta, Gamma, Theta, Vega, Rho for all options
- **Open Interest** - OI changes, unusual OI patterns
- **Options Volume** - Absolute and relative volume, volume/OI ratios
- **Bid-Ask Spreads** - Options liquidity metrics
- **Put-Call Ratios** - Market-wide and symbol-specific
- **Options Flow** - Large block trades, sweep orders, unusual activity

**Underlying Stock Data:**
- **Price Data** - OHLCV (Open, High, Low, Close, Volume) at multiple timeframes
- **Trading Volume** - Absolute and relative volume changes
- **Market Depth** - Order book data, bid-ask spreads
- **Volatility Metrics** - Historical volatility, realized volatility
- **Stock Performance** - Percent changes, drawdown metrics

#### 3.2.3 Legal & Regulatory Intelligence
- **Legal Proceedings** - Lawsuits, investigations, settlements
- **Regulatory Filings** - SEC filings (10-K, 10-Q, 8-K, 13-F)
- **Regulatory Decisions** - FDA approvals, antitrust rulings
- **Compliance Changes** - New regulations affecting industries

#### 3.2.4 Geopolitical Events
- **International Relations** - Treaties, alliances, conflicts
- **Trade Policies** - Tariffs, trade agreements, sanctions
- **Political Decisions** - Elections, policy changes, legislation
- **Global Events** - Natural disasters, pandemics, major incidents

#### 3.2.5 Corporate Actions
- **Mergers & Acquisitions** - Deals, rumors, regulatory approvals
- **Dividends** - Declarations, changes, special dividends
- **Stock Splits** - Forward and reverse splits (forward, reverse)
- **Earnings** - Results, guidance, conference calls
- **Share Buybacks** - Announcements and execution
- **Spin-offs** - Parent company splitting off subsidiaries
- **Rights Issues** - Offerings to existing shareholders
- **Tender Offers** - Buyback offers at premium prices

**CRITICAL: Corporate Actions Impact on P&L Calculations**
- **Dividend Adjustments:**
  - Ex-dividend dates MUST be tracked for accurate position valuation
  - Dividend payments received MUST be added to realized P&L
  - Options positions: dividend capture strategies affect profitability
  - Short positions: dividend payments are a COST (subtract from P&L)
- **Stock Split Adjustments:**
  - Position quantities MUST be adjusted (2:1 split → 2x shares, 0.5x price)
  - Cost basis MUST be adjusted proportionally
  - Options contracts adjust automatically (track contract multiplier changes)
  - Reverse splits: combine positions and adjust basis
- **Merger/Acquisition Adjustments:**
  - Cash mergers: position closed at acquisition price
  - Stock-for-stock: position converted at exchange ratio
  - Mixed deals: combination of cash and stock adjustments
- **Spin-off Adjustments:**
  - Cost basis allocated between parent and spun-off entity
  - New positions created in spun-off company
  - Options positions may require manual adjustments

#### 3.2.6 Macroeconomic Indicators & Recession Detection
- **Federal Reserve** - Meeting schedules, minutes, speeches, statements
- **Interest Rates** - Current rates, rate changes, forward guidance
- **Economic Data** - GDP, unemployment, inflation, retail sales
- **Treasury Yields** - Yield curve movements and inversions (10Y-2Y spread as recession predictor)
- **Currency Markets** - Exchange rates, currency strength indices
- **Leading Economic Indicators** - Conference Board LEI, ISM PMI, building permits
- **Credit Spreads** - Corporate bond spreads, high-yield spreads (widen before recessions)
- **Consumer Confidence** - University of Michigan, Conference Board (early warning signals)
- **Labor Market** - Initial jobless claims, JOLTS (job openings decline before recessions)

#### 3.2.7 Political Intelligence
- **Supreme Court** - Docket schedules, decision dates, rulings
- **Trade Decisions** - USTR announcements, trade disputes
- **Legislative Calendar** - Bill schedules, committee hearings
- **Political Events** - Debates, elections, policy announcements

#### 3.2.8 Seasonal & Sentiment Patterns

**⚠️ CRITICAL:** These patterns represent "animal spirits" and behavioral biases that create tradeable opportunities.

- **Holiday Timing** - Market closures, pre/post-holiday patterns
  - Low volume = higher volatility (less rational pricing)
  - Holiday optimism/pessimism affects sentiment
- **Seasonal Trends** - "Sell in May," January effect, quarter-end positioning
  - Self-fulfilling prophecies driven by herd behavior
  - Tax loss harvesting (December) - emotion-driven selling
- **Market Sentiment Indices** - VIX, put-call ratios, sentiment surveys
  - **VIX > 30:** Fear dominates (contrarian opportunity)
  - **VIX < 12:** Complacency (risk of sudden reversal)
  - Extreme readings indicate irrational market psychology
- **Consumer Confidence** - University of Michigan, Conference Board indices
  - Lagging indicator but affects retail trader behavior

**Behavioral Indicators (Animal Spirits):**
- **Fear & Greed Index (CNN Business):** Quantified market emotion
- **AAII Sentiment Survey:** Individual investor bullishness/bearishness
- **Investor Intelligence Survey:** Newsletter writer sentiment
- **CBOE Put/Call Ratios:** Options traders' fear vs greed positioning
- **Margin Debt Levels:** Excessive borrowing indicates euphoria (risk)
- **Short Interest:** High short interest = potential short squeeze fuel

#### 3.2.9 Retail Intelligence
- **Sales Data** - Top products sold at major retailers:
  - Costco (membership trends, product categories)
  - Amazon (best sellers, trending products)
  - Walmart (sales patterns, inventory levels)
  - Target (product categories, seasonal trends)
  - Best Buy (electronics, consumer tech)
- **Supply Chain Signals** - Inventory levels, shipping data
- **Consumer Trends** - Product categories gaining/losing popularity

#### 3.2.10 Government & Institutional Intelligence

**U.S. Congress:**
- **Legislative Data:**
  - Congress.gov API - Bills, amendments, voting records, committees
  - ProPublica Congress API - Votes, members, statements, bills
  - GovTrack.us API - Legislative tracking
  - Senate.gov & House.gov RSS feeds
- **Congressional Research:**
  - CBO reports (web scraping)
  - GAO reports (API/web scraping)
  - CRS reports (via EveryCRSReport.com)
- **Lobbying & Finance:**
  - OpenSecrets API - Campaign finance, lobbying
  - FEC API - Campaign contributions
  - Senate Lobbying Disclosure Database

**U.S. Department of Treasury:**
- **TreasuryDirect API** - Securities, bonds, T-bills, auctions
- **Fiscal Service API** - Daily Treasury Statement, debt data
- **OFAC Sanctions Lists API** - SDN list updates
- **IRS.gov** - Tax regulations, rulings (web scraping)

**U.S. Department of Agriculture (USDA):**
- **USDA NASS API** - Crop production, livestock data, agricultural prices
- **USDA ERS** - Commodity outlook, farm income data
- **USDA AMS API** - Commodity prices, market news
- **FAS** - Global agricultural trade, export sales

**Food and Drug Administration (FDA):**
- **OpenFDA APIs:**
  - Drug APIs - Adverse events, approvals, recalls
  - Device APIs - Medical device events, classifications
  - Food APIs - Food recalls, enforcement reports
- **FDA.gov Web Scraping** - Drug approvals, breakthrough designations, warning letters
- **ClinicalTrials.gov API** - Drug and device trials

**Environmental Protection Agency (EPA):**
- **EPA AQS API** - Air quality monitoring
- **EPA ECHO API** - Facility compliance, enforcement actions
- **TRI API** - Toxic release inventory
- **GHGRP** - Greenhouse gas emissions
- **Superfund Sites API** - Contaminated sites

**Health and Human Services (HHS):**
- **HealthData.gov API** - Healthcare datasets
- **CMS APIs** - Medicare/Medicaid data, hospital compare
- **CDC WONDER API** - Mortality, disease surveillance
- **NIH Reporter API** - Research grants
- **PubMed API** - Biomedical literature

**U.S. Department of Labor (DOL):**
- **Bureau of Labor Statistics (BLS) API** - Employment statistics, unemployment rates, inflation (CPI), wages
  - Employment Situation Report (monthly jobs report)
  - JOLTS (Job Openings and Labor Turnover Survey)
  - Unemployment Insurance Weekly Claims
  - Employment Cost Index (ECI)
  - Producer Price Index (PPI)
  - Industry-specific employment data
  - Occupational employment statistics
- **OSHA API** - Workplace safety violations, enforcement actions
- **WARN Act Database** - Mass layoff notifications, plant closures (state-level data)
- **Private Sector Job Data:**
  - **Layoffs.fyi API** - Tech sector layoffs (community-sourced)
  - **Company Press Releases** - Hiring freezes, layoffs, workforce expansions
  - **LinkedIn API** - Job postings trends, hiring activity by company
  - **Indeed API** - Job market trends, hiring demand
  - **Glassdoor API** - Company reviews, hiring sentiment
- **Timing and Event Tracking:**
  - Monthly jobs report (first Friday of month) - Major market moving event
  - Weekly initial jobless claims (Thursday 8:30 AM ET) - Leading indicator
  - JOLTS report (monthly, 1-month lag) - Job openings as recession signal
  - Layoff announcements - Immediate sentiment impact on company/sector
  - Mass hiring announcements - Bullish signal for company/sector
  - Industry-specific employment trends - Sector rotation indicators

**Employment Data as Market Indicators:**
- **Leading Indicators:** Initial jobless claims spike → recession warning → defensive positioning
- **Coincident Indicators:** Nonfarm payrolls, employment rate → current economic health
- **Lagging Indicators:** Long-term unemployment → recession confirmation
- **Sector-Specific Impact:**
  - Tech layoffs → Tech sector weakness → Short tech stocks/ETFs
  - Healthcare hiring → Healthcare sector strength → Long healthcare stocks
  - Manufacturing employment decline → Industrial sector weakness → Rotate to services
  - Retail hiring (holiday season) → Consumer sector strength → Long retail

#### 3.2.11 Additional Data Extraction Technologies by Sector

**Legal Intelligence Extraction:**
- **SEC EDGAR API** - Regulatory filings (10-K, 10-Q, 8-K)
- **CourtListener API** - Federal and state court opinions
- **Justia API** - Legal case law and statutes
- **Document Parsing:** Apache Tika, PyPDF2, pdfplumber for legal PDFs
- **Web Scraping:** Scrapy/Playwright for government regulatory sites

**Industrial Intelligence Extraction:**
- **Industry Publication RSS feeds** (Manufacturing.net, IndustryWeek)
- **ISM Manufacturing Index API**
- **FRED API** - Industrial production data
- **USPTO API** - Patent data
- **Google Patents API** - Patent search

**Geopolitical Intelligence Extraction:**
- **GDELT Project** - Global events database (free)
- **NewsAPI** - Aggregated news from 80,000+ sources
- **ACLED API** - Armed conflict data
- **EventRegistry** - Real-time global news events
- **UN Data APIs** - COMTRADE, UNdata
- **OECD Data API** - Economic and policy data

**Corporate News Extraction (Affordable Options):**
- **NewsAPI** - Aggregated business news (affordable)
- **MarketAux API** - Financial news (budget-friendly)
- **Benzinga News API** - Real-time financial news
- **Alpha Vantage News API** - Free tier available
- **PR Newswire API** - Press releases
- **RSS Feeds** - Major publications (WSJ, FT, Reuters)

**Corporate Actions Data Sources (MANDATORY FOR ACCURATE P&L):**
- **Alpha Vantage** - Dividends, splits via API
  - Free tier: basic corporate actions
  - Must track: ex-dividend dates, payment dates, dividend amounts
- **Polygon.io** - Comprehensive corporate actions feed
  - Dividends, splits, mergers, spin-offs
  - Real-time and historical data
- **Finnhub** - Corporate actions, dividends, splits (free tier)
  - Good for POC validation
- **IEX Cloud** - Corporate actions and events
  - Ex-dividend dates critical for P&L accuracy
- **SEC EDGAR** - 8-K filings for material events
  - Mergers, acquisitions, spin-offs, tender offers
  - DEF 14A: proxy statements (merger/acquisition terms)
- **Schwab API** - Real-time corporate action notifications
  - Automatic adjustments to positions
  - Dividend payments credited to account
  - Split adjustments applied automatically

**Margin Interest Rate Data Sources:**
- **Schwab API** - Current margin rates via account API
  - Real-time margin balance tracking
  - Interest accrual visible in account statements
- **Schwab Website** - Published margin rate schedule
  - Updated periodically (monitor for changes)
  - https://www.schwab.com/margin-rates
- **Manual Updates** - Quarterly verification required
  - Compare API rates vs published rates
  - Alert if rates change by >0.25%

**Central Bank Data Extraction:**
- **Federal Reserve:**
  - FRED API - 800,000+ time series (free)
  - Federal Reserve Board API - H.4.1, H.3, flow of funds
  - FOMC web scraping - Meeting minutes, statements
  - FedBeige Book - Web scraping
- **European Central Bank:**
  - ECB Statistical Data Warehouse API
  - ECB RSS Feeds - Press releases, speeches
  - ECB Data Portal API
- **World Bank:**
  - World Bank Data API (free)
  - World Bank Climate Data API
  - World Bank Documents API
- **IMF:**
  - IMF Data API (free)
  - IMF eLibrary API
  - Web scraping for country reports

#### 3.2.12 Business Sector Classification and Analysis

**CRITICAL:** The system must track and analyze business sectors comprehensively to identify sector rotation opportunities, sector-specific impacts from news/events, and employment trends by industry.

**Primary Sector Classification (GICS - Global Industry Classification Standard):**

**11 Major Sectors:**

1. **Energy (10)**
   - Oil, Gas & Consumable Fuels (Exxon, Chevron, ConocoPhillips)
   - Energy Equipment & Services (Halliburton, Schlumberger, Baker Hughes)
   - ETFs: XLE (Energy Select Sector SPDR)
   - Employment Indicators: Oil rig count, energy sector jobs (BLS)
   - News Impact: OPEC decisions, oil prices, renewable energy policies

2. **Materials (15)**
   - Chemicals (Dow, DuPont, LyondellBasell)
   - Metals & Mining (Freeport-McMoRan, Newmont, Southern Copper)
   - Construction Materials (Martin Marietta, Vulcan Materials)
   - Paper & Forest Products
   - ETFs: XLB (Materials Select Sector SPDR)
   - Employment Indicators: Manufacturing jobs, mining employment (BLS)
   - News Impact: Commodity prices, infrastructure spending, housing starts

3. **Industrials (20)**
   - Aerospace & Defense (Boeing, Lockheed Martin, Raytheon)
   - Airlines (Delta, United, American Airlines)
   - Machinery (Caterpillar, Deere, Parker-Hannifin)
   - Construction & Engineering
   - Transportation (FedEx, UPS, Union Pacific)
   - ETFs: XLI (Industrial Select Sector SPDR)
   - Employment Indicators: Manufacturing employment, construction jobs (BLS)
   - News Impact: Infrastructure bills, defense spending, trade policies

4. **Consumer Discretionary (25)**
   - Automobiles (Tesla, Ford, GM)
   - Retail (Amazon, Home Depot, Lowe's, Target, Walmart)
   - Hotels & Restaurants (Marriott, Hilton, McDonald's, Starbucks)
   - Apparel (Nike, Lululemon, VF Corp)
   - Leisure Products & Entertainment
   - ETFs: XLY (Consumer Discretionary Select Sector SPDR)
   - Employment Indicators: Retail trade jobs, leisure & hospitality (BLS)
   - News Impact: Consumer confidence, holiday sales, gas prices

5. **Consumer Staples (30)**
   - Food Products (General Mills, Kraft Heinz, Mondelez)
   - Beverages (Coca-Cola, PepsiCo, Monster Beverage)
   - Household Products (Procter & Gamble, Colgate-Palmolive)
   - Tobacco (Altria, Philip Morris)
   - Food & Drug Retail (Costco, Kroger, Walgreens)
   - ETFs: XLP (Consumer Staples Select Sector SPDR)
   - Employment Indicators: Retail trade, food services (BLS)
   - News Impact: FDA actions, commodity prices, consumer spending

6. **Health Care (35)**
   - Pharmaceuticals (Pfizer, Johnson & Johnson, Merck, Eli Lilly)
   - Biotechnology (Amgen, Gilead, Biogen, Moderna)
   - Medical Devices (Medtronic, Abbott Labs, Stryker)
   - Health Care Providers (UnitedHealth, CVS Health, Cigna)
   - ETFs: XLV (Health Care Select Sector SPDR)
   - Employment Indicators: Healthcare jobs, hospital employment (BLS)
   - News Impact: FDA approvals, drug trials, healthcare policy, Medicare/Medicaid

7. **Financials (40)**
   - Banks (JPMorgan, Bank of America, Wells Fargo, Citigroup)
   - Insurance (Berkshire Hathaway, Progressive, Allstate)
   - Capital Markets (Goldman Sachs, Morgan Stanley, BlackRock)
   - Consumer Finance (American Express, Visa, Mastercard)
   - REITs (Real Estate Investment Trusts)
   - ETFs: XLF (Financial Select Sector SPDR)
   - Employment Indicators: Financial activities employment (BLS)
   - News Impact: Fed rate decisions, banking regulations, credit conditions

8. **Information Technology (45)**
   - Software (Microsoft, Oracle, Salesforce, Adobe)
   - Hardware (Apple, Dell, HP Inc.)
   - Semiconductors (NVIDIA, Intel, AMD, Qualcomm, Broadcom)
   - IT Services (Accenture, IBM, Cognizant)
   - Communications Equipment (Cisco, Arista Networks)
   - ETFs: XLK (Technology Select Sector SPDR)
   - Employment Indicators: Information sector jobs, tech sector layoffs/hiring
   - News Impact: Chip shortages, AI developments, tech regulations, earnings

9. **Communication Services (50)**
   - Media & Entertainment (Disney, Netflix, Comcast, Paramount)
   - Interactive Media (Meta, Google/Alphabet, Twitter/X)
   - Telecommunications (Verizon, AT&T, T-Mobile)
   - ETFs: XLC (Communication Services Select Sector SPDR)
   - Employment Indicators: Information sector, media jobs (BLS)
   - News Impact: FCC regulations, streaming wars, social media policy

10. **Utilities (55)**
    - Electric Utilities (NextEra Energy, Duke Energy, Southern Company)
    - Gas Utilities (Sempra Energy, Atmos Energy)
    - Water Utilities (American Water Works)
    - Renewable Energy Utilities
    - ETFs: XLU (Utilities Select Sector SPDR)
    - Employment Indicators: Utilities sector employment (BLS)
    - News Impact: Energy policy, rate regulations, renewable mandates

11. **Real Estate (60)**
    - REITs - Residential (AvalonBay, Equity Residential)
    - REITs - Commercial (Simon Property Group, Boston Properties)
    - REITs - Healthcare (Welltower, Healthpeak)
    - REITs - Industrial (Prologis, Duke Realty)
    - Real Estate Management & Development
    - ETFs: XLRE (Real Estate Select Sector SPDR)
    - Employment Indicators: Construction jobs, real estate agents (BLS)
    - News Impact: Interest rates, housing starts, commercial occupancy

**Sector Analysis Requirements:**

**A. News Impact by Sector:**
- System MUST classify news events by affected sector(s)
- Track sector-specific sentiment from news analysis
- Identify cross-sector impacts (e.g., oil prices → transportation costs → airlines)
- Monitor sector rotation signals from news flow

**B. Employment Data by Sector:**
- Map BLS industry codes to GICS sectors
- Track sector-specific employment trends (hiring vs. layoffs)
- Monitor mass layoff events by sector (WARN Act data)
- Identify sector strength/weakness from employment data
- Use employment as leading indicator for sector performance

**C. Sector Correlation and Rotation:**
- Calculate inter-sector correlations
- Track sector ETF performance relative to S&P 500
- Identify defensive vs. cyclical sector rotation
- Monitor sector leadership changes
- Risk-on vs. risk-off sector positioning

**D. Database Schema Requirements:**
```sql
-- Sector master table
CREATE TABLE sectors (
    sector_id INTEGER PRIMARY KEY,
    sector_code INTEGER NOT NULL,  -- GICS code
    sector_name VARCHAR NOT NULL,
    sector_category VARCHAR NOT NULL,  -- Defensive, Cyclical, Sensitive
    etf_ticker VARCHAR,
    description TEXT
);

-- Company to sector mapping
CREATE TABLE company_sectors (
    ticker VARCHAR PRIMARY KEY,
    sector_id INTEGER REFERENCES sectors(sector_id),
    industry VARCHAR,
    sub_industry VARCHAR,
    market_cap_category VARCHAR  -- Large, Mid, Small cap
);

-- Sector employment data
CREATE TABLE sector_employment (
    id INTEGER PRIMARY KEY,
    sector_id INTEGER REFERENCES sectors(sector_id),
    report_date DATE NOT NULL,
    employment_count INTEGER,
    unemployment_rate DOUBLE,
    job_openings INTEGER,
    layoff_count INTEGER,
    hiring_count INTEGER,
    data_source VARCHAR,  -- BLS, WARN, Layoffs.fyi
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sector news sentiment
CREATE TABLE sector_news_sentiment (
    id INTEGER PRIMARY KEY,
    sector_id INTEGER REFERENCES sectors(sector_id),
    news_date TIMESTAMP NOT NULL,
    sentiment_score DOUBLE,  -- -1.0 to 1.0
    news_count INTEGER,
    major_events TEXT[],
    impact_magnitude VARCHAR  -- High, Medium, Low
);
```

**E. API Integration Requirements for Tier 1:**

**BLS API Integration (MANDATORY):**
```python
# Example BLS API call for sector employment
import requests

BLS_API_KEY = os.getenv('BLS_API_KEY')  # Free API key from BLS
BLS_BASE_URL = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

# Series IDs for major sectors (CES - Current Employment Statistics)
SECTOR_SERIES = {
    'mining_logging': 'CES1000000001',  # Energy/Materials
    'construction': 'CES2000000001',     # Industrials
    'manufacturing': 'CES3000000001',    # Industrials/Materials
    'trade_transport': 'CES4000000001',  # Consumer Discretionary
    'information': 'CES5000000001',      # Technology/Communication
    'financial': 'CES5500000001',        # Financials
    'professional': 'CES6000000001',     # IT Services
    'education_health': 'CES6500000001', # Health Care
    'leisure': 'CES7000000001',          # Consumer Discretionary
}

def fetch_sector_employment(series_id, start_year, end_year):
    """Fetch employment data from BLS API"""
    payload = {
        'seriesid': [series_id],
        'startyear': str(start_year),
        'endyear': str(end_year),
        'registrationkey': BLS_API_KEY
    }
    response = requests.post(BLS_BASE_URL, json=payload)
    return response.json()
```

**F. Sector Decision-Making Integration:**

- **Sector Rotation Strategy:** Identify sectors transitioning from weak to strong
- **Defensive Positioning:** Rotate to Utilities, Consumer Staples, Healthcare during downturns
- **Cyclical Positioning:** Rotate to Energy, Financials, Industrials during recovery
- **Employment Signals:**
  - Tech layoffs → Avoid/short tech sector
  - Healthcare hiring → Long healthcare sector
  - Retail hiring surge → Long consumer discretionary
  - Manufacturing decline → Rotate out of industrials

### 3.3 Core Features

#### 3.3.1 Multi-Source Data Ingestion
- Real-time data collection from all sources
- Data normalization and standardization
- Quality assurance and validation
- Historical data storage and versioning
- API integration with data providers

#### 3.3.2 Natural Language Processing (NLP) & Sentiment Analysis

**⚠️ CRITICAL FOR DAY TRADING:** Sentiment drives intraday price movements more than fundamentals. System MUST capture and quantify "animal spirits" in real-time.

**Sentiment Analysis (MANDATORY):**
- **News Sentiment:**
  - Real-time sentiment scoring (-1.0 to +1.0) for all news articles
  - Tone analysis: bearish, neutral, bullish
  - Urgency detection: breaking news, developing story, background
  - Emotional intensity: panic, fear, caution, optimism, euphoria
  - Track sentiment velocity (how fast sentiment changes)

- **Social Media Sentiment (Reddit, Twitter/X, StockTwits):**
  - **"Meme stock" detection:** Identify viral momentum stocks
  - **Retail trader sentiment:** WSB (WallStreetBets) mood tracking
  - **FOMO indicators:** Rapid increase in mentions, hashtag trending
  - **Panic indicators:** Sudden negative sentiment spikes
  - Volume of mentions (viral attention correlates with volatility)
  - Sentiment divergence: social media vs institutional news

- **Options Sentiment Indicators:**
  - Put/Call ratio (fear vs greed)
  - Implied volatility skew (demand for puts vs calls)
  - Unusual options activity (large bets indicate conviction)
  - Options flow: bullish vs bearish positioning

- **Market-Wide Sentiment:**
  - VIX (fear gauge) and VIX futures
  - Advance/decline ratios
  - New highs vs new lows
  - Market breadth indicators
  - Sector rotation (risk-on vs risk-off)

**Behavioral Finance Patterns to Detect:**
- **Overreaction:** Extreme sentiment → mean reversion opportunity
- **Underreaction:** Delayed response → momentum continuation
- **Herding:** Crowds following crowds (exploitable with timing)
- **Anchoring:** Traders stuck on old price levels (resistance/support)
- **Confirmation Bias:** Ignoring contradictory information (creates mispricing)
- **Recency Bias:** Overweighting recent events (volatility overestimation)

**Traditional NLP Features (Still Important):**
- Entity recognition (companies, people, products)
- Event extraction and classification
- Topic modeling and trend detection
- Language translation for international sources
- Named entity disambiguation

#### 3.3.3 Impact Prediction Engine
- Company-specific impact prediction
- Sector and industry impact analysis
- Magnitude estimation (% expected movement)
- Confidence scoring for predictions
- Time-to-impact estimation

#### 3.3.4 Impact Graph Generation
- **Node Types:**
  - Events (news, announcements, decisions)
  - Companies (affected entities)
  - Sectors and industries
  - Economic indicators
  - Products and services
- **Edge Types:**
  - Direct relationships (supplier-customer)
  - Correlation relationships
  - Causal relationships (event → impact)
  - Competitive relationships
- **Edge Weights:**
  - Strength of relationship (0.0 to 1.0)
  - Confidence level
  - Historical correlation coefficient
- **Graph Analysis:**
  - Multi-hop impact propagation
  - Central node identification
  - Cluster detection (related companies)
  - Shortest path analysis (causal chains)

#### 3.3.5 Machine Learning Models
- **Supervised Learning:**
  - Gradient Boosting (XGBoost, LightGBM) for impact magnitude
  - Random Forests for classification (positive/negative impact)
  - Neural Networks for complex pattern recognition
  - **✅ DEPLOYED: Price Predictor v3.0** (60-feature neural network)
    - Architecture: [256, 128, 64, 32] with LeakyReLU
    - Features: 60 (identification, time, treasury, Greeks, sentiment, price, momentum, volatility, interactions, directionality)
    - Training: 24,300 samples, DirectionalLoss (90% direction + 10% MSE)
    - Performance: 56.3% (5-day), 56.6% (20-day) accuracy - **PROFITABLE** (>55% threshold)
    - Inference: ONNX Runtime with AVX2 SIMD normalization (8x speedup)
    - Integration: C++23 module in `src/market_intelligence/price_predictor.cppm`
- **Deep Learning:**
  - Transformer models (BERT, GPT-based) for NLP
  - LSTM/GRU for time series prediction
  - Graph Neural Networks (GNN) for impact graph analysis
  - Attention mechanisms for feature importance
- **Ensemble Methods:**
  - Model averaging and stacking
  - Confidence-weighted predictions
  - Bayesian model combination

### 3.4 Functional Requirements

#### FR1.1: Data Collection
- System SHALL collect data from all specified sources in real-time
- System SHALL handle API rate limits and implement retry logic
- System SHALL store raw data for audit and reprocessing
- System SHALL timestamp all data with UTC timezone

#### FR1.2: Data Processing
- System SHALL process incoming data within 1 second of receipt
- System SHALL extract entities, events, and relationships
- System SHALL calculate sentiment scores with confidence intervals
- System SHALL normalize data to standardized format

#### FR1.3: Impact Analysis
- System SHALL predict impact on relevant companies within 5 seconds
- System SHALL generate impact magnitude estimates with confidence scores
- System SHALL identify both direct and indirect impacts (multi-hop)
- System SHALL prioritize analysis based on event significance

#### FR1.4: Graph Generation
- System SHALL generate impact graphs with nodes and weighted edges
- System SHALL update graphs incrementally as new data arrives
- System SHALL identify connected components and clusters
- System SHALL calculate centrality metrics for all nodes

#### FR1.5: Output & Alerts
- System SHALL provide API access to predictions and graphs
- System SHALL generate alerts for high-impact events
- System SHALL rank opportunities by expected return and confidence
- System SHALL provide explanations for predictions (interpretability)

### 3.5 Technical Requirements

#### TR1.1: Performance
- Data ingestion latency < 500ms
- NLP processing latency < 2 seconds per document
- Impact prediction latency < 5 seconds
- Graph update latency < 3 seconds
- Support for 10,000+ concurrent data streams

#### TR1.2: Accuracy
- Sentiment analysis accuracy > 85%
- Entity recognition F1 score > 0.90
- Impact direction prediction accuracy > 70%
- Impact magnitude MAPE < 20%

#### TR1.3: Data Management
- Store minimum 10 years of historical data
- Support for petabyte-scale data storage
- Real-time data access with < 10ms query latency
- Data versioning and audit trails

---

## 4. Sub-Project 2: Trading Correlation Analysis Tool

### 4.1 Overview

A time-series analysis system that discovers statistical relationships between securities across multiple timeframes, including time-lagged correlations to identify leading and lagging indicators.

**📋 Complete Architecture:** For detailed system architecture, correlation algorithms, fluent API design, database schemas, API specifications, and implementation guidelines, see the **[Trading Correlation Analysis Tool - Architecture Design Document](./architecture/trading-correlation-analysis-tool.md)**.

The architecture document includes:
- High-level system architecture with Mermaid diagrams (single-machine 64-core deployment)
- Parallel correlation algorithms (C++23 + MPI + OpenMP + Intel MKL)
- Fluent composable API design with UPC++ for distributed computing
- CUDA-accelerated matrix correlation (cuBLAS, cuFFT)
- Time-lagged cross-correlation with FFT optimization
- Complete database schemas (PostgreSQL + TimescaleDB, DuckDB analytical views)
- REST/WebSocket API specifications (OpenAPI 3.0)
- Shared infrastructure design with Market Intelligence Engine (cost optimization)
- Loose coupling architecture (separate processes, API-only communication)
- Quick 2-4 week POC guide (Python + DuckDB + CUDA)
- Consistent toolset across both tools
- Performance benchmarks and optimization strategies

**Tier 1 Implementation Focus:**
- Start with **Python + DuckDB** for quick POC (Week 1-2, $0 cost)
- Add **CUDA acceleration** with CuPy for GPU speedup (Week 3)
- Add **PostgreSQL storage** for results (Week 4)
- Implement **C++23/MPI/OpenMP** production engine after POC validation (Month 2+)

**Shared Infrastructure:**
- Same 64-128 core server as Market Intelligence Engine
- Cores 32-63 allocated to Correlation Tool
- Shared PostgreSQL/Redis/GPU (separate namespaces)
- **Cost savings:** $3,450-9,700 one-time, $150-280/month vs separate servers

### 4.2 Data Sources

#### 4.2.1 Historical Market Data
- **Price Data** - OHLCV for all securities (minimum 10 years)
- **Adjusted Prices** - Split and dividend adjusted
- **Volume Data** - Trading volume, dollar volume
- **Volatility Data** - Historical volatility calculations
- **Market Capitalization** - Historical market cap data

#### 4.2.2 Historical News Data (Optional)
- **News Archives** - Historical news articles with timestamps
- **Corporate Events** - Historical earnings, announcements
- **Economic Events** - Historical economic releases
- **Sentiment Archives** - Historical sentiment scores

### 4.3 Analysis Dimensions

#### 4.3.1 Timeframe Analysis Levels

**Intra-Day Correlations:**
- 1-minute resolution
- 5-minute resolution
- 15-minute resolution
- 1-hour resolution
- Market open/close effects

**Inter-Day (Within Week):**
- Daily returns
- Day-of-week effects
- Week-over-week patterns
- Monday vs. Friday effects

**Intra-Month Correlations:**
- Weekly aggregated data
- Beginning/middle/end of month patterns
- Options expiration effects
- Earnings season patterns

**Intra-Quarter Correlations:**
- Monthly aggregated data
- Quarter-over-quarter patterns
- Seasonal industry rotations
- Quarterly reporting cycles

#### 4.3.2 Correlation Types

**Contemporaneous Correlations:**
- Pearson correlation coefficients
- Spearman rank correlations
- Rolling correlations (windows: 30, 60, 90, 180, 360 days)
- Dynamic conditional correlations

**Time-Lagged Correlations (Convolution Analysis):**

**⚠️ CRITICAL FOR PROFITABILITY:** Accurately predicting the LAG between cause and effect is KEY to profitable trading. Knowing WHEN an event will impact prices is more valuable than knowing IF it will impact.

**Lag Prediction Requirements:**
- **Cross-correlation functions:** Identify optimal lag periods
- **Lag periods to analyze:**
  - Intra-day: 1-60 minutes (high-frequency cause-effect)
  - Inter-day: 1-30 days (medium-term propagation)
  - Seasonal: weeks to months (long-term structural relationships)

**Examples of Profitable Lag Exploitation:**
- **NVDA earnings → AMD stock (15-minute lag):**
  - NVDA reports strong earnings at 4:00 PM
  - System predicts AMD will follow in 15 minutes
  - Buy AMD calls at 4:01 PM, sell at 4:20 PM (exploit lag)
  - Profit from predictable delayed reaction

- **Fed rate decision → Bank stocks (30-minute lag):**
  - Fed announces rate cut at 2:00 PM
  - System predicts bank stocks rally in 30 minutes (delayed analysis)
  - Enter positions immediately, exit after delayed reaction
  - Capture momentum before crowd realizes impact

- **Oil price spike → Energy sector (2-hour lag):**
  - Crude oil jumps 5% at market open
  - System predicts XLE (energy ETF) lags by 2 hours
  - Trade energy stocks/options in the gap period

- **S&P 500 futures → Individual stocks (5-minute lag):**
  - ES futures spike down at 9:35 AM
  - System predicts individual stocks follow in 5 minutes
  - Short high-beta stocks immediately
  - Cover when correlation catches up

**Lag Prediction Profitability Formula:**
```
Profit Opportunity = Price Change × (1 - e^(-lag_accuracy))

Where:
  - lag_accuracy: How precisely we predict the lag (0-1 scale)
  - Perfect prediction (lag_accuracy = 1): Capture full move
  - No prediction (lag_accuracy = 0): Random entry, no edge
  - Typical accuracy (lag_accuracy = 0.7): Capture 70% of move

Example:
  - Expected move: $2.00 per share
  - Lag accuracy: 80%
  - Capturable profit: $2.00 × (1 - e^(-0.8)) = $2.00 × 0.55 = $1.10
  - With perfect timing: capture $1.10 of $2.00 move
```

**Lag Timing Features (MANDATORY):**
- **Predicted lag:** Expected time delay (minutes, hours, days)
- **Lag confidence:** How confident in the timing (0-1 scale)
- **Historical lag distribution:** Show variability (sometimes 10min, sometimes 30min)
- **Optimal entry window:** When to enter (immediately vs wait for confirmation)
- **Optimal exit window:** When lag effect exhausted (mean reversion begins)
- **Volume-adjusted lag:** Higher volume = faster propagation
- **Sentiment-adjusted lag:** Extreme sentiment accelerates lag (panic spreads faster)
- Leading indicators (Stock A predicts Stock B)
- Lagging indicators (Stock A follows Stock B)
- Optimal lag identification

**Positive and Negative Correlations:**
- Direct relationships (ρ > 0.5)
- Inverse relationships (ρ < -0.5)
- Weak correlations for outlier detection
- Correlation stability analysis

### 4.4 Core Features

#### 4.4.1 Correlation Computation Engine
- Parallel computation for large security universes
- Rolling window calculations
- Online (incremental) correlation updates
- Statistical significance testing (p-values)
- Correlation clustering algorithms

#### 4.4.2 Time-Lagged Analysis
- Cross-correlation function computation
- Optimal lag period identification
- Lead-lag relationship discovery
- Granger causality testing
- Transfer entropy analysis

#### 4.4.3 Pattern Recognition
- Correlation regime detection (stable/unstable periods)
- Breakpoint detection (when correlations change)
- Correlation network topology analysis
- Community detection in correlation networks
- Hierarchical clustering of securities

#### 4.4.4 Visualization & Reporting
- Correlation matrices (heatmaps)
- Time-series correlation plots
- Network graphs of correlations
- Lead-lag relationship diagrams
- Statistical significance indicators

#### 4.4.5 Backtesting Framework
- Historical correlation-based strategy testing
- Pairs trading backtests
- Basket trading simulations
- Performance metrics calculation
- Risk-adjusted returns analysis

### 4.5 Functional Requirements

#### FR2.1: Data Ingestion
- System SHALL ingest historical market data for analysis
- System SHALL support data from multiple exchanges and markets
- System SHALL handle missing data and outliers appropriately
- System SHALL normalize prices for splits and dividends
- **System SHALL track all corporate actions with effective dates:**
  - Dividends (declaration, ex-dividend, record, payment dates)
  - Stock splits (announcement, effective date, split ratio)
  - Mergers and acquisitions (announcement, shareholder approval, close date)
  - Spin-offs (distribution date, cost basis allocation)
- **System SHALL fetch and store margin interest rates:**
  - Daily margin balance tracking
  - Current margin rate tiers (updated quarterly minimum)
  - Historical margin rates for backtesting accuracy

#### FR2.2: Correlation Calculation
- System SHALL calculate correlations for all timeframes
- System SHALL compute rolling correlations with configurable windows
- System SHALL identify statistically significant correlations (p < 0.05)
- System SHALL update correlations incrementally with new data

#### FR2.3: Time-Lag Analysis
- System SHALL compute cross-correlation functions for all pairs
- System SHALL identify optimal lag periods automatically
- System SHALL distinguish between leading and lagging relationships
- System SHALL test for Granger causality

#### FR2.4: Pattern Detection
- System SHALL detect correlation regime changes
- System SHALL identify correlation breakpoints
- System SHALL cluster securities by correlation patterns
- System SHALL detect anomalous correlation behavior

#### FR2.5: Output & Integration
- System SHALL provide correlation data via API
- System SHALL generate correlation matrices on demand
- System SHALL export lead-lag relationships for trading engine
- System SHALL provide confidence scores for all relationships

### 4.6 Technical Requirements

#### TR2.1: Performance
- Correlation calculation for 5,000 securities in < 60 seconds
- Rolling correlation updates in < 5 seconds
- Cross-correlation computation in < 10 seconds per pair
- Support for real-time correlation monitoring

#### TR2.2: Accuracy
- Correlation coefficient precision to 4 decimal places
- Statistical significance testing with multiple comparison correction
- Robust correlation estimators to handle outliers
- Confidence intervals for all correlation estimates

#### TR2.3: Scalability
- Support for 10,000+ securities
- Handle 10+ years of minute-level data
- Parallel processing on multi-core systems
- Distributed computation for large-scale analysis

---

## 5. Sub-Project 3: Intelligent Trading Decision Engine

### 5.1 Overview

A high-performance machine learning system that integrates insights from the Market Intelligence Engine and Correlation Analysis Tool to select trading opportunities, predict movements, and execute trading strategies. **Initial focus is on algorithmic options day trading**, with stock trading strategies to be developed subsequently.

**📋 Complete Architecture:** For detailed system architecture, mathematical foundations (RL, DNN, GNN), explainability design, low-cost data sources, charting tools, and implementation guidelines, see the **[Intelligent Trading Decision Engine - Architecture Design Document](./architecture/intelligent-trading-decision-engine.md)**.

The architecture document includes:
- High-level system architecture with Mermaid diagrams (shared 64-core server deployment)
- Mathematical foundations: Reinforcement Learning (PPO, DQN, A3C), Deep Neural Networks (Transformers, LSTMs), Graph Neural Networks
- **Explainability & Interpretability Design** - NOT a black box! Every decision explained
  - SHAP (SHapley Additive exPlanations) for feature attribution
  - Attention mechanism visualization
  - Decision tree extraction from neural networks
  - Multi-level explanations (5 levels from executive summary to full analysis)
  - Natural language generation for human-readable rationale
- Fluent composable API design (C++23 builder pattern, consistent with other tools)
- C++23 ultra-low latency order execution engine (< 10ms order submission)
- Risk management with microsecond-level checks
- Complete database schemas (PostgreSQL + TimescaleDB, DuckDB backtesting)
- REST/WebSocket API specifications (OpenAPI 3.0)
- **Low-cost historical data collection** (Yahoo Finance, FRED, SEC - all free)
- **Charting & visualization stack** (Lightweight Charts, Plotly/Dash, Streamlit - all free)
- Real-time trading dashboard with decision explanations
- Message format design (JSON + zstd, consistent with other tools)
- Shared infrastructure with Market Intelligence and Correlation tools (cores 43-63)
- Quick 4-6 week POC guide (Python + DuckDB + simple models)
- Portfolio optimization mathematics (Markowitz, Kelly Criterion)
- Performance benchmarks: < 300ms end-to-end decision latency

**Tier 1 Implementation Focus:**
- Start with **simple rule-based logic** for POC (Week 1, $0 cost)
- Add **XGBoost/LightGBM** with SHAP explainability (Week 2-3)
- Add **Streamlit dashboard** for visualization (Week 4)
- Implement **RL agent + DNN** after POC validation (Month 2+)
- **Explainability-first:** Every decision must be explainable to humans

**Shared Infrastructure:**
- Same 64-128 core server as other two tools
- Cores 43-63 (21 cores) allocated to Trading Decision Engine
- Shared PostgreSQL/Redis/GPU (separate namespaces: td:*)
- **Zero additional hardware cost** - uses existing server

**Key Differentiator:**
- **100% Explainable** - No black box AI
- Every decision traceable with human-readable rationale
- SHAP values, attention weights, decision trees
- What-if scenario analysis
- Full audit trail

### 5.2 Input Sources

**⚠️ FUNDAMENTAL PRINCIPLE:** Trading decisions must balance rational analysis (fundamentals, correlations) with behavioral factors (sentiment, momentum, "animal spirits"). Day trading especially requires understanding crowd psychology.

#### 5.2.1 From Market Intelligence Engine
- Impact predictions with confidence scores
- Event classifications and severity
- Impact graphs with relationship strengths
- **Sentiment scores and trends (CRITICAL FOR DAY TRADING):**
  - News sentiment: bearish, neutral, bullish
  - Social media sentiment (Reddit, Twitter/X, StockTwits)
  - Sentiment velocity (rate of change)
  - Sentiment divergence (retail vs institutional)
  - Emotional intensity (panic, fear, caution, optimism, euphoria)
- News significance rankings

#### 5.2.2 From Correlation Analysis Tool
- Correlation matrices (all timeframes)
- Lead-lag relationships
- Correlation regime indicators
- Statistical significance measures
- Correlation stability metrics

#### 5.2.3 Additional Market Data
- Real-time options chains (complete strikes and expirations)
- Real-time underlying stock price feeds
- Order book data (options and stocks)
- Trading volume (options and stocks)
- Implied volatility surfaces
- Options Greeks (delta, gamma, theta, vega, rho)
- **Technical indicators (momentum and sentiment proxies):**
  - RSI (overbought/oversold = extreme sentiment)
  - MACD (momentum direction)
  - Bollinger Bands (volatility and extremes)
  - Volume indicators (conviction and participation)
  - Money flow indicators (smart money vs dumb money)

#### 5.2.4 Behavioral & Sentiment Indicators (CRITICAL FOR DAY TRADING)
**⚠️ NEW CATEGORY:** Quantifying "animal spirits" for short-term trading decisions.

- **Market-Wide Sentiment:**
  - VIX and VIX futures (fear gauge)
  - Fear & Greed Index (CNN Business)
  - Put/Call ratios (CBOE, equity-only)
  - Advance/decline line
  - New highs vs new lows
  - Market breadth (% stocks above 50-day MA)

- **Stock-Specific Sentiment:**
  - Social media mentions (volume and tone)
  - Unusual options activity (smart money positioning)
  - Analyst rating changes (herd following)
  - Insider trading activity (confidence indicator)
  - Short interest levels (contrarian indicator)
  - Retail order flow (sentiment proxy)

- **Order Flow & Volume Analysis (Smart Money Tracking):**
  - **Volume surge detection:** Volume > 2x average indicates conviction
  - **Block trades:** Large institutional orders (>10,000 shares)
  - **Dark pool activity:** Off-exchange institutional buying/selling
  - **Tape reading:** Bid vs ask volume (buying vs selling pressure)
  - **Time & sales:** Aggressive buying (lifting asks) vs selling (hitting bids)
  - **Institutional flow:** Track large money movements
    - Pension funds, hedge funds, mutual funds
    - 13F filings for quarterly positions
    - Whale watching: large account activity
  - **Retail vs institutional:** Differentiate smart money from dumb money
    - Retail: small trades, market orders, emotional timing
    - Institutional: large trades, limit orders, strategic timing
  - **Smart money indicators:**
    - Institutional accumulation during selloffs (bullish)
    - Institutional distribution during rallies (bearish)
    - Follow the smart money, fade retail panic

- **Momentum Indicators (Behavioral Proxies):**
  - Price momentum (rate of change)
  - Volume surge (attention and conviction)
  - Breakout patterns (herd behavior)
  - Gap up/down (overnight sentiment shift)
  - Intraday volatility (panic or euphoria)

### 5.2.4 Supported Trading Instruments & Valuation Methods

**CRITICAL:** Complete specification of all tradeable instruments and their valuation methodologies.

#### Stock Trades

**Order Types Supported:**
- **Market Orders:** Immediate execution at current market price
- **Limit Orders:** Execute only at specified price or better
- **Stop Orders:** Trigger market order when price reaches stop level
- **Stop-Limit Orders:** Trigger limit order when price reaches stop level
- **Trailing Stop:** Dynamic stop that follows price movement
- **Good-Till-Canceled (GTC):** Order remains active until filled or canceled
- **Day Orders:** Cancel if not filled by end of trading day
- **Fill-or-Kill (FOK):** Execute entire order immediately or cancel
- **Immediate-or-Cancel (IOC):** Execute immediately, cancel unfilled portion

**Stock Trade Types:**
- **Long Positions:** Buy and hold (bullish)
- **Short Positions:** Sell borrowed shares (bearish)
- **Pairs Trading:** Long/short correlated pairs
- **Basket Trading:** Multiple stocks as single trade
- **Dollar-Cost Averaging:** Systematic purchases over time

**Valuation:** Fundamental (DCF, P/E, PEG) + Technical + ML predictions

#### Money Market Instruments

**Cash Management:**
- **Money Market Funds:** Sweep uninvested cash (1-2% yield)
- **Treasury Bills:** Short-term government debt (< 1 year)
- **Commercial Paper:** Corporate short-term debt
- **Certificates of Deposit:** Bank time deposits

**Usage in Platform:**
- Uninvested cash → Money market sweep (automatic)
- Margin collateral → T-Bills (higher quality)
- Emergency reserves → Ultra-short duration funds
- **NOT actively traded** - used for cash management only

#### Options Trades (PRIMARY FOCUS - Phase 1)

**Basic Options:**
- **Call Options:** Right to buy at strike price
- **Put Options:** Right to sell at strike price
- **American Style:** Exercise any time before expiration
- **European Style:** Exercise only at expiration

**Spread Strategies:**
- **Vertical Spreads:**
  - Bull Call Spread: Long lower strike call, short higher strike call
  - Bear Put Spread: Long higher strike put, short lower strike put
  - Defined risk/reward, lower capital requirement

- **Calendar (Time) Spreads:**
  - Long later expiration, short near expiration
  - Profit from time decay differential
  - Neutral to slightly directional

- **Diagonal Spreads:**
  - Different strikes AND different expirations
  - Flexible risk/reward profiles

**Volatility Strategies:**
- **Straddle:** Long call + long put at same strike (expect big move)
- **Strangle:** Long OTM call + long OTM put (cheaper than straddle)
- **Iron Condor:** Sell OTM put spread + sell OTM call spread (profit from range)
- **Iron Butterfly:** Sell ATM straddle + buy OTM strangle (range-bound profit)
- **Butterfly Spread:** Limited risk, limited profit (3 strikes)

**Greeks-Based Strategies:**
- **Delta-Neutral:** Hedge directional risk, profit from volatility
- **Gamma Scalping:** Profit from gamma as underlying moves
- **Theta Harvesting:** Sell options, collect time decay
- **Vega Plays:** Trade implied volatility changes
- **Rho Strategies:** Interest rate sensitivity plays (rare)

**Advanced Strategies:**
- **Ratio Spreads:** Unequal number of long/short options
- **Back Spreads:** Reverse ratio spreads
- **Box Spreads:** Arbitrage play (4-leg synthetic loan)
- **Conversions/Reversals:** Arbitrage between stock and options
- **Synthetic Positions:** Replicate stock with options (synthetic long/short)

#### Options Valuation Methods

**PRIMARY: Trinomial Tree Model**

Mathematical Foundation:
```
Trinomial Tree for American Options:
  - At each node, price can move: Up, Flat, or Down
  - More accurate than binomial for short time steps
  - Better for early exercise determination
  - Handles American-style options correctly

Parameters:
  S = Current stock price
  K = Strike price
  T = Time to expiration
  r = Risk-free rate
  σ = Volatility (implied or historical)
  q = Dividend yield

Price movements per step (Δt):
  u = e^(σ√(3Δt))     (up)
  m = 1                (middle/flat)
  d = e^(-σ√(3Δt))    (down)

Probabilities:
  p_u = ((e^((r-q)Δt/2) - e^(-σ√(Δt/3))) / (e^(σ√(Δt/3)) - e^(-σ√(Δt/3))))^2
  p_m = 1 - p_u - p_d
  p_d = ((e^(σ√(Δt/3)) - e^((r-q)Δt/2)) / (e^(σ√(Δt/3)) - e^(-σ√(Δt/3))))^2

Backward induction:
  V(S,t) = e^(-rΔt)[p_u·V(Su,t+Δt) + p_m·V(Sm,t+Δt) + p_d·V(Sd,t+Δt)]

Early exercise check (American):
  V(S,t) = max(Intrinsic Value, Continuation Value)

Implementation: C++23 for speed, parallelize tree construction with OpenMP
```

**SECONDARY: Black-Scholes Model (Short-Term European Options)**

Mathematical Foundation:
```
Black-Scholes Formula:
  C = S·N(d₁) - K·e^(-rT)·N(d₂)  (Call)
  P = K·e^(-rT)·N(-d₂) - S·N(-d₁) (Put)

Where:
  d₁ = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
  d₂ = d₁ - σ√T
  N(x) = Cumulative normal distribution

Greeks (analytic formulas):
  Delta (Δ): ∂V/∂S
  Gamma (Γ): ∂²V/∂S²
  Theta (Θ): ∂V/∂t
  Vega (ν): ∂V/∂σ
  Rho (ρ): ∂V/∂r

When to use:
  - European options only (no early exercise)
  - Short-term options (< 30 days)
  - Liquid markets (tight bid-ask)
  - Fast computation needed (real-time pricing)

Implementation: C++23 with Intel MKL for N(x) calculation
```

**TERTIARY: Monte Carlo Simulation (Complex/Exotic Options)**

Used for:
- Path-dependent options (Asian, lookback)
- Barrier options (knock-in, knock-out)
- Multi-asset options (baskets, rainbow)
- Validation of tree methods

Implementation: CUDA for parallel scenarios (100K+ paths in milliseconds)

**Greeks Calculation:**
- **Numerical Methods:** Finite differences from tree/MC results
- **Analytic (Black-Scholes):** When applicable
- **Automatic Differentiation:** For ML-based pricing

#### Broker Platform Selection (Tier 1 POC - Real Money)

**CRITICAL:** Tier 1 POC uses existing Schwab account with $30k margin. This is REAL MONEY trading - profitability from day one is essential.

**TIER 1 POC BROKER: Charles Schwab**

**Account Details:**
- **Existing Account:** $30,000 margin trading account
- **Account Type:** Margin account (pattern day trading enabled)
- **API Access:** Schwab Developer API (already configured)
- **Commission:** $0 stock, $0.65/contract options
- **Data Access:** Free real-time market data (Level 1)
- **Options Support:** Full (all strategies supported)
- **Existing Integration:** SchwabFirstAPI repository

**Schwab API Integration:**

**📋 Complete Schwab API Integration Guide:** For detailed C++23 implementation design, OAuth 2.0 token management, security implications, and integration patterns, see the **[Schwab API Integration - Implementation Guide](./architecture/schwab-api-integration.md)**.

The Schwab integration document includes:
- Complete OAuth 2.0 authentication flow with token refresh
- C++23 module architecture for Schwab API client
- Secure credential and token storage strategies
- Rate limiting implementation
- HTTP client library evaluation (cpr, libcurl, Boost.Beast)
- JSON parsing with nlohmann/json
- Thread-safe token management with std::atomic
- Security best practices for $30k account
- CMake 4.1.2 build configuration
- Integration phases and implementation checklist
- Risk mitigations for real money trading

Reference Repository: https://github.com/oldboldpilot/SchwabFirstAPI
- Existing options trading intelligence implementation
- Proven Schwab API connectivity
- Market data access patterns
- Options chain retrieval
- Order execution workflows
- OAuth patterns and credential management
- Can reuse/adapt existing code for BigBrotherAnalytics C++23 implementation

**Schwab API Capabilities:**
- **REST API:** Orders, positions, account data, market data
- **Streaming API:** Real-time quotes, options chains, Level 1 data
- **Market Data:** Free real-time with funded account
- **Options Chains:** Complete chains with all strikes and expirations
- **Greeks:** Delta, Gamma, Theta, Vega provided by API
- **Historical Data:** OHLC, options history
- **Account Management:** Balances, positions, buying power
- **Order Types:** All standard types (market, limit, stop, etc.)

**Tier 1 POC Risk Management ($30k Account):**

**⚠️ CRITICAL: Daily Profitability Requirements**

With $30k real money at stake, the system MUST be profitable every single day:

```
Daily Risk Limits (Conservative):
  - Maximum daily loss: $900 (3% of capital)
  - Maximum position size: $1,500 (5% of capital)
  - Maximum positions: 10 concurrent
  - Stop loss: MANDATORY on every trade
  - Position size via Kelly Criterion (conservative 50% of full Kelly)

Daily Profitability Goals:
  - Minimum target: $150/day (0.5% daily return)
  - Realistic target: $300-600/day (1-2% daily return)
  - Stretch target: $900+/day (3%+ daily return)

Risk Management Strategy:
  1. SIMULATE every trade before execution (Monte Carlo)
  2. Only trade if expected value > $50
  3. Only trade if probability of profit > 70%
  4. HARD stop loss at 3% daily loss ($900)
  5. Close all positions before market close (no overnight risk initially)
  6. Conservative position sizing (50% Kelly)
  7. Diversify across 5-10 uncorrelated trades
  8. Real-time monitoring with automatic stop losses
```

**Integration with SchwabFirstAPI:**
```
BigBrotherAnalytics will integrate with existing SchwabFirstAPI codebase:
  - Reuse Schwab API connection logic
  - Adapt market data retrieval patterns
  - Extend options intelligence with ML predictions
  - Add Market Intelligence and Correlation Tool signals
  - Implement trinomial tree valuation
  - Add explainability layer
  - Real-time execution via Schwab API

Repository: https://github.com/oldboldpilot/SchwabFirstAPI
```

**Alternative Brokers (Future Tier 2/3):**

| Broker | Use Case | When to Add |
|--------|----------|-------------|
| **Interactive Brokers** | Institutional-grade, international | Tier 2 (>$100K capital) |
| **Tradier** | Backup/redundancy, options focus | Tier 2 (multi-broker) |
| **TD Ameritrade** | Options analytics (thinkorswim platform) | Tier 2 (analysis) |

**Tier 1 POC Broker: Charles Schwab (FINAL)**
- Existing $30k margin account
- Proven API access via SchwabFirstAPI
- Free real-time market data
- Full options support
- Zero additional setup cost
- Can begin real trading immediately after validation

### 5.3 Trading Strategies (Priority Order)

**Implementation Priority:**
1. **Phase 1 (Initial Focus):** Options Day Trading
2. **Phase 2 (Future):** Stock Day Trading
3. **Phase 3 (Future):** Short-Term Trading (stocks and options)
4. **Phase 4 (Future):** Long-Term Strategic Investing (stocks)

#### 5.3.1 Strategy 1: Algorithmic Options Day Trading (INITIAL PRIORITY)

**Objective:** Exploit intra-day options price movements and volatility changes through fully automated ultra-low latency trading

**⚠️ CRITICAL PHILOSOPHY:** Day trading success requires understanding that markets are driven by **EMOTION and MOMENTUM** in the short term, not just rational analysis. The system must detect and exploit "animal spirits" - the psychological forces that move markets minute-to-minute.

**Characteristics:**
- Position holding period: Minutes to hours (intra-day only)
- Trade frequency: 20-100+ options trades per day
- Position size: Small to medium (1-3% of portfolio per trade)
- Leverage: Inherent in options (controlling 100 shares per contract)
- Automation: 100% automated, no human intervention
- Execution speed: < 1ms signal-to-order latency
- **Decision basis: 60% sentiment/momentum, 40% fundamentals/correlation**

**Options Strategies to Employ (Sentiment-Aware):**
- **Directional plays:** Calls/Puts based on **predicted movement AND sentiment momentum**
- **Volatility plays:** Straddles/Strangles on **high-impact news AND panic/euphoria spikes**
- **Delta-neutral strategies:** Profit from **volatility changes driven by fear/greed**
- **Calendar spreads:** Exploit time decay differences **while fading short-term emotion**
- **Vertical spreads:** Defined risk/reward **capturing sentiment overreaction**
- **Iron condors:** Profit from range-bound underlying **during low emotion periods**
- **Greeks arbitrage:** Exploit mispriced gamma, vega, theta **caused by irrational demand**
- **NEW: Sentiment fade trades:** Sell premium when VIX > 30 (extreme fear)
- **NEW: Momentum continuation:** Buy calls/puts when sentiment velocity high
- **NEW: Contrarian plays:** Opposite positioning when sentiment at extremes

**Signal Sources (Rational + Behavioral):**

**Rational Signals:**
- Breaking news impact predictions (implied volatility spikes)
- Earnings announcements and guidance (volatility expansion)
- Intra-day correlation signals (underlying and options)
- Greeks arbitrage opportunities
- Implied volatility surface anomalies

**Behavioral Signals (CRITICAL FOR DAY TRADING):**
- **Sentiment momentum:** Rapid change in social media/news tone
- **"Meme stock" alerts:** Viral stocks with retail trader attention
- **Unusual options activity:** Large institutional bets (smart money following)
- **Volume surges:** Breakouts driven by herd behavior
- **VIX spikes:** Fear creating mispriced options (sell premium)
- **Put/Call ratio extremes:** Contrarian indicators (>1.2 or <0.7)
- **Reddit/WSB trending:** Retail momentum (ride it or fade it)
- **Panic selling patterns:** Capitulation bottoms (buy calls)
- **Euphoria patterns:** Blow-off tops (buy puts)
- **FOMO indicators:** Rapid price acceleration with high volume

**Entry Criteria (Combining Logic + Sentiment):**
- **Rational Criteria:**
  - High-confidence impact prediction (> 80%) for directional plays
  - Implied volatility mispricing detected (> 2 standard deviations)
  - Sufficient options liquidity (bid-ask spread < 5% of mid price)
  - Favorable risk-reward ratio (> 3:1 for options)
  - Technical confirmation on underlying
  - Greeks within acceptable ranges for strategy

- **Behavioral/Sentiment Criteria (MANDATORY FOR DAY TRADING):**
  - **Sentiment alignment:** Momentum direction matches sentiment trend
  - **Sentiment extremes:** VIX > 25 (sell premium) or unusual options activity
  - **Social media momentum:** Increasing mentions + positive sentiment (ride momentum)
  - **Contrarian setup:** Extreme fear (VIX > 35) or euphoria (meme stock blow-off)
  - **Volume confirmation:** High volume = conviction = momentum continuation
  - **Institutional vs retail:** Smart money positioning (large options orders)
  - **No conflicting signals:** Sentiment and fundamentals aligned (or consciously fading emotion)

**Exit Criteria (Logic + Sentiment):**
- **Profit Targets:**
  - Target profit reached (10-50% on options premium)
  - Exceptional momentum: let winners run if sentiment accelerating

- **Stop Losses:**
  - Stop-loss triggered (30-50% loss on premium)
  - Sentiment reversal: cut losses if momentum shifts against position

- **Time-Based:**
  - End of trading day (no overnight options positions initially)
  - Time decay approaching inflection point

- **Sentiment-Based (CRITICAL):**
  - **Sentiment reversal:** Social media/news tone flips (exit momentum trades)
  - **Volatility normalization:** VIX returns to normal after spike (exit volatility plays)
  - **Volume exhaustion:** Momentum dying (decreasing volume on rallies/selloffs)
  - **Extreme profit taking:** If up 100%+ and sentiment reaching euphoria (take profit)
  - **Dead cat bounce:** False reversal pattern (don't get trapped)
  - **Capitulation:** Final panic selling (buy opportunity, but wait for confirmation)

**Risk Management:**
- Maximum position size: 3% of portfolio per options trade
- Maximum daily loss limit: 5% of portfolio
- Position-level stop-losses (hard stops at 50% loss)
- Portfolio-level circuit breakers
- Greeks limits (max delta exposure, max gamma exposure)
- Avoid holding through earnings (unless strategy)
- Bid-ask spread limits
- Slippage and commission accounting (critical for options)

#### 5.3.2 Strategy 2: Algorithmic Stock Day Trading (FUTURE - Phase 2)

**Objective:** Exploit intra-day stock price movements through fully automated high-frequency trading

**Note:** This strategy will be developed after options day trading is validated and profitable.

#### 5.3.3 Strategy 3: Short-Term Trading - Stocks and Options (FUTURE - Phase 3)

**Objective:** Capture medium-term trends driven by events, correlations, and market cycles

**Characteristics:**
- Position holding period: 1-120 days
- Trade frequency: 1-10 trades per week
- Position size: Medium (2-5% of portfolio per position)
- Leverage: 1x to 1.5x
- Automation: High automation with human oversight

**Signal Sources:**
- Multi-day impact predictions
- Inter-day and intra-month correlations
- Earnings expectations and surprises
- Geopolitical event impacts
- Sector rotation signals

**Entry Criteria:**
- Medium-to-high confidence prediction (> 60%)
- Supporting correlation evidence
- Fundamental catalyst identified
- Technical trend confirmation
- Risk-adjusted expected return > 15%

**Exit Criteria:**
- Target profit reached (5-25%)
- Stop-loss triggered (3-10%)
- 120-day maximum holding period
- Signal deterioration
- Better opportunity identified

**Risk Management:**
- Maximum position size: 5% of portfolio
- Maximum positions: 20 concurrent
- Portfolio diversification requirements
- Sector exposure limits
- Correlation-based hedging

#### 5.3.4 Strategy 4: Long-Term Strategic Investing - Stocks Only (FUTURE - Phase 4)

**Objective:** Build wealth through multi-year positions in fundamentally strong companies

**Note:** This strategy focuses exclusively on stocks (not options) for long-term wealth building.

**Characteristics:**
- Position holding period: 1-10+ years
- Trade frequency: 1-5 trades per month
- Position size: Large (5-15% of portfolio per position)
- Leverage: None (1x)
- Automation: Low automation, high human judgment

**Signal Sources:**
- Long-term impact graphs and relationships
- Intra-quarter and multi-year correlations
- Fundamental analysis integration
- Macroeconomic trends
- Industry disruption predictions

**Entry Criteria:**
- Strong long-term thesis
- Attractive valuation metrics
- Positive long-term impact graph positioning
- Favorable industry trends
- Management quality indicators

**Exit Criteria:**
- Thesis invalidation
- Valuation extremes reached
- Better opportunity identified
- Fundamental deterioration
- Portfolio rebalancing needs

**Risk Management:**
- Maximum position size: 15% of portfolio
- Maximum positions: 10-15 concurrent
- Diversification across sectors and themes
- Regular fundamental review
- Tax-aware trading decisions

### 5.4 Core Features

#### 5.4.1 Options Selection Engine (Initial Priority)
- Options opportunity scoring algorithm
- Implied volatility analysis and mispricing detection
- Greeks analysis and optimization
- Multi-strategy options ranking system
- Underlying stock selection for options plays
- Liquidity filtering (bid-ask spreads, volume)
- Watchlist generation for high-probability setups

#### 5.4.2 Stock Selection Engine (Future)
- Stock opportunity scoring algorithm
- Multi-factor ranking system
- Portfolio construction optimizer
- Diversification engine
- Watchlist generation

#### 5.4.3 Movement Prediction (Underlying & Options)
- **Options-specific (Priority):**
  - Implied volatility prediction with confidence intervals
  - Expected move calculation
  - Probability of profit (POP) estimation
  - Greeks forecasting (future delta, gamma, theta, vega)
  - Options price target calculation
- **Underlying stock prediction:**
  - Price target calculation with confidence intervals
  - Time-to-target estimation
- Probability distribution modeling
- Expected value calculation
- Risk assessment (VaR, CVaR)

#### 5.4.3 Position Sizing
- Kelly Criterion implementation
- Risk-adjusted position sizing
- Volatility-based sizing
- Correlation-adjusted sizing
- Capital allocation optimization

#### 5.4.4 Order Execution
- Smart order routing
- VWAP/TWAP execution algorithms
- Slippage minimization
- Order splitting and timing
- Market impact modeling

#### 5.4.5 Portfolio Management
- Real-time position tracking
- **P&L Calculation and Attribution (CRITICAL - See detailed requirements below)**
- Risk metrics monitoring
- Rebalancing automation
- Tax loss harvesting

**MANDATORY P&L CALCULATION REQUIREMENTS:**

**1. Margin Interest Accounting (CRITICAL FOR MARGIN ACCOUNTS):**
- **Daily Margin Interest Accrual:**
  - MUST track daily margin balance (borrowed amount)
  - MUST calculate daily interest: `Daily Interest = (Margin Balance × Annual Rate) / 360`
  - Schwab margin rates (as of 2024):
    - $0-$24,999: 12.825% annual
    - $25,000-$49,999: 12.325% annual
    - $50,000-$99,999: 11.825% annual
    - (Rates variable - MUST fetch current rates via API or manual update)
  - Interest accrues daily, charged monthly
  - MUST subtract accumulated interest from realized P&L
  - Example: $10k margin × 12.825% / 360 = $3.56/day interest cost

- **Impact on Strategy Profitability:**
  - Day trading options: margin interest minimal (intraday positions)
  - Swing trading (2-7 days): margin interest significant cost
  - Position sizing MUST account for daily interest drag
  - Break-even calculation: `Profit Target > (Entry Cost + Commission + Margin Interest)`
  - Monthly reporting: total margin interest expense as separate line item

- **Margin Balance Tracking:**
  - Track real-time margin utilization: `(Borrowed Amount / Margin Limit) × 100%`
  - Alert if margin utilization > 80% (risk of margin call)
  - MUST track margin balance changes:
    - Increases: new positions opened, stock price drops (long), price rises (short)
    - Decreases: positions closed, deposits, stock price rises (long), price drops (short)

**2. Corporate Actions Impact on P&L (MANDATORY):**
- **Dividend Adjustments:**
  - Ex-dividend date tracking for all positions
  - LONG positions: ADD dividend payments to realized P&L
  - SHORT positions: SUBTRACT dividend payments from realized P&L (cost)
  - Options: dividend expectations affect option pricing (dividends favor puts, hurt calls)
  - Quarterly dividend tracking: MUST forecast dividend impact on positions
  - Example: Short 100 shares XYZ paying $0.50 dividend = -$50 realized P&L

- **Stock Split Adjustments:**
  - Forward splits (e.g., 2:1, 3:1): multiply shares, divide cost basis
  - Reverse splits (e.g., 1:5, 1:10): divide shares, multiply cost basis
  - MUST adjust position quantities and cost basis on split effective date
  - Options: contracts adjust automatically (100 shares → varies by split ratio)
  - P&L calculation MUST use split-adjusted prices for historical comparisons

- **Merger/Acquisition Adjustments:**
  - Cash mergers: position liquidated at offer price (realized P&L)
  - Stock-for-stock: position converted at exchange ratio (unrealized → new position)
  - Mixed deals: partial cash (realized), partial stock (unrealized)
  - Tender offers: premium to market price (capture spread)
  - Tracking required: announcement date, shareholder vote, regulatory approval, close date

- **Spin-off Adjustments:**
  - Cost basis allocation: distribute original cost between parent and spun-off entity
  - Allocation method: typically based on first-day trading prices
  - Example: $100 position → $70 parent + $30 spin-off (based on market values)
  - New position created in spin-off company (track separately)

**3. Comprehensive P&L Formula:**
```
Realized P&L =
  (Exit Price - Entry Price) × Quantity
  - Entry Commission
  - Exit Commission
  - Margin Interest (daily accrual × days held)
  + Dividends Received (long positions)
  - Dividends Paid (short positions)
  ± Corporate Action Adjustments (mergers, spin-offs)

Unrealized P&L =
  (Current Price - Entry Price) × Quantity
  - Accrued Margin Interest (to date)
  + Expected Dividends (ex-div dates before planned exit)
  ± Pending Corporate Action Impact
```

**4. Reporting Requirements:**
- Daily P&L reports MUST include:
  - Gross P&L (before costs)
  - Commission costs
  - Margin interest costs (accrued to date)
  - Dividend income/expense
  - Corporate action impacts
  - Net P&L (after all costs)
  - **Estimated tax liability (real-time)**
  - **After-tax P&L**
- Monthly P&L reports MUST include:
  - Total margin interest paid (separate line item)
  - Total dividends received/paid
  - Corporate action summary (splits, mergers, spin-offs)
  - **Complete tax reporting data (details below)**

**5. Tax Calculation Requirements (MANDATORY REAL-TIME TRACKING):**

**CRITICAL:** Traders must know their true after-tax profitability at all times. Tax liability can consume 25-50%+ of gross profits.

**A. Capital Gains Tax Calculation:**

- **Holding Period Classification:**
  - Short-term capital gains: positions held ≤ 365 days
  - Long-term capital gains: positions held > 365 days
  - Track exact holding period per position (entry date → exit date)
  - Day trading (intraday): always short-term

- **Short-Term Capital Gains (Taxed as Ordinary Income):**
  - Federal tax brackets (2024):
    - 10%: $0 - $11,600 (single) / $0 - $23,200 (married)
    - 12%: $11,601 - $47,150 / $23,201 - $94,300
    - 22%: $47,151 - $100,525 / $94,301 - $201,050
    - 24%: $100,526 - $191,950 / $201,051 - $383,900
    - 32%: $191,951 - $243,725 / $383,901 - $487,450
    - 35%: $243,726 - $609,350 / $487,451 - $731,200
    - 37%: $609,351+ / $731,201+
  - MUST calculate marginal tax rate based on total income
  - MUST track year-to-date trading income for bracket calculation

- **Long-Term Capital Gains (Preferential Rates):**
  - 0%: Taxable income up to $47,025 (single) / $94,050 (married)
  - 15%: $47,026 - $518,900 / $94,051 - $583,750
  - 20%: $518,901+ / $583,751+

**B. Additional Federal Taxes:**

- **Net Investment Income Tax (NIIT):**
  - 3.8% surtax on investment income
  - Applies if Modified AGI exceeds:
    - $200,000 (single)
    - $250,000 (married filing jointly)
  - Applied to lesser of: (net investment income) OR (AGI above threshold)
  - MUST track cumulative AGI to determine NIIT applicability

- **Medicare Tax (for Professional Traders):**
  - If classified as "trader in business" (mark-to-market election):
    - 2.9% Medicare tax on net trading income
    - Additional 0.9% Medicare surtax if income > $200,000 (single) / $250,000 (married)
  - MUST support both investor and trader classification
  - Default: investor (capital gains treatment)
  - Optional: trader in business (ordinary income, allows more deductions)

**C. State Income Tax:**

- **State-Specific Rates (Must Support):**
  - California: 1% - 13.3% (highest in nation)
  - New York: 4% - 10.9%
  - Texas: 0% (no state income tax)
  - Florida: 0% (no state income tax)
  - MUST allow user to configure state of residence
  - MUST update rates when state changes tax law

- **Multi-State Considerations:**
  - Track state of residence during trade period
  - Apply correct state rate based on residence date

**D. Wash Sale Rules (CRITICAL - IRS Required):**

- **Wash Sale Definition:**
  - Sell security at a loss
  - Purchase "substantially identical" security within 30 days before or after sale
  - Loss is DISALLOWED, added to cost basis of new position

- **Tracking Requirements:**
  - MUST track all sales at a loss
  - MUST monitor purchases 30 days before and after loss sale
  - MUST identify substantially identical securities:
    - Same stock (e.g., AAPL shares)
    - Options on same stock (same strike, similar expiration)
  - MUST disallow losses and adjust cost basis automatically
  - MUST flag wash sales on trade confirmations

- **Example:**
  - Buy 100 AAPL @ $180 (Jan 5)
  - Sell 100 AAPL @ $170 (Jan 15) → $1,000 loss
  - Buy 100 AAPL @ $175 (Jan 20) → WASH SALE
  - Original $1,000 loss DISALLOWED
  - New cost basis: $175 + $10 (disallowed loss) = $185

**E. Comprehensive Tax Formula:**

```
Gross Trading Income =
  Short-Term Gains - Short-Term Losses (after wash sales)
  + Long-Term Gains - Long-Term Losses (after wash sales)
  + Net Dividend Income (qualified vs ordinary)

Federal Capital Gains Tax =
  (Short-Term Gains × Marginal Tax Rate)
  + (Long-Term Gains × Long-Term Rate)
  + (Net Investment Income × NIIT Rate if applicable)

State Income Tax =
  (Total Trading Income × State Rate)

Total Tax Liability =
  Federal Capital Gains Tax
  + State Income Tax
  + Medicare Tax (if trader in business)

After-Tax Profit =
  Gross Trading Income
  - Total Tax Liability
  - Commissions
  - Margin Interest
```

**F. Real-Time Tax Tracking:**

- **Daily Tax Liability Estimation:**
  - Calculate estimated tax on all realized gains/losses to date
  - Update after every trade close
  - Show running total: "YTD Tax Liability: $X,XXX"
  - Alert if tax liability > 30% of cash balance (estimated payment warning)

- **Position-Level Tax Impact:**
  - Before closing position, show:
    - Gross P&L
    - Estimated tax if closed now
    - After-tax P&L
    - Holding period (days until long-term treatment)
  - Example: "Close now: +$1,000 gross, -$370 tax (37%), +$630 after-tax"
  - Example: "Wait 15 more days for long-term: save $170 in taxes"

- **Quarterly Estimated Tax Calculations:**
  - Calculate quarterly estimated tax payments (due Apr 15, Jun 15, Sep 15, Jan 15)
  - Alert 7 days before due date: "Estimated tax payment: $X,XXX due [date]"
  - Track payments made vs liability to date
  - Warn if underpayment penalty likely (< 90% of current year or 100%/110% of prior year)

**G. Tax Reporting Data Export:**

- **Form 8949 Support (Capital Gains/Losses):**
  - Export all trades with:
    - Description (100 shares AAPL)
    - Date acquired, date sold
    - Proceeds, cost basis, wash sale adjustments
    - Short-term vs long-term classification
  - Generate preliminary Form 8949 for CPA review

- **Schedule D Support (Summary):**
  - Total short-term gains/losses
  - Total long-term gains/losses
  - Net capital gain/loss

- **Form 1099-B Reconciliation:**
  - Import broker 1099-B (Schwab format)
  - Compare broker cost basis vs system cost basis
  - Flag discrepancies for review
  - Adjust for wash sales not reported by broker

**H. Tax Optimization Features:**

- **Tax Loss Harvesting Recommendations:**
  - Identify positions with unrealized losses
  - Calculate potential tax benefit of harvesting
  - Warn about wash sale risk if similar positions exist
  - Suggest: "Harvest $5,000 loss → save $1,850 in taxes (37% rate)"

- **Long-Term vs Short-Term Decision Support:**
  - For positions near 365-day mark, show:
    - Days until long-term treatment
    - Tax savings if held to long-term
    - Risk of holding (volatility, opportunity cost)

- **Year-End Tax Planning:**
  - November/December: show unrealized gains/losses
  - Calculate tax liability if all positions closed today
  - Suggest offsetting gains with losses
  - Warn if large tax bill likely

**I. User Configuration:**

- **Required Settings:**
  - Filing status (single, married filing jointly, etc.)
  - State of residence
  - Estimated non-trading income (for marginal rate calculation)
  - Trader vs investor classification
  - Prior year AGI (for estimated payment calculations)

- **Optional Settings:**
  - Specific deductions/credits (reduces effective rate)
  - Expected itemized deductions
  - Other investment income (for NIIT calculation)

**J. Tax Liability Alerts:**

- **Real-Time Alerts:**
  - Daily: "YTD tax liability: $X,XXX (XX% of gross profits)"
  - After large winning trade: "Trade added $X,XXX to tax liability"
  - Quarterly: "Estimated tax payment of $X,XXX due [date]"
  - Year-end: "Projected total tax liability: $X,XXX for [year]"

- **Warning Thresholds:**
  - Tax liability > 40% of account balance (liquidity warning)
  - Underpayment penalty likely (< 90% of current year tax)
  - Approaching NIIT threshold ($200k/$250k)
  - Large wash sale disallowance (> $1,000)

**K. Tax Code References (MANDATORY COMPLIANCE):**

**Federal Tax Code (IRS):**
- **IRC § 1(h):** Capital gains tax rates (0%, 15%, 20%)
- **IRC § 1(j):** Tax brackets for ordinary income
- **IRC § 1411:** Net Investment Income Tax (NIIT) - 3.8% surtax
- **IRC § 1091:** Wash sale rules (61-day window)
- **IRC § 1222:** Holding period definitions (short-term vs long-term)
- **IRC § 475(f):** Mark-to-market election for traders
- **IRC § 6654:** Underpayment penalty calculations

**IRS Publications:**
- **IRS Publication 550:** Investment Income and Expenses (capital gains treatment)
- **IRS Publication 551:** Basis of Assets (wash sales, corporate actions)
- **IRS Publication 564:** Mutual Fund Distributions (dividend treatment)
- **IRS Publication 505:** Tax Withholding and Estimated Tax (quarterly payments)
- **IRS Publication 17:** Federal Income Tax Guide (general tax calculations)

**IRS Forms:**
- **Form 8949:** Sales and Other Dispositions of Capital Assets
- **Schedule D (Form 1040):** Capital Gains and Losses
- **Form 1099-B:** Proceeds from Broker Transactions (broker reports)
- **Form 4797:** Sales of Business Property (if mark-to-market election)

**California Tax Code (Example State):**
- **California Revenue and Taxation Code § 17024:** Capital gains treatment (no preferential rate)
- **California Revenue and Taxation Code § 17041-17045.7:** Tax brackets (1% - 13.3%)
- **California Revenue and Taxation Code § 17551:** Wash sale conformity with federal
- **California FTB Publication 1001:** Supplemental Guidelines to California Adjustments
- **California FTB Publication 1005:** Pension and Annuity Guidelines (dividend treatment)

**L. Annual Tax Rate Updates (CRITICAL - MANDATORY YEARLY MAINTENANCE):**

**⚠️ WARNING:** Tax rates, brackets, and rules change EVERY YEAR. Failure to update will result in incorrect tax calculations and potential IRS penalties for users.

**Annual Update Schedule:**
- **November 1 - November 30:** IRS publishes next year's tax brackets and rates
  - Monitor IRS Revenue Procedure (typically Rev. Proc. YYYY-XX released ~November)
  - Example: Rev. Proc. 2023-34 contained 2024 tax year adjustments
  - IRS website: https://www.irs.gov/newsroom/tax-inflation-adjustments

- **December 1 - December 15:** Update system with new tax year rates
  - Update federal_tax_brackets table with new year data
  - Update longterm_capital_gains_brackets if changed
  - Update standard deduction amounts
  - Update NIIT thresholds (rare, but check)
  - Update estimated payment safe harbor percentages

- **December 16 - December 31:** Testing and validation
  - Test tax calculations with new rates
  - Verify wash sale rules (rarely change, but review IRS notices)
  - Update documentation with new tax year

- **January 1:** New tax year rates go live
  - System automatically uses new rates for new tax year
  - Historical rates preserved for prior year reporting

**State Tax Rate Updates:**
- **Quarterly Review (January, April, July, October):**
  - California FTB: https://www.ftb.ca.gov/
  - New York DTF: https://www.tax.ny.gov/
  - Texas: Verify still 0% (rare changes)
  - Florida: Verify still 0% (rare changes)
  - Other states as applicable to user base

**Sources to Monitor:**

1. **Federal (IRS):**
   - IRS Revenue Procedures (annual inflation adjustments)
   - IRS Notices (mid-year changes, rare)
   - IRS Tax Topics: https://www.irs.gov/taxtopics
   - IRS Forms and Publications (updated annually)
   - IRS Newsroom: https://www.irs.gov/newsroom

2. **California:**
   - California Franchise Tax Board (FTB)
   - CA Revenue and Taxation Code updates
   - FTB Tax News: https://www.ftb.ca.gov/about-ftb/newsroom/tax-news/index.html
   - CA Legislative updates affecting tax rates

3. **Other States:**
   - State Department of Revenue websites
   - State legislative tracking services
   - Tax Foundation updates: https://taxfoundation.org/

**Automated Update Process:**

1. **Monitoring System:**
   - Subscribe to IRS e-news: https://www.irs.gov/newsroom/e-news-subscriptions
   - Subscribe to state tax agency newsletters
   - Set up Google Alerts for "IRS tax brackets [year]"
   - Set up Google Alerts for "California tax brackets [year]"

2. **Calendar Reminders:**
   - November 1: "Check for new IRS Revenue Procedure"
   - November 15: "Verify IRS published new tax year rates"
   - December 1: "Update system with new tax rates"
   - December 20: "Test new tax year calculations"
   - Quarterly: "Review state tax rate changes"

3. **Version Control:**
   - All tax rate changes MUST be git committed
   - Commit message format: "tax: Update [year] federal/state tax rates per IRS Rev. Proc. YYYY-XX"
   - Tag releases: "v2024-tax-rates", "v2025-tax-rates"

4. **User Notification:**
   - December 15: Email users: "Tax rates updated for [year] tax year"
   - Alert users if their tax liability estimates changed significantly
   - Provide link to IRS Revenue Procedure for transparency

**Validation Requirements:**

1. **Cross-Check Sources:**
   - IRS official publications (primary source)
   - Tax software vendors (TurboTax, H&R Block - secondary validation)
   - CPA review (annual verification by tax professional)

2. **Test Cases:**
   - Calculate sample tax liability with known income amounts
   - Compare against IRS tax tables
   - Verify bracket transitions (edge cases)
   - Test all filing statuses

3. **Documentation:**
   - Update CHANGELOG.md with tax rate changes
   - Update user-facing documentation
   - Update API documentation if tax endpoints changed

**Emergency Updates:**

- **Mid-Year Tax Law Changes (Rare):**
  - Monitor for emergency IRS notices
  - Implement within 30 days of passage
  - Example: TCJA 2017 (major overhaul)
  - Example: COVID-19 tax relief provisions

**Responsibility Assignment:**
- **Primary:** Lead Developer (system updates)
- **Secondary:** CPA/Tax Professional (validation)
- **Tertiary:** DevOps (deployment, testing)

**Consequences of Not Updating:**
- ❌ Incorrect tax liability calculations
- ❌ Users underpay/overpay estimated taxes
- ❌ IRS underpayment penalties for users
- ❌ Loss of user trust
- ❌ Potential legal liability

**Annual Update Checklist:**
```
[ ] November 1: Check IRS website for Revenue Procedure
[ ] November 15: Download and review IRS Rev. Proc. YYYY-XX
[ ] November 20: Extract new tax bracket data
[ ] December 1: Update federal_tax_brackets table
[ ] December 2: Update longterm_capital_gains_brackets table
[ ] December 3: Update standard_deduction amounts
[ ] December 5: Review state tax changes (CA, NY, etc.)
[ ] December 7: Update state_tax_rates table
[ ] December 10: Update documentation with new rates
[ ] December 12: Run test suite with new rates
[ ] December 15: Deploy to staging environment
[ ] December 18: CPA validation of calculations
[ ] December 20: Deploy to production
[ ] December 22: Send user notification
[ ] December 31: Final verification before new tax year
```

**M. Automated Tax Rate Discovery & Historical Reproduction (MANDATORY):**

**⚠️ CRITICAL REQUIREMENT:** System MUST automatically determine and store tax rates for all calendar years to ensure accurate historical calculations and backtesting.

**1. Automatic Tax Rate Determination:**

**API Integration for Automated Rate Discovery:**
- **TaxJar API:** State sales tax rates (includes income tax data)
  - https://www.taxjar.com/api/
  - Pricing: $99/month for Pro plan
  - Coverage: All 50 states + DC
  - Historical data: 2013-present

- **Tax Foundation API:** Comprehensive federal and state tax data
  - https://taxfoundation.org/
  - Free tier available (limited calls)
  - Historical tax brackets and rates

- **IRS Data API:** Federal tax information
  - https://www.irs.gov/statistics
  - Free, official source
  - Historical tax tables available

- **State-Specific APIs:**
  - California FTB Tax Calculator API (if available)
  - New York DTF e-Services
  - Auto-discovery from state websites

**Automated Scraping/Parsing (Fallback):**
- Automated download of IRS Publication 17 (annual)
- Parse IRS tax tables from PDF/HTML
- Extract state tax brackets from official websites
- Validate extracted data against multiple sources

**2. Historical Tax Rate Storage Requirements:**

**Complete Historical Record (2015-2035 minimum):**
- **Every calendar year MUST have complete tax rates stored:**
  - Federal ordinary income brackets (all filing statuses)
  - Long-term capital gains brackets (all filing statuses)
  - State income tax brackets (all supported states)
  - Standard deductions (federal and state)
  - NIIT thresholds (Net Investment Income Tax)
  - Medicare tax thresholds
  - Estimated payment safe harbor percentages

**Version Control for Tax Rules:**
- **Each tax year stored with:**
  - Effective date (January 1, YYYY)
  - End date (December 31, YYYY or NULL if current)
  - Source document (IRS Rev. Proc. YYYY-XX)
  - Source URL (link to official publication)
  - Hash/checksum of rate data (detect changes)
  - Created timestamp (when added to system)
  - Created by (user or automated process)

**Rule Versioning:**
- Tax rules can change mid-year (rare, but possible)
- MUST support multiple versions per tax year
- Example: CARES Act 2020 changed tax rules mid-year
- Each version tagged with effective date range

**3. Calculation Reproduction Requirements:**

**CRITICAL:** Any tax calculation must be reproducible years later using historical tax rates.

**Calculation Metadata Storage:**
- **For every realized gain/loss, store:**
  - Trade date and settlement date
  - Tax year applied
  - Federal tax rate used (at time of calculation)
  - State tax rate used (at time of calculation)
  - NIIT rate used (if applicable)
  - Tax bracket applied
  - Source document reference
  - Calculation timestamp

**Reproduction Process:**
- Given: trade_id, exit_date
- Lookup: tax rates effective on exit_date
- Recalculate: using exact rates from that date
- Verify: matches original calculation
- Report: any discrepancies (indicates rate update needed)

**Audit Trail:**
```sql
-- Example: Reproduce tax calculation from 3 years ago
SELECT
    rgl.*,
    ftb.tax_rate as federal_rate_used,
    str.tax_rate as state_rate_used,
    'Reproduced from historical rates' as note
FROM realized_gains_losses rgl
JOIN federal_tax_brackets ftb
    ON ftb.tax_year = EXTRACT(YEAR FROM rgl.exit_date)
    AND ftb.filing_status = (SELECT filing_status FROM tax_config WHERE account_id = rgl.account_id)
    AND rgl.realized_gain_loss BETWEEN ftb.income_min AND COALESCE(ftb.income_max, 999999999)
JOIN state_tax_rates str
    ON str.tax_year = EXTRACT(YEAR FROM rgl.exit_date)
    AND str.state_code = (SELECT state_of_residence FROM tax_config WHERE account_id = rgl.account_id)
WHERE rgl.trade_id = 'TRADE-2021-12345';
```

**4. Validation and Verification:**

**Multi-Source Validation:**
- Compare fetched rates against 3+ independent sources
- Flag discrepancies for human review
- Example sources:
  - IRS official publications (primary)
  - TurboTax rate tables (secondary)
  - H&R Block rate tables (tertiary)
  - TaxJar API (tertiary)

**Automated Testing:**
```python
# Example validation test
def test_2024_federal_tax_brackets():
    """Verify 2024 federal tax brackets match IRS Rev. Proc. 2023-34"""
    rates = get_federal_tax_brackets(tax_year=2024, filing_status='single')

    # Test against known IRS values
    assert rates[0].income_min == 0
    assert rates[0].income_max == 11600
    assert rates[0].tax_rate == 0.10

    assert rates[6].income_min == 609351
    assert rates[6].income_max is None  # Highest bracket
    assert rates[6].tax_rate == 0.37

    # Verify source document
    assert rates[0].source_document == 'IRS Rev. Proc. 2023-34'
```

**5. Backwards Compatibility:**

**Historical Backfill (2015-2024):**
- On first deployment, automatically fetch and store historical rates
- Backfill process:
  ```
  FOR year IN 2015..2024:
      - Fetch IRS Rev. Proc. for year
      - Extract federal tax brackets
      - Extract LTCG brackets
      - Store with effective_date = January 1, year
      - Fetch state rates from archives
      - Validate against known sources
  ```

**Backtesting Accuracy:**
- Backtests run on historical data MUST use tax rates from that period
- Example: Backtest 2020 trades → use 2020 tax rates
- Never use current tax rates for historical calculations

**6. Data Export for Compliance:**

**Tax Rate Snapshots:**
- Export complete tax rate snapshot for each year
- Format: JSON, CSV, SQL dump
- Include all metadata (source documents, URLs)
- Store in version control (git)
- Example: `tax_rates_2024.json`, `tax_rates_2024.sql`

**Reproducibility Package:**
```json
{
  "tax_year": 2024,
  "effective_date": "2024-01-01",
  "end_date": "2024-12-31",
  "source_document": "IRS Rev. Proc. 2023-34",
  "source_url": "https://www.irs.gov/pub/irs-drop/rp-23-34.pdf",
  "federal_brackets": {
    "single": [
      {"min": 0, "max": 11600, "rate": 0.10},
      {"min": 11601, "max": 47150, "rate": 0.12},
      ...
    ]
  },
  "ltcg_brackets": { ... },
  "state_rates": { ... },
  "validation": {
    "validated_against": ["IRS Publication 17", "TurboTax 2024"],
    "validated_by": "cpa@example.com",
    "validated_date": "2023-12-20"
  }
}
```

**Tax Database (Enhanced):**
- Store current tax year rates (federal, state, local)
- **Store historical rates for ALL supported years (2015-present minimum)**
- **Store effective_date and end_date for all tax rates**
- **Flag current_year rates vs historical rates**
- **API integration with tax data providers (TaxJar, Tax Foundation)**
- **Automated rate discovery and validation**
- Manual override capability for custom tax situations
- **Audit trail: track who updated rates and when**
- **Version control: track all rate changes with timestamps**
- **Hash/checksum: detect unauthorized rate changes**
- **Export capability: JSON, CSV, SQL dumps for each year**
- **Reproduction guarantee: any calculation can be reproduced years later**

#### 5.4.6 Risk Management
- Position-level stop-losses
- Portfolio-level risk limits
- Correlation-based risk assessment
- Stress testing and scenario analysis
- Exposure monitoring (sector, geographic, factor)

#### 5.4.7 Performance Analytics
- Strategy-level performance tracking
- Attribution analysis (alpha sources)
- Benchmark comparison
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Trade analysis (win rate, profit factor)

### 5.5 Machine Learning Components

#### 5.5.1 Reinforcement Learning
- Q-learning for order execution
- Policy gradient methods for strategy selection
- Actor-critic for position sizing
- Environment: Historical market simulator
- Reward: Risk-adjusted returns

#### 5.5.2 Ensemble Prediction
- Combine signals from both sub-projects
- Model averaging with confidence weighting
- Bayesian model combination
- Prediction uncertainty quantification
- Model performance monitoring and re-weighting

#### 5.5.3 Feature Engineering
- Technical indicators (momentum, mean reversion, volatility)
- Fundamental features from impact analysis
- Correlation-based features
- Sentiment and news features
- Macroeconomic features
- **✅ DEPLOYED: 60-Feature Comprehensive Model (v3.0)**
  - **Identification (3):** symbol_encoded, is_option, days_to_expiry
  - **Time (8):** day_of_week, day_of_month, month, quarter, hour, minute, is_market_open, days_since_start
  - **Treasury Rates (7):** DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10
  - **Greeks (6):** delta, gamma, theta, vega, rho, implied_volatility
  - **Sentiment (2):** sentiment_score, sentiment_magnitude
  - **Price (5):** current_price, sma_20, sma_50, ema_12, ema_26
  - **Momentum (7):** return_1d, return_5d, return_20d, rsi_14, macd, macd_signal, bb_width
  - **Volatility (4):** volatility_20d, atr_14, bb_upper, bb_lower
  - **Interactions (10):** price_volume, momentum_volatility, rsi_bb, macd_atr, sma_cross, price_sma50, volatility_volume, rsi_macd, bb_position, atr_price
  - **Directionality (8):** win_rate_5d, win_rate_20d, avg_win_5d, avg_loss_5d, avg_win_20d, avg_loss_20d, win_loss_ratio_5d, win_loss_ratio_20d
  - **Normalization:** StandardScaler with AVX2 SIMD (8-way parallel, 8x speedup)
  - **Implementation:** `src/market_intelligence/feature_extractor.cppm` (620 lines)

#### 5.5.4 Model Training & Validation
- Walk-forward optimization
- Cross-validation on time series
- Out-of-sample testing
- Paper trading validation
- A/B testing for strategy variants

### 5.6 Functional Requirements

#### FR3.1: Signal Integration
- System SHALL integrate signals from both sub-projects
- System SHALL normalize and weight signals appropriately
- System SHALL handle conflicting signals with priority rules
- System SHALL track signal quality and performance

#### FR3.2: Stock Selection
- System SHALL generate ranked lists of opportunities
- System SHALL score opportunities based on expected returns and risk
- System SHALL apply filters (liquidity, volatility, sector limits)
- System SHALL update opportunity lists in real-time

#### FR3.3: Position Management
- System SHALL calculate optimal position sizes
- System SHALL place orders with appropriate execution algorithms
- System SHALL monitor positions in real-time
- System SHALL execute stop-losses and take-profit orders automatically
- **System SHALL calculate accurate P&L including ALL costs:**
  - Entry and exit commissions ($0 stocks, $0.65/contract options)
  - Margin interest (daily accrual based on margin balance and current rates)
  - Dividend income (long positions) and dividend expense (short positions)
  - Corporate action adjustments (splits, mergers, spin-offs)
- **System SHALL adjust positions automatically for corporate actions:**
  - Stock splits: update quantity and cost basis on effective date
  - Dividends: credit/debit account on payment date
  - Mergers: convert or liquidate positions per deal terms
  - Spin-offs: allocate cost basis and create new positions
- **System SHALL track margin utilization in real-time:**
  - Current margin balance (borrowed amount)
  - Daily margin interest accrual
  - Margin utilization percentage
  - Alert if utilization exceeds 80% (margin call risk)
- **System SHALL maintain complete position history:**
  - All corporate action adjustments with timestamps
  - Margin interest charges by position
  - Dividend payments received/paid
  - Split-adjusted cost basis history

#### FR3.4: Strategy Execution
- System SHALL execute day trading strategy fully automatically
- System SHALL execute short-term strategy with human approval
- System SHALL provide recommendations for long-term strategy
- System SHALL adapt to market conditions dynamically

#### FR3.5: Risk Control
- System SHALL enforce all position and portfolio limits
- System SHALL trigger circuit breakers when thresholds exceeded
- System SHALL calculate and monitor risk metrics continuously
- System SHALL generate risk alerts for human review

#### FR3.6: Reporting & Monitoring
- System SHALL provide real-time dashboard of positions and P&L
- System SHALL generate daily, weekly, monthly performance reports
- System SHALL track and report on strategy-specific metrics
- System SHALL provide trade-level audit trails
- **System SHALL provide detailed cost breakdowns in all P&L reports:**
  - Gross P&L (price changes only)
  - Commission costs (entry + exit)
  - Margin interest costs (daily accrual)
  - Dividend income/expense
  - Corporate action impacts
  - Net P&L (after all costs)
  - **Estimated tax liability (real-time)**
  - **After-tax P&L (net of all costs including taxes)**
- **System SHALL generate monthly cost analysis reports:**
  - Total margin interest paid
  - Average daily margin balance
  - Margin interest as % of gross P&L
  - Total dividends received/paid
  - Corporate action summary (all events affecting positions)
- **System SHALL alert on significant cost events:**
  - Ex-dividend dates for held positions (3 days advance notice)
  - Upcoming stock splits affecting positions (7 days advance notice)
  - Margin interest exceeding 10% of position P&L
  - Margin rate changes (notify within 24 hours)

#### FR3.7: Tax Calculation & Reporting (MANDATORY)
- **System SHALL calculate real-time tax liability on all closed positions:**
  - Classify each trade as short-term (≤365 days) or long-term (>365 days)
  - Apply correct federal tax rate based on holding period and user's tax bracket
  - Apply state income tax based on user's state of residence
  - Calculate Net Investment Income Tax (NIIT) if applicable
  - Track Medicare tax if user elected mark-to-market (trader in business)
- **System SHALL implement wash sale tracking per IRC § 1091:**
  - Monitor 61-day window (30 days before + sale date + 30 days after)
  - Identify substantially identical securities
  - Disallow losses on wash sales automatically
  - Adjust cost basis of replacement position
  - Flag all wash sales on trade confirmations and reports
- **System SHALL maintain year-to-date tax accounting:**
  - Total short-term gains and losses
  - Total long-term gains and losses
  - Net dividend income (qualified vs ordinary)
  - Running tax liability estimate
  - Quarterly estimated tax payment calculations
- **System SHALL generate tax reporting exports:**
  - Form 8949 data (all capital gains/losses)
  - Schedule D summary
  - Wash sale adjustments
  - Cost basis reconciliation vs Form 1099-B
- **System SHALL provide tax optimization features:**
  - Tax loss harvesting recommendations
  - Long-term vs short-term holding decision support
  - Year-end tax planning suggestions
  - Estimated quarterly payment alerts (7 days before due date)
- **System SHALL alert on tax events:**
  - Position approaching 365-day holding period (7 days notice)
  - Estimated tax liability exceeds 40% of account balance
  - Potential underpayment penalty warning
  - Approaching NIIT threshold ($200k single / $250k married)
- **System SHALL support user tax configuration:**
  - Filing status, state of residence
  - Non-trading income (for marginal rate calculation)
  - Trader vs investor classification
  - Custom tax rates for edge cases

### 5.7 Technical Requirements

#### TR3.1: Performance
- Signal processing latency < 100ms
- Order placement latency < 50ms
- Risk calculation latency < 200ms
- Dashboard update frequency: 1 second

#### TR3.2: Reliability
- System uptime > 99.9% during market hours
- Automated failover for critical components
- Data backup every 15 minutes
- Disaster recovery capability (RTO < 5 minutes)

#### TR3.3: Security
- Encrypted communication with brokers
- Multi-factor authentication for access
- Role-based access control
- Audit logging of all actions

---

## 6. Systems Integration & Usage Scenarios

**📋 Complete Systems Integration Architecture:** For detailed integration workflows, causal chain tracking, correlation chain prediction, sector-level analysis, budget-driven scenarios, self-simulation, profit/loss explanation, and mistake learning, see the **[Systems Integration Architecture Document](./architecture/systems-integration.md)**.

The systems integration document includes:
- **Complete hardware/software architecture** for Tier 1/2/3 deployments
- **Causal chain tracking**: Follow events from Fed decisions → Interest rates → Bank stocks
- **Correlation chain prediction**: Use time-lagged correlations for predictive trading (NVDA ↑ → AMD follows in 15min)
- **Sector-level impact analysis**: Track impacts across entire business sectors
- **Budget-driven strategy selection**: Automatic allocation for $1K to $1M+ budgets
- **Self-simulation framework**: Simulate all trades before execution (profitable from day one)
- **Profit explanation**: Complete analysis of why trades were profitable
- **Mistake analysis & learning**: Analyze losses, identify patterns, update models
- **Daily operation workflows**: Complete integration of all three tools
- **Weekly learning cycle**: Continuous improvement from successes and failures
- **🆕 Hypothetical scenario planning**: Propose "what-if" scenarios, generate trading strategies with complete justification and success probabilities, monitor in real-time, store for human retrieval

**📋 Additional Integration Documents:**
- **[Schwab API Integration Guide](./architecture/schwab-api-integration.md)**: Complete C++23 implementation design for Schwab Developer API with OAuth 2.0 token management, secure credential storage, rate limiting, and integration patterns from SchwabFirstAPI repository
- **[Database Strategy Analysis](./architecture/database-strategy-analysis.md)**: DuckDB-First vs PostgreSQL comparison with decision scenarios and migration strategy
- **[Trading Types and Strategies](./architecture/trading-types-and-strategies.md)**: Comprehensive reference for stock/options trading types, pricing models (Black-Scholes, binomial, trinomial), Greeks calculations, volatility strategies, P/L calculations, and tax implications
- **[Risk Metrics and Evaluation](./architecture/risk-metrics-and-evaluation.md)**: Complete risk management framework including VaR, stress testing, correlation analysis, position sizing (Kelly criterion), and real-time risk monitoring

### 6.1 Key Integration Features

**Causal Chain Following:**
```
Congress Passes Bill → Regulation Changes → Industry Impact → Company Stocks
Fed Rate Decision → Interest Rates → Sector Effects → Individual Stocks
Geopolitical Event → Commodity Prices → Supply Chain → Stock Prices
FDA Approval → Direct Impact → Competitor Effects → Sector Movement
```

**Correlation Chain Prediction:**
```
Leader Stock Moves → Correlated Stock Follows (with lag)
NVDA +5% (T0) → AMD +4.1% (T0+15min) → SMH +2.1% (T0+30min)
JPM +2% → BAC +1.8% (lag 10min) → Regional Banks (lag 30min)
```

**Sector-Level Analysis:**
```
Event → Direct Companies → Competitors → Suppliers → Entire Sector
Track correlations within sectors
Generate sector ETF trades vs individual stocks
```

**Budget-Aware Trading:**
- $1K: Focused options day trading (2-3 positions)
- $10K: Mixed strategies, 5-10 positions
- $100K: Diversified long-term, 15-25 positions
- $1M: Multi-strategy, 30-50 positions

**Self-Simulation (Validate Before Execute):**
- Monte Carlo simulation (1,000-10,000 scenarios)
- Only execute trades with positive expected value
- Probability of profit calculated
- Worst-case/best-case analysis
- Profitable from day one approach

**Full Explainability:**
- Every trade: Why we took it
- Every profit: Why we made money
- Every loss: What went wrong, how to improve
- Model updates based on mistakes
- Continuous learning loop

### 6.2 Usage Scenario Examples

**Scenario 1: Small Budget ($1,000)**
- Strategy: Options day trading
- Positions: 2-3 simultaneous
- Timeframe: Intra-day only
- Expected: 3-5% daily returns
- Explainability: Full trade-by-trade analysis

**Scenario 2: Medium Budget ($10,000)**
- Strategy: Mixed (short-term + options)
- Positions: 5-10 diversified
- Timeframe: 1-30 days
- Expected: 2-3% weekly returns
- Learning: Weekly model updates from mistakes

**Scenario 3: Large Budget ($100,000)**
- Strategy: Long-term + tactical
- Positions: 15-25 diversified
- Timeframe: 1-12 months
- Expected: 12-15% annual (beat S&P 500)
- Analysis: Quarterly attribution and learning

**Scenario 4: Institutional ($1,000,000)**
- Strategy: Multi-strategy portfolio
- Positions: 30-50 diversified
- All timeframes: Day/short/long
- Compliance: Full audit trail
- Learning: Continuous model refinement

For complete scenarios with explanations, causal chains, and learning examples, see the **[Systems Integration Architecture](./architecture/systems-integration.md)**.

### 6.3 Hypothetical Scenario Planning (What-If Analysis)

**🆕 NEW CAPABILITY:** Generate hypothetical scenarios that haven't occurred yet and propose trading strategies in real-time.

**Example Scenarios:**
- "What if Fed announces emergency rate cut of 0.5%?"
- "What if China blockades Taiwan?"
- "What if FDA approves Alzheimer's breakthrough drug?"
- "What if Congress passes major climate legislation?"

**System Response:**
1. Builds complete causal chain (multi-hop analysis)
2. Identifies affected securities via correlation
3. Finds historical precedents (similar events)
4. Calculates success probabilities
5. Proposes ranked trading strategies
6. Generates complete justification with risk assessment
7. Stores proposal in database for human retrieval
8. Monitors news feeds for scenario occurrence
9. Alerts when scenario detected
10. Can auto-execute approved strategies

**Output:**
- Complete proposal with 5-10 proposed trades
- Success probability for each trade (based on historical precedents)
- Expected P&L and risk/reward ratios
- Complete causal chain explanation
- Human-readable justification
- Retrievable via API: `/api/v1/scenarios/{proposal_id}`

**See:** Systems Integration Architecture Section 13.5 for complete implementation design

### 6.4 Database Strategy Decision: DuckDB-First

**DECISION: Start with DuckDB, migrate to PostgreSQL only if profitable**

**📋 Complete Analysis:** See **[Database Strategy Analysis](./architecture/database-strategy-analysis.md)** for comprehensive comparison with 10 decision scenarios.

**Rationale:**
- **Setup time:** DuckDB 30 seconds vs PostgreSQL 4-12 hours
- **POC velocity:** 10x faster iteration with DuckDB
- **Backtesting:** DuckDB 3-5x faster for analytics
- **Risk:** Zero wasted effort if POC fails
- **Migration:** 1-2 days when ready for PostgreSQL

**Tier 1 POC (Months 1-4): DuckDB Only**
- All data in DuckDB + Parquet files
- Perfect for rapid iteration and backtesting
- Zero database administration overhead
- Focus on algorithms, not database tuning

**Tier 2 Production (Month 5+, if profitable): Add PostgreSQL**
- PostgreSQL: Real-time operational data (trades, positions)
- DuckDB: Analytics and backtesting (historical data)
- Dual database: Best of both worlds
- Migration time: 1-2 days

**Cost Savings:**
- Don't invest time in PostgreSQL until profitability proven
- Can fail fast and pivot if needed
- PostgreSQL added only after validation

---

## 7. Cross-Project Integration Details

### 7.1 Data Flow Architecture

```
Market Intelligence Engine
         ↓
    [Impact Predictions, Graphs, Sentiment]
         ↓
         +------------------------+
         |                        |
         ↓                        ↓
Trading Decision Engine    Correlation Tool
         ↑                        |
         |                        |
         +--------[Correlations]--+
                  ↓
            [Trading Signals]
                  ↓
          [Order Execution]
```

### 6.2 API Requirements

#### 6.2.1 Market Intelligence → Trading Engine
- `/api/v1/predictions` - Latest impact predictions
- `/api/v1/impact-graph` - Impact graph for specific event/company
- `/api/v1/sentiment` - Sentiment scores and trends
- `/api/v1/events` - Significant events feed

#### 6.2.2 Correlation Tool → Trading Engine
- `/api/v1/correlations` - Correlation matrices by timeframe
- `/api/v1/lead-lag` - Lead-lag relationships
- `/api/v1/regimes` - Current correlation regime status
- `/api/v1/pairs` - Recommended pairs for pairs trading

#### 6.2.3 Internal Trading Engine APIs
- `/api/v1/signals` - Combined trading signals
- `/api/v1/positions` - Current positions
- `/api/v1/orders` - Order management
- `/api/v1/performance` - Performance metrics

### 6.3 Shared Components

#### 6.3.1 Data Storage
- Time-series database (e.g., InfluxDB, TimescaleDB)
- Graph database for impact graphs (e.g., Neo4j)
- Relational database for structured data (e.g., PostgreSQL)
- Object storage for raw data (e.g., S3)
- Caching layer (e.g., Redis)

#### 6.3.2 Message Queue
- Event streaming platform (e.g., Apache Kafka)
- Real-time data distribution
- Asynchronous processing coordination
- Replay capability for testing

#### 6.3.3 Monitoring & Observability
- Centralized logging (e.g., ELK stack)
- Metrics collection (e.g., Prometheus)
- Distributed tracing (e.g., Jaeger)
- Alerting system (e.g., PagerDuty)

---

## 7. Non-Functional Requirements

### 7.1 Performance Requirements

- **Latency:**
  - End-to-end signal-to-order latency < 200ms for day trading
  - Dashboard updates < 1 second
  - API response times < 100ms (p95)

- **Throughput:**
  - Process 100,000+ news articles per day
  - Handle 10,000+ securities simultaneously
  - Execute 1,000+ orders per day
  - Support 50+ concurrent API users

- **Scalability:**
  - Horizontal scaling for computation
  - Auto-scaling based on load
  - Support for geographic distribution

### 7.2 Reliability Requirements

- **Availability:**
  - 99.9% uptime during market hours
  - 99% uptime during off-market hours
  - Scheduled maintenance windows outside market hours

- **Fault Tolerance:**
  - No single point of failure
  - Automated failover mechanisms
  - Data replication across availability zones
  - Graceful degradation under load

- **Data Integrity:**
  - Zero data loss for critical operations
  - Transaction consistency guarantees
  - Regular data validation checks
  - Audit trails for all trading actions

### 7.3 Security Requirements

- **Authentication & Authorization:**
  - Multi-factor authentication required
  - Role-based access control (RBAC)
  - API key management for integrations
  - Session management and timeout policies

- **Data Protection:**
  - Encryption at rest (AES-256)
  - Encryption in transit (TLS 1.3)
  - Secure key management
  - PII data handling compliance

- **Compliance:**
  - Regulatory reporting capabilities
  - Audit logging of all actions
  - Data retention policies
  - SEC/FINRA compliance requirements

### 7.4 Maintainability Requirements

- **Code Quality:**
  - Comprehensive unit test coverage (> 80%)
  - Integration tests for all APIs
  - Code documentation and comments
  - **Static Analysis and Linting (Mandatory):**
    - **Python Static Analysis:**
      - **mypy:** Type checking with strict mode enabled
      - **pylint:** Code quality and PEP 8 compliance (minimum score: 8.5/10)
      - **pytype:** Google's static type checker for additional type safety
    - **C++ Static Analysis:**
      - **clang-tidy:** C++ Core Guidelines checks, modernize checks, and performance checks
      - **cppcheck:** Additional static analysis for undefined behavior and memory leaks
    - **Enforcement:** All code changes MUST pass static analysis before merge
    - **CI/CD Integration:** Automated static analysis on every pull request

- **API Design Standards:**
  - **Fluent/Chainable APIs Required:** All components (C++, Python, Rust) MUST provide fluent interfaces
  - **Method Chaining:** Enable readable, expressive code through method chaining pattern
  - **Builder Pattern:** Use for complex object construction
  - **Immutability Where Appropriate:** Fluent APIs should return new instances when modifying state

- **C++23 Specific Requirements:**
  - **C++23 Modules Required:** All C++ components MUST use modules instead of headers
  - **Trailing Return Syntax:** Use trailing return type syntax (`auto func() -> Type`) for all functions
  - **Modern Features:** Leverage `std::expected`, `std::mdspan`, `std::flat_map`, deducing `this`
  - **Examples:**
    ```cpp
    // C++23 Module with Fluent API and Trailing Return Syntax
    module;
    #include <expected>
    #include <vector>
    #include <string>
    export module bigbrother.correlation;

    export namespace bigbrother::correlation {
        enum class CorrelationMethod { Pearson, Spearman };

        class CorrelationEngine {
        public:
            auto withSymbols(std::vector<std::string> symbols) -> CorrelationEngine& {
                symbols_ = std::move(symbols);
                return *this;
            }

            auto withWindow(int window) -> CorrelationEngine& {
                window_ = window;
                return *this;
            }

            auto withMethod(CorrelationMethod method) -> CorrelationEngine& {
                method_ = method;
                return *this;
            }

            auto calculate() -> std::expected<CorrelationResult, Error>;

        private:
            std::vector<std::string> symbols_;
            int window_ = 20;
            CorrelationMethod method_ = CorrelationMethod::Pearson;
        };
    }
    ```
    ```python
    # Python Fluent API Example
    result = (TradingDecisionEngine()
        .with_strategy("delta_neutral")
        .with_risk_tolerance(0.15)
        .with_position_size(1000)
        .simulate()
        .execute())
    ```
    ```rust
    // Rust Fluent API Example
    let result = PricingEngine::new()
        .with_model(PricingModel::BlackScholes)
        .with_option_type(OptionType::Call)
        .with_strike(150.0)
        .calculate()?;
    ```

- **Deployment:**
  - Containerized deployments (Docker)
  - Infrastructure as Code (Terraform)
  - CI/CD pipelines for automated deployment
  - Blue-green deployment capability

- **Monitoring:**
  - Comprehensive logging
  - Performance metrics collection
  - Alerting for anomalies
  - Dashboard for system health

### 7.5 Usability Requirements

- **User Interface:**
  - Intuitive dashboard design
  - Responsive design for mobile access
  - Customizable layouts and views
  - Dark mode support

- **Documentation:**
  - API documentation (OpenAPI/Swagger)
  - User guides and tutorials
  - Architecture documentation
  - Runbooks for operators

---

## 8. Data Requirements

### 8.1 Data Sources & Acquisition

**Note:** Comprehensive data extraction technologies are detailed in Section 3.2.10 and 3.2.11. This section provides acquisition strategy and cost considerations.

#### 8.1.1 Market Data Providers (Affordable Options)

**Primary Providers (Cost-Effective):**
- **Polygon.io** - Real-time and historical market data (startup-friendly pricing)
  - Required: OHLCV, options chains, order book
  - Latency: < 100ms
  - Cost: ~$200-500/month for startup tier

- **Alpha Vantage** - Free tier + paid options
  - Good for initial development and testing
  - Free tier: 500 API calls/day
  - Premium: ~$50-150/month

- **IEX Cloud** - Transparent pricing, good free tier
  - Pay-per-request model
  - Free tier available for development

- **Finnhub** - Free tier with generous limits
  - Real-time data and corporate actions
  - Free tier: 60 API calls/minute

**Historical Data:**
  - **Yahoo Finance API** - Free historical data (community libraries)
  - **Polygon.io** - 10+ years of daily data included
  - **Alpha Vantage** - Historical data in free tier
  - Format: JSON, CSV via API

#### 8.1.2 News & Sentiment (Budget-Friendly)

**Affordable News APIs:**
- **NewsAPI** - $449/month for business tier (most affordable)
- **MarketAux** - $49-199/month (budget option)
- **Alpha Vantage News API** - Included in premium tier
- **RSS Feeds** - Free from major publications (WSJ, Reuters, FT)
- **PR Newswire** - Pay-per-release or subscription

**Social Media (Free/Low-Cost):**
- **Twitter/X API** - Free tier available (with limits)
- **Reddit API** - Free via PRAW library
- **StockTwits API** - Free tier available
- **Web scraping** - For public data (with rate limiting)

#### 8.1.3 Alternative Data (Free/Low-Cost)

**Government Data (All Free):**
- **FRED API** - Federal Reserve Economic Data (free, unlimited)
- **SEC EDGAR API** - Free regulatory filings
- **Congress.gov API** - Free legislative data
- **All U.S. Government APIs** - Generally free (see Section 3.2.10)

**International Data (Free):**
- **World Bank Data API** - Free
- **IMF Data API** - Free
- **ECB APIs** - Free
- **OECD Data API** - Free

**Alternative Sources:**
- **GDELT Project** - Free global events database
- **EventRegistry** - Free tier available
- **Web Scraping** - For public retail data (legal compliance required)

### 8.2 Data Storage Requirements

- **Hot Storage (Real-time):**
  - Last 30 days of all data
  - Low-latency access (< 10ms)
  - Estimated size: 1-5 TB

- **Warm Storage (Recent):**
  - Last 2 years of all data
  - Medium-latency access (< 100ms)
  - Estimated size: 20-100 TB

- **Cold Storage (Archive):**
  - 10+ years of historical data
  - High-latency access acceptable (< 1 second)
  - Estimated size: 100+ TB

### 8.3 Data Quality & Governance

- Data validation pipelines
- Outlier detection and handling
- Missing data imputation strategies
- Data lineage tracking
- Version control for datasets
- Privacy and compliance policies

---

## 9. Technology Stack - Unified High-Performance Architecture

**CRITICAL:** Speed is of the essence. All technology choices prioritize performance, low latency, massive parallelization, AND affordability. This is a startup-friendly stack that leverages open-source tools and cost-effective solutions without compromising on performance.

**Design Principles:**
- **Unified Database Layer:** PostgreSQL with extensions (instead of multiple databases)
- **Open-Source First:** Prioritize proven open-source tools
- **Cloud-Agnostic:** Start with private servers, cloud-ready when needed
- **Minimal Vendor Lock-in:** Use standard protocols and formats
- **Performance-First:** Ultra-low latency for trading execution
- **Fluent API Design:** All components must provide fluent/chainable APIs for improved code readability and developer experience
- **Modern C++23:** Use modules, trailing return syntax, and latest language features for all C++ code

### 9.1 Programming Languages (Performance-First)

- **C++23:** Primary language for ultra-low latency components
  - Market data ingestion and processing
  - Order execution engine
  - Critical path algorithms
  - Real-time correlation calculations
  - Target: < 1ms latency for critical operations
  - **Key C++23 Features & Modern Primitives:**
    - **Modules:** Replace headers with `export module` for faster compilation and better encapsulation
    - **Trailing Return Syntax:** Use `auto func() -> Type` for all functions (mandatory standard)
    - **Smart Pointers (Mandatory):**
      - `std::unique_ptr` for exclusive ownership (default choice)
      - `std::shared_ptr` for shared ownership (when needed)
      - `std::weak_ptr` for breaking circular references
      - **Never use raw `new`/`delete`** - always use smart pointers or RAII
    - **Memory Management & Optimization:**
      - **Move Semantics:** Efficiently transfer ownership of resources
      - **Rvalue References:** Bind temporary objects to avoid unnecessary copying
      - **Copy-on-Write (COW):** Share resources until modification for large data structures
      - `std::span` for lightweight array views without ownership
      - `std::mdspan` for multi-dimensional array views
    - **Error Handling:**
      - `std::expected<T, E>` for error handling without exceptions (mandatory for hot paths)
      - `std::variant` for type-safe unions
      - `std::optional` for values that may or may not exist
    - **Compile-Time Programming:**
      - `constexpr` for compile-time computation (use extensively)
      - `consteval` for immediate functions
      - Improved `constexpr` support in C++23
    - **Concurrency & Parallelism:**
      - `std::atomic` for lock-free atomic operations
      - `std::jthread` for automatic thread joining
      - Coroutines for asynchronous programming
    - **Modern Containers & Algorithms:**
      - `std::flat_map` and `std::flat_set` for cache-friendly containers (mandatory)
      - Ranges library for composable algorithms
      - Views for lazy evaluation
    - **Performance Features:**
      - Deducing `this` for better performance and fluent APIs
      - `[[likely]]` and `[[unlikely]]` attributes for branch prediction
  - **Coding Standards:**
    - **C++ Core Guidelines Compliance (Mandatory):**
      - Follow [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) for all implementation details
      - **Static Analysis Enforcement:**
        - **clang-tidy:** Run with core-guidelines checks, modernize-*, performance-*, readability-* checks
        - **cppcheck:** Run with --enable=all for comprehensive analysis
        - Both tools MUST be run before code commits
        - Zero warnings policy for production code
      - Key guidelines to emphasize:
        - I.11: Never transfer ownership by raw pointer (use smart pointers)
        - F.15: Prefer simple and conventional ways of passing information (span, const&)
        - C.20: Use RAII wherever possible
        - ES.46: Avoid lossy (narrowing, truncating) arithmetic conversions
        - CP.1: Assume your code will run as part of a multi-threaded program
    - **Standard Template Library (STL) First:**
      - Prefer STL algorithms over hand-written loops (std::transform, std::for_each, etc.)
      - Use STL containers by default (std::vector, std::flat_map, std::unordered_map)
      - Leverage STL numeric algorithms (std::accumulate, std::reduce, std::transform_reduce)
      - Use STL utilities (std::pair, std::tuple, std::tie)
      - Only write custom implementations when STL performance is insufficient
    - **Module System:**
      - ALL C++ files must use modules (no .h/.hpp headers)
      - Use `export module` for public interfaces
      - Use `import` for dependencies
    - **Function Syntax:**
      - ALL functions must use trailing return type syntax (`auto func() -> Type`)
    - **API Design:**
      - ALL APIs must support method chaining (fluent interface)
      - Use builder pattern for complex object construction
    - **Memory Management:**
      - Use smart pointers for all dynamic allocations
      - Prefer `std::unique_ptr` by default
      - Use `std::shared_ptr` only when shared ownership is required
      - Use RAII (Resource Acquisition Is Initialization) pattern
      - **Never use raw `new`/`delete`** in production code
    - **Error Handling:**
      - Use `std::expected` for error handling in hot paths
      - Use `std::variant` for type-safe unions
      - Use `std::optional` for values that may not exist
      - Reserve exceptions for truly exceptional cases only
    - **Performance:**
      - Use `constexpr` for compile-time computation
      - Use `std::span` for safe array access without ownership
      - Use `std::flat_map` instead of `std::map` for better cache locality
      - Use move semantics (`std::move`) and rvalue references extensively
      - Use STL parallel algorithms when appropriate (`std::execution::par`)
    - **Concurrency:**
      - Use `std::atomic` for lock-free shared state
      - Use `std::jthread` instead of `std::thread`
      - Follow lock-free programming guidelines when possible

- **Rust:** High-performance components requiring memory safety
  - Concurrent data structures
  - Network protocols
  - Safety-critical financial calculations
  - Shared nothing parallelism
  - Zero-cost abstractions

- **Python 3.14+ (GIL-Free):** AI/ML development with true parallelism
  - **GIL-Free Mode:** Exploit true multi-threading without Global Interpreter Lock
  - Model training (offline with parallel data loading)
  - Feature engineering (parallel processing across cores)
  - ML inference preprocessing (parallel batch preparation)
  - C++ bindings (pybind11) for performance-critical sections
  - **Key Python 3.14+ Features:**
    - Free-threaded mode for CPU-bound parallel workloads
    - JIT compilation improvements (copy-and-patch JIT)
    - Better memory efficiency
    - Enhanced type system
  - **Parallelism Strategy:**
    - Use Python's native threading for CPU-bound ML tasks
    - Leverage multiprocessing where isolation is needed
    - Combine with CUDA for GPU parallelism
    - C++ extensions for ultimate performance

- **SQL:** Database queries (optimized for time-series)

#### 9.1.1 C++23 Modern Primitives Reference

**Efficient C++23 Primitives and Libraries:**

| Feature | Description | Use Cases | Example |
|---------|-------------|-----------|---------|
| **Smart Pointers** | Automatic memory management | All dynamic allocations | `std::unique_ptr<Data>` |
| **std::expected** | Value or error without exceptions | Error handling in hot paths | `std::expected<Result, Error>` |
| **std::span** | Lightweight array view | Safe array manipulations | `std::span<const double>` |
| **std::mdspan** | Multi-dimensional array view | Matrix operations | `std::mdspan<double, 2>` |
| **Move Semantics** | Transfer ownership efficiently | Resource management | `std::move(large_data)` |
| **Rvalue References** | Bind temporary objects | Container optimization | `void func(Data&&)` |
| **std::variant** | Type-safe union | Flexible data types | `std::variant<int, double>` |
| **std::optional** | Value that may not exist | Nullable values | `std::optional<Price>` |
| **constexpr** | Compile-time computation | Performance optimization | `constexpr auto calc()` |
| **std::atomic** | Lock-free operations | Multithreaded access | `std::atomic<int>` |
| **std::flat_map** | Cache-friendly map | High-performance lookup | Replace `std::map` |
| **Ranges** | Composable algorithms | Data manipulation | `data \| views::filter` |
| **Copy-on-Write** | Share until modified | Large data structures | String, vectors |
| **std::jthread** | Auto-joining threads | Concurrent operations | Replace `std::thread` |
| **Coroutines** | Asynchronous programming | I/O operations | `co_await`, `co_return` |

**Example Usage:**

```cpp
// Smart Pointers - Automatic memory management
auto engine = std::make_unique<CorrelationEngine>();
auto shared_data = std::make_shared<PriceData>();

// std::expected - Error handling without exceptions
auto calculate_price(const Option& opt) -> std::expected<double, Error> {
    if (!opt.valid()) {
        return std::unexpected(Error{"Invalid option"});
    }
    return opt.strike * 1.05;
}

// std::span - Safe array access
auto process_prices(std::span<const double> prices) -> double {
    return std::accumulate(prices.begin(), prices.end(), 0.0);
}

// std::variant - Type-safe union
using OrderType = std::variant<MarketOrder, LimitOrder, StopOrder>;
auto execute(const OrderType& order) -> void {
    std::visit([](auto&& o) { o.execute(); }, order);
}

// constexpr - Compile-time computation
constexpr auto fibonacci(int n) -> int {
    return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2);
}

// std::atomic - Lock-free operations
std::atomic<int> counter{0};
counter.fetch_add(1, std::memory_order_relaxed);

// Ranges - Composable algorithms
auto filtered = prices
    | std::views::filter([](auto p) { return p > 100.0; })
    | std::views::transform([](auto p) { return p * 1.05; });
```

### 9.2 Parallel Computing & High-Performance Libraries

**Message Passing & Distribution:**
- **MPI (Message Passing Interface):** Distributed parallel computing across cores
- **OpenMP:** Shared memory parallelization within nodes
- **UPC++ (Unified Parallel C++):** Partitioned global address space programming
- **pdsh:** Parallel distributed shell for cluster management

**Performance Libraries:**
- **Intel oneAPI / MKL:** Optimized math libraries
- **BLAS/LAPACK:** Linear algebra operations
- **Eigen:** C++ template library for linear algebra
- **Boost:** High-performance C++ libraries
- **Threading Building Blocks (TBB):** Parallel programming

### 9.3 Machine Learning & AI (GPU-Accelerated)

**Model Serving (Critical):**
- **vLLM:** High-throughput, low-latency LLM inference serving
  - GPU-accelerated inference
  - Continuous batching
  - PagedAttention for memory efficiency
  - Target: > 10,000 predictions/second
  - Python 3.14+ GIL-free for parallel request handling

**Training Frameworks:**
- **PyTorch with CUDA:** Primary deep learning framework
  - Multi-GPU training with DDP (DistributedDataParallel)
  - Mixed precision training (FP16/BF16)
  - Python 3.14+ GIL-free for parallel data loading and preprocessing
  - **✅ DEPLOYED:** Price Predictor v3.0 trained with PyTorch (24,300 samples, DirectionalLoss)
- **TensorRT:** NVIDIA inference optimization
- **ONNX Runtime:** Cross-platform inference
  - **✅ DEPLOYED:** Production price predictor (60-feature model, 56.3%/56.6% accuracy)
  - C++ integration: `src/market_intelligence/price_predictor.cppm` (525 lines)
  - AVX2 SIMD normalization: 8-way parallel StandardScaler (8x speedup)
  - Model file: `models/price_predictor.onnx` (58,947 parameters)
- **XGBoost/LightGBM:** Gradient boosting (with GPU support)
- **cuML (RAPIDS):** GPU-accelerated machine learning
- **JAX:** High-performance numerical computing with XLA compilation
  - **✅ USED:** Greeks calculation (340K-384K samples/sec on A100 GPU)

**NLP (GPU-Accelerated):**
- **Hugging Face Transformers with CUDA**
- **spaCy with GPU support**
- **Custom CUDA kernels for specialized operations**

**Python Parallelism Strategy (GIL-Free):**
- **True Multi-threading:** Exploit Python 3.14+ free-threaded mode for CPU-bound ML tasks
  - Parallel feature extraction across cores
  - Concurrent model inference for different instruments
  - Parallel data preprocessing pipelines
- **Combined GPU + CPU Parallelism:**
  - GPU for training and heavy inference
  - CPU threads for data preparation and post-processing
  - Overlap computation and I/O
- **C++ Extensions:** Critical sections in C++23 with pybind11 bindings
- **Performance Target:** Near-linear scaling with core count for embarrassingly parallel tasks

### 9.4 Data Processing (High-Performance)

**Real-Time Stream Processing:**
- **Custom C++ pipelines:** Zero-copy, lock-free data structures
- **Apache Kafka:** Event streaming (producer/consumer)
- **ZeroMQ:** Low-latency messaging
- **Shared memory IPC:** Inter-process communication

**Batch Processing & Analytics (Parallel):**
- **DuckDB:** Blazing-fast analytical queries on Parquet/CSV files
  - Multi-threaded, vectorized execution
  - Direct file queries without loading
  - Perfect for backtesting and historical analysis
  - **Zero setup, zero cost**
- **Apache Spark with C++ UDFs:** Large-scale parallel processing (when needed)
- **Dask:** Parallel computing in Python
- **GNU Parallel:** Shell-level parallelization

**Time Series (Optimized):**
- **DuckDB:** Fast aggregations and window functions for time-series
- **Custom C++ time-series libraries:** Microsecond-level operations
- **TA-Lib with C++ bindings:** Technical indicators
- **NumPy with MKL:** Vectorized operations
- **Polars:** Fast DataFrame library (Rust-based, similar performance to DuckDB)

### 9.5 Database Strategy (DuckDB-First Approach)

**DUCKDB-FIRST APPROACH FOR TIER 1:** Start with DuckDB exclusively (zero setup, instant start). Add PostgreSQL ONLY in Tier 2 after proving profitability. This eliminates setup complexity and allows focus on algorithm validation.

**IMPORTANT:** See [docs/architecture/database-strategy-analysis.md](./architecture/database-strategy-analysis.md) for detailed decision analysis and migration strategy.

### 9.5.1 DuckDB - Embedded Analytics Database (Rapid Development)

**CRITICAL FOR QUICK START:** DuckDB enables instant deployment with zero infrastructure setup. Perfect for initial development phase with no subscription fees.

**Key Features:**
- **Embedded database** - No server, just a file (like SQLite but for analytics)
- **Zero setup time** - `pip install duckdb` and you're running
- **Blazing fast analytics** - Columnar storage, vectorized execution
- **Direct file queries** - Query CSV, JSON, Parquet without loading
- **Arrow integration** - Zero-copy data sharing with pandas, polars
- **SQL interface** - Standard SQL with extensions
- **Multi-threaded** - Automatic parallelization across cores
- **Cost:** FREE (open-source, MIT license)

**Use Cases for DuckDB:**

1. **Initial Development & Prototyping:**
   - Start building immediately without PostgreSQL setup
   - Iterate rapidly on data models
   - Test queries and analytics workflows
   - No configuration needed

2. **Analytical Workloads:**
   - Historical data analysis (10+ years of market data)
   - Backtesting strategies on large datasets
   - Ad-hoc queries on archived data
   - Exploratory data analysis (EDA)
   - Statistical analysis and aggregations

3. **Data Lake Queries:**
   - Query Parquet files directly from disk
   - No need to load into database first
   - Perfect for cold storage (archived data)
   - Example: `SELECT * FROM 'data/market_history/*.parquet'`

4. **ETL & Data Processing:**
   - Fast data transformation pipelines
   - Join data from multiple sources
   - Export to Parquet for archival
   - Pre-process before loading to PostgreSQL

5. **Local Development:**
   - Each developer runs local DuckDB instance
   - No shared database infrastructure needed
   - Portable database files
   - Version control friendly

**Performance Characteristics:**
- **Scan speed:** Billions of rows per second on modern hardware
- **Aggregations:** 10-100x faster than traditional row-based databases
- **Joins:** Optimized for analytical workloads
- **Compression:** Automatic columnar compression
- **Memory:** Out-of-core processing for datasets larger than RAM

**Integration with Python:**
```python
import duckdb
import pandas as pd

# Create in-memory database (or use file: duckdb.connect('trading.db'))
con = duckdb.connect()

# Query Parquet files directly
result = con.execute("""
    SELECT date, symbol, close, volume
    FROM 'data/prices/*.parquet'
    WHERE date >= '2020-01-01'
    ORDER BY date DESC
""").df()  # Returns pandas DataFrame

# Query CSV files
news = con.execute("SELECT * FROM 'data/news/*.csv'").df()

# Query pandas DataFrame directly
con.execute("SELECT * FROM result WHERE close > 100").df()
```

**Transition Strategy (DuckDB-First):**
- **Tier 1 POC (Weeks 1-12):** Use DuckDB EXCLUSIVELY for rapid development and validation
- **Decision Point (Month 4):** Evaluate profitability - only proceed if proven
- **Tier 2 Production (Month 5+, OPTIONAL):** Add PostgreSQL for enhanced operations
- **Long-term (if scaled):** DuckDB for analytics, PostgreSQL for high-frequency operations
- **Note:** Many traders continue with DuckDB-only successfully at POC scale

### 9.5.2 PostgreSQL - Production Database (Tier 2 Only, Optional)

**DEFERRED TO TIER 2:** PostgreSQL is NOT installed in Tier 1 POC. Add only after proving profitability (Month 5+).

**PostgreSQL 16+ Features (when added):**
- **Production operations** for high-frequency trading at scale
- **Core relational database** for operational data (trades, positions)
- **Proven performance** at institutional scale
- **ACID compliance** for financial operations
- **Rich ecosystem** of tools and extensions
- **Zero licensing cost**
- **Migration from DuckDB:** 1-2 days

**Critical Extensions:**

1. **TimescaleDB** - Time-Series Data
   - Automatic partitioning and compression
   - Optimized for time-series queries
   - Continuous aggregates for real-time analytics
   - Hypertables for market data, price history
   - Target: < 10ms query latency for recent data
   - **Replaces:** QuestDB, InfluxDB, specialized time-series DBs

2. **pgvector** - Vector Embeddings & Semantic Search
   - Store and query embeddings for NLP models
   - Similarity search for news articles, documents
   - Index types: HNSW for speed, IVFFlat for memory efficiency
   - **Replaces:** Pinecone, Weaviate, Milvus, dedicated vector DBs

3. **Apache AGE** (A Graph Extension) - Graph Database
   - Native graph database within PostgreSQL
   - Cypher query language support
   - Impact graph analysis and traversal
   - Multi-hop relationship queries
   - **Replaces:** Neo4j, dedicated graph databases

4. **pg_partman** - Partition Management
   - Automated partition creation and maintenance
   - Time-based and serial-based partitioning
   - Background worker for partition management

5. **pg_cron** - Job Scheduling
   - Schedule database maintenance tasks
   - Automated data archival and cleanup
   - Report generation scheduling

**PostgreSQL Performance Tuning:**
- **Shared memory buffers:** 25% of RAM (64GB for 256GB system)
- **Connection pooling:** PgBouncer for connection management
- **Parallel queries:** Max parallel workers = core count
- **JIT compilation:** Enabled for complex queries
- **BRIN indexes:** For time-series data (block range indexes)
- **Custom tablespaces:** Separate fast NVMe for hot data

**Caching & In-Memory Layer:**
- **Redis** - Primary cache
  - Session data and real-time state
  - Pub/sub for real-time notifications
  - Rate limiting and quota management
  - Distributed locks
  - Target: < 1ms latency
  - Cost: Free (open-source)

- **PostgreSQL Shared Buffers** - Database-level cache
  - Keep hot data in memory
  - Automatic cache management

- **Custom shared memory** (C++) - Ultra-fast IPC
  - Zero-copy data sharing between processes
  - Market data distribution
  - Order book snapshots

**Data Archival Strategy (DuckDB + PostgreSQL):**

**Multi-Tier Storage Approach:**

1. **Hot Tier (PostgreSQL/Redis):** Last 30 days, < 10ms access
   - Real-time trading operations
   - Active positions and orders
   - Live market data feed
   - In-memory caching with Redis

2. **Warm Tier (PostgreSQL compressed):** Last 2 years, < 100ms access
   - Recent historical analysis
   - Strategy backtesting on recent data
   - TimescaleDB compression policies
   - Weekly analytical queries

3. **Cold Tier (Parquet files + DuckDB):** 10+ years, < 1s access
   - **Store data as Parquet files** on disk (efficient compression)
   - **Query with DuckDB** directly without loading to database
   - Perfect for long-term backtesting
   - Example: 10 years of minute-level data for 5,000 stocks
   - Storage: ~500GB compressed Parquet (vs. ~5TB uncompressed)
   - Query speed: Seconds to scan billions of rows

**DuckDB Cold Storage Example:**
```python
import duckdb

# Query 10 years of historical data stored as Parquet files
con = duckdb.connect()

# Calculate correlations across entire history
result = con.execute("""
    SELECT
        symbol_a, symbol_b,
        corr(returns_a, returns_b) as correlation,
        count(*) as data_points
    FROM (
        SELECT
            a.symbol as symbol_a,
            b.symbol as symbol_b,
            a.close / lag(a.close) OVER (PARTITION BY a.symbol ORDER BY a.date) - 1 as returns_a,
            b.close / lag(b.close) OVER (PARTITION BY b.symbol ORDER BY b.date) - 1 as returns_b
        FROM 'data/archive/*.parquet' a
        JOIN 'data/archive/*.parquet' b ON a.date = b.date
        WHERE a.date >= '2014-01-01'
    )
    GROUP BY symbol_a, symbol_b
    HAVING correlation > 0.7
    ORDER BY correlation DESC
""").df()

print(f"Analyzed 10 years of data in seconds, found {len(result)} correlations")
```

**Storage Cost Comparison (10 years, 5,000 stocks, minute-level):**
- Raw CSV: ~10 TB
- PostgreSQL uncompressed: ~5 TB
- PostgreSQL TimescaleDB compressed: ~1 TB
- Parquet files: ~500 GB (10x reduction!)
- DuckDB can query Parquet directly without loading

**Benefits of DuckDB for Archival:**
- No need to load data into database
- Query files directly from disk
- Fast scans even on HDD (sequential reads)
- Easy to backup (just copy Parquet files)
- Portable across systems
- Zero maintenance overhead

### 9.5.1 Unified Data Extraction & Processing Stack

**CRITICAL:** This section consolidates all data extraction, processing, and ingestion technologies into a coherent, maintainable architecture.

**Data Extraction Layer (Web Scraping & API Integration):**

1. **Scrapy** - Primary web scraping framework
   - Asynchronous, high-performance
   - Built-in rate limiting and robotics.txt support
   - Middleware for rotating proxies and headers
   - Use for: Government sites, news sites, corporate IR pages
   - Cost: Free (open-source)

2. **Playwright** - Modern browser automation
   - Headless browser for JavaScript-heavy sites
   - Better than Selenium for modern web apps
   - API testing and dynamic content extraction
   - Use for: Complex SPAs, authenticated scraping
   - Cost: Free (open-source)

3. **httpx / aiohttp** - Async HTTP clients
   - Concurrent API requests
   - Connection pooling and retry logic
   - Use for: REST API integrations (FRED, SEC, etc.)
   - Cost: Free (open-source)

4. **Requests** - Simple synchronous HTTP
   - For simple API calls and one-off requests
   - Wide compatibility
   - Cost: Free (open-source)

**Document Processing (PDFs, Office Docs):**

1. **Apache Tika** - Universal document parser
   - Handles PDF, Word, Excel, PowerPoint
   - Extract text and metadata
   - Use for: SEC filings, regulatory documents
   - Cost: Free (open-source)

2. **pdfplumber** - Detailed PDF extraction
   - Table extraction from PDFs
   - Layout-aware text extraction
   - Use for: Financial statements, government reports
   - Cost: Free (open-source)

3. **Tabula-py** - PDF table extraction
   - Specialized for tabular data in PDFs
   - Use for: Statistical reports, data tables
   - Cost: Free (open-source)

4. **Tesseract OCR** - Optical character recognition
   - For scanned documents
   - Use when PDFs are images
   - Cost: Free (open-source)

**Data Pipeline Orchestration:**

1. **Apache Airflow** - Workflow orchestration
   - DAG-based scheduling
   - Monitoring and alerting
   - Extensive operator library
   - Schedule all data extraction jobs
   - Cost: Free (open-source)
   - **Alternative:** Prefect (more modern, also free)

2. **Apache Kafka** - Event streaming platform
   - Real-time data streaming
   - Durable message queue
   - Publish/subscribe for news feeds
   - Stream market data updates
   - Cost: Free (open-source)
   - **Lightweight alternative:** Redis Streams (for smaller scale)

3. **Celery** - Distributed task queue
   - Async task execution
   - Priority queues
   - Use for: Background processing, batch jobs
   - Works with Redis as broker
   - Cost: Free (open-source)

**Data Validation & Quality:**

1. **Pydantic** - Data validation
   - Schema validation for API responses
   - Type checking and serialization
   - Cost: Free (open-source)

2. **Great Expectations** - Data quality framework
   - Automated data validation
   - Data profiling and documentation
   - Pipeline testing
   - Cost: Free (open-source core)

3. **pandas-profiling (ydata-profiling)** - Data exploration
   - Automated EDA reports
   - Data quality checks
   - Cost: Free (open-source)

**NLP & Text Processing:**

1. **spaCy** - Industrial NLP
   - Named entity recognition (companies, people, locations)
   - Dependency parsing
   - Fast and production-ready
   - Pre-trained models available
   - Cost: Free (open-source)

2. **Hugging Face Transformers** - Modern NLP models
   - BERT, GPT, FinBERT for sentiment
   - Fine-tunable on domain data
   - Use for: Sentiment analysis, text classification
   - Cost: Free (open-source models)

3. **NLTK** - Traditional NLP toolkit
   - Tokenization, stemming, lemmatization
   - Backup for basic text processing
   - Cost: Free (open-source)

4. **sentence-transformers** - Semantic embeddings
   - Generate embeddings for semantic search
   - Store in pgvector
   - Cost: Free (open-source)

**API Management & Rate Limiting:**

1. **python-ratelimit** - Rate limit decorator
   - Enforce API rate limits
   - Prevent hitting provider limits
   - Cost: Free (open-source)

2. **tenacity** - Retry logic
   - Exponential backoff
   - Configurable retry strategies
   - Cost: Free (open-source)

3. **requests-cache** - HTTP caching
   - Cache API responses
   - Reduce API costs
   - Cost: Free (open-source)

**Data Format & Serialization:**

1. **Apache Arrow** - Columnar data format
   - Zero-copy data sharing
   - Language-agnostic
   - Fast analytics
   - Cost: Free (open-source)

2. **Parquet** - Columnar storage format
   - Efficient compression
   - Ideal for archival data
   - Cost: Free (open-source)

3. **Protocol Buffers (protobuf)** - Serialization
   - Efficient binary serialization
   - Schema evolution
   - Use for: Internal service communication
   - Cost: Free (open-source)

**Monitoring & Observability:**

1. **Prometheus** - Metrics collection
   - Time-series metrics database
   - Scraping and alerting
   - Cost: Free (open-source)

2. **Grafana** - Visualization
   - Dashboards for monitoring
   - Alerts and notifications
   - Cost: Free (open-source)

3. **ELK Stack** - Log aggregation (Optional, start simple)
   - Elasticsearch: Search and analytics
   - Logstash: Log processing
   - Kibana: Visualization
   - Cost: Free (open-source)
   - **Simpler alternative:** Loki + Grafana (lighter weight)

4. **Sentry** - Error tracking
   - Application error monitoring
   - Stack traces and context
   - Free tier: 5K events/month
   - Cost: Free tier then $26/month

**Estimated Monthly Costs by Phase:**

**Phase 1 - Validation (Months 1-2): $0/month**
- Market Data: Alpha Vantage Free + Yahoo Finance ($0)
- Government Data: FRED, SEC, Congress APIs ($0)
- News: RSS feeds + NewsAPI free tier ($0)
- Database: DuckDB ($0)
- Infrastructure: Own hardware ($0)
- All libraries: Open-source ($0)
- **Total: $0/month**

**Phase 2 - Initial Production (Months 3-4): $200-300/month**
- Market Data: Polygon.io Starter ($200-250)
- News: NewsAPI or MarketAux Basic ($50)
- Database: PostgreSQL + DuckDB ($0)
- Infrastructure: Own hardware ($0)
- Monitoring: Sentry free tier ($0)
- **Total: $250-300/month**

**Phase 3 - Full Production (Month 5+): $500-1,000/month**
- Market Data: Polygon.io + options data ($400-500)
- News: NewsAPI Business ($449) or MarketAux Pro ($199)
- Database: PostgreSQL + DuckDB ($0)
- Infrastructure: Own hardware ($0)
- Monitoring: Sentry paid ($26)
- Misc APIs: $25-50
- **Total: $500-1,000/month**

**Compare to Enterprise:** $25,000+/month with Bloomberg Terminal
**Annual Savings:** $288,000 - $300,000/year

### 9.6 Infrastructure (Private Server Deployment)

**Operating System:**
- **Red Hat Enterprise Linux (RHEL) with OpenShift**
  - Container orchestration with Kubernetes
  - Enterprise support and security
- **Ubuntu Server 22.04 LTS (Alternative)**
  - Long-term support
  - Strong HPC ecosystem

**Deployment & Configuration:**
- **Ansible:** Infrastructure as code, automated deployment
- **Podman/Docker:** Container runtime
- **OpenShift/Kubernetes:** Container orchestration (when needed)
- **systemd:** Service management and monitoring

**Hardware Requirements:**
- **CPU:** 32+ cores (Intel Xeon or AMD EPYC)
- **GPU:** NVIDIA A100/H100 or RTX 4090 (for CUDA/vLLM)
- **RAM:** 256GB+ (for in-memory operations)
- **Storage:** NVMe SSDs (low-latency data access)
- **Network:** 10Gb+ Ethernet (low-latency data feeds)

**Monitoring & Observability:**
- **Prometheus:** Metrics collection
- **Grafana:** Visualization and dashboards
- **ELK Stack (Elasticsearch, Logstash, Kibana):** Log aggregation
- **perf/VTune:** Performance profiling tools
- **Custom latency monitoring:** Microsecond-level tracking

**CI/CD:**
- **GitHub Actions:** Automated testing and deployment
- **CMake 4.1.2+:** Latest C++23 build system with modern features
- **Compilers (Latest and Greatest):**
  - **Clang 21.1.5** (LLVM Project) - chosen for C/C++/Fortran compilation
    - Full C++23 support with latest optimizations
    - **Flang 21.1.5** - Modern Fortran compiler for numerical libraries
    - **MLIR** infrastructure for advanced optimizations
    - OpenMP 21 runtime included
    - Reason for choice: Superior compatibility with WSL2, no glibc/pthread conflicts
    - Alternative: GCC 15+ also supported but requires careful glibc matching
  - **Rust 1.75+** for latest language features
  - **Python 3.14+** with free-threaded mode enabled
- **Cargo:** Rust build system
- **Conan/vcpkg:** C++ package management
- **Poetry/uv:** Python dependency management (fast resolver)
- **Static Analysis Tools (Mandatory):**
  - **Python:** mypy, pylint, pytype (all installed via uv)
  - **C++:** clang-tidy 21 (from Clang 21 toolchain), cppcheck (latest)
    - clang-tidy now built-in with Clang 21.1.5 installation
    - C++ Core Guidelines enforcement
    - modernize-* checks for C++23 features
  - **Pre-commit hooks:** Automatic static analysis on git commit
  - **CI/CD enforcement:** All PRs must pass static analysis checks

### 9.6.1 Tier 1 Development Deployment Stack (Detailed)

**CRITICAL:** This section specifies the exact technology stack and setup procedures for Tier 1 development deployment (Months 1-4). This is the foundation for rapid prototyping and validation before production scaling.

#### 9.6.1.1 Operating System Selection

**Primary Choice: Red Hat Enterprise Linux (RHEL) 9+**
- **Advantages:**
  - Enterprise-grade stability and security
  - Integrated OpenShift for container orchestration
  - Long-term support (10 years)
  - Optimized for HPC workloads
  - Built-in SELinux for security
  - Commercial support available

**Alternative: Ubuntu Server 22.04 LTS**
- **Advantages:**
  - Strong community support
  - Excellent HPC ecosystem
  - Easy package availability
  - Free for personal/commercial use
  - Long-term support (5 years with ESM extension)

**Decision Matrix:**
```
Use RHEL if:
- Enterprise support needed
- OpenShift orchestration required
- Maximum stability critical
- Budget allows subscription (~$350-800/year)

Use Ubuntu if:
- Zero subscription cost required
- Community support sufficient
- Latest packages needed faster
- Development/startup phase
```

#### 9.6.1.2 Core Development Tools Installation

**1. Clang/LLVM 21 Toolchain (Built from Source)**

**CHOSEN TOOLCHAIN: Clang 21.1.5 + Flang 21 + MLIR + OpenMP**

Why LLVM/Clang 21:
- **Latest stable release** (Clang 21.1.5, released 2025)
- **Full C++23 support** with excellent standards conformance
- **Flang Fortran compiler** for numerical library integration
- **Superior WSL2 compatibility** - no glibc/pthread version conflicts
- **Integrated OpenMP 21** runtime (no separate installation)
- **Better diagnostics** than GCC for template errors
- **MLIR infrastructure** for advanced compiler optimizations

Build from Source (Recommended for WSL2):
```bash
# Install build dependencies
sudo apt-get install -y build-essential cmake ninja-build python3 \
    libz3-dev libxml2-dev zlib1g-dev

# Download LLVM 21.1.5
cd ~/toolchain-build
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.5/llvm-project-21.1.5.src.tar.xz
tar -xf llvm-project-21.1.5.src.tar.xz

# Configure (includes Clang, Flang, MLIR, OpenMP)
mkdir llvm-build && cd llvm-build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;flang;openmp" \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DLLVM_ENABLE_LTO=OFF \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  ../llvm-project-21.1.5.src/llvm

# Build (takes 45-90 minutes)
ninja -j$(nproc)
sudo ninja install

# Verify installation
clang --version          # Should show 21.1.5
clang++ --version        # C++ compiler
flang-new --version      # Fortran compiler

# Verify installations
gcc-15 --version             # Should show GCC 15.x
ld --version                 # Should show latest binutils
cmake --version              # Should show CMake 4.1.2+
```

**2. Python 3.14+ with uv (Development Environment Manager)**

Why uv:
- Rust-based, ultra-fast dependency resolver (10-100x faster than pip)
- Better than Poetry for large projects
- Handles virtual environments automatically
- Lockfile support for reproducibility
- Compatible with pip and requirements.txt

Installation:
```bash
# Install uv (Rust-based Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.14 (when available, or 3.13 currently)
# Option 1: Via uv (recommended)
uv python install 3.14      # Or 3.13 currently

# Option 2: Via Homebrew
brew install python@3.14    # Or python@3.13

# Create project environment with uv
cd /path/to/BigBrotherAnalytics
uv venv --python 3.14       # Creates .venv with Python 3.14
source .venv/bin/activate   # Activate environment

# Install dependencies (faster than pip)
uv pip install -r requirements.txt

# For GIL-free mode (when Python 3.14 available):
uv python install 3.14t     # 't' suffix for free-threaded build
```

**3. C++23 with OpenMP and OpenMPI**

Complete Setup:
```bash
# OpenMP (already included in GCC 15)
# Verify OpenMP support
echo '#include <omp.h>
int main() {
    #pragma omp parallel
    printf("Hello from thread %d\n", omp_get_thread_num());
    return 0;
}' | gcc-15 -fopenmp -x c++ - -o test_omp && ./test_omp

# OpenMPI installation
brew install open-mpi        # Latest OpenMPI 5.x

# Verify MPI installation
mpirun --version
mpicc --version

# Test MPI
echo '#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Hello from rank %d\n", rank);
    MPI_Finalize();
    return 0;
}' > test_mpi.cpp
mpic++ -std=c++23 test_mpi.cpp -o test_mpi
mpirun -np 4 ./test_mpi

# Install UPC++ and Berkeley Distributed Components (PGAS)
# RECOMMENDED: Use Ansible playbook for automated installation
# See: playbooks/install-upcxx-berkeley.yml

# Quick install via Homebrew (simpler)
brew install upcxx           # Basic UPC++ installation

# OR: Complete Berkeley components installation (GASNet-EX + UPC++ + BUPC)
# For full PGAS stack, see: https://github.com/oldboldpilot/ClusterSetupAndConfigs
# Complete installation guide in ClusterSetupAndConfigs/DEPLOYMENT_GUIDE.md

# Automated installation with Ansible:
ansible-playbook playbooks/install-upcxx-berkeley.yml

# This installs:
#   - GASNet-EX 2024.5.0 (communication layer)
#   - UPC++ 2024.3.0 (PGAS programming model)
#   - Berkeley UPC (optional, for legacy code)
#   - OpenSHMEM 1.5.2 (optional)
# All configured for MPI conduit and optimized for performance

# Manual installation (if not using Ansible):
# Download and install GASNet-EX
wget https://gasnet.lbl.gov/download/GASNet-2024.5.0.tar.gz
tar xzf GASNet-2024.5.0.tar.gz
cd GASNet-2024.5.0
./configure --prefix=/opt/berkeley/gasnet --enable-mpi --enable-par \
    CC=$(brew --prefix)/bin/gcc CXX=$(brew --prefix)/bin/g++
make -j $(nproc) && make install

# Download and install UPC++
wget https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-2024.3.0.tar.gz
tar xzf upcxx-2024.3.0.tar.gz
cd upcxx-2024.3.0
./install /opt/berkeley/upcxx --with-gasnet=/opt/berkeley/gasnet

# Set environment variables
export UPCXX_INSTALL=/opt/berkeley/upcxx
export PATH=/opt/berkeley/upcxx/bin:$PATH
export LD_LIBRARY_PATH=/opt/berkeley/upcxx/lib:$LD_LIBRARY_PATH

# Verify installation
upcxx --version

# Test UPC++
echo '#include <upcxx/upcxx.hpp>
int main() {
    upcxx::init();
    std::cout << "Hello from rank " << upcxx::rank_me()
              << " of " << upcxx::rank_n() << std::endl;
    upcxx::finalize();
    return 0;
}' > test_upcxx.cpp
upcxx -std=c++23 test_upcxx.cpp -o test_upcxx
upcxx-run -n 4 ./test_upcxx

# For complete cluster setup and advanced configurations:
# See: https://github.com/oldboldpilot/ClusterSetupAndConfigs
```

**4. CUDA and PyTorch**

CUDA Installation:
```bash
# For RHEL 9
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install cuda-toolkit-13-0

# For Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-0

# Set environment variables
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA
nvcc --version
nvidia-smi

# Test CUDA
echo '#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n", deviceCount);
    return 0;
}' > test_cuda.cu
nvcc -o test_cuda test_cuda.cu
./test_cuda
```

PyTorch with CUDA:
```bash
# Using uv for ALL Python package installation (10-100x faster than pip)
source .venv/bin/activate

# Install PyTorch with CUDA 13.0 support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
           print(f'CUDA available: {torch.cuda.is_available()}'); \
           print(f'CUDA version: {torch.version.cuda}'); \
           print(f'GPU count: {torch.cuda.device_count()}')"

# Install vLLM for high-throughput inference
uv pip install vllm

# Install other ML libraries
uv pip install \
    transformers accelerate \
    sentence-transformers \
    xgboost lightgbm \
    scikit-learn \
    polars pyarrow duckdb
```

#### 9.6.1.3 Infrastructure Automation with Ansible

**Ansible Setup:**
```bash
# Install Ansible
brew install ansible         # Via Homebrew
# OR
uv pip install ansible      # Via uv/pip

# Verify Ansible
ansible --version
```

**Ansible Playbook for Tier 1 Setup:**
```yaml
# File: playbooks/tier1-setup.yml
---
- name: BigBrotherAnalytics Tier 1 Development Setup
  hosts: localhost
  become: yes
  vars:
    gcc_version: "15"
    python_version: "3.14"
    cuda_version: "12.3"

  tasks:
    - name: Install system dependencies (RHEL)
      dnf:
        name:
          - git
          - wget
          - curl
          - vim
          - htop
          - tmux
        state: latest
      when: ansible_os_family == "RedHat"

    - name: Install system dependencies (Ubuntu)
      apt:
        name:
          - git
          - wget
          - curl
          - vim
          - htop
          - tmux
          - build-essential
        state: latest
        update_cache: yes
      when: ansible_os_family == "Debian"

    - name: Install Homebrew
      shell: |
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      become_user: "{{ ansible_user_id }}"
      args:
        creates: /usr/local/bin/brew

    - name: Install GCC, binutils, CMake via Homebrew
      homebrew:
        name:
          - gcc@{{ gcc_version }}
          - binutils
          - cmake
          - ninja
          - open-mpi
        state: latest
      become_user: "{{ ansible_user_id }}"

    - name: Install UPC++ and Berkeley Distributed Components (PGAS)
      include_tasks: ../playbooks/install-upcxx-berkeley.yml
      # Installs: GASNet-EX, UPC++, Berkeley UPC
      # See: https://github.com/oldboldpilot/ClusterSetupAndConfigs for details

    - name: Install uv (Python package manager)
      shell: curl -LsSf https://astral.sh/uv/install.sh | sh
      become_user: "{{ ansible_user_id }}"
      args:
        creates: ~/.cargo/bin/uv

    - name: Install Python {{ python_version }}
      shell: ~/.cargo/bin/uv python install {{ python_version }}
      become_user: "{{ ansible_user_id }}"

    - name: Install PostgreSQL 16
      include_tasks: postgres_install.yml

    - name: Install Redis
      package:
        name: redis
        state: latest

    - name: Clone BigBrotherAnalytics repository
      git:
        repo: 'https://github.com/yourusername/BigBrotherAnalytics.git'
        dest: /opt/bigbrother
        version: main
      become_user: "{{ ansible_user_id }}"

    - name: Create Python virtual environment
      shell: |
        cd /opt/bigbrother
        ~/.cargo/bin/uv venv --python {{ python_version }}
      become_user: "{{ ansible_user_id }}"

    - name: Install Python dependencies
      shell: |
        cd /opt/bigbrother
        source .venv/bin/activate
        ~/.cargo/bin/uv pip install -r requirements.txt
      become_user: "{{ ansible_user_id }}"
```

**Run Ansible Playbook (ONE-COMMAND INSTALLATION):**

**🚀 RECOMMENDED: Complete All-in-One Playbook**
```bash
# Installs EVERYTHING in one run (2-4 hours):
# - Homebrew + GCC 15 + CMake
# - OpenMP/MPI/UPC++/GASNet-EX (PGAS)
# - Intel MKL
# - NVIDIA CUDA (if GPU detected)
# - PostgreSQL 16 + TimescaleDB + AGE + pgvector
# - Redis
# - Python 3.14 + uv + all ML frameworks
# - PyTorch + CUDA + Transformers + Hugging Face
# - Complete verification

ansible-playbook playbooks/complete-tier1-setup.yml

# After installation:
source /etc/profile.d/bigbrother_env.sh
cd /opt/bigbrother
source .venv/bin/activate
./scripts/verify_complete_setup.sh
```

**Alternative: Modular Installation**
```bash
# Step-by-step (if you prefer control)
ansible-playbook playbooks/tier1-setup.yml
ansible-playbook playbooks/install-upcxx-berkeley.yml

# Run on remote server
ansible-playbook -i inventory.ini playbooks/complete-tier1-setup.yml
```

**See:** `playbooks/README.md` for complete playbook documentation

#### 9.6.1.4 Container Orchestration (Optional for Tier 1)

**OpenShift/Kubernetes (RHEL):**
```bash
# Install OpenShift Local (formerly CodeReady Containers)
# For development/testing on local machine
wget https://developers.redhat.com/content-gateway/rest/mirror/pub/openshift-v4/clients/crc/latest/crc-linux-amd64.tar.xz
tar xf crc-linux-amd64.tar.xz
sudo cp crc-linux-*/crc /usr/local/bin/

# Setup OpenShift Local
crc setup
crc start

# Access OpenShift console
crc console
```

**Docker/Podman (Alternative):**
```bash
# For Ubuntu
sudo apt install docker.io docker-compose

# For RHEL (Podman is pre-installed)
sudo dnf install podman podman-compose

# Verify
docker --version  # or podman --version
docker-compose --version  # or podman-compose --version
```

#### 9.6.1.5 Complete Tier 1 Environment Verification

**System Verification Script:**
```bash
#!/bin/bash
# File: scripts/verify_tier1_setup.sh

echo "=== BigBrotherAnalytics Tier 1 Setup Verification ==="

# Check GCC
echo -n "GCC C++23: "
gcc-15 --version | head -1

# Check OpenMP
echo -n "OpenMP: "
echo '#include <omp.h>
int main() { return omp_get_max_threads(); }' | gcc-15 -fopenmp -x c++ - -o /tmp/test_omp && /tmp/test_omp && echo "✓ Working" || echo "✗ Failed"

# Check MPI
echo -n "OpenMPI: "
mpirun --version | head -1

# Check UPC++
echo -n "UPC++: "
upcxx --version

# Check Python
echo -n "Python: "
~/.cargo/bin/uv python list | grep "3.14"

# Check CUDA
echo -n "CUDA: "
nvcc --version | grep release

# Check PyTorch CUDA
echo -n "PyTorch CUDA: "
source .venv/bin/activate && python -c "import torch; print('✓ Available' if torch.cuda.is_available() else '✗ Not available')"

# Check PostgreSQL
echo -n "PostgreSQL: "
psql --version

# Check Redis
echo -n "Redis: "
redis-cli --version

# Check DuckDB
echo -n "DuckDB: "
source .venv/bin/activate && python -c "import duckdb; print(f'✓ {duckdb.__version__}')"

echo ""
echo "=== Verification Complete ==="
```

#### 9.6.1.6 Tier 1 Development Workflow

**Daily Development Cycle:**
```bash
# 1. Activate environment
cd /opt/bigbrother
source .venv/bin/activate

# 2. Pull latest changes
git pull origin main

# 3. Install/update dependencies (if requirements.txt changed)
uv pip install -r requirements.txt

# 4. Compile C++ components
cd src/cpp
cmake -B build -G Ninja \
    -DCMAKE_CXX_COMPILER=g++-15 \
    -DCMAKE_CXX_STANDARD=23 \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_OPENMP=ON \
    -DENABLE_MPI=ON \
    -DENABLE_CUDA=ON
cmake --build build -j $(nproc)

# 5. Run tests
cd ../..
pytest tests/

# 6. Start development services
docker-compose up -d postgres redis

# 7. Run data collectors
python scripts/collect_free_data.py

# 8. Start Jupyter for exploration
jupyter lab --no-browser

# 9. Profile performance
perf record -g python scripts/backtest.py
perf report
```

#### 9.6.1.7 Tier 1 Resource Requirements

**Minimum Hardware:**
- **CPU:** 8+ cores
- **RAM:** 16GB
- **Storage:** 500GB SSD
- **GPU:** Optional (any NVIDIA GPU)

**Recommended Hardware:**
- **CPU:** 16-32 cores (AMD Ryzen 9 / Intel i9 / Xeon)
- **RAM:** 32-64GB
- **Storage:** 1TB NVMe SSD
- **GPU:** NVIDIA RTX 3090/4090 (24GB VRAM)

**Cost Estimate:**
- **DIY Workstation:** $2,000-5,000 (one-time)
- **Used Server:** $1,000-3,000 (eBay/Craigslist)
- **Software:** $0 (all open-source)

**Monthly Operational Costs (Tier 1):**
- Data subscriptions: $0 (free tiers only)
- Electricity: ~$50-100/month
- Internet: Included
- **Total: $50-100/month**

#### 9.6.1.8 eBay Hardware Search Guide (64+ Core Enterprise Servers)

**CRITICAL:** For production-like Tier 1 testing with massive parallelization (MPI across 64+ cores), purchase used enterprise servers from eBay at 80-90% discount vs. new.

**Target Specifications (64+ Cores):**
- **CPU:** Dual-socket with 32+ cores per socket = 64+ total cores
- **RAM:** 256GB+ ECC DDR4
- **Storage:** NVMe SSDs in RAID configuration
- **GPU:** NVIDIA Tesla/Quadro/RTX (24-48GB VRAM)
- **Network:** 10GbE or better
- **Total Cost:** $3,000-7,000 (vs. $20,000-40,000 new)

**eBay Search Terms (Copy-Paste Ready):**

**Budget 64-Core Systems ($2,500-4,000):**
```
Search: "dual epyc 7551 server"
- 2x AMD EPYC 7551 (32 cores each) = 64 cores, 128 threads
- Typical: Supermicro, Dell, HP
- Price Range: $2,000-3,500
- Add GPU: Tesla P40 24GB ($400-600)
```

**Intel Xeon Option ($2,000-3,500):**
```
Search: "dual xeon platinum 8168 server"
- 2x Intel Xeon Platinum 8168 (24 cores each) = 48 cores, 96 threads
- Typical: Dell R740, HP DL380 Gen10
- Price Range: $2,500-4,000
- Add GPU: RTX A4000 16GB ($800-1,200)
```

**Ultra High-Core ($5,000-8,000):**
```
Search: "dual epyc 7742 server"
- 2x AMD EPYC 7742 (64 cores each) = 128 cores, 256 threads
- Near-linear MPI scaling
- Price Range: $5,000-7,000
- Add GPU: RTX 4090 24GB or A40 48GB
```

**Pre-Built eBay Searches:**

| Configuration | eBay Search URL | Cores | Typical Price |
|---------------|----------------|-------|---------------|
| Budget 64-core | [dual epyc 7551 server](https://www.ebay.com/sch/i.html?_nkw=dual+epyc+7551+server) | 64c/128t | $2K-3.5K |
| Intel 48-core | [dual xeon platinum 8168](https://www.ebay.com/sch/i.html?_nkw=dual+xeon+platinum+8168) | 48c/96t | $2.5K-4K |
| High 80-core | [dual epyc 7601 server](https://www.ebay.com/sch/i.html?_nkw=dual+epyc+7601+server) | 64c/128t | $3K-5K |
| Ultra 128-core | [dual epyc 7742 server](https://www.ebay.com/sch/i.html?_nkw=dual+epyc+7742+server) | 128c/256t | $5K-7K |
| GPU Servers | [server nvidia tesla p40](https://www.ebay.com/sch/i.html?_nkw=server+nvidia+tesla+p40) | Varies | +$400-600 |

**eBay Buying Checklist:**

✅ **Verify Before Purchase:**
- [ ] Exact CPU model (check core count on ark.intel.com or amd.com)
- [ ] RAM included (minimum 128GB, prefer 256GB+)
- [ ] PCIe x16 slots available for GPU
- [ ] Power supply sufficient (750W+ for GPU)
- [ ] Seller rating 98%+ with 100+ sales
- [ ] Includes rails/hardware if rack-mount
- [ ] BIOS unlocked (not locked to previous owner)
- [ ] RAID controller supports JBOD/HBA mode

✅ **Questions for Sellers:**
- "CPUs and RAM included as listed?"
- "BIOS password removed?"
- "Power-on hours?"
- "All PCIe slots functional?"
- "Successfully POSTs and boots?"
- "Why selling?" (look for datacenter decommissions)

**Recommended eBay Configurations:**

**Option 1: Budget 64-Core Powerhouse ($3,000-4,500)**
```
Base Server: Dual AMD EPYC 7551 (64 cores, 128 threads)
- Search: "dual epyc 7551 server" or "supermicro epyc"
- Server cost: $2,500-3,500
- RAM: 256GB ECC DDR4 (usually included)
- Storage: 4x 960GB SSD
- GPU: Add NVIDIA Tesla P40 24GB (~$500)
- Network: 10GbE (usually included)

Total Cost: ~$3,000-4,000
Performance: Excellent for MPI/OpenMP workloads
Power Draw: 400-600W under load
```

**Option 2: Intel Xeon Gold/Platinum ($3,500-5,000)**
```
Base Server: Dual Intel Xeon (40-56 cores total)
- Search: "dual xeon gold 6248 server"
- Server cost: $3,000-4,500
- RAM: 384GB ECC DDR4
- Storage: 2x 1.92TB NVMe
- GPU: NVIDIA RTX A4000 16GB (~$1,000)
- Models: Dell R740, HP DL380 Gen10, Supermicro

Total Cost: ~$4,000-5,500
Performance: Excellent single-thread + parallel
```

**Option 3: Ultra 128-Core Beast ($6,000-9,000)**
```
Base Server: Dual AMD EPYC 7742 (128 cores, 256 threads)
- Search: "dual epyc 7742 server"
- Server cost: $5,000-6,500
- RAM: 512GB+ ECC DDR4
- Storage: 8x 1.92TB NVMe RAID
- GPU: NVIDIA RTX 4090 24GB or A40 48GB
- Near-linear scaling to 256 threads

Total Cost: ~$7,000-9,000
Performance: Production-equivalent
Best for: Validating before full deployment
```

**Additional Costs:**
- Shipping: $100-250 (servers are 40-80 lbs)
- Rack/Mounting: $200-500 (if needed)
- 10GbE Switch: $150-400
- PDU/Power cables: $50-150
- Extra RAM: $200-1,000 (optional upgrade)

**Total Tier 1 Hardware Investment (64-Core System):**
```
Server (64+ cores):       $2,500-6,500
GPU (NVIDIA):            $500-2,000
Shipping & Accessories:  $300-800
Network Equipment:       $150-400
───────────────────────────────────
Total (one-time):        $3,450-9,700

vs. New Enterprise:      $20,000-40,000
Savings:                 $16,000-30,000 (80-85%)
```

**Monthly Operational Costs (64-Core Server):**
```
Electricity (dual-socket server):  $150-250/month
  - Idle: 200-300W
  - Full load: 500-700W
  - 24/7 operation @ $0.12/kWh
Data subscriptions (Tier 1):      $0/month (free APIs)
RHEL subscription (optional):     $30/month
Internet (1Gb+ recommended):      Included
────────────────────────────────────────────
Total Monthly:                    $150-280/month
```

**ROI Analysis (64-Core eBay Server vs. Cloud):**
```
Hardware Cost:           $4,000 (one-time)
Monthly Operating:       $200/month

AWS c6i.32xlarge Equivalent:
- 128 vCPUs (64 physical cores equivalent)
- 256GB RAM
- Cost: $5,440/month

Break-even: 0.7 months (22 days)
Annual Savings: $60,280
3-Year Savings: $180,840
```

**Why eBay Enterprise Servers are Perfect for Tier 1:**
1. **Massive Parallelization**: Test MPI across 64-128 real cores
2. **ECC Memory**: Detect memory errors during long computations
3. **Enterprise Components**: Same hardware as production
4. **Multiple PCIe Slots**: Add 2-4 GPUs if needed
5. **10GbE+**: Fast data transfers between storage and compute
6. **Proven Reliability**: These ran in datacenters for years
7. **Easy Upgrades**: Standard components, readily available
8. **80-90% Discount**: Get $20K hardware for $3-7K

**Recommended eBay Sellers (High Volume, Good Ratings):**
- SaveMyServer
- TechMikeNY
- NetworkTigers
- Aventis Systems
- Look for: 99%+ feedback, 1000+ sales

---

### 9.7 Brokerage & Execution (Low-Latency APIs)

**Options Trading APIs (Priority):**
- **Interactive Brokers (IBKR) API:** Options trading with low latency
- **Tastytrade API:** Options-focused trading platform
- **TradeStation API:** Options and futures
- **TD Ameritrade (thinkorswim) API:** Options trading
- **Other institutional brokers with FIX protocol support**

**Execution Requirements:**
- **Direct Market Access (DMA):** Minimize routing latency
- **Co-location consideration:** For future ultra-low latency needs
- **Custom C++ FIX engine:** Ultra-fast order routing
- **Redundant connections:** Failover and reliability

**Order Management:**
- **Custom OMS in C++:** Ultra-low latency order management
- **Position tracking in shared memory:** Real-time updates
- **Risk checks in microseconds:** Pre-trade validation

### 9.8 Performance Optimization Tools

**Profiling & Optimization:**
- **perf:** Linux performance analysis
- **Intel VTune Profiler:** CPU performance analysis
- **NVIDIA Nsight:** GPU performance analysis
- **Valgrind:** Memory profiling
- **gperftools:** Google performance tools

**Benchmarking:**
- **Google Benchmark:** C++ microbenchmarking
- **Criterion:** Rust benchmarking
- **Custom latency measurement:** Nanosecond-level timing

---

## 9.9 Cloud Deployment (Deferred)

**Rationale for Private Deployment First:**
- **Security:** Sensitive trading algorithms and data
- **Cost:** Cloud costs prohibitive for 24/7 high-performance computing
- **Performance:** Direct hardware access, no virtualization overhead
- **Control:** Full control over infrastructure and networking

**Future Cloud Considerations:**
- AWS EC2 (c6i/c7i instances with local NVMe)
- Google Cloud (n2/c2 instances)
- Azure (HBv3/HBv4 HPC instances)
- Bare metal cloud providers (for ultimate performance)

---

## 9.10 Distributed Architecture & Fault Tolerance

**CRITICAL:** The system must be resilient to failures and support multi-site deployment for high availability.

### 9.10.1 Multi-Server Architecture

**Initial Deployment:**
- Single high-performance server (32+ cores) for development and initial trading
- All components co-located for minimal latency

**Future Multi-Server Deployment:**
- Multiple geographically distributed servers
- Active-active or active-passive configurations
- Automatic failover between sites
- Data replication and consistency

### 9.10.2 Fault Tolerance Requirements

**Component-Level Resilience:**
- **Process Monitoring:** Automatic restart of failed processes
- **Health Checks:** Continuous monitoring of all services
- **Graceful Degradation:** System continues with reduced functionality if non-critical components fail
- **Circuit Breakers:** Prevent cascading failures

**Data-Level Resilience:**
- **Real-time Replication:** Market data replicated across servers with < 10ms delay
- **Database Replication:** PostgreSQL/TimescaleDB with synchronous/asynchronous replication
- **State Synchronization:** Trading positions and state replicated in real-time
- **Data Consistency:** Raft consensus or similar for critical state

**Network-Level Resilience:**
- **Redundant Data Feeds:** Multiple market data providers with automatic failover
- **Redundant Broker Connections:** Multiple broker connections for order routing
- **Network Partitioning Handling:** Split-brain prevention and resolution
- **Heartbeat Monitoring:** Detect network failures within 100ms

### 9.10.3 Site Failover Strategy

**Automatic Failover Scenarios:**
1. **Server Failure:** Backup server takes over within 5 seconds
2. **Network Partition:** Sites operate independently, reconcile on recovery
3. **Data Feed Failure:** Switch to backup feed within 100ms
4. **Broker Connection Failure:** Reroute orders through backup broker

**Failover Components:**
- **Leader Election:** Raft or ZooKeeper for distributed coordination
- **State Transfer:** Snapshot and replay mechanism for position synchronization
- **Order Reconciliation:** Match and reconcile orders across sites
- **Trade De-duplication:** Prevent duplicate orders during failover

**Recovery Time Objectives (RTO):**
- Critical trading components: RTO < 5 seconds
- Market data processing: RTO < 1 second
- Historical data access: RTO < 30 seconds
- Reporting and analytics: RTO < 5 minutes

**Recovery Point Objectives (RPO):**
- Trade execution state: RPO = 0 (no data loss)
- Market data: RPO < 1 second
- Historical analysis data: RPO < 1 minute

### 9.10.4 Distributed Coordination

**Technology Options:**
- **Raft Consensus:** For strongly consistent state (etcd, Consul)
- **ZooKeeper:** Distributed coordination and configuration
- **Hazelcast/Redis Cluster:** Distributed caching and state
- **MPI Communicators:** For parallel computation coordination

**Coordination Requirements:**
- **Configuration Management:** Centralized configuration distributed to all sites
- **Service Discovery:** Automatic discovery of services across sites
- **Distributed Locking:** Prevent conflicting trades across sites
- **Consensus on Trades:** Ensure single execution even with multiple sites

### 9.10.5 Testing Fault Tolerance

**Chaos Engineering:**
- Regular testing of failover scenarios
- Random process kills (chaos monkey)
- Network partition simulations
- Latency injection tests
- Load testing under degraded conditions

**Disaster Recovery Drills:**
- Monthly DR drills to validate procedures
- Automated testing of backup systems
- Verification of RTO/RPO targets
- Documentation of lessons learned

### 9.10.6 Monitoring & Alerting for Resilience

**Critical Metrics:**
- Site health status (up/down)
- Replication lag between sites
- Failover event frequency
- Data consistency checks
- Network partition detection

**Alerting:**
- PagerDuty/OpsGenie for critical alerts
- Automated escalation procedures
- SMS/phone alerts for trading disruptions
- Dashboard for real-time system health

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data provider outages | Medium | High | Multiple redundant providers, cached data |
| Model overfitting | High | High | Rigorous validation, walk-forward testing, regularization |
| System latency issues | Medium | High | Performance optimization, hardware acceleration |
| Scalability bottlenecks | Medium | Medium | Load testing, horizontal scaling design |
| Security breaches | Low | Very High | Security audits, encryption, access controls |
| Integration failures | Medium | Medium | Comprehensive testing, API versioning |

### 10.2 Financial Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model underperformance | Medium | High | Conservative position sizing, diversification |
| Market regime changes | High | High | Adaptive models, regime detection |
| Black swan events | Low | Very High | Portfolio hedging, circuit breakers, stop-losses |
| Execution slippage | Medium | Medium | Smart order routing, impact modeling |
| Broker issues | Low | High | Multiple broker relationships |
| Regulatory changes | Medium | Medium | Compliance monitoring, flexible architecture |

### 10.3 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Human error | Medium | Medium | Automation, validation checks, review processes |
| Infrastructure failures | Low | High | Redundancy, failover systems, backups |
| Data quality issues | Medium | High | Validation pipelines, monitoring, alerts |
| Team expertise gaps | Medium | Medium | Training, hiring, documentation |
| Vendor dependency | Medium | Medium | Multiple vendors, contractual protections |

### 10.4 Regulatory & Compliance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Algorithmic trading regulations | Medium | High | Legal consultation, compliance features |
| Market manipulation concerns | Low | Very High | Ethical guidelines, audit trails, transparency |
| Data privacy violations | Low | High | Privacy policies, data governance |
| Reporting requirements | Medium | Medium | Automated reporting, compliance systems |

---

## 11. Development Phases & Milestones

### Phase 1: Foundation (Months 1-3)
- **Milestone 1.1:** Requirements finalization and architecture design
- **Milestone 1.2:** Development environment setup
- **Milestone 1.3:** Data provider evaluation and contracts
- **Milestone 1.4:** Core data infrastructure (databases, pipelines)
- **Milestone 1.5:** Initial data collection and storage

### Phase 2: Market Intelligence Engine (Months 3-8)
- **Milestone 2.1:** Data ingestion pipelines for all sources
- **Milestone 2.2:** NLP models for news and sentiment analysis
- **Milestone 2.3:** Impact prediction models (initial version)
- **Milestone 2.4:** Impact graph generation system
- **Milestone 2.5:** API development and testing
- **Milestone 2.6:** Historical backtesting of predictions

### Phase 3: Correlation Analysis Tool (Months 6-10)
- **Milestone 3.1:** Historical data acquisition (10+ years)
- **Milestone 3.2:** Correlation computation engine
- **Milestone 3.3:** Time-lagged analysis implementation
- **Milestone 3.4:** Pattern recognition algorithms
- **Milestone 3.5:** Visualization and reporting tools
- **Milestone 3.6:** API development and integration

### Phase 4: Trading Decision Engine (Months 9-15)
- **Milestone 4.1:** Signal integration framework
- **Milestone 4.2:** Day trading strategy development and backtesting
- **Milestone 4.3:** Short-term strategy development and backtesting
- **Milestone 4.4:** Long-term strategy framework
- **Milestone 4.5:** Position sizing and risk management
- **Milestone 4.6:** Broker integration and order execution
- **Milestone 4.7:** Portfolio management system
- **Milestone 4.8:** Paper trading validation (3 months minimum)

### Phase 5: Integration & Testing (Months 15-18)
- **Milestone 5.1:** End-to-end integration testing
- **Milestone 5.2:** Performance optimization
- **Milestone 5.3:** Security audits
- **Milestone 5.4:** Load and stress testing
- **Milestone 5.5:** User acceptance testing
- **Milestone 5.6:** Documentation completion

### Phase 6: Deployment & Monitoring (Month 18+)
- **Milestone 6.1:** Production deployment (limited capital)
- **Milestone 6.2:** Live trading with small positions
- **Milestone 6.3:** Performance monitoring and validation
- **Milestone 6.4:** Gradual capital increase
- **Milestone 6.5:** Continuous improvement and optimization

---

## 12. Success Metrics & KPIs

### 12.1 Market Intelligence Engine Metrics

- **Accuracy Metrics:**
  - Impact prediction accuracy (direction): Target > 70%
  - Impact magnitude MAPE: Target < 20%
  - Sentiment analysis accuracy: Target > 85%
  - Entity recognition F1 score: Target > 0.90

- **Performance Metrics:**
  - Data processing latency: Target < 1 second
  - Prediction generation time: Target < 5 seconds
  - System uptime: Target > 99.5%

- **Coverage Metrics:**
  - News articles processed per day: Target > 50,000
  - Companies covered: Target > 5,000
  - Data source reliability: Target > 98%

### 12.2 Correlation Analysis Tool Metrics

- **Accuracy Metrics:**
  - Correlation coefficient precision: 4 decimal places
  - Statistical significance: p < 0.05 for reported correlations
  - Lead-lag identification accuracy: Target > 65%

- **Performance Metrics:**
  - Correlation computation time (5K securities): Target < 60 seconds
  - Real-time correlation updates: Target < 5 seconds
  - System uptime: Target > 99%

- **Coverage Metrics:**
  - Securities analyzed: Target > 5,000
  - Historical data depth: Target > 10 years
  - Correlation pairs tracked: Target > 1 million

### 12.3 Trading Decision Engine Metrics

- **Financial Performance:**
  - **Primary KPI:** Sharpe Ratio > 1.5
  - Annual return vs. S&P 500: Target alpha > 5%
  - Maximum drawdown: Target < 15%
  - Win rate: Target > 55%
  - Profit factor: Target > 1.5

- **Strategy-Specific Metrics:**
  - **Day Trading:**
    - Win rate: Target > 60%
    - Average profit per trade: Target > 0.5%
    - Daily Sharpe ratio: Target > 2.0
  - **Short-Term Trading:**
    - Win rate: Target > 50%
    - Average profit per trade: Target > 10%
    - Holding period return: Target > 15% annualized
  - **Long-Term Investing:**
    - Win rate: Target > 60%
    - Average holding period return: Target > 100%
    - Annual return: Target > 15%

- **Risk Metrics:**
  - Value at Risk (VaR) 95%: Monitor and limit
  - Conditional VaR (CVaR): Monitor and limit
  - Position concentration: Max 15% per position
  - Sector concentration: Max 30% per sector

- **Operational Metrics:**
  - Order fill rate: Target > 99%
  - Average slippage: Target < 0.05%
  - System latency (signal to order): Target < 200ms
  - System uptime during market hours: Target > 99.9%

### 12.4 Overall Platform Metrics

- **User Satisfaction:**
  - System reliability score: Target > 4.5/5
  - Performance vs. expectations: Target > 4/5
  - Feature completeness: Target > 4/5

- **Development Metrics:**
  - Code test coverage: Target > 80%
  - Deployment frequency: Target weekly
  - Mean time to recovery (MTTR): Target < 30 minutes
  - Change failure rate: Target < 5%

---

## 13. Dependencies & Assumptions

### 13.1 External Dependencies

- **Data Providers:**
  - Assumption: Reliable and affordable data feeds available
  - Risk: Provider outages, price increases, contract terminations

- **Broker APIs:**
  - Assumption: Stable APIs with sufficient throughput
  - Risk: API changes, execution quality degradation

- **Cloud Infrastructure:**
  - Assumption: AWS/GCP/Azure availability and performance
  - Risk: Outages, cost increases, vendor lock-in

- **Third-Party Libraries:**
  - Assumption: Maintained open-source libraries
  - Risk: Security vulnerabilities, breaking changes, abandonment

### 13.2 Internal Dependencies

- **Team Expertise:**
  - Required: ML/AI engineers, quantitative developers, financial experts
  - Assumption: Team can be hired or trained

- **Capital:**
  - Required: Development costs, data costs, trading capital
  - Assumption: Sufficient funding available

- **Time:**
  - Assumption: 18-month development timeline is acceptable
  - Risk: Market conditions may change during development

### 13.3 Regulatory Assumptions

- Algorithmic trading is permitted in target markets
- Data usage complies with provider terms and regulations
- System design meets regulatory requirements (if applicable)
- Changes in regulations won't require complete redesign

### 13.4 Market Assumptions

- Historical patterns have some predictive power for future
- Market efficiency is not perfect; opportunities exist
- Transaction costs don't eliminate all edge
- Sufficient liquidity in target securities
- No major market structure changes during development

---

## 14. Open Questions & Decisions Needed

### 14.1 Strategic Questions

1. **Target Market:** Which markets will we trade? (US only vs. global)
2. **Asset Classes:** Equities only or also ETFs, options, futures, forex?
3. **Capital Scale:** What is the target AUM for the system?
4. **Regulatory Status:** Will this be registered as an investment advisor?
5. **Business Model:** Internal use, licensing, fund management, or SaaS?

### 14.2 Technical Questions

1. **Cloud Provider:** AWS, GCP, Azure, or multi-cloud?
2. **Programming Language:** Python for all components or mixed?
3. **Real-Time vs. Batch:** What percentage of processing is real-time?
4. **Model Retraining:** How frequently should models be retrained?
5. **Hardware Acceleration:** Do we need GPUs or TPUs for inference?

### 14.3 Data Questions

1. **Data Vendors:** Which specific providers will we contract with?
2. **Data Retention:** How long do we keep raw vs. processed data?
3. **Alternative Data:** Which alternative data sources are worth the cost?
4. **Data Quality:** What is acceptable data quality threshold?
5. **Historical Depth:** How much historical data is truly necessary?

### 14.4 Risk Questions

1. **Risk Tolerance:** What is the acceptable maximum drawdown?
2. **Leverage:** Will we use leverage, and to what extent?
3. **Concentration Limits:** What are reasonable position/sector limits?
4. **Stop-Loss Strategy:** Tight stops vs. allowing room for volatility?
5. **Hedging:** Should we implement portfolio hedging strategies?

### 14.5 Operational Questions

1. **Team Size:** How many people needed for development and operations?
2. **24/7 Monitoring:** Do we need round-the-clock system monitoring?
3. **Disaster Recovery:** What is acceptable RTO and RPO?
4. **Testing:** How long should paper trading validation last?
5. **Human Oversight:** What level of human review is required for trades?

---

## 15. Next Steps

### 15.1 Immediate Actions

1. **Review and Feedback:** Stakeholders review this PRD and provide feedback
2. **Prioritization:** Rank features by importance and feasibility
3. **Team Formation:** Identify required roles and begin hiring/contracting
4. **Vendor Evaluation:** Evaluate and select data providers and brokers
5. **Proof of Concept:** Build small-scale PoC for critical components

### 15.2 Short-Term (Next 30 Days)

1. Finalize PRD based on feedback
2. Create detailed technical architecture document
3. Set up development environment and repositories
4. Sign contracts with initial data providers
5. Begin Phase 1 development (Foundation)

### 15.3 Medium-Term (Next 90 Days)

1. Complete data infrastructure setup
2. Begin data collection and storage
3. Prototype first ML models for Market Intelligence Engine
4. Conduct feasibility studies for correlation analysis
5. Set up CI/CD pipelines and monitoring

---

## 16. Document Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | Initial draft |
| 0.4.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | Major update: Added comprehensive data extraction technologies for all government departments and sectors (Congress, Treasury, USDA, FDA, EPA, HHS, etc.). Unified technology stack with PostgreSQL-centric architecture (TimescaleDB, pgvector, Apache AGE). Consolidated expensive tools into affordable open-source alternatives. Added detailed cost estimates ($250-1K/month vs $25K+/month). Removed Bloomberg Terminal and other expensive dependencies. |
| 0.5.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **Critical Addition: Zero-Fee Rapid Deployment Strategy.** Added DuckDB as embedded analytics database for instant prototyping without infrastructure setup. Created comprehensive "Quick Start" section with zero-subscription deployment path using only free data sources (Yahoo Finance, Alpha Vantage free tier, FRED, SEC, all government APIs). Added phased cost structure: $0/month Months 1-2 (validation), $200/month Month 3 (if validated), $250-1K Month 4+ (if profitable). Updated cost comparison to show three tiers: Enterprise ($300K/year), Production ($3-12K/year), Zero-Fee ($0/year initial). Includes complete code examples for DuckDB usage, Parquet file queries, and correlation analysis. Emphasizes validate-then-scale approach to minimize financial risk. |
| 0.6.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **Major Addition: Complete Architecture Documents for Both Core Tools.** Created comprehensive architecture design documents for Market Intelligence Engine and Trading Correlation Analysis Tool with 6,000+ lines of technical specifications. Added Tier 1 deployment stack with Homebrew (GCC 15+), uv (Python 3.14+), Ansible automation, OpenShift/Kubernetes support. Documented C++23/MPI/OpenMP/UPC++ implementations with fluent composable APIs. Added CUDA/cuBLAS/Intel MKL acceleration strategies. Created eBay hardware procurement guide for 64-128 core enterprise servers ($3-9K vs $20-40K new, 80-90% savings). Documented shared infrastructure design (both tools on same server, $3.5-9.7K total vs $7-19K separate). Added loose coupling architecture with API-only communication. Created 2-4 week POC guides using Python/DuckDB/CUDA before C++23 production implementation. Includes 20+ Mermaid diagrams, OpenAPI 3.0 specs, complete database schemas, and ready-to-execute code examples. Emphasizes consistent toolset across tools for maintainability. Total documentation: Market Intelligence (1,900 lines), Correlation Tool (3,300 lines). |
| 0.7.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **Complete Platform Architecture: Added Trading Decision Engine.** Created comprehensive architecture design document (3,900+ lines) for Intelligent Trading Decision Engine completing the three-pillar platform. **Key Feature: 100% Explainable AI - NOT a black box.** Every trading decision includes multi-level explanations (SHAP, attention visualization, decision trees, natural language summaries). Added mathematical foundations: Reinforcement Learning (PPO/DQN/A3C), Deep Neural Networks (Transformers/LSTMs), Graph Neural Networks for impact propagation. Documented complete explainability framework with 5 explanation levels from executive summary to full analysis. Added low-cost historical data collection (Yahoo Finance, FRED, SEC EDGAR - all free). Comprehensive charting stack: Lightweight Charts (TradingView-like), Plotly/Dash dashboards, Streamlit prototyping (all free, $0 cost). C++23 ultra-low latency order execution (< 10ms). Shared infrastructure design: all three tools on single 64-core server (cores 0-21: MI, 22-42: Correlation, 43-63: Trading). Portfolio optimization with Markowitz framework and Kelly Criterion. Message format with JSON+zstd compression (consistent across all tools). 4-6 week POC guide with simple rules → XGBoost+SHAP → RL+DNN progression. Total platform documentation: 13,300+ lines across 4 documents (PRD + 3 architecture docs). Zero additional hardware cost - complete platform on single eBay server ($3.9-9.3K). Emphasizes explainability, interpretability, and human understanding of all AI decisions. |
| 0.8.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **Systems Integration & Complete Platform Workflows.** Created comprehensive Systems Integration Architecture document (5,500+ lines) showing how all three tools work together. **Key Features: Causal chain tracking (Fed decision → rates → bank stocks), correlation chain prediction (NVDA ↑ → AMD follows with lag), sector-level analysis (track entire business sectors).** Added complete usage scenarios for budgets from $1K to $1M+ with detailed workflows. Self-simulation framework: simulate all trades before execution, only trade when profitable (Monte Carlo 1,000-10,000 scenarios). Profit explanation framework: analyze why trades were profitable with attribution analysis. Mistake analysis & learning loop: classify mistakes, identify patterns, update models, validate improvements. Daily operation workflow: pre-market → execution → monitoring → analysis → learning. Budget-driven strategy selection: automatic allocation based on capital size. Tier 1/2/3 deployment specifications with Tier 1 focus (existing computer, $0 cost). Complete hardware/software architecture diagrams. Integration communication patterns showing loose coupling. Weekly learning cycle with backtest validation. Total documentation: 18,830+ lines across 5 documents (PRD + 4 architecture docs). Platform is self-improving, self-explaining, and budget-aware. All integration patterns preserve explainability and enable continuous learning from both successes and mistakes. |
| 0.9.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **Berkeley PGAS Components & Validation Framework.** Added complete UPC++ and Berkeley Distributed Components installation via Ansible playbook (based on ClusterSetupAndConfigs repository). Installs GASNet-EX 2024.5.0, UPC++ 2024.3.0, Berkeley UPC, and OpenSHMEM with automated configuration. Created playbooks/install-upcxx-berkeley.yml with verification and testing. Added playbooks/README.md documenting all playbooks and PGAS installation. Updated PRD and all architecture documents with Berkeley components references and ClusterSetupAndConfigs links. **Added comprehensive backtesting & validation framework** to systems integration: 7-level validation (unit tests → integration → historical backtest → walk-forward → Monte Carlo → paper trading → limited live). DuckDB-based backtesting processes 10 years of data in seconds. Walk-forward validation prevents overfitting. Complete validation checklist with must-pass criteria (Sharpe > 1.5, win rate > 55%, max DD < 20%). Only deploy real money after ALL validation levels pass. Emphasizes rigorous testing before production. References ClusterSetupAndConfigs repo for complete cluster management and PGAS deployment details. Total documentation: 19,400+ lines (added 570 lines of validation/PGAS content). |
| 1.0.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **PLANNING PHASE COMPLETE - All-in-One Installation.** Created comprehensive Ansible playbook (complete-tier1-setup.yml, 500+ lines) that installs ENTIRE Tier 1 environment in one command. Single playbook installs: Homebrew, GCC 15 (C++23), Python 3.14+, uv package manager, CMake 4.1.2+, Ninja, OpenMP, OpenMPI 5.x, UPC++ 2024.3.0, GASNet-EX 2024.5.0, Intel MKL, NVIDIA CUDA 12.3 (auto-detects GPU), PostgreSQL 16 with TimescaleDB/AGE/pgvector extensions, Redis, DuckDB, PyTorch with CUDA, Hugging Face Transformers, Stable-Baselines3, XGBoost, SHAP, spaCy, all data collection tools (yfinance, FRED API, Scrapy, Playwright), visualization tools (Plotly, Streamlit, Dash), monitoring (Prometheus, Sentry), complete environment configuration, verification script, and QUICKSTART.md guide. Installation time: 2-4 hours. Cost: $0 (100% open-source). **PLANNING PHASE COMPLETE:** All documentation (20,000+ lines), all architecture documents (5 docs), all Ansible playbooks (3 playbooks), complete technology stack specified, ready for Tier 1 POC implementation. Platform status: Design complete, implementation ready, zero initial investment required. |
| 1.1.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **Final Updates: CUDA 13.0, uv Standardization, Uninstall Capability.** Updated all playbooks and documentation to use latest NVIDIA CUDA Toolkit 13.0 (was 12.3). Standardized ALL Python package management to use uv (NOT pip) throughout playbooks and documentation - emphasizes 10-100x speed improvement. Fixed complete-tier1-setup.yml for cross-platform compatibility (Ubuntu, RHEL, WSL2) with OS-specific package names, paths, and service management. Added systemd detection for WSL compatibility (graceful fallback to manual service start). Created uninstall-tier1.yml playbook (200+ lines) for clean removal with data backup, optional Homebrew/data preservation. Added idempotency documentation - playbook safe to re-run, won't duplicate installations. Updated playbooks/README.md with uninstall instructions, idempotency notes, and current technology versions. Updated PRD Section 9.6.1.3 with CUDA 13.0 references and uv emphasis. Clarified installation strategy: Homebrew for toolchain (GCC, Python, CMake, MPI), uv for ALL Python packages, system package managers only for OS-specific components (PostgreSQL, Redis). Complete platform ready for one-command installation on Ubuntu, RHEL, or WSL2 with ability to cleanly uninstall. |
| 1.2.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **Trading Instruments, Options Valuation, and Broker Selection.** Added comprehensive Section 5.2.4 to PRD specifying ALL supported trading instruments: stock order types (market, limit, stop, trailing stop, GTC, FOK, IOC), stock trade types (long, short, pairs, basket, DCA), money market instruments (MMF, T-Bills, commercial paper for cash management), and complete options strategies (basic calls/puts, vertical spreads, calendar spreads, diagonal spreads, straddles, strangles, iron condors, butterflies, delta-neutral, gamma scalping, theta harvesting, ratio spreads, synthetics). **Added complete options valuation methods with C++23 implementations:** Trinomial tree model (PRIMARY for American options) with full mathematical foundation, OpenMP parallelization, early exercise logic. Black-Scholes model (SECONDARY for European short-term <30 days) with analytic Greeks formulas using Intel MKL. Model selection logic (automatic choice based on option type and expiry). Parallel portfolio pricing with OpenMP. **Added Tier 1 POC broker evaluation:** Comprehensive comparison of 7 brokers (Tradier, IBKR, TD Ameritrade, Alpaca, Robinhood, E*TRADE, Schwab) with account minimums, commissions, API access, options support, and suitability ratings. **RECOMMENDED Tradier for POC:** $0 minimum, free API, full options support, free real-time data, $0.35/contract, unlimited paper trading. Complete Tradier setup guide for Tier 1 POC. Added to Trading Decision Engine architecture (Section 14b): Complete C++23 trinomial tree implementation (500+ lines), Black-Scholes with MKL integration, automatic model selection, portfolio-level parallel pricing. Mathematical formulas included for both methods. Platform now has complete options trading capability specification ready for implementation. Total new content: 700+ lines across PRD and architecture doc. |
| 1.3.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **Schwab API Integration & CMake 4.1.2 Update.** Created comprehensive Schwab API Integration architecture document (schwab-api-integration.md, 500+ lines) documenting C++23 implementation design for existing $30k Schwab account. **Updated broker from Tradier to Schwab** for Tier 1 POC due to existing account and SchwabFirstAPI integration. Added complete OAuth 2.0 authentication flow with token refresh strategies (proactive/reactive/hybrid). Documented secure credential storage options (environment variables, encrypted config, system keyring, HSM). C++23 module design for schwab.api.client with std::jthread for background token refresh, std::atomic for thread-safe tokens. Rate limiting with token bucket algorithm. HTTP client library evaluation (cpr RECOMMENDED). Complete data structures for options chains, orders, positions. Security implications for $30k real money documented. Implementation phases: OAuth → Market Data → Trading → Integration (4 weeks). **Updated ALL CMake references from 3.28 to 4.1.2** across all documentation and playbooks (better C++23 modules support). Updated playbooks/complete-tier1-setup.yml comments and README.md. Schwab integration references SchwabFirstAPI repository for OAuth patterns and credential management. Emphasizes this is DESIGN DOCUMENTATION (no code implemented yet). Added link to Schwab integration guide in PRD Section 5.2.4. Platform ready for C++23 Schwab API implementation with rigorous security and risk management for real money trading. Total documentation: 22,200+ lines (added 800+ lines). |
| 1.4.0 | 2025-11-06 | Olumuyiwa Oluwasanmi | **FINAL: Scenario Planning & DuckDB-First Decision.** Added hypothetical scenario planning capability (Section 13.5 in Systems Integration, 700+ lines): propose "what-if" scenarios, system generates complete trading strategies with full causal chain justification, success probabilities based on historical precedents, real-time monitoring for scenario occurrence, human-retrievable proposals via API. Examples: Fed emergency cut, China-Taiwan conflict, FDA drug approval/ban, unemployment spike. **Created Database Strategy Analysis document** (database-strategy-analysis.md, 600+ lines) comparing DuckDB-first vs PostgreSQL approaches with 10 detailed scenarios. **DECISION: DuckDB-First for Tier 1 POC** - rationale: 30-second setup vs 4-12 hours, 10x faster iteration, 3-5x faster backtesting, zero risk if POC fails, proven 1-2 day migration path when profitable. Tier 1 uses DuckDB only, Tier 2 adds PostgreSQL for dual-database architecture. Updated PRD Section 6 with scenario planning overview and database decision. Added scenario proposal database schema (DuckDB), API endpoints for scenario management, real-time monitoring implementation. Platform now supports predictive scenario analysis with complete audit trail. **PLANNING PHASE OFFICIALLY COMPLETE** - all decisions made, all architecture documented, ready for C++23 implementation. Total documentation: 23,500+ lines across 7 architecture documents. Status: FINAL, implementation can begin. |

---

## 17. Appendices

### Appendix A: Glossary

- **Alpha:** Risk-adjusted excess return compared to a benchmark
- **Sharpe Ratio:** Risk-adjusted return metric (return - risk-free rate) / standard deviation
- **Maximum Drawdown:** Largest peak-to-trough decline in portfolio value
- **Correlation Coefficient:** Measure of linear relationship between two variables (-1 to +1)
- **Lead-Lag Relationship:** When one security's movements predict another's with a time delay
- **Convolution:** Mathematical operation for analyzing time-lagged relationships
- **Impact Graph:** Network representation of how events affect companies
- **Granger Causality:** Statistical test for whether one time series predicts another
- **VaR (Value at Risk):** Potential loss at a given confidence level
- **VWAP:** Volume-Weighted Average Price

### Appendix B: References

- Relevant academic papers on quantitative trading
- Books on algorithmic trading and machine learning
- Industry reports on alternative data
- Regulatory guidance documents
- Technical documentation for proposed tools/libraries

### Appendix C: Stakeholder Contact Information

[To be populated]

### Appendix D: Related Documents

- Technical Architecture Document (to be created)
- API Specification Document (to be created)
- Data Privacy and Security Policy (to be created)
- Trading Operations Manual (to be created)

---

**End of Product Requirements Document**

---

## Approval & Sign-off

This document requires approval from the following stakeholders before proceeding to implementation:

- [ ] Product Owner
- [ ] Technical Lead
- [ ] Financial/Trading Expert
- [ ] Compliance Officer (if applicable)
- [ ] Executive Sponsor

**Approval Date:** _________________

**Approved By:** _________________
