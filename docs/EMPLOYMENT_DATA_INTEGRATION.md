# Employment Data Integration Guide

**Author:** Olumuyiwa Oluwasanmi
**Date:** 2025-11-08
**Version:** 1.0.0

**Tier 1 Extension: Department of Labor API & Sector Analysis**

This guide explains how to integrate employment statistics and sector analysis into BigBrotherAnalytics for enhanced trading decisions based on labor market indicators.

---

## Overview

Employment data serves as a critical leading and coincident indicator for:
- **Sector rotation signals** - Identify sectors transitioning from weak to strong
- **Recession detection** - Jobless claims spikes warn of economic downturns
- **Company-specific impacts** - Layoffs signal weakness, hiring signals strength
- **Defensive positioning** - Rotate to defensive sectors during labor market deterioration

---

## Components Added

### 1. Database Schema

**Location:** `scripts/database_schema_employment.sql`

**Tables:**
- `sectors` - 11 GICS sector definitions with ETF mappings
- `company_sectors` - Maps tickers to sectors
- `sector_employment` - BLS employment data by sector
- `employment_events` - Layoffs, hiring, freeze events
- `jobless_claims` - Weekly initial jobless claims (leading indicator)
- `monthly_jobs_report` - NFP (Nonfarm Payrolls) summary
- `sector_news_sentiment` - Sentiment analysis by sector
- `sector_performance` - Sector ETF performance and rotation signals

**To initialize:**
```bash
# Connect to DuckDB and run schema
duckdb data/bigbrother.duckdb < scripts/database_schema_employment.sql
```

### 2. BLS API Integration

**Location:** `scripts/data_collection/bls_employment.py`

**Features:**
- Fetches employment data from Bureau of Labor Statistics API
- Tracks 19 different employment series (total nonfarm, by industry, etc.)
- Collects weekly jobless claims (leading indicator)
- Stores data in DuckDB for analysis

**Usage:**
```bash
# Set BLS API key (optional but recommended)
export BLS_API_KEY="your_api_key_here"

# Get free API key at: https://data.bls.gov/registrationEngine/

# Run data collection
python scripts/data_collection/bls_employment.py
```

---

## Data Sources

### Bureau of Labor Statistics (BLS)

**Free API Access:**
- **Unauthenticated:** 25 queries/day
- **With API Key:** 500 queries/day (free registration)
- **Register:** https://data.bls.gov/registrationEngine/

**Key Series:**

1. **Current Employment Statistics (CES)** - Monthly employment by industry
   - Total nonfarm payrolls: `CES0000000001`
   - Manufacturing: `CES3000000001`
   - Technology/Information: `CES5000000001`
   - Financial activities: `CES5500000001`
   - Healthcare: `CES6500000001`

2. **Unemployment Insurance Claims** - Weekly jobless claims
   - Initial claims: `ICSA` (released Thursday 8:30 AM ET)
   - Continued claims: `CCSA`

3. **JOLTS (Job Openings and Labor Turnover Survey)** - Monthly
   - Job openings: `JTS00000000JOL`
   - Layoffs & discharges: `JTS00000000LDL`
   - Quits: `JTS00000000QUL`

### Private Sector Data

1. **Layoffs.fyi** - Tech sector layoffs (community-sourced)
   - Real-time layoff announcements
   - Company, date, employee count
   - https://layoffs.fyi/

2. **WARN Act Database** - Mass layoff notifications
   - State-level databases (60-90 day advance notice required)
   - Example: https://edd.ca.gov/jobs_and_training/Layoff_Services_WARN.htm

3. **Company Press Releases & News**
   - Parse hiring/layoff announcements
   - Track via NewsAPI, RSS feeds

---

## GICS Sector Classification

**11 Major Sectors:**

| Sector ID | Sector Name | ETF | Category | Key Industries |
|-----------|-------------|-----|----------|----------------|
| 1 | Energy | XLE | Cyclical | Oil, gas, energy services |
| 2 | Materials | XLB | Cyclical | Chemicals, metals, mining |
| 3 | Industrials | XLI | Sensitive | Aerospace, machinery, transport |
| 4 | Consumer Discretionary | XLY | Sensitive | Retail, autos, hotels |
| 5 | Consumer Staples | XLP | Defensive | Food, beverages, household |
| 6 | Health Care | XLV | Defensive | Pharma, biotech, devices |
| 7 | Financials | XLF | Sensitive | Banks, insurance, capital mkts |
| 8 | Information Technology | XLK | Sensitive | Software, hardware, semis |
| 9 | Communication Services | XLC | Sensitive | Media, entertainment, telecom |
| 10 | Utilities | XLU | Defensive | Electric, gas, water utilities |
| 11 | Real Estate | XLRE | Sensitive | REITs (all types) |

**Sector Categories:**
- **Defensive:** Stable earnings, less economic sensitivity (Utilities, Staples, Healthcare)
- **Cyclical:** Highly economic sensitive (Energy, Materials)
- **Sensitive:** Moderate sensitivity, growth-oriented (Tech, Financials, Industrials)

---

## Trading Signals from Employment Data

### 1. Leading Indicators (Predict Future Moves)

**Initial Jobless Claims (Weekly):**
- **Spike (>10% increase)** → Recession warning → Rotate to defensive sectors
- **Decline (<300K)** → Economic strength → Rotate to cyclical sectors
- **4-week average trending up** → Economic weakness building

**JOLTS Job Openings (Monthly):**
- **Declining openings** → Labor market cooling → Recession risk
- **Layoffs rising** → Corporate stress → Bearish signal
- **High quit rate** → Worker confidence → Bullish signal

### 2. Coincident Indicators (Current State)

**Nonfarm Payrolls (Monthly - First Friday):**
- **Major market moving event** (8:30 AM ET)
- **Beat expectations** → Stock market rally, rate hike concerns
- **Miss expectations** → Stock market drop, recession fears
- **Sector-specific data** → Rotate based on sector strength

**Unemployment Rate:**
- **Rising above 4.5%** → Recession concerns → Defensive positioning
- **Below 4.0%** → Strong economy → Cyclical positioning

### 3. Sector-Specific Signals

**Tech Layoffs:**
- **Example:** Meta, Google, Amazon announce 10K+ layoffs
- **Signal:** Short/avoid tech sector (XLK)
- **Pairs trade:** Long defensive (XLV, XLP, XLU) vs. Short tech (XLK)

**Healthcare Hiring:**
- **Signal:** Healthcare expansion → Long healthcare (XLV)
- **Thesis:** Aging population, healthcare demand growth

**Manufacturing Employment Decline:**
- **Signal:** Industrial weakness → Rotate out of industrials (XLI)
- **Alternative:** Rotate to services, technology

**Retail Hiring Surge (Holiday Season):**
- **Signal:** Strong consumer spending expected → Long consumer discretionary (XLY)
- **Timeframe:** October-November hiring for holiday season

---

## Implementation Roadmap

### Phase 1: Database Setup (Week 1)
- [x] Create database schema (`database_schema_employment.sql`)
- [ ] Run schema creation in DuckDB
- [ ] Verify tables and indexes created
- [ ] Seed 11 GICS sectors

### Phase 2: Data Collection (Week 2)
- [x] Implement BLS API client (`bls_employment.py`)
- [ ] Obtain BLS API key
- [ ] Collect 5 years of historical employment data
- [ ] Collect 1 year of weekly jobless claims
- [ ] Verify data quality and completeness

### Phase 3: Company-Sector Mapping (Week 3)
- [ ] Create mapping of stock tickers to GICS sectors
- [ ] Populate `company_sectors` table with current holdings
- [ ] Add sector classification to existing stock data
- [ ] Validate sector assignments

### Phase 4: Employment Event Tracking (Week 4)
- [ ] Implement Layoffs.fyi scraper/API integration
- [ ] Parse WARN Act database for mass layoffs
- [ ] Track company hiring announcements from press releases
- [ ] Store in `employment_events` table

### Phase 5: Sector Analysis Module (Week 5)
- [ ] Calculate sector-level employment trends
- [ ] Identify sectors with improving vs. declining employment
- [ ] Generate sector rotation signals
- [ ] Build sector sentiment analysis (news + employment)

### Phase 6: Decision Engine Integration (Week 6)
- [ ] Add employment signals to trading decision engine
- [ ] Implement sector-based filters
- [ ] Create sector rotation strategy
- [ ] Backtest sector rotation performance
- [ ] Validate profitability

---

## Example Queries

### Latest Employment by Sector
```sql
SELECT * FROM latest_sector_employment;
```

### Recent Layoff Events
```sql
SELECT sector_name, event_date, company_name, event_type, employee_count
FROM recent_employment_events
WHERE event_type = 'layoff'
ORDER BY event_date DESC
LIMIT 20;
```

### Employment Trends (3-month change)
```sql
SELECT * FROM sector_employment_trends
WHERE trend = 'Declining'
ORDER BY pct_change_3mo ASC;
```

### Jobless Claims Spike Detection
```sql
SELECT week_ending, initial_claims,
       LAG(initial_claims) OVER (ORDER BY week_ending) as prev_week,
       ((initial_claims - LAG(initial_claims) OVER (ORDER BY week_ending)) * 100.0 /
        LAG(initial_claims) OVER (ORDER BY week_ending)) as pct_change
FROM jobless_claims
WHERE week_ending >= CURRENT_DATE - INTERVAL '6 months'
  AND ((initial_claims - LAG(initial_claims) OVER (ORDER BY week_ending)) * 100.0 /
       LAG(initial_claims) OVER (ORDER BY week_ending)) > 10
ORDER BY pct_change DESC;
```

### Sectors with Strong Employment Growth
```sql
SELECT s.sector_name, s.etf_ticker,
       COUNT(ee.id) as hiring_events,
       SUM(ee.employee_count) as total_hired
FROM sectors s
JOIN employment_events ee ON s.sector_id = ee.sector_id
WHERE ee.event_type = 'hiring'
  AND ee.event_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY s.sector_name, s.etf_ticker
ORDER BY total_hired DESC;
```

---

## Configuration

### Environment Variables

Add to `.env` or `api_keys.yaml`:

```yaml
# Bureau of Labor Statistics API
BLS_API_KEY: "your_bls_api_key_here"

# Layoffs.fyi (if API becomes available)
LAYOFFS_FYI_API_KEY: "your_api_key_here"
```

### Schedule Data Updates

**Daily:**
- Check for new employment events (layoffs, hiring announcements)
- Update sector news sentiment

**Weekly (Thursday):**
- Fetch initial jobless claims (released 8:30 AM ET)
- Alert if claims spike >10% week-over-week

**Monthly (First Friday):**
- Fetch nonfarm payrolls report (released 8:30 AM ET)
- Update sector employment statistics
- Recalculate sector rotation signals

---

## Testing

### Unit Tests
```bash
# Test BLS API connection
python -c "from scripts.data_collection.bls_employment import BLSEmploymentCollector; c = BLSEmploymentCollector(); print(c.fetch_series(['CES0000000001'], 2024, 2024))"

# Test database schema
duckdb data/bigbrother.duckdb "SELECT * FROM sectors;"
```

### Integration Test
```bash
# Full data collection pipeline
python scripts/data_collection/bls_employment.py

# Verify data
duckdb data/bigbrother.duckdb "SELECT COUNT(*) FROM sector_employment_raw;"
```

---

## References

**BLS Documentation:**
- API Home: https://www.bls.gov/developers/home.htm
- Series ID Finder: https://data.bls.gov/cgi-bin/surveymost
- API Registration: https://data.bls.gov/registrationEngine/

**Economic Calendar:**
- Initial Jobless Claims: Every Thursday 8:30 AM ET
- Nonfarm Payrolls: First Friday of month 8:30 AM ET
- JOLTS Report: Monthly (1-month lag)

**PRD References:**
- Section 3.2.10: Government & Institutional Intelligence
- Section 3.2.11: U.S. Department of Labor (DOL)
- Section 3.2.12: Business Sector Classification and Analysis

---

## Next Steps

1. **Run database schema initialization**
2. **Obtain BLS API key** (free, 5 minutes)
3. **Execute initial data collection** (5 years of history)
4. **Implement sector-company mapping**
5. **Build sector rotation strategy**
6. **Backtest with employment signals**

**Estimated Time to Complete:** 4-6 weeks (Tier 1 Extension)

---

**Questions or Issues?**
See main PRD (`docs/PRD.md`) for complete requirements and context.
