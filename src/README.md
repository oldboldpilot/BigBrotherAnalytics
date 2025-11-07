# Source Code Directory

This directory contains all source code for the BigBrotherAnalytics trading platform.

## Structure

### market_intelligence/
Market Intelligence & Impact Analysis Engine
- Multi-source data ingestion (news, market data, economic indicators)
- NLP sentiment analysis (FinBERT, entity recognition)
- Event extraction and classification
- Impact prediction and graph generation

### correlation_engine/
Trading Correlation Analysis Tool
- Statistical correlation analysis (Pearson, Spearman)
- Time-lagged cross-correlations
- Leading/lagging indicator identification
- C++23 performance-critical components

### trading_decision/
Intelligent Trading Decision Engine
- Options trading strategies (straddles, strangles, volatility arbitrage)
- ML ensemble (RL + DNN + GNN)
- Signal aggregation and decision-making
- Portfolio optimization

### risk_management/
Risk Management System
- Position sizing (Kelly Criterion)
- Stop loss management
- Daily loss limits
- Monte Carlo pre-trade simulation
- Portfolio constraints

### explainability/
Explainability Layer
- SHAP values for ML predictions
- LIME for local explanations
- Feature importance tracking
- Decision visualization and logging

### schwab_api/
Schwab API Integration
- OAuth 2.0 authentication (C++23)
- Market data retrieval
- Options chain parsing
- Order placement and management
- Real-time WebSocket streaming

### utils/
Shared Utilities
- Logging configuration
- Database connections (DuckDB)
- Configuration management
- Helper functions
