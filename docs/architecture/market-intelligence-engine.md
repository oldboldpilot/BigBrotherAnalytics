# Market Intelligence & Impact Analysis Engine - Architecture Design Document

**Version:** 1.0.0
**Date:** November 6, 2025
**Status:** Draft - Design Phase
**Author:** Olumuyiwa Oluwasanmi
**Related:** [Product Requirements Document](../PRD.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Component Architecture](#3-component-architecture)
4. [Data Flow Architecture](#4-data-flow-architecture)
5. [Database Schema Design](#5-database-schema-design)
6. [API Specifications](#6-api-specifications)
7. [Technology Stack](#7-technology-stack)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Implementation Guidelines](#9-implementation-guidelines)
10. [Performance Considerations](#10-performance-considerations)

---

## 1. Overview

### 1.1 Purpose

The Market Intelligence & Impact Analysis Engine is a sophisticated ML-powered system that:
- Ingests data from 15+ government agencies and multiple market sources
- Processes news, events, and regulatory information in real-time
- Predicts market impacts on specific securities with confidence scores
- Generates impact graphs showing causal chains and relationship strengths
- Provides actionable intelligence for trading decisions

### 1.2 Key Capabilities

```mermaid
mindmap
  root((Market Intelligence Engine))
    Data Ingestion
      Government APIs
      Market Data Feeds
      News Aggregators
      Social Media
    NLP Processing
      Entity Recognition
      Sentiment Analysis
      Event Classification
      Topic Modeling
    Impact Analysis
      Prediction Models
      Confidence Scoring
      Multi-hop Analysis
      Time-to-Impact
    Graph Generation
      Impact Graphs
      Relationship Mapping
      Causal Chains
      Network Analysis
    Output & Alerts
      REST APIs
      Real-time Alerts
      Ranked Opportunities
      Explanations
```

### 1.3 Design Principles

- **Speed First:** < 5 second end-to-end latency for critical predictions
- **Modular:** Each component can be developed and deployed independently
- **Scalable:** Handle 100,000+ articles per day, 10,000+ securities
- **Cost-Effective:** Use open-source tools and affordable data sources
- **Reliable:** 99.5%+ uptime with graceful degradation
- **Observable:** Full monitoring, logging, and tracing

---

## 2. System Architecture

### 2.1 High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        DS1[Government APIs<br/>FRED, SEC, FDA, etc.]
        DS2[Market Data<br/>Polygon.io, Alpha Vantage]
        DS3[News APIs<br/>NewsAPI, RSS Feeds]
        DS4[Social Media<br/>Twitter, Reddit]
    end

    subgraph "Ingestion Layer"
        IL1[API Collectors]
        IL2[Web Scrapers]
        IL3[Stream Processors]
        IL4[Rate Limiters]
    end

    subgraph "Message Queue"
        MQ[Apache Kafka<br/>Event Streaming]
    end

    subgraph "Processing Layer"
        PL1[Document Parser]
        PL2[NLP Pipeline]
        PL3[Entity Extractor]
        PL4[Event Classifier]
    end

    subgraph "ML Layer"
        ML1[Impact Predictor]
        ML2[Sentiment Analyzer]
        ML3[Graph Generator]
        ML4[Confidence Scorer]
    end

    subgraph "Storage Layer"
        ST1[(PostgreSQL<br/>TimescaleDB)]
        ST2[(DuckDB<br/>Analytics)]
        ST3[(Redis<br/>Cache)]
        ST4[Parquet Files<br/>Archive]
    end

    subgraph "API Layer"
        API1[REST API<br/>FastAPI]
        API2[WebSocket<br/>Real-time]
        API3[GraphQL<br/>Flexible Queries]
    end

    subgraph "Monitoring"
        MON1[Prometheus]
        MON2[Grafana]
        MON3[Sentry]
    end

    DS1 & DS2 & DS3 & DS4 --> IL1 & IL2 & IL3
    IL1 & IL2 & IL3 --> IL4
    IL4 --> MQ
    MQ --> PL1 --> PL2 --> PL3 --> PL4
    PL4 --> ML1 & ML2 & ML3 & ML4
    ML1 & ML2 & ML3 & ML4 --> ST1 & ST2 & ST3
    ST1 & ST2 & ST3 --> API1 & API2 & API3
    ST4 -.Archival.-> ST2

    API1 & API2 & API3 -.Metrics.-> MON1
    MON1 --> MON2
    API1 & API2 & API3 -.Errors.-> MON3

    style MQ fill:#f9f,stroke:#333,stroke-width:2px
    style ST1 fill:#bbf,stroke:#333,stroke-width:2px
    style API1 fill:#bfb,stroke:#333,stroke-width:2px
```

### 2.2 Deployment Tiers

```mermaid
graph LR
    subgraph "Tier 1: Zero-Fee (Months 1-2)"
        T1A[Python Scripts]
        T1B[DuckDB]
        T1C[Free APIs Only]
        T1D[Single Machine]
    end

    subgraph "Tier 2: Initial Production (Months 3-4)"
        T2A[Airflow Orchestration]
        T2B[PostgreSQL + DuckDB]
        T2C[Paid APIs Added]
        T2D[Redis Cache]
    end

    subgraph "Tier 3: Full Production (Month 5+)"
        T3A[Kafka Streaming]
        T3B[Multi-node Postgres]
        T3C[All Data Sources]
        T3D[Full Monitoring]
    end

    T1A --> T2A
    T1B --> T2B
    T1C --> T2C
    T1D --> T2D

    T2A --> T3A
    T2B --> T3B
    T2C --> T3C
    T2D --> T3D

    style T1A fill:#afa,stroke:#333,stroke-width:2px
    style T2A fill:#ffa,stroke:#333,stroke-width:2px
    style T3A fill:#faa,stroke:#333,stroke-width:2px
```

---

## 3. Component Architecture

### 3.1 Data Ingestion Components

```mermaid
graph TB
    subgraph "API Collectors"
        AC1[Government API Collector<br/>FRED, SEC, Congress]
        AC2[Market Data Collector<br/>Polygon.io, Finnhub]
        AC3[News API Collector<br/>NewsAPI, MarketAux]
    end

    subgraph "Web Scrapers"
        WS1[Scrapy Spiders<br/>FDA, EPA, Court Sites]
        WS2[Playwright Scrapers<br/>JavaScript-heavy sites]
        WS3[RSS Feed Readers<br/>News sources]
    end

    subgraph "Rate Limiting & Retry"
        RL1[Rate Limiter<br/>python-ratelimit]
        RL2[Retry Logic<br/>tenacity]
        RL3[Cache<br/>requests-cache]
    end

    subgraph "Data Validation"
        DV1[Pydantic Schemas]
        DV2[Great Expectations]
        DV3[Data Quality Checks]
    end

    subgraph "Output Queue"
        OQ[Kafka Topics<br/>raw-data-*]
    end

    AC1 & AC2 & AC3 --> RL1
    WS1 & WS2 & WS3 --> RL1
    RL1 --> RL2 --> RL3
    RL3 --> DV1 --> DV2 --> DV3
    DV3 --> OQ

    style OQ fill:#f9f,stroke:#333,stroke-width:2px
```

### 3.2 Processing Pipeline Components

```mermaid
graph TB
    subgraph "Document Processing"
        DP1[Apache Tika<br/>PDF/Office Parsing]
        DP2[pdfplumber<br/>Table Extraction]
        DP3[Tesseract OCR<br/>Scanned Docs]
        DP4[Text Cleaner<br/>Normalization]
    end

    subgraph "NLP Pipeline"
        NLP1[spaCy<br/>NER & Parsing]
        NLP2[Transformers<br/>BERT/FinBERT]
        NLP3[sentence-transformers<br/>Embeddings]
        NLP4[Custom Models<br/>Fine-tuned]
    end

    subgraph "Entity & Event Extraction"
        EE1[Company Recognition]
        EE2[Person Recognition]
        EE3[Event Classification]
        EE4[Relationship Extraction]
    end

    subgraph "Storage"
        ST1[(PostgreSQL<br/>Entities & Events)]
        ST2[(pgvector<br/>Embeddings)]
    end

    DP1 & DP2 & DP3 --> DP4
    DP4 --> NLP1 --> NLP2 --> NLP3
    NLP1 & NLP2 & NLP3 & NLP4 --> EE1 & EE2 & EE3 & EE4
    EE1 & EE2 & EE3 & EE4 --> ST1 & ST2

    style ST1 fill:#bbf,stroke:#333,stroke-width:2px
    style ST2 fill:#bbf,stroke:#333,stroke-width:2px
```

### 3.3 ML & Prediction Components

```mermaid
graph TB
    subgraph "Feature Engineering"
        FE1[Technical Features<br/>Price, Volume]
        FE2[Sentiment Features<br/>NLP Output]
        FE3[Event Features<br/>Classified Events]
        FE4[Macro Features<br/>Economic Data]
        FE5[Graph Features<br/>Network Metrics]
    end

    subgraph "Impact Prediction Models"
        IPM1[XGBoost<br/>Direction Prediction]
        IPM2[LightGBM<br/>Magnitude Prediction]
        IPM3[Neural Network<br/>Complex Patterns]
        IPM4[Ensemble<br/>Model Averaging]
    end

    subgraph "Confidence Scoring"
        CS1[Prediction Variance]
        CS2[Historical Accuracy]
        CS3[Feature Importance]
        CS4[Ensemble Agreement]
    end

    subgraph "Output"
        OUT1[Impact Predictions]
        OUT2[Confidence Scores]
        OUT3[Time-to-Impact]
        OUT4[Explanations]
    end

    FE1 & FE2 & FE3 & FE4 & FE5 --> IPM1 & IPM2 & IPM3
    IPM1 & IPM2 & IPM3 --> IPM4
    IPM4 --> CS1 & CS2 & CS3 & CS4
    CS1 & CS2 & CS3 & CS4 --> OUT1 & OUT2 & OUT3 & OUT4

    style IPM4 fill:#faa,stroke:#333,stroke-width:2px
    style OUT1 fill:#afa,stroke:#333,stroke-width:2px
```

### 3.4 Graph Generation Components

```mermaid
graph TB
    subgraph "Graph Building"
        GB1[Node Creator<br/>Companies, Events]
        GB2[Edge Creator<br/>Relationships]
        GB3[Weight Calculator<br/>Strength & Confidence]
        GB4[Graph Assembler<br/>NetworkX/AGE]
    end

    subgraph "Graph Analysis"
        GA1[Centrality Metrics<br/>PageRank, Betweenness]
        GA2[Community Detection<br/>Clustering]
        GA3[Path Analysis<br/>Shortest Paths]
        GA4[Impact Propagation<br/>Multi-hop]
    end

    subgraph "Storage"
        GS1[(Apache AGE<br/>Graph DB in PostgreSQL)]
        GS2[(DuckDB<br/>Graph Analytics)]
    end

    subgraph "Visualization Data"
        VD1[JSON/GraphSON]
        VD2[D3.js Format]
        VD3[Cytoscape Format]
    end

    GB1 --> GB2 --> GB3 --> GB4
    GB4 --> GA1 & GA2 & GA3 & GA4
    GA1 & GA2 & GA3 & GA4 --> GS1 & GS2
    GS1 & GS2 --> VD1 & VD2 & VD3

    style GS1 fill:#bbf,stroke:#333,stroke-width:2px
    style VD1 fill:#afa,stroke:#333,stroke-width:2px
```

---

## 4. Data Flow Architecture

### 4.1 Real-Time Data Flow

```mermaid
sequenceDiagram
    participant DS as Data Source
    participant IC as Ingestion Collector
    participant K as Kafka
    participant NLP as NLP Pipeline
    participant ML as ML Predictor
    participant DB as PostgreSQL
    participant Cache as Redis
    participant API as REST API
    participant Client as Trading Engine

    DS->>IC: Raw Data (API/Scrape)
    IC->>IC: Rate Limit & Validate
    IC->>K: Publish to Topic
    K->>NLP: Consume Message
    NLP->>NLP: Extract Entities & Events
    NLP->>K: Publish Processed Data
    K->>ML: Consume Processed Data
    ML->>ML: Feature Engineering
    ML->>ML: Run Prediction Models
    ML->>ML: Calculate Confidence
    ML->>DB: Store Predictions
    ML->>Cache: Cache Recent Predictions
    ML->>K: Publish Alert (if significant)
    Client->>API: Request Predictions
    API->>Cache: Check Cache
    alt Cache Hit
        Cache-->>API: Return Cached Data
    else Cache Miss
        API->>DB: Query Database
        DB-->>API: Return Data
        API->>Cache: Update Cache
    end
    API-->>Client: Return Predictions

    Note over DS,Client: End-to-End Latency Target: < 5 seconds
```

### 4.2 Batch Processing Flow

```mermaid
flowchart TD
    Start([Scheduled Job<br/>Airflow DAG]) --> Check{Check Data<br/>Sources}

    Check -->|Government APIs| Gov[Fetch FRED, SEC,<br/>Congress, FDA, EPA]
    Check -->|Market Data| Market[Fetch Historical<br/>Market Data]
    Check -->|News Archives| News[Fetch News<br/>Archives]

    Gov --> Validate1[Validate &<br/>Transform]
    Market --> Validate2[Validate &<br/>Transform]
    News --> Validate3[Validate &<br/>Transform]

    Validate1 & Validate2 & Validate3 --> Store[Store Raw Data<br/>Parquet Files]

    Store --> Load[Load to DuckDB<br/>for Analysis]

    Load --> Analyze[Run Batch<br/>Analytics]

    Analyze --> Correlations[Calculate<br/>Correlations]
    Analyze --> Patterns[Detect<br/>Patterns]
    Analyze --> Features[Generate<br/>Features]

    Correlations & Patterns & Features --> Train[Train/Update<br/>ML Models]

    Train --> Evaluate[Evaluate Model<br/>Performance]

    Evaluate --> Deploy{Performance<br/>Good?}

    Deploy -->|Yes| UpdateProd[Update Production<br/>Models]
    Deploy -->|No| Alert[Alert for<br/>Review]

    UpdateProd --> Archive[Archive to<br/>PostgreSQL]
    Alert --> Archive

    Archive --> End([Complete])

    style Start fill:#afa,stroke:#333,stroke-width:2px
    style End fill:#afa,stroke:#333,stroke-width:2px
    style Deploy fill:#faa,stroke:#333,stroke-width:2px
```

### 4.3 Impact Graph Generation Flow

```mermaid
flowchart TB
    Start([New Event<br/>Detected]) --> Parse[Parse Event<br/>Extract Entities]

    Parse --> Identify[Identify Primary<br/>Entity]

    Identify --> Query1[Query Historical<br/>Relationships]
    Identify --> Query2[Query Market<br/>Correlations]
    Identify --> Query3[Query Supply<br/>Chain Data]

    Query1 & Query2 & Query3 --> Build[Build Node Set]

    Build --> Connect[Create Edges<br/>with Weights]

    Connect --> Classify[Classify Edge<br/>Types]

    Classify --> Calculate[Calculate Impact<br/>Propagation]

    Calculate --> Iterate{More Hops<br/>Needed?}

    Iterate -->|Yes, <3 hops| Expand[Expand to<br/>Next Level]
    Expand --> Connect

    Iterate -->|No| Analyze[Analyze Graph<br/>Metrics]

    Analyze --> Rank[Rank Impacted<br/>Entities]

    Rank --> Store[(Store in<br/>Apache AGE)]

    Store --> Cache[(Cache in<br/>Redis)]

    Cache --> Notify[Notify Trading<br/>Engine]

    Notify --> End([Complete])

    style Start fill:#afa,stroke:#333,stroke-width:2px
    style End fill:#afa,stroke:#333,stroke-width:2px
    style Calculate fill:#faa,stroke:#333,stroke-width:2px
```

---

## 5. Database Schema Design

### 5.1 PostgreSQL Schema - Core Tables

```mermaid
erDiagram
    RAW_DATA ||--o{ PROCESSED_DOCUMENTS : "processed_into"
    PROCESSED_DOCUMENTS ||--o{ ENTITIES : "contains"
    PROCESSED_DOCUMENTS ||--o{ EVENTS : "contains"
    ENTITIES ||--o{ ENTITY_RELATIONSHIPS : "source"
    ENTITIES ||--o{ ENTITY_RELATIONSHIPS : "target"
    EVENTS ||--o{ EVENT_IMPACTS : "causes"
    ENTITIES ||--o{ EVENT_IMPACTS : "affects"
    EVENTS ||--o{ IMPACT_PREDICTIONS : "triggers"
    ENTITIES ||--o{ IMPACT_PREDICTIONS : "targets"

    RAW_DATA {
        uuid id PK
        timestamp collected_at
        text source_name
        text source_type
        jsonb raw_content
        text url
        text checksum
        timestamp created_at
    }

    PROCESSED_DOCUMENTS {
        uuid id PK
        uuid raw_data_id FK
        text document_type
        text title
        text content
        jsonb metadata
        vector embedding
        timestamp processed_at
        timestamp created_at
    }

    ENTITIES {
        uuid id PK
        text entity_type
        text name
        text ticker
        jsonb attributes
        vector embedding
        float confidence
        timestamp created_at
        timestamp updated_at
    }

    EVENTS {
        uuid id PK
        uuid document_id FK
        text event_type
        text description
        timestamp event_time
        text severity
        jsonb details
        float significance
        timestamp created_at
    }

    ENTITY_RELATIONSHIPS {
        uuid id PK
        uuid source_entity_id FK
        uuid target_entity_id FK
        text relationship_type
        float strength
        jsonb metadata
        timestamp valid_from
        timestamp valid_to
    }

    EVENT_IMPACTS {
        uuid id PK
        uuid event_id FK
        uuid entity_id FK
        text impact_type
        float predicted_magnitude
        float confidence
        int time_to_impact_hours
        jsonb explanation
        timestamp created_at
    }

    IMPACT_PREDICTIONS {
        uuid id PK
        uuid event_id FK
        uuid entity_id FK
        text direction
        float magnitude
        float confidence
        int time_horizon_hours
        jsonb features
        text model_version
        timestamp created_at
    }
```

### 5.2 TimescaleDB Hypertables - Time Series Data

```sql
-- Market data time series
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    vwap NUMERIC,
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'time');

-- Create continuous aggregates for different intervals
CREATE MATERIALIZED VIEW market_data_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume
FROM market_data
GROUP BY bucket, symbol;

-- News sentiment time series
CREATE TABLE news_sentiment (
    time TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    entity_id UUID NOT NULL,
    sentiment_score NUMERIC,
    article_count INTEGER,
    PRIMARY KEY (time, source, entity_id)
);

SELECT create_hypertable('news_sentiment', 'time');

-- Economic indicators
CREATE TABLE economic_indicators (
    time TIMESTAMPTZ NOT NULL,
    indicator_name TEXT NOT NULL,
    value NUMERIC,
    source TEXT,
    PRIMARY KEY (time, indicator_name)
);

SELECT create_hypertable('economic_indicators', 'time');
```

### 5.3 Apache AGE - Graph Schema

```sql
-- Create graph for impact analysis
SELECT create_graph('impact_graph');

-- Company nodes
SELECT * FROM cypher('impact_graph', $$
    CREATE (:Company {
        id: 'uuid',
        name: 'string',
        ticker: 'string',
        sector: 'string',
        market_cap: float
    })
$$) as (result agtype);

-- Event nodes
SELECT * FROM cypher('impact_graph', $$
    CREATE (:Event {
        id: 'uuid',
        type: 'string',
        description: 'string',
        timestamp: timestamp,
        severity: float
    })
$$) as (result agtype);

-- Impact relationships
SELECT * FROM cypher('impact_graph', $$
    MATCH (e:Event), (c:Company)
    WHERE e.id = 'event_uuid' AND c.id = 'company_uuid'
    CREATE (e)-[:IMPACTS {
        magnitude: float,
        confidence: float,
        time_to_impact: int,
        impact_type: 'string'
    }]->(c)
$$) as (result agtype);

-- Supply chain relationships
SELECT * FROM cypher('impact_graph', $$
    MATCH (c1:Company), (c2:Company)
    WHERE c1.id = 'supplier_uuid' AND c2.id = 'customer_uuid'
    CREATE (c1)-[:SUPPLIES_TO {
        volume: float,
        dependency_score: float
    }]->(c2)
$$) as (result agtype);

-- Multi-hop impact query
SELECT * FROM cypher('impact_graph', $$
    MATCH path = (e:Event)-[:IMPACTS*1..3]->(c:Company)
    WHERE e.id = 'event_uuid'
    RETURN path, c.name, length(path) as hops
    ORDER BY hops, c.market_cap DESC
$$) as (path agtype, company_name agtype, hops agtype);
```

### 5.4 DuckDB Schemas - Analytics

```sql
-- Historical market data (query Parquet files)
CREATE VIEW historical_prices AS
SELECT * FROM read_parquet('data/market/*.parquet');

-- News archive
CREATE VIEW news_archive AS
SELECT * FROM read_parquet('data/news/*.parquet');

-- Correlation analysis results
CREATE TABLE correlation_results (
    symbol_a VARCHAR,
    symbol_b VARCHAR,
    timeframe VARCHAR,
    correlation DOUBLE,
    p_value DOUBLE,
    sample_size INTEGER,
    calculated_at TIMESTAMP
);

-- Backtesting results
CREATE TABLE backtest_results (
    strategy_name VARCHAR,
    symbol VARCHAR,
    entry_date DATE,
    exit_date DATE,
    entry_price DOUBLE,
    exit_price DOUBLE,
    return_pct DOUBLE,
    holding_days INTEGER,
    metadata JSON
);
```

---

## 6. API Specifications

### 6.1 REST API Endpoints

```yaml
openapi: 3.0.0
info:
  title: Market Intelligence & Impact Analysis API
  version: 1.0.0
  description: API for accessing market intelligence predictions and impact graphs

servers:
  - url: http://localhost:8000/api/v1
    description: Local development
  - url: https://api.bigbrother.analytics/v1
    description: Production

paths:
  /predictions:
    get:
      summary: Get impact predictions
      parameters:
        - name: entity_id
          in: query
          schema:
            type: string
            format: uuid
        - name: min_confidence
          in: query
          schema:
            type: number
            minimum: 0
            maximum: 1
        - name: time_horizon
          in: query
          schema:
            type: string
            enum: [1h, 4h, 1d, 1w]
        - name: limit
          in: query
          schema:
            type: integer
            default: 100
      responses:
        '200':
          description: List of predictions
          content:
            application/json:
              schema:
                type: object
                properties:
                  predictions:
                    type: array
                    items:
                      $ref: '#/components/schemas/Prediction'
                  total:
                    type: integer
                  timestamp:
                    type: string
                    format: date-time

  /predictions/{prediction_id}:
    get:
      summary: Get specific prediction details
      parameters:
        - name: prediction_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Prediction details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PredictionDetail'

  /impact-graph:
    get:
      summary: Get impact graph for an event
      parameters:
        - name: event_id
          in: query
          required: true
          schema:
            type: string
            format: uuid
        - name: max_hops
          in: query
          schema:
            type: integer
            default: 3
            maximum: 5
        - name: min_impact
          in: query
          schema:
            type: number
            default: 0.1
      responses:
        '200':
          description: Impact graph
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ImpactGraph'

  /events:
    get:
      summary: Get recent significant events
      parameters:
        - name: from_date
          in: query
          schema:
            type: string
            format: date-time
        - name: severity
          in: query
          schema:
            type: string
            enum: [low, medium, high, critical]
        - name: event_type
          in: query
          schema:
            type: string
      responses:
        '200':
          description: List of events
          content:
            application/json:
              schema:
                type: object
                properties:
                  events:
                    type: array
                    items:
                      $ref: '#/components/schemas/Event'

  /entities/{entity_id}/sentiment:
    get:
      summary: Get sentiment analysis for entity
      parameters:
        - name: entity_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
        - name: timeframe
          in: query
          schema:
            type: string
            enum: [1h, 4h, 1d, 1w, 1m]
            default: 1d
      responses:
        '200':
          description: Sentiment data
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SentimentAnalysis'

  /alerts:
    get:
      summary: Get active alerts
      parameters:
        - name: priority
          in: query
          schema:
            type: string
            enum: [low, medium, high, critical]
        - name: acknowledged
          in: query
          schema:
            type: boolean
      responses:
        '200':
          description: List of alerts
          content:
            application/json:
              schema:
                type: object
                properties:
                  alerts:
                    type: array
                    items:
                      $ref: '#/components/schemas/Alert'

components:
  schemas:
    Prediction:
      type: object
      properties:
        id:
          type: string
          format: uuid
        event_id:
          type: string
          format: uuid
        entity_id:
          type: string
          format: uuid
        entity_name:
          type: string
        ticker:
          type: string
        direction:
          type: string
          enum: [positive, negative, neutral]
        magnitude:
          type: number
          description: Predicted percentage change
        confidence:
          type: number
          minimum: 0
          maximum: 1
        time_to_impact_hours:
          type: integer
        model_version:
          type: string
        created_at:
          type: string
          format: date-time

    PredictionDetail:
      allOf:
        - $ref: '#/components/schemas/Prediction'
        - type: object
          properties:
            explanation:
              type: object
              properties:
                key_factors:
                  type: array
                  items:
                    type: string
                feature_importance:
                  type: object
                related_events:
                  type: array
                  items:
                    type: string
            historical_accuracy:
              type: object
              properties:
                similar_predictions:
                  type: integer
                accuracy_rate:
                  type: number

    ImpactGraph:
      type: object
      properties:
        event_id:
          type: string
          format: uuid
        nodes:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
              type:
                type: string
                enum: [event, company, sector, indicator]
              name:
                type: string
              attributes:
                type: object
        edges:
          type: array
          items:
            type: object
            properties:
              source:
                type: string
              target:
                type: string
              type:
                type: string
              weight:
                type: number
              confidence:
                type: number
        metrics:
          type: object
          properties:
            total_nodes:
              type: integer
            total_edges:
              type: integer
            max_impact:
              type: number
            affected_sectors:
              type: array
              items:
                type: string

    Event:
      type: object
      properties:
        id:
          type: string
          format: uuid
        type:
          type: string
        description:
          type: string
        timestamp:
          type: string
          format: date-time
        severity:
          type: string
          enum: [low, medium, high, critical]
        source:
          type: string
        entities:
          type: array
          items:
            type: string

    SentimentAnalysis:
      type: object
      properties:
        entity_id:
          type: string
          format: uuid
        timeframe:
          type: string
        overall_sentiment:
          type: number
          minimum: -1
          maximum: 1
        sentiment_trend:
          type: string
          enum: [improving, stable, deteriorating]
        article_count:
          type: integer
        time_series:
          type: array
          items:
            type: object
            properties:
              timestamp:
                type: string
                format: date-time
              sentiment:
                type: number
              volume:
                type: integer

    Alert:
      type: object
      properties:
        id:
          type: string
          format: uuid
        priority:
          type: string
          enum: [low, medium, high, critical]
        type:
          type: string
        message:
          type: string
        entity_id:
          type: string
          format: uuid
        event_id:
          type: string
          format: uuid
        created_at:
          type: string
          format: date-time
        acknowledged:
          type: boolean
        acknowledged_at:
          type: string
          format: date-time
```

### 6.2 WebSocket API - Real-time Updates

```javascript
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/v1/stream');

// Subscribe to prediction updates
ws.send(JSON.stringify({
    action: 'subscribe',
    channels: ['predictions', 'alerts'],
    filters: {
        min_confidence: 0.7,
        entities: ['AAPL', 'TSLA', 'MSFT']
    }
}));

// Receive updates
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch(data.type) {
        case 'prediction':
            handleNewPrediction(data.payload);
            break;
        case 'alert':
            handleAlert(data.payload);
            break;
        case 'sentiment_update':
            handleSentimentUpdate(data.payload);
            break;
    }
};
```

---

## 7. Technology Stack

### 7.1 Performance-Critical Architecture

**CRITICAL:** The system uses a hybrid C++23/Python architecture:
- **C++23** for ultra-low latency data paths (< 1ms)
- **Python 3.14+ GIL-Free** for ML development and non-critical paths
- **CUDA** for GPU-accelerated ML inference
- **MPI/OpenMP/UPC++** for massive parallelization

```mermaid
graph TB
    subgraph "High-Performance Core (C++23)"
        CPP1[Market Data Processor<br/>C++23 + MPI]
        CPP2[Real-time Stream Handler<br/>C++23 + OpenMP]
        CPP3[Feature Extractor<br/>C++23 + UPC++]
        CPP4[Correlation Engine<br/>C++23 + MPI]
        CPP5[Time-Series Ops<br/>C++23 SIMD]
    end

    subgraph "GPU-Accelerated ML (CUDA + PyTorch)"
        GPU1[Model Inference<br/>vLLM + CUDA]
        GPU2[Embedding Generation<br/>PyTorch + CUDA]
        GPU3[Batch Predictions<br/>TensorRT]
        GPU4[NLP Models<br/>Transformers + CUDA]
    end

    subgraph "Python ML/AI Layer (GIL-Free)"
        PY1[Model Training<br/>PyTorch Multi-thread]
        PY2[Feature Engineering<br/>Parallel Processing]
        PY3[Data Preprocessing<br/>Polars/DuckDB]
        PY4[Ensemble Models<br/>XGBoost/LightGBM]
    end

    subgraph "Data Ingestion (Python)"
        DI1[API Collectors<br/>aiohttp/httpx]
        DI2[Web Scrapers<br/>Scrapy]
        DI3[Document Parsers<br/>Apache Tika]
    end

    subgraph "Storage Layer"
        ST1[(PostgreSQL 16+<br/>TimescaleDB, AGE, pgvector)]
        ST2[(DuckDB<br/>Analytics)]
        ST3[(Redis<br/>< 1ms Cache)]
        ST4[Parquet Files<br/>Archive]
    end

    subgraph "API & Orchestration"
        API1[FastAPI<br/>REST/WebSocket]
        OR1[Apache Airflow<br/>Scheduling]
    end

    DI1 & DI2 & DI3 --> CPP2
    CPP2 --> CPP1
    CPP1 --> CPP3
    CPP3 --> GPU1 & GPU2
    GPU1 & GPU2 --> PY1
    PY1 --> GPU3 & GPU4
    GPU3 & GPU4 --> ST1 & ST2 & ST3
    ST1 & ST2 & ST3 --> API1
    CPP4 --> ST1
    OR1 --> DI1 & DI2
    ST4 -.Archive.-> ST2

    style CPP1 fill:#faa,stroke:#333,stroke-width:3px
    style GPU1 fill:#f9f,stroke:#333,stroke-width:3px
    style ST1 fill:#bbf,stroke:#333,stroke-width:2px
```

### 7.2 Component Technology Mapping

| Component | Language | Framework/Library | Purpose | Latency Target |
|-----------|----------|-------------------|---------|----------------|
| **Critical Path** ||||
| Market Data Ingestion | C++23 | MPI, ZeroMQ | Process incoming market data | < 100μs |
| Stream Processing | C++23 | OpenMP, lock-free queues | Real-time event handling | < 500μs |
| Feature Extraction | C++23 | UPC++, Eigen | Parallel feature computation | < 1ms |
| Correlation Calc | C++23 | MPI, Intel MKL | Multi-core correlation | < 100ms |
| Time-Series Ops | C++23 | SIMD, TA-Lib | Technical indicators | < 10μs/op |
| **ML Inference** ||||
| LLM Inference | Python | vLLM + CUDA | High-throughput predictions | < 50ms |
| Embeddings | Python | PyTorch + CUDA | Semantic embeddings | < 20ms |
| Batch Inference | Python/C++ | TensorRT | Optimized inference | < 10ms |
| NLP Models | Python | Transformers + CUDA | Text analysis | < 100ms |
| **ML Training** ||||
| Model Training | Python 3.14+ | PyTorch + CUDA | GPU training | N/A |
| Feature Engineering | Python 3.14+ | Polars, NumPy+MKL | Parallel processing | < 1s |
| Ensemble Training | Python | XGBoost/LightGBM | Gradient boosting | N/A |
| **Data Layer** ||||
| Data Ingestion | Python | aiohttp, Scrapy | API/web scraping | < 500ms |
| Document Parsing | Python | Apache Tika, pdfplumber | PDF/doc parsing | < 2s |
| NLP Pipeline | Python | spaCy, Transformers | Entity extraction | < 2s |
| Analytics | SQL/Python | DuckDB | Batch analytics | Varies |
| **Infrastructure** ||||
| Database | SQL | PostgreSQL 16+ | Structured data | < 10ms |
| Time-Series | SQL | TimescaleDB | Market data | < 10ms |
| Graph | SQL | Apache AGE | Impact graphs | < 100ms |
| Vectors | SQL | pgvector | Embeddings | < 50ms |
| Cache | Key-Value | Redis | Hot data | < 1ms |
| API | Python | FastAPI + Uvicorn | REST/WebSocket | < 100ms |
| Orchestration | Python | Apache Airflow | Job scheduling | N/A |

### 7.3 C++23 Core Components

```cpp
// Example: High-performance market data processor
// File: src/core/market_data_processor.hpp

#include <expected>      // C++23 error handling
#include <mdspan>        // C++23 multi-dimensional arrays
#include <flat_map>      // C++23 cache-friendly containers
#include <mpi.h>         // Message Passing Interface
#include <omp.h>         // OpenMP
#include <upcxx/upcxx.hpp> // UPC++

namespace market_intelligence {

// Cache-friendly data structure using C++23 std::flat_map
class MarketDataCache {
private:
    std::flat_map<std::string, MarketTick> tick_cache_;
    std::shared_mutex cache_mutex_;

public:
    // C++23 deducing this for better performance
    template<typename Self>
    auto get_tick(this Self&& self, const std::string& symbol)
        -> std::expected<MarketTick, Error> {
        if constexpr (std::is_const_v<std::remove_reference_t<Self>>) {
            std::shared_lock lock(self.cache_mutex_);
            auto it = self.tick_cache_.find(symbol);
            if (it != self.tick_cache_.end()) {
                return it->second;
            }
            return std::unexpected(Error::NotFound);
        } else {
            // Non-const version
            std::unique_lock lock(self.cache_mutex_);
            // ... update logic
        }
    }
};

// Parallel correlation calculation with MPI
class CorrelationEngine {
public:
    // Calculate correlations across N securities using MPI
    static auto calculate_correlations(
        std::span<const PriceData> prices,
        int window_size
    ) -> std::vector<CorrelationPair> {

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Distribute work across MPI ranks
        const size_t total_pairs = (prices.size() * (prices.size() - 1)) / 2;
        const size_t pairs_per_rank = total_pairs / size;
        const size_t start_idx = rank * pairs_per_rank;
        const size_t end_idx = (rank == size - 1) ? total_pairs : start_idx + pairs_per_rank;

        std::vector<CorrelationPair> local_results;

        // OpenMP parallel loop within each MPI rank
        #pragma omp parallel for schedule(dynamic)
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            auto [i, j] = index_to_pair(idx, prices.size());
            auto corr = compute_correlation_simd(
                prices[i].returns,
                prices[j].returns,
                window_size
            );

            if (corr.has_value() && std::abs(corr.value()) > 0.5) {
                #pragma omp critical
                local_results.push_back({i, j, corr.value()});
            }
        }

        // Gather results from all ranks
        std::vector<CorrelationPair> all_results;
        gather_results_mpi(local_results, all_results, rank, size);

        return all_results;
    }

private:
    // SIMD-optimized correlation calculation
    static auto compute_correlation_simd(
        std::span<const double> x,
        std::span<const double> y,
        size_t window
    ) -> std::expected<double, Error> {
        // Use Intel MKL or manual SIMD (AVX-512)
        // ... SIMD implementation
    }
};

// UPC++ for distributed shared memory
class DistributedMarketData {
private:
    upcxx::global_ptr<MarketTick> distributed_ticks_;
    size_t local_size_;

public:
    DistributedMarketData(size_t total_size) {
        // Allocate distributed shared memory
        local_size_ = total_size / upcxx::rank_n();
        distributed_ticks_ = upcxx::new_array<MarketTick>(local_size_);
    }

    // One-sided RDMA get (no synchronization needed)
    MarketTick get_remote_tick(int rank, size_t offset) {
        auto remote_ptr = distributed_ticks_ + offset;
        return upcxx::rget(remote_ptr).wait();
    }

    // One-sided RDMA put
    void put_remote_tick(int rank, size_t offset, const MarketTick& tick) {
        auto remote_ptr = distributed_ticks_ + offset;
        upcxx::rput(tick, remote_ptr).wait();
    }
};

} // namespace market_intelligence
```

### 7.4 CUDA-Accelerated ML Inference

```python
# GPU-accelerated inference with vLLM
from vllm import LLM, SamplingParams
import torch

class GPUInferenceEngine:
    """High-throughput ML inference on GPU"""

    def __init__(self, model_name: str, num_gpus: int = 1):
        # Initialize vLLM for high-throughput inference
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.9,
            max_num_seqs=256,  # Batch size
            dtype="half",  # FP16 for speed
        )

        # Configure CUDA for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    async def batch_predict_impact(
        self,
        events: list[Event],
        entities: list[Entity]
    ) -> list[ImpactPrediction]:
        """Batch predict impacts using GPU acceleration"""

        # Prepare prompts
        prompts = self._prepare_prompts(events, entities)

        # Generate predictions with vLLM (continuous batching)
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=256
        )

        # vLLM automatically batches and uses PagedAttention
        outputs = self.llm.generate(prompts, sampling_params)

        # Parse outputs
        predictions = [self._parse_prediction(o) for o in outputs]

        return predictions

# PyTorch CUDA for embedding generation
class EmbeddingGenerator:
    """Generate embeddings on GPU"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model = self.model.cuda()  # Move to GPU
        self.model.eval()

    @torch.no_grad()  # Disable gradient computation
    @torch.cuda.amp.autocast()  # Mixed precision
    def generate_embeddings_batch(
        self,
        texts: list[str],
        batch_size: int = 256
    ) -> np.ndarray:
        """Generate embeddings in batches on GPU"""

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                device='cuda',
                show_progress_bar=False,
                normalize_embeddings=True
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

# TensorRT for optimized inference
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

class TensorRTPredictor:
    """Optimized inference with TensorRT"""

    def __init__(self, engine_path: str):
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate GPU memory
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Run inference on GPU"""

        # Copy input to GPU
        cuda.memcpy_htod(self.inputs[0], features)

        # Run inference
        self.context.execute_v2(bindings=self.bindings)

        # Copy output from GPU
        output = np.empty(self.outputs[0].shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.outputs[0])

        return output
```

### 7.5 Python 3.14+ GIL-Free Parallel Processing

```python
# Python 3.14+ GIL-free mode for true multi-threading
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Enable GIL-free mode (Python 3.14+)
# Run with: python -X gil=0 script.py

class ParallelFeatureExtractor:
    """Extract features in parallel using GIL-free Python"""

    def __init__(self, num_threads: int = 32):
        self.num_threads = num_threads
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

    def extract_features_parallel(
        self,
        documents: list[Document]
    ) -> list[Features]:
        """Extract features from documents in parallel threads"""

        # With GIL-free mode, CPU-bound tasks can truly parallelize
        futures = [
            self.executor.submit(self._extract_single, doc)
            for doc in documents
        ]

        # Collect results
        features = [f.result() for f in futures]

        return features

    def _extract_single(self, document: Document) -> Features:
        """CPU-intensive feature extraction (benefits from GIL-free)"""

        # Complex feature engineering
        technical_features = self._compute_technical_features(document)
        sentiment_features = self._compute_sentiment_features(document)
        graph_features = self._compute_graph_features(document)

        return Features(
            technical=technical_features,
            sentiment=sentiment_features,
            graph=graph_features
        )

# Parallel data preprocessing with polars (Rust-based, fast)
import polars as pl

def preprocess_market_data_parallel(file_paths: list[str]) -> pl.DataFrame:
    """Load and preprocess market data in parallel"""

    # Polars automatically uses multiple threads
    df = pl.scan_parquet(file_paths).select([
        pl.col("timestamp"),
        pl.col("symbol"),
        pl.col("close").pct_change().alias("returns"),
        pl.col("volume").log1p().alias("log_volume"),
        # Rolling calculations (parallelized internally)
        pl.col("returns").rolling_mean(window_size=20).alias("ma_20"),
        pl.col("returns").rolling_std(window_size=20).alias("std_20"),
    ]).collect(streaming=True)  # Streaming for large datasets

    return df
```

### 7.6 Tier 1 Development Environment Setup

**CRITICAL:** Complete Tier 1 deployment stack for Market Intelligence Engine development. This setup uses Homebrew for latest GCC/binutils, uv for Python environment management, and supports RHEL with OpenShift or Ubuntu Server.

#### 7.6.1 Operating System Selection

**Primary: Red Hat Enterprise Linux (RHEL) 9+ with OpenShift**
- Enterprise-grade stability
- Integrated container orchestration
- 10-year support lifecycle
- Optimized for HPC workloads
- Cost: ~$350-800/year

**Alternative: Ubuntu Server 22.04 LTS**
- Strong community support
- Free for all use cases
- Excellent HPC ecosystem
- 5-year LTS support

#### 7.6.2 Homebrew-Based Toolchain Installation

**Why Homebrew:**
- Latest GCC 15+ with complete C++23 support
- Latest binutils for optimized linking
- Easy version management without conflicts
- Isolated from system packages

```bash
# Install Homebrew on Linux
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ~/.bashrc
source ~/.bashrc

# Install complete C++23 toolchain
brew install gcc@15          # GCC 15 with full C++23
brew install binutils        # Latest GNU binutils
brew install cmake           # CMake 3.28+
brew install ninja           # Ninja build system
brew install open-mpi        # OpenMPI 5.x
brew install upcxx           # UPC++ for PGAS

# Verify installations
gcc-15 --version             # GCC 15.x
ld --version                 # Latest binutils
cmake --version              # CMake 3.28+
mpirun --version             # OpenMPI 5.x
```

#### 7.6.3 Python 3.14+ with uv (Fast Package Manager)

**Why uv:**
- Rust-based: 10-100x faster than pip
- Automatic virtual environment management
- Lockfile support for reproducibility
- Compatible with all pip workflows

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.14 (or 3.13 currently available)
uv python install 3.14       # Via uv (recommended)
# OR
brew install python@3.14     # Via Homebrew

# Create project environment
cd /path/to/BigBrotherAnalytics
uv venv --python 3.14        # Creates .venv
source .venv/bin/activate

# Install dependencies (ultra-fast)
uv pip install -r requirements.txt

# For GIL-free mode (Python 3.14+):
uv python install 3.14t      # 't' = free-threaded build
```

#### 7.6.4 C++23 with OpenMP and OpenMPI

```bash
# OpenMP verification (included in GCC 15)
echo '#include <omp.h>
int main() {
    #pragma omp parallel
    printf("Thread %d\n", omp_get_thread_num());
    return 0;
}' | gcc-15 -fopenmp -x c++ - -o test_omp && ./test_omp

# Test MPI installation
echo '#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Rank %d\n", rank);
    MPI_Finalize();
    return 0;
}' > test_mpi.cpp
mpic++ -std=c++23 test_mpi.cpp -o test_mpi
mpirun -np 4 ./test_mpi

# Test UPC++
echo '#include <upcxx/upcxx.hpp>
int main() {
    upcxx::init();
    std::cout << "Rank " << upcxx::rank_me()
              << " of " << upcxx::rank_n() << std::endl;
    upcxx::finalize();
    return 0;
}' > test_upcxx.cpp
upcxx -std=c++23 test_upcxx.cpp -o test_upcxx
upcxx-run -n 4 ./test_upcxx
```

#### 7.6.5 CUDA 12.3 and PyTorch

```bash
# CUDA Toolkit Installation (RHEL 9)
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install cuda-toolkit-12-3

# CUDA Toolkit Installation (Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update && sudo apt install cuda-toolkit-12-3

# Environment variables
echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA
nvcc --version
nvidia-smi

# Install PyTorch with CUDA support (using uv)
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123

# Install vLLM for high-throughput inference
uv pip install vllm transformers accelerate

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 7.6.6 Infrastructure Automation with Ansible

```bash
# Install Ansible
brew install ansible         # Via Homebrew
# OR
uv pip install ansible      # Via uv

# Verify
ansible --version
```

**Complete Tier 1 Ansible Playbook:**
```yaml
# File: playbooks/tier1-market-intelligence-setup.yml
---
- name: Market Intelligence Engine - Tier 1 Setup
  hosts: localhost
  become: yes
  vars:
    gcc_version: "15"
    python_version: "3.14"
    cuda_version: "12.3"
    project_dir: "/opt/bigbrother"

  tasks:
    - name: Install system dependencies (RHEL)
      dnf:
        name: [git, wget, curl, vim, htop, tmux]
        state: latest
      when: ansible_os_family == "RedHat"

    - name: Install system dependencies (Ubuntu)
      apt:
        name: [git, wget, curl, vim, htop, tmux, build-essential]
        state: latest
        update_cache: yes
      when: ansible_os_family == "Debian"

    - name: Install Homebrew
      shell: |
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      become_user: "{{ ansible_user_id }}"
      args:
        creates: /home/linuxbrew/.linuxbrew/bin/brew

    - name: Install toolchain via Homebrew
      homebrew:
        name: [gcc@15, binutils, cmake, ninja, open-mpi, upcxx]
        state: latest
      become_user: "{{ ansible_user_id }}"

    - name: Install uv
      shell: curl -LsSf https://astral.sh/uv/install.sh | sh
      become_user: "{{ ansible_user_id }}"
      args:
        creates: ~/.cargo/bin/uv

    - name: Install Python
      shell: ~/.cargo/bin/uv python install {{ python_version }}
      become_user: "{{ ansible_user_id }}"

    - name: Install PostgreSQL 16 with extensions
      include_tasks: postgres_setup.yml

    - name: Install Redis
      package:
        name: redis
        state: latest

    - name: Setup project environment
      shell: |
        cd {{ project_dir }}
        ~/.cargo/bin/uv venv --python {{ python_version }}
        source .venv/bin/activate
        ~/.cargo/bin/uv pip install -r requirements.txt
      become_user: "{{ ansible_user_id }}"
```

#### 7.6.7 Complete Environment Verification

```bash
#!/bin/bash
# File: scripts/verify_tier1_environment.sh

echo "=== Market Intelligence Engine - Tier 1 Verification ==="

# GCC C++23
echo -n "GCC C++23: "
gcc-15 --version | head -1

# OpenMP
echo -n "OpenMP: "
echo '#include <omp.h>
int main() { return omp_get_max_threads(); }' | \
gcc-15 -fopenmp -x c++ - -o /tmp/test && /tmp/test && echo "✓" || echo "✗"

# MPI
echo -n "OpenMPI: "
mpirun --version | head -1

# UPC++
echo -n "UPC++: "
upcxx --version

# Python
echo -n "Python 3.14+: "
~/.cargo/bin/uv python list | grep "3.14"

# CUDA
echo -n "CUDA: "
nvcc --version | grep release

# PyTorch CUDA
echo -n "PyTorch CUDA: "
source .venv/bin/activate && python -c "import torch; \
print('✓' if torch.cuda.is_available() else '✗')"

# PostgreSQL
echo -n "PostgreSQL 16+: "
psql --version

# Redis
echo -n "Redis: "
redis-cli --version

# DuckDB
echo -n "DuckDB: "
source .venv/bin/activate && python -c "import duckdb; print(f'✓ {duckdb.__version__}')"

echo ""
echo "=== Verification Complete ==="
```

#### 7.6.8 Development Workflow

```bash
# Activate environment
cd /opt/bigbrother
source .venv/bin/activate

# Update dependencies
uv pip install -r requirements.txt

# Compile C++ components
cd src/cpp
cmake -B build -G Ninja \
    -DCMAKE_CXX_COMPILER=g++-15 \
    -DCMAKE_CXX_STANDARD=23 \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_OPENMP=ON \
    -DENABLE_MPI=ON \
    -DENABLE_CUDA=ON
cmake --build build -j $(nproc)

# Run tests
cd ../.. && pytest tests/

# Start services
docker-compose up -d postgres redis

# Profile performance
perf record -g python scripts/run_pipeline.py
perf report
```

#### 7.6.9 OpenShift Setup (RHEL Only)

```bash
# Install OpenShift Local for development
wget https://developers.redhat.com/content-gateway/rest/mirror/pub/openshift-v4/clients/crc/latest/crc-linux-amd64.tar.xz
tar xf crc-linux-amd64.tar.xz
sudo cp crc-linux-*/crc /usr/local/bin/

# Setup and start
crc setup
crc start

# Access console
crc console
```

#### 7.6.10 Tier 1 Hardware Requirements

**Minimum:**
- CPU: 8+ cores
- RAM: 16GB
- Storage: 500GB SSD
- GPU: Optional

**Recommended:**
- CPU: 16-32 cores (Ryzen 9/i9/Xeon)
- RAM: 32-64GB
- Storage: 1TB NVMe
- GPU: RTX 3090/4090 (24GB VRAM)

**Cost:**
- Hardware: $2,000-5,000 (one-time)
- RHEL subscription: $350-800/year (optional)
- Software: $0 (all open-source)

---

### 7.7 PostgreSQL Database Setup

# 2. Install PostgreSQL extensions
sudo apt install -y \
    postgresql-16-postgis-3 \
    postgresql-16-timescaledb \
    postgresql-16-age

# 3. Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install --upgrade pip
pip install \
    fastapi uvicorn pydantic \
    sqlalchemy psycopg2-binary \
    apache-airflow \
    kafka-python \
    redis \
    scrapy playwright aiohttp requests \
    pandas polars pyarrow duckdb \
    spacy transformers sentence-transformers \
    torch torchvision \
    xgboost lightgbm scikit-learn \
    apache-tika pdfplumber tabula-py \
    prometheus-client grafana-client \
    sentry-sdk \
    pytest pytest-cov black flake8

# 5. Download spaCy model
python -m spacy download en_core_web_lg

# 6. Install Playwright browsers
playwright install

# 7. Initialize PostgreSQL database
sudo -u postgres createdb market_intelligence
sudo -u postgres psql market_intelligence -c "CREATE EXTENSION timescaledb;"
sudo -u postgres psql market_intelligence -c "CREATE EXTENSION age;"
sudo -u postgres psql market_intelligence -c "CREATE EXTENSION vector;"

# 8. Start services
sudo systemctl start postgresql
sudo systemctl start redis-server

# 9. Optional: Install Kafka (or use Redis Streams for lightweight)
# Download from https://kafka.apache.org/downloads
```

---

## 8. Deployment Architecture

### 8.1 Single-Node Deployment (Tier 1 & 2)

```mermaid
graph TB
    subgraph "Single Physical Server"
        subgraph "Application Layer"
            APP1[FastAPI Server<br/>Port 8000]
            APP2[WebSocket Server<br/>Port 8001]
            APP3[Celery Workers<br/>Background Tasks]
        end

        subgraph "Data Layer"
            DB1[(PostgreSQL<br/>Port 5432)]
            DB2[(Redis<br/>Port 6379)]
            DB3[DuckDB Files<br/>/data/duckdb]
        end

        subgraph "Orchestration"
            OR1[Airflow Scheduler]
            OR2[Airflow Webserver<br/>Port 8080]
        end

        subgraph "Monitoring"
            MON1[Prometheus<br/>Port 9090]
            MON2[Grafana<br/>Port 3000]
        end

        subgraph "Data Collectors"
            DC1[API Collectors<br/>Scheduled]
            DC2[Web Scrapers<br/>Scheduled]
        end
    end

    Client[External Clients] --> APP1
    Client --> APP2
    APP1 & APP2 --> DB1
    APP1 & APP2 --> DB2
    APP3 --> DB1
    APP3 --> DB2
    OR1 --> DC1 & DC2
    DC1 & DC2 --> DB1
    DC1 & DC2 --> DB3
    APP1 & APP2 -.Metrics.-> MON1
    MON1 --> MON2

    style DB1 fill:#bbf,stroke:#333,stroke-width:2px
    style APP1 fill:#bfb,stroke:#333,stroke-width:2px
```

### 8.2 Multi-Node Deployment (Tier 3)

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Nginx/HAProxy]
    end

    subgraph "Application Servers"
        APP1[API Server 1]
        APP2[API Server 2]
        APP3[API Server 3]
    end

    subgraph "Worker Nodes"
        WK1[Celery Worker 1<br/>Ingestion]
        WK2[Celery Worker 2<br/>Processing]
        WK3[Celery Worker 3<br/>ML Inference]
    end

    subgraph "Message Queue Cluster"
        K1[Kafka Broker 1]
        K2[Kafka Broker 2]
        K3[Kafka Broker 3]
    end

    subgraph "Database Cluster"
        PG1[(PostgreSQL Primary)]
        PG2[(PostgreSQL Replica 1)]
        PG3[(PostgreSQL Replica 2)]
    end

    subgraph "Cache Cluster"
        R1[(Redis Primary)]
        R2[(Redis Replica)]
    end

    subgraph "Monitoring Stack"
        PROM[Prometheus]
        GRAF[Grafana]
        ALERT[AlertManager]
    end

    Client[External Clients] --> LB
    LB --> APP1 & APP2 & APP3
    APP1 & APP2 & APP3 --> PG1
    APP1 & APP2 & APP3 --> R1
    PG1 --> PG2 & PG3
    R1 --> R2
    WK1 & WK2 & WK3 --> K1 & K2 & K3
    WK1 & WK2 & WK3 --> PG1
    APP1 & APP2 & APP3 -.Metrics.-> PROM
    WK1 & WK2 & WK3 -.Metrics.-> PROM
    PROM --> GRAF
    PROM --> ALERT

    style PG1 fill:#bbf,stroke:#333,stroke-width:3px
    style LB fill:#faa,stroke:#333,stroke-width:2px
```

### 8.3 Containerized Deployment (Docker Compose)

```yaml
version: '3.8'

services:
  # PostgreSQL with extensions
  postgres:
    image: timescale/timescaledb-ha:pg16
    environment:
      POSTGRES_DB: market_intelligence
      POSTGRES_USER: miadmin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U miadmin"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # FastAPI application
  api:
    build:
      context: .
      dockerfile: Dockerfile
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://miadmin:${DB_PASSWORD}@postgres:5432/market_intelligence
      REDIS_URL: redis://redis:6379
    volumes:
      - ./app:/app
      - ./data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Celery worker
  celery_worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A app.celery_app worker --loglevel=info
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://miadmin:${DB_PASSWORD}@postgres:5432/market_intelligence
      REDIS_URL: redis://redis:6379
    volumes:
      - ./app:/app
      - ./data:/data

  # Airflow (simplified)
  airflow:
    image: apache/airflow:2.7.0
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql://miadmin:${DB_PASSWORD}@postgres:5432/airflow
    ports:
      - "8080:8080"
    depends_on:
      - postgres
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

---

## 9. Implementation Guidelines

### 9.1 Development Workflow

```mermaid
graph LR
    A[Plan Feature] --> B[Write Tests]
    B --> C[Implement Code]
    C --> D[Run Tests]
    D --> E{Tests Pass?}
    E -->|No| C
    E -->|Yes| F[Code Review]
    F --> G{Approved?}
    G -->|No| C
    G -->|Yes| H[Merge to Main]
    H --> I[Deploy to Staging]
    I --> J[Integration Tests]
    J --> K{Tests Pass?}
    K -->|No| C
    K -->|Yes| L[Deploy to Production]

    style A fill:#afa,stroke:#333
    style L fill:#afa,stroke:#333
    style E fill:#faa,stroke:#333
    style G fill:#faa,stroke:#333
    style K fill:#faa,stroke:#333
```

### 9.2 Project Structure

```
market-intelligence-engine/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   │
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── predictions.py
│   │   │   ├── events.py
│   │   │   ├── graphs.py
│   │   │   └── sentiment.py
│   │   └── websocket.py
│   │
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── api_collectors.py
│   │   │   ├── scrapers.py
│   │   │   └── validators.py
│   │   ├── processing/
│   │   │   ├── __init__.py
│   │   │   ├── document_parser.py
│   │   │   ├── nlp_pipeline.py
│   │   │   ├── entity_extractor.py
│   │   │   └── event_classifier.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── feature_engineering.py
│   │   │   ├── impact_predictor.py
│   │   │   ├── sentiment_analyzer.py
│   │   │   └── models/
│   │   │       ├── xgboost_model.py
│   │   │       ├── lstm_model.py
│   │   │       └── ensemble.py
│   │   └── graph/
│   │       ├── __init__.py
│   │       ├── graph_builder.py
│   │       ├── graph_analyzer.py
│   │       └── impact_propagation.py
│   │
│   ├── db/                     # Database layer
│   │   ├── __init__.py
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── session.py          # Database sessions
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── prediction_repo.py
│   │   │   ├── event_repo.py
│   │   │   └── entity_repo.py
│   │   └── migrations/         # Alembic migrations
│   │
│   ├── schemas/                # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── prediction.py
│   │   ├── event.py
│   │   ├── entity.py
│   │   └── graph.py
│   │
│   ├── tasks/                  # Celery tasks
│   │   ├── __init__.py
│   │   ├── ingestion_tasks.py
│   │   ├── processing_tasks.py
│   │   └── ml_tasks.py
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── logging.py
│       ├── metrics.py
│       └── helpers.py
│
├── dags/                       # Airflow DAGs
│   ├── daily_data_ingestion.py
│   ├── model_training.py
│   └── data_quality_checks.py
│
├── data/                       # Data storage
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   ├── models/                 # Trained models
│   └── archive/                # Parquet archives
│
├── tests/                      # Tests
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_ingestion.py
│   │   ├── test_processing.py
│   │   ├── test_ml.py
│   │   └── test_graph.py
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_pipeline.py
│   └── fixtures/
│
├── scripts/                    # Utility scripts
│   ├── setup_db.py
│   ├── seed_data.py
│   └── backfill_historical.py
│
├── monitoring/                 # Monitoring configs
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
│
└── docs/                       # Documentation
    ├── api.md
    ├── deployment.md
    └── development.md
```

### 9.3 Phase 1 Implementation Checklist

**Week 1-2: Foundation**
- [ ] Set up development environment
- [ ] Initialize PostgreSQL with extensions (TimescaleDB, AGE, pgvector)
- [ ] Create database schema and migrations
- [ ] Set up DuckDB for analytics
- [ ] Configure Redis cache
- [ ] Implement basic FastAPI structure
- [ ] Set up logging and monitoring

**Week 3-4: Data Ingestion**
- [ ] Implement free API collectors (FRED, Alpha Vantage, Yahoo Finance)
- [ ] Build government API collectors (SEC, Congress, FDA, EPA)
- [ ] Create RSS feed readers
- [ ] Implement rate limiting and retry logic
- [ ] Set up data validation with Pydantic
- [ ] Store raw data in PostgreSQL and Parquet files
- [ ] Create DuckDB views for analytics

**Week 5-6: Basic Processing**
- [ ] Implement document parsing (Apache Tika, pdfplumber)
- [ ] Build basic NLP pipeline with spaCy
- [ ] Create entity extraction (companies, people, locations)
- [ ] Implement simple sentiment analysis
- [ ] Store processed data in PostgreSQL
- [ ] Create embeddings with sentence-transformers

**Week 7-8: Simple ML Models**
- [ ] Implement basic feature engineering
- [ ] Train simple regression model (impact magnitude)
- [ ] Train classification model (impact direction)
- [ ] Create confidence scoring
- [ ] Store predictions in PostgreSQL
- [ ] Build simple prediction API endpoint

**Week 9-10: Graph & API**
- [ ] Implement basic graph builder with Apache AGE
- [ ] Create company-event relationships
- [ ] Build impact propagation logic
- [ ] Complete REST API endpoints
- [ ] Add WebSocket for real-time updates
- [ ] Create API documentation

**Week 11-12: Testing & Refinement**
- [ ] Write unit tests (>80% coverage)
- [ ] Write integration tests
- [ ] Perform load testing
- [ ] Optimize database queries
- [ ] Add monitoring dashboards
- [ ] Document deployment process

---

## 10. Performance Considerations

### 10.1 Performance Targets

```mermaid
graph TB
    subgraph "Latency Targets"
        L1[API Response<br/>< 100ms p95]
        L2[Prediction Generation<br/>< 5 seconds]
        L3[Graph Generation<br/>< 3 seconds]
        L4[Data Ingestion<br/>< 500ms per item]
    end

    subgraph "Throughput Targets"
        T1[Process 100K+<br/>articles/day]
        T2[Handle 10K+<br/>API requests/min]
        T3[Track 10K+<br/>securities]
        T4[Generate 1K+<br/>predictions/hour]
    end

    subgraph "Optimization Strategies"
        O1[Caching with Redis]
        O2[Connection Pooling]
        O3[Database Indexing]
        O4[Async Processing]
        O5[Batch Operations]
        O6[Query Optimization]
    end

    L1 & L2 & L3 & L4 --> O1 & O2 & O3
    T1 & T2 & T3 & T4 --> O4 & O5 & O6

    style L1 fill:#afa,stroke:#333
    style T1 fill:#afa,stroke:#333
    style O1 fill:#ffa,stroke:#333
```

### 10.2 Caching Strategy

```python
# Multi-layer caching strategy

# Layer 1: Application-level cache (Redis)
@cache(ttl=300)  # 5 minutes
async def get_recent_predictions(entity_id: str, min_confidence: float):
    """Cache recent predictions in Redis"""
    key = f"predictions:{entity_id}:{min_confidence}"
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)

    # Query database
    predictions = await db.query_predictions(entity_id, min_confidence)

    # Cache result
    await redis.setex(key, 300, json.dumps(predictions))
    return predictions

# Layer 2: Database-level materialized views
CREATE MATERIALIZED VIEW recent_high_confidence_predictions AS
SELECT * FROM impact_predictions
WHERE created_at > NOW() - INTERVAL '1 hour'
  AND confidence > 0.7
ORDER BY confidence DESC;

-- Refresh every minute
CREATE OR REPLACE FUNCTION refresh_predictions_view()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY recent_high_confidence_predictions;
END;
$$ LANGUAGE plpgsql;

# Layer 3: Query result caching in PostgreSQL
ALTER TABLE impact_predictions SET (
    autovacuum_enabled = true,
    autovacuum_vacuum_scale_factor = 0.05
);

CREATE INDEX CONCURRENTLY idx_predictions_entity_confidence
ON impact_predictions(entity_id, confidence DESC, created_at DESC);
```

### 10.3 Database Optimization

```sql
-- Partitioning strategy for large tables
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT
) PARTITION BY RANGE (time);

-- Create monthly partitions
CREATE TABLE market_data_2025_01 PARTITION OF market_data
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE market_data_2025_02 PARTITION OF market_data
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- Auto-create partitions with pg_partman
SELECT partman.create_parent(
    p_parent_table => 'public.market_data',
    p_control => 'time',
    p_type => 'native',
    p_interval => '1 month',
    p_premake => 3
);

-- Composite indexes for common queries
CREATE INDEX CONCURRENTLY idx_predictions_composite
ON impact_predictions(entity_id, created_at DESC, confidence DESC)
WHERE confidence > 0.5;

-- Partial indexes for active data
CREATE INDEX CONCURRENTLY idx_events_recent
ON events(event_time DESC)
WHERE event_time > NOW() - INTERVAL '7 days';

-- GiST index for pgvector
CREATE INDEX CONCURRENTLY idx_embeddings_hnsw
ON processed_documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### 10.4 Async Processing Patterns

```python
# Async data collection with concurrent requests
import asyncio
import aiohttp
from typing import List

async def fetch_data_source(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch data from a single source"""
    async with session.get(url) as response:
        return await response.json()

async def collect_all_sources(sources: List[str]) -> List[dict]:
    """Collect from all sources concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data_source(session, url) for url in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        return [r for r in results if not isinstance(r, Exception)]

# Batch processing with connection pooling
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def batch_insert_predictions(predictions: List[dict]):
    """Batch insert predictions efficiently"""
    async with async_session() as session:
        async with session.begin():
            # Use COPY for bulk insert (fastest)
            await session.execute(
                text("""
                    COPY impact_predictions (
                        event_id, entity_id, direction,
                        magnitude, confidence
                    ) FROM STDIN WITH CSV
                """),
                predictions
            )
```

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-06 | Initial architecture design document |

---

## References

1. [BigBrotherAnalytics PRD](../PRD.md)
2. [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
3. [TimescaleDB Best Practices](https://docs.timescale.com/timescaledb/latest/how-to-guides/hypertables/)
4. [Apache AGE Documentation](https://age.apache.org/)
5. [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
6. [DuckDB SQL Reference](https://duckdb.org/docs/sql/introduction)

---

**Document Status:** Ready for implementation
**Next Steps:** Begin Phase 1 implementation (Week 1-2: Foundation)
**Contact:** For questions or clarifications, refer to the main PRD
