#!/usr/bin/env python3
"""
BigBrotherAnalytics: News Ingestion Script (Python Wrapper)
Fetches news using C++ NewsAPI collector and stores in DuckDB

This script orchestrates the C++ news ingestion module via Python bindings.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 5+: News Ingestion System
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import duckdb

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import C++ bindings
try:
    from build import news_ingestion_py
    HAS_CPP_MODULE = True
except ImportError:
    HAS_CPP_MODULE = False
    logging.warning("C++ news ingestion module not available - using Python fallback")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Python Fallback Sentiment Analyzer
# ============================================================================

def simple_sentiment(text: str):
    """
    Simple keyword-based sentiment analysis (Python fallback)

    Matches the expanded keyword set from C++ sentiment analyzer.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (score, label, positive_keywords, negative_keywords)
    """
    if not text:
        return 0.0, 'neutral', [], []

    text_lower = text.lower()

    # Expanded positive keywords (matching C++ implementation)
    positive_words = [
        # Strong positive
        'surge', 'soar', 'rally', 'jump', 'climb', 'spike', 'breakout', 'boom',
        'excellent', 'outstanding', 'exceptional', 'remarkable', 'impressive',

        # Earnings & performance
        'beat', 'exceed', 'outperform', 'strong', 'robust', 'solid', 'record',
        'growth', 'accelerate', 'expand', 'increase', 'gain', 'profit', 'revenue',

        # Market sentiment
        'bullish', 'optimistic', 'positive', 'upgrade', 'raise', 'boost',
        'momentum', 'upside', 'breakthrough', 'success',

        # Fundamentals
        'earnings', 'guidance', 'margins', 'cash flow', 'dividend', 'buyback',
        'innovation', 'competitive', 'market share', 'efficiency'
    ]

    # Expanded negative keywords (matching C++ implementation)
    negative_words = [
        # Strong negative
        'plunge', 'crash', 'tumble', 'sink', 'collapse', 'plummet', 'selloff', 'slump',
        'terrible', 'awful', 'disastrous', 'catastrophic', 'severe',

        # Earnings & performance
        'miss', 'disappoint', 'weak', 'soft', 'decline', 'fall', 'loss', 'deficit',
        'slowdown', 'decelerate', 'contract', 'decrease', 'cut', 'reduce',

        # Market sentiment
        'bearish', 'pessimistic', 'negative', 'downgrade', 'lower', 'warning',
        'concern', 'risk', 'uncertainty', 'volatility',

        # Issues & problems
        'crisis', 'recession', 'bankruptcy', 'lawsuit', 'probe', 'scandal',
        'layoff', 'restructure', 'writedown', 'impairment', 'default'
    ]

    # Count keyword matches
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    # Extract matched keywords
    pos_kw = [w for w in positive_words if w in text_lower]
    neg_kw = [w for w in negative_words if w in text_lower]

    # Calculate score
    total = pos_count + neg_count
    if total == 0:
        return 0.0, 'neutral', [], []

    # Score formula: balance between keyword counts and text length
    # Formula: (positive - negative) / sqrt(total_keywords) normalized by text length
    word_count = max(len(text_lower.split()), 1)
    score = (pos_count - neg_count) / (total ** 0.5)  # Dampened by sqrt
    score = score / (word_count ** 0.3)  # Light normalization by text length
    score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]

    # Determine label
    if score > 0.1:
        label = 'positive'
    elif score < -0.1:
        label = 'negative'
    else:
        label = 'neutral'

    return score, label, pos_kw, neg_kw


def load_api_key() -> Optional[str]:
    """
    Load NewsAPI key from configs/api_keys.yaml

    Returns:
        API key or None
    """
    try:
        import yaml
        config_path = BASE_DIR / 'configs' / 'api_keys.yaml'

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('news_api', {}).get('api_key')
    except Exception as e:
        logger.error(f"Error loading API key: {e}")

    return None


def get_traded_symbols(db_path: str, limit: int = 10) -> List[str]:
    """
    Get list of traded symbols from database

    Args:
        db_path: Path to DuckDB database
        limit: Maximum number of symbols to fetch

    Returns:
        List of stock symbols
    """
    try:
        conn = duckdb.connect(str(db_path), read_only=True)

        # Get symbols from positions table (current portfolio)
        result = conn.execute("""
            SELECT DISTINCT symbol
            FROM positions
            WHERE is_bot_managed = 1
            ORDER BY symbol
            LIMIT ?
        """, [limit]).fetchall()

        conn.close()

        symbols = [row[0] for row in result]
        logger.info(f"Fetched {len(symbols)} symbols from database")

        return symbols

    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return []


def run_cpp_ingestion(api_key: str, symbols: List[str], db_path: str) -> bool:
    """
    Run news ingestion using C++ module

    Args:
        api_key: NewsAPI key
        symbols: List of symbols to fetch news for
        db_path: Path to DuckDB database

    Returns:
        True if successful
    """
    if not HAS_CPP_MODULE:
        logger.error("C++ module not available")
        return False

    try:
        # Configure NewsAPI
        config = news_ingestion_py.NewsAPIConfig()
        config.api_key = api_key
        config.requests_per_day = 100
        config.lookback_days = 7
        config.timeout_seconds = 30

        # Create collector
        collector = news_ingestion_py.NewsAPICollector(config)

        # Calculate date range (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        logger.info(f"Fetching news from {from_date} to {to_date}")

        # Fetch news for all symbols
        result = collector.fetch_news_batch(symbols, from_date, to_date)

        if not result:
            logger.error("Failed to fetch news")
            return False

        # Extract articles from result
        all_articles = []
        for symbol, articles in result.value().items():
            logger.info(f"  {symbol}: {len(articles)} articles")
            all_articles.extend(articles)

        logger.info(f"Total articles fetched: {len(all_articles)}")

        # Store articles in database
        if all_articles:
            store_result = collector.store_articles(all_articles, db_path)

            if store_result:
                logger.info("Successfully stored articles in database")
                return True
            else:
                logger.error(f"Failed to store articles: {store_result.error().message}")
                return False
        else:
            logger.warning("No articles to store")
            return True

    except Exception as e:
        logger.error(f"Error during C++ ingestion: {e}", exc_info=True)
        return False


def run_python_fallback_ingestion(api_key: str, symbols: List[str], db_path: str) -> bool:
    """
    Fallback: Run news ingestion using pure Python

    Args:
        api_key: NewsAPI key
        symbols: List of symbols to fetch news for
        db_path: Path to DuckDB database

    Returns:
        True if successful
    """
    logger.info("Using Python fallback implementation")

    try:
        from newsapi import NewsApiClient
        import hashlib
        from datetime import datetime, timedelta

        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=api_key)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        logger.info(f"Fetching news from {from_date} to {to_date}")

        # Connect to database
        conn = duckdb.connect(str(db_path))

        total_articles = 0

        # Fetch news for each symbol
        for symbol in symbols:
            try:
                logger.info(f"Fetching news for {symbol}...")

                # Fetch from NewsAPI
                articles = newsapi.get_everything(
                    q=symbol,
                    from_param=from_date,
                    to=to_date,
                    language='en',
                    sort_by='publishedAt',
                    page_size=20
                )

                if articles.get('status') != 'ok':
                    logger.warning(f"  NewsAPI error for {symbol}: {articles.get('message')}")
                    continue

                article_list = articles.get('articles', [])
                logger.info(f"  Found {len(article_list)} articles for {symbol}")

                # Store each article
                for article in article_list:
                    # Generate article ID
                    article_id = hashlib.md5(article.get('url', '').encode()).hexdigest()

                    # Extract data
                    title = article.get('title', '')
                    description = article.get('description', '')
                    content = article.get('content', '')
                    url = article.get('url', '')
                    author = article.get('author', '')
                    published_at = article.get('publishedAt', '')

                    source = article.get('source', {})
                    source_name = source.get('name', '')
                    source_id = source.get('id', '')

                    # Sentiment analysis
                    text_to_analyze = f"{title} {description}"
                    score, label, pos_kw, neg_kw = simple_sentiment(text_to_analyze)

                    # Insert into database (with deduplication)
                    try:
                        conn.execute("""
                            INSERT INTO news_articles (
                                article_id, symbol, title, description, content, url,
                                source_name, source_id, author, published_at, fetched_at,
                                sentiment_score, sentiment_label,
                                positive_keywords, negative_keywords
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
                            ON CONFLICT (article_id) DO NOTHING
                        """, [
                            article_id, symbol, title, description, content, url,
                            source_name, source_id, author, published_at,
                            score, label, pos_kw, neg_kw
                        ])

                        total_articles += 1

                    except Exception as e:
                        logger.debug(f"  Skipping duplicate or error: {e}")

                # Rate limiting
                import time
                time.sleep(1.0)  # 1 second between requests

            except Exception as e:
                logger.error(f"  Error fetching news for {symbol}: {e}")
                continue

        conn.commit()
        conn.close()

        logger.info(f"Stored {total_articles} articles in database")
        return True

    except Exception as e:
        logger.error(f"Error during Python fallback ingestion: {e}", exc_info=True)
        return False


def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("BigBrotherAnalytics - News Ingestion")
    logger.info("=" * 80)

    # Load API key
    api_key = load_api_key() or os.getenv('NEWS_API_KEY')

    if not api_key:
        logger.error("NewsAPI key not found!")
        logger.error("  Add to configs/api_keys.yaml or set NEWS_API_KEY environment variable")
        logger.error("  Get free API key at: https://newsapi.org/")
        sys.exit(1)

    logger.info(f"API key loaded: {api_key[:10]}...")

    # Database path
    db_path = str(BASE_DIR / 'data' / 'bigbrother.duckdb')

    # Get symbols from database
    symbols = get_traded_symbols(db_path, limit=10)

    if not symbols:
        logger.error("No symbols found in database")
        sys.exit(1)

    logger.info(f"Symbols to fetch: {', '.join(symbols)}")

    # Run ingestion (C++ or Python fallback)
    if HAS_CPP_MODULE:
        success = run_cpp_ingestion(api_key, symbols, db_path)
    else:
        success = run_python_fallback_ingestion(api_key, symbols, db_path)

    if success:
        logger.info("\n" + "=" * 80)
        logger.info("News ingestion completed successfully!")
        logger.info("=" * 80)
        sys.exit(0)
    else:
        logger.error("\n" + "=" * 80)
        logger.error("News ingestion failed!")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()
