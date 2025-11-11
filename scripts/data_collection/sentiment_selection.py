"""
Power-of-2-Choices Sentiment Selection

Implements the "power of 2 choices" algorithm for sentiment analysis:
- Fetch news from multiple sources (NewsAPI + AlphaVantage)
- For each article, compare sentiment scores from both sources
- Select the score closest to neutral (least extreme) to reduce noise
- If only one source has data, use that source's sentiment

This approach helps reduce false signals from overly optimistic or
pessimistic sources by choosing the more conservative estimate.

Algorithm:
1. Match articles from both sources (by URL or title similarity)
2. For matched articles: Choose sentiment score closest to 0 (least extreme)
3. For unmatched articles: Use the only available sentiment score

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 5+: Multi-Source News Ingestion
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class MultiSourceArticle:
    """
    Article with sentiment scores from multiple sources
    """
    # Article metadata
    article_id: str
    symbol: str
    title: str
    description: str
    content: str
    url: str
    source_name: str
    published_at: int
    fetched_at: int

    # Sentiment scores from different sources
    newsapi_score: Optional[float] = None
    newsapi_label: Optional[str] = None
    newsapi_positive_keywords: Optional[List[str]] = None
    newsapi_negative_keywords: Optional[List[str]] = None

    alphavantage_score: Optional[float] = None
    alphavantage_label: Optional[str] = None
    alphavantage_relevance: Optional[float] = None

    # Selected sentiment (power-of-2-choices result)
    selected_score: Optional[float] = None
    selected_label: Optional[str] = None
    selected_source: Optional[str] = None  # "newsapi", "alphavantage", or "keyword"
    selection_reason: Optional[str] = None


def choose_least_extreme_sentiment(
    score1: float,
    label1: str,
    score2: float,
    label2: str,
    source1: str = "source1",
    source2: str = "source2"
) -> Tuple[float, str, str, str]:
    """
    Power-of-2-Choices: Select sentiment score closest to neutral

    Args:
        score1: Sentiment score from first source (-1.0 to 1.0)
        label1: Sentiment label from first source
        score2: Sentiment score from second source (-1.0 to 1.0)
        label2: Sentiment label from second source
        source1: Name of first source
        source2: Name of second source

    Returns:
        Tuple of (selected_score, selected_label, selected_source, reason)

    Example:
        >>> choose_least_extreme_sentiment(0.8, "positive", 0.3, "positive", "newsapi", "alphavantage")
        (0.3, "positive", "alphavantage", "closer to neutral (|0.3| < |0.8|)")
    """
    abs_score1 = abs(score1)
    abs_score2 = abs(score2)

    if abs_score1 < abs_score2:
        # score1 is closer to neutral
        reason = f"closer to neutral (|{score1:.2f}| < |{score2:.2f}|)"
        return score1, label1, source1, reason
    elif abs_score2 < abs_score1:
        # score2 is closer to neutral
        reason = f"closer to neutral (|{score2:.2f}| < |{score1:.2f}|)"
        return score2, label2, source2, reason
    else:
        # Equal distance - prefer average
        avg_score = (score1 + score2) / 2.0
        avg_label = "neutral" if abs(avg_score) < 0.1 else (label1 if abs(score1) == abs(score2) else label2)
        reason = f"equal distance, using average ({avg_score:.2f})"
        return avg_score, avg_label, "average", reason


def match_articles_by_url(
    newsapi_articles: List[Dict],
    alphavantage_articles: List[Dict]
) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    """
    Match articles from both sources by URL

    Args:
        newsapi_articles: Articles from NewsAPI
        alphavantage_articles: Articles from AlphaVantage

    Returns:
        Tuple of:
        - matched_pairs: List of (newsapi_article, alphavantage_article) tuples
        - newsapi_only: Articles only in NewsAPI
        - alphavantage_only: Articles only in AlphaVantage
    """
    # Create URL lookup maps
    newsapi_by_url = {article['url']: article for article in newsapi_articles}
    alphavantage_by_url = {article['url']: article for article in alphavantage_articles}

    # Find matches
    matched_pairs = []
    newsapi_urls = set(newsapi_by_url.keys())
    alphavantage_urls = set(alphavantage_by_url.keys())
    common_urls = newsapi_urls & alphavantage_urls

    for url in common_urls:
        matched_pairs.append((newsapi_by_url[url], alphavantage_by_url[url]))

    # Find unique articles
    newsapi_only = [newsapi_by_url[url] for url in newsapi_urls - common_urls]
    alphavantage_only = [alphavantage_by_url[url] for url in alphavantage_urls - common_urls]

    logging.info(f"Article matching: {len(matched_pairs)} matched, "
                 f"{len(newsapi_only)} NewsAPI-only, {len(alphavantage_only)} AlphaVantage-only")

    return matched_pairs, newsapi_only, alphavantage_only


def match_articles_by_title_similarity(
    newsapi_articles: List[Dict],
    alphavantage_articles: List[Dict],
    similarity_threshold: float = 0.7
) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    """
    Match articles by title similarity (Jaccard similarity)

    Args:
        newsapi_articles: Articles from NewsAPI
        alphavantage_articles: Articles from AlphaVantage
        similarity_threshold: Minimum Jaccard similarity to consider a match

    Returns:
        Tuple of (matched_pairs, newsapi_only, alphavantage_only)
    """
    def jaccard_similarity(title1: str, title2: str) -> float:
        """Compute Jaccard similarity between two titles"""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0

    matched_pairs = []
    used_newsapi = set()
    used_alphavantage = set()

    # Try to match each NewsAPI article with AlphaVantage article
    for i, newsapi_article in enumerate(newsapi_articles):
        newsapi_title = newsapi_article.get('title', '')
        best_match_idx = None
        best_similarity = 0.0

        for j, alphavantage_article in enumerate(alphavantage_articles):
            if j in used_alphavantage:
                continue

            alphavantage_title = alphavantage_article.get('title', '')
            similarity = jaccard_similarity(newsapi_title, alphavantage_title)

            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = j

        if best_match_idx is not None:
            matched_pairs.append((newsapi_article, alphavantage_articles[best_match_idx]))
            used_newsapi.add(i)
            used_alphavantage.add(best_match_idx)

    # Collect unmatched articles
    newsapi_only = [article for i, article in enumerate(newsapi_articles) if i not in used_newsapi]
    alphavantage_only = [article for i, article in enumerate(alphavantage_articles) if i not in used_alphavantage]

    logging.info(f"Title matching: {len(matched_pairs)} matched, "
                 f"{len(newsapi_only)} NewsAPI-only, {len(alphavantage_only)} AlphaVantage-only")

    return matched_pairs, newsapi_only, alphavantage_only


def create_multi_source_articles(
    newsapi_articles: List[Dict],
    alphavantage_articles: List[Dict],
    symbol: str,
    use_url_matching: bool = True,
    use_title_matching: bool = True
) -> List[MultiSourceArticle]:
    """
    Create multi-source articles with power-of-2-choices sentiment selection

    Args:
        newsapi_articles: Articles from NewsAPI (with sentiment from keyword analysis)
        alphavantage_articles: Articles from AlphaVantage (with AI sentiment)
        symbol: Stock symbol
        use_url_matching: Try to match articles by URL first
        use_title_matching: Try to match articles by title similarity

    Returns:
        List of MultiSourceArticle objects with selected sentiments
    """
    result_articles = []

    # Step 1: Match articles by URL (exact match)
    matched_pairs = []
    newsapi_remaining = newsapi_articles
    alphavantage_remaining = alphavantage_articles

    if use_url_matching:
        matched_pairs, newsapi_remaining, alphavantage_remaining = match_articles_by_url(
            newsapi_articles, alphavantage_articles
        )

    # Step 2: Match remaining articles by title similarity
    if use_title_matching and newsapi_remaining and alphavantage_remaining:
        title_matches, newsapi_only, alphavantage_only = match_articles_by_title_similarity(
            newsapi_remaining, alphavantage_remaining
        )
        matched_pairs.extend(title_matches)
    else:
        newsapi_only = newsapi_remaining
        alphavantage_only = alphavantage_remaining

    # Step 3: Process matched articles with power-of-2-choices
    for newsapi_article, alphavantage_article in matched_pairs:
        # Extract sentiment scores
        newsapi_score = newsapi_article.get('sentiment_score', 0.0)
        newsapi_label = newsapi_article.get('sentiment_label', 'neutral')
        alphavantage_score = alphavantage_article.get('overall_sentiment_score', 0.0)
        alphavantage_label = alphavantage_article.get('overall_sentiment_label', 'Neutral')

        # Apply power-of-2-choices
        selected_score, selected_label, selected_source, reason = choose_least_extreme_sentiment(
            newsapi_score, newsapi_label,
            alphavantage_score, alphavantage_label,
            "newsapi_keyword", "alphavantage_ai"
        )

        # Create multi-source article
        article = MultiSourceArticle(
            article_id=generate_article_id(newsapi_article.get('url', '')),
            symbol=symbol,
            title=newsapi_article.get('title', ''),
            description=newsapi_article.get('description', ''),
            content=newsapi_article.get('content', ''),
            url=newsapi_article.get('url', ''),
            source_name=newsapi_article.get('source_name', ''),
            published_at=newsapi_article.get('published_at', 0),
            fetched_at=int(datetime.now().timestamp()),
            newsapi_score=newsapi_score,
            newsapi_label=newsapi_label,
            newsapi_positive_keywords=newsapi_article.get('positive_keywords', []),
            newsapi_negative_keywords=newsapi_article.get('negative_keywords', []),
            alphavantage_score=alphavantage_score,
            alphavantage_label=alphavantage_label,
            alphavantage_relevance=get_ticker_relevance(alphavantage_article, symbol),
            selected_score=selected_score,
            selected_label=selected_label,
            selected_source=selected_source,
            selection_reason=reason
        )
        result_articles.append(article)

    # Step 4: Add NewsAPI-only articles (use keyword sentiment)
    for newsapi_article in newsapi_only:
        article = MultiSourceArticle(
            article_id=generate_article_id(newsapi_article.get('url', '')),
            symbol=symbol,
            title=newsapi_article.get('title', ''),
            description=newsapi_article.get('description', ''),
            content=newsapi_article.get('content', ''),
            url=newsapi_article.get('url', ''),
            source_name=newsapi_article.get('source_name', ''),
            published_at=newsapi_article.get('published_at', 0),
            fetched_at=int(datetime.now().timestamp()),
            newsapi_score=newsapi_article.get('sentiment_score', 0.0),
            newsapi_label=newsapi_article.get('sentiment_label', 'neutral'),
            newsapi_positive_keywords=newsapi_article.get('positive_keywords', []),
            newsapi_negative_keywords=newsapi_article.get('negative_keywords', []),
            selected_score=newsapi_article.get('sentiment_score', 0.0),
            selected_label=newsapi_article.get('sentiment_label', 'neutral'),
            selected_source="newsapi_keyword",
            selection_reason="only available from NewsAPI"
        )
        result_articles.append(article)

    # Step 5: Add AlphaVantage-only articles (use AI sentiment)
    for alphavantage_article in alphavantage_only:
        article = MultiSourceArticle(
            article_id=generate_article_id(alphavantage_article.get('url', '')),
            symbol=symbol,
            title=alphavantage_article.get('title', ''),
            description=alphavantage_article.get('summary', ''),
            content="",
            url=alphavantage_article.get('url', ''),
            source_name=alphavantage_article.get('source', ''),
            published_at=parse_alphavantage_timestamp(alphavantage_article.get('time_published', '')),
            fetched_at=int(datetime.now().timestamp()),
            alphavantage_score=alphavantage_article.get('overall_sentiment_score', 0.0),
            alphavantage_label=alphavantage_article.get('overall_sentiment_label', 'Neutral'),
            alphavantage_relevance=get_ticker_relevance(alphavantage_article, symbol),
            selected_score=alphavantage_article.get('overall_sentiment_score', 0.0),
            selected_label=alphavantage_article.get('overall_sentiment_label', 'Neutral'),
            selected_source="alphavantage_ai",
            selection_reason="only available from AlphaVantage"
        )
        result_articles.append(article)

    logging.info(f"Created {len(result_articles)} multi-source articles for {symbol}")
    return result_articles


def generate_article_id(url: str) -> str:
    """Generate unique article ID from URL"""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def parse_alphavantage_timestamp(time_str: str) -> int:
    """
    Parse AlphaVantage timestamp to Unix timestamp

    Args:
        time_str: Timestamp in format "20231115T093000"

    Returns:
        Unix timestamp (seconds since epoch)
    """
    try:
        dt = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
        return int(dt.timestamp())
    except ValueError:
        logging.warning(f"Failed to parse AlphaVantage timestamp: {time_str}")
        return 0


def get_ticker_relevance(alphavantage_article: Dict, symbol: str) -> Optional[float]:
    """
    Extract ticker-specific relevance score from AlphaVantage article

    Args:
        alphavantage_article: AlphaVantage article dict
        symbol: Stock symbol to get relevance for

    Returns:
        Relevance score (0.0 to 1.0) or None if not found
    """
    ticker_sentiment_list = alphavantage_article.get('ticker_sentiment', [])
    for ticker_sent in ticker_sentiment_list:
        if ticker_sent.get('ticker', '').upper() == symbol.upper():
            return float(ticker_sent.get('relevance_score', 0.0))
    return None


def convert_to_db_format(article: MultiSourceArticle) -> Dict:
    """
    Convert MultiSourceArticle to database format

    Returns:
        Dictionary ready for DuckDB insertion
    """
    return {
        'article_id': article.article_id,
        'symbol': article.symbol,
        'title': article.title,
        'description': article.description,
        'content': article.content,
        'url': article.url,
        'source_name': article.source_name,
        'published_at': article.published_at,
        'fetched_at': article.fetched_at,

        # Selected sentiment (power-of-2-choices result)
        'sentiment_score': article.selected_score,
        'sentiment_label': article.selected_label,
        'sentiment_source': article.selected_source,

        # NewsAPI sentiment
        'newsapi_sentiment_score': article.newsapi_score,
        'newsapi_sentiment_label': article.newsapi_label,
        'positive_keywords': ','.join(article.newsapi_positive_keywords or []),
        'negative_keywords': ','.join(article.newsapi_negative_keywords or []),

        # AlphaVantage sentiment
        'alphavantage_sentiment_score': article.alphavantage_score,
        'alphavantage_sentiment_label': article.alphavantage_label,
        'alphavantage_relevance': article.alphavantage_relevance,

        # Selection metadata
        'selection_reason': article.selection_reason
    }
