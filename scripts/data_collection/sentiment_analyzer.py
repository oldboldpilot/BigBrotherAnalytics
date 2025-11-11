#!/usr/bin/env python3
"""
BigBrotherAnalytics: Keyword-Based Sentiment Analysis
Simple, fast sentiment analysis without ML dependencies

This module provides basic sentiment analysis using keyword matching.
No ML libraries required - perfect for lightweight deployment.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 5+: News Ingestion System
"""

import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Keyword-based sentiment analyzer for financial news.

    Uses predefined lists of positive and negative financial keywords
    to calculate sentiment scores without ML dependencies.
    """

    # Positive financial keywords
    POSITIVE_KEYWORDS = {
        'profit', 'profits', 'profitable', 'gain', 'gains', 'growth', 'grow', 'growing',
        'surge', 'surged', 'surges', 'surging', 'bull', 'bullish', 'rally', 'rallied',
        'upgrade', 'upgraded', 'upgrades', 'beat', 'beats', 'beating', 'exceed', 'exceeded',
        'exceeds', 'outperform', 'outperformed', 'outperforming', 'strong', 'stronger',
        'strength', 'success', 'successful', 'positive', 'optimistic', 'optimism',
        'improve', 'improved', 'improving', 'improvement', 'rise', 'rises', 'rising',
        'rose', 'increase', 'increased', 'increasing', 'up', 'higher', 'high', 'record',
        'advance', 'advanced', 'advancing', 'expansion', 'expand', 'expanding', 'boom',
        'breakthrough', 'win', 'wins', 'winning', 'won', 'leader', 'leading', 'innovation',
        'innovative', 'opportunity', 'opportunities', 'recovery', 'recover', 'recovering',
    }

    # Negative financial keywords
    NEGATIVE_KEYWORDS = {
        'loss', 'losses', 'lose', 'losing', 'lost', 'decline', 'declined', 'declining',
        'fall', 'falls', 'falling', 'fell', 'drop', 'dropped', 'dropping', 'drops',
        'bear', 'bearish', 'downgrade', 'downgrades', 'downgraded', 'miss', 'missed',
        'misses', 'missing', 'underperform', 'underperformed', 'underperforming',
        'weak', 'weaker', 'weakness', 'failure', 'fail', 'failed', 'failing', 'fails',
        'negative', 'pessimistic', 'pessimism', 'worsen', 'worsened', 'worsening',
        'worse', 'decrease', 'decreased', 'decreasing', 'down', 'lower', 'low',
        'plunge', 'plunged', 'plunging', 'crash', 'crashed', 'crashing', 'slump',
        'slumped', 'slumping', 'risk', 'risks', 'risky', 'concern', 'concerns',
        'concerned', 'concerning', 'warning', 'warnings', 'warn', 'warned', 'trouble',
        'troubled', 'crisis', 'recession', 'bankruptcy', 'bankrupt', 'deficit',
    }

    # Intensifiers (multiply sentiment impact)
    INTENSIFIERS = {
        'very', 'extremely', 'highly', 'significantly', 'substantially',
        'dramatically', 'sharply', 'rapidly', 'strongly', 'massively',
    }

    # Negation words (flip sentiment)
    NEGATIONS = {
        'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
        'hardly', 'scarcely', 'barely', 'doesn\'t', 'isn\'t', 'wasn\'t',
        'shouldn\'t', 'wouldn\'t', 'couldn\'t', 'won\'t', 'can\'t', 'don\'t',
    }

    def __init__(self):
        """Initialize sentiment analyzer."""
        logger.info("Initialized keyword-based sentiment analyzer")
        logger.info(f"Positive keywords: {len(self.POSITIVE_KEYWORDS)}")
        logger.info(f"Negative keywords: {len(self.NEGATIVE_KEYWORDS)}")

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text using keyword matching.

        Args:
            text: Text to analyze (title + description)

        Returns:
            Dictionary with:
            - score: float between -1.0 and 1.0
            - label: 'positive', 'negative', or 'neutral'
            - positive_keywords: list of matched positive keywords
            - negative_keywords: list of matched negative keywords
            - details: additional analysis details
        """
        if not text or not text.strip():
            return {
                'score': 0.0,
                'label': 'neutral',
                'positive_keywords': [],
                'negative_keywords': [],
                'details': 'Empty text'
            }

        # Normalize text
        text_lower = text.lower()
        words = self._tokenize(text_lower)

        # Find keyword matches
        positive_matches = []
        negative_matches = []
        positive_score = 0.0
        negative_score = 0.0

        for i, word in enumerate(words):
            # Check for intensifier
            intensifier = 1.0
            if i > 0 and words[i-1] in self.INTENSIFIERS:
                intensifier = 1.5

            # Check for negation (within 3 words before)
            negated = False
            for j in range(max(0, i-3), i):
                if words[j] in self.NEGATIONS:
                    negated = True
                    break

            # Check positive keywords
            if word in self.POSITIVE_KEYWORDS:
                if negated:
                    negative_score += 1.0 * intensifier
                    negative_matches.append(word)
                else:
                    positive_score += 1.0 * intensifier
                    positive_matches.append(word)

            # Check negative keywords
            elif word in self.NEGATIVE_KEYWORDS:
                if negated:
                    positive_score += 1.0 * intensifier
                    positive_matches.append(word)
                else:
                    negative_score += 1.0 * intensifier
                    negative_matches.append(word)

        # Calculate final score
        total_keywords = positive_score + negative_score
        if total_keywords == 0:
            raw_score = 0.0
        else:
            raw_score = (positive_score - negative_score) / total_keywords

        # Normalize to -1 to 1 range (already in range, but ensure)
        score = max(-1.0, min(1.0, raw_score))

        # Determine label (using thresholds)
        if score > 0.1:
            label = 'positive'
        elif score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'score': round(score, 3),
            'label': label,
            'positive_keywords': list(set(positive_matches)),
            'negative_keywords': list(set(negative_matches)),
            'details': {
                'positive_score': round(positive_score, 2),
                'negative_score': round(negative_score, 2),
                'total_words': len(words),
                'keyword_density': round(total_keywords / len(words), 3) if words else 0
            }
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment analysis results
        """
        return [self.analyze(text) for text in texts]

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of words (lowercase, alphanumeric only)
        """
        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s\']', ' ', text)

        # Split into words
        words = text.split()

        # Filter out very short words (but keep common negations like "no")
        words = [w for w in words if len(w) >= 2 or w in self.NEGATIONS]

        return words

    def get_sentiment_summary(self, articles: List[Dict]) -> Dict:
        """
        Get summary statistics for a list of analyzed articles.

        Args:
            articles: List of articles with sentiment analysis

        Returns:
            Dictionary with summary statistics
        """
        if not articles:
            return {
                'total_articles': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'avg_score': 0.0,
            }

        positive = sum(1 for a in articles if a.get('sentiment_label') == 'positive')
        negative = sum(1 for a in articles if a.get('sentiment_label') == 'negative')
        neutral = sum(1 for a in articles if a.get('sentiment_label') == 'neutral')

        scores = [a.get('sentiment_score', 0) for a in articles]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            'total_articles': len(articles),
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'avg_score': round(avg_score, 3),
            'positive_pct': round(100.0 * positive / len(articles), 1),
            'negative_pct': round(100.0 * negative / len(articles), 1),
            'neutral_pct': round(100.0 * neutral / len(articles), 1),
        }


# Convenience function for quick analysis
def analyze_sentiment(text: str) -> Dict:
    """
    Quick sentiment analysis function.

    Args:
        text: Text to analyze

    Returns:
        Sentiment analysis result
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(text)


if __name__ == '__main__':
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer()

    test_articles = [
        "Apple stock surges on strong quarterly earnings and positive outlook",
        "Company reports massive losses amid declining sales and weak demand",
        "Analysts remain cautiously optimistic about market recovery prospects",
        "Stock price drops sharply following disappointing revenue miss",
        "Innovative breakthrough drives record profits and market leadership",
        "Economic concerns weigh on investor sentiment as risks increase",
    ]

    print("\nSentiment Analysis Test Results:")
    print("=" * 80)

    for article in test_articles:
        result = analyzer.analyze(article)
        print(f"\nText: {article}")
        print(f"Score: {result['score']:+.3f} | Label: {result['label'].upper()}")
        print(f"Positive: {result['positive_keywords']}")
        print(f"Negative: {result['negative_keywords']}")

    print("\n" + "=" * 80)
