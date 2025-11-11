#!/usr/bin/env python3
"""
Manual sentiment analysis simulation to verify keyword effectiveness
"""

import re

def load_keywords(filepath):
    """Load positive and negative keywords from the C++ file"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract positive keywords
    pos_match = re.search(r'positive_keywords_\s*=\s*\{([^}]+)\};', content, re.DOTALL)
    positive = set(re.findall(r'"([^"]+)"', pos_match.group(1))) if pos_match else set()

    # Extract negative keywords
    neg_match = re.search(r'negative_keywords_\s*=\s*\{([^}]+)\};', content, re.DOTALL)
    negative = set(re.findall(r'"([^"]+)"', neg_match.group(1))) if neg_match else set()

    return positive, negative

def tokenize(text):
    """Tokenize text into lowercase words (mimics C++ tokenizer)"""
    words = []
    word = ""
    for c in text.lower():
        if c.isalnum() or c == "'":
            word += c
        elif word and len(word) >= 2:
            words.append(word)
            word = ""
    if word and len(word) >= 2:
        words.append(word)
    return words

def analyze_sentiment(text, positive_kw, negative_kw):
    """Simulate sentiment analysis (simplified version of C++ logic)"""
    words = tokenize(text)

    pos_matches = []
    neg_matches = []

    for word in words:
        if word in positive_kw:
            pos_matches.append(word)
        elif word in negative_kw:
            neg_matches.append(word)

    pos_score = len(pos_matches)
    neg_score = len(neg_matches)
    total = pos_score + neg_score

    if total == 0:
        score = 0.0
        label = "neutral"
    else:
        score = (pos_score - neg_score) / total
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"

    return {
        'score': score,
        'label': label,
        'positive_score': pos_score,
        'negative_score': neg_score,
        'positive_keywords': pos_matches,
        'negative_keywords': neg_matches,
        'total_words': len(words)
    }

def print_result(text, result):
    """Pretty print analysis result"""
    print(f"\nText: \"{text}\"")
    print(f"Score: {result['score']:.3f} ({result['label']})")
    print(f"Positive: {result['positive_score']} | Negative: {result['negative_score']}")
    print(f"Positive Keywords: {', '.join(result['positive_keywords']) if result['positive_keywords'] else 'none'}")
    print(f"Negative Keywords: {', '.join(result['negative_keywords']) if result['negative_keywords'] else 'none'}")
    print(f"Keyword Density: {(result['positive_score'] + result['negative_score']) / result['total_words']:.3f}")
    print("-" * 80)

if __name__ == "__main__":
    filepath = "/home/muyiwa/Development/BigBrotherAnalytics/src/market_intelligence/sentiment_analyzer.cppm"
    positive_kw, negative_kw = load_keywords(filepath)

    print("=" * 80)
    print("SENTIMENT ANALYZER TEST (Manual Python Simulation)")
    print("=" * 80)
    print(f"Loaded {len(positive_kw)} positive keywords")
    print(f"Loaded {len(negative_kw)} negative keywords")
    print("=" * 80)

    test_cases = [
        # Very positive - financial performance
        "Company reports strong revenue growth and raises guidance with impressive earnings beat",

        # Very positive - market sentiment
        "Stock surges in bullish rally with strong momentum and breakout above resistance",

        # Very positive - company performance
        "Firm accelerates expansion with strategic acquisition and competitive advantages",

        # Very negative - financial performance
        "Firm misses earnings expectations and cuts workforce amid restructuring charges",

        # Very negative - market sentiment
        "Stock plunges in bearish selloff with panic selling and breakdown below support",

        # Very negative - company performance
        "Company struggles with deteriorating margins and faces bankruptcy risks",

        # Mixed sentiment
        "Despite strong revenue growth, company warns of margin pressure and rising costs",

        # Neutral
        "Stock consolidates in tight range as investors await next catalyst",

        # Economic indicators - positive
        "Economic expansion accelerates with robust GDP growth and stimulus measures",

        # Economic indicators - negative
        "Recession fears mount as inflation surges and economy shows signs of contraction",

        # Analyst actions - positive
        "Analysts upgrade stock to overweight citing strong fundamentals and upside potential",

        # Analyst actions - negative
        "Downgrade to underweight as analysts cite deteriorating fundamentals and downside risks",
    ]

    for text in test_cases:
        result = analyze_sentiment(text, positive_kw, negative_kw)
        print_result(text, result)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nNote: This is a simplified simulation. The actual C++ analyzer includes:")
    print("  - Intensifier handling (very, extremely, etc.)")
    print("  - Negation detection (not good -> negative)")
    print("  - More sophisticated scoring")
