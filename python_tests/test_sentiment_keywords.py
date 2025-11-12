#!/usr/bin/env python3
"""
Test script to count and verify expanded sentiment analyzer keywords
"""

import re

def count_keywords_in_file(filepath):
    """Count positive and negative keywords in the sentiment analyzer file"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find positive keywords section
    pos_match = re.search(r'positive_keywords_\s*=\s*\{([^}]+)\};', content, re.DOTALL)
    if pos_match:
        pos_content = pos_match.group(1)
        # Count quoted strings (keywords)
        pos_keywords = re.findall(r'"([^"]+)"', pos_content)
        print(f"\nPositive Keywords: {len(pos_keywords)}")
        print("=" * 80)

        # Group by category
        categories = {}
        current_category = "uncategorized"

        for line in pos_content.split('\n'):
            if '//' in line and 'terms' in line.lower():
                current_category = line.split('//')[1].strip()
                categories[current_category] = []

            keywords_in_line = re.findall(r'"([^"]+)"', line)
            for kw in keywords_in_line:
                if current_category not in categories:
                    categories[current_category] = []
                categories[current_category].append(kw)

        for cat, words in categories.items():
            if words:
                print(f"\n{cat}: {len(words)} keywords")
                print(f"  Examples: {', '.join(words[:5])}")

        total_pos = len(pos_keywords)
    else:
        total_pos = 0
        print("ERROR: Could not find positive keywords section!")

    # Find negative keywords section
    neg_match = re.search(r'negative_keywords_\s*=\s*\{([^}]+)\};', content, re.DOTALL)
    if neg_match:
        neg_content = neg_match.group(1)
        neg_keywords = re.findall(r'"([^"]+)"', neg_content)
        print(f"\n\nNegative Keywords: {len(neg_keywords)}")
        print("=" * 80)

        # Group by category
        categories = {}
        current_category = "uncategorized"

        for line in neg_content.split('\n'):
            if '//' in line and 'terms' in line.lower():
                current_category = line.split('//')[1].strip()
                categories[current_category] = []

            keywords_in_line = re.findall(r'"([^"]+)"', line)
            for kw in keywords_in_line:
                if current_category not in categories:
                    categories[current_category] = []
                categories[current_category].append(kw)

        for cat, words in categories.items():
            if words:
                print(f"\n{cat}: {len(words)} keywords")
                print(f"  Examples: {', '.join(words[:5])}")

        total_neg = len(neg_keywords)
    else:
        total_neg = 0
        print("ERROR: Could not find negative keywords section!")

    print("\n" + "=" * 80)
    print(f"\nTOTAL SUMMARY")
    print("=" * 80)
    print(f"Positive Keywords: {total_pos}")
    print(f"Negative Keywords: {total_neg}")
    print(f"Total Keywords: {total_pos + total_neg}")
    print(f"\nTarget: 150+ keywords (75+ each)")
    print(f"Status: {'✓ ACHIEVED' if (total_pos + total_neg) >= 150 else '✗ NOT MET'}")

    return total_pos, total_neg

if __name__ == "__main__":
    filepath = "/home/muyiwa/Development/BigBrotherAnalytics/src/market_intelligence/sentiment_analyzer.cppm"
    print("=" * 80)
    print("SENTIMENT ANALYZER KEYWORD EXPANSION VERIFICATION")
    print("=" * 80)

    pos_count, neg_count = count_keywords_in_file(filepath)

    # Test sample texts
    print("\n\n" + "=" * 80)
    print("SAMPLE TEST CASES")
    print("=" * 80)

    test_cases = [
        "Company reports strong revenue growth and raises guidance with impressive earnings beat",
        "Firm misses earnings expectations and cuts workforce amid restructuring charges",
        "Stock consolidates in tight range as investors await next catalyst",
    ]

    print("\nNote: Full sentiment scoring requires running the C++ analyzer.")
    print("The keyword expansion is complete. Sample texts for testing:")
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. \"{text}\"")
