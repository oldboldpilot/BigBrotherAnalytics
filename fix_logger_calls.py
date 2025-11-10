#!/usr/bin/env python3
"""
Fix all Logger calls to use std::format externally.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import re
from pathlib import Path

def fix_logger_call(match):
    """Convert Logger::getInstance().method("format {}", args) to Logger::getInstance().method(std::format("format {}", args))"""
    indent = match.group(1)
    namespace = match.group(2)  # might be empty or "utils::"
    method = match.group(3)
    content = match.group(4)
    
    # If already wrapped in std::format, skip
    if 'std::format(' in content:
        return match.group(0)
    
    # Extract format string and arguments
    # Pattern: "string...", args
    parts = content.split('", ', 1)
    if len(parts) == 2:
        format_str = parts[0] + '"'
        rest_args = parts[1].rstrip(');')
        # Wrap in std::format
        return f'{indent}{namespace}Logger::getInstance().{method}(std::format({format_str}, {rest_args}));'
    else:
        # No format arguments, just string
        return match.group(0)

def process_file(filepath):
    """Process a single file."""
    print(f"Processing {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Pattern matches:
    # Logger::getInstance().info("format {}", arg)
    # Logger::getInstance().debug("format {} {}", arg1, arg2)
    # utils::Logger::getInstance().warn("format", args...)
    
    # Multi-line logger calls pattern
    pattern = r'([ \t]*)((?:utils::)?Logger::getInstance\(\)\.(info|debug|warn|error|critical)\()([^;]+\);)'
    
    # Only fix lines with format placeholders {}
    def replace_if_has_format(match):
        if '{' in match.group(4):
            return fix_logger_call(match)
        return match.group(0)
    
    content = re.sub(pattern, replace_if_has_format, content, flags=re.MULTILINE)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Updated {filepath}")
        return True
    else:
        print(f"  - No changes needed for {filepath}")
        return False

def main():
    """Main entry point."""
    root = Path('/home/muyiwa/Development/BigBrotherAnalytics/src')
    
    # Find all .cppm and .cpp files
    files = list(root.rglob('*.cppm')) + list(root.rglob('*.cpp'))
    
    updated = 0
    for filepath in files:
        if process_file(filepath):
            updated += 1
    
    print(f"\n✓ Updated {updated} files")

if __name__ == '__main__':
    main()
