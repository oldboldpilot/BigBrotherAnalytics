#!/bin/bash
cd /home/muyiwa/Development/BigBrotherAnalytics/build
ninja -v 2>&1 | tee /home/muyiwa/Development/BigBrotherAnalytics/build_verbose.log
echo "Build exit code: $?"
echo "=== Last 50 lines of build output ==="
tail -50 /home/muyiwa/Development/BigBrotherAnalytics/build_verbose.log
