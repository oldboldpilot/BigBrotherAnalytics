#!/bin/bash
#
# BigBrother Trading Dashboard Launcher
#

echo "=========================================="
echo "BigBrother Trading Dashboard"
echo "=========================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if database exists
DB_PATH="$PROJECT_ROOT/data/bigbrother.duckdb"
if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database not found at $DB_PATH"
    echo "Please ensure the database exists before running the dashboard."
    exit 1
fi

echo "Database: $DB_PATH"
echo "Status: Found ✓"
echo ""

# Check for sample data
echo "Checking for data..."
cd "$PROJECT_ROOT"

POSITION_COUNT=$(uv run python -c "import duckdb; conn = duckdb.connect('data/bigbrother.duckdb', read_only=True); print(conn.execute('SELECT COUNT(*) FROM positions').fetchone()[0]); conn.close()" 2>/dev/null)

if [ "$POSITION_COUNT" -eq "0" ]; then
    echo "Warning: No positions found in database"
    echo ""
    read -p "Would you like to generate sample data? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Generating sample data..."
        uv run python dashboard/generate_sample_data.py
        echo ""
    fi
else
    echo "Positions: $POSITION_COUNT ✓"
    echo ""
fi

# Start the dashboard
echo "Starting Streamlit dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

cd "$SCRIPT_DIR"
uv run streamlit run app.py
