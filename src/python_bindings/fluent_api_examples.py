#!/usr/bin/env python3
"""
DuckDB Fluent API Examples

Demonstrates the fluent interface pattern for DuckDB bindings,
matching the Schwab API design. Shows method chaining for both
configuration and query building.

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09

Tagged: PYTHON_BINDINGS, FLUENT_API, EXAMPLES
"""

import bigbrother_duckdb as duckdb

# =============================================================================
# 1. FLUENT CONFIGURATION - Configure connection with method chaining
# =============================================================================

print("\n=== Example 1: Fluent Configuration ===")

# Traditional approach (still supported for backward compatibility)
db = duckdb.Connection("data/bigbrother.duckdb")
# db.set_read_only(False)
# db.set_max_memory(1024 * 1024 * 1024)

# Fluent approach (new - method chaining)
db = duckdb.Connection("data/bigbrother.duckdb") \
    .set_read_only(False) \
    .set_max_memory(2 * 1024 * 1024 * 1024) \
    .enable_auto_checkpoint(True) \
    .set_thread_pool_size(4) \
    .enable_logging(True)

print("Configured database connection with fluent API")
print("  - Read-only: False")
print("  - Max memory: 2GB")
print("  - Auto-checkpoint: Enabled")
print("  - Thread pool: 4 threads")
print("  - Logging: Enabled")


# =============================================================================
# 2. FLUENT QUERY BUILDER - Build SQL with method chaining
# =============================================================================

print("\n=== Example 2: Fluent Query Builder ===")

# Build a query using fluent interface
result = db.query() \
    .select(["symbol", "price", "volume"]) \
    .from_table("quotes") \
    .where("price > 100") \
    .order_by("volume", "DESC") \
    .limit(10) \
    .execute()

print("Built query:")
print("  SELECT symbol, price, volume")
print("  FROM quotes")
print("  WHERE price > 100")
print("  ORDER BY volume DESC")
print("  LIMIT 10")


# =============================================================================
# 3. ADVANCED QUERY BUILDER - Complex queries with multiple conditions
# =============================================================================

print("\n=== Example 3: Advanced Query Builder ===")

# Build complex query with multiple WHERE clauses
query_string = db.query() \
    .select(["id", "ticker", "sector", "market_cap"]) \
    .from_table("companies") \
    .where("market_cap > 1000000000") \
    .where("sector = 'Technology'") \
    .or_where("sector = 'Healthcare'") \
    .order_by("market_cap", "DESC") \
    .limit(20) \
    .build()  # Use .build() to get SQL string without executing

print("Generated SQL:")
print(query_string)


# =============================================================================
# 4. PAGINATION - Using limit and offset with fluent API
# =============================================================================

print("\n=== Example 4: Pagination ===")

# Page 1 (items 1-20)
page1 = db.query() \
    .select(["id", "name", "value"]) \
    .from_table("records") \
    .order_by("id", "ASC") \
    .limit(20) \
    .offset(0) \
    .execute()

# Page 2 (items 21-40)
page2 = db.query() \
    .select(["id", "name", "value"]) \
    .from_table("records") \
    .order_by("id", "ASC") \
    .limit(20) \
    .offset(20) \
    .execute()

print("Page 1: Retrieved records with offset=0, limit=20")
print("Page 2: Retrieved records with offset=20, limit=20")


# =============================================================================
# 5. FLUENT EMPLOYMENT DATA ACCESSOR
# =============================================================================

print("\n=== Example 5: Employment Data Accessor ===")

# Access employment data with fluent interface
employment_data = db.employment() \
    .for_sector("Technology") \
    .between_dates("2024-01-01", "2025-01-01") \
    .limit(100) \
    .get()

print("Employment data query:")
print("  Sector: Technology")
print("  Date range: 2024-01-01 to 2025-01-01")
print("  Limit: 100 records")


# =============================================================================
# 6. EMPLOYMENT DATA - Date range variations
# =============================================================================

print("\n=== Example 6: Employment Data - Date Variations ===")

# From a specific date to latest
recent_employment = db.employment() \
    .from_date("2024-06-01") \
    .limit(50) \
    .get()

print("Recent employment data (from June 2024):")
print("  From date: 2024-06-01")
print("  To date: Latest available")

# Specific date range for sector analysis
sector_history = db.employment() \
    .for_sector("Healthcare") \
    .between_dates("2023-01-01", "2024-12-31") \
    .limit(200) \
    .get()

print("\nSector history (Healthcare 2023-2024):")
print("  Sector: Healthcare")
print("  Date range: 2023-01-01 to 2024-12-31")


# =============================================================================
# 7. FLUENT SECTOR DATA ACCESSOR
# =============================================================================

print("\n=== Example 7: Sector Data Accessor ===")

# Access sector data with employment information
sectors = db.sectors() \
    .with_employment_data() \
    .sort_by_growth("DESC") \
    .limit(10) \
    .get()

print("Sectors with employment data:")
print("  Include employment data: Yes")
print("  Sort by: Growth (descending)")
print("  Limit: Top 10 sectors")


# =============================================================================
# 8. SECTOR DATA - Multiple sort options
# =============================================================================

print("\n=== Example 8: Sector Data - Sort Options ===")

# Sort by growth
growth_sectors = db.sectors() \
    .sort_by_growth("DESC") \
    .limit(5) \
    .get()

print("Top 5 sectors by growth")

# Sort by performance
performance_sectors = db.sectors() \
    .with_rotation_data() \
    .sort_by_performance("DESC") \
    .limit(5) \
    .get()

print("Top 5 sectors by performance (with rotation data)")


# =============================================================================
# 9. CHAINING CONFIGURATION AND QUERIES
# =============================================================================

print("\n=== Example 9: Combined Configuration and Queries ===")

# Configure database and run queries in sequence
results = db \
    .set_read_only(False) \
    .set_max_memory(1024 * 1024 * 1024) \
    .query() \
    .select(["symbol", "price"]) \
    .from_table("quotes") \
    .where("price > 150") \
    .limit(10) \
    .execute()

print("Configured DB and executed query in fluent chain")


# =============================================================================
# 10. REUSABLE QUERY BUILDER WITH RESET
# =============================================================================

print("\n=== Example 10: Reusable Query Builder ===")

builder = db.query()

# Build first query
query1 = builder \
    .select(["id", "name"]) \
    .from_table("employees") \
    .where("department = 'Sales'") \
    .limit(50) \
    .execute()

# Reset and build different query with same builder
query2 = builder \
    .reset() \
    .select(["id", "salary"]) \
    .from_table("payroll") \
    .where("salary > 100000") \
    .order_by("salary", "DESC") \
    .limit(20) \
    .execute()

print("Built two different queries with same builder after reset")


# =============================================================================
# 11. BACKWARD COMPATIBILITY - Old API still works
# =============================================================================

print("\n=== Example 11: Backward Compatibility ===")

# Traditional methods still work
traditional_result = db.execute("SELECT * FROM employment LIMIT 1")
print(f"Traditional execute() method still works: {len(traditional_result.columns)} columns")

# Convert to dataframe (old method)
employment_df = db.to_dataframe("employment")
print(f"Traditional to_dataframe() method still works: {len(employment_df)} rows")

# Get employment data the old way
employment_stats = db.get_employment_statistics()
print("Traditional get_employment_statistics() method still works")


# =============================================================================
# 12. BUILD QUERY WITHOUT EXECUTING
# =============================================================================

print("\n=== Example 12: Build Query Without Executing ===")

# Use .build() to get SQL string for inspection, logging, or optimization
sql_query = db.query() \
    .select(["symbol", "price", "volume"]) \
    .from_table("quotes") \
    .where("price > 100") \
    .where("volume > 1000000") \
    .order_by("volume", "DESC") \
    .limit(50) \
    .build()

print("Generated SQL query:")
print(sql_query)
print("\nQuery can be used for:")
print("  - Debugging/inspection")
print("  - Logging")
print("  - Performance analysis")
print("  - Passing to other systems")


# =============================================================================
# COMPARISON: Fluent vs Traditional Approach
# =============================================================================

print("\n=== Comparison: Fluent vs Traditional ===")

print("\nTraditional Approach:")
print("""
db = duckdb.Connection("data.duckdb")
result = db.execute(
    'SELECT symbol, price FROM quotes WHERE price > 100 ORDER BY volume DESC LIMIT 10'
)
""")

print("\nFluent Approach (Method Chaining):")
print("""
result = db.query() \\
    .select(["symbol", "price"]) \\
    .from_table("quotes") \\
    .where("price > 100") \\
    .order_by("volume", "DESC") \\
    .limit(10) \\
    .execute()
""")

print("\nBenefits of Fluent Approach:")
print("  + More readable and self-documenting")
print("  + Type-safe (if using C++ bindings)")
print("  + Composable and reusable")
print("  + Easier to debug complex queries")
print("  + Better IDE autocompletion support")
print("  + Can build queries programmatically")


# =============================================================================
# DESIGN PATTERN: Following Schwab API Style
# =============================================================================

print("\n=== Design Pattern: Schwab API Style ===")

print("""
The fluent API follows the Schwab API design pattern:

    schwab.marketData().getQuote("SPY")
    schwab.orders().placeOrder(request)

Similarly, BigBrother DuckDB uses:

    db.query()
        .select([...])
        .from_table("...")
        .where("...")
        .execute()

    db.employment()
        .for_sector("Technology")
        .between_dates("2024-01-01", "2025-01-01")
        .get()

    db.sectors()
        .with_employment_data()
        .sort_by_growth("DESC")
        .get()
""")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DuckDB Fluent API Examples Complete")
    print("="*60)
    print("\nFor more information:")
    print("  - Check duckdb_fluent.hpp for C++ implementation")
    print("  - Check duckdb_bindings.cpp for Python bindings")
    print("  - Run test_duckdb_fluent.cpp for comprehensive tests")
