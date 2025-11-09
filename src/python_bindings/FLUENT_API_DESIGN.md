# DuckDB Fluent API Design Document

## Overview

The DuckDB Fluent API implements a fluent interface pattern (method chaining) for database operations, matching the Schwab API design principles. This enables intuitive, readable, and composable database queries.

**Design Date:** 2025-11-09
**Author:** Olumuyiwa Oluwasanmi
**Pattern:** Builder Pattern + Fluent Interface

---

## Table of Contents

1. [Architecture](#architecture)
2. [Fluent Configuration Methods](#fluent-configuration-methods)
3. [QueryBuilder](#querybuilder)
4. [Data Accessors](#data-accessors)
5. [Python Bindings](#python-bindings)
6. [Usage Examples](#usage-examples)
7. [Backward Compatibility](#backward-compatibility)
8. [Thread Safety](#thread-safety)
9. [Performance Considerations](#performance-considerations)

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│         DuckDBConnection (Main Class)               │
├─────────────────────────────────────────────────────┤
│ • Fluent Configuration Methods                      │
│   - setReadOnly()                                   │
│   - setMaxMemory()                                  │
│   - enableAutoCheckpoint()                          │
│   - setThreadPoolSize()                             │
│   - enableLogging()                                 │
├─────────────────────────────────────────────────────┤
│ • Fluent Query Builders                             │
│   - query() → QueryBuilder                          │
│   - employment() → EmploymentDataAccessor           │
│   - sectors() → SectorDataAccessor                  │
├─────────────────────────────────────────────────────┤
│ • Legacy Methods (Backward Compatible)              │
│   - execute()                                       │
│   - to_dataframe()                                  │
│   - get_employment_data()                           │
│   - get_employment_statistics()                     │
└─────────────────────────────────────────────────────┘

                        ↓

┌─────────────────────────────────────────────────────┐
│            QueryBuilder                             │
├─────────────────────────────────────────────────────┤
│ • SQL Construction Methods                          │
│   - select(columns)                                 │
│   - from(table)                                     │
│   - where(condition)                                │
│   - orWhere(condition)                              │
│   - orderBy(column, direction)                      │
│   - limit(count)                                    │
│   - offset(count)                                   │
│ • Execution                                         │
│   - execute() → SQL string                          │
│   - build() → SQL string (no exec)                  │
│   - reset() → Clear builder                         │
└─────────────────────────────────────────────────────┘

                        ↓

┌─────────────────────────────────────────────────────┐
│        Data Accessors (Specialized)                 │
├─────────────────────────────────────────────────────┤
│ • EmploymentDataAccessor                            │
│   - forSector(sector)                               │
│   - betweenDates(start, end)                        │
│   - fromDate(start)                                 │
│   - toDate(end)                                     │
│   - limit(count)                                    │
│   - get() → Query string                            │
│                                                     │
│ • SectorDataAccessor                                │
│   - withEmploymentData()                            │
│   - withRotationData()                              │
│   - sortByGrowth(direction)                         │
│   - sortByPerformance(direction)                    │
│   - limit(count)                                    │
│   - get() → Query string                            │
└─────────────────────────────────────────────────────┘
```

### Files

- **duckdb_fluent.hpp** - C++ header with fluent interface classes
- **duckdb_bindings.cpp** - Python bindings and DuckDBConnection integration
- **test_duckdb_fluent.cpp** - Comprehensive test suite
- **fluent_api_examples.py** - Python usage examples
- **FLUENT_API_DESIGN.md** - This document

---

## Fluent Configuration Methods

Configuration methods enable database setup through method chaining.

### Method Signatures

```cpp
auto setReadOnly(bool read_only) -> DuckDBConnection&;
auto setMaxMemory(size_t bytes) -> DuckDBConnection&;
auto enableAutoCheckpoint(bool enable) -> DuckDBConnection&;
auto setThreadPoolSize(int threads) -> DuckDBConnection&;
auto enableLogging(bool enable) -> DuckDBConnection&;
```

### C++ Usage

```cpp
auto db = DuckDBConnection("data.duckdb")
    .setReadOnly(false)
    .setMaxMemory(2 * 1024 * 1024 * 1024)  // 2GB
    .enableAutoCheckpoint(true)
    .setThreadPoolSize(4)
    .enableLogging(true);
```

### Python Usage

```python
db = duckdb.Connection("data/bigbrother.duckdb") \
    .set_read_only(False) \
    .set_max_memory(2 * 1024 * 1024 * 1024) \
    .enable_auto_checkpoint(True) \
    .set_thread_pool_size(4) \
    .enable_logging(True)
```

### Parameters

| Method | Parameter | Type | Default | Description |
|--------|-----------|------|---------|-------------|
| setReadOnly | read_only | bool | false | Set read-only mode |
| setMaxMemory | bytes | size_t | 0 (unlimited) | Max memory for queries |
| enableAutoCheckpoint | enable | bool | true | Enable auto-checkpoint |
| setThreadPoolSize | threads | int | 0 (auto-detect) | Number of threads |
| enableLogging | enable | bool | false | Enable query logging |

---

## QueryBuilder

The QueryBuilder class implements fluent SQL construction using the builder pattern.

### Class Definition

```cpp
class QueryBuilder {
public:
    auto select(std::vector<std::string> columns) -> QueryBuilder&;
    auto selectAll() -> QueryBuilder&;
    auto from(std::string table) -> QueryBuilder&;
    auto where(std::string condition) -> QueryBuilder&;
    auto orWhere(std::string condition) -> QueryBuilder&;
    auto orderBy(std::string column, std::string direction = "ASC") -> QueryBuilder&;
    auto limit(int count) -> QueryBuilder&;
    auto offset(int count) -> QueryBuilder&;
    auto execute() -> std::string;
    auto build() -> std::string;
    auto reset() -> QueryBuilder&;
};
```

### Method Details

#### select(columns)
Specify columns to select.

```cpp
.select({"id", "name", "value"})
.select({"price", "volume"})
```

#### selectAll()
Select all columns (SELECT *).

```cpp
.selectAll()
```

#### from(table)
Specify the table to query.

```cpp
.from("employees")
.from("quotes")
```

#### where(condition)
Add AND condition to WHERE clause.

```cpp
.where("salary > 50000")
.where("department = 'Engineering'")
```

Multiple WHERE calls are AND'd together:
```
WHERE salary > 50000 AND department = 'Engineering'
```

#### orWhere(condition)
Add OR condition to WHERE clause.

```cpp
.where("status = 'active'")
.orWhere("status = 'pending'")
```

Generates:
```sql
WHERE status = 'active' OR status = 'pending'
```

#### orderBy(column, direction)
Order results by column.

```cpp
.orderBy("volume", "DESC")
.orderBy("date", "ASC")
.orderBy("name")  // Defaults to ASC
```

#### limit(count)
Limit result set size.

```cpp
.limit(10)
.limit(100)
```

#### offset(count)
Skip rows for pagination.

```cpp
.limit(20)
.offset(40)  // Rows 41-60
```

#### execute()
Build and return the SQL query string.

```cpp
std::string sql = builder.execute();
```

#### build()
Build query without modifying state.

```cpp
std::string sql = builder.build();
```

#### reset()
Clear builder to initial state.

```cpp
builder.reset()
    .select({"different"})
    .from("new_table");
```

### C++ Examples

#### Basic Query
```cpp
auto query = db.query()
    .select({"id", "name", "salary"})
    .from("employees")
    .where("salary > 50000")
    .orderBy("salary", "DESC")
    .limit(10);

std::string sql = query.execute();
// SELECT id, name, salary FROM employees WHERE salary > 50000 ORDER BY salary DESC LIMIT 10
```

#### Complex Query with Multiple Conditions
```cpp
auto query = db.query()
    .select({"id", "ticker", "sector", "market_cap"})
    .from("companies")
    .where("market_cap > 1000000000")
    .where("sector = 'Technology'")
    .orWhere("sector = 'Healthcare'")
    .orderBy("market_cap", "DESC")
    .limit(20);
```

#### Pagination
```cpp
// Page 1
auto page1 = db.query()
    .from("records")
    .limit(20)
    .offset(0);

// Page 2
auto page2 = db.query()
    .from("records")
    .limit(20)
    .offset(20);
```

### Python Examples

#### Basic Query
```python
query = db.query() \
    .select(["id", "name", "salary"]) \
    .from_table("employees") \
    .where("salary > 50000") \
    .order_by("salary", "DESC") \
    .limit(10)

sql = query.execute()
```

#### Using .build() for Inspection
```python
sql = db.query() \
    .select(["price", "volume"]) \
    .from_table("quotes") \
    .where("price > 100") \
    .build()

print(f"Generated SQL: {sql}")
```

---

## Data Accessors

Specialized accessor classes for specific data domains.

### EmploymentDataAccessor

Fluent interface for employment data queries.

```cpp
class EmploymentDataAccessor {
public:
    auto forSector(std::string sector) -> EmploymentDataAccessor&;
    auto betweenDates(std::string start, std::string end) -> EmploymentDataAccessor&;
    auto fromDate(std::string start) -> EmploymentDataAccessor&;
    auto toDate(std::string end) -> EmploymentDataAccessor&;
    auto limit(int count) -> EmploymentDataAccessor&;
    auto get() -> std::string;
};
```

#### Methods

##### forSector(sector)
Filter by sector name.

```cpp
.forSector("Technology")
.forSector("Healthcare")
```

##### betweenDates(start, end)
Filter by date range (inclusive).

```cpp
.betweenDates("2024-01-01", "2025-01-01")
```

Date format: YYYY-MM-DD

##### fromDate(start)
Filter from specific date to latest.

```cpp
.fromDate("2024-06-01")
```

##### toDate(end)
Filter from earliest to specific date.

```cpp
.toDate("2024-12-31")
```

##### limit(count)
Limit number of records.

```cpp
.limit(100)
```

##### get()
Execute and return query string.

```cpp
std::string query = accessor.get();
```

#### C++ Example

```cpp
auto employment = db.employment()
    .forSector("Technology")
    .betweenDates("2024-01-01", "2025-01-01")
    .limit(100)
    .get();
```

#### Python Example

```python
employment = db.employment() \
    .for_sector("Technology") \
    .between_dates("2024-01-01", "2025-01-01") \
    .limit(100) \
    .get()
```

### SectorDataAccessor

Fluent interface for sector data queries.

```cpp
class SectorDataAccessor {
public:
    auto withEmploymentData() -> SectorDataAccessor&;
    auto withRotationData() -> SectorDataAccessor&;
    auto sortByGrowth(std::string direction = "DESC") -> SectorDataAccessor&;
    auto sortByPerformance(std::string direction = "DESC") -> SectorDataAccessor&;
    auto limit(int count) -> SectorDataAccessor&;
    auto get() -> std::string;
};
```

#### Methods

##### withEmploymentData()
Include employment data in results.

```cpp
.withEmploymentData()
```

##### withRotationData()
Include sector rotation data.

```cpp
.withRotationData()
```

##### sortByGrowth(direction)
Sort by growth metrics.

```cpp
.sortByGrowth("DESC")
.sortByGrowth("ASC")
```

##### sortByPerformance(direction)
Sort by performance metrics.

```cpp
.sortByPerformance("DESC")
```

##### limit(count)
Limit number of sectors.

```cpp
.limit(10)
```

##### get()
Execute and return query string.

```cpp
std::string query = accessor.get();
```

#### C++ Example

```cpp
auto sectors = db.sectors()
    .withEmploymentData()
    .sortByGrowth("DESC")
    .limit(10)
    .get();
```

#### Python Example

```python
sectors = db.sectors() \
    .with_employment_data() \
    .sort_by_growth("DESC") \
    .limit(10) \
    .get()
```

---

## Python Bindings

Python method names follow snake_case convention while maintaining fluent interface.

### Configuration Methods

```python
# Fluent configuration methods return self for chaining
db.set_read_only(bool) -> Connection
db.set_max_memory(int) -> Connection
db.enable_auto_checkpoint(bool) -> Connection
db.set_thread_pool_size(int) -> Connection
db.enable_logging(bool) -> Connection
```

### Query Builder

```python
# Create builder
builder = db.query() -> QueryBuilder

# QueryBuilder methods (fluent)
builder.select(list) -> QueryBuilder
builder.select_all() -> QueryBuilder
builder.from_table(str) -> QueryBuilder
builder.where(str) -> QueryBuilder
builder.or_where(str) -> QueryBuilder
builder.order_by(str, str = "ASC") -> QueryBuilder
builder.limit(int) -> QueryBuilder
builder.offset(int) -> QueryBuilder

# Execution
sql = builder.execute() -> str
sql = builder.build() -> str
builder.reset() -> QueryBuilder
```

### Employment Accessor

```python
accessor = db.employment() -> EmploymentDataAccessor

accessor.for_sector(str) -> EmploymentDataAccessor
accessor.between_dates(str, str) -> EmploymentDataAccessor
accessor.from_date(str) -> EmploymentDataAccessor
accessor.to_date(str) -> EmploymentDataAccessor
accessor.limit(int) -> EmploymentDataAccessor
query = accessor.get() -> str
```

### Sector Accessor

```python
accessor = db.sectors() -> SectorDataAccessor

accessor.with_employment_data() -> SectorDataAccessor
accessor.with_rotation_data() -> SectorDataAccessor
accessor.sort_by_growth(str = "DESC") -> SectorDataAccessor
accessor.sort_by_performance(str = "DESC") -> SectorDataAccessor
accessor.limit(int) -> SectorDataAccessor
query = accessor.get() -> str
```

---

## Usage Examples

### Example 1: Basic Configuration and Query

```python
import bigbrother_duckdb as duckdb

# Fluent configuration
db = duckdb.Connection("data/bigbrother.duckdb") \
    .set_read_only(False) \
    .set_max_memory(2 * 1024 * 1024 * 1024)

# Fluent query
result = db.query() \
    .select(["symbol", "price", "volume"]) \
    .from_table("quotes") \
    .where("price > 100") \
    .order_by("volume", "DESC") \
    .limit(10) \
    .execute()

print(result)
```

### Example 2: Employment Data Analysis

```python
# Access employment data for specific sector and date range
employment = db.employment() \
    .for_sector("Technology") \
    .between_dates("2024-01-01", "2025-01-01") \
    .limit(100) \
    .get()

print(employment)
```

### Example 3: Sector Analysis with Rotation

```python
# Get top sectors by growth with employment data
sectors = db.sectors() \
    .with_employment_data() \
    .with_rotation_data() \
    .sort_by_growth("DESC") \
    .limit(10) \
    .get()

print(sectors)
```

### Example 4: Complex Query with Multiple Conditions

```python
# Find large companies in specific sectors
companies = db.query() \
    .select(["id", "ticker", "sector", "market_cap"]) \
    .from_table("companies") \
    .where("market_cap > 1000000000") \
    .where("sector = 'Technology'") \
    .or_where("sector = 'Healthcare'") \
    .order_by("market_cap", "DESC") \
    .limit(50) \
    .execute()

print(companies)
```

### Example 5: Pagination

```python
page_size = 20
current_page = 2  # Page 2 (items 21-40)

results = db.query() \
    .select(["id", "name", "value"]) \
    .from_table("records") \
    .order_by("id", "ASC") \
    .limit(page_size) \
    .offset((current_page - 1) * page_size) \
    .execute()

print(results)
```

### Example 6: Query Inspection

```python
# Build query without executing (useful for debugging)
sql = db.query() \
    .select(["id", "name", "salary"]) \
    .from_table("employees") \
    .where("salary > 50000") \
    .where("department = 'Engineering'") \
    .order_by("salary", "DESC") \
    .limit(100) \
    .build()

print(f"Generated SQL: {sql}")

# Can also inspect queries from accessors
emp_sql = db.employment() \
    .for_sector("Finance") \
    .from_date("2024-01-01") \
    .limit(50) \
    .get()

print(f"Employment query: {emp_sql}")
```

### Example 7: Reusing Builder with Reset

```python
builder = db.query()

# First query
result1 = builder \
    .select(["id", "name"]) \
    .from_table("employees") \
    .where("active = true") \
    .execute()

# Reset and build different query
result2 = builder \
    .reset() \
    .select(["id", "salary"]) \
    .from_table("payroll") \
    .where("salary > 100000") \
    .order_by("salary", "DESC") \
    .limit(50) \
    .execute()
```

---

## Backward Compatibility

All existing methods remain unchanged and functional. The fluent API is purely additive.

### Legacy Methods

```python
# Traditional execute
result = db.execute("SELECT * FROM employment WHERE date >= '2024-01-01'")

# Traditional dataframe loading
df_dict = db.to_dataframe("employment")

# Traditional employment queries
stats = db.get_employment_statistics()
latest = db.get_latest_employment(limit=10)
employment_data = db.get_employment_data("2024-01-01", "2024-12-31")

# Traditional table info
table_info = db.get_table_info("employment")
tables = db.list_tables()
row_count = db.get_row_count("employment")
```

### Migration Path

No migration is required. Code can be updated gradually:

```python
# Old style (still works)
result = db.execute("SELECT * FROM quotes WHERE price > 100 LIMIT 10")

# New style (fluent - optional)
result = db.query() \
    .from_table("quotes") \
    .where("price > 100") \
    .limit(10) \
    .execute()

# Mixed style (during transition)
db = duckdb.Connection("data.duckdb") \
    .set_read_only(False)  # Fluent config

result = db.execute("SELECT ...")  # Traditional query
```

---

## Thread Safety

### Configuration Methods

Configuration methods are **NOT thread-safe** during initialization. Configure the connection before shared access.

```cpp
// SAFE: Configure before sharing
auto db = DuckDBConnection("data.duckdb")
    .setReadOnly(false)
    .setMaxMemory(2GB);

// Use db in multiple threads (read-only configuration)
```

### Query Execution

Query builders are **thread-safe** once created. Each thread can have its own builder or query instance.

```cpp
// Each thread can create its own query builder
auto thread1_builder = db.query()
    .select({"id"})
    .from("table1");

auto thread2_builder = db.query()
    .select({"name"})
    .from("table2");

// Execute independently in different threads
```

### Best Practices

1. Configure the connection once at startup
2. Create separate builders for each thread
3. Use GIL-free execution for queries (handled by pybind11)
4. Don't share mutable builders between threads

---

## Performance Considerations

### Query Building Performance

Query building is **O(n)** where n is the number of SQL clauses:
- `select()`: O(m) where m = number of columns
- `from()`: O(1)
- `where()`: O(1) per condition
- `orderBy()`: O(1) per order clause
- `limit()`: O(1)
- `offset()`: O(1)

Total: O(m + k) where m = columns, k = conditions

### Memory Usage

- QueryBuilder: ~1KB base + column names
- EmploymentDataAccessor: ~256B
- SectorDataAccessor: ~256B

Minimal overhead compared to query execution.

### Query Execution

- Building SQL string: microseconds
- Actual query execution: depends on DuckDB
- No performance penalty vs. raw SQL execution

### Optimization Tips

1. Use `.build()` to inspect generated SQL before execution
2. Reset builders to reuse for different queries
3. Limit result sets with `.limit()` to reduce memory
4. Use pagination (`.limit()` + `.offset()`) for large datasets

---

## Design Patterns

### Builder Pattern

Each class returns `*this` (C++) or `self` (Python) to enable chaining.

```cpp
auto& setReadOnly(bool value) {
    read_only_ = value;
    return *this;  // Enable chaining
}
```

### Fluent Interface

Method names are verbs that read like natural language:

```python
db.query()
    .select([...])      # What columns?
    .from_table("...")  # From which table?
    .where("...")       # With what condition?
    .order_by("...")    # Sorted how?
    .limit(10)          # How many results?
    .execute()          # Do it!
```

### Separation of Concerns

- **DuckDBConnection**: Manages connection lifecycle
- **QueryBuilder**: Constructs generic SQL queries
- **EmploymentDataAccessor**: Domain-specific employment queries
- **SectorDataAccessor**: Domain-specific sector queries

---

## Testing

Comprehensive test suite in `test_duckdb_fluent.cpp`:

- Configuration method chaining
- Query builder SQL generation
- Accessor query generation
- Reset functionality
- Backward compatibility
- Combined fluent operations

Run tests:
```bash
g++ -std=c++23 -o test_fluent test_duckdb_fluent.cpp \
    -I/path/to/duckdb -L/path/to/duckdb -lduckdb
./test_fluent
```

---

## Reference Pattern: Schwab API

The fluent API design mirrors Schwab API patterns:

### Schwab Style
```cpp
auto quote = schwab.marketData().getQuote("SPY");
auto chain = schwab.marketData().getOptionChain(request);
auto order = schwab.orders().placeOrder(request);
```

### BigBrother DuckDB Style
```python
data = db.query()
    .select([...])
    .from_table(...)
    .where(...)
    .execute()

employment = db.employment()
    .for_sector(...)
    .between_dates(...)
    .get()

sectors = db.sectors()
    .with_employment_data()
    .sort_by_growth()
    .get()
```

Both follow:
1. Accessor methods for domains
2. Method chaining for configuration
3. Specialized accessors for specific data types
4. Clear, readable, verb-based method names

---

## Future Enhancements

Potential additions:

1. **Query Caching**: Cache frequently-built queries
2. **Query Optimization**: Suggest optimizations based on patterns
3. **Async Support**: Async query execution with futures
4. **Transaction Support**: Fluent transaction management
5. **Join Support**: Fluent JOIN clause building
6. **Aggregation Support**: Fluent GROUP BY and aggregate functions
7. **Sub-query Support**: Nested query building
8. **Query Logging**: Built-in query logging for debugging

---

## Summary

The DuckDB Fluent API provides:

- **Readable**: Natural language-like query construction
- **Composable**: Reusable builders and accessors
- **Type-Safe**: C++23 trailing return syntax
- **Fast**: Zero overhead vs. raw SQL
- **Compatible**: Existing code continues to work
- **Thread-Safe**: Multiple concurrent queries
- **Extensible**: Easy to add new accessors

Maximum developer experience with minimum runtime cost.

---

## Contact & Support

For questions or issues:
1. Check examples in `fluent_api_examples.py`
2. Review tests in `test_duckdb_fluent.cpp`
3. Read inline documentation in header files
4. Refer to DuckDB official documentation
