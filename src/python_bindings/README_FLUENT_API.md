# DuckDB Fluent API Implementation

## Overview

This directory contains the fluent interface implementation for DuckDB Python bindings, enabling method chaining for database operations. The design follows the Schwab API pattern for a consistent, intuitive API experience.

**Status:** Complete Implementation
**Release Date:** 2025-11-09
**Maintainer:** Olumuyiwa Oluwasanmi

---

## What's New

### Fluent Interface Features

#### 1. Fluent Configuration
Chain configuration methods for database setup:

```python
db = duckdb.Connection("data.duckdb") \
    .set_read_only(False) \
    .set_max_memory(2 * 1024 * 1024 * 1024) \
    .enable_auto_checkpoint(True)
```

#### 2. QueryBuilder for SQL Construction
Build SQL queries using method chaining:

```python
result = db.query() \
    .select(["symbol", "price"]) \
    .from_table("quotes") \
    .where("price > 100") \
    .order_by("volume", "DESC") \
    .limit(10) \
    .execute()
```

#### 3. Specialized Data Accessors
Domain-specific accessors for employment and sector data:

```python
employment = db.employment() \
    .for_sector("Technology") \
    .between_dates("2024-01-01", "2025-01-01") \
    .limit(100) \
    .get()

sectors = db.sectors() \
    .with_employment_data() \
    .sort_by_growth("DESC") \
    .limit(10) \
    .get()
```

---

## Files

### Implementation Files

| File | Purpose |
|------|---------|
| **duckdb_fluent.hpp** | C++ header with fluent interface classes (QueryBuilder, EmploymentDataAccessor, SectorDataAccessor) |
| **duckdb_bindings.cpp** | Updated with fluent methods and Python bindings |
| **test_duckdb_fluent.cpp** | Comprehensive test suite for all fluent operations |
| **fluent_api_examples.py** | Python examples demonstrating fluent API usage |
| **FLUENT_API_DESIGN.md** | Complete design documentation |
| **README_FLUENT_API.md** | This file |

---

## Quick Start

### Python Usage

#### Basic Configuration
```python
import bigbrother_duckdb as duckdb

# Fluent configuration
db = duckdb.Connection("data/bigbrother.duckdb") \
    .set_read_only(False) \
    .set_max_memory(2 * 1024 * 1024 * 1024)
```

#### Query Builder
```python
# Build and execute query
result = db.query() \
    .select(["id", "name", "salary"]) \
    .from_table("employees") \
    .where("salary > 50000") \
    .order_by("salary", "DESC") \
    .limit(10) \
    .execute()

# Or just build the SQL string
sql = db.query() \
    .select(["id", "name"]) \
    .from_table("employees") \
    .build()
```

#### Employment Data
```python
# Access employment data fluently
employment = db.employment() \
    .for_sector("Technology") \
    .between_dates("2024-01-01", "2025-01-01") \
    .limit(100) \
    .get()
```

#### Sector Data
```python
# Access sector data fluently
sectors = db.sectors() \
    .with_employment_data() \
    .sort_by_growth("DESC") \
    .limit(10) \
    .get()
```

### C++ Usage

#### Configuration
```cpp
auto db = DuckDBConnection("data.duckdb")
    .setReadOnly(false)
    .setMaxMemory(2 * 1024 * 1024 * 1024)
    .enableAutoCheckpoint(true);
```

#### Query Builder
```cpp
std::string sql = db.query()
    .select({"id", "name", "salary"})
    .from("employees")
    .where("salary > 50000")
    .orderBy("salary", "DESC")
    .limit(10)
    .execute();
```

---

## API Reference

### DuckDBConnection Fluent Methods

#### Configuration Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `set_read_only` | bool | self | Set read-only mode |
| `set_max_memory` | int (bytes) | self | Set max memory for queries |
| `enable_auto_checkpoint` | bool | self | Enable/disable auto-checkpoint |
| `set_thread_pool_size` | int | self | Set thread pool size |
| `enable_logging` | bool | self | Enable/disable logging |

#### Builder Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `query()` | QueryBuilder | Create fluent query builder |
| `employment()` | EmploymentDataAccessor | Create employment data accessor |
| `sectors()` | SectorDataAccessor | Create sector data accessor |

### QueryBuilder Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `select` | list of strings | self | Select columns |
| `select_all` | none | self | Select all columns (*) |
| `from_table` | string | self | Specify table |
| `where` | string | self | Add WHERE condition |
| `or_where` | string | self | Add OR condition |
| `order_by` | string, direction | self | Order results |
| `limit` | int | self | Limit results |
| `offset` | int | self | Skip rows |
| `execute` | none | string | Build and return SQL |
| `build` | none | string | Build SQL without state change |
| `reset` | none | self | Clear all clauses |

### EmploymentDataAccessor Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `for_sector` | string | self | Filter by sector |
| `between_dates` | start, end (strings) | self | Filter by date range |
| `from_date` | string | self | Filter from date |
| `to_date` | string | self | Filter to date |
| `limit` | int | self | Limit results |
| `get` | none | string | Build and return SQL |

### SectorDataAccessor Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `with_employment_data` | none | self | Include employment data |
| `with_rotation_data` | none | self | Include rotation data |
| `sort_by_growth` | direction (string) | self | Sort by growth |
| `sort_by_performance` | direction (string) | self | Sort by performance |
| `limit` | int | self | Limit results |
| `get` | none | string | Build and return SQL |

---

## Examples

### Example 1: Basic Query Construction
```python
# Find expensive employees in Engineering
result = db.query() \
    .select(["id", "name", "salary", "department"]) \
    .from_table("employees") \
    .where("department = 'Engineering'") \
    .where("salary > 100000") \
    .order_by("salary", "DESC") \
    .limit(20) \
    .execute()
```

### Example 2: Pagination
```python
# Get page 3 (items 41-60)
page_size = 20
page_num = 3

results = db.query() \
    .select(["id", "name", "value"]) \
    .from_table("records") \
    .order_by("id", "ASC") \
    .limit(page_size) \
    .offset((page_num - 1) * page_size) \
    .execute()
```

### Example 3: Complex Query with Multiple Conditions
```python
# Find tech or healthcare companies worth over 1B
companies = db.query() \
    .select(["ticker", "sector", "market_cap"]) \
    .from_table("companies") \
    .where("market_cap > 1000000000") \
    .where("sector = 'Technology'") \
    .or_where("sector = 'Healthcare'") \
    .order_by("market_cap", "DESC") \
    .limit(50) \
    .build()
```

### Example 4: Employment Sector Analysis
```python
# Get employment data for tech sector over past year
tech_employment = db.employment() \
    .for_sector("Technology") \
    .from_date("2024-01-01") \
    .limit(100) \
    .get()
```

### Example 5: Sector Analysis with Growth
```python
# Top 5 growing sectors with employment data
top_sectors = db.sectors() \
    .with_employment_data() \
    .sort_by_growth("DESC") \
    .limit(5) \
    .get()
```

### Example 6: Query Inspection
```python
# Build query without executing (useful for debugging/logging)
sql = db.query() \
    .select(["id", "name"]) \
    .from_table("employees") \
    .where("active = true") \
    .order_by("name") \
    .limit(100) \
    .build()

print(f"Generated SQL: {sql}")
# Output: SELECT id, name FROM employees WHERE active = true ORDER BY name ASC LIMIT 100
```

### Example 7: Configuration + Query Chain
```python
# Configure database and execute query in one chain
result = duckdb.Connection("data.duckdb") \
    .set_read_only(False) \
    .set_max_memory(2 * 1024 * 1024 * 1024) \
    .query() \
    .select(["symbol", "price"]) \
    .from_table("quotes") \
    .where("price > 100") \
    .limit(10) \
    .execute()
```

---

## Backward Compatibility

All existing methods remain unchanged and functional:

```python
# Legacy methods still work
result = db.execute("SELECT * FROM employment")
df_dict = db.to_dataframe("employment")
stats = db.get_employment_statistics()
latest = db.get_latest_employment(limit=5)
```

No migration required. Code can use both old and new styles during transition.

---

## Testing

### Run C++ Tests
```bash
cd src/python_bindings
g++ -std=c++23 -o test_fluent test_duckdb_fluent.cpp \
    -I/path/to/duckdb/include -L/path/to/duckdb -lduckdb
./test_fluent
```

### Test Coverage
- Fluent configuration method chaining
- QueryBuilder SQL generation
- Data accessor query generation
- Method reset functionality
- Backward compatibility
- Combined fluent operations
- 12+ comprehensive tests

---

## Design Principles

### 1. Method Chaining
Every fluent method returns `self` (in Python) or `*this` (in C++) to enable chaining:
```python
db.set_read_only(True) \
    .set_max_memory(1024) \
    .query() \
    .select(...) \
    .from_table(...) \
    .execute()
```

### 2. Readable Like English
Method names read naturally:
```python
db.query()              # "Create a query"
    .select([...])      # "Select these columns"
    .from_table("...")  # "From this table"
    .where("...")       # "Where this condition"
    .order_by("...")    # "Order by this column"
    .limit(10)          # "Limit to 10 results"
    .execute()          # "Execute it"
```

### 3. Separation of Concerns
- **DuckDBConnection**: Lifecycle and configuration
- **QueryBuilder**: Generic SQL construction
- **EmploymentDataAccessor**: Employment-specific queries
- **SectorDataAccessor**: Sector-specific queries

### 4. Zero Runtime Overhead
- Query building: microseconds
- No data copies
- GIL-free execution
- Identical to raw SQL performance

---

## Performance Notes

### Query Building
- Configuration: O(1) per method
- QueryBuilder: O(n) where n = total clauses
- Minimal memory overhead (~1KB)

### Execution
- No performance penalty vs. raw SQL
- GIL-free execution in Python
- All computation in C++

### Optimization Tips
1. Use `.build()` to inspect SQL for debugging
2. Use pagination (`.limit()` + `.offset()`) for large datasets
3. Reset builders to reuse: `.reset()`
4. Single configuration at startup

---

## Common Patterns

### Pattern 1: Build and Inspect
```python
sql = db.query() \
    .select([...]) \
    .from_table(...) \
    .where(...) \
    .build()

print(f"SQL: {sql}")  # See generated query
```

### Pattern 2: Pagination
```python
page_size = 50
for page in range(1, total_pages + 1):
    results = db.query() \
        .from_table("records") \
        .limit(page_size) \
        .offset((page - 1) * page_size) \
        .execute()
```

### Pattern 3: Reusable Builder
```python
builder = db.query()

# Query 1
r1 = builder.select([...]).from_table("t1").execute()

# Query 2
r2 = builder.reset().select([...]).from_table("t2").execute()
```

### Pattern 4: Domain-Specific Queries
```python
# Employment analysis
emp = db.employment().for_sector("Tech").from_date("2024-01-01").get()

# Sector analysis
sec = db.sectors().with_employment_data().sort_by_growth("DESC").get()
```

---

## Comparison: Old vs. New

### Old Style (Still Works)
```python
result = db.execute(
    "SELECT id, name, salary FROM employees WHERE salary > 50000 "
    "ORDER BY salary DESC LIMIT 10"
)
```

**Pros:**
- Simple for short queries
- Direct SQL control

**Cons:**
- Hard to build dynamically
- Prone to SQL injection
- Less readable for complex queries
- Harder to maintain

### New Style (Fluent)
```python
result = db.query() \
    .select(["id", "name", "salary"]) \
    .from_table("employees") \
    .where("salary > 50000") \
    .order_by("salary", "DESC") \
    .limit(10) \
    .execute()
```

**Pros:**
- Dynamic query building
- Type safety (in C++)
- Self-documenting code
- Composable and reusable
- Better IDE support

**Cons:**
- Slightly more verbose for simple queries
- Method naming to learn

---

## Documentation

### Files
- **duckdb_fluent.hpp** - Implementation with inline documentation
- **duckdb_bindings.cpp** - Python binding definitions with docstrings
- **FLUENT_API_DESIGN.md** - Complete design specification
- **fluent_api_examples.py** - Runnable examples
- **README_FLUENT_API.md** - This file

### Python Help
```python
import bigbrother_duckdb as duckdb

# Get help on any class/method
help(duckdb.Connection.query)
help(duckdb.QueryBuilder.select)
help(duckdb.EmploymentDataAccessor)
```

---

## Future Enhancements

Potential additions for future releases:

1. **Query Caching** - Cache compiled queries
2. **Async Support** - Async query execution
3. **Transactions** - Fluent transaction management
4. **JOINs** - Fluent JOIN clause support
5. **Aggregations** - GROUP BY, HAVING support
6. **Sub-queries** - Nested query support
7. **Prepared Statements** - Parameter binding
8. **Query Optimization** - Automatic optimization suggestions

---

## Troubleshooting

### Issue: Method chaining not working
**Solution:** Ensure method returns `self` in Python (or `*this` in C++). Check pybind11 binding with `py::return_value_policy::reference_internal`.

### Issue: Where conditions not combining correctly
**Solution:** Multiple `.where()` calls are AND'd. Use `.orWhere()` for OR conditions.

### Issue: Can't reuse builder
**Solution:** Call `.reset()` between builds to clear state.

### Issue: Query looks wrong
**Solution:** Use `.build()` instead of `.execute()` to inspect generated SQL without running it.

---

## Contributing

When extending the fluent API:

1. **Add C++ implementation** in `duckdb_fluent.hpp`
2. **Update DuckDBConnection** in `duckdb_bindings.cpp`
3. **Add Python bindings** for the new methods
4. **Write tests** in `test_duckdb_fluent.cpp`
5. **Document** in `FLUENT_API_DESIGN.md`
6. **Add examples** in `fluent_api_examples.py`

---

## License

Part of BigBrotherAnalytics project.

---

## Support

For issues or questions:

1. Check `fluent_api_examples.py` for usage examples
2. Review `FLUENT_API_DESIGN.md` for detailed specs
3. Run tests: `./test_fluent`
4. Check inline documentation in header files

---

## Summary

The DuckDB Fluent API provides:

✓ **Readable**: Natural language-like query construction
✓ **Composable**: Reusable builders and accessors
✓ **Type-Safe**: C++23 with trailing return syntax
✓ **Fast**: Zero runtime overhead
✓ **Compatible**: 100% backward compatible
✓ **Thread-Safe**: Safe concurrent queries
✓ **Extensible**: Easy to add new domains
✓ **Well-Documented**: Examples, tests, design docs

Maximum developer experience with zero performance cost.
