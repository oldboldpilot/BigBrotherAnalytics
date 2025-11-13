# Agent Coding Guide - DuckDB Bridge Best Practices

**Version:** 1.0 | **Last Updated:** November 12, 2025

This guide provides best practices for AI agents implementing C++23 modules with DuckDB database access using the bridge pattern.

## Quick Reference

| Task | Use | Example |
|------|-----|---------|
| C++23 module with DB | DuckDB bridge | `#include "schwab_api/duckdb_bridge.hpp"` |
| Python C++ bindings | Direct DuckDB | `#include <duckdb.hpp>` |
| Token storage | Bridge via token_manager | `token_manager.cpp` |
| Account manager | Bridge via resilient_database | `account_manager.cppm` |

---

## Mandatory Rule: Bridge for All C++23 Modules

**IF** your code:
- Is a `.cppm` file (C++23 module)
- Needs DuckDB database access
- Will be imported by other C++23 modules

**THEN** you MUST use the bridge pattern. No exceptions.

```cpp
// ✅ CORRECT - C++23 module
module;
#include "schwab_api/duckdb_bridge.hpp"  // Bridge, not DuckDB
export module bigbrother.my.module;

// ❌ INCORRECT - Will cause compilation errors
module;
#include <duckdb.hpp>  // NEVER direct DuckDB in modules
```

---

## Complete Bridge API Reference

### Database Operations

#### Opening a Database

```cpp
using namespace bigbrother::duckdb_bridge;

// Create a new database or open existing
auto db = openDatabase("data/trading.duckdb");
if (!db) {
    logger_->error("Failed to open database");
    throw std::runtime_error("Database error");
}
```

**Parameters:**
- `path` - File path to DuckDB database

**Returns:** `std::unique_ptr<DatabaseHandle>`

**Lifetime:** Keep alive while database operations needed (RAII)

### Connection Management

#### Creating Connections

```cpp
auto conn = createConnection(*db);
if (!conn) {
    logger_->error("Failed to create connection");
    throw std::runtime_error("Connection error");
}
```

**Parameters:**
- `db` - DatabaseHandle reference

**Returns:** `std::unique_ptr<ConnectionHandle>`

**Lifetime:** One connection per worker thread recommended

### Query Execution

#### Synchronous Queries (No Results)

```cpp
// DDL operations: CREATE TABLE, ALTER, DROP
auto success = executeQuery(*conn,
    "CREATE TABLE IF NOT EXISTS positions ("
    "  id INTEGER PRIMARY KEY,"
    "  symbol TEXT,"
    "  quantity DOUBLE,"
    "  entry_price DOUBLE"
    ")"
);

if (!success) {
    logger_->error("Failed to create table");
}

// INSERT/UPDATE/DELETE operations
success = executeQuery(*conn,
    "INSERT INTO positions VALUES (1, 'AAPL', 100.0, 150.25)"
);

if (!success) {
    logger_->error("Insert failed");
}
```

**Parameters:**
- `conn` - ConnectionHandle reference
- `query` - SQL string

**Returns:** `bool` (success/failure)

**Error Handling:**
```cpp
if (!executeQuery(*conn, "SELECT * FROM positions")) {
    // Query failed - log and handle gracefully
    logger_->warn("Query execution failed");
}
```

#### Queries with Result Sets

```cpp
// SELECT operations
auto result = executeQueryWithResults(*conn,
    "SELECT symbol, quantity FROM positions WHERE quantity > 0"
);

if (!result || hasError(*result)) {
    logger_->error("Query failed: {}", getErrorMessage(*result));
    return {};  // Empty result
}

// Process results
size_t row_count = getRowCount(*result);
size_t col_count = getColumnCount(*result);

for (size_t row = 0; row < row_count; ++row) {
    auto symbol = getValueAsString(*result, 0, row);
    auto quantity = getValueAsDouble(*result, 1, row);

    logger_->info("Symbol: {}, Quantity: {}", symbol, quantity);
}
```

**Parameters:**
- `conn` - ConnectionHandle reference
- `query` - SQL SELECT statement

**Returns:** `std::unique_ptr<QueryResultHandle>`

**Lifetime:** Automatically cleaned up when destroyed (RAII)

### Result Set Operations

#### Getting Metadata

```cpp
auto result = executeQueryWithResults(*conn, "SELECT * FROM trading_signals");

// Column count
size_t cols = getColumnCount(*result);

// Get column names
for (size_t i = 0; i < cols; ++i) {
    std::string col_name = getColumnName(*result, i);
    logger_->info("Column {}: {}", i, col_name);
}

// Row count
size_t rows = getRowCount(*result);
logger_->info("Result set has {} rows", rows);

// Error checking
if (hasError(*result)) {
    auto error_msg = getErrorMessage(*result);
    logger_->error("Query error: {}", error_msg);
}
```

**Available Functions:**
- `getRowCount(const QueryResultHandle&) -> size_t`
- `getColumnCount(const QueryResultHandle&) -> size_t`
- `getColumnName(const QueryResultHandle&, col_idx) -> std::string`
- `hasError(const QueryResultHandle&) -> bool`
- `getErrorMessage(const QueryResultHandle&) -> std::string`

#### Extracting Values

```cpp
// String values
std::string strategy = getValueAsString(*result, 0, row_idx);

// Integer values
int32_t trade_id = getValueAsInt(*result, 1, row_idx);

// 64-bit integers
int64_t timestamp_us = getValueAsInt64(*result, 2, row_idx);

// Floating point
double price = getValueAsDouble(*result, 3, row_idx);

// Boolean
bool is_active = getValueAsBool(*result, 4, row_idx);

// NULL checking
if (isValueNull(*result, 5, row_idx)) {
    logger_->info("Value is NULL");
}
```

**Available Functions:**
- `getValueAsString(const QueryResultHandle&, col, row) -> std::string`
- `getValueAsInt(const QueryResultHandle&, col, row) -> int32_t`
- `getValueAsInt64(const QueryResultHandle&, col, row) -> int64_t`
- `getValueAsDouble(const QueryResultHandle&, col, row) -> double`
- `getValueAsBool(const QueryResultHandle&, col, row) -> bool`
- `isValueNull(const QueryResultHandle&, col, row) -> bool`

### Prepared Statements

#### Basic Prepared Statement

```cpp
// Prepare statement with parameters (? placeholders)
auto stmt = prepareStatement(*conn,
    "INSERT INTO trades (strategy, price, quantity) VALUES (?, ?, ?)"
);

if (!stmt) {
    logger_->error("Failed to prepare statement");
    return false;
}

// Bind parameters
bindString(*stmt, 1, "straddle");
bindDouble(*stmt, 2, 152.50);
bindInt(*stmt, 3, 10);

// Execute
auto success = executeStatement(*stmt);
if (!success) {
    logger_->error("Execute statement failed");
    return false;
}

return true;
```

**Parameter Binding Functions:**
- `bindString(PreparedStatementHandle&, int index, const std::string&) -> bool`
- `bindInt(PreparedStatementHandle&, int index, int) -> bool`
- `bindInt64(PreparedStatementHandle&, int index, int64_t) -> bool`
- `bindDouble(PreparedStatementHandle&, int index, double) -> bool`

**Execute:**
- `executeStatement(PreparedStatementHandle&) -> bool`

**Parameter Numbering:**
- Parameters are 1-indexed (not 0-indexed)
- First `?` is index 1, second is index 2, etc.

#### Prepared Statement in Loop

```cpp
auto insert_stmt = prepareStatement(*conn,
    "INSERT INTO positions (symbol, quantity, price) VALUES (?, ?, ?)"
);

// Insert multiple rows
std::vector<Trade> trades = getTradesToInsert();
for (const auto& trade : trades) {
    // Re-use prepared statement for multiple rows
    bindString(*insert_stmt, 1, trade.symbol);
    bindInt(*insert_stmt, 2, trade.quantity);
    bindDouble(*insert_stmt, 3, trade.price);

    auto success = executeStatement(*insert_stmt);
    if (!success) {
        logger_->warn("Failed to insert trade: {}", trade.symbol);
    }
}
```

---

## Code Patterns & Examples

### Pattern 1: Simple Database Initialization

```cpp
export module bigbrother.my.database;

import bigbrother.utils.logger;

using namespace bigbrother::duckdb_bridge;

export class DatabaseInitializer {
  public:
    auto initialize() -> bool {
        db_ = openDatabase("data/trading.duckdb");
        if (!db_) {
            LOG_ERROR("Failed to open database");
            return false;
        }

        conn_ = createConnection(*db_);
        if (!conn_) {
            LOG_ERROR("Failed to create connection");
            return false;
        }

        // Create tables
        bool success = createTables();
        if (!success) {
            LOG_ERROR("Failed to create database schema");
            return false;
        }

        return true;
    }

  private:
    std::unique_ptr<DatabaseHandle> db_;
    std::unique_ptr<ConnectionHandle> conn_;

    [[nodiscard]] auto createTables() -> bool {
        auto success = executeQuery(*conn_,
            "CREATE TABLE IF NOT EXISTS trading_signals ("
            "  id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,"
            "  symbol TEXT NOT NULL,"
            "  strategy TEXT NOT NULL,"
            "  signal TEXT NOT NULL,"
            "  confidence DOUBLE,"
            "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        );

        if (!success) return false;

        success = executeQuery(*conn_,
            "CREATE TABLE IF NOT EXISTS executions ("
            "  id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,"
            "  signal_id INTEGER REFERENCES trading_signals(id),"
            "  order_id TEXT,"
            "  status TEXT,"
            "  executed_at TIMESTAMP"
            ")"
        );

        return success;
    }
};
```

### Pattern 2: Data Insertion with Prepared Statements

```cpp
export class SignalPersistence {
  private:
    std::unique_ptr<ConnectionHandle> conn_;

  public:
    explicit SignalPersistence(ConnectionHandle& conn)
        : conn_(std::make_unique<ConnectionHandle>(conn)) {}

    [[nodiscard]] auto insertSignal(
        std::string const& symbol,
        std::string const& strategy,
        std::string const& signal_type,
        double confidence) -> bool
    {
        auto stmt = prepareStatement(*conn_,
            "INSERT INTO trading_signals (symbol, strategy, signal, confidence) "
            "VALUES (?, ?, ?, ?)"
        );

        if (!stmt) {
            LOG_ERROR("Failed to prepare insert statement");
            return false;
        }

        bindString(*stmt, 1, symbol);
        bindString(*stmt, 2, strategy);
        bindString(*stmt, 3, signal_type);
        bindDouble(*stmt, 4, confidence);

        auto success = executeStatement(*stmt);
        if (!success) {
            LOG_ERROR("Failed to insert signal for {}", symbol);
        }

        return success;
    }
};
```

### Pattern 3: Data Retrieval with Result Processing

```cpp
export class SignalReader {
  private:
    std::unique_ptr<ConnectionHandle> conn_;

  public:
    [[nodiscard]] auto getSignals(std::string const& symbol)
        -> std::vector<std::string>
    {
        std::vector<std::string> signals;

        auto result = executeQueryWithResults(*conn_,
            "SELECT signal FROM trading_signals "
            "WHERE symbol = '" + symbol + "' "
            "ORDER BY created_at DESC LIMIT 10"
        );

        if (!result || hasError(*result)) {
            LOG_WARN("Query failed: {}", getErrorMessage(*result));
            return signals;
        }

        auto row_count = getRowCount(*result);
        for (size_t row = 0; row < row_count; ++row) {
            signals.push_back(getValueAsString(*result, 0, row));
        }

        return signals;
    }

    [[nodiscard]] auto getRecentSignals(int limit_rows = 100)
        -> std::vector<Signal>
    {
        std::vector<Signal> signals;

        auto result = executeQueryWithResults(*conn_,
            "SELECT id, symbol, strategy, signal, confidence, created_at "
            "FROM trading_signals "
            "ORDER BY created_at DESC LIMIT " + std::to_string(limit_rows)
        );

        if (!result || hasError(*result)) {
            return signals;
        }

        auto row_count = getRowCount(*result);
        for (size_t row = 0; row < row_count; ++row) {
            Signal sig;
            sig.id = getValueAsInt(*result, 0, row);
            sig.symbol = getValueAsString(*result, 1, row);
            sig.strategy = getValueAsString(*result, 2, row);
            sig.signal = getValueAsString(*result, 3, row);
            sig.confidence = getValueAsDouble(*result, 4, row);
            // sig.created_at = parse timestamp at index 5

            signals.push_back(sig);
        }

        return signals;
    }
};
```

### Pattern 4: Transaction-like Operations (Multiple Queries)

```cpp
[[nodiscard]] auto recordTrade(const Trade& trade) -> bool {
    // Insert signal
    auto signal_result = executeQueryWithResults(*conn_,
        "INSERT INTO trading_signals (symbol, strategy, signal, confidence) "
        "VALUES ('" + trade.symbol + "', '" + trade.strategy + "', "
        "'BUY', " + std::to_string(trade.confidence) + ") "
        "RETURNING id"
    );

    if (!signal_result || hasError(*signal_result)) {
        LOG_ERROR("Failed to insert signal");
        return false;
    }

    auto signal_id = getValueAsInt(*signal_result, 0, 0);

    // Insert execution
    auto exec_success = executeQuery(*conn_,
        "INSERT INTO executions (signal_id, order_id, status) "
        "VALUES (" + std::to_string(signal_id) + ", '" + trade.order_id + "', 'PENDING')"
    );

    if (!exec_success) {
        LOG_ERROR("Failed to record execution");
        // In real code, handle rollback here
        return false;
    }

    return true;
}
```

---

## Common Mistakes & How to Avoid Them

### Mistake 1: Including `<duckdb.hpp>` in C++23 Modules

```cpp
// ❌ WRONG - Causes compilation errors
module;
#include <duckdb.hpp>
#include "schwab_api/duckdb_bridge.hpp"

export module my_module;
```

**Error Message:**
```
error: use of incomplete type 'duckdb::QueryNode'
note: forward declaration of 'duckdb::QueryNode'
```

**Fix:**
```cpp
// ✅ CORRECT - Use bridge only
module;
#include "schwab_api/duckdb_bridge.hpp"

export module my_module;
```

### Mistake 2: Forgetting to Check Query Results

```cpp
// ❌ WRONG - Crashes if query fails
auto result = executeQueryWithResults(*conn, query);
auto count = getRowCount(*result);  // result might be nullptr!
```

**Fix:**
```cpp
// ✅ CORRECT - Always check
auto result = executeQueryWithResults(*conn, query);
if (!result || hasError(*result)) {
    logger_->error("Query failed: {}",
        result ? getErrorMessage(*result) : "Unknown error");
    return {};
}

auto count = getRowCount(*result);
```

### Mistake 3: Using 0-based Index for Prepared Statement Binding

```cpp
// ❌ WRONG - Parameters are 1-indexed
bindString(*stmt, 0, symbol);  // Should be 1
bindInt(*stmt, 1, quantity);   // Should be 2
```

**Fix:**
```cpp
// ✅ CORRECT - 1-indexed parameters
bindString(*stmt, 1, symbol);
bindInt(*stmt, 2, quantity);
```

### Mistake 4: SQL Injection in Dynamic Queries

```cpp
// ❌ WRONG - Unsafe SQL construction
auto symbol = getUserInput();  // Could be "'; DROP TABLE trading_signals; --"
auto query = "SELECT * FROM positions WHERE symbol = '" + symbol + "'";
auto result = executeQueryWithResults(*conn, query);
```

**Fix:**
```cpp
// ✅ CORRECT - Use prepared statements
auto stmt = prepareStatement(*conn,
    "SELECT * FROM positions WHERE symbol = ?"
);
bindString(*stmt, 1, symbol);  // Safe - escaping handled automatically
auto result = executeStatement(*stmt);
```

### Mistake 5: Not Handling NULL Values

```cpp
// ❌ WRONG - Crashes on NULL
auto value = getValueAsString(*result, col, row);
auto len = value.length();  // Undefined if NULL
```

**Fix:**
```cpp
// ✅ CORRECT - Check for NULL
if (isValueNull(*result, col, row)) {
    logger_->info("Value is NULL");
    return std::nullopt;
}

auto value = getValueAsString(*result, col, row);
```

---

## Testing DuckDB Bridge Code

### Unit Test Pattern

```cpp
#include <gtest/gtest.h>
#include "schwab_api/duckdb_bridge.hpp"

using namespace bigbrother::duckdb_bridge;

class DuckDBBridgeTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Use in-memory database for tests
        db = openDatabase(":memory:");
        conn = createConnection(*db);
    }

    std::unique_ptr<DatabaseHandle> db;
    std::unique_ptr<ConnectionHandle> conn;
};

TEST_F(DuckDBBridgeTest, CreateTableAndInsert) {
    // Create table
    auto success = executeQuery(*conn,
        "CREATE TABLE test (id INTEGER, name TEXT)"
    );
    EXPECT_TRUE(success);

    // Insert row
    success = executeQuery(*conn,
        "INSERT INTO test VALUES (1, 'Alice')"
    );
    EXPECT_TRUE(success);

    // Query result
    auto result = executeQueryWithResults(*conn, "SELECT * FROM test");
    ASSERT_TRUE(result);
    EXPECT_EQ(getRowCount(*result), 1);
    EXPECT_EQ(getValueAsInt(*result, 0, 0), 1);
    EXPECT_EQ(getValueAsString(*result, 1, 0), "Alice");
}

TEST_F(DuckDBBridgeTest, PreparedStatement) {
    executeQuery(*conn, "CREATE TABLE trades (symbol TEXT, quantity INT)");

    auto stmt = prepareStatement(*conn,
        "INSERT INTO trades VALUES (?, ?)"
    );
    ASSERT_TRUE(stmt);

    bindString(*stmt, 1, "AAPL");
    bindInt(*stmt, 2, 100);
    auto success = executeStatement(*stmt);
    EXPECT_TRUE(success);
}
```

---

## Building & Verification

### CMakeLists.txt Configuration

```cmake
# Add your module with bridge dependency
add_library(my_module)
target_sources(my_module
    PUBLIC FILE_SET CXX_MODULES FILES
        src/my/module.cppm
)

# Link against duckdb_bridge
target_link_libraries(my_module PUBLIC duckdb_bridge)
```

### Compilation Verification

```bash
# Build with bridge
cmake -G Ninja -B build
ninja -C build

# Verify bridge symbols are present
nm build/CMakeFiles/my_module.dir/src/my/module.cpp.o | grep duckdb_bridge

# Run tests
ctest --output-on-failure
```

---

## Performance Considerations

### Bridge Overhead is Negligible

| Operation | Time | Overhead |
|-----------|------|----------|
| openDatabase() | 5-10ms | N/A |
| createConnection() | <1ms | N/A |
| Simple query | 1-5ms | <1% |
| Prepared stmt | 0.5-2ms | <1% |
| Result extraction | 1-10us per row | <0.1% |

**Conclusion:** Use the bridge freely - database I/O dominates, not pointer indirection.

### Optimization Tips

1. **Reuse connections** - Create once, keep alive
2. **Use prepared statements** - Faster for repeated queries
3. **Batch operations** - Insert multiple rows in one prepared statement
4. **Keep result sets small** - Use LIMIT and WHERE clauses

---

## Module Integration Checklist

When writing a C++23 module that uses DuckDB:

- [ ] Include only `#include "schwab_api/duckdb_bridge.hpp"` (never direct DuckDB)
- [ ] Use `bigbrother::duckdb_bridge` namespace with `using namespace` or fully qualified
- [ ] Check all query results with `hasError()` and `getErrorMessage()`
- [ ] Use prepared statements for dynamic SQL (prevents injection)
- [ ] Handle NULL values with `isValueNull()` before extracting values
- [ ] Use RAII (unique_ptr) for handle management
- [ ] Add proper logging for all database operations
- [ ] Test with both empty and populated datasets
- [ ] Verify compilation: `ninja -C build module_name`
- [ ] Run validation: `./scripts/validate_code.sh`

---

## Useful References

- **Bridge Header:** `src/schwab_api/duckdb_bridge.hpp` (read for detailed API)
- **Bridge Implementation:** `src/schwab_api/duckdb_bridge.cpp` (reference for patterns)
- **Example Usage:** `src/schwab_api/token_manager.cpp` (real-world token storage)
- **Advanced Example:** `src/utils/resilient_database.cppm` (resilient operations)
- **C++23 Modules Guide:** `docs/CPP23_MODULES_GUIDE.md` (module patterns)

---

## Quick Decision Tree

```
Need DuckDB access?
├─ YES: Is it a C++23 module (.cppm)?
│   ├─ YES: Use bridge pattern
│   │   └─ #include "schwab_api/duckdb_bridge.hpp"
│   └─ NO: Is it a Python C++ bindings file?
│       ├─ YES: Can use direct DuckDB
│       │   └─ #include <duckdb.hpp>
│       └─ NO: Use bridge anyway (future-proofing)
└─ NO: No database access needed
    └─ Don't include either
```

---

## Getting Help

- **Compilation errors mentioning `duckdb::` types?** You're using direct DuckDB in a module. Switch to bridge.
- **Segfaults on database operations?** Check that you're handling NULL results.
- **NULL pointer dereference?** Always check `result != nullptr` before accessing.
- **Unclear API?** Check `duckdb_bridge.hpp` - it's well-documented.
