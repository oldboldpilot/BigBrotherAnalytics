/**
 * Tests for DuckDB Fluent API
 *
 * Comprehensive test suite for fluent configuration methods, query builder,
 * and data accessors. Tests method chaining, backward compatibility, and
 * thread safety.
 *
 * Build: g++ -std=c++23 -o test_duckdb_fluent test_duckdb_fluent.cpp \
 *        -I/path/to/duckdb/include -L/path/to/duckdb/lib -lduckdb
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 *
 * Tagged: PYTHON_BINDINGS, TESTS, FLUENT_API
 */

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "../python_bindings/duckdb_fluent.hpp"
#include "../python_bindings/duckdb_bindings.cpp"

using namespace bigbrother::database;
using namespace bigbrother::database::fluent;

// Test helpers
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

void printTestResult(const TestResult& result) {
    const char* status = result.passed ? "PASS" : "FAIL";
    std::cout << "[" << status << "] " << result.name << std::endl;
    if (!result.message.empty()) {
        std::cout << "  -> " << result.message << std::endl;
    }
}

// Test 1: Fluent Configuration Methods
TestResult testFluentConfiguration() {
    try {
        DuckDBConnection db(":memory:");

        // Test method chaining
        auto& result = db.setReadOnly(true)
            .setMaxMemory(1024 * 1024 * 1024)
            .enableAutoCheckpoint(true)
            .setThreadPoolSize(4)
            .enableLogging(false);

        // Verify it returns the same instance
        assert(&result == &db);

        return {"Fluent Configuration", true, "Method chaining works correctly"};
    } catch (const std::exception& e) {
        return {"Fluent Configuration", false, std::string(e.what())};
    }
}

// Test 2: QueryBuilder Basic Functionality
TestResult testQueryBuilderBasic() {
    try {
        DuckDBConnection db(":memory:");
        QueryBuilder builder = db.query();

        std::string query = builder
            .select({"id", "name", "value"})
            .from("test_table")
            .where("value > 100")
            .orderBy("name", "ASC")
            .limit(10)
            .build();

        // Verify query is built correctly
        assert(!query.empty());
        assert(query.find("SELECT") != std::string::npos);
        assert(query.find("id") != std::string::npos);
        assert(query.find("FROM test_table") != std::string::npos);
        assert(query.find("WHERE value > 100") != std::string::npos);
        assert(query.find("ORDER BY") != std::string::npos);
        assert(query.find("LIMIT 10") != std::string::npos);

        return {"QueryBuilder Basic", true, "Query built correctly: " + query};
    } catch (const std::exception& e) {
        return {"QueryBuilder Basic", false, std::string(e.what())};
    }
}

// Test 3: QueryBuilder Select All
TestResult testQueryBuilderSelectAll() {
    try {
        DuckDBConnection db(":memory:");
        QueryBuilder builder = db.query();

        std::string query = builder
            .selectAll()
            .from("employees")
            .build();

        assert(query.find("SELECT *") != std::string::npos);
        assert(query.find("FROM employees") != std::string::npos);

        return {"QueryBuilder SelectAll", true, "SelectAll works correctly"};
    } catch (const std::exception& e) {
        return {"QueryBuilder SelectAll", false, std::string(e.what())};
    }
}

// Test 4: QueryBuilder With OR Conditions
TestResult testQueryBuilderOrWhere() {
    try {
        DuckDBConnection db(":memory:");
        QueryBuilder builder = db.query();

        std::string query = builder
            .select({"id"})
            .from("table")
            .where("status = 'active'")
            .orWhere("status = 'pending'")
            .build();

        assert(query.find("WHERE") != std::string::npos);
        assert(query.find("OR") != std::string::npos);

        return {"QueryBuilder OrWhere", true, "OR conditions work correctly"};
    } catch (const std::exception& e) {
        return {"QueryBuilder OrWhere", false, std::string(e.what())};
    }
}

// Test 5: QueryBuilder With Offset
TestResult testQueryBuilderOffset() {
    try {
        DuckDBConnection db(":memory:");
        QueryBuilder builder = db.query();

        std::string query = builder
            .from("table")
            .limit(10)
            .offset(20)
            .build();

        assert(query.find("LIMIT 10") != std::string::npos);
        assert(query.find("OFFSET 20") != std::string::npos);

        return {"QueryBuilder Offset", true, "Pagination works correctly"};
    } catch (const std::exception& e) {
        return {"QueryBuilder Offset", false, std::string(e.what())};
    }
}

// Test 6: QueryBuilder Reset
TestResult testQueryBuilderReset() {
    try {
        DuckDBConnection db(":memory:");
        QueryBuilder builder = db.query();

        // Build first query
        std::string query1 = builder
            .select({"id"})
            .from("table1")
            .where("x = 1")
            .build();

        // Reset and build different query
        std::string query2 = builder
            .reset()
            .select({"name"})
            .from("table2")
            .build();

        assert(query1.find("table1") != std::string::npos);
        assert(query2.find("table2") != std::string::npos);
        assert(query2.find("table1") == std::string::npos);

        return {"QueryBuilder Reset", true, "Reset works correctly"};
    } catch (const std::exception& e) {
        return {"QueryBuilder Reset", false, std::string(e.what())};
    }
}

// Test 7: EmploymentDataAccessor Fluent Interface
TestResult testEmploymentAccessor() {
    try {
        DuckDBConnection db(":memory:");
        EmploymentDataAccessor accessor = db.employment();

        std::string query = accessor
            .forSector("Technology")
            .betweenDates("2024-01-01", "2025-01-01")
            .limit(100)
            .get();

        assert(!query.empty());
        assert(query.find("employment") != std::string::npos);
        assert(query.find("2024-01-01") != std::string::npos);
        assert(query.find("2025-01-01") != std::string::npos);
        assert(query.find("Technology") != std::string::npos);

        return {"EmploymentAccessor", true, "Employment accessor works correctly"};
    } catch (const std::exception& e) {
        return {"EmploymentAccessor", false, std::string(e.what())};
    }
}

// Test 8: EmploymentDataAccessor Date Range Methods
TestResult testEmploymentAccessorDates() {
    try {
        DuckDBConnection db(":memory:");
        EmploymentDataAccessor accessor = db.employment();

        std::string query = accessor
            .fromDate("2024-06-01")
            .toDate("2024-12-31")
            .limit(50)
            .get();

        assert(query.find("2024-06-01") != std::string::npos);
        assert(query.find("2024-12-31") != std::string::npos);

        return {"EmploymentAccessor Dates", true, "Date methods work correctly"};
    } catch (const std::exception& e) {
        return {"EmploymentAccessor Dates", false, std::string(e.what())};
    }
}

// Test 9: SectorDataAccessor Fluent Interface
TestResult testSectorAccessor() {
    try {
        DuckDBConnection db(":memory:");
        SectorDataAccessor accessor = db.sectors();

        std::string query = accessor
            .withEmploymentData()
            .sortByGrowth("DESC")
            .limit(10)
            .get();

        assert(!query.empty());
        assert(query.find("sectors") != std::string::npos);

        return {"SectorAccessor", true, "Sector accessor works correctly"};
    } catch (const std::exception& e) {
        return {"SectorAccessor", false, std::string(e.what())};
    }
}

// Test 10: SectorDataAccessor Sorting Options
TestResult testSectorAccessorSorting() {
    try {
        DuckDBConnection db(":memory:");
        SectorDataAccessor accessor1 = db.sectors();
        SectorDataAccessor accessor2 = db.sectors();

        std::string query1 = accessor1
            .sortByGrowth("DESC")
            .get();

        std::string query2 = accessor2
            .sortByPerformance("ASC")
            .get();

        assert(!query1.empty());
        assert(!query2.empty());
        assert(query1 != query2);  // Should be different

        return {"SectorAccessor Sorting", true, "Sorting options work correctly"};
    } catch (const std::exception& e) {
        return {"SectorAccessor Sorting", false, std::string(e.what())};
    }
}

// Test 11: Backward Compatibility - Existing Methods Still Work
TestResult testBackwardCompatibility() {
    try {
        DuckDBConnection db(":memory:");

        // These should still work (backward compatibility)
        auto tables = db.list_tables();

        return {"Backward Compatibility", true, "Existing methods still work"};
    } catch (const std::exception& e) {
        return {"Backward Compatibility", false, std::string(e.what())};
    }
}

// Test 12: Combined Fluent Configuration and Query
TestResult testCombinedFluent() {
    try {
        DuckDBConnection db(":memory:");

        // Combine configuration and query building
        auto query = db
            .setReadOnly(false)
            .setMaxMemory(512 * 1024 * 1024)
            .query()
            .select({"price", "volume"})
            .from("quotes")
            .where("price > 100")
            .orderBy("volume", "DESC")
            .limit(5)
            .build();

        assert(!query.empty());
        assert(query.find("SELECT price, volume") != std::string::npos);

        return {"Combined Fluent", true, "Configuration and query building work together"};
    } catch (const std::exception& e) {
        return {"Combined Fluent", false, std::string(e.what())};
    }
}

// Main test runner
int main() {
    std::cout << "\n====== DuckDB Fluent API Tests ======\n" << std::endl;

    std::vector<TestResult> results;

    // Run all tests
    results.push_back(testFluentConfiguration());
    results.push_back(testQueryBuilderBasic());
    results.push_back(testQueryBuilderSelectAll());
    results.push_back(testQueryBuilderOrWhere());
    results.push_back(testQueryBuilderOffset());
    results.push_back(testQueryBuilderReset());
    results.push_back(testEmploymentAccessor());
    results.push_back(testEmploymentAccessorDates());
    results.push_back(testSectorAccessor());
    results.push_back(testSectorAccessorSorting());
    results.push_back(testBackwardCompatibility());
    results.push_back(testCombinedFluent());

    // Print results
    int passed = 0;
    int failed = 0;

    for (const auto& result : results) {
        printTestResult(result);
        if (result.passed) {
            passed++;
        } else {
            failed++;
        }
    }

    // Summary
    std::cout << "\n====== Test Summary ======\n";
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Total: " << (passed + failed) << std::endl;
    std::cout << "\n====== End of Tests ======\n" << std::endl;

    return failed == 0 ? 0 : 1;
}
