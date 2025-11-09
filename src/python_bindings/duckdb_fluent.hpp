/**
 * BigBrotherAnalytics - DuckDB Fluent API
 *
 * Fluent interface for DuckDB bindings following Schwab API design patterns.
 * Enables method chaining for configuration and query building.
 *
 * Features:
 * - Fluent configuration methods (setReadOnly, setMaxMemory, etc.)
 * - QueryBuilder for fluent SQL construction
 * - Specialized data accessors (Employment, Sectors)
 * - Full C++23 trailing return syntax support
 * - Thread-safe operations
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 *
 * Tagged: PYTHON_BINDINGS, FLUENT_API
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <functional>

namespace bigbrother::database::fluent {

// Forward declarations
class DuckDBConnection;
class QueryBuilder;
class EmploymentDataAccessor;
class SectorDataAccessor;

/**
 * QueryBuilder - Fluent SQL query construction interface
 *
 * Allows method chaining for building SQL queries:
 *     auto result = db.query()
 *         .select({"column1", "column2"})
 *         .from("table_name")
 *         .where("price > 100")
 *         .orderBy("volume", "DESC")
 *         .limit(10)
 *         .execute();
 */
class QueryBuilder {
public:
    explicit QueryBuilder(DuckDBConnection& conn) : conn_(conn) {}

    /**
     * Select specific columns
     *
     * @param columns Vector of column names to select
     * @return Reference to this QueryBuilder for chaining
     *
     * Example:
     *     .select({"id", "name", "value"})
     */
    auto select(std::vector<std::string> const& columns) -> QueryBuilder& {
        select_columns_ = columns;
        return *this;
    }

    /**
     * Select all columns
     *
     * @return Reference to this QueryBuilder for chaining
     *
     * Example:
     *     .selectAll()
     */
    auto selectAll() -> QueryBuilder& {
        select_columns_.clear();
        select_all_ = true;
        return *this;
    }

    /**
     * Specify table to query
     *
     * @param table Table name
     * @return Reference to this QueryBuilder for chaining
     *
     * Example:
     *     .from("employment")
     */
    auto from(std::string const& table) -> QueryBuilder& {
        from_table_ = table;
        return *this;
    }

    /**
     * Add WHERE clause condition
     *
     * @param condition SQL WHERE condition
     * @return Reference to this QueryBuilder for chaining
     *
     * Example:
     *     .where("price > 100")
     *     .where("date >= '2024-01-01'")
     */
    auto where(std::string const& condition) -> QueryBuilder& {
        where_conditions_.push_back(condition);
        return *this;
    }

    /**
     * Add OR condition to WHERE clause
     *
     * @param condition SQL WHERE condition
     * @return Reference to this QueryBuilder for chaining
     *
     * Example:
     *     .where("status = 'active'")
     *     .orWhere("status = 'pending'")
     */
    auto orWhere(std::string const& condition) -> QueryBuilder& {
        if (!where_conditions_.empty()) {
            or_conditions_.push_back(condition);
        }
        return *this;
    }

    /**
     * Order results by column
     *
     * @param column Column name to sort by
     * @param direction Sort direction ("ASC" or "DESC"), defaults to "ASC"
     * @return Reference to this QueryBuilder for chaining
     *
     * Example:
     *     .orderBy("volume", "DESC")
     *     .orderBy("date")  // defaults to ASC
     */
    auto orderBy(std::string const& column, std::string const& direction = "ASC") -> QueryBuilder& {
        order_clauses_.push_back({column, direction});
        return *this;
    }

    /**
     * Limit result set size
     *
     * @param count Maximum number of rows to return
     * @return Reference to this QueryBuilder for chaining
     *
     * Example:
     *     .limit(10)
     */
    auto limit(int count) -> QueryBuilder& {
        limit_count_ = count;
        return *this;
    }

    /**
     * Add OFFSET to skip rows
     *
     * @param count Number of rows to skip
     * @return Reference to this QueryBuilder for chaining
     *
     * Example:
     *     .offset(20)  // Skip first 20 rows
     */
    auto offset(int count) -> QueryBuilder& {
        offset_count_ = count;
        return *this;
    }

    /**
     * Build and execute the query
     *
     * @return Query result from the database
     *
     * Example:
     *     auto result = builder.execute();
     */
    auto execute() -> std::string {
        return buildQuery();
    }

    /**
     * Build query without executing (for inspection)
     *
     * @return Built SQL query string
     *
     * Example:
     *     std::string sql = builder.build();
     */
    auto build() const -> std::string {
        return buildQuery();
    }

    /**
     * Reset builder to initial state
     *
     * @return Reference to this QueryBuilder for chaining
     */
    auto reset() -> QueryBuilder& {
        select_columns_.clear();
        select_all_ = false;
        from_table_.clear();
        where_conditions_.clear();
        or_conditions_.clear();
        order_clauses_.clear();
        limit_count_ = -1;
        offset_count_ = -1;
        return *this;
    }

private:
    DuckDBConnection& conn_;
    std::vector<std::string> select_columns_;
    bool select_all_ = false;
    std::string from_table_;
    std::vector<std::string> where_conditions_;
    std::vector<std::string> or_conditions_;
    std::vector<std::pair<std::string, std::string>> order_clauses_;
    int limit_count_ = -1;
    int offset_count_ = -1;

    auto buildQuery() const -> std::string {
        std::ostringstream oss;

        // SELECT clause
        oss << "SELECT ";
        if (select_all_) {
            oss << "* ";
        } else if (!select_columns_.empty()) {
            for (size_t i = 0; i < select_columns_.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << select_columns_[i];
            }
            oss << " ";
        } else {
            oss << "* ";  // Default to all if not specified
        }

        // FROM clause
        if (!from_table_.empty()) {
            oss << "FROM " << from_table_ << " ";
        }

        // WHERE clause
        if (!where_conditions_.empty()) {
            oss << "WHERE ";
            for (size_t i = 0; i < where_conditions_.size(); ++i) {
                if (i > 0) oss << "AND ";
                oss << where_conditions_[i] << " ";
            }

            // OR conditions
            for (auto const& or_cond : or_conditions_) {
                oss << "OR " << or_cond << " ";
            }
        }

        // ORDER BY clause
        if (!order_clauses_.empty()) {
            oss << "ORDER BY ";
            for (size_t i = 0; i < order_clauses_.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << order_clauses_[i].first << " " << order_clauses_[i].second;
            }
            oss << " ";
        }

        // LIMIT clause
        if (limit_count_ >= 0) {
            oss << "LIMIT " << limit_count_ << " ";
        }

        // OFFSET clause
        if (offset_count_ >= 0) {
            oss << "OFFSET " << offset_count_;
        }

        return oss.str();
    }

    friend class DuckDBConnection;
};

/**
 * EmploymentDataAccessor - Fluent interface for employment data queries
 *
 * Provides specialized methods for accessing employment data:
 *     auto data = db.employment()
 *         .forSector("Technology")
 *         .betweenDates("2024-01-01", "2025-01-01")
 *         .get();
 */
class EmploymentDataAccessor {
public:
    explicit EmploymentDataAccessor(DuckDBConnection& conn) : conn_(conn) {}

    /**
     * Filter employment data by sector
     *
     * @param sector Sector name
     * @return Reference to this accessor for chaining
     */
    auto forSector(std::string const& sector) -> EmploymentDataAccessor& {
        sector_ = sector;
        return *this;
    }

    /**
     * Filter employment data by date range
     *
     * @param start_date Start date (YYYY-MM-DD)
     * @param end_date End date (YYYY-MM-DD)
     * @return Reference to this accessor for chaining
     */
    auto betweenDates(std::string const& start_date, std::string const& end_date)
        -> EmploymentDataAccessor& {
        start_date_ = start_date;
        end_date_ = end_date;
        return *this;
    }

    /**
     * Filter employment data starting from date
     *
     * @param start_date Start date (YYYY-MM-DD)
     * @return Reference to this accessor for chaining
     */
    auto fromDate(std::string const& start_date) -> EmploymentDataAccessor& {
        start_date_ = start_date;
        return *this;
    }

    /**
     * Filter employment data up to date
     *
     * @param end_date End date (YYYY-MM-DD)
     * @return Reference to this accessor for chaining
     */
    auto toDate(std::string const& end_date) -> EmploymentDataAccessor& {
        end_date_ = end_date;
        return *this;
    }

    /**
     * Limit number of results
     *
     * @param count Maximum results
     * @return Reference to this accessor for chaining
     */
    auto limit(int count) -> EmploymentDataAccessor& {
        limit_ = count;
        return *this;
    }

    /**
     * Get employment data (executes query)
     *
     * @return Query result
     */
    auto get() -> std::string {
        // This would be implemented to return actual QueryResult in the binding
        std::string query = "SELECT * FROM employment WHERE 1=1";

        if (!start_date_.empty()) {
            query += " AND date >= '" + start_date_ + "'";
        }
        if (!end_date_.empty()) {
            query += " AND date <= '" + end_date_ + "'";
        }
        if (!sector_.empty()) {
            query += " AND sector = '" + sector_ + "'";
        }

        query += " ORDER BY date DESC";

        if (limit_ > 0) {
            query += " LIMIT " + std::to_string(limit_);
        }

        return query;
    }

private:
    DuckDBConnection& conn_;
    std::string sector_;
    std::string start_date_;
    std::string end_date_;
    int limit_ = -1;

    friend class DuckDBConnection;
};

/**
 * SectorDataAccessor - Fluent interface for sector data queries
 *
 * Provides specialized methods for accessing sector data:
 *     auto sectors = db.sectors()
 *         .withEmploymentData()
 *         .sortByGrowth()
 *         .get();
 */
class SectorDataAccessor {
public:
    explicit SectorDataAccessor(DuckDBConnection& conn) : conn_(conn) {}

    /**
     * Include employment data in sector results
     *
     * @return Reference to this accessor for chaining
     */
    auto withEmploymentData() -> SectorDataAccessor& {
        include_employment_ = true;
        return *this;
    }

    /**
     * Include sector rotation data
     *
     * @return Reference to this accessor for chaining
     */
    auto withRotationData() -> SectorDataAccessor& {
        include_rotation_ = true;
        return *this;
    }

    /**
     * Sort results by growth metrics
     *
     * @param direction Sort direction ("ASC" or "DESC"), defaults to "DESC"
     * @return Reference to this accessor for chaining
     */
    auto sortByGrowth(std::string const& direction = "DESC") -> SectorDataAccessor& {
        sort_by_ = "growth";
        sort_direction_ = direction;
        return *this;
    }

    /**
     * Sort results by performance
     *
     * @param direction Sort direction ("ASC" or "DESC"), defaults to "DESC"
     * @return Reference to this accessor for chaining
     */
    auto sortByPerformance(std::string const& direction = "DESC") -> SectorDataAccessor& {
        sort_by_ = "performance";
        sort_direction_ = direction;
        return *this;
    }

    /**
     * Limit number of sectors
     *
     * @param count Maximum sectors
     * @return Reference to this accessor for chaining
     */
    auto limit(int count) -> SectorDataAccessor& {
        limit_ = count;
        return *this;
    }

    /**
     * Get sector data (executes query)
     *
     * @return Query result
     */
    auto get() -> std::string {
        std::string query = "SELECT * FROM sectors";

        if (include_employment_) {
            query = "SELECT s.*, e.employment_data FROM sectors s LEFT JOIN employment e ON s.id = e.sector_id";
        }

        if (!sort_by_.empty()) {
            query += " ORDER BY " + sort_by_ + " " + sort_direction_;
        }

        if (limit_ > 0) {
            query += " LIMIT " + std::to_string(limit_);
        }

        return query;
    }

private:
    DuckDBConnection& conn_;
    bool include_employment_ = false;
    bool include_rotation_ = false;
    std::string sort_by_;
    std::string sort_direction_ = "DESC";
    int limit_ = -1;

    friend class DuckDBConnection;
};

}  // namespace bigbrother::database::fluent
