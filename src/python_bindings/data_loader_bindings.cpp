/**
 * Python bindings for C++ data loading
 * Ensures training and inference use IDENTICAL data ingestion logic
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>

// DuckDB C API
#include <duckdb.h>

namespace py = pybind11;

/**
 * Historical price data (matches bot's data structure)
 */
struct HistoricalData {
    std::vector<std::string> dates;
    std::vector<float> closes;
    std::vector<float> volumes;
};

/**
 * Load historical prices from DuckDB (EXACTLY as the bot does)
 *
 * Query: SELECT date, close, volume FROM stock_prices
 *        WHERE symbol = ? ORDER BY date DESC LIMIT ?
 *
 * Returns data in MOST RECENT FIRST order (same as bot)
 */
class DataLoader {
public:
    /**
     * Load historical data for a single symbol
     *
     * @param db_path Path to DuckDB database
     * @param symbol Symbol to load
     * @param limit Number of days to load (default: 100)
     * @return HistoricalData struct with dates, closes, volumes
     */
    static HistoricalData loadHistoricalPrices(
        std::string const& db_path,
        std::string const& symbol,
        int limit = 100) {

        HistoricalData data;

        // Open database in READ-ONLY mode
        duckdb_database db = nullptr;
        duckdb_connection conn = nullptr;

        duckdb_config config;
        duckdb_create_config(&config);
        duckdb_set_config(config, "access_mode", "READ_ONLY");

        char* error_msg = nullptr;
        if (duckdb_open_ext(db_path.c_str(), &db, config, &error_msg) == DuckDBError) {
            std::string err = error_msg ? error_msg : "Unknown error";
            duckdb_free(error_msg);
            duckdb_destroy_config(&config);
            throw std::runtime_error("Failed to open database: " + err);
        }
        duckdb_destroy_config(&config);

        if (duckdb_connect(db, &conn) == DuckDBError) {
            duckdb_close(&db);
            throw std::runtime_error("Failed to create connection");
        }

        // Execute query (EXACTLY as bot does)
        // ORDER BY date DESC ensures MOST RECENT FIRST (critical for feature extraction!)
        auto query = std::format(
            "SELECT date, close, volume FROM stock_prices "
            "WHERE symbol = '{}' ORDER BY date DESC LIMIT {}",
            symbol, limit
        );

        duckdb_result result;
        if (duckdb_query(conn, query.c_str(), &result) == DuckDBError) {
            std::string err = duckdb_result_error(&result);
            duckdb_destroy_result(&result);
            duckdb_disconnect(&conn);
            duckdb_close(&db);
            throw std::runtime_error("Query failed: " + err);
        }

        // Parse results (column indices: 0=date, 1=close, 2=volume)
        size_t row_count = duckdb_row_count(&result);

        data.dates.reserve(row_count);
        data.closes.reserve(row_count);
        data.volumes.reserve(row_count);

        for (size_t row = 0; row < row_count; ++row) {
            // Get date (convert to string)
            auto date_val = duckdb_value_varchar(&result, 0, row);
            data.dates.push_back(std::string(date_val));
            duckdb_free(date_val);

            // Get close price
            if (!duckdb_value_is_null(&result, 1, row)) {
                double close_val = duckdb_value_double(&result, 1, row);
                data.closes.push_back(static_cast<float>(close_val));
            } else {
                data.closes.push_back(0.0f);
            }

            // Get volume
            if (!duckdb_value_is_null(&result, 2, row)) {
                double volume_val = duckdb_value_double(&result, 2, row);
                data.volumes.push_back(static_cast<float>(volume_val));
            } else {
                data.volumes.push_back(0.0f);
            }
        }

        // Cleanup
        duckdb_destroy_result(&result);
        duckdb_disconnect(&conn);
        duckdb_close(&db);

        return data;
    }

    /**
     * Python-friendly wrapper returning numpy arrays
     */
    static py::tuple loadHistoricalPrices_py(
        std::string const& db_path,
        std::string const& symbol,
        int limit = 100) {

        auto data = loadHistoricalPrices(db_path, symbol, limit);

        // Convert closes to numpy array
        auto closes_array = py::array_t<float>(data.closes.size());
        auto closes_buf = closes_array.request();
        float* closes_ptr = static_cast<float*>(closes_buf.ptr);
        std::copy(data.closes.begin(), data.closes.end(), closes_ptr);

        // Convert volumes to numpy array
        auto volumes_array = py::array_t<float>(data.volumes.size());
        auto volumes_buf = volumes_array.request();
        float* volumes_ptr = static_cast<float*>(volumes_buf.ptr);
        std::copy(data.volumes.begin(), data.volumes.end(), volumes_ptr);

        // Return tuple of (dates, closes, volumes)
        return py::make_tuple(data.dates, closes_array, volumes_array);
    }

    /**
     * Load training data from training_data.duckdb
     * Returns: (dates, symbols, closes, opens, highs, lows, volumes, rsi_14, ...)
     */
    static py::dict loadTrainingData(
        std::string const& db_path,
        std::string const& table_name = "train") {

        // Open database in READ-ONLY mode
        duckdb_database db = nullptr;
        duckdb_connection conn = nullptr;

        duckdb_config config;
        duckdb_create_config(&config);
        duckdb_set_config(config, "access_mode", "READ_ONLY");

        char* error_msg = nullptr;
        if (duckdb_open_ext(db_path.c_str(), &db, config, &error_msg) == DuckDBError) {
            std::string err = error_msg ? error_msg : "Unknown error";
            duckdb_free(error_msg);
            duckdb_destroy_config(&config);
            throw std::runtime_error("Failed to open database: " + err);
        }
        duckdb_destroy_config(&config);

        if (duckdb_connect(db, &conn) == DuckDBError) {
            duckdb_close(&db);
            throw std::runtime_error("Failed to create connection");
        }

        // Query training data
        auto query = std::format("SELECT * FROM {}", table_name);

        duckdb_result result;
        if (duckdb_query(conn, query.c_str(), &result) == DuckDBError) {
            std::string err = duckdb_result_error(&result);
            duckdb_destroy_result(&result);
            duckdb_disconnect(&conn);
            duckdb_close(&db);
            throw std::runtime_error("Query failed: " + err);
        }

        size_t row_count = duckdb_row_count(&result);
        size_t col_count = duckdb_column_count(&result);

        // Create dictionary to return
        py::dict result_dict;

        // Get column names
        for (size_t col = 0; col < col_count; ++col) {
            const char* col_name = duckdb_column_name(&result, col);

            // Create list for this column
            py::list col_data;

            for (size_t row = 0; row < row_count; ++row) {
                if (duckdb_value_is_null(&result, col, row)) {
                    col_data.append(py::none());
                } else {
                    // Try to get as double (works for most numeric types)
                    double val = duckdb_value_double(&result, col, row);
                    col_data.append(val);
                }
            }

            result_dict[col_name] = col_data;
        }

        // Cleanup
        duckdb_destroy_result(&result);
        duckdb_disconnect(&conn);
        duckdb_close(&db);

        return result_dict;
    }
};

PYBIND11_MODULE(data_loader_cpp, m) {
    m.doc() = "C++ data loading module - ensures perfect parity between training and inference";

    m.def("load_historical_prices", &DataLoader::loadHistoricalPrices_py,
          "Load historical prices for a symbol (most recent first, same as bot)",
          py::arg("db_path"),
          py::arg("symbol"),
          py::arg("limit") = 100);

    m.def("load_training_data", &DataLoader::loadTrainingData,
          "Load training data from DuckDB table",
          py::arg("db_path"),
          py::arg("table_name") = "train");
}
