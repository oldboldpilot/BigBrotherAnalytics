/**
 * BigBrotherAnalytics - Error Handling & Retry Logic Example
 *
 * Demonstrates comprehensive error handling including:
 * - Exponential backoff retry
 * - Circuit breaker pattern
 * - Connection monitoring
 * - Database resilience
 * - Graceful degradation
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 * Phase: 4 (Week 2)
 */

import bigbrother.utils.logger;
import bigbrother.utils.retry;
import bigbrother.utils.connection_monitor;
import bigbrother.utils.resilient_database;
import bigbrother.schwab_api;

#include <iostream>
#include <thread>

using namespace bigbrother;

/**
 * Example 1: Basic Retry with Exponential Backoff
 */
auto example_basic_retry() -> void {
    utils::Logger::getInstance().info("=== Example 1: Basic Retry ===");

    int attempt_count = 0;

    // Simulate an operation that fails twice then succeeds
    auto unreliable_operation = [&attempt_count]() -> types::Result<std::string> {
        attempt_count++;
        utils::Logger::getInstance().info("Attempt {}", attempt_count);

        if (attempt_count < 3) {
            return types::makeError<std::string>(
                types::ErrorCode::NetworkError,
                "Simulated transient network error"
            );
        }

        return std::string{"Success!"};
    };

    // Execute with retry
    auto result = utils::retryNetworkOperation<std::string>(
        unreliable_operation,
        "unreliable_operation"
    );

    if (result) {
        utils::Logger::getInstance().info("Result: {}", *result);
    } else {
        utils::Logger::getInstance().error("Failed: {}", result.error().message);
    }

    // Check retry statistics
    auto stats = utils::getRetryEngine().getStats("unreliable_operation");
    if (stats) {
        utils::Logger::getInstance().info(
            "Stats: {} attempts, {} successes, {} retries, success rate: {:.1f}%",
            stats->total_attempts.load(),
            stats->total_successes.load(),
            stats->total_retries.load(),
            stats->getSuccessRate() * 100.0
        );
    }

    std::cout << "\n";
}

/**
 * Example 2: Circuit Breaker
 */
auto example_circuit_breaker() -> void {
    utils::Logger::getInstance().info("=== Example 2: Circuit Breaker ===");

    // Simulate an operation that always fails
    auto failing_operation = []() -> types::Result<int> {
        return types::makeError<int>(
            types::ErrorCode::ServiceUnavailable,
            "Service is down"
        );
    };

    // Try operation 6 times to trigger circuit breaker
    for (int i = 0; i < 6; ++i) {
        utils::Logger::getInstance().info("Attempt {} to call failing service", i + 1);

        auto result = utils::retryNetworkOperation<int>(
            failing_operation,
            "failing_service"
        );

        if (!result) {
            utils::Logger::getInstance().warn("Operation failed: {}", result.error().message);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Check circuit breaker state
    auto breaker_state = utils::getRetryEngine().getCircuitBreakerState("failing_service");
    if (breaker_state) {
        std::string state_str = (*breaker_state == utils::CircuitBreaker::State::Open) ? "OPEN" :
                               (*breaker_state == utils::CircuitBreaker::State::Closed) ? "CLOSED" :
                               "HALF_OPEN";
        utils::Logger::getInstance().info("Circuit breaker state: {}", state_str);
    }

    std::cout << "\n";
}

/**
 * Example 3: Connection Monitoring
 */
auto example_connection_monitoring() -> void {
    utils::Logger::getInstance().info("=== Example 3: Connection Monitoring ===");

    int health_check_count = 0;

    // Health check function that fails every 3rd check
    auto health_check = [&health_check_count]() -> types::Result<void> {
        health_check_count++;
        utils::Logger::getInstance().info("Health check #{}", health_check_count);

        if (health_check_count % 3 == 0) {
            return types::makeError<void>(
                types::ErrorCode::NetworkError,
                "Health check failed"
            );
        }

        return {};
    };

    // Reconnect function
    auto reconnect = []() -> types::Result<void> {
        utils::Logger::getInstance().info("Attempting reconnection...");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return {};
    };

    // Create connection monitor with fast interval for demo
    utils::HeartbeatConfig config;
    config.interval = std::chrono::seconds(2);
    config.timeout = std::chrono::seconds(1);
    config.failure_threshold = 2;

    utils::ConnectionMonitor monitor(
        "test_service",
        health_check,
        reconnect,
        config
    );

    // Start monitoring
    monitor.start();

    // Run for 10 seconds
    utils::Logger::getInstance().info("Monitoring for 10 seconds...");
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // Check final health
    auto health = monitor.getHealth();
    utils::Logger::getInstance().info(
        "Final health: {} total checks, {} successes, {} failures, success rate: {:.1f}%",
        health.total_checks,
        health.total_successes,
        health.total_failures,
        health.getSuccessRate() * 100.0
    );

    monitor.stop();

    std::cout << "\n";
}

/**
 * Example 4: Database Resilience
 */
auto example_database_resilience() -> void {
    utils::Logger::getInstance().info("=== Example 4: Database Resilience ===");

    // Create resilient database
    utils::ResilientDatabaseConfig config;
    config.db_path = "/tmp/test_resilient.duckdb";
    config.enable_in_memory_fallback = true;
    config.max_retry_attempts = 3;

    utils::ResilientDatabase db(config);

    // Connect
    auto connect_result = db.connect();
    if (!connect_result) {
        utils::Logger::getInstance().error("Connection failed: {}", connect_result.error().message);
        return;
    }

    utils::Logger::getInstance().info("Database connected");

    // Create table
    auto create_result = db.execute(R"(
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY,
            value VARCHAR
        )
    )");

    if (create_result) {
        utils::Logger::getInstance().info("Table created");
    }

    // Insert with transaction
    auto transaction_result = db.executeInTransaction([&db]() -> types::Result<void> {
        auto insert1 = db.execute("INSERT INTO test_table VALUES (1, 'First')");
        if (!insert1) return std::unexpected(insert1.error());

        auto insert2 = db.execute("INSERT INTO test_table VALUES (2, 'Second')");
        if (!insert2) return std::unexpected(insert2.error());

        return {};
    });

    if (transaction_result) {
        utils::Logger::getInstance().info("Transaction committed");
    } else {
        utils::Logger::getInstance().error("Transaction failed: {}",
                                         transaction_result.error().message);
    }

    // Query data
    auto query_result = db.query("SELECT * FROM test_table");
    if (query_result) {
        auto& result = *query_result;
        utils::Logger::getInstance().info("Query returned {} rows", result->RowCount());

        for (size_t i = 0; i < result->RowCount(); ++i) {
            auto id = result->GetValue(0, i).GetValue<int>();
            auto value = result->GetValue(1, i).ToString();
            utils::Logger::getInstance().info("  Row {}: id={}, value={}", i + 1, id, value);
        }
    }

    // Show health
    auto health = db.getHealth();
    utils::Logger::getInstance().info(
        "Database health: {} operations, {} successful, success rate: {:.1f}%",
        health.total_operations.load(),
        health.successful_operations.load(),
        health.getSuccessRate() * 100.0
    );

    std::cout << "\n";
}

/**
 * Example 5: Schwab API with Retry Logic
 */
auto example_schwab_with_retry() -> void {
    utils::Logger::getInstance().info("=== Example 5: Schwab API with Retry ===");

    // Note: This example shows the pattern - actual implementation requires credentials

    // Simulate Schwab API call with retry
    auto get_quote = [](std::string const& symbol) -> types::Result<double> {
        utils::Logger::getInstance().info("Fetching quote for {}", symbol);

        // Simulate network call that might fail
        static int attempt = 0;
        attempt++;

        if (attempt < 2) {
            return types::makeError<double>(
                types::ErrorCode::RateLimitExceeded,
                "Rate limit exceeded, please retry"
            );
        }

        // Return simulated price
        return 450.75;
    };

    // Execute with retry
    auto quote_operation = [&]() -> types::Result<double> {
        return get_quote("SPY");
    };

    auto result = utils::retryNetworkOperation<double>(
        quote_operation,
        "schwab_get_quote"
    );

    if (result) {
        utils::Logger::getInstance().info("Quote for SPY: ${:.2f}", *result);
    } else {
        utils::Logger::getInstance().error("Failed to get quote: {}", result.error().message);
    }

    std::cout << "\n";
}

/**
 * Main function
 */
auto main() -> int {
    // Initialize logger
    utils::Logger::getInstance().initialize(
        "logs/error_handling_example.log",
        utils::LogLevel::DEBUG,
        true  // Console output
    );

    utils::Logger::getInstance().info("╔══════════════════════════════════════════════════════════╗");
    utils::Logger::getInstance().info("║   Error Handling & Retry Logic Examples                ║");
    utils::Logger::getInstance().info("╚══════════════════════════════════════════════════════════╝");
    std::cout << "\n";

    try {
        example_basic_retry();
        example_circuit_breaker();
        example_connection_monitoring();
        example_database_resilience();
        example_schwab_with_retry();

        // Print all retry statistics
        utils::Logger::getInstance().info("=== All Retry Statistics ===");
        auto all_stats = utils::getRetryEngine().getAllStats();
        for (auto const& stats : all_stats) {
            utils::Logger::getInstance().info(
                "{}: {} attempts, {} successes, {} failures, {} retries",
                stats.operation_name,
                stats.total_attempts.load(),
                stats.total_successes.load(),
                stats.total_failures.load(),
                stats.total_retries.load()
            );
        }

    } catch (std::exception const& e) {
        utils::Logger::getInstance().error("Example failed: {}", e.what());
        return 1;
    }

    utils::Logger::getInstance().info("\nAll examples completed successfully!");
    return 0;
}
