/**
 * BigBrotherAnalytics - Token Receiver Example
 *
 * Demonstrates usage of the token_receiver.cppm module for OAuth token refresh.
 *
 * This example shows:
 * 1. Creating a TokenReceiver with callback
 * 2. Starting the receiver to listen for tokens
 * 3. Querying the latest token
 * 4. Graceful shutdown
 *
 * To test this example:
 * 1. Build: cd build && cmake --build .
 * 2. Run: ./bin/test_token_receiver
 * 3. In another terminal, send a token:
 *    echo "test_access_token_12345" | nc -U /tmp/bigbrother_token.sock
 *    or
 *    echo "test_access_token_12345" | nc localhost 9999
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: November 13, 2025
 */

#include <chrono>
#include <iostream>
#include <thread>

import bigbrother.utils.token_receiver;
import bigbrother.utils.logger;

using namespace bigbrother::utils;

int main() {
    // Initialize logger
    auto& logger = Logger::getInstance();
    logger.initialize("logs/token_receiver_test.log", LogLevel::DEBUG, true);

    std::cout << "=============================================================\n";
    std::cout << "Token Receiver Test - OAuth Token Refresh Demo\n";
    std::cout << "=============================================================\n\n";

    // Configure token receiver
    TokenReceiverConfig config;
    config.unix_socket_path = "/tmp/bigbrother_token.sock";
    config.tcp_host = "127.0.0.1";
    config.tcp_port = 9999;
    config.prefer_unix_socket = true; // Try Unix socket first
    config.max_token_size = 8192;
    config.accept_timeout = std::chrono::milliseconds(500);
    config.receive_timeout = std::chrono::milliseconds(5000);

    // Create token receiver with callback
    TokenReceiver receiver(
        [&logger](std::string const& token) {
            std::cout << "\n[CALLBACK] New token received!\n";
            std::cout << "Token: " << token.substr(0, std::min<size_t>(50, token.size()));
            if (token.size() > 50) {
                std::cout << "... (truncated)";
            }
            std::cout << "\n";
            std::cout << "Token size: " << token.size() << " bytes\n\n";

            logger.info("Token callback invoked: received {} bytes", token.size());
        },
        config);

    // Start the receiver
    std::cout << "Starting token receiver...\n";
    if (!receiver.start()) {
        std::cerr << "Failed to start token receiver!\n";
        std::cerr << "Error: " << receiver.getLastError() << "\n";
        return 1;
    }

    std::cout << "Token receiver started successfully!\n";
    std::cout << "Listening on:\n";
    std::cout << "  - Unix socket: " << config.unix_socket_path << "\n";
    std::cout << "  - TCP socket: " << config.tcp_host << ":" << config.tcp_port << "\n";
    std::cout << "\nSend tokens using:\n";
    std::cout << "  echo \"your_token_here\" | nc -U " << config.unix_socket_path << "\n";
    std::cout << "  echo \"your_token_here\" | nc " << config.tcp_host << " " << config.tcp_port
              << "\n";
    std::cout << "\nPress Ctrl+C to stop...\n\n";

    // Main loop - monitor for token updates
    auto last_count = receiver.getTokenCount();
    std::string last_token;

    for (int i = 0; i < 120; ++i) { // Run for 2 minutes (120 seconds)
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Check if new token received
        auto current_count = receiver.getTokenCount();
        if (current_count != last_count) {
            std::cout << "[STATUS] Token count: " << current_count << "\n";
            last_count = current_count;

            // Get and display latest token
            auto token = receiver.getLatestToken();
            if (token != last_token) {
                std::cout << "[STATUS] Latest token: "
                          << token.substr(0, std::min<size_t>(50, token.size()));
                if (token.size() > 50) {
                    std::cout << "... (truncated)";
                }
                std::cout << "\n";
                last_token = token;
            }
        }

        // Display progress every 10 seconds
        if ((i + 1) % 10 == 0) {
            std::cout << "[INFO] Running for " << (i + 1) << " seconds, "
                      << "tokens received: " << current_count << "\n";
        }
    }

    // Stop receiver
    std::cout << "\n\nStopping token receiver...\n";
    receiver.stop();

    // Final statistics
    std::cout << "\n=============================================================\n";
    std::cout << "Final Statistics:\n";
    std::cout << "  Total tokens received: " << receiver.getTokenCount() << "\n";

    auto final_token = receiver.getLatestToken();
    if (!final_token.empty()) {
        std::cout << "  Last token: "
                  << final_token.substr(0, std::min<size_t>(50, final_token.size()));
        if (final_token.size() > 50) {
            std::cout << "... (truncated)";
        }
        std::cout << "\n";
        std::cout << "  Token size: " << final_token.size() << " bytes\n";
    } else {
        std::cout << "  No tokens received\n";
    }

    auto error = receiver.getLastError();
    if (!error.empty()) {
        std::cout << "  Last error: " << error << "\n";
    }

    std::cout << "=============================================================\n";
    std::cout << "Token receiver test completed successfully!\n";

    return 0;
}
