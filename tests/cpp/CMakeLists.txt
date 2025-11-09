# C++ Tests
# GoogleTest imported targets are configured in root CMakeLists.txt

if(GTest_FOUND)
    # Options Pricing Tests
    add_executable(test_options_pricing
        test_black_scholes.cpp
        test_greeks.cpp
        test_implied_volatility.cpp
    )

    target_link_libraries(test_options_pricing
        PRIVATE
        options_pricing
        GTest::gtest
        GTest::gtest_main
    )

    add_test(NAME OptionsPricingTests COMMAND test_options_pricing)

    # Correlation Engine Tests
    add_executable(test_correlation
        test_pearson.cpp
        test_spearman.cpp
        test_time_lagged.cpp
    )

    target_link_libraries(test_correlation
        PRIVATE
        correlation_engine
        GTest::gtest
        GTest::gtest_main
    )

    add_test(NAME CorrelationTests COMMAND test_correlation)

    # Schwab API Integration Tests
    add_executable(test_order_manager_integration
        test_order_manager_integration.cpp
    )

    target_link_libraries(test_order_manager_integration
        PRIVATE
        schwab_api
        GTest::gtest
        GTest::gtest_main
    )

    add_test(NAME OrderManagerIntegrationTests COMMAND test_order_manager_integration)

    # Schwab Account Manager Integration Tests
    add_executable(test_account_manager_integration
        test_account_manager_integration.cpp
    )

    target_link_libraries(test_account_manager_integration
        PRIVATE
        schwab_api
        GTest::gtest
        GTest::gtest_main
    )

    add_test(NAME AccountManagerIntegrationTests COMMAND test_account_manager_integration)

    # Schwab End-to-End Workflow Tests
    add_executable(test_schwab_e2e_workflow
        test_schwab_e2e_workflow.cpp
    )

    target_link_libraries(test_schwab_e2e_workflow
        PRIVATE
        schwab_api
        risk_management
        GTest::gtest
        GTest::gtest_main
        spdlog::spdlog
    )

    add_test(NAME SchwabE2EWorkflowTests COMMAND test_schwab_e2e_workflow)

else()
    message(STATUS "Google Test not found - C++ tests disabled")
    message(STATUS "Install with: sudo apt-get install libgtest-dev")
endif()
