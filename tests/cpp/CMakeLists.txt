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

    # Risk Management Modules Tests
    add_executable(test_risk_modules
        test_risk_modules.cpp
    )

    target_link_libraries(test_risk_modules
        PRIVATE
        risk_management
        options_pricing
        utils
        GTest::gtest
        GTest::gtest_main
    )

    add_test(NAME RiskModulesTests COMMAND test_risk_modules)

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
    )

    add_test(NAME SchwabE2EWorkflowTests COMMAND test_schwab_e2e_workflow)

    # Yahoo Finance API Tests
    add_executable(test_yahoo_finance
        test_yahoo_finance.cpp
    )

    target_link_libraries(test_yahoo_finance
        PRIVATE
        market_intelligence
        GTest::gtest
        GTest::gtest_main
    )

    add_test(NAME YahooFinanceAPITests COMMAND test_yahoo_finance)

    # Schwab API Tests
    add_executable(test_schwab_api
        test_schwab_api.cpp
    )

    target_link_libraries(test_schwab_api
        PRIVATE
        market_intelligence
        GTest::gtest
        GTest::gtest_main
    )

    add_test(NAME SchwabAPITests COMMAND test_schwab_api)

    # Options Strategies Regression Tests (52 strategies)
    add_executable(test_options_strategies
        test_options_strategies.cpp
    )

    # Match the compilation flags used by options_strategies library
    target_compile_options(test_options_strategies PRIVATE
        -O3
        -march=native
        -mavx2
        -mfma
        -fopenmp-simd
        -ffast-math
        -funroll-loops
        -ftree-vectorize
    )

    target_link_libraries(test_options_strategies
        PRIVATE
        options_strategies
        GTest::gtest
        GTest::gtest_main
    )

    add_test(NAME OptionsStrategiesTests COMMAND test_options_strategies)

    # simdjson Wrapper Unit Tests
    add_executable(test_simdjson_wrapper
        test_simdjson_wrapper.cpp
    )

    target_link_libraries(test_simdjson_wrapper
        PRIVATE
        utils
        GTest::gtest
        GTest::gtest_main
    )

    add_test(NAME SimdJsonWrapperTests COMMAND test_simdjson_wrapper)

else()
    message(STATUS "Google Test not found - C++ tests disabled")
    message(STATUS "Install with: sudo apt-get install libgtest-dev")
endif()
