# C++ Tests

# Find Google Test (optional for now)
find_package(GTest)

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
        GTest::GTest
        GTest::Main
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
        GTest::GTest
        GTest::Main
    )

    add_test(NAME CorrelationTests COMMAND test_correlation)

else()
    message(STATUS "Google Test not found - C++ tests disabled")
    message(STATUS "Install with: sudo apt-get install libgtest-dev")
endif()
