# C++ Benchmarks
# Requires Google Benchmark library built with libc++

# Use manually built Google Benchmark from /usr/local (built with libc++)
if(EXISTS "/usr/local/lib/libbenchmark.a" AND EXISTS "/usr/local/include/benchmark/benchmark.h")
    # JSON Parser Performance Benchmarks (simdjson vs nlohmann/json)
    add_executable(benchmark_json_parsers
        benchmark_json_parsers.cpp
    )

    target_include_directories(benchmark_json_parsers PRIVATE /usr/local/include)

    target_link_libraries(benchmark_json_parsers
        PRIVATE
        utils                  # For simdjson_wrapper
        /usr/local/lib/libbenchmark.a      # Google Benchmark (static, libc++)
        /usr/local/lib/libbenchmark_main.a # Google Benchmark main (static, libc++)
        nlohmann_json::nlohmann_json       # For comparison
        pthread                            # Required by benchmark
        rt                                 # Required by benchmark
    )

    # Match the optimization flags used in production
    target_compile_options(benchmark_json_parsers PRIVATE
        -O3
        -march=native
    )

    message(STATUS "Google Benchmark found - C++ benchmarks enabled")
else()
    message(STATUS "Google Benchmark not found - C++ benchmarks disabled")
    message(STATUS "Build from source with libc++:")
    message(STATUS "  cd /tmp && git clone https://github.com/google/benchmark.git")
    message(STATUS "  cd benchmark && cmake -B build -DCMAKE_CXX_FLAGS=\"-stdlib=libc++\" && cmake --build build && sudo cmake --install build")
endif()
