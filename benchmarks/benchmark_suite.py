#!/usr/bin/env python3
"""
BigBrotherAnalytics - Comprehensive Performance Benchmark Suite

Validates the 50-100x speedup claims for all Python bindings:
- DuckDB: 5-10x faster target
- Correlation: 60-100x faster target (OpenMP)
- Risk/Options: 50-100x faster target
- GIL-free execution verification

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import time
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import traceback

# Import C++ bindings
try:
    import bigbrother_correlation as corr_cpp
    CORRELATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Correlation bindings not available: {e}")
    CORRELATION_AVAILABLE = False

try:
    import bigbrother_options as opts_cpp
    OPTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Options bindings not available: {e}")
    OPTIONS_AVAILABLE = False

try:
    import bigbrother_risk as risk_cpp
    RISK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Risk bindings not available: {e}")
    RISK_AVAILABLE = False

try:
    import bigbrother_duckdb as db_cpp
    DUCKDB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DuckDB bindings not available: {e}")
    DUCKDB_AVAILABLE = False

# Pure Python implementations for comparison
try:
    import duckdb
    DUCKDB_PY_AVAILABLE = True
except ImportError:
    print("Warning: Pure Python DuckDB not available")
    DUCKDB_PY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: SciPy not available")
    SCIPY_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""
    name: str
    implementation: str
    data_size: str
    execution_time: float
    iterations: int
    avg_time: float
    std_dev: float
    min_time: float
    max_time: float
    memory_mb: float = 0.0
    additional_info: Dict = None


class BenchmarkRunner:
    """Manages benchmark execution and statistics"""

    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []

    def time_function(self, func: Callable, *args, **kwargs) -> float:
        """Time a single function call"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return end - start

    def run_benchmark(
        self,
        name: str,
        implementation: str,
        data_size: str,
        func: Callable,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Run a benchmark with warmup and multiple iterations"""

        # Warmup
        for _ in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"  Error during warmup: {e}")
                raise

        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            try:
                elapsed = self.time_function(func, *args, **kwargs)
                times.append(elapsed)
            except Exception as e:
                print(f"  Error during benchmark: {e}")
                raise

        times = np.array(times)

        result = BenchmarkResult(
            name=name,
            implementation=implementation,
            data_size=data_size,
            execution_time=np.sum(times),
            iterations=self.benchmark_runs,
            avg_time=np.mean(times),
            std_dev=np.std(times),
            min_time=np.min(times),
            max_time=np.max(times)
        )

        self.results.append(result)
        return result

    def print_result(self, result: BenchmarkResult):
        """Print a single benchmark result"""
        print(f"  {result.implementation:20s} | "
              f"{result.avg_time*1000:8.2f} ms | "
              f"std={result.std_dev*1000:6.2f} ms | "
              f"min={result.min_time*1000:8.2f} ms")

    def compare_results(self, baseline: BenchmarkResult, optimized: BenchmarkResult):
        """Compare two benchmark results"""
        speedup = baseline.avg_time / optimized.avg_time
        improvement_pct = ((baseline.avg_time - optimized.avg_time) / baseline.avg_time) * 100

        print(f"\n  Speedup: {speedup:.2f}x")
        print(f"  Improvement: {improvement_pct:.1f}%")
        print(f"  Time saved per call: {(baseline.avg_time - optimized.avg_time)*1000:.2f} ms")

        return speedup


# ============================================================================
# CORRELATION BENCHMARKS
# ============================================================================

def benchmark_correlation(runner: BenchmarkRunner):
    """Benchmark correlation calculations"""

    if not CORRELATION_AVAILABLE:
        print("\nSkipping correlation benchmarks (bindings not available)")
        return

    print("\n" + "="*80)
    print("CORRELATION BENCHMARKS")
    print("="*80)

    # Test different data sizes
    sizes = {
        'small': 100,
        'medium': 1_000,
        'large': 10_000,
        'xlarge': 100_000
    }

    for size_name, n in sizes.items():
        print(f"\n{size_name.upper()}: {n:,} data points")
        print("-" * 80)

        # Generate test data
        np.random.seed(42)
        x = np.random.randn(n).tolist()
        y = np.random.randn(n).tolist()

        # Python implementation (NumPy/SciPy)
        def python_pearson():
            return np.corrcoef(x, y)[0, 1]

        try:
            result_py = runner.run_benchmark(
                "Pearson Correlation",
                "NumPy",
                size_name,
                python_pearson
            )
            runner.print_result(result_py)
        except Exception as e:
            print(f"  NumPy failed: {e}")
            result_py = None

        # C++ implementation
        def cpp_pearson():
            return corr_cpp.pearson(x, y)

        try:
            result_cpp = runner.run_benchmark(
                "Pearson Correlation",
                "C++ Bindings",
                size_name,
                cpp_pearson
            )
            runner.print_result(result_cpp)
        except Exception as e:
            print(f"  C++ failed: {e}")
            result_cpp = None

        # Compare
        if result_py and result_cpp:
            runner.compare_results(result_py, result_cpp)

    # Spearman correlation
    print("\n" + "-"*80)
    print("SPEARMAN CORRELATION")
    print("-" * 80)

    n = 10_000
    np.random.seed(42)
    x = np.random.randn(n).tolist()
    y = np.random.randn(n).tolist()

    if SCIPY_AVAILABLE:
        def python_spearman():
            return stats.spearmanr(x, y)[0]

        try:
            result_py = runner.run_benchmark(
                "Spearman Correlation",
                "SciPy",
                "10K",
                python_spearman
            )
            runner.print_result(result_py)
        except Exception as e:
            print(f"  SciPy failed: {e}")
            result_py = None
    else:
        result_py = None

    def cpp_spearman():
        return corr_cpp.spearman(x, y)

    try:
        result_cpp = runner.run_benchmark(
            "Spearman Correlation",
            "C++ Bindings",
            "10K",
            cpp_spearman
        )
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ failed: {e}")
        result_cpp = None

    if result_py and result_cpp:
        runner.compare_results(result_py, result_cpp)

    # Correlation matrix (OpenMP parallelized)
    print("\n" + "-"*80)
    print("CORRELATION MATRIX (OpenMP Parallel)")
    print("-" * 80)

    n_symbols = 50
    n_points = 1000

    np.random.seed(42)
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    data = [np.random.randn(n_points).tolist() for _ in range(n_symbols)]

    # Python implementation
    def python_corr_matrix():
        df = pd.DataFrame({sym: d for sym, d in zip(symbols, data)})
        return df.corr().values

    try:
        result_py = runner.run_benchmark(
            "Correlation Matrix",
            "Pandas",
            f"{n_symbols} symbols",
            python_corr_matrix
        )
        runner.print_result(result_py)
    except Exception as e:
        print(f"  Pandas failed: {e}")
        result_py = None

    # C++ implementation
    def cpp_corr_matrix():
        return corr_cpp.correlation_matrix(symbols, data, "pearson")

    try:
        result_cpp = runner.run_benchmark(
            "Correlation Matrix",
            "C++ Bindings (OpenMP)",
            f"{n_symbols} symbols",
            cpp_corr_matrix
        )
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ failed: {e}")
        result_cpp = None

    if result_py and result_cpp:
        runner.compare_results(result_py, result_cpp)


# ============================================================================
# OPTIONS PRICING BENCHMARKS
# ============================================================================

def benchmark_options(runner: BenchmarkRunner):
    """Benchmark options pricing"""

    if not OPTIONS_AVAILABLE:
        print("\nSkipping options benchmarks (bindings not available)")
        return

    print("\n" + "="*80)
    print("OPTIONS PRICING BENCHMARKS")
    print("="*80)

    # Pure Python Black-Scholes implementation
    def python_black_scholes_call(S, K, r, T, sigma):
        """Pure Python Black-Scholes implementation"""
        from math import log, sqrt, exp
        from scipy.stats import norm

        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        return S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)

    # Test parameters
    S, K, r, T, sigma = 100.0, 105.0, 0.041, 1.0, 0.25

    # Benchmark Black-Scholes
    print("\nBLACK-SCHOLES (European Options)")
    print("-" * 80)

    if SCIPY_AVAILABLE:
        def py_bs():
            return python_black_scholes_call(S, K, r, T, sigma)

        try:
            result_py = runner.run_benchmark(
                "Black-Scholes Call",
                "Pure Python",
                "single",
                py_bs
            )
            runner.print_result(result_py)
        except Exception as e:
            print(f"  Pure Python failed: {e}")
            result_py = None
    else:
        result_py = None

    def cpp_bs():
        return opts_cpp.black_scholes_call(S, K, sigma, T, r)

    try:
        result_cpp = runner.run_benchmark(
            "Black-Scholes Call",
            "C++ Bindings",
            "single",
            cpp_bs
        )
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ failed: {e}")
        result_cpp = None

    if result_py and result_cpp:
        runner.compare_results(result_py, result_cpp)

    # Benchmark Trinomial Tree
    print("\nTRINOMIAL TREE (American Options)")
    print("-" * 80)

    def cpp_trinomial():
        return opts_cpp.trinomial_call(S, K, sigma, T, r, steps=100)

    try:
        result_cpp = runner.run_benchmark(
            "Trinomial Tree Call",
            "C++ Bindings",
            "100 steps",
            cpp_trinomial
        )
        runner.print_result(result_cpp)
        print(f"  (No pure Python comparison - too slow)")
    except Exception as e:
        print(f"  C++ failed: {e}")

    # Benchmark Greeks
    print("\nGREEKS CALCULATION")
    print("-" * 80)

    def cpp_greeks():
        return opts_cpp.calculate_greeks(S, K, sigma, T, r)

    try:
        result_cpp = runner.run_benchmark(
            "Greeks Calculation",
            "C++ Bindings",
            "5 greeks",
            cpp_greeks
        )
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ failed: {e}")

    # Batch pricing
    print("\nBATCH PRICING (1000 options)")
    print("-" * 80)

    n_options = 1000
    strikes = np.linspace(90, 110, n_options)

    if SCIPY_AVAILABLE:
        def py_batch():
            return [python_black_scholes_call(S, k, r, T, sigma) for k in strikes]

        try:
            result_py = runner.run_benchmark(
                "Batch Black-Scholes",
                "Pure Python",
                "1000 options",
                py_batch
            )
            runner.print_result(result_py)
        except Exception as e:
            print(f"  Pure Python failed: {e}")
            result_py = None
    else:
        result_py = None

    def cpp_batch():
        return [opts_cpp.black_scholes_call(S, k, sigma, T, r) for k in strikes]

    try:
        result_cpp = runner.run_benchmark(
            "Batch Black-Scholes",
            "C++ Bindings",
            "1000 options",
            cpp_batch
        )
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ failed: {e}")
        result_cpp = None

    if result_py and result_cpp:
        runner.compare_results(result_py, result_cpp)


# ============================================================================
# RISK MANAGEMENT BENCHMARKS
# ============================================================================

def benchmark_risk(runner: BenchmarkRunner):
    """Benchmark risk management functions"""

    if not RISK_AVAILABLE:
        print("\nSkipping risk benchmarks (bindings not available)")
        return

    print("\n" + "="*80)
    print("RISK MANAGEMENT BENCHMARKS")
    print("="*80)

    # Kelly Criterion
    print("\nKELLY CRITERION")
    print("-" * 80)

    def python_kelly(win_prob, win_loss_ratio):
        """Pure Python Kelly Criterion"""
        return max(0.0, min(0.25, (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio))

    win_prob, wl_ratio = 0.65, 2.0

    def py_kelly():
        return python_kelly(win_prob, wl_ratio)

    try:
        result_py = runner.run_benchmark(
            "Kelly Criterion",
            "Pure Python",
            "single",
            py_kelly
        )
        runner.print_result(result_py)
    except Exception as e:
        print(f"  Pure Python failed: {e}")
        result_py = None

    def cpp_kelly():
        return risk_cpp.kelly_criterion(win_prob, wl_ratio)

    try:
        result_cpp = runner.run_benchmark(
            "Kelly Criterion",
            "C++ Bindings",
            "single",
            cpp_kelly
        )
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ failed: {e}")
        result_cpp = None

    if result_py and result_cpp:
        runner.compare_results(result_py, result_cpp)

    # Monte Carlo Simulation
    print("\nMONTE CARLO SIMULATION (OpenMP Parallel)")
    print("-" * 80)

    spot, vol, drift = 100.0, 0.25, 0.05

    # Pure Python Monte Carlo
    def python_monte_carlo(S, sigma, mu, n_sims):
        """Pure Python Monte Carlo"""
        np.random.seed(42)
        T = 0.25
        dt = T / 100

        results = []
        for _ in range(n_sims):
            price = S
            for _ in range(100):
                dW = np.random.randn() * np.sqrt(dt)
                price *= np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)
            results.append(price - S)

        results = np.array(results)
        return {
            'expected_value': np.mean(results),
            'std_deviation': np.std(results),
            'probability_of_profit': np.sum(results > 0) / n_sims,
            'var_95': np.percentile(results, 5),
            'var_99': np.percentile(results, 1)
        }

    for n_sims in [1_000, 10_000, 50_000]:
        print(f"\n  Simulations: {n_sims:,}")

        def py_mc():
            return python_monte_carlo(spot, vol, drift, n_sims)

        try:
            result_py = runner.run_benchmark(
                "Monte Carlo",
                "Pure Python",
                f"{n_sims} sims",
                py_mc
            )
            runner.print_result(result_py)
        except Exception as e:
            print(f"    Pure Python failed: {e}")
            result_py = None

        def cpp_mc():
            return risk_cpp.monte_carlo(spot, vol, drift, n_sims)

        try:
            result_cpp = runner.run_benchmark(
                "Monte Carlo",
                "C++ Bindings (OpenMP)",
                f"{n_sims} sims",
                cpp_mc
            )
            runner.print_result(result_cpp)
        except Exception as e:
            print(f"    C++ failed: {e}")
            result_cpp = None

        if result_py and result_cpp:
            runner.compare_results(result_py, result_cpp)


# ============================================================================
# DUCKDB BENCHMARKS
# ============================================================================

def benchmark_duckdb(runner: BenchmarkRunner):
    """Benchmark DuckDB bindings"""

    if not DUCKDB_AVAILABLE or not DUCKDB_PY_AVAILABLE:
        print("\nSkipping DuckDB benchmarks (bindings not available)")
        return

    print("\n" + "="*80)
    print("DUCKDB BENCHMARKS")
    print("="*80)

    db_path = 'data/bigbrother.duckdb'

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    # Test queries
    queries = [
        ("Row Count", "SELECT COUNT(*) FROM sector_employment_raw"),
        ("Aggregate", "SELECT sector_code, AVG(employment_count) FROM sector_employment_raw GROUP BY sector_code"),
        ("Filter + Sort", "SELECT * FROM sector_employment_raw WHERE employment_count > 100000 ORDER BY report_date DESC LIMIT 1000"),
    ]

    for query_name, query in queries:
        print(f"\n{query_name.upper()}")
        print("-" * 80)

        # Pure Python DuckDB
        def py_query():
            conn = duckdb.connect(db_path, read_only=True)
            result = conn.execute(query).fetchall()
            conn.close()
            return result

        try:
            result_py = runner.run_benchmark(
                query_name,
                "Python DuckDB",
                "full table",
                py_query
            )
            runner.print_result(result_py)
        except Exception as e:
            print(f"  Python DuckDB failed: {e}")
            result_py = None

        # C++ bindings
        def cpp_query():
            conn = db_cpp.connect(db_path)
            result = conn.execute(query)
            return result.to_pandas_dict()

        try:
            result_cpp = runner.run_benchmark(
                query_name,
                "C++ Bindings",
                "full table",
                cpp_query
            )
            runner.print_result(result_cpp)
        except Exception as e:
            print(f"  C++ failed: {e}")
            result_cpp = None

        if result_py and result_cpp:
            runner.compare_results(result_py, result_cpp)


# ============================================================================
# GIL-FREE MULTI-THREADING TEST
# ============================================================================

def benchmark_gil_free(runner: BenchmarkRunner):
    """Test GIL-free execution with multi-threading"""

    if not CORRELATION_AVAILABLE:
        print("\nSkipping GIL-free benchmarks (bindings not available)")
        return

    print("\n" + "="*80)
    print("GIL-FREE MULTI-THREADING TEST")
    print("="*80)

    n = 10_000
    np.random.seed(42)
    data = [(np.random.randn(n).tolist(), np.random.randn(n).tolist()) for _ in range(100)]

    # Single-threaded
    print("\nSINGLE-THREADED (100 correlations)")
    print("-" * 80)

    def single_threaded():
        results = []
        for x, y in data:
            results.append(corr_cpp.pearson(x, y))
        return results

    try:
        result_single = runner.run_benchmark(
            "100 Correlations",
            "Single Thread",
            "100 x 10K",
            single_threaded
        )
        runner.print_result(result_single)
    except Exception as e:
        print(f"  Failed: {e}")
        result_single = None

    # Multi-threaded
    print("\nMULTI-THREADED (100 correlations, 4 threads)")
    print("-" * 80)

    def multi_threaded():
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(corr_cpp.pearson, x, y) for x, y in data]
            results = [f.result() for f in futures]
        return results

    try:
        result_multi = runner.run_benchmark(
            "100 Correlations",
            "4 Threads (GIL-free)",
            "100 x 10K",
            multi_threaded
        )
        runner.print_result(result_multi)
    except Exception as e:
        print(f"  Failed: {e}")
        result_multi = None

    if result_single and result_multi:
        speedup = runner.compare_results(result_single, result_multi)
        print(f"\n  GIL-free verification: {'PASS' if speedup > 1.5 else 'FAIL'}")
        print(f"  Expected ~4x with 4 threads, got {speedup:.2f}x")


# ============================================================================
# MAIN BENCHMARK SUITE
# ============================================================================

def main():
    """Run complete benchmark suite"""

    print("="*80)
    print("BigBrotherAnalytics - Comprehensive Performance Benchmark Suite")
    print("="*80)
    print(f"\nEnvironment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")
    print(f"  Pandas: {pd.__version__}")

    print(f"\nBindings Status:")
    print(f"  Correlation: {'Available' if CORRELATION_AVAILABLE else 'Not Available'}")
    print(f"  Options: {'Available' if OPTIONS_AVAILABLE else 'Not Available'}")
    print(f"  Risk: {'Available' if RISK_AVAILABLE else 'Not Available'}")
    print(f"  DuckDB: {'Available' if DUCKDB_AVAILABLE else 'Not Available'}")

    runner = BenchmarkRunner(warmup_runs=3, benchmark_runs=10)

    try:
        # Run all benchmarks
        benchmark_correlation(runner)
        benchmark_options(runner)
        benchmark_risk(runner)
        benchmark_duckdb(runner)
        benchmark_gil_free(runner)

        # Summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        # Group results by benchmark type
        benchmarks = {}
        for result in runner.results:
            key = (result.name, result.data_size)
            if key not in benchmarks:
                benchmarks[key] = {}
            benchmarks[key][result.implementation] = result

        print("\nSpeedup Summary:")
        print("-" * 80)

        speedups = []
        for (name, size), implementations in benchmarks.items():
            if 'C++ Bindings' in implementations or 'C++ Bindings (OpenMP)' in implementations:
                cpp_key = 'C++ Bindings (OpenMP)' if 'C++ Bindings (OpenMP)' in implementations else 'C++ Bindings'
                cpp_result = implementations[cpp_key]

                # Find baseline
                baseline = None
                for impl in ['Pure Python', 'NumPy', 'SciPy', 'Pandas', 'Python DuckDB']:
                    if impl in implementations:
                        baseline = implementations[impl]
                        break

                if baseline:
                    speedup = baseline.avg_time / cpp_result.avg_time
                    speedups.append((name, size, speedup, baseline.implementation))
                    print(f"  {name:30s} ({size:10s}): {speedup:6.2f}x vs {baseline.implementation}")

        # Statistics
        if speedups:
            speedup_values = [s[2] for s in speedups]
            print(f"\nOverall Statistics:")
            print(f"  Average speedup: {np.mean(speedup_values):.2f}x")
            print(f"  Median speedup:  {np.median(speedup_values):.2f}x")
            print(f"  Min speedup:     {np.min(speedup_values):.2f}x")
            print(f"  Max speedup:     {np.max(speedup_values):.2f}x")

        # Save results to JSON
        results_file = 'benchmarks/results.json'
        os.makedirs('benchmarks', exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'results': [
                    {
                        'name': r.name,
                        'implementation': r.implementation,
                        'data_size': r.data_size,
                        'avg_time_ms': r.avg_time * 1000,
                        'std_dev_ms': r.std_dev * 1000,
                        'min_time_ms': r.min_time * 1000,
                        'max_time_ms': r.max_time * 1000,
                    }
                    for r in runner.results
                ],
                'speedups': [
                    {
                        'benchmark': name,
                        'data_size': size,
                        'speedup': speedup,
                        'baseline': baseline
                    }
                    for name, size, speedup, baseline in speedups
                ]
            }, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
