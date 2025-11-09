#!/usr/bin/env python3
"""
BigBrotherAnalytics - Performance Benchmark Suite

Comprehensive benchmarks for Python bindings with the following targets:
- DuckDB: 5-10x speedup
- Correlation: 60-100x speedup (with OpenMP)
- Risk/Options: 50-100x speedup
- GIL-free execution verification

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import time
import numpy as np
import pandas as pd
import duckdb
from typing import Callable, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import json
import traceback
from math import log, sqrt, exp, isnan
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import threading

# ============================================================================
# BENCHMARK FRAMEWORK
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run"""
    name: str
    implementation: str
    data_size: str
    avg_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    iterations: int
    memory_mb: float = 0.0
    notes: str = ""


class BenchmarkRunner:
    """Manages benchmark execution and statistics"""

    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []

    def time_function(self, func: Callable, *args, **kwargs) -> float:
        """Time a single function call (returns time in seconds)"""
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
            avg_time_ms=np.mean(times) * 1000,
            std_dev_ms=np.std(times) * 1000,
            min_time_ms=np.min(times) * 1000,
            max_time_ms=np.max(times) * 1000,
            iterations=self.benchmark_runs
        )

        self.results.append(result)
        return result

    def print_result(self, result: BenchmarkResult):
        """Print a single benchmark result"""
        print(f"  {result.implementation:25s} | "
              f"{result.avg_time_ms:10.3f} ms | "
              f"σ={result.std_dev_ms:7.3f} ms")

    def compare_results(self, baseline: BenchmarkResult, optimized: BenchmarkResult) -> float:
        """Compare two benchmark results and return speedup ratio"""
        if optimized.avg_time_ms == 0:
            speedup = float('inf')
        else:
            speedup = baseline.avg_time_ms / optimized.avg_time_ms

        improvement_pct = ((baseline.avg_time_ms - optimized.avg_time_ms) / baseline.avg_time_ms) * 100

        print(f"\n  Speedup: {speedup:.2f}x | Improvement: {improvement_pct:.1f}% | Time saved: {(baseline.avg_time_ms - optimized.avg_time_ms):.2f} ms/call")

        return speedup


# ============================================================================
# PURE PYTHON IMPLEMENTATIONS FOR COMPARISON
# ============================================================================

class PythonImplementations:
    """Pure Python implementations for benchmarking"""

    @staticmethod
    def pearson_correlation(x: List[float], y: List[float]) -> float:
        """Pearson correlation using NumPy"""
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def spearman_correlation(x: List[float], y: List[float]) -> float:
        """Spearman correlation using SciPy"""
        return stats.spearmanr(x, y)[0]

    @staticmethod
    def black_scholes_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
        """Black-Scholes European call option pricing"""
        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        return S * stats.norm.cdf(d1) - K * exp(-r*T) * stats.norm.cdf(d2)

    @staticmethod
    def kelly_criterion(win_prob: float, win_loss_ratio: float) -> float:
        """Kelly Criterion calculation"""
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        return max(0.0, min(0.25, kelly))

    @staticmethod
    def monte_carlo_simulation(spot: float, sigma: float, drift: float, n_sims: int) -> Dict[str, float]:
        """Monte Carlo simulation for price paths"""
        T = 0.25
        dt = T / 100
        np.random.seed(42)

        results = []
        for _ in range(n_sims):
            price = spot
            for _ in range(100):
                dW = np.random.randn() * np.sqrt(dt)
                price *= np.exp((drift - 0.5*sigma**2)*dt + sigma*dW)
            results.append(price - spot)

        results = np.array(results)
        return {
            'expected_value': float(np.mean(results)),
            'std_deviation': float(np.std(results)),
            'probability_of_profit': float(np.sum(results > 0) / n_sims),
            'var_95': float(np.percentile(results, 5)),
            'var_99': float(np.percentile(results, 1))
        }

    @staticmethod
    def correlation_matrix(n_symbols: int, n_points: int) -> np.ndarray:
        """Calculate correlation matrix using Pandas"""
        np.random.seed(42)
        data = {f'SYM{i:03d}': np.random.randn(n_points) for i in range(n_symbols)}
        df = pd.DataFrame(data)
        return df.corr().values


# ============================================================================
# DUCKDB BENCHMARKS
# ============================================================================

def benchmark_duckdb(runner: BenchmarkRunner):
    """Benchmark DuckDB operations"""

    print("\n" + "="*90)
    print("DUCKDB QUERY BENCHMARKS")
    print("="*90)

    db_path = os.path.join(os.path.dirname(__file__), 'data', 'bigbrother.duckdb')

    if not os.path.exists(db_path):
        print(f"\nDatabase not found: {db_path}")
        return

    try:
        import bigbrother_duckdb as db_cpp
        cpp_available = True
    except ImportError:
        print("\nC++ DuckDB bindings not available")
        cpp_available = False
        return

    # Query 1: Simple COUNT
    print("\n1. COUNT QUERY (simple aggregation)")
    print("-" * 90)

    query1 = "SELECT COUNT(*) as count FROM sector_employment_raw"

    def py_count():
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(query1).fetchall()
        conn.close()
        return result

    def cpp_count():
        conn = db_cpp.connect(db_path)
        result = conn.execute(query1)
        return result.to_pandas_dict()

    try:
        result_py = runner.run_benchmark("Count Query", "Python DuckDB", "full table", py_count)
        runner.print_result(result_py)
    except Exception as e:
        print(f"  Python DuckDB failed: {e}")
        result_py = None

    try:
        result_cpp = runner.run_benchmark("Count Query", "C++ DuckDB Bindings", "full table", cpp_count)
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ DuckDB failed: {e}")
        result_cpp = None

    if result_py and result_cpp:
        speedup = runner.compare_results(result_py, result_cpp)

    # Query 2: JOIN and GROUP BY Aggregation
    print("\n2. JOIN + GROUP BY AGGREGATION")
    print("-" * 90)

    query2 = "SELECT s.sector_code, s.sector_name, AVG(e.employment_count) as avg_emp, COUNT(*) as cnt FROM sector_employment_raw e JOIN sectors s ON e.series_id LIKE s.sector_code::VARCHAR || '%' GROUP BY s.sector_code, s.sector_name"

    def py_groupby():
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(query2).fetchall()
        conn.close()
        return result

    def cpp_groupby():
        conn = db_cpp.connect(db_path)
        result = conn.execute(query2)
        return result.to_pandas_dict()

    try:
        result_py = runner.run_benchmark("Group By", "Python DuckDB", "full table", py_groupby)
        runner.print_result(result_py)
    except Exception as e:
        print(f"  Python DuckDB failed: {e}")
        result_py = None

    try:
        result_cpp = runner.run_benchmark("Group By", "C++ DuckDB Bindings", "full table", cpp_groupby)
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ DuckDB failed: {e}")
        result_cpp = None

    if result_py and result_cpp:
        speedup = runner.compare_results(result_py, result_cpp)

    # Query 3: Filter and Sort
    print("\n3. FILTER, SORT, AND LIMIT")
    print("-" * 90)

    query3 = "SELECT * FROM sector_employment_raw WHERE employment_count > 100000 ORDER BY report_date DESC LIMIT 1000"

    def py_filter():
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(query3).fetchall()
        conn.close()
        return result

    def cpp_filter():
        conn = db_cpp.connect(db_path)
        result = conn.execute(query3)
        return result.to_pandas_dict()

    try:
        result_py = runner.run_benchmark("Filter/Sort", "Python DuckDB", "filtered", py_filter)
        runner.print_result(result_py)
    except Exception as e:
        print(f"  Python DuckDB failed: {e}")
        result_py = None

    try:
        result_cpp = runner.run_benchmark("Filter/Sort", "C++ DuckDB Bindings", "filtered", cpp_filter)
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ DuckDB failed: {e}")
        result_cpp = None

    if result_py and result_cpp:
        speedup = runner.compare_results(result_py, result_cpp)

    # Query 4: Complex JOIN (if data allows)
    print("\n4. COMPLEX QUERY (multi-table operations)")
    print("-" * 90)

    query4 = """
    SELECT
        DATE_TRUNC('month', e.report_date) as month,
        s.sector_name,
        AVG(e.employment_count) as avg_employment,
        STDDEV(e.employment_count) as stddev_employment,
        COUNT(*) as observation_count
    FROM sector_employment_raw e
    JOIN sectors s ON e.series_id LIKE s.sector_code::VARCHAR || '%'
    GROUP BY DATE_TRUNC('month', e.report_date), s.sector_name
    ORDER BY month DESC, avg_employment DESC
    LIMIT 500
    """

    def py_complex():
        conn = duckdb.connect(db_path, read_only=True)
        result = conn.execute(query4).fetchall()
        conn.close()
        return result

    def cpp_complex():
        conn = db_cpp.connect(db_path)
        result = conn.execute(query4)
        return result.to_pandas_dict()

    try:
        result_py = runner.run_benchmark("Complex Query", "Python DuckDB", "processed", py_complex)
        runner.print_result(result_py)
    except Exception as e:
        print(f"  Python DuckDB failed: {e}")
        result_py = None

    try:
        result_cpp = runner.run_benchmark("Complex Query", "C++ DuckDB Bindings", "processed", cpp_complex)
        runner.print_result(result_cpp)
    except Exception as e:
        print(f"  C++ DuckDB failed: {e}")
        result_cpp = None

    if result_py and result_cpp:
        speedup = runner.compare_results(result_py, result_cpp)


# ============================================================================
# PURE PYTHON FINANCIAL CALCULATIONS
# ============================================================================

def benchmark_pure_python(runner: BenchmarkRunner):
    """Benchmark pure Python financial calculations"""

    print("\n" + "="*90)
    print("PURE PYTHON FINANCIAL CALCULATIONS (Baseline)")
    print("="*90)

    py_impl = PythonImplementations()

    # Correlation benchmarks
    print("\n1. CORRELATION CALCULATIONS")
    print("-" * 90)

    for size_name, n in [("small", 100), ("medium", 1_000), ("large", 10_000)]:
        print(f"\n  {size_name.upper()} ({n:,} points)")

        np.random.seed(42)
        x = np.random.randn(n).tolist()
        y = np.random.randn(n).tolist()

        def pearson_py():
            return py_impl.pearson_correlation(x, y)

        try:
            result = runner.run_benchmark(f"Pearson ({size_name})", "NumPy", size_name, pearson_py)
            runner.print_result(result)
        except Exception as e:
            print(f"    Failed: {e}")

    # Spearman correlation
    print(f"\n  SPEARMAN CORRELATION (10K points)")
    np.random.seed(42)
    x = np.random.randn(10_000).tolist()
    y = np.random.randn(10_000).tolist()

    def spearman_py():
        return py_impl.spearman_correlation(x, y)

    try:
        result = runner.run_benchmark("Spearman", "SciPy", "10K", spearman_py)
        runner.print_result(result)
    except Exception as e:
        print(f"    Failed: {e}")

    # Options pricing
    print("\n2. BLACK-SCHOLES OPTION PRICING")
    print("-" * 90)

    S, K, r, T, sigma = 100.0, 105.0, 0.041, 1.0, 0.25

    def bs_py():
        return py_impl.black_scholes_call(S, K, r, T, sigma)

    try:
        result = runner.run_benchmark("Black-Scholes Single", "Pure Python", "single", bs_py)
        runner.print_result(result)
    except Exception as e:
        print(f"    Failed: {e}")

    # Batch pricing
    n_options = 1000
    strikes = np.linspace(90, 110, n_options)

    def bs_batch_py():
        return [py_impl.black_scholes_call(S, k, r, T, sigma) for k in strikes]

    try:
        result = runner.run_benchmark("Black-Scholes Batch", "Pure Python", f"{n_options} options", bs_batch_py)
        runner.print_result(result)
    except Exception as e:
        print(f"    Failed: {e}")

    # Risk calculations
    print("\n3. RISK MANAGEMENT CALCULATIONS")
    print("-" * 90)

    win_prob, wl_ratio = 0.65, 2.0

    def kelly_py():
        return py_impl.kelly_criterion(win_prob, wl_ratio)

    try:
        result = runner.run_benchmark("Kelly Criterion", "Pure Python", "single", kelly_py)
        runner.print_result(result)
    except Exception as e:
        print(f"    Failed: {e}")

    # Monte Carlo
    for n_sims in [1_000, 10_000]:
        print(f"\n  Monte Carlo ({n_sims:,} simulations)")

        def mc_py():
            return py_impl.monte_carlo_simulation(100.0, 0.25, 0.05, n_sims)

        try:
            result = runner.run_benchmark("Monte Carlo", "Pure Python", f"{n_sims} sims", mc_py)
            runner.print_result(result)
        except Exception as e:
            print(f"    Failed: {e}")


# ============================================================================
# GIL-FREE MULTI-THREADING TEST
# ============================================================================

def benchmark_gil_free(runner: BenchmarkRunner):
    """Test GIL-free execution (pure Python + DuckDB)"""

    print("\n" + "="*90)
    print("GIL-FREE MULTI-THREADING TEST (DuckDB)")
    print("="*90)

    try:
        import bigbrother_duckdb as db_cpp
    except ImportError:
        print("\nDuckDB C++ bindings not available for GIL test")
        return

    db_path = os.path.join(os.path.dirname(__file__), 'data', 'bigbrother.duckdb')

    if not os.path.exists(db_path):
        print(f"\nDatabase not found: {db_path}")
        return

    # Prepare multiple queries
    queries = [
        "SELECT COUNT(*) FROM sector_employment_raw",
        "SELECT AVG(employment_count), STDDEV(employment_count), MIN(employment_count), MAX(employment_count) FROM sector_employment_raw",
        "SELECT * FROM sector_employment_raw WHERE employment_count > 100000 LIMIT 100",
    ]

    # Single-threaded
    print("\nSINGLE-THREADED (3 queries x 10 iterations)")
    print("-" * 90)

    def single_threaded():
        results = []
        for q in queries:
            conn = db_cpp.connect(db_path)
            result = conn.execute(q)
            results.append(result.to_pandas_dict())
        return results

    try:
        result_single = runner.run_benchmark(
            "3 DuckDB Queries",
            "Single Thread",
            "sequential",
            single_threaded
        )
        runner.print_result(result_single)
    except Exception as e:
        print(f"  Failed: {e}")
        result_single = None

    # Multi-threaded
    print("\nMULTI-THREADED (3 queries x 10 iterations, 3 threads - GIL-free)")
    print("-" * 90)

    def multi_threaded():
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(lambda q: db_cpp.connect(db_path).execute(q).to_pandas_dict(), q) for q in queries]
            results = [f.result() for f in futures]
        return results

    try:
        result_multi = runner.run_benchmark(
            "3 DuckDB Queries",
            "3 Threads (GIL-free)",
            "parallel",
            multi_threaded
        )
        runner.print_result(result_multi)
    except Exception as e:
        print(f"  Failed: {e}")
        result_multi = None

    if result_single and result_multi:
        speedup = runner.compare_results(result_single, result_multi)
        is_gil_free = speedup > 1.3
        print(f"\n  GIL-free verification: {'PASS ✓' if is_gil_free else 'FAIL ✗'}")
        print(f"  Expected >1.3x speedup with 3 threads, got {speedup:.2f}x")


# ============================================================================
# MAIN BENCHMARK SUITE
# ============================================================================

def main():
    """Run complete benchmark suite"""

    print("\n" + "="*90)
    print("BigBrotherAnalytics - Comprehensive Performance Benchmark Suite")
    print("="*90)

    # Environment info
    print(f"\nEnvironment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")
    print(f"  Pandas: {pd.__version__}")
    print(f"  DuckDB: {duckdb.__version__}")

    # Check bindings
    print(f"\nBinding Status:")
    bindings = {
        'DuckDB': 'bigbrother_duckdb',
        'Correlation': 'bigbrother_correlation',
        'Risk': 'bigbrother_risk',
        'Options': 'bigbrother_options',
    }

    binding_status = {}
    for name, module in bindings.items():
        try:
            __import__(module)
            binding_status[name] = 'Available'
            print(f"  {name}: Available ✓")
        except ImportError:
            binding_status[name] = 'Not Available'
            print(f"  {name}: Not Available (missing dependencies)")

    # Run benchmarks
    runner = BenchmarkRunner(warmup_runs=3, benchmark_runs=10)

    try:
        benchmark_pure_python(runner)
        benchmark_duckdb(runner)
        benchmark_gil_free(runner)

        # Summary and statistics
        print("\n" + "="*90)
        print("BENCHMARK SUMMARY")
        print("="*90)

        if runner.results:
            # Group results by test
            benchmarks = {}
            for result in runner.results:
                key = result.name
                if key not in benchmarks:
                    benchmarks[key] = {}
                benchmarks[key][result.implementation] = result

            # Calculate speedups
            print("\nSpeedup Analysis:")
            print("-" * 90)

            speedups = []
            for test_name, implementations in benchmarks.items():
                if len(implementations) > 1:
                    # Find baseline (pure Python/NumPy/SciPy)
                    baseline = None
                    cpp_impl = None

                    baseline_names = ['Pure Python', 'NumPy', 'SciPy', 'Python DuckDB']
                    cpp_names = ['C++ DuckDB Bindings', 'C++ Bindings', 'C++ Bindings (OpenMP)']

                    for name in baseline_names:
                        if name in implementations:
                            baseline = implementations[name]
                            break

                    for name in cpp_names:
                        if name in implementations:
                            cpp_impl = implementations[name]
                            break

                    if baseline and cpp_impl:
                        speedup = baseline.avg_time_ms / cpp_impl.avg_time_ms
                        speedups.append((test_name, speedup, baseline.implementation, cpp_impl.implementation))
                        print(f"  {test_name:30s}: {speedup:6.2f}x ({baseline.implementation} → {cpp_impl.implementation})")

            # Statistics
            if speedups:
                speedup_values = [s[1] for s in speedups]
                print(f"\nPerformance Statistics:")
                print(f"  Average speedup:  {np.mean(speedup_values):6.2f}x")
                print(f"  Median speedup:   {np.median(speedup_values):6.2f}x")
                print(f"  Min speedup:      {np.min(speedup_values):6.2f}x")
                print(f"  Max speedup:      {np.max(speedup_values):6.2f}x")
                print(f"  Std deviation:    {np.std(speedup_values):6.2f}x")

        # Save detailed results to JSON
        results_file = os.path.join(os.path.dirname(__file__), 'benchmarks', 'results.json')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'environment': {
                    'python_version': sys.version.split()[0],
                    'numpy_version': np.__version__,
                    'pandas_version': pd.__version__,
                    'duckdb_version': duckdb.__version__,
                },
                'bindings_status': binding_status,
                'results': [asdict(r) for r in runner.results]
            }, f, indent=2)

        print(f"\nDetailed results saved to: {results_file}")

        print("\n" + "="*90)
        print("BENCHMARK COMPLETE")
        print("="*90)

        return 0

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
