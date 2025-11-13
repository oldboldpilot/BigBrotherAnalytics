#!/usr/bin/env python3
"""
Monte Carlo SIMD Performance Benchmark

Comprehensive benchmark of SIMD-optimized Monte Carlo simulator:
- Stock simulations with varying simulation counts
- Parallel vs sequential execution
- Throughput and latency measurements
- Memory efficiency analysis

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-13
"""

import time
import sys
from statistics import mean, stdev
import bigbrother_risk as risk

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")

def print_subheader(text):
    """Print formatted subheader"""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-' * 70}{Colors.ENDC}")

def benchmark_monte_carlo_throughput():
    """Benchmark Monte Carlo throughput with varying simulation counts"""
    print_header("Monte Carlo SIMD Throughput Benchmark")

    # Test configuration
    entry = 100.0
    target = 110.0
    stop = 95.0
    volatility = 0.25

    test_configs = [
        (1_000, "1K simulations", 10),
        (5_000, "5K simulations", 10),
        (10_000, "10K simulations", 10),
        (50_000, "50K simulations", 5),
        (100_000, "100K simulations", 3),
        (250_000, "250K simulations", 3),
    ]

    results = []

    for num_sims, label, iterations in test_configs:
        print_subheader(f"{label} (avg of {iterations} runs)")

        times_ms = []

        for i in range(iterations):
            simulator = (risk.MonteCarloSimulator.create()
                        .with_simulations(num_sims)
                        .with_parallel(True))

            start = time.perf_counter()
            result = simulator.simulate_stock(entry, target, stop, volatility)
            elapsed_ms = (time.perf_counter() - start) * 1000

            times_ms.append(elapsed_ms)

        avg_time = mean(times_ms)
        std_time = stdev(times_ms) if len(times_ms) > 1 else 0
        throughput = num_sims / avg_time * 1000  # sims/sec

        print(f"  {Colors.OKGREEN}Average time:{Colors.ENDC} {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"  {Colors.OKGREEN}Throughput:{Colors.ENDC}   {throughput:,.0f} simulations/second")
        print(f"  {Colors.OKGREEN}Min/Max:{Colors.ENDC}      {min(times_ms):.2f}ms / {max(times_ms):.2f}ms")

        if result:
            print(f"  {Colors.OKCYAN}Stats:{Colors.ENDC}        Mean=${result.mean_pnl:.2f}, "
                  f"VaR=${result.var_95:.2f}, Win={result.win_probability*100:.1f}%")

        results.append({
            'simulations': num_sims,
            'avg_time_ms': avg_time,
            'throughput': throughput,
            'label': label
        })

    # Print summary table
    print_subheader("Performance Summary")
    print(f"\n  {'Simulations':<15} {'Avg Time':<15} {'Throughput':<25} {'Speedup':<10}")
    print(f"  {'-' * 65}")

    baseline_throughput = results[0]['throughput']
    for r in results:
        speedup = r['throughput'] / baseline_throughput
        print(f"  {r['simulations']:<15,} {r['avg_time_ms']:<14.2f}ms "
              f"{r['throughput']:<24,.0f} {speedup:<9.2f}x")

def benchmark_parallel_vs_sequential():
    """Compare parallel vs sequential Monte Carlo performance"""
    print_header("Parallel vs Sequential Performance")

    entry = 100.0
    target = 110.0
    stop = 95.0
    volatility = 0.25

    test_configs = [
        (10_000, "10K simulations", 5),
        (50_000, "50K simulations", 3),
        (100_000, "100K simulations", 3),
    ]

    for num_sims, label, iterations in test_configs:
        print_subheader(label)

        # Parallel execution
        parallel_times = []
        for _ in range(iterations):
            simulator = (risk.MonteCarloSimulator.create()
                        .with_simulations(num_sims)
                        .with_parallel(True))

            start = time.perf_counter()
            result = simulator.simulate_stock(entry, target, stop, volatility)
            elapsed_ms = (time.perf_counter() - start) * 1000
            parallel_times.append(elapsed_ms)

        # Sequential execution
        sequential_times = []
        for _ in range(iterations):
            simulator = (risk.MonteCarloSimulator.create()
                        .with_simulations(num_sims)
                        .with_parallel(False))

            start = time.perf_counter()
            result = simulator.simulate_stock(entry, target, stop, volatility)
            elapsed_ms = (time.perf_counter() - start) * 1000
            sequential_times.append(elapsed_ms)

        parallel_avg = mean(parallel_times)
        sequential_avg = mean(sequential_times)
        speedup = sequential_avg / parallel_avg

        print(f"  {Colors.OKGREEN}Parallel:{Colors.ENDC}     {parallel_avg:.2f}ms "
              f"({num_sims/parallel_avg*1000:,.0f} sims/sec)")
        print(f"  {Colors.WARNING}Sequential:{Colors.ENDC}   {sequential_avg:.2f}ms "
              f"({num_sims/sequential_avg*1000:,.0f} sims/sec)")
        print(f"  {Colors.BOLD}Speedup:{Colors.ENDC}      {speedup:.2f}x")

def benchmark_statistics_accuracy():
    """Verify SIMD statistics accuracy vs expected values"""
    print_header("SIMD Statistics Accuracy Verification")

    # Known distribution test
    print_subheader("Normal Distribution (μ=0, σ=1)")

    num_sims = 100_000
    simulator = (risk.MonteCarloSimulator.create()
                .with_simulations(num_sims)
                .with_parallel(True)
                .with_seed(42))  # Fixed seed for reproducibility

    # Simulate with zero drift (should have mean near 0)
    result = simulator.simulate_stock(100.0, 110.0, 90.0, 0.01)  # Low volatility

    print(f"  {Colors.OKGREEN}Simulations:{Colors.ENDC}  {result.num_simulations:,}")
    print(f"  {Colors.OKGREEN}Mean P&L:{Colors.ENDC}     ${result.mean_pnl:.4f} (expected: ~$0.00)")
    print(f"  {Colors.OKGREEN}Std Dev:{Colors.ENDC}      ${result.std_pnl:.4f}")
    print(f"  {Colors.OKGREEN}VaR (95%):{Colors.ENDC}    ${result.var_95:.4f}")
    print(f"  {Colors.OKGREEN}CVaR (95%):{Colors.ENDC}   ${result.cvar_95:.4f}")
    print(f"  {Colors.OKGREEN}Win Rate:{Colors.ENDC}     {result.win_probability*100:.2f}%")
    print(f"  {Colors.OKGREEN}Min/Max:{Colors.ENDC}      ${result.min_pnl:.2f} / ${result.max_pnl:.2f}")

def print_system_info():
    """Print system information"""
    print_header("System Information")

    import platform
    import os

    print(f"  {Colors.OKBLUE}Platform:{Colors.ENDC}     {platform.system()} {platform.release()}")
    print(f"  {Colors.OKBLUE}Python:{Colors.ENDC}       {platform.python_version()}")
    print(f"  {Colors.OKBLUE}CPU Cores:{Colors.ENDC}    {os.cpu_count()}")

    # Try to detect CPU features
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'avx512f' in cpuinfo.lower():
                simd = "AVX-512 (8 doubles/iteration)"
            elif 'avx2' in cpuinfo.lower():
                simd = "AVX2 (4 doubles/iteration)"
            else:
                simd = "Scalar (no SIMD)"
            print(f"  {Colors.OKBLUE}SIMD Support:{Colors.ENDC} {simd}")
    except:
        pass

def main():
    """Run all benchmarks"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     Monte Carlo SIMD Performance Benchmark Suite                  ║")
    print("║     BigBrotherAnalytics Risk Management                           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")

    print_system_info()

    try:
        benchmark_monte_carlo_throughput()
        benchmark_parallel_vs_sequential()
        benchmark_statistics_accuracy()

        print_header("Benchmark Complete")
        print(f"\n{Colors.OKGREEN}✓ All benchmarks completed successfully{Colors.ENDC}\n")

        print(f"{Colors.BOLD}SIMD Optimizations Summary:{Colors.ENDC}")
        print(f"  • AVX-512: 8 doubles processed simultaneously")
        print(f"  • AVX2: 4 doubles processed simultaneously")
        print(f"  • FMA instructions for multiply-add operations")
        print(f"  • Horizontal reduction for efficient summation")
        print(f"  • OpenMP parallel execution across CPU cores")
        print()

    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
