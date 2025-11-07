# Trading Correlation Analysis Tool - Architecture Design Document

**Version:** 1.0.0
**Date:** November 6, 2025
**Status:** Draft - Design Phase
**Author:** Olumuyiwa Oluwasanmi
**Related:** [Product Requirements Document](../PRD.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Component Architecture](#3-component-architecture)
4. [Correlation Algorithms](#4-correlation-algorithms)
5. [Database Schema Design](#5-database-schema-design)
6. [API Specifications](#6-api-specifications)
7. [Technology Stack](#7-technology-stack)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Implementation Guidelines](#9-implementation-guidelines)
10. [Performance Considerations](#10-performance-considerations)

---

## 1. Overview

### 1.1 Purpose

The Trading Correlation Analysis Tool is a high-performance time-series analysis system that:
- Discovers statistical relationships between securities across multiple timeframes
- Computes time-lagged correlations to identify leading and lagging indicators
- Detects correlation regime changes and breakpoints
- Provides correlation data for trading strategy development
- Leverages C++23/MPI/OpenMP for massive parallelization across 64+ cores

### 1.2 Key Capabilities

```mermaid
mindmap
  root((Correlation Analysis Tool))
    Data Sources
      Historical Market Data
      Price Time Series
      Volume Data
      Volatility Metrics
    Correlation Types
      Contemporaneous
      Time-Lagged
      Rolling Windows
      Dynamic Conditional
    Timeframes
      Intra-Day 1m-1h
      Inter-Day Daily
      Intra-Month Weekly
      Intra-Quarter Monthly
    Analysis Methods
      Pearson Correlation
      Spearman Rank
      Cross-Correlation
      Granger Causality
    Pattern Detection
      Regime Changes
      Breakpoints
      Clustering
      Anomalies
    Output
      Correlation Matrices
      Lead-Lag Relationships
      Network Graphs
      Trading Signals
```

### 1.3 Design Principles

- **Ultra-High Performance:** Parallel correlation computation across 64+ cores with C++23/MPI
- **Massive Scale:** Handle 10,000+ securities with 10+ years of data
- **Real-Time Updates:** Incremental correlation updates as new data arrives
- **GPU Acceleration:** CUDA for matrix operations and large-scale computations
- **Zero Setup:** DuckDB for instant analytics on Parquet files
- **Single Machine:** Tier 1 deployment on one powerful server (64-128 cores)

---

## 2. System Architecture

### 2.1 High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        DS1[Historical Market Data<br/>10+ Years]
        DS2[Real-Time Market Feed<br/>Polygon.io/Finnhub]
        DS3[Parquet Archives<br/>DuckDB-Queryable]
    end

    subgraph "Data Loading Layer (C++23)"
        DL1[Parallel Data Loader<br/>MPI + OpenMP]
        DL2[Data Validator<br/>C++23 std::expected]
        DL3[Time-Series Aligner<br/>Multi-threaded]
    end

    subgraph "Correlation Engine (C++23 + CUDA)"
        CE1[Parallel Correlation<br/>MPI Distribution]
        CE2[Rolling Correlations<br/>OpenMP Threads]
        CE3[Cross-Correlation<br/>CUDA/cuBLAS]
        CE4[Statistical Tests<br/>Intel MKL]
    end

    subgraph "Pattern Detection (C++23)"
        PD1[Regime Detection<br/>Change Point Analysis]
        PD2[Breakpoint Detection<br/>Structural Breaks]
        PD3[Clustering<br/>Hierarchical/K-means]
        PD4[Anomaly Detection<br/>Outlier Analysis]
    end

    subgraph "Storage Layer"
        ST1[(PostgreSQL<br/>TimescaleDB<br/>Correlation Results)]
        ST2[(DuckDB<br/>Analytics<br/>Parquet Queries)]
        ST3[(Redis<br/>Cache<br/>Real-time Data)]
    end

    subgraph "API & Visualization"
        API1[REST API<br/>FastAPI]
        API2[WebSocket<br/>Real-time Updates]
        VIZ1[Correlation Matrices<br/>Heatmaps]
        VIZ2[Network Graphs<br/>D3.js/Cytoscape]
    end

    DS1 & DS2 & DS3 --> DL1 --> DL2 --> DL3
    DL3 --> CE1 & CE2 & CE3 & CE4
    CE1 & CE2 & CE3 & CE4 --> PD1 & PD2 & PD3 & PD4
    PD1 & PD2 & PD3 & PD4 --> ST1 & ST2 & ST3
    ST1 & ST2 & ST3 --> API1 & API2
    API1 & API2 --> VIZ1 & VIZ2

    style CE1 fill:#faa,stroke:#333,stroke-width:3px
    style ST2 fill:#bbf,stroke:#333,stroke-width:2px
    style API1 fill:#bfb,stroke:#333,stroke-width:2px
```

### 2.2 Tier 1 Single-Machine Architecture

```mermaid
graph TB
    subgraph "Single Server (64-128 Cores)"
        subgraph "C++ Core Engine"
            CPP1[MPI Master Process<br/>Rank 0]
            CPP2[MPI Worker 1<br/>Cores 0-15]
            CPP3[MPI Worker 2<br/>Cores 16-31]
            CPP4[MPI Worker 3<br/>Cores 32-47]
            CPP5[MPI Worker 4<br/>Cores 48-63]
        end

        subgraph "GPU Acceleration"
            GPU1[CUDA Kernel<br/>Matrix Operations]
            GPU2[cuBLAS<br/>Linear Algebra]
            GPU3[cuSOLVER<br/>Statistical Tests]
        end

        subgraph "Data Layer"
            DB1[(PostgreSQL<br/>Port 5432)]
            DB2[DuckDB Files<br/>/data/correlations]
            DB3[(Redis<br/>Port 6379)]
            FS1[Parquet Files<br/>/data/archive]
        end

        subgraph "Python API Layer"
            PY1[FastAPI Server<br/>Port 8001]
            PY2[WebSocket Server<br/>Port 8002]
        end
    end

    CPP1 --> CPP2 & CPP3 & CPP4 & CPP5
    CPP2 & CPP3 & CPP4 & CPP5 --> GPU1 & GPU2
    GPU1 & GPU2 & GPU3 --> DB1 & DB3
    FS1 -.DuckDB Query.-> DB2
    DB2 --> DB1
    DB1 & DB3 --> PY1 & PY2

    Client[External Clients] --> PY1 & PY2

    style CPP1 fill:#faa,stroke:#333,stroke-width:3px
    style GPU1 fill:#f9f,stroke:#333,stroke-width:2px
    style DB2 fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 3. Component Architecture

### 3.1 Data Loading Pipeline

```mermaid
flowchart TB
    Start([Load Historical Data]) --> Source{Data Source?}

    Source -->|Parquet Files| Duck[DuckDB Direct Query]
    Source -->|PostgreSQL| PG[TimescaleDB Query]
    Source -->|CSV/JSON| File[File Reader]

    Duck --> Parse1[Parse to C++ Structures]
    PG --> Parse2[Parse to C++ Structures]
    File --> Parse3[Parse to C++ Structures]

    Parse1 & Parse2 & Parse3 --> Align[Time-Series Alignment<br/>Handle Missing Data]

    Align --> Validate[Validate Data Quality<br/>std::expected]

    Validate --> Split[MPI Data Distribution<br/>Split by Rank]

    Split --> Local1[Local Rank 0<br/>Securities 0-2499]
    Split --> Local2[Local Rank 1<br/>Securities 2500-4999]
    Split --> Local3[Local Rank 2<br/>Securities 5000-7499]
    Split --> Local4[Local Rank 3<br/>Securities 7500-9999]

    Local1 & Local2 & Local3 & Local4 --> Ready[Ready for<br/>Correlation Calc]

    Ready --> End([Begin Correlation])

    style Start fill:#afa,stroke:#333,stroke-width:2px
    style Split fill:#faa,stroke:#333,stroke-width:2px
    style End fill:#afa,stroke:#333,stroke-width:2px
```

### 3.2 Parallel Correlation Engine

```mermaid
graph TB
    subgraph "MPI Distribution (4 Ranks on 64-Core Server)"
        MPI1[Rank 0<br/>Cores 0-15<br/>Pairs 0-25%]
        MPI2[Rank 1<br/>Cores 16-31<br/>Pairs 25-50%]
        MPI3[Rank 2<br/>Cores 32-47<br/>Pairs 50-75%]
        MPI4[Rank 3<br/>Cores 48-63<br/>Pairs 75-100%]
    end

    subgraph "OpenMP Within Each Rank"
        OMP1[Thread 0-15<br/>Rank 0 Work]
        OMP2[Thread 16-31<br/>Rank 1 Work]
        OMP3[Thread 32-47<br/>Rank 2 Work]
        OMP4[Thread 48-63<br/>Rank 3 Work]
    end

    subgraph "CUDA Acceleration (Optional)"
        CUDA1[GPU Kernel<br/>Matrix Multiplication]
        CUDA2[cuBLAS<br/>Dot Products]
    end

    subgraph "Results Collection"
        Gather[MPI_Gather<br/>Collect Results]
        Filter[Filter Significant<br/>|r| > 0.5]
        Store[(Store in PostgreSQL)]
    end

    MPI1 --> OMP1 --> CUDA1
    MPI2 --> OMP2 --> CUDA1
    MPI3 --> OMP3 --> CUDA2
    MPI4 --> OMP4 --> CUDA2

    CUDA1 & CUDA2 --> Gather
    Gather --> Filter --> Store

    style MPI1 fill:#faa,stroke:#333,stroke-width:2px
    style CUDA1 fill:#f9f,stroke:#333,stroke-width:2px
    style Store fill:#bbf,stroke:#333,stroke-width:2px
```

### 3.3 Multi-Timeframe Analysis Pipeline

```mermaid
graph LR
    subgraph "Input Data"
        I1[Minute-Level Data<br/>2+ Years]
    end

    subgraph "Timeframe Aggregation (DuckDB)"
        T1[1-Minute<br/>Intra-Day]
        T2[5-Minute<br/>Intra-Day]
        T3[1-Hour<br/>Intra-Day]
        T4[Daily<br/>Inter-Day]
        T5[Weekly<br/>Intra-Month]
        T6[Monthly<br/>Intra-Quarter]
    end

    subgraph "Parallel Correlation (C++23)"
        C1[Correlations<br/>1-Minute]
        C2[Correlations<br/>5-Minute]
        C3[Correlations<br/>1-Hour]
        C4[Correlations<br/>Daily]
        C5[Correlations<br/>Weekly]
        C6[Correlations<br/>Monthly]
    end

    subgraph "Pattern Analysis"
        P1[Intra-Day Patterns]
        P2[Day-of-Week Effects]
        P3[Monthly Patterns]
        P4[Quarterly Patterns]
    end

    I1 --> T1 & T2 & T3 & T4 & T5 & T6
    T1 --> C1 --> P1
    T2 --> C2 --> P1
    T3 --> C3 --> P1
    T4 --> C4 --> P2
    T5 --> C5 --> P3
    T6 --> C6 --> P4

    P1 & P2 & P3 & P4 --> Results[(Correlation<br/>Database)]

    style C1 fill:#faa,stroke:#333,stroke-width:2px
    style Results fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 4. Correlation Algorithms

### 4.1 Parallel Correlation Computation (C++23 + MPI + OpenMP)

```cpp
// File: src/correlation/parallel_correlation.hpp
// High-performance correlation calculation with MPI and OpenMP

#include <expected>
#include <span>
#include <flat_map>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <mkl.h>  // Intel MKL for BLAS operations

namespace correlation {

// Time-series data structure
struct TimeSeries {
    std::string symbol;
    std::vector<double> returns;
    std::vector<double> volumes;
    std::vector<int64_t> timestamps;
};

// Correlation result
struct CorrelationPair {
    std::string symbol_a;
    std::string symbol_b;
    double correlation;
    double p_value;
    size_t sample_size;
    int optimal_lag;  // For time-lagged correlations
};

// High-performance correlation engine
class ParallelCorrelationEngine {
private:
    int mpi_rank_;
    int mpi_size_;
    int num_threads_;

public:
    ParallelCorrelationEngine() {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        num_threads_ = omp_get_max_threads();
    }

    // Compute all pairwise correlations in parallel
    auto compute_all_correlations(
        std::span<const TimeSeries> time_series,
        int window_size = -1  // -1 for full period
    ) -> std::expected<std::vector<CorrelationPair>, Error> {

        const size_t n = time_series.size();
        const size_t total_pairs = (n * (n - 1)) / 2;

        // Distribute pairs across MPI ranks
        const size_t pairs_per_rank = total_pairs / mpi_size_;
        const size_t start_idx = mpi_rank_ * pairs_per_rank;
        const size_t end_idx = (mpi_rank_ == mpi_size_ - 1)
            ? total_pairs
            : start_idx + pairs_per_rank;

        std::vector<CorrelationPair> local_results;
        local_results.reserve(pairs_per_rank / 2);  // Estimate half are significant

        // OpenMP parallel loop within MPI rank
        #pragma omp parallel
        {
            std::vector<CorrelationPair> thread_results;

            #pragma omp for schedule(dynamic, 10) nowait
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                auto [i, j] = pair_index_to_indices(idx, n);

                // Compute correlation using Intel MKL
                auto result = compute_correlation_mkl(
                    time_series[i].returns,
                    time_series[j].returns,
                    window_size
                );

                if (result.has_value() && std::abs(result.value().correlation) > 0.5) {
                    thread_results.push_back(result.value());
                }
            }

            // Merge thread results
            #pragma omp critical
            {
                local_results.insert(
                    local_results.end(),
                    thread_results.begin(),
                    thread_results.end()
                );
            }
        }

        // MPI_Gather all results to rank 0
        std::vector<CorrelationPair> all_results;
        if (mpi_rank_ == 0) {
            all_results = gather_results_from_all_ranks(local_results);
        } else {
            send_results_to_rank0(local_results);
        }

        return all_results;
    }

    // Compute rolling correlations efficiently
    auto compute_rolling_correlations(
        const TimeSeries& ts_a,
        const TimeSeries& ts_b,
        int window_size,
        int step_size = 1
    ) -> std::expected<std::vector<double>, Error> {

        if (ts_a.returns.size() != ts_b.returns.size()) {
            return std::unexpected(Error::SizeMismatch);
        }

        const size_t n = ts_a.returns.size();
        if (n < window_size) {
            return std::unexpected(Error::InsufficientData);
        }

        const size_t num_windows = (n - window_size) / step_size + 1;
        std::vector<double> rolling_corr(num_windows);

        // Parallel rolling correlation with OpenMP
        #pragma omp parallel for schedule(static)
        for (size_t w = 0; w < num_windows; ++w) {
            const size_t start_idx = w * step_size;
            const size_t end_idx = start_idx + window_size;

            // Use Intel MKL for correlation calculation
            auto corr = pearson_correlation_mkl(
                std::span(ts_a.returns).subspan(start_idx, window_size),
                std::span(ts_b.returns).subspan(start_idx, window_size)
            );

            rolling_corr[w] = corr;
        }

        return rolling_corr;
    }

private:
    // Use Intel MKL for optimized correlation
    static double pearson_correlation_mkl(
        std::span<const double> x,
        std::span<const double> y
    ) {
        const int n = x.size();

        // Use MKL's correlation function
        double corr;
        VSLSSTaskPtr task;

        // Create VSL task
        vsldSSNewTask(&task, &n, &n, &n, x.data(), nullptr, nullptr);
        vsldSSEditTask(task, VSL_SS_ED_COR, y.data());

        // Compute correlation
        vsldSSCompute(task, VSL_SS_COR, VSL_SS_METHOD_FAST);

        // Extract result
        vsldSSEditTask(task, VSL_SS_ED_COR, &corr);

        // Cleanup
        vslSSDeleteTask(&task);

        return corr;
    }

    // Convert 1D pair index to 2D matrix indices
    static auto pair_index_to_indices(size_t idx, size_t n)
        -> std::pair<size_t, size_t> {
        // Convert triangular matrix index to (i, j) coordinates
        size_t i = n - 2 - static_cast<size_t>(
            std::sqrt(static_cast<double>(4 * n * (n - 1) - 8 * idx - 7)) / 2.0 - 0.5
        );
        size_t j = idx + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
        return {i, j};
    }

    // Gather results from all MPI ranks
    auto gather_results_from_all_ranks(
        const std::vector<CorrelationPair>& local_results
    ) -> std::vector<CorrelationPair> {

        // Get result counts from all ranks
        int local_count = local_results.size();
        std::vector<int> counts(mpi_size_);
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate displacements
        std::vector<int> displacements(mpi_size_);
        int total_count = 0;
        for (int i = 0; i < mpi_size_; ++i) {
            displacements[i] = total_count;
            total_count += counts[i];
        }

        // Gather all results
        std::vector<CorrelationPair> all_results(total_count);
        MPI_Gatherv(
            local_results.data(), local_count, MPI_CORRELATION_PAIR,
            all_results.data(), counts.data(), displacements.data(),
            MPI_CORRELATION_PAIR, 0, MPI_COMM_WORLD
        );

        return all_results;
    }
};

} // namespace correlation
```

### 4.2 Time-Lagged Correlation (Cross-Correlation)

```cpp
// File: src/correlation/cross_correlation.hpp
// Cross-correlation for lead-lag relationship detection

#include <complex>
#include <fftw3.h>  // For FFT-based cross-correlation
#include <mkl_vsl.h>

namespace correlation {

class CrossCorrelationAnalyzer {
public:
    // Compute cross-correlation function for all lags
    auto compute_ccf(
        std::span<const double> x,
        std::span<const double> y,
        int max_lag = 60
    ) -> std::expected<std::vector<double>, Error> {

        const size_t n = std::min(x.size(), y.size());

        // Use FFT for efficient cross-correlation (O(n log n))
        std::vector<std::complex<double>> x_fft(n);
        std::vector<std::complex<double>> y_fft(n);
        std::vector<std::complex<double>> ccf_fft(n);

        // FFT plans
        fftw_plan plan_x = fftw_plan_dft_r2c_1d(
            n, const_cast<double*>(x.data()),
            reinterpret_cast<fftw_complex*>(x_fft.data()),
            FFTW_ESTIMATE
        );

        fftw_plan plan_y = fftw_plan_dft_r2c_1d(
            n, const_cast<double*>(y.data()),
            reinterpret_cast<fftw_complex*>(y_fft.data()),
            FFTW_ESTIMATE
        );

        // Execute FFTs
        fftw_execute(plan_x);
        fftw_execute(plan_y);

        // Compute element-wise product (X * conj(Y))
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; ++i) {
            ccf_fft[i] = x_fft[i] * std::conj(y_fft[i]);
        }

        // Inverse FFT
        std::vector<double> ccf(n);
        fftw_plan plan_ifft = fftw_plan_dft_c2r_1d(
            n, reinterpret_cast<fftw_complex*>(ccf_fft.data()),
            ccf.data(), FFTW_ESTIMATE
        );
        fftw_execute(plan_ifft);

        // Normalize
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; ++i) {
            ccf[i] /= n;
        }

        // Extract relevant lags
        std::vector<double> result(2 * max_lag + 1);
        for (int lag = -max_lag; lag <= max_lag; ++lag) {
            size_t idx = (lag + n) % n;
            result[lag + max_lag] = ccf[idx];
        }

        // Cleanup
        fftw_destroy_plan(plan_x);
        fftw_destroy_plan(plan_y);
        fftw_destroy_plan(plan_ifft);

        return result;
    }

    // Find optimal lag (max absolute correlation)
    auto find_optimal_lag(
        const std::vector<double>& ccf,
        int max_lag
    ) -> std::pair<int, double> {

        int optimal_lag = 0;
        double max_corr = 0.0;

        for (int lag = -max_lag; lag <= max_lag; ++lag) {
            double corr = std::abs(ccf[lag + max_lag]);
            if (corr > max_corr) {
                max_corr = corr;
                optimal_lag = lag;
            }
        }

        return {optimal_lag, ccf[optimal_lag + max_lag]};
    }

    // Granger causality test
    auto granger_causality_test(
        std::span<const double> x,
        std::span<const double> y,
        int max_lag
    ) -> std::expected<GrangerResult, Error> {

        // Implement Granger causality using MKL regression
        // Returns F-statistic and p-value

        // ... implementation using MKL for linear regression
    }
};

} // namespace correlation
```

### 4.3 CUDA-Accelerated Matrix Operations

```cpp
// File: src/correlation/cuda_correlation.cu
// GPU-accelerated correlation for large-scale calculations

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace correlation {

// CUDA kernel for parallel correlation calculation
__global__ void compute_correlations_kernel(
    const double* returns_matrix,  // [n_securities x n_days]
    double* correlation_matrix,    // [n_securities x n_securities]
    int n_securities,
    int n_days
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_securities && j < n_securities && i < j) {
        // Compute correlation between securities i and j
        double sum_x = 0, sum_y = 0, sum_xy = 0;
        double sum_x2 = 0, sum_y2 = 0;

        for (int t = 0; t < n_days; ++t) {
            double x = returns_matrix[i * n_days + t];
            double y = returns_matrix[j * n_days + t];

            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }

        double n = n_days;
        double numerator = n * sum_xy - sum_x * sum_y;
        double denominator = sqrt(
            (n * sum_x2 - sum_x * sum_x) *
            (n * sum_y2 - sum_y * sum_y)
        );

        double corr = (denominator > 1e-10) ? numerator / denominator : 0.0;

        // Store in upper triangle
        correlation_matrix[i * n_securities + j] = corr;
        correlation_matrix[j * n_securities + i] = corr;  // Symmetric
    }
}

// GPU correlation engine
class CUDACorrelationEngine {
private:
    cublasHandle_t cublas_handle_;
    int device_id_;

public:
    CUDACorrelationEngine(int device_id = 0) : device_id_(device_id) {
        cudaSetDevice(device_id_);
        cublasCreate(&cublas_handle_);
    }

    ~CUDACorrelationEngine() {
        cublasDestroy(cublas_handle_);
    }

    // Compute correlation matrix on GPU
    auto compute_correlation_matrix(
        const std::vector<TimeSeries>& time_series
    ) -> std::expected<std::vector<std::vector<double>>, Error> {

        const int n_securities = time_series.size();
        const int n_days = time_series[0].returns.size();

        // Allocate device memory
        double* d_returns;
        double* d_correlations;
        cudaMalloc(&d_returns, n_securities * n_days * sizeof(double));
        cudaMalloc(&d_correlations, n_securities * n_securities * sizeof(double));

        // Copy data to device
        std::vector<double> returns_flat;
        returns_flat.reserve(n_securities * n_days);
        for (const auto& ts : time_series) {
            returns_flat.insert(returns_flat.end(),
                              ts.returns.begin(),
                              ts.returns.end());
        }
        cudaMemcpy(d_returns, returns_flat.data(),
                   n_securities * n_days * sizeof(double),
                   cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 block(16, 16);
        dim3 grid(
            (n_securities + block.x - 1) / block.x,
            (n_securities + block.y - 1) / block.y
        );

        compute_correlations_kernel<<<grid, block>>>(
            d_returns, d_correlations, n_securities, n_days
        );

        // Copy results back
        std::vector<double> corr_flat(n_securities * n_securities);
        cudaMemcpy(corr_flat.data(), d_correlations,
                   n_securities * n_securities * sizeof(double),
                   cudaMemcpyDeviceToHost);

        // Convert to 2D matrix
        std::vector<std::vector<double>> correlation_matrix(
            n_securities, std::vector<double>(n_securities)
        );
        for (int i = 0; i < n_securities; ++i) {
            for (int j = 0; j < n_securities; ++j) {
                correlation_matrix[i][j] = corr_flat[i * n_securities + j];
            }
        }

        // Cleanup
        cudaFree(d_returns);
        cudaFree(d_correlations);

        return correlation_matrix;
    }

    // Use cuBLAS for correlation via matrix multiplication
    auto compute_correlation_cublas(
        const double* returns_matrix,  // [n_securities x n_days]
        int n_securities,
        int n_days
    ) -> std::vector<double> {

        // Correlation = (X^T * X) / (n - 1), after standardization
        // Use cuBLAS DGEMM for matrix multiplication

        double* d_input;
        double* d_output;
        cudaMalloc(&d_input, n_securities * n_days * sizeof(double));
        cudaMalloc(&d_output, n_securities * n_securities * sizeof(double));

        cudaMemcpy(d_input, returns_matrix,
                   n_securities * n_days * sizeof(double),
                   cudaMemcpyHostToDevice);

        const double alpha = 1.0 / (n_days - 1);
        const double beta = 0.0;

        // C = alpha * A * A^T + beta * C
        cublasDgemm(
            cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            n_securities, n_securities, n_days,
            &alpha,
            d_input, n_securities,
            d_input, n_securities,
            &beta,
            d_output, n_securities
        );

        // Copy result back
        std::vector<double> correlation_matrix(n_securities * n_securities);
        cudaMemcpy(correlation_matrix.data(), d_output,
                   n_securities * n_securities * sizeof(double),
                   cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        return correlation_matrix;
    }
};

} // namespace correlation
```

### 4.4 Fluent Composable API Design (C++23 + UPC++)

**CRITICAL:** Design a fluent, composable API for correlation analysis that enables easy chaining of operations and distributed computing with UPC++.

```cpp
// File: src/correlation/fluent_api.hpp
// Fluent API for composable correlation analysis

#include <expected>
#include <ranges>
#include <upcxx/upcxx.hpp>

namespace correlation::api {

// Fluent correlation builder
class CorrelationAnalysis {
private:
    std::vector<std::string> symbols_;
    std::string timeframe_ = "1day";
    int window_size_ = -1;  // -1 means full period
    double min_correlation_ = 0.5;
    bool compute_pvalues_ = true;
    bool compute_lags_ = false;
    int max_lag_ = 60;
    upcxx::global_ptr<double> distributed_data_;  // UPC++ shared data

public:
    // Fluent API methods (chainable)
    auto for_symbols(std::vector<std::string> symbols) -> CorrelationAnalysis& {
        symbols_ = std::move(symbols);
        return *this;
    }

    auto with_timeframe(std::string timeframe) -> CorrelationAnalysis& {
        timeframe_ = std::move(timeframe);
        return *this;
    }

    auto using_window(int size) -> CorrelationAnalysis& {
        window_size_ = size;
        return *this;
    }

    auto above_threshold(double min_corr) -> CorrelationAnalysis& {
        min_correlation_ = min_corr;
        return *this;
    }

    auto with_significance_test() -> CorrelationAnalysis& {
        compute_pvalues_ = true;
        return *this;
    }

    auto with_lead_lag_analysis(int max_lag = 60) -> CorrelationAnalysis& {
        compute_lags_ = true;
        max_lag_ = max_lag;
        return *this;
    }

    // Execute computation using UPC++ for distributed data
    auto compute() -> std::expected<CorrelationResults, Error> {
        // Load data into UPC++ distributed shared memory
        distribute_data_upcxx();

        // Compute correlations using distributed data
        auto results = compute_distributed_correlations();

        return results;
    }

    // Execute with MPI parallelization
    auto compute_parallel() -> std::expected<CorrelationResults, Error> {
        return compute_pairwise_correlations_mpi(/*...*/);
    }

    // Execute with GPU acceleration
    auto compute_gpu() -> std::expected<CorrelationResults, Error> {
        CUDACorrelationMatrix gpu_engine;
        return gpu_engine.compute(/*...*/);
    }

private:
    void distribute_data_upcxx() {
        // Allocate distributed shared memory with UPC++
        const size_t local_size = symbols_.size() / upcxx::rank_n();

        distributed_data_ = upcxx::new_array<double>(
            local_size * /* days */ 252 * 10  // 10 years daily
        );

        // Load local portion of data
        // Each rank loads its subset of securities
        // ... implementation
    }

    auto compute_distributed_correlations()
        -> std::expected<CorrelationResults, Error> {

        // Each rank computes correlations for its securities
        // Using one-sided RDMA to access remote data

        std::vector<CorrelationPair> local_results;

        for (size_t local_i = 0; local_i < local_symbols_.size(); ++local_i) {
            // Access remote securities using UPC++ one-sided get
            for (int remote_rank = 0; remote_rank < upcxx::rank_n(); ++remote_rank) {
                for (size_t remote_j = 0; remote_j < remote_symbols_per_rank; ++remote_j) {
                    // Get remote data without synchronization
                    auto remote_data = upcxx::rget(
                        distributed_data_ + remote_j * n_days,
                        n_days
                    ).wait();

                    // Compute correlation locally
                    double corr = compute_pearson(local_data[local_i], remote_data);

                    if (std::abs(corr) >= min_correlation_) {
                        local_results.push_back(/*...*/);
                    }
                }
            }
        }

        // Barrier to ensure all ranks complete
        upcxx::barrier();

        return local_results;
    }
};

// Example usage of fluent API
auto example_fluent_usage() {
    // Create composable correlation analysis
    auto results = CorrelationAnalysis()
        .for_symbols({"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"})
        .with_timeframe("1day")
        .using_window(30)  // 30-day rolling
        .above_threshold(0.7)
        .with_significance_test()
        .with_lead_lag_analysis(max_lag = 30)
        .compute_parallel();  // Execute with MPI

    if (results.has_value()) {
        for (const auto& pair : results.value().pairs) {
            std::cout << pair.symbol_a << " <-> " << pair.symbol_b
                      << ": " << pair.correlation << std::endl;
        }
    }
}

// Composable analysis pipeline
class CorrelationPipeline {
public:
    static auto builder() -> CorrelationPipeline& {
        static CorrelationPipeline instance;
        return instance;
    }

    // Chain multiple analyses
    auto pearson() -> CorrelationPipeline& {
        analyses_.push_back(AnalysisType::Pearson);
        return *this;
    }

    auto spearman() -> CorrelationPipeline& {
        analyses_.push_back(AnalysisType::Spearman);
        return *this;
    }

    auto cross_correlation() -> CorrelationPipeline& {
        analyses_.push_back(AnalysisType::CrossCorrelation);
        return *this;
    }

    auto granger_causality() -> CorrelationPipeline& {
        analyses_.push_back(AnalysisType::GrangerCausality);
        return *this;
    }

    auto execute() -> std::expected<PipelineResults, Error> {
        // Execute all analyses in pipeline
        PipelineResults results;

        for (const auto& analysis : analyses_) {
            switch (analysis) {
                case AnalysisType::Pearson:
                    results.pearson = compute_pearson_all();
                    break;
                case AnalysisType::CrossCorrelation:
                    results.cross_corr = compute_ccf_all();
                    break;
                // ... other analyses
            }
        }

        return results;
    }

private:
    std::vector<AnalysisType> analyses_;
};

// Usage
auto pipeline_results = CorrelationPipeline::builder()
    .pearson()
    .cross_correlation()
    .granger_causality()
    .execute();

} // namespace correlation::api
```

### 4.5 DuckDB Analytical Queries

```sql
-- File: sql/correlation_analytics.sql
-- DuckDB queries for correlation analysis on Parquet files

-- Calculate pairwise correlations across all stocks
-- (Use DuckDB for exploration, C++ for production)

CREATE OR REPLACE VIEW pairwise_correlations AS
SELECT
    a.symbol as symbol_a,
    b.symbol as symbol_b,
    corr(a.returns, b.returns) as correlation,
    count(*) as sample_size,
    -- T-statistic for correlation significance
    corr(a.returns, b.returns) * sqrt(count(*) - 2) /
        sqrt(1 - pow(corr(a.returns, b.returns), 2)) as t_stat
FROM read_parquet('data/returns/*.parquet') a
JOIN read_parquet('data/returns/*.parquet') b
    ON a.date = b.date
WHERE a.symbol < b.symbol  -- Avoid duplicates (upper triangle)
  AND a.date >= '2020-01-01'
GROUP BY a.symbol, b.symbol
HAVING abs(corr(a.returns, b.returns)) > 0.5
   AND count(*) > 100;  -- Minimum sample size

-- Rolling correlations with window functions
CREATE OR REPLACE VIEW rolling_correlations_30d AS
SELECT
    a.symbol as symbol_a,
    b.symbol as symbol_b,
    a.date,
    corr(a.returns, b.returns) OVER (
        PARTITION BY a.symbol, b.symbol
        ORDER BY a.date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as rolling_corr_30d
FROM read_parquet('data/returns/*.parquet') a
JOIN read_parquet('data/returns/*.parquet') b
    ON a.date = b.date
WHERE a.symbol IN ('AAPL', 'MSFT', 'GOOGL')
  AND b.symbol IN ('AAPL', 'MSFT', 'GOOGL')
  AND a.symbol < b.symbol
ORDER BY a.symbol, b.symbol, a.date;

-- Detect correlation regime changes
CREATE OR REPLACE VIEW correlation_regime_changes AS
WITH rolling_stats AS (
    SELECT
        symbol_a,
        symbol_b,
        date,
        rolling_corr_30d,
        -- Moving average of correlation
        avg(rolling_corr_30d) OVER (
            PARTITION BY symbol_a, symbol_b
            ORDER BY date
            ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
        ) as ma_correlation,
        -- Standard deviation
        stddev(rolling_corr_30d) OVER (
            PARTITION BY symbol_a, symbol_b
            ORDER BY date
            ROWS BETWEEN 60 PRECEDING AND CURRENT ROW
        ) as std_correlation
    FROM rolling_correlations_30d
)
SELECT
    symbol_a,
    symbol_b,
    date,
    rolling_corr_30d,
    ma_correlation,
    -- Detect regime change (> 2 std deviations)
    CASE
        WHEN abs(rolling_corr_30d - ma_correlation) > 2 * std_correlation
        THEN 'REGIME_CHANGE'
        ELSE 'STABLE'
    END as regime_status
FROM rolling_stats
WHERE std_correlation > 0.01  -- Avoid division by zero
ORDER BY date DESC;

-- Cross-correlation at different lags (simplified version)
-- For production, use C++ with FFT
CREATE OR REPLACE MACRO cross_corr(x, y, lag) AS
    corr(
        x[lag + 1:],
        y[1:length(y) - lag]
    );
```

---

## 5. Database Schema Design

### 5.1 PostgreSQL Schema - Correlation Results

```mermaid
erDiagram
    SECURITIES ||--o{ CORRELATION_PAIRS : "has_correlations"
    SECURITIES ||--o{ TIME_SERIES_DATA : "has_data"
    CORRELATION_PAIRS ||--o{ ROLLING_CORRELATIONS : "has_rolling"
    CORRELATION_PAIRS ||--o{ CROSS_CORRELATIONS : "has_lagged"
    CORRELATION_PAIRS ||--o{ REGIME_CHANGES : "has_regimes"
    SECURITIES ||--o{ CLUSTERS : "belongs_to"

    SECURITIES {
        uuid id PK
        text symbol
        text name
        text sector
        text industry
        numeric market_cap
        timestamp created_at
        timestamp updated_at
    }

    TIME_SERIES_DATA {
        timestamp time PK
        text symbol PK
        numeric open
        numeric high
        numeric low
        numeric close
        bigint volume
        numeric returns
        numeric log_returns
    }

    CORRELATION_PAIRS {
        uuid id PK
        text symbol_a
        text symbol_b
        text timeframe
        numeric correlation
        numeric p_value
        int sample_size
        timestamp calculated_at
        text calculation_method
    }

    ROLLING_CORRELATIONS {
        uuid id PK
        uuid correlation_pair_id FK
        timestamp window_end
        int window_size
        numeric correlation
        timestamp calculated_at
    }

    CROSS_CORRELATIONS {
        uuid id PK
        uuid correlation_pair_id FK
        int lag
        numeric correlation
        timestamp calculated_at
    }

    REGIME_CHANGES {
        uuid id PK
        uuid correlation_pair_id FK
        timestamp change_date
        numeric correlation_before
        numeric correlation_after
        numeric confidence
        text regime_type
    }

    CLUSTERS {
        uuid id PK
        text cluster_name
        int cluster_number
        jsonb securities
        numeric avg_intra_correlation
        timestamp created_at
    }
```

### 5.2 TimescaleDB Tables

```sql
-- Time-series data (hypertable)
CREATE TABLE time_series_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    returns NUMERIC,  -- Simple returns
    log_returns NUMERIC,  -- Log returns
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('time_series_data', 'time');

-- Create indexes
CREATE INDEX idx_time_series_symbol ON time_series_data(symbol, time DESC);
CREATE INDEX idx_time_series_returns ON time_series_data(returns) WHERE returns IS NOT NULL;

-- Continuous aggregate for daily returns
CREATE MATERIALIZED VIEW daily_returns
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    (last(close, time) - first(open, time)) / first(open, time) AS returns
FROM time_series_data
GROUP BY day, symbol;

-- Correlation pairs (regular table, frequently updated)
CREATE TABLE correlation_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol_a TEXT NOT NULL,
    symbol_b TEXT NOT NULL,
    timeframe TEXT NOT NULL,  -- '1min', '1hour', '1day', '1week', '1month'
    correlation NUMERIC NOT NULL,
    p_value NUMERIC,
    sample_size INTEGER,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    calculation_method TEXT,  -- 'pearson', 'spearman', 'kendall'
    UNIQUE(symbol_a, symbol_b, timeframe, calculated_at)
);

CREATE INDEX idx_corr_pairs_symbols ON correlation_pairs(symbol_a, symbol_b, timeframe);
CREATE INDEX idx_corr_pairs_strength ON correlation_pairs(abs(correlation) DESC);
CREATE INDEX idx_corr_pairs_recent ON correlation_pairs(calculated_at DESC);

-- Rolling correlations (hypertable for time-series)
CREATE TABLE rolling_correlations (
    time TIMESTAMPTZ NOT NULL,
    symbol_a TEXT NOT NULL,
    symbol_b TEXT NOT NULL,
    window_size INTEGER NOT NULL,
    correlation NUMERIC NOT NULL,
    PRIMARY KEY (time, symbol_a, symbol_b, window_size)
);

SELECT create_hypertable('rolling_correlations', 'time');

-- Cross-correlations (lag analysis)
CREATE TABLE cross_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol_a TEXT NOT NULL,
    symbol_b TEXT NOT NULL,
    lag INTEGER NOT NULL,  -- Positive: A leads B, Negative: B leads A
    correlation NUMERIC NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(symbol_a, symbol_b, lag, calculated_at)
);

CREATE INDEX idx_cross_corr_symbols ON cross_correlations(symbol_a, symbol_b, lag);
CREATE INDEX idx_cross_corr_strength ON cross_correlations(abs(correlation) DESC);

-- Lead-lag relationships (derived from cross-correlations)
CREATE TABLE lead_lag_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    leader_symbol TEXT NOT NULL,
    lagger_symbol TEXT NOT NULL,
    optimal_lag INTEGER NOT NULL,  -- In minutes/days depending on timeframe
    max_correlation NUMERIC NOT NULL,
    confidence NUMERIC,  -- Based on p-value and stability
    timeframe TEXT NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lead_lag_leader ON lead_lag_relationships(leader_symbol, max_correlation DESC);
CREATE INDEX idx_lead_lag_lagger ON lead_lag_relationships(lagger_symbol);
```

### 5.3 DuckDB Analytical Schemas

```sql
-- DuckDB views for analytical queries on Parquet archives

-- Load historical returns
CREATE VIEW historical_returns AS
SELECT * FROM read_parquet('data/returns/*.parquet');

-- Compute correlations on-the-fly
CREATE MACRO compute_correlation(symbol_a, symbol_b, start_date, end_date) AS (
    SELECT corr(a.returns, b.returns) as correlation
    FROM historical_returns a
    JOIN historical_returns b ON a.date = b.date
    WHERE a.symbol = symbol_a
      AND b.symbol = symbol_b
      AND a.date BETWEEN start_date AND end_date
);

-- Find highly correlated pairs
CREATE VIEW high_correlations AS
SELECT
    a.symbol as symbol_a,
    b.symbol as symbol_b,
    corr(a.returns, b.returns) as correlation,
    count(*) as sample_size
FROM read_parquet('data/returns/*.parquet') a
JOIN read_parquet('data/returns/*.parquet') b
    ON a.date = b.date
WHERE a.symbol < b.symbol
  AND a.date >= current_date - INTERVAL 2 YEAR
GROUP BY a.symbol, b.symbol
HAVING abs(corr(a.returns, b.returns)) > 0.7
ORDER BY abs(corr(a.returns, b.returns)) DESC;

-- Sector-wise correlations
CREATE VIEW sector_correlations AS
SELECT
    s1.sector as sector_a,
    s2.sector as sector_b,
    avg(corr(ts1.returns, ts2.returns)) as avg_correlation,
    count(DISTINCT ts1.symbol || '-' || ts2.symbol) as pair_count
FROM read_parquet('data/returns/*.parquet') ts1
JOIN securities s1 ON ts1.symbol = s1.symbol
JOIN read_parquet('data/returns/*.parquet') ts2 ON ts1.date = ts2.date
JOIN securities s2 ON ts2.symbol = s2.symbol
WHERE s1.sector < s2.sector
GROUP BY s1.sector, s2.sector
ORDER BY avg_correlation DESC;
```

---

## 6. API Specifications

### 6.1 REST API Endpoints

```yaml
openapi: 3.0.0
info:
  title: Trading Correlation Analysis API
  version: 1.0.0
  description: API for accessing correlation analysis results and lead-lag relationships

servers:
  - url: http://localhost:8001/api/v1
    description: Local development

paths:
  /correlations:
    get:
      summary: Get correlation matrix or pairs
      parameters:
        - name: symbols
          in: query
          schema:
            type: array
            items:
              type: string
          description: List of symbols (if empty, returns all)
        - name: timeframe
          in: query
          schema:
            type: string
            enum: [1min, 5min, 1hour, 1day, 1week, 1month]
            default: 1day
        - name: min_correlation
          in: query
          schema:
            type: number
            minimum: 0
            maximum: 1
            default: 0.5
        - name: format
          in: query
          schema:
            type: string
            enum: [pairs, matrix, network]
            default: pairs
      responses:
        '200':
          description: Correlation results
          content:
            application/json:
              schema:
                oneOf:
                  - $ref: '#/components/schemas/CorrelationPairs'
                  - $ref: '#/components/schemas/CorrelationMatrix'
                  - $ref: '#/components/schemas/CorrelationNetwork'

  /correlations/rolling:
    get:
      summary: Get rolling correlations over time
      parameters:
        - name: symbol_a
          in: query
          required: true
          schema:
            type: string
        - name: symbol_b
          in: query
          required: true
          schema:
            type: string
        - name: window_size
          in: query
          schema:
            type: integer
            default: 30
        - name: from_date
          in: query
          schema:
            type: string
            format: date
      responses:
        '200':
          description: Rolling correlation time series
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RollingCorrelation'

  /correlations/lead-lag:
    get:
      summary: Get lead-lag relationships
      parameters:
        - name: symbol
          in: query
          schema:
            type: string
          description: Get relationships where this symbol is involved
        - name: role
          in: query
          schema:
            type: string
            enum: [leader, lagger, any]
            default: any
        - name: min_correlation
          in: query
          schema:
            type: number
            default: 0.6
        - name: timeframe
          in: query
          schema:
            type: string
            enum: [1min, 1hour, 1day]
      responses:
        '200':
          description: Lead-lag relationships
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LeadLagRelationships'

  /correlations/clusters:
    get:
      summary: Get correlation-based security clusters
      parameters:
        - name: method
          in: query
          schema:
            type: string
            enum: [hierarchical, kmeans, spectral]
            default: hierarchical
        - name: num_clusters
          in: query
          schema:
            type: integer
            minimum: 2
            maximum: 50
        - name: timeframe
          in: query
          schema:
            type: string
            default: 1day
      responses:
        '200':
          description: Security clusters
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Clusters'

  /correlations/regimes:
    get:
      summary: Get correlation regime changes
      parameters:
        - name: symbol_a
          in: query
          required: true
          schema:
            type: string
        - name: symbol_b
          in: query
          required: true
          schema:
            type: string
        - name: from_date
          in: query
          schema:
            type: string
            format: date
      responses:
        '200':
          description: Regime change events
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RegimeChanges'

  /correlations/compute:
    post:
      summary: Trigger correlation computation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                symbols:
                  type: array
                  items:
                    type: string
                timeframes:
                  type: array
                  items:
                    type: string
                window_sizes:
                  type: array
                  items:
                    type: integer
                priority:
                  type: string
                  enum: [low, normal, high]
      responses:
        '202':
          description: Computation job accepted
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                    format: uuid
                  status:
                    type: string
                  estimated_completion:
                    type: string
                    format: date-time

components:
  schemas:
    CorrelationPairs:
      type: object
      properties:
        pairs:
          type: array
          items:
            type: object
            properties:
              symbol_a:
                type: string
              symbol_b:
                type: string
              correlation:
                type: number
              p_value:
                type: number
              sample_size:
                type: integer
              timeframe:
                type: string
        total:
          type: integer
        timeframe:
          type: string

    CorrelationMatrix:
      type: object
      properties:
        symbols:
          type: array
          items:
            type: string
        matrix:
          type: array
          items:
            type: array
            items:
              type: number
        timeframe:
          type: string

    CorrelationNetwork:
      type: object
      properties:
        nodes:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
              symbol:
                type: string
              sector:
                type: string
        edges:
          type: array
          items:
            type: object
            properties:
              source:
                type: string
              target:
                type: string
              weight:
                type: number

    RollingCorrelation:
      type: object
      properties:
        symbol_a:
          type: string
        symbol_b:
          type: string
        window_size:
          type: integer
        data:
          type: array
          items:
            type: object
            properties:
              date:
                type: string
                format: date
              correlation:
                type: number

    LeadLagRelationships:
      type: object
      properties:
        relationships:
          type: array
          items:
            type: object
            properties:
              leader:
                type: string
              lagger:
                type: string
              optimal_lag:
                type: integer
              max_correlation:
                type: number
              confidence:
                type: number
              timeframe:
                type: string

    Clusters:
      type: object
      properties:
        method:
          type: string
        num_clusters:
          type: integer
        clusters:
          type: array
          items:
            type: object
            properties:
              cluster_id:
                type: integer
              securities:
                type: array
                items:
                  type: string
              avg_correlation:
                type: number
              representative:
                type: string

    RegimeChanges:
      type: object
      properties:
        symbol_a:
          type: string
        symbol_b:
          type: string
        changes:
          type: array
          items:
            type: object
            properties:
              date:
                type: string
                format: date
              correlation_before:
                type: number
              correlation_after:
                type: number
              confidence:
                type: number
              regime_type:
                type: string
```

---

## 7. Technology Stack

### 7.1 Performance-Critical Architecture

```mermaid
graph TB
    subgraph "C++23 Correlation Engine"
        CPP1[Data Loader<br/>MPI Parallel Read]
        CPP2[Pearson Correlation<br/>Intel MKL BLAS]
        CPP3[Rolling Correlation<br/>OpenMP Threads]
        CPP4[Cross-Correlation<br/>FFTW + MKL]
        CPP5[Statistical Tests<br/>MKL VSL]
    end

    subgraph "CUDA Acceleration"
        GPU1[Matrix Correlation<br/>cuBLAS DGEMM]
        GPU2[FFT Cross-Correlation<br/>cuFFT]
        GPU3[Parallel Reduction<br/>CUDA Kernels]
    end

    subgraph "Python Analytics"
        PY1[DuckDB Queries<br/>Parquet Files]
        PY2[Pattern Detection<br/>scikit-learn]
        PY3[Clustering<br/>scipy/cuML]
        PY4[Visualization<br/>Plotly/Seaborn]
    end

    subgraph "Storage"
        ST1[(PostgreSQL<br/>TimescaleDB)]
        ST2[DuckDB<br/>Analytics]
        ST3[(Redis<br/>Cache)]
        ST4[Parquet<br/>Archive]
    end

    CPP1 --> CPP2 & CPP3 & CPP4
    CPP2 & CPP3 & CPP4 --> CPP5
    CPP2 --> GPU1
    CPP4 --> GPU2
    CPP5 --> GPU3
    GPU1 & GPU2 & GPU3 --> ST1 & ST3
    ST4 --> PY1
    PY1 --> PY2 & PY3
    PY2 & PY3 --> ST1
    ST1 & ST3 --> PY4

    style CPP2 fill:#faa,stroke:#333,stroke-width:3px
    style GPU1 fill:#f9f,stroke:#333,stroke-width:3px
    style ST2 fill:#bbf,stroke:#333,stroke-width:2px
```

### 7.2 Component Technology Mapping

| Component | Language | Library/Framework | Purpose | Performance Target |
|-----------|----------|-------------------|---------|-------------------|
| **Critical Path (C++23)** |||||
| Data Loading | C++23 | MPI I/O, std::mdspan | Parallel file reading | < 10s for 10 years |
| Pearson Correlation | C++23 | Intel MKL BLAS | Vectorized computation | < 60s for 5K pairs |
| Rolling Correlation | C++23 | OpenMP, MKL | Multi-threaded windows | < 5s updates |
| Cross-Correlation | C++23 | FFTW3, MKL VSL | FFT-based CCF | < 10s per pair |
| Statistical Tests | C++23 | MKL VSL | P-values, confidence | < 1ms per test |
| Matrix Operations | C++23 | Eigen, std::mdspan | Linear algebra | < 100μs/op |
| **GPU Acceleration** |||||
| Matrix Correlation | CUDA | cuBLAS | Massive matrix ops | < 5s for 10K×10K |
| FFT Cross-Corr | CUDA | cuFFT | GPU-accelerated FFT | < 1s per pair |
| Parallel Reduction | CUDA | Custom kernels | Sum, mean, variance | < 100μs |
| **Analytics (Python)** |||||
| DuckDB Queries | Python/SQL | DuckDB | Parquet analytics | Seconds |
| Clustering | Python | scikit-learn/cuML | Security grouping | Minutes |
| Pattern Detection | Python | scipy, ruptures | Regime changes | Seconds |
| Visualization | Python | Plotly, NetworkX | Charts, graphs | N/A |
| **Storage** |||||
| Time-Series DB | SQL | PostgreSQL+Timescale | Store results | < 10ms |
| Analytics DB | SQL | DuckDB | Query archives | < 1s |
| Cache | Key-Value | Redis | Hot correlations | < 1ms |

### 7.3 C++23 Implementation Examples

**Complete Correlation Engine with MPI + OpenMP + Intel MKL:**

```cpp
// File: src/correlation/correlation_engine.cpp

#include <expected>
#include <span>
#include <mdspan>
#include <vector>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <mkl.h>

namespace correlation {

// Multi-dimensional view using C++23 mdspan
using MatrixView = std::mdspan<double, std::dextents<size_t, 2>>;

class CorrelationEngine {
private:
    int mpi_rank_;
    int mpi_size_;
    int omp_threads_;

public:
    CorrelationEngine() {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        omp_threads_ = omp_get_max_threads();

        // Configure Intel MKL
        mkl_set_num_threads(omp_threads_);
        mkl_set_dynamic(0);  // Disable dynamic threading
    }

    // Compute full correlation matrix in parallel
    auto compute_correlation_matrix(
        const std::vector<std::vector<double>>& returns_data,
        const std::vector<std::string>& symbols
    ) -> std::expected<std::vector<std::vector<double>>, Error> {

        const size_t n_securities = returns_data.size();
        const size_t n_days = returns_data[0].size();

        // Flatten to row-major matrix for MKL
        std::vector<double> returns_matrix(n_securities * n_days);
        for (size_t i = 0; i < n_securities; ++i) {
            std::copy(returns_data[i].begin(), returns_data[i].end(),
                     returns_matrix.begin() + i * n_days);
        }

        // Standardize data (mean=0, std=1)
        standardize_matrix_parallel(returns_matrix.data(), n_securities, n_days);

        // Compute correlation matrix using MKL DGEMM
        // Correlation = (X^T * X) / (n - 1)
        std::vector<double> corr_matrix(n_securities * n_securities, 0.0);

        const double alpha = 1.0 / (n_days - 1);
        const double beta = 0.0;

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            n_securities, n_securities, n_days,
            alpha,
            returns_matrix.data(), n_days,
            returns_matrix.data(), n_days,
            beta,
            corr_matrix.data(), n_securities
        );

        // Convert to 2D vector
        std::vector<std::vector<double>> result(
            n_securities, std::vector<double>(n_securities)
        );

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n_securities; ++i) {
            for (size_t j = 0; j < n_securities; ++j) {
                result[i][j] = corr_matrix[i * n_securities + j];
            }
        }

        return result;
    }

    // Compute pairwise correlations with MPI distribution
    auto compute_pairwise_correlations_mpi(
        const std::vector<std::vector<double>>& returns_data,
        const std::vector<std::string>& symbols,
        double min_correlation = 0.5
    ) -> std::expected<std::vector<CorrelationPair>, Error> {

        const size_t n = symbols.size();
        const size_t total_pairs = (n * (n - 1)) / 2;

        // Distribute pairs across MPI ranks
        const size_t pairs_per_rank = total_pairs / mpi_size_;
        const size_t start_idx = mpi_rank_ * pairs_per_rank;
        const size_t end_idx = (mpi_rank_ == mpi_size_ - 1)
            ? total_pairs
            : start_idx + pairs_per_rank;

        std::vector<CorrelationPair> local_results;

        // OpenMP parallel loop within MPI rank
        #pragma omp parallel
        {
            std::vector<CorrelationPair> thread_local_results;

            #pragma omp for schedule(dynamic, 100) nowait
            for (size_t pair_idx = start_idx; pair_idx < end_idx; ++pair_idx) {
                auto [i, j] = pair_index_to_ij(pair_idx, n);

                // Compute Pearson correlation using MKL
                double corr = vdCorrPearson(
                    returns_data[i].data(),
                    returns_data[j].data(),
                    returns_data[i].size()
                );

                if (std::abs(corr) >= min_correlation) {
                    // Compute p-value
                    double p_value = compute_correlation_pvalue(
                        corr, returns_data[i].size()
                    );

                    thread_local_results.push_back({
                        .symbol_a = symbols[i],
                        .symbol_b = symbols[j],
                        .correlation = corr,
                        .p_value = p_value,
                        .sample_size = returns_data[i].size(),
                        .optimal_lag = 0
                    });
                }
            }

            // Merge thread results
            #pragma omp critical
            {
                local_results.insert(local_results.end(),
                                   thread_local_results.begin(),
                                   thread_local_results.end());
            }
        }

        // Gather results from all MPI ranks to rank 0
        std::vector<CorrelationPair> all_results;
        if (mpi_rank_ == 0) {
            all_results = mpi_gather_correlations(local_results);
        } else {
            mpi_send_correlations(local_results, 0);
        }

        return all_results;
    }

private:
    // Standardize matrix in parallel (mean=0, std=1)
    void standardize_matrix_parallel(
        double* matrix,
        size_t n_securities,
        size_t n_days
    ) {
        #pragma omp parallel for
        for (size_t i = 0; i < n_securities; ++i) {
            double* row = matrix + i * n_days;

            // Compute mean using MKL
            double mean = cblas_dasum(n_days, row, 1) / n_days;

            // Subtract mean
            for (size_t j = 0; j < n_days; ++j) {
                row[j] -= mean;
            }

            // Compute standard deviation using MKL
            double std_dev = cblas_dnrm2(n_days, row, 1) / std::sqrt(n_days - 1);

            // Divide by std dev
            if (std_dev > 1e-10) {
                cblas_dscal(n_days, 1.0 / std_dev, row, 1);
            }
        }
    }

    // Compute p-value for correlation coefficient
    static double compute_correlation_pvalue(double r, size_t n) {
        if (n < 3) return 1.0;

        // T-statistic: t = r * sqrt(n - 2) / sqrt(1 - r^2)
        double t_stat = r * std::sqrt(n - 2) / std::sqrt(1 - r * r);

        // Degrees of freedom
        int df = n - 2;

        // Use MKL VSL for t-distribution CDF
        double p_value;
        vdCdfStudentT(1, &t_stat, df, &p_value);

        // Two-tailed test
        return 2.0 * (1.0 - p_value);
    }

    // MKL-based Pearson correlation
    static double vdCorrPearson(
        const double* x,
        const double* y,
        size_t n
    ) {
        VSLSSTaskPtr task;
        double corr;

        // Create VSL task for correlation
        int n_int = static_cast<int>(n);
        int p = 2;  // 2 variables

        // Interleave x and y for VSL
        std::vector<double> data(2 * n);
        for (size_t i = 0; i < n; ++i) {
            data[2 * i] = x[i];
            data[2 * i + 1] = y[i];
        }

        vsldSSNewTask(&task, &p, &n_int, &p, data.data(), nullptr, nullptr);

        // Compute correlation matrix
        double cor_storage[4];  // 2x2 correlation matrix
        vsldSSEditCorParameterization(task, cor_storage, &p,
                                     VSL_SS_ED_COR_PARAMETERIZATION_FULL);

        vsldSSCompute(task, VSL_SS_COR, VSL_SS_METHOD_FAST);

        // Extract correlation (off-diagonal element)
        corr = cor_storage[1];  // [0,1] element

        vslSSDeleteTask(&task);

        return corr;
    }
};

} // namespace correlation
```

### 7.4 CUDA Correlation Kernel

```cpp
// File: src/correlation/cuda_kernels.cu

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>

namespace correlation::cuda {

// Parallel correlation matrix computation on GPU
class CUDACorrelationMatrix {
private:
    cublasHandle_t cublas_;
    int device_id_;

public:
    CUDACorrelationMatrix(int device_id = 0) : device_id_(device_id) {
        cudaSetDevice(device_id_);
        cublasCreate(&cublas_);
    }

    ~CUDACorrelationMatrix() {
        cublasDestroy(cublas_);
    }

    // Compute correlation matrix using cuBLAS
    // Input: returns_matrix [n_securities × n_days]
    // Output: correlation_matrix [n_securities × n_securities]
    auto compute(
        const double* h_returns,
        int n_securities,
        int n_days
    ) -> std::vector<double> {

        // Allocate device memory
        double *d_returns, *d_standardized, *d_corr;
        cudaMalloc(&d_returns, n_securities * n_days * sizeof(double));
        cudaMalloc(&d_standardized, n_securities * n_days * sizeof(double));
        cudaMalloc(&d_corr, n_securities * n_securities * sizeof(double));

        // Copy to device
        cudaMemcpy(d_returns, h_returns,
                   n_securities * n_days * sizeof(double),
                   cudaMemcpyHostToDevice);

        // Standardize data on GPU (mean=0, std=1)
        standardize_rows_gpu<<<n_securities, 256>>>(
            d_returns, d_standardized, n_securities, n_days
        );

        // Compute correlation matrix: C = (1/(n-1)) * X * X^T
        const double alpha = 1.0 / (n_days - 1);
        const double beta = 0.0;

        cublasDgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
            n_securities, n_securities, n_days,
            &alpha,
            d_standardized, n_securities,
            d_standardized, n_securities,
            &beta,
            d_corr, n_securities
        );

        // Copy result back
        std::vector<double> correlation_matrix(n_securities * n_securities);
        cudaMemcpy(correlation_matrix.data(), d_corr,
                   n_securities * n_securities * sizeof(double),
                   cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_returns);
        cudaFree(d_standardized);
        cudaFree(d_corr);

        return correlation_matrix;
    }
};

// GPU kernel for standardization
__global__ void standardize_rows_gpu(
    const double* input,
    double* output,
    int n_rows,
    int n_cols
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const double* row_data = input + row * n_cols;
    double* row_output = output + row * n_cols;

    // Compute mean using parallel reduction
    __shared__ double shared_sum[256];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    double local_sum = 0.0;
    for (int i = tid; i < n_cols; i += stride) {
        local_sum += row_data[i];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    double mean = shared_sum[0] / n_cols;

    // Compute variance
    double local_var = 0.0;
    for (int i = tid; i < n_cols; i += stride) {
        double diff = row_data[i] - mean;
        local_var += diff * diff;
    }

    shared_sum[tid] = local_var;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    double variance = shared_sum[0] / (n_cols - 1);
    double std_dev = sqrt(variance);

    // Standardize
    for (int i = tid; i < n_cols; i += stride) {
        row_output[i] = (row_data[i] - mean) / std_dev;
    }
}

} // namespace correlation::cuda
```

---

## 8. Deployment Architecture

### 8.1 Tier 1 Single-Machine Deployment

```mermaid
graph TB
    subgraph "64-Core Server (Single Machine)"
        subgraph "MPI Processes (4 Ranks)"
            R0["Rank 0 (Master)<br/>Cores 0-15<br/>+ Python API"]
            R1["Rank 1 (Worker)<br/>Cores 16-31"]
            R2["Rank 2 (Worker)<br/>Cores 32-47"]
            R3["Rank 3 (Worker)<br/>Cores 48-63"]
        end

        subgraph "GPU (Optional)"
            GPU["NVIDIA GPU<br/>CUDA Kernels<br/>24-48GB VRAM"]
        end

        subgraph "Storage (Local NVMe)"
            PG[("PostgreSQL<br/>TimescaleDB<br/>Correlation Results")]
            Duck["DuckDB<br/>Parquet Queries<br/>Historical Data"]
            Redis[("Redis<br/>Real-time Cache")]
            Files["Parquet Files<br/>10 Years Data<br/>~500GB"]
        end

        subgraph "API Server (Python)"
            API["FastAPI<br/>Port 8001<br/>REST/WebSocket"]
        end
    end

    R0 <--> R1 & R2 & R3
    R0 & R1 & R2 & R3 --> GPU
    GPU --> PG
    R0 --> API
    Files --> Duck
    Duck --> PG
    PG & Redis --> API

    Client["External<br/>Clients"] --> API

    style R0 fill:#faa,stroke:#333,stroke-width:3px
    style GPU fill:#f9f,stroke:#333,stroke-width:2px
    style PG fill:#bbf,stroke:#333,stroke-width:2px
```

### 8.2 Resource Allocation (64-Core Server)

```mermaid
pie title CPU Core Allocation (64 Cores Total)
    "MPI Rank 0 + API" : 16
    "MPI Rank 1" : 16
    "MPI Rank 2" : 16
    "MPI Rank 3" : 16
```

```mermaid
pie title Memory Allocation (256GB RAM)
    "PostgreSQL Buffer" : 64
    "C++ Correlation Engine" : 128
    "Redis Cache" : 16
    "Python API & OS" : 48
```

### 8.3 Deployment Configuration

**systemd Service Files:**

```ini
# File: /etc/systemd/system/correlation-engine.service
[Unit]
Description=Trading Correlation Analysis Engine
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=bigbrother
Group=bigbrother
WorkingDirectory=/opt/bigbrother
Environment="PATH=/home/linuxbrew/.linuxbrew/bin:/usr/local/cuda-12.3/bin:/usr/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/home/linuxbrew/.linuxbrew/lib"

# Run with MPI (4 ranks on 64 cores)
ExecStart=/home/linuxbrew/.linuxbrew/bin/mpirun \
    -np 4 \
    --bind-to socket \
    --map-by socket \
    /opt/bigbrother/build/correlation_engine \
    --config /opt/bigbrother/config/correlation.yaml

Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

```ini
# File: /etc/systemd/system/correlation-api.service
[Unit]
Description=Correlation Analysis API Server
After=network.target correlation-engine.service
Requires=correlation-engine.service

[Service]
Type=simple
User=bigbrother
Group=bigbrother
WorkingDirectory=/opt/bigbrother
Environment="PATH=/opt/bigbrother/.venv/bin:/usr/bin"

ExecStart=/opt/bigbrother/.venv/bin/uvicorn \
    app.correlation_api:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 4

Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

### 8.4 Docker Compose (Alternative to systemd)

```yaml
# File: docker-compose.correlation.yml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb-ha:pg16
    environment:
      POSTGRES_DB: correlations
      POSTGRES_USER: corruser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    shm_size: 16g  # Shared memory for PostgreSQL

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 16gb --maxmemory-policy allkeys-lru

  # C++ correlation engine (runs with MPI)
  correlation_engine:
    build:
      context: .
      dockerfile: Dockerfile.correlation
    command: >
      mpirun -np 4
      --bind-to socket
      --use-hwthread-cpus
      /app/build/correlation_engine
      --config /app/config/correlation.yaml
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://corruser:${DB_PASSWORD}@postgres:5432/correlations
      REDIS_URL: redis://redis:6379
    volumes:
      - ./data:/data
      - ./config:/app/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Python API server
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    command: uvicorn app.correlation_api:app --host 0.0.0.0 --port 8001 --workers 4
    ports:
      - "8001:8001"
    depends_on:
      - postgres
      - redis
      - correlation_engine
    environment:
      DATABASE_URL: postgresql://corruser:${DB_PASSWORD}@postgres:5432/correlations
      REDIS_URL: redis://redis:6379
    volumes:
      - ./app:/app

volumes:
  postgres_data:
  redis_data:
```

---

## 9. Implementation Guidelines

### 9.1 Project Structure

```
trading-correlation-tool/
├── README.md
├── CMakeLists.txt              # C++23 build configuration
├── requirements.txt            # Python dependencies
├── docker-compose.correlation.yml
│
├── src/
│   ├── cpp/                    # C++23 correlation engine
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp
│   │   ├── correlation/
│   │   │   ├── correlation_engine.hpp
│   │   │   ├── correlation_engine.cpp
│   │   │   ├── cross_correlation.hpp
│   │   │   ├── cross_correlation.cpp
│   │   │   ├── cuda_correlation.cu
│   │   │   ├── cuda_correlation.hpp
│   │   │   └── statistical_tests.cpp
│   │   ├── data/
│   │   │   ├── data_loader.hpp
│   │   │   ├── data_loader.cpp
│   │   │   └── time_series.hpp
│   │   ├── pattern/
│   │   │   ├── regime_detector.hpp
│   │   │   ├── regime_detector.cpp
│   │   │   ├── clustering.hpp
│   │   │   └── clustering.cpp
│   │   └── utils/
│   │       ├── mpi_utils.hpp
│   │       ├── mpi_utils.cpp
│   │       └── statistics.hpp
│   │
│   └── python/                 # Python API and analytics
│       ├── app/
│       │   ├── __init__.py
│       │   ├── correlation_api.py
│       │   ├── models.py
│       │   └── schemas.py
│       ├── analytics/
│       │   ├── __init__.py
│       │   ├── duckdb_queries.py
│       │   ├── pattern_detection.py
│       │   └── visualization.py
│       └── db/
│           ├── __init__.py
│           ├── postgres.py
│           └── duckdb_connection.py
│
├── sql/                        # SQL schemas and queries
│   ├── schema.sql
│   ├── views.sql
│   └── analytics.sql
│
├── data/                       # Data storage
│   ├── raw/                    # Raw market data
│   ├── parquet/                # Parquet archives
│   ├── duckdb/                 # DuckDB database files
│   └── cache/                  # Temporary cache
│
├── scripts/                    # Utility scripts
│   ├── setup_db.sh
│   ├── download_data.py
│   ├── run_correlation.sh
│   └── verify_setup.sh
│
├── tests/                      # Tests
│   ├── cpp/
│   │   ├── test_correlation.cpp
│   │   └── test_cross_correlation.cpp
│   └── python/
│       ├── test_api.py
│       └── test_analytics.py
│
├── config/                     # Configuration
│   ├── correlation.yaml
│   └── database.yaml
│
├── playbooks/                  # Ansible playbooks
│   ├── tier1-correlation-setup.yml
│   └── postgres_setup.yml
│
└── docs/
    ├── api.md
    ├── algorithms.md
    └── performance.md
```

### 9.2 CMake Configuration

```cmake
# File: CMakeLists.txt
cmake_minimum_required(VERSION 3.28)
project(TradingCorrelationTool VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(MKL REQUIRED)
find_package(FFTW3 REQUIRED)

# Compiler options for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(
        -march=native        # Optimize for current CPU
        -mtune=native
        -O3                  # Maximum optimization
        -flto                # Link-time optimization
        -ffast-math          # Fast math operations
        -funroll-loops       # Loop unrolling
        -fopenmp             # OpenMP support
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -march=native
        -O3
        -flto=thin
        -ffast-math
        -fopenmp
    )
endif()

# CUDA options
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # RTX 2000-4000 series, A100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Source files
set(CPP_SOURCES
    src/cpp/main.cpp
    src/cpp/correlation/correlation_engine.cpp
    src/cpp/correlation/cross_correlation.cpp
    src/cpp/correlation/statistical_tests.cpp
    src/cpp/data/data_loader.cpp
    src/cpp/pattern/regime_detector.cpp
    src/cpp/pattern/clustering.cpp
    src/cpp/utils/mpi_utils.cpp
)

set(CUDA_SOURCES
    src/cpp/correlation/cuda_correlation.cu
)

# Executable
add_executable(correlation_engine ${CPP_SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(correlation_engine
    PRIVATE
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        CUDA::cublas
        CUDA::cufft
        CUDA::cudart
        MKL::MKL
        FFTW3::fftw3
)

# Include directories
target_include_directories(correlation_engine
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/cpp
        ${MKL_INCLUDE_DIRS}
)

# Install
install(TARGETS correlation_engine DESTINATION bin)
```

### 9.3 Build and Run Instructions

```bash
# Build C++ correlation engine
cd /opt/bigbrother
mkdir -p build && cd build

cmake .. \
    -G Ninja \
    -DCMAKE_CXX_COMPILER=g++-15 \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_OPENMP=ON \
    -DENABLE_MPI=ON \
    -DENABLE_CUDA=ON

cmake --build . -j $(nproc)

# Run with MPI (4 ranks on 64 cores)
mpirun -np 4 \
    --bind-to socket \
    --map-by socket \
    --report-bindings \
    ./correlation_engine \
    --symbols data/symbols.txt \
    --data data/parquet/*.parquet \
    --output results/correlations.csv \
    --min-correlation 0.5 \
    --timeframe 1day

# Run with GPU acceleration
CUDA_VISIBLE_DEVICES=0 mpirun -np 4 \
    ./correlation_engine \
    --use-gpu \
    --symbols data/sp500.txt \
    --output results/correlations_gpu.csv

# Profile with perf
perf record -g mpirun -np 4 ./correlation_engine --symbols data/test.txt
perf report
```

---

## 10. Performance Considerations

### 10.1 Performance Benchmarks

```mermaid
graph LR
    subgraph "Dataset Size"
        D1[100 Securities<br/>1 Year Daily]
        D2[1,000 Securities<br/>2 Years Daily]
        D3[5,000 Securities<br/>5 Years Daily]
        D4[10,000 Securities<br/>10 Years Daily]
    end

    subgraph "Computation Time (Single-Threaded)"
        T1[< 1 second]
        T2[~10 seconds]
        T3[~5 minutes]
        T4[~45 minutes]
    end

    subgraph "With MPI 4 Ranks + OpenMP 64 Cores"
        M1[< 0.1 second]
        M2[~0.5 seconds]
        M3[~8 seconds]
        M4[~60 seconds]
    end

    subgraph "With CUDA GPU"
        G1[< 0.05 second]
        G2[< 0.2 seconds]
        G3[~2 seconds]
        G4[~12 seconds]
    end

    D1 --> T1 --> M1 --> G1
    D2 --> T2 --> M2 --> G2
    D3 --> T3 --> M3 --> G3
    D4 --> T4 --> M4 --> G4

    style G4 fill:#afa,stroke:#333,stroke-width:2px
```

### 10.2 Optimization Strategies

**Memory Optimization:**
```cpp
// Use memory-mapped files for large datasets
#include <sys/mman.h>
#include <fcntl.h>

class MemoryMappedData {
private:
    void* mapped_addr_;
    size_t file_size_;
    int fd_;

public:
    auto map_file(const std::string& filename)
        -> std::expected<std::span<const double>, Error> {

        fd_ = open(filename.c_str(), O_RDONLY);
        if (fd_ == -1) {
            return std::unexpected(Error::FileOpenFailed);
        }

        // Get file size
        struct stat sb;
        fstat(fd_, &sb);
        file_size_ = sb.st_size;

        // Memory map the file
        mapped_addr_ = mmap(
            nullptr, file_size_,
            PROT_READ, MAP_PRIVATE | MAP_POPULATE,
            fd_, 0
        );

        if (mapped_addr_ == MAP_FAILED) {
            close(fd_);
            return std::unexpected(Error::MmapFailed);
        }

        // Advise kernel about access pattern
        madvise(mapped_addr_, file_size_, MADV_SEQUENTIAL | MADV_WILLNEED);

        return std::span<const double>(
            static_cast<const double*>(mapped_addr_),
            file_size_ / sizeof(double)
        );
    }

    ~MemoryMappedData() {
        if (mapped_addr_ != nullptr) {
            munmap(mapped_addr_, file_size_);
            close(fd_);
        }
    }
};
```

**Cache Optimization:**
```cpp
// C++23 std::flat_map for cache-friendly storage
#include <flat_map>

class CorrelationCache {
private:
    // Cache-friendly hash map (contiguous storage)
    std::flat_map<std::pair<std::string, std::string>, double> cache_;

public:
    auto get_or_compute(
        const std::string& symbol_a,
        const std::string& symbol_b,
        auto compute_fn
    ) -> double {
        auto key = std::make_pair(
            std::min(symbol_a, symbol_b),
            std::max(symbol_a, symbol_b)
        );

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;  // Cache hit
        }

        // Cache miss - compute
        double result = compute_fn(symbol_a, symbol_b);
        cache_[key] = result;

        return result;
    }
};
```

### 10.3 Scalability Analysis

**Expected Performance (64-Core Server with GPU):**

| Securities | Pairs | Timeframe | MPI+OpenMP | MPI+OpenMP+GPU | Speedup |
|------------|-------|-----------|------------|----------------|---------|
| 100 | 4,950 | 1 year | 0.5s | 0.1s | 5x |
| 500 | 124,750 | 2 years | 5s | 1s | 5x |
| 1,000 | 499,500 | 5 years | 25s | 4s | 6.25x |
| 5,000 | 12,497,500 | 5 years | 60s | 12s | 5x |
| 10,000 | 49,995,000 | 10 years | 180s | 35s | 5.1x |

**Scaling Efficiency:**
- **MPI Scaling:** 85-95% efficiency up to 64 ranks (on 64-core machine)
- **OpenMP Scaling:** 90-95% efficiency up to 64 threads
- **GPU Speedup:** 5-10x for large matrices (5K+ securities)

---

## 11. Shared Infrastructure & Loose Coupling Design

### 11.1 Shared Server Architecture (Cost Optimization)

**CRITICAL:** The Correlation Analysis Tool and Market Intelligence Engine share the same 64-128 core server to minimize costs. Both tools run on the same hardware with process isolation.

```mermaid
graph TB
    subgraph "Single 64-128 Core Server (Shared Infrastructure)"
        subgraph "Market Intelligence Engine"
            MIE1[MPI Ranks 0-1<br/>Cores 0-31<br/>News Processing]
            MIE2[Python API<br/>Port 8000]
        end

        subgraph "Correlation Analysis Tool"
            CAT1[MPI Ranks 2-3<br/>Cores 32-63<br/>Correlation Calc]
            CAT2[Python API<br/>Port 8001]
        end

        subgraph "Shared Resources"
            GPU["NVIDIA GPU<br/>Time-Shared<br/>Both Tools"]
            PG[("PostgreSQL<br/>Separate Databases<br/>market_intelligence + correlations")]
            Redis[("Redis<br/>Separate Key Prefixes<br/>mi:* and corr:*")]
            Duck["DuckDB Files<br/>Separate Directories<br/>/data/mi + /data/corr"]
        end
    end

    MIE1 & CAT1 --> GPU
    MIE1 --> PG
    CAT1 --> PG
    MIE2 --> PG & Redis
    CAT2 --> PG & Redis
    MIE1 --> Duck
    CAT1 --> Duck

    style GPU fill:#f9f,stroke:#333,stroke-width:2px
    style PG fill:#bbf,stroke:#333,stroke-width:2px
```

**Resource Allocation on Shared Server:**

| Resource | Market Intelligence | Correlation Tool | Total |
|----------|-------------------|------------------|-------|
| CPU Cores | 0-31 (32 cores) | 32-63 (32 cores) | 64 cores |
| RAM | 128GB | 128GB | 256GB |
| GPU | Time-shared | Time-shared | 1 GPU |
| Postgres | DB: market_intelligence | DB: correlations | 1 Instance |
| Redis | Prefix: mi:* | Prefix: corr:* | 1 Instance |
| Disk | /data/mi/* | /data/corr/* | Shared NVMe |

**Cost Impact:**
```
Single Server (Both Tools):        $3,450-9,700 (one-time)
Electricity:                       $150-280/month

vs. Two Separate Servers:          $6,900-19,400 (one-time)
                                   $300-560/month

Savings with Shared Infrastructure: $3,450-9,700 one-time
                                    $150-280/month
```

### 11.2 Loose Coupling Design

**Tools communicate via well-defined APIs, not direct coupling:**

```mermaid
graph LR
    subgraph "Market Intelligence Engine"
        MIE_API[REST API<br/>Port 8000<br/>/api/v1/predictions]
    end

    subgraph "Correlation Analysis Tool"
        CAT_API[REST API<br/>Port 8001<br/>/api/v1/correlations]
    end

    subgraph "Trading Decision Engine (Future)"
        TDE[Trading Engine<br/>Port 8002]
    end

    subgraph "Shared Message Bus (Optional)"
        KAFKA[Redis Streams<br/>or Kafka<br/>Events Only]
    end

    MIE_API -.HTTP.-> TDE
    CAT_API -.HTTP.-> TDE
    MIE_API -.Publish.-> KAFKA
    CAT_API -.Publish.-> KAFKA
    KAFKA -.Subscribe.-> TDE

    style MIE_API fill:#bfb,stroke:#333,stroke-width:2px
    style CAT_API fill:#bfb,stroke:#333,stroke-width:2px
    style KAFKA fill:#f9f,stroke:#333,stroke-width:2px
```

**Loose Coupling Principles:**

1. **Separate Databases:**
   - Market Intelligence: `market_intelligence` database
   - Correlation Tool: `correlations` database
   - No direct table dependencies

2. **Separate Processes:**
   - Different MPI processes
   - Different Python API servers
   - Independent restarts

3. **API-Only Communication:**
   - REST APIs for synchronous calls
   - Redis Streams/Kafka for async events (optional)
   - No shared memory between tools

4. **Shared Infrastructure Only:**
   - Same physical server
   - Same PostgreSQL instance (different DBs)
   - Same Redis instance (different key prefixes)
   - Same GPU (time-shared via CUDA streams)

5. **Consistent Toolset:**
   - Both use: C++23, Python, CUDA, MKL, OpenMP/MPI, UPC++
   - Both use: PostgreSQL, DuckDB, Redis
   - Both use: Homebrew, uv, Ansible
   - Same deployment procedures

---

## 12. Quick Proof-of-Concept Implementation Guide

**🚀 Goal:** Build a testable POC in 2-4 weeks using Tier 1 tools (Python, DuckDB, CUDA) before implementing full C++23/MPI production system.

### 12.1 POC Phase 1: DuckDB-Only (Week 1)

**Zero infrastructure setup - just DuckDB:**

```python
# File: poc/correlation_poc.py
# Quick POC using DuckDB and free data

import duckdb
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# 1. Download free historical data
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD']
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years

print("Downloading data...")
data_frames = []
for symbol in symbols:
    df = yf.download(symbol, start=start_date, end=end_date)
    df['Symbol'] = symbol
    df['Returns'] = df['Close'].pct_change()
    data_frames.append(df.reset_index())

# Combine all data
all_data = pd.concat(data_frames, ignore_index=True)

# 2. Store in Parquet (efficient storage)
all_data.to_parquet('data/market_data.parquet', index=False)

# 3. Compute correlations with DuckDB
con = duckdb.connect()

# Calculate all pairwise correlations
correlations = con.execute("""
    SELECT
        a.Symbol as symbol_a,
        b.Symbol as symbol_b,
        corr(a.Returns, b.Returns) as correlation,
        count(*) as sample_size
    FROM 'data/market_data.parquet' a
    JOIN 'data/market_data.parquet' b
        ON a.Date = b.Date
    WHERE a.Symbol < b.Symbol
      AND a.Returns IS NOT NULL
      AND b.Returns IS NOT NULL
    GROUP BY a.Symbol, b.Symbol
    HAVING abs(corr(a.Returns, b.Returns)) > 0.5
    ORDER BY abs(corr(a.Returns, b.Returns)) DESC
""").df()

print(f"\nFound {len(correlations)} significant correlations:")
print(correlations.head(10))

# 4. Rolling correlations
rolling_corr = con.execute("""
    SELECT
        a.Symbol as symbol_a,
        b.Symbol as symbol_b,
        a.Date,
        corr(a.Returns, b.Returns) OVER (
            PARTITION BY a.Symbol, b.Symbol
            ORDER BY a.Date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as rolling_corr_30d
    FROM 'data/market_data.parquet' a
    JOIN 'data/market_data.parquet' b
        ON a.Date = b.Date
    WHERE a.Symbol = 'AAPL' AND b.Symbol = 'MSFT'
    ORDER BY a.Date
""").df()

print(f"\nRolling 30-day correlation (AAPL vs MSFT):")
print(rolling_corr.tail(10))

# Save results
correlations.to_parquet('results/correlations.parquet')
rolling_corr.to_parquet('results/rolling_correlations.parquet')

print("\n✓ POC Complete! Correlations computed in seconds with DuckDB.")
```

**Expected Output:**
```
Downloading data...
Found 15 significant correlations:
   symbol_a symbol_b  correlation  sample_size
0      NVDA      AMD       0.8234         1256
1      AAPL     MSFT       0.7891         1256
2     GOOGL     META       0.7654         1256
...

✓ POC Complete! Correlations computed in seconds with DuckDB.
```

### 12.2 POC Phase 2: Add Python Parallel Processing (Week 2)

```python
# File: poc/parallel_correlation_poc.py
# POC with Python parallel processing (GIL-free Python 3.14+)

import duckdb
import polars as pl
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

def compute_correlation_pair(symbol_a, symbol_b, data):
    """Compute correlation for one pair"""
    df_a = data.filter(pl.col('Symbol') == symbol_a)
    df_b = data.filter(pl.col('Symbol') == symbol_b)

    # Merge on date
    merged = df_a.join(df_b, on='Date', suffix='_b')

    # Compute correlation
    corr = np.corrcoef(
        merged['Returns'].to_numpy(),
        merged['Returns_b'].to_numpy()
    )[0, 1]

    return {
        'symbol_a': symbol_a,
        'symbol_b': symbol_b,
        'correlation': corr,
        'sample_size': len(merged)
    }

# Load data with polars (faster than pandas)
data = pl.read_parquet('data/market_data.parquet')

# Get all symbol pairs
symbols = data['Symbol'].unique().to_list()
pairs = list(combinations(symbols, 2))

print(f"Computing {len(pairs)} correlations in parallel...")

# Parallel computation (benefits from Python 3.14+ GIL-free)
with ThreadPoolExecutor(max_workers=16) as executor:
    results = list(executor.map(
        lambda p: compute_correlation_pair(p[0], p[1], data),
        pairs
    ))

# Convert to DataFrame
correlations = pl.DataFrame(results)

# Filter significant correlations
significant = correlations.filter(
    pl.col('correlation').abs() > 0.5
).sort('correlation', descending=True)

print(f"Found {len(significant)} significant correlations")
print(significant.head(10))
```

### 12.3 POC Phase 3: Add CUDA Acceleration (Week 3)

```python
# File: poc/cuda_correlation_poc.py
# POC with CUDA acceleration using CuPy

import cupy as cp
import numpy as np
import pandas as pd

# Load data
data = pd.read_parquet('data/market_data.parquet')

# Pivot to matrix format (securities × days)
returns_pivot = data.pivot(index='Date', columns='Symbol', values='Returns')
returns_matrix = returns_pivot.fillna(0).values.T  # [n_securities × n_days]

print(f"Computing correlation matrix on GPU...")
print(f"Securities: {returns_matrix.shape[0]}, Days: {returns_matrix.shape[1]}")

# Transfer to GPU
returns_gpu = cp.asarray(returns_matrix)

# Standardize on GPU (mean=0, std=1)
mean = cp.mean(returns_gpu, axis=1, keepdims=True)
std = cp.std(returns_gpu, axis=1, keepdims=True)
returns_standardized = (returns_gpu - mean) / (std + 1e-10)

# Compute correlation matrix: corr = (X @ X.T) / (n - 1)
n_days = returns_gpu.shape[1]
correlation_matrix = (returns_standardized @ returns_standardized.T) / (n_days - 1)

# Transfer back to CPU
correlation_cpu = cp.asnumpy(correlation_matrix)

print(f"✓ Correlation matrix computed on GPU")
print(f"Shape: {correlation_cpu.shape}")
print(f"Sample correlations:")
print(correlation_cpu[:5, :5])

# Find high correlations
high_corr_indices = np.where(np.abs(correlation_cpu) > 0.7)
symbols = returns_pivot.columns.to_list()

print(f"\nHigh correlations (>0.7):")
for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
    if i < j:  # Upper triangle only
        print(f"{symbols[i]} <-> {symbols[j]}: {correlation_cpu[i, j]:.3f}")
```

### 12.4 POC Phase 4: Add PostgreSQL Storage (Week 4)

```python
# File: poc/postgres_storage_poc.py
# Store correlation results in PostgreSQL

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="correlations",
    user="corruser",
    password="your_password",
    host="localhost",
    port="5432"
)

# Load correlation results
correlations = pd.read_parquet('results/correlations.parquet')

# Insert into PostgreSQL
with conn.cursor() as cur:
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS correlation_pairs (
            id SERIAL PRIMARY KEY,
            symbol_a TEXT NOT NULL,
            symbol_b TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            correlation NUMERIC NOT NULL,
            sample_size INTEGER,
            calculated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Batch insert
    values = [
        (row['symbol_a'], row['symbol_b'], '1day',
         row['correlation'], row['sample_size'])
        for _, row in correlations.iterrows()
    ]

    execute_values(
        cur,
        """
        INSERT INTO correlation_pairs
        (symbol_a, symbol_b, timeframe, correlation, sample_size)
        VALUES %s
        """,
        values
    )

    conn.commit()

print(f"✓ Inserted {len(correlations)} correlations into PostgreSQL")

# Query results
with conn.cursor() as cur:
    cur.execute("""
        SELECT symbol_a, symbol_b, correlation
        FROM correlation_pairs
        WHERE abs(correlation) > 0.8
        ORDER BY abs(correlation) DESC
        LIMIT 10
    """)

    results = cur.fetchall()
    print("\nTop 10 strongest correlations from PostgreSQL:")
    for row in results:
        print(f"{row[0]} <-> {row[1]}: {row[2]:.3f}")

conn.close()
```

### 12.5 POC Complete Technology Stack

**Tier 1 POC Implementation (Weeks 1-4):**

```
Week 1: DuckDB Only
  ✓ Install: pip install duckdb yfinance pandas
  ✓ Download free data (Yahoo Finance)
  ✓ Compute correlations with SQL
  ✓ Cost: $0

Week 2: Add Python Parallelization
  ✓ Install: pip install polars numpy scipy
  ✓ Parallel processing with ThreadPoolExecutor
  ✓ Python 3.14+ GIL-free mode
  ✓ Cost: $0

Week 3: Add CUDA Acceleration
  ✓ Install: pip install cupy-cuda12x
  ✓ GPU correlation matrix computation
  ✓ 5-10x speedup on large datasets
  ✓ Cost: $0 (using existing GPU)

Week 4: Add PostgreSQL
  ✓ Setup: PostgreSQL + TimescaleDB
  ✓ Store correlation results
  ✓ Query historical correlations
  ✓ Cost: $0 (open-source)

Total POC Cost: $0
Total POC Time: 2-4 weeks
```

### 12.6 Consistent Toolset Across Both Tools

**Design Philosophy:** Both tools use the same technology stack for maintainability, reduced learning curve, and shared infrastructure.

| Component | Market Intelligence | Correlation Tool | Shared? |
|-----------|-------------------|------------------|---------|
| **Languages** ||||
| C++23 | ✓ (critical path) | ✓ (critical path) | Yes |
| Python 3.14+ | ✓ (ML, API) | ✓ (analytics, API) | Yes |
| SQL | ✓ (queries) | ✓ (queries) | Yes |
| **Parallel Computing** ||||
| MPI | ✓ (data processing) | ✓ (correlation calc) | Yes |
| OpenMP | ✓ (multi-threading) | ✓ (multi-threading) | Yes |
| UPC++ | ✓ (shared memory) | ✓ (shared memory) | Yes |
| CUDA | ✓ (ML inference) | ✓ (matrix ops) | Yes |
| Intel MKL | ✓ (math ops) | ✓ (BLAS, stats) | Yes |
| **Databases** ||||
| PostgreSQL 16+ | ✓ (predictions) | ✓ (correlations) | Same instance |
| TimescaleDB | ✓ (time-series) | ✓ (time-series) | Same instance |
| Apache AGE | ✓ (impact graphs) | ✗ (not needed) | Same instance |
| pgvector | ✓ (embeddings) | ✗ (not needed) | Same instance |
| DuckDB | ✓ (analytics) | ✓ (analytics) | Separate files |
| Redis | ✓ (cache) | ✓ (cache) | Same instance |
| **DevOps** ||||
| Homebrew | ✓ (toolchain) | ✓ (toolchain) | Yes |
| uv | ✓ (Python mgmt) | ✓ (Python mgmt) | Yes |
| Ansible | ✓ (deployment) | ✓ (deployment) | Yes |
| systemd | ✓ (services) | ✓ (services) | Yes |
| Docker/Podman | ✓ (containers) | ✓ (containers) | Yes |
| **Monitoring** ||||
| Prometheus | ✓ (metrics) | ✓ (metrics) | Same instance |
| Grafana | ✓ (dashboards) | ✓ (dashboards) | Same instance |
| Sentry | ✓ (errors) | ✓ (errors) | Same instance |

**Benefits of Consistent Toolset:**
- Single set of dependencies to manage
- Shared infrastructure reduces costs
- Common monitoring and logging
- Same deployment procedures
- Easy to maintain and debug
- Team members can work on both tools

### 12.7 POC Success Criteria

**Before proceeding to full C++23/MPI implementation:**

✅ **Validation Criteria:**
- [ ] Successfully compute correlations for 500+ securities
- [ ] Identify 10+ statistically significant lead-lag relationships
- [ ] Detect correlation regime changes
- [ ] Achieve < 5 second computation time for 100 securities (Python/DuckDB)
- [ ] Successfully query 5+ years of historical data from Parquet
- [ ] GPU acceleration shows 3-5x speedup vs CPU
- [ ] Correlations match known market relationships (e.g., NVDA-AMD high correlation)

**Only proceed with C++23/MPI production implementation if POC validates the approach!**

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-06 | Initial architecture design document for Trading Correlation Analysis Tool |

---

## References

1. [BigBrotherAnalytics PRD](../PRD.md)
2. [Market Intelligence Engine Architecture](./market-intelligence-engine.md)
3. [Intel MKL Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
4. [OpenMPI Documentation](https://www.open-mpi.org/doc/)
5. [CUDA cuBLAS Guide](https://docs.nvidia.com/cuda/cublas/)
6. [DuckDB Performance Guide](https://duckdb.org/docs/guides/performance/)

---

**Document Status:** Ready for implementation
**Next Steps:** Begin Tier 1 implementation with free market data
**Deployment:** Single 64-core server with GPU
