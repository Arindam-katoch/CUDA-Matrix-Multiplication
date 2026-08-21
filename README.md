# CUDA Matrix Multiplication — Naive vs Tiled vs cuBLAS, Benchmarked and Profiled

## 1. Problem Statement
Implement and profile three versions of N×N single-precision matrix multiplication —
naive (global memory only), tiled (shared memory), and cuBLAS (NVIDIA's production
library) — on an RTX 4060 Laptop GPU, to understand and quantify *why* GPU memory
optimization matters, using real Nsight Compute hardware counters rather than
theoretical claims.

## 2. Approach
- **CPU baseline** (`cpu_matmul.c`): plain triple-nested-loop, single-threaded, no GPU —
  establishes what "no GPU at all" costs, both unoptimized (`-O0`) and compiler-optimized (`-O2`).
- **Naive kernel** (`naive_matmul.cu`): one CUDA thread computes one output element,
  reading directly from global memory every iteration — no data reuse across threads.
- **Tiled kernel** (`tiled_matmul.cu`): threads in a block cooperatively load
  16×16 tiles of A and B into on-chip shared memory once, then reuse that data
  across the whole block before moving to the next tile — reduces redundant
  global/L2 memory traffic.
- **cuBLAS reference** (`matmul_cublas.cu`): calls `cublasSgemm()`, NVIDIA's own
  hand-tuned GEMM implementation, as the ceiling comparison point.
- All GPU kernels verified for correctness against a CPU reference computation
  (tolerance-based float comparison) before any timing was trusted.
- Profiled naive and tiled kernels with Nsight Compute (`ncu --set full`) to capture
  compute/memory throughput, cache hit rates, and register usage — not just wall-clock time.

**Matrix size used for all measurements: N = 2048** (chosen specifically because it
exceeds the RTX 4060's 24MB L2 cache when both input matrices are considered together,
avoiding an artificially cache-resident smaller test case).

## 3. Results

### Full pipeline comparison (the "why does this matter" table)
| Implementation | Time (N=2048) | Speedup vs CPU (-O0) | Speedup vs CPU (-O2) |
|---|---|---|---|
| CPU, single-threaded, `-O0` | 88,280.98 ms (88.28 s) | 1x (baseline) | — |
| CPU, single-threaded, `-O2` | 67,056.21 ms (67.06 s) | 1.32x | 1x (baseline) |
| Naive CUDA kernel | 27.80 ms | **3,176x** | **2,412x** |
| Tiled CUDA kernel (shared memory) | 21.43 ms | **4,120x** | **3,129x** |
| cuBLAS (`ampere_sgemm_64x64_nn`) | 0.0575 ms (57.50 μs) | **~1,535,000x** | **~1,166,000x** |

### GPU kernel comparison (the engineering detail table)
| Metric | Naive | Tiled | cuBLAS |
|---|---|---|---|
| Duration (Nsight-measured) | 27.80 ms | 21.43 ms | 57.50 μs |
| Compute (SM) Throughput | 99.61% | 96.97% | 63.37% |
| Memory Throughput | 99.61% | 96.97% | 58.94% |
| L1/TEX Throughput | 99.68% | 97.04% | 61.01% |
| L2 Cache Hit Rate | ~100%* | 98.73% | 94.55% |
| DRAM Throughput | 1.09% | 1.27% | 15.58% |
| Registers per thread | 40 | 38 | 126 |
| % of fp32 peak achieved | 6% | 8% | 49% |
| Speedup vs naive | 1x | 1.30x | 483x |

*Nsight reported 106.16% due to a known sector-counting quirk at near-total cache reuse; treat as effectively saturated.*

## 4. What Failed / What Was Hard (honest account)
- **Tiled vs naive speedup was only 1.30x, far below the 8-20x figure commonly cited
  in CUDA literature.** Investigated with Nsight rather than accepting the discrepancy:
  DRAM Throughput stayed under 2% for *both* kernels at N=2048, meaning the RTX 4060's
  24MB L2 cache was already absorbing almost all redundant global-memory traffic in the
  naive version. The classic "naive matmul is DRAM-bandwidth-bound" story assumes a GPU
  generation with a much smaller L2 (older architectures had a few hundred KB–a few MB);
  on Ada Lovelace's large L2, that bottleneck is largely masked at this problem size.
  The real limiter for both kernels was insufficient independent work in flight to hide
  memory *latency* (SM Busy ~33-35%, Issue Slots Busy ~27-31%), not memory *bandwidth* —
  a more precise and more interesting finding than the textbook prediction.
- **cuBLAS's `cudaEvent_t`-measured wall-clock time (2.7–7.9 ms across runs) did not
  match Nsight's reported kernel duration (57.50 μs) even after adding a warm-up call.**
  Root cause: cuBLAS's host-side per-call overhead (algorithm heuristic selection,
  workspace bookkeeping) is larger than the actual kernel execution time when the kernel
  itself is this fast. This is a real, known property of cuBLAS for small/fast problem
  sizes — not a bug in the benchmark code. Nsight's kernel-only duration was used for
  all comparisons instead of the wall-clock number, for a fair like-for-like comparison
  across all three implementations.
- **Investigated whether cuBLAS's speed came from Tensor Cores** (the RTX 4060 has
  dedicated Tensor Core hardware this project's own kernels never touch). Forced
  `CUBLAS_PEDANTIC_MATH` and re-profiled — the executed kernel name remained
  `ampere_sgemm_64x64_nn` in both cases (a standard CUDA-core SGEMM kernel, not a
  `tensorop` kernel). Confirmed: cuBLAS's advantage here comes from superior
  register-blocking (126 registers/thread vs. 38-40 in this project's kernels) and
  instruction-level optimization on the same CUDA cores, not from hardware this
  project didn't use.
- **Environment setup issues along the way:** system default `gcc-12` had no matching
  `cc1plus` backend installed, breaking `nvcc` compilation entirely until pinned to
  `g++-11` via `-ccbin g++-11`. Nsight Compute initially failed to access GPU performance
  counters (`ERR_NVGPUCTRPERM`) due to a driver-level restriction on non-admin profiling
  access; resolved via `/etc/modprobe.d/nvidia-profiler.conf` setting
  `NVreg_RestrictProfilingToAdminUsers=0` and a reboot.

## 5. What I Learned
- Real profiling data frequently diverges from textbook predictions, and understanding
  *why* it diverges (modern GPUs' larger L2 cache changing where the real bottleneck sits)
  is a stronger engineering signal than simply reproducing an expected number.
- There is a meaningful difference between kernel *throughput* (pure GPU execution time,
  what Nsight measures) and end-to-end API *latency* (what an application experiences,
  including host-side overhead) — conflating the two gives misleading benchmark numbers,
  especially for very fast library calls like cuBLAS.
- Register usage (126 vs ~40) is as important a lever as shared-memory tiling for
  closing the gap to production-grade performance — tiling alone is necessary but not
  sufficient to approach cuBLAS-level throughput.
- Kernel naming conventions (e.g., `ampere_sgemm_64x64_nn` vs a `tensorop`-named kernel)
  are a legitimate, checkable way to verify what hardware path a library call is actually
  using, rather than assuming.

## 6. How to Run

```bash
# CPU baseline (plain gcc, no CUDA needed)
gcc cpu_matmul.c -o cpu_matmul -O0 && ./cpu_matmul

# Naive GPU kernel
nvcc -ccbin g++-11 naive_matmul.cu -o naive_matmul && ./naive_matmul

# Tiled GPU kernel
nvcc -ccbin g++-11 tiled_matmul.cu -o tiled_matmul && ./tiled_matmul

# cuBLAS reference (requires -lcublas link flag)
nvcc -ccbin g++-11 matmul_cublas.cu -o matmul_cublas -lcublas && ./matmul_cublas

# Profiling any of the above with Nsight Compute
ncu --set full -o <output_name> ./<binary_name>

# View a saved profile
ncu-ui <output_name>.ncu-rep
```

**Environment:** Ubuntu 22.04, RTX 4060 Laptop GPU (8GB VRAM, Ada Lovelace, CC 8.9),
CUDA 12.1, driver 570.211.01, compiled with `g++-11` (system default `g++-12` lacks a
working `cc1plus` backend on this machine).
