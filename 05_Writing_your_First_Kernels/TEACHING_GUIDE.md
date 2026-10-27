# Section 05: Learning to Write CUDA Kernels

An approachable companion to `05_Writing_your_First_Kernels` in the Infatoshi CUDA course.

This guide assumes you understand C arrays, pointers, loops, functions, and allocation. It explains every topic in section 05, adds worked examples and exercises, and identifies places where the original examples need care. The explanations and additional examples below are teaching material; the original course files remain the starting point for comparison.

The goal is to answer four questions for any kernel:

1. Which output does each thread produce?
2. Where does its input live, and how does it reach that input?
3. Does it depend on work performed by another thread or operation?
4. How can we prove it is correct and measure its cost?

## Contents

1. [The CPU–GPU relationship](#1-the-cpugpu-relationship)
2. [Threads, blocks, grids, and warps](#2-threads-blocks-grids-and-warps)
3. [Calculating indices](#3-calculating-indices)
4. [Memory and data movement](#4-memory-and-data-movement)
5. [A complete vector-add program](#5-a-complete-vector-add-program)
6. [Synchronization and correctness](#6-synchronization-and-correctness)
7. [Matrix multiplication](#7-matrix-multiplication)
8. [Shared-memory tiling](#8-shared-memory-tiling)
9. [Measuring and profiling](#9-measuring-and-profiling)
10. [Atomics and races](#10-atomics-and-races)
11. [Streams and events](#11-streams-and-events)
12. [How these ideas connect to inference](#12-how-these-ideas-connect-to-inference)
13. [Reading the original files critically](#13-reading-the-original-files-critically)
14. [Practice plan and answers](#14-practice-plan-and-answers)
15. [Reference sheet and further reading](#15-reference-sheet-and-further-reading)

## 1. The CPU–GPU relationship

The **host** is the CPU side of your program. The **device** is the GPU. Your application uses the CPU to prepare data, allocate GPU buffers, launch computations, and consume results.

A **kernel** is a function executed by many GPU threads. You write its body once; each thread executes that body with its own thread index and local variables.

For example, a CPU adds two vectors with a loop:

```cpp
for (int i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
```

A first GPU implementation assigns one value of `i` to each thread. Thread 0 adds element 0, thread 1 adds element 1, and so on. This works because one output does not depend on another output.

That does not mean ten million additions happen in one instant. The GPU has finite execution resources and schedules the work in groups. Large workloads provide enough independent work to keep those resources busy.

The basic data path is:

```text
CPU arrays              GPU arrays                     CPU output
    |                       |                               ^
    +---- input copies ---->+---- kernel writes result ------+
                                              output copy
```

Transfers and launches have a cost. A GPU can lose to a CPU on a tiny operation even when the GPU kernel itself is fast. In an inference engine, keeping intermediate tensors on the GPU avoids paying for a round trip after every operation.

### Function annotations

| Annotation | Where the function runs | Typical purpose |
|---|---|---|
| `__global__` | GPU | Kernel entry point, usually launched by the CPU |
| `__device__` | GPU | Helper called by GPU code |
| `__host__` | CPU | Ordinary host function; normally implicit |
| `__host__ __device__` | Both, as separately compiled versions | Small helper valid in both environments |

A `__global__` function returns `void`; it writes results through arguments. The word “global” here does not mean “global memory.”

CUDA source files usually have the `.cu` extension and use CUDA C++. `nvcc` coordinates compilation of host and device code. Depending on compilation options, the executable can contain native GPU instructions and/or PTX, an intermediate representation that the driver can compile for a GPU. Host code is compiled for the host platform; it is not universally x86.

## 2. Threads, blocks, grids, and warps

CUDA organizes a kernel launch into a hierarchy:

```text
Grid: one kernel launch
  Block 0                 Block 1                 Block 2 ...
    threads 0..255          threads 0..255          threads 0..255
    shared memory           shared memory           shared memory
```

**Thread:** one logical execution of the kernel. Its index identifies its job.

**Block:** a group of threads that can cooperate using block shared memory and block barriers.

**Grid:** all blocks launched for that kernel invocation. Ordinary blocks must be able to execute independently, in any order.

**Warp:** a group of 32 threads within a block on the NVIDIA GPUs relevant to this course. Hardware schedules instructions for warps. A 256-thread block contains eight warps. A 100-thread block needs four warps, with only four active lanes in its final warp.

A streaming multiprocessor (**SM**) is a GPU execution unit that hosts blocks and their warps. Several blocks may reside on one SM if resources permit. A logical CUDA thread is not permanently assigned its own physical CUDA core.

### Launch syntax

```cpp
my_kernel<<<grid, block, shared_bytes, stream>>>(arguments);
```

| Parameter | Meaning |
|---|---|
| `grid` | Number of blocks along each dimension |
| `block` | Number of threads per block along each dimension |
| `shared_bytes` | Additional dynamic shared memory per block; default 0 |
| `stream` | Queue in which to submit the operation; default stream if omitted |

For example:

```cpp
dim3 block(16, 16);  // z defaults to 1; 256 threads per block
dim3 grid(8, 4);     // 32 blocks
my_kernel<<<grid, block>>>(arguments);
```

This launches `16 * 16 * 8 * 4 = 8192` logical threads. The 2D arrangement makes indexing convenient; it does not create a physical rectangular arrangement of processors.

For the GPUs discussed here, the maximum block size is 1024 threads, subject to additional dimension and resource limits. Starting with 128 or 256 threads for a 1D kernel is a useful experiment. The largest legal block is not automatically the fastest.

### Divergence and occupancy

If threads in one warp take different branches, the hardware must execute the needed paths for the appropriate active threads. This is **branch divergence**. A boundary check affecting the last warp is usually necessary and inexpensive relative to incorrect memory access.

**Occupancy** describes resident active warps relative to the SM's maximum. Registers, shared memory, and block limits can restrict it. More resident warps can help hide latency, but maximizing occupancy alone does not guarantee minimum runtime.

Do not assume that threads in a warp can communicate through memory safely without synchronization. Modern scheduling makes that assumption particularly unsafe.

## 3. Calculating indices

### 3.1 One-dimensional data

The central formula is:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockIdx.x * blockDim.x` is the block's starting position. `threadIdx.x` selects a position within that block.

Suppose `n = 10` and each block has four threads:

| Block | Local thread indices | Global indices | Valid outputs |
|---|---|---|---|
| 0 | 0, 1, 2, 3 | 0, 1, 2, 3 | All four |
| 1 | 0, 1, 2, 3 | 4, 5, 6, 7 | All four |
| 2 | 0, 1, 2, 3 | 8, 9, 10, 11 | Only 8 and 9 |

We need three blocks, so we round the division upward:

```cpp
int blocks = (n + threads - 1) / threads;
```

With positive integers, `10 / 4` gives 2, while `(10 + 4 - 1) / 4` gives 3. The kernel must still check `i < n`. Rounding up creates enough threads; it does not make the extra elements valid.

These simple formulas assume moderate positive sizes that fit in the chosen integer type. For large tensors, use appropriately wide index arithmetic and avoid overflow in products and ceiling division. For an empty tensor, skip the launch rather than launching zero blocks.

### 3.2 Two-dimensional data

A row-major matrix places each row consecutively in a flat allocation:

```text
Matrix:          Flat array:
  a b c          a b c d e f
  d e f          0 1 2 3 4 5
```

For a matrix with `columns` columns:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int offset = row * columns + col;
```

Map x to columns so adjacent x threads usually access adjacent elements. This helps memory coalescing, discussed below.

### 3.3 Three-dimensional data

For a volume with sizes `nx`, `ny`, and `nz`, with x varying fastest:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
if (x < nx && y < ny && z < nz) {
    int offset = x + nx * (y + ny * z);
    // Access element offset.
}
```

The course's `01 CUDA Basics/01_idxing.cu` does something related but different: it flattens the block ID, flattens the local thread ID, then combines them to give every launched thread a unique serial ID. That is useful for understanding launch structure. For indexing a real tensor, derive the address from the tensor's dimensions and layout.

GPU `printf` output can arrive in an unexpected order. A higher ID printing first does not mean its index calculation is wrong.

### 3.4 Extension: a grid-stride loop

One thread can process several elements:

```cpp
__global__ void add_grid_stride(const float* a, const float* b,
                                float* c, size_t n) {
    size_t start = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = start; i < n; i += stride)
        c[i] = a[i] + b[i];
}
```

With eight total threads, thread 2 handles indices `2, 10, 18, ...`. Neighboring threads still access neighboring elements during each iteration. This separates the amount of data from the number of launched threads.

## 4. Memory and data movement

### 4.1 The memory types you will use

| Memory | Visibility | Typical use | Important limitation |
|---|---|---|---|
| Registers | One thread | Accumulators, loop variables | Limited registers can constrain occupancy |
| Local memory | One thread | Spills and some thread-local arrays | Device-memory backing, despite the name “local” |
| Shared memory | One block | Reused tiles and communication | Limited capacity; explicit coordination needed |
| Global memory | Device threads with access to the allocation | Large arrays and tensors | Access pattern and traffic matter |
| Constant memory | Read by device code; populated from host | Small read-only parameters | Most effective when warp lanes read the same address |
| Host memory | CPU by default | Input preparation and results | Transfers or special allocation/mapping required |

L1 and L2 caches are hardware-managed caches, not substitutes for the shared-memory programming interface. L2 is shared across SMs. Global-memory accesses can hit in cache, so there is no single latency that describes every access.

The GPU's VRAM chips are generally separate from the GPU compute die. Registers and shared memory are on-chip resources.

### 4.2 Allocation sizes are bytes

```cpp
size_t bytes = size_t(n) * sizeof(float);
float* d_a = nullptr;
cudaMalloc(&d_a, bytes);
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
// Launch a kernel using d_a.
cudaFree(d_a);
```

This is a fragment; the complete example below adds error checks. `d_a` is a pointer variable stored on the host whose value identifies device memory. Ordinary host code must not dereference that device allocation.

`h_` and `d_` are naming conventions, not language features. `cudaMalloc` does not initialize the buffer. Copy input or initialize it before a kernel reads it.

`cudaMemcpyDeviceToDevice` also covers copies between buffers on the same GPU; it does not inherently mean a multi-GPU transfer.

### 4.3 Coalescing: think across a warp

Imagine adjacent threads reading these addresses:

```text
Good starting pattern:  a[0], a[1], a[2], a[3], ...
Strided pattern:        a[0], a[1024], a[2048], a[3072], ...
```

The first lets the memory system service accesses using relatively few memory transactions. The second generally touches many more regions. Alignment, access width, active lanes, and caching affect the details.

The important question is: **What addresses do neighboring threads access at the same instruction?** A single thread reading sequentially across time does not by itself establish coalescing.

## 5. A complete vector-add program

Save this code as `vector_add_tutorial.cu` in a scratch directory. This is a complete standalone program. Later code blocks marked as fragments require their surrounding allocations and host code.

It deliberately uses 1003 elements so the final block is partial. It checks CUDA calls, compares with a CPU calculation, and measures repeated kernel execution with events.

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                     \
        std::fprintf(stderr, "%s:%d: %s: %s\n",                   \
                     __FILE__, __LINE__, #call,                    \
                     cudaGetErrorString(err));                    \
        std::exit(EXIT_FAILURE);                                  \
    }                                                             \
} while (0)

__global__ void vector_add(const float* a, const float* b,
                           float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main() {
    constexpr int n = 1003;
    constexpr int threads = 256;
    constexpr int repeats = 50;
    const int blocks = (n + threads - 1) / threads;
    const size_t bytes = size_t(n) * sizeof(float);

    std::vector<float> a(n), b(n), result(n);
    for (int i = 0; i < n; ++i) {
        a[i] = float(i % 17) * 0.25f;
        b[i] = -float(i % 11) * 0.5f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // Warm up, and distinguish launch errors from execution errors.
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    bool passed = true;
    for (int i = 0; i < n; ++i) {
        float expected = a[i] + b[i];
        float tolerance = 1e-6f + 1e-5f * std::fabs(expected);
        if (!std::isfinite(result[i]) ||
            std::fabs(result[i] - expected) > tolerance) {
            std::fprintf(stderr, "Mismatch at %d: got %g, expected %g\n",
                         i, result[i], expected);
            passed = false;
            break;
        }
    }
    std::printf("%s; average device timeline interval: %.6f ms\n",
                passed ? "PASS" : "FAIL", total_ms / repeats);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

### What each stage does

1. Host vectors own CPU memory and deterministic input values.
2. Three `cudaMalloc` calls reserve GPU buffers of the same byte size.
3. Two copies initialize the input buffers. No initialization of `d_c` is needed because the kernel writes every valid output.
4. Each of four blocks launches 256 threads. Threads with `i >= 1003` do nothing.
5. The warm-up launches work before measurement and catches basic failures.
6. Events bracket repeated launches in the same stream. Waiting for `stop` makes the elapsed-time query meaningful.
7. The result is copied back and compared element by element.
8. Events and GPU allocations are released. Host vectors release themselves.

The error macro stops the program on failure. It is suitable for a small standalone experiment; production code should use structured resource ownership and cleanup on error.

### Building on your machines

When this guide was prepared, this Windows session exposed CUDA Toolkit 11.8 and an RTX 2080 SUPER. You also mentioned access to RTX 4090 cards. Use the target corresponding to the GPU where you will run the binary: `sm_75` for the 2080 SUPER, `sm_89` for a 4090. The architecture mapping comes from [NVIDIA's compute-capability table](https://developer.nvidia.com/cuda/gpus).

On Windows, run in an environment with a compatible MSVC host compiler available:

```powershell
nvcc -O2 -lineinfo -std=c++17 -arch=sm_75 vector_add_tutorial.cu -o vector_add_tutorial.exe
.\vector_add_tutorial.exe
```

On a Linux/WSL CUDA development environment with a 4090:

```bash
nvcc -O2 -lineinfo -std=c++17 -arch=sm_89 vector_add_tutorial.cu -o vector_add_tutorial
./vector_add_tutorial
```

`-lineinfo` helps profilers associate device instructions with source. `-O2` enables host optimization; do not use device debug flag `-G` for performance measurements. The `.cu` examples in the original course sometimes use POSIX `clock_gettime`, which needs adaptation for a native Windows build. This example uses CUDA events instead.

The GPU test is not a hardware-independent unit test: compiling needs a CUDA toolkit and compatible host compiler, and running needs a supported GPU/driver. Expected output begins with `PASS`; no fixed runtime is promised. This guide's code has not been compiled or executed in this session because a host C++ compiler was not available on the current PATH.

## 6. Synchronization and correctness

Synchronization is about dependencies. First identify **who produces a value** and **who consumes it**. Then choose an operation with the right scope.

| Mechanism | Called from | What it coordinates |
|---|---|---|
| Statements in one thread | GPU code | That thread's own dependent work |
| `__syncwarp(mask)` | GPU code | Participating lanes named by a warp mask |
| `__syncthreads()` | GPU code | Threads in one block and their relevant memory accesses |
| Same-stream operation order | Host submission | Dependencies between operations queued in that stream |
| `cudaStreamWaitEvent` | Host submission | A stream waits for work marked by an event |
| `cudaStreamSynchronize` | CPU | CPU waits for a stream's work |
| `cudaDeviceSynchronize` | CPU | CPU waits for preceding work on the current device |

### No barrier is needed for private work

```cpp
// Fragment inside a kernel: each thread owns a different i.
float x = a[i] * b[i];
c[i] = x + 1.0f;
```

The addition uses the multiplication result in the same thread. Other threads do not have to finish multiplying before this thread adds one.

Likewise, if two kernels are submitted to the same stream, the second can consume the first kernel's outputs without a CPU-side device synchronization between them.

### A barrier is needed when threads share intermediate work

Imagine thread 0 writes `tile[0]`, then thread 1 reads `tile[0]`. Without coordination, thread 1 could read before thread 0 writes. A block barrier after the cooperative writes and before the reads establishes the required ordering.

For the tiled algorithms in this guide, every thread must participate in every block barrier. Do not put the barrier inside an `if` that only some threads take, or return invalid edge threads before cooperative loads and barriers. Instead, make invalid loads write zero and keep the barrier unconditional.

An ordinary `__syncthreads()` does not synchronize different blocks. If a second phase needs outputs from every block, a separate kernel in the same stream is a straightforward design.

Calling `cudaDeviceSynchronize()` after a racy kernel does not repair the race. It only waits for the already-incorrect work to finish.

### Floating-point correctness

Floating-point addition is not associative: regrouping additions can change the rounded answer. GPU fused multiply-add and different reduction orders can produce small differences from a CPU reference.

Use a comparison such as:

```text
absolute_error <= absolute_tolerance + relative_tolerance * abs(reference)
```

Check unexpected NaNs and infinities explicitly. Choose tolerances based on data type, operation, input scale, and accumulation length. Do not loosen them until an unexplained error disappears. Start with tiny hand-computable cases and inspect the first mismatch.

`logf` means single-precision logarithm; `log` generally selects double precision for a double argument. Both can be used in appropriate device code. Approximate intrinsics such as `__logf` and `--use_fast_math` involve numerical tradeoffs. Keep default math behavior while establishing correctness.

## 7. Matrix multiplication

For `A` with shape `M × K` and `B` with shape `K × N`, the result `C` has shape `M × N`:

```text
C[row, col] = sum over k of A[row, k] * B[k, col]
```

For example:

```text
A = [1 2]      B = [5 6]      C = [19 22]
    [3 4]          [7 8]          [43 50]

C[0,1] = 1*6 + 2*8 = 22
```

Assign one output element to one thread. That thread loops over `K` and accumulates a dot product.

```cpp
// Kernel definition; requires a host driver to allocate/copy/validate data.
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
```

Launch fragment:

```cpp
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
CUDA_CHECK(cudaGetLastError());
```

The strides differ because the matrices have different widths: `A` uses `K`, while `B` and `C` use `N`. Confusing those widths is a common bug that square test matrices can hide.

No atomics are needed because one thread owns each `C[row, col]`. The early return is safe here because this kernel has no block-wide cooperative phase.

The approximate arithmetic count is `2*M*N*K` floating-point operations, counting a multiply and add separately. This simple kernel repeatedly requests input values across threads. Caches can help, but we can explicitly reuse input tiles with shared memory.

## 8. Shared-memory tiling

A **tile** is a small rectangular piece of a matrix. A block computes one output tile by moving along the `K` dimension in steps.

For each step:

1. Threads cooperatively load one tile from `A` and one from `B` into shared memory.
2. A barrier ensures the tiles are ready.
3. Each thread accumulates part of its output using those shared values.
4. A second barrier ensures nobody overwrites the tiles while another thread still reads them.

With a `16 × 16` tile, 256 threads load 256 values from each input matrix. They then perform 4096 multiply-add contributions. Each loaded input is reused across multiple outputs.

### A boundary-safe tiled kernel

```cpp
// Kernel definition, not a complete standalone application.
template<int T>
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;
    float sum = 0.0f;

    for (int base = 0; base < K; base += T) {
        As[ty][tx] = (row < M && base + tx < K)
            ? A[row * K + base + tx] : 0.0f;
        Bs[ty][tx] = (base + ty < K && col < N)
            ? B[(base + ty) * N + col] : 0.0f;

        __syncthreads();  // Complete this tile's cooperative loads.
        for (int k = 0; k < T; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();  // Finish reads before the next tile overwrites them.
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

Launch with exactly the matching block shape:

```cpp
constexpr int T = 16;
dim3 block(T, T);
dim3 grid((N + T - 1) / T, (M + T - 1) / T);
matmul_tiled<T><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
CUDA_CHECK(cudaGetLastError());
```

The template argument makes tile dimensions known at compile time. Shared-memory use is `2*T*T*sizeof(float)`, which is 2048 bytes for `T=16`.

### Why zero padding is necessary

Let `K=18` and `T=16`. The first iteration handles columns/rows 0 through 15. The second handles 16 and 17, while the remaining 14 positions are outside the input. Loading zeros makes those extra multiply-add contributions zero.

Threads outside the valid output rectangle still help load tiles and participate in barriers. They simply skip the final store. Do not add an early return at the beginning of this tiled kernel.

### Why tiling does not guarantee a speedup

Tiling adds barriers, address calculations, and shared-memory accesses. It can reduce global traffic but also limit occupancy. Small matrices may not benefit. A larger tile changes both reuse and resource cost, so it must be measured.

Shared memory is organized into banks; different addresses mapping to the same bank can serialize accesses. Reading the same shared address can use broadcast behavior. Learn to inspect access patterns before adding padding blindly.

This kernel teaches reuse. It does not use Tensor Cores, register tiling, or advanced pipelines and is not expected to match a tuned library GEMM.

## 9. Measuring and profiling

### 9.1 Separate correctness from performance

Use this sequence:

1. Initialize inputs deterministically.
2. Compare against a reference, including partial blocks and rectangular matrices.
3. Check CUDA errors and run relevant sanitizer tools.
4. Warm up, then measure repeated runs without a profiler attached.
5. Profile to test a specific hypothesis.
6. Change one thing, recheck correctness, and measure again.

The original profiling matrix examples do not initialize their inputs or verify their outputs. They demonstrate profiling structure, but their timing alone is not evidence of a correct implementation.

### 9.2 Define the timing boundary

| Measurement | Includes | Question it answers |
|---|---|---|
| Kernel interval using events | Device timeline between events | How expensive is the submitted GPU computation? |
| Host timer around launch plus synchronization | Submission, waiting, execution | What does this synchronous call cost the application? |
| Full operation | Transfers and compute; allocations if included | How much does this complete operation cost? |

A host timer around an asynchronous launch without a wait mostly measures submission. The course's timer around launch plus `cudaDeviceSynchronize` measures a broader scope than the kernel alone.

Events also need interpretation: for extremely short kernels, the device timeline can include gaps while the CPU submits more work. The 1003-element tutorial is a correctness demonstration, not a reliable bandwidth benchmark. Increase the workload and report the exact measurement scope.

Use several trials and report a median and spread. Record GPU model, toolkit version, compiler flags, shape, data type, and launch dimensions. Avoid benchmarking alongside unrelated GPU jobs.

### 9.3 Useful performance estimates

Vector add reads two floats and writes one float per element. A useful requested-traffic estimate is:

```text
bytes = 3 * N * sizeof(float) = 12*N
effective bandwidth in GB/s = bytes / seconds / 1e9
```

For a hypothetical `N=10,000,000` and `0.2 ms`, this gives `600 GB/s`. That is an illustration, not a measured result. Cache reuse and actual memory transactions can make requested bandwidth differ from physical DRAM bandwidth.

Matrix-multiply throughput can be estimated as:

```text
GFLOP/s = (2*M*N*K) / seconds / 1e9
```

Vector add performs little arithmetic per byte, so large instances tend to be limited by data movement. Matrix multiplication can reuse data extensively, allowing a much higher ratio of arithmetic to memory traffic. This ratio is **arithmetic intensity**.

Do not describe a speedup over a simple CPU triple loop as a speedup over optimized CPU matrix libraries. The reference and benchmark baseline serve different purposes.

### 9.4 Pick the right tool

| Tool | Use it to investigate |
|---|---|
| Nsight Systems (`nsys`) | CPU/GPU timeline, transfers, idle gaps, stream overlap |
| Nsight Compute (`ncu`) | One kernel's memory behavior, occupancy, instruction and execution bottlenecks |
| NVTX | Named application regions that make a timeline understandable |
| Compute Sanitizer | Memory and synchronization correctness issues |
| `nvidia-smi` | Device status; not precise timing of short kernels |

Example Linux/WSL commands, assuming the tools are installed and support your environment:

```bash
nsys profile --trace=cuda,nvtx --stats=true -o vector_timeline ./vector_add_tutorial
ncu --set basic --launch-count 1 ./vector_add_tutorial
compute-sanitizer --tool memcheck --leak-check full ./vector_add_tutorial
compute-sanitizer --tool initcheck ./vector_add_tutorial
compute-sanitizer --tool racecheck ./vector_add_tutorial
compute-sanitizer --tool synccheck ./vector_add_tutorial
```

On Windows substitute `.\vector_add_tutorial.exe` for the program path. Profilers may need additional installation, compatible drivers, or performance-counter permission. Consult the installed version's help if an option differs.

`memcheck` checks invalid memory access; `initcheck` detects uninitialized device global-memory reads; `racecheck` targets shared-memory access hazards; `synccheck` checks synchronization misuse. A clean result does not prove arbitrary cross-block or cross-stream logic correct. See [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

The one-launch `ncu` command profiles the first launch, which is the warm-up in our example. Select later launches when that matters. Profiling can replay and perturb work, so use ordinary timed runs for your final latency numbers. See [Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

Open `.nsys-rep` files with Nsight Systems. Nsight Compute uses its own reports.

NVTX ranges label host-side regions. A range around an asynchronous kernel launch does not automatically span the whole device execution. Correlate it with CUDA activity on the timeline. CUPTI is the lower-level tooling interface; you do not need to build a profiler to complete this chapter.

## 10. Atomics and races

Suppose many threads execute:

```cpp
*counter = *counter + 1;
```

This involves reading the old value, computing a new value, and writing it. Two threads might both read 7 and both write 8. Two increments then produce only one net increment.

| Step | Thread A | Thread B |
|---|---|---|
| 1 | Reads 7 | |
| 2 | | Reads 7 |
| 3 | Writes 8 | |
| 4 | | Writes 8 |

This table illustrates a possible lost update. A program with an unsynchronized data race has no guaranteed numerical result; do not rely on any particular incorrect count.

Use an atomic read-modify-write:

```cpp
__global__ void count_threads(int* counter) {
    atomicAdd(counter, 1);
}
```

Initialize `counter` to zero before launching. For `1000` blocks of `1000` threads, the intended result is `1,000,000`, assuming a successful legal launch and an appropriate integer range.

`atomicAdd(address, value)` adds to the target and returns its previous value. That return value can reserve unique slots:

```cpp
// Fragment: capacity must cover the maximum number of selected items.
if (selected) {
    int slot = atomicAdd(count, 1);
    output[slot] = item;
}
```

The order of reserved slots is not deterministic across threads. Also, incrementing `count` does not mean the corresponding `output[slot]` is already written for a concurrent consumer.

### Atomicity is narrower than synchronization

An atomic operation protects its target operation at its supported scope. It does not lock an entire kernel, synchronize every thread, or automatically publish unrelated data. Legacy CUDA atomic operations have relaxed ordering; producer/consumer protocols require additional reasoning. See [NVIDIA's atomic-function semantics](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#atomic-functions).

Do not mix unsynchronized ordinary writes with atomics on the same counter and assume the atomics make everything safe.

### Contention and reductions

If a million threads update one address, they compete for that location. A common improvement is to combine values within each block and make one global atomic update per block. The reduction must itself use correct barriers or warp primitives.

Floating-point atomic sums can differ between runs because the addition order can change. Integer counting without overflow avoids that particular rounding issue.

Other operations include `atomicMin`, `atomicMax`, `atomicExch`, and `atomicCAS`. Compare-and-swap conditionally replaces a value if the existing value matches the expected value. Support varies with type, operation, and architecture.

Use the built-in operations for these exercises. The course README's illustrative software locks are not general, safe replacements: its shared lock is separate for every block, and its later mutex example passes the address of an ordinary host-side struct to a kernel. Spin-lock protocols also require careful memory ordering and scheduling reasoning.

## 11. Streams and events

A **stream** is an ordered queue of GPU operations. It is not an SM, a CPU thread, or a reserved slice of the GPU.

Within one stream:

```text
copy inputs -> kernel -> copy output
```

The dependency order is preserved. Different streams may progress independently, but concurrency is not guaranteed. Resource availability, dependencies, transfer engines, and submission behavior affect overlap.

### 11.1 A single asynchronous pipeline

The following is a host-code fragment. It uses `vector_add` and `CUDA_CHECK` from the complete example, with positive `n`, `bytes`, `blocks`, and `threads` already defined.

```cpp
cudaStream_t stream;
CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

float *h_a, *h_b, *h_c;
float *d_a, *d_b, *d_c;
CUDA_CHECK(cudaMallocHost(&h_a, bytes));
CUDA_CHECK(cudaMallocHost(&h_b, bytes));
CUDA_CHECK(cudaMallocHost(&h_c, bytes));
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMalloc(&d_b, bytes));
CUDA_CHECK(cudaMalloc(&d_c, bytes));

for (int i = 0; i < n; ++i) {
    h_a[i] = float(i);
    h_b[i] = 1.0f;
}
CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream));
CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream));
vector_add<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream));

// The CPU may do independent work here. Do not reuse these buffers yet.
CUDA_CHECK(cudaStreamSynchronize(stream));
// h_c can now be read and checked against h_a + h_b.

CUDA_CHECK(cudaFree(d_a));
CUDA_CHECK(cudaFree(d_b));
CUDA_CHECK(cudaFree(d_c));
CUDA_CHECK(cudaFreeHost(h_a));
CUDA_CHECK(cudaFreeHost(h_b));
CUDA_CHECK(cudaFreeHost(h_c));
CUDA_CHECK(cudaStreamDestroy(stream));
```

This sequence is asynchronous relative to the host where supported, but its own dependent stages do not overlap with one another. Its benefit is that the CPU can do other work while it progresses.

### 11.2 Pinned memory

Ordinary `malloc` memory is pageable. `cudaMallocHost` allocates page-locked, or pinned, host memory, enabling the transfer mechanisms needed for reliable asynchronous host/device overlap. Free it with `cudaFreeHost`, not `free`.

Pageable buffers do not inherently cause a segmentation fault when used for CUDA copies. The runtime may stage transfers or block the host, reducing asynchrony. Pinned memory is primarily about transfer behavior and performance, and excessive pinning consumes resources the OS cannot page out. See [NVIDIA's transfer-overlap explanation](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc).

Keep a source buffer unchanged until its asynchronous copy completes. Do not read a destination buffer before completion. These lifetime rules matter even if a particular small experiment happens to work without waiting.

### 11.3 Dependencies across streams

Suppose stream A produces data that stream B consumes. Submission order on the CPU alone does not establish the required dependency between independent streams.

```cpp
// Fragment: streams and buffers already exist.
cudaEvent_t ready;
CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
producer<<<grid, block, 0, streamA>>>(data);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaEventRecord(ready, streamA));

CUDA_CHECK(cudaStreamWaitEvent(streamB, ready, 0));
consumer<<<grid, block, 0, streamB>>>(data);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaStreamSynchronize(streamB));
CUDA_CHECK(cudaEventDestroy(ready));
```

`producer` and `consumer` are placeholders for your own kernels. Recording the event marks completion of preceding work in stream A. The wait delays following work in stream B until that point, without making the CPU wait at the `cudaStreamWaitEvent` call.

The advanced course example multiplies a buffer by two in stream 1 and adds one in stream 2. Its event dependency is essential for obtaining `2*x + 1`. These two kernels are intentionally ordered, so their presence in two streams does not make them overlap.

### 11.4 Overlap independent chunks

To overlap transfers and compute, split a workload into independent chunks with separate buffer regions:

```text
Stream A: [H2D chunk 0] [compute chunk 0] [D2H chunk 0]
Stream B:       [H2D chunk 1] [compute chunk 1] [D2H chunk 1]
```

This is a possible timeline, not a promise. Validate results and inspect Nsight Systems to see whether overlap actually occurs. Reusing the same device buffer for both chunks can create a race. Reusing a pinned staging buffer before its copy completes can also corrupt inputs.

### 11.5 Default streams, priorities, and callbacks

The legacy default stream can introduce implicit synchronization with other blocking streams. Per-thread default-stream mode has different semantics. Explicit streams created with `cudaStreamNonBlocking` make many teaching experiments easier to reason about; “nonblocking” here specifically controls interaction with the legacy default stream.

Stream priorities are scheduling preferences, not dependency guarantees. Use events to establish order. A high-priority stream does not guarantee immediate execution or interruption of an already-running kernel. These distinctions are documented in the [CUDA programming guide's asynchronous execution section](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution).

The course uses `cudaStreamAddCallback`. For new host-function examples, study `cudaLaunchHostFunc`. Host functions/callbacks must not call CUDA APIs directly; they may notify ordinary host-side control code to submit later work. In the course example the callback is queued before the result copy, so it does not mean that the CPU result buffer is ready. See the [CUDA guide's host-functions section](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#host-functions).

## 12. How these ideas connect to inference

| Chapter concept | Inference application | Next exercise |
|---|---|---|
| Elementwise indexing | Activations, residual addition, scaling | Fuse bias, addition, and ReLU |
| Reductions and synchronization | Softmax and normalization | Compute a row maximum and row sum |
| Matrix multiplication | Linear layers and attention products | Support rectangular shapes and compare implementations |
| Shared-memory reuse | Attention tiles and matrix tiles | Explain which values a block reuses |
| Atomics | Counters, accumulation, work assignment | Compare many atomic updates with partial reductions |
| Streams/events | Overlapping independent transfers and computation | Build and trace a two-chunk pipeline |
| Profiling | Finding the layer or transfer that limits latency | Measure an entire sequence, then inspect its expensive kernel |

An illustrative fusion exercise computes:

```text
y[i] = max(0, a[i] + b[i] + bias[i])
```

A fused kernel can avoid writing and rereading intermediate arrays and reduce launch count. Verify both its values and its full-operation latency. Fusion can also increase register use, so it still needs measurement.

You can learn all of this on one GPU. Distributed inference introduces a separate layer of communication and scheduling; it is not required to understand section 05.

## 13. Reading the original files critically

Use the following map while comparing this guide with the course. Paths are relative to this guide's directory.

| Original file or topic | What to learn | Correction or limitation |
|---|---|---|
| `01 CUDA Basics/01_idxing.cu` | Flatten block and thread coordinates | Output order is nondeterministic; launch serial IDs differ from general tensor indexing |
| Basics READMEs | Host/device split and hierarchy | Registers and CUDA local memory are distinct; threads do not map permanently one-to-one onto cores |
| `02 Kernels/00_vector_add_v1.cu` | CPU reference, launch, warm-up | Its CPU timer includes launch and synchronization; add runtime error checks |
| Ceiling-division comment in v1 | Round up the number of blocks | 1025 elements with 256 threads need five blocks, not four |
| `02 Kernels/01_vector_add_v2.cu` | 1D versus 3D indexing | Actual block shape is `16*8*8=1024`; the `16*16*8` comment is inconsistent |
| v2 performance explanation | Compare launch layouts experimentally | Counting source expressions as “stores” does not reveal generated machine instructions |
| `02 Kernels/02 matmul.cu` | One thread per output, dot-product loop | Allocates a GPU-result host buffer but never copies back or verifies the GPU result |
| Kernel synchronization README | Coordinate dependencies | Correct spelling is `__syncwarp`; private multiply-then-add needs no inter-thread barrier |
| Kernel race discussion | Identify unsafe shared access | A device wait after the kernel cannot fix races inside it |
| `03 Profiling/00 nvtx_matmul.cu` | Label allocation, transfer, and compute phases | Host matrices are left uninitialized; initialize and validate before interpreting results |
| `03 Profiling/01_naive_matmul.cu` and `02_tiled_matmul.cu` | Compare reuse strategies | Device inputs are uninitialized; outputs are never validated; check each API and launch |
| Profiling README | Systems first, individual kernels second | Open `.nsys-rep` in Systems; sanitizer modes detect different error classes |
| `04 Atomics/00_atomicAdd.cu` | Lost updates versus atomic counting | Non-atomic result is not guaranteed; add error checks |
| Atomics README locks | Motivation for mutual exclusion | Shared lock does not protect across blocks; host struct address is not an ordinary device argument |
| `05 Streams/01_stream_basics.cu` | Order copies and compute | `malloc` buffers limit asynchronous overlap; stream-2 synchronization supplies the B-input dependency |
| `05 Streams/02_stream_advanced.cu` | Event-based producer/consumer ordering | Prints an uninitialized event handle before creating it; remove that diagnostic |
| Advanced stream callback | Host notification | Notification precedes D2H copy and cannot certify host-output readiness |
| Streams README priorities/pinning | Understand concurrency | Priorities do not enforce order; pageable memory does not inherently cause invalid-address failures |

Some original files also rely on indirect includes for math or library functions. Include the headers you actually use. Treat examples as learning material and verify assumptions when adapting them.

## 14. Practice plan and answers

Work through these sessions in order. Predict behavior before running, and keep a short note containing your prediction, result, and explanation.

### Session 1: Thread indexing

Run the course indexing example with smaller dimensions. Draw the block and thread IDs by hand. Explain why changing print order should not affect the set of IDs.

**Exercise:** With 256 threads per block, what index does block 3, thread 17 handle? How many blocks cover 1003 elements, and how many launched threads are unused?

**Answer:** `3*256+17=785`. Four blocks launch 1024 threads; 21 have no valid output.

### Session 2: Vector add and boundaries

Build the complete program. Test `n=1, 31, 32, 33, 255, 256, 257, 1003`. Add negative values and zeros. If you add an `n=0` case, handle it without a zero-block launch.

**Exercise:** Why is `if (i < n)` still necessary after rounding up the block count?

**Answer:** Rounding up ensures coverage but may launch extra threads. The check prevents those threads from accessing beyond the arrays.

### Session 3: Naive matrix multiplication

Build a host driver using the allocation/copy/check pattern from vector add. Start with the 2-by-2 hand example, then test `M=17, N=19, K=23`. Compare every output with a CPU reference.

**Exercise:** Why test non-square dimensions?

**Answer:** A mistaken `N` versus `K` stride can be hidden when both dimensions are equal. Odd dimensions also exercise partial blocks.

### Session 4: Tiled matrix multiplication

Add the tiled kernel to the same driver. Test `K=1, 15, 16, 17, 31` with odd `M` and `N`. Explain what each thread loads for a boundary tile. Run shared-memory race and synchronization checks.

**Exercise:** What can happen if the second barrier is removed?

**Answer:** Faster threads can load the next tile over shared values that slower threads still need for the current tile. This is a shared-memory race.

### Session 5: Timing and profiling

Measure vector add for several input sizes and block sizes of 128, 256, and 512. Record device interval and complete-operation latency separately. For matmul, compare naive and tiled versions only after both pass the same tests.

**Exercise:** Can a kernel be faster while the whole operation barely improves?

**Answer:** Yes. Transfers, allocations, or other stages may dominate total time. Optimize the part that matters for the application's measured boundary.

Use a table like this:

| GPU / flags | Operation / shape | Block | Correct? | Median ms | Spread | Timing scope |
|---|---|---|---|---|---|---|
| Your measured configuration | Your test case | Your launch | Pass/fail | Measured | Measured | Explicit boundary |

### Session 6: Atomics

Run the atomic counter example repeatedly. Explain the lost-update interleaving. Then implement per-block integer counting with one atomic update per block and compare it with one atomic per thread.

**Exercise:** Why is atomics-free vector addition correct while atomics-free shared counting is not?

**Answer:** Vector-add threads write different elements. Counter threads contend for the same element and perform a read-modify-write sequence.

### Session 7: Streams

First build a correct single-stream pipeline. Then split independent data into two chunks, use separate buffer regions and streams, and inspect the timeline. Keep pinned buffers alive until their operations complete.

**Exercise:** A producer runs in stream A and its consumer runs in stream B. Is launching A first enough?

**Answer:** No. Record an event after the producer and make B wait before submitting dependent work, or use a suitable explicit host wait. Priorities do not replace this dependency.

### Completion checkpoint

You are ready to move on when you can:

- Derive a thread's output address without copying a formula blindly.
- Explain why edge threads exist and how they behave around barriers.
- Write, validate, and time a small kernel with checked CUDA calls.
- Explain both barriers in tiled matrix multiplication.
- Distinguish an atomic update from a barrier and from a stream dependency.
- Read a timeline and identify whether transfers, launches, or compute dominate.
- Describe what your benchmark includes and what its reference establishes.

## 15. Reference sheet and further reading

| Task | Typical expression or API |
|---|---|
| Global 1D index | `blockIdx.x * blockDim.x + threadIdx.x` |
| Blocks for positive moderate `n` | `(n + threads - 1) / threads` |
| Row-major address | `row * columns + col` |
| Device allocation / release | `cudaMalloc` / `cudaFree` |
| Pinned host allocation / release | `cudaMallocHost` / `cudaFreeHost` |
| Synchronous copy interface | `cudaMemcpy` |
| Stream-associated copy | `cudaMemcpyAsync` |
| Check launch status | `cudaGetLastError` immediately after launch |
| Check completion during debugging | Checked `cudaDeviceSynchronize` or narrower stream/event wait |
| Block coordination | `__syncthreads()` |
| Atomic increment | `atomicAdd(counter, 1)` |
| Cross-stream dependency | `cudaEventRecord` + `cudaStreamWaitEvent` |
| Device timeline timing | Record two events, wait for stop, query elapsed time |

The local section 05 source and README files are the main reading for this guide. These official references support the corrections and provide deeper detail:

- [CUDA 11.8 Programming Guide](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html): a versioned reference matching the toolkit found in this session.
- [CUDA atomic functions](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#atomic-functions): operation, type, scope, and ordering details.
- [CUDA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus): architecture targets for compilation.
- [Overlapping transfers and computation](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc): streams, pinned buffers, and overlap conditions.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html): correctness tooling and its limits.
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html): profiling individual kernels.

As you read later inference code, keep asking: which thread owns the result, which memory holds the data, which dependency must be enforced, and what measurement supports the optimization?
