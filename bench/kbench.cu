// Micro-benchmark for the ternary PQ2_0 kernels of the NInfer sm_86 port.
//
// Purpose: measure the REAL throughput of each of the three kernels on the shapes this model
// actually uses, so the T>1 rewrite has a target instead of a guess.
//
//   T = 1        ternary_pq2_gemv_kernel        (decoded one row per warp)
//   T = 2..4     ternary_pq2_gemv_tile_kernel<4>(weights read once per tile of tokens)
//   T = 8        ternary_rowsplit_gemm_kernel<...,8>  (the reference tiled kernel)
//
// Reported per case: ms/iteration, effective weight bandwidth (GB/s), and tokens/s.
#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ninfer::ops::detail;

namespace {

struct Shape {
    int n;          // output rows
    int k;          // reduction width (multiple of 128)
    const char* tag;
};

constexpr int kGroupK = 128;
constexpr int kCodeBytes = 32;
constexpr int kScaleBytes = 2;
constexpr int kWarpsPerBlock = kGemvWarpsPerBlock; // 8

double now_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return static_cast<double>(ms);
}

double bench_gemv(const Shape& s, int tokens, int iters) {
    const int groups = s.k / kGroupK;
    const std::size_t code_bytes = static_cast<std::size_t>(s.n) * groups * kCodeBytes;
    const std::size_t scale_bytes = static_cast<std::size_t>(s.n) * groups * kScaleBytes;

    std::uint8_t *codes = nullptr, *scales = nullptr;
    __nv_bfloat16 *x = nullptr, *out = nullptr;
    cudaMalloc(&codes, code_bytes);
    cudaMalloc(&scales, scale_bytes);
    cudaMalloc(&x, static_cast<std::size_t>(s.k) * tokens * sizeof(__nv_bfloat16));
    cudaMalloc(&out, static_cast<std::size_t>(s.n) * tokens * sizeof(__nv_bfloat16));
    cudaMemset(codes, 0x5a, code_bytes);
    cudaMemset(scales, 0x30, scale_bytes);
    cudaMemset(x, 0, static_cast<std::size_t>(s.k) * tokens * sizeof(__nv_bfloat16));

    const unsigned grid = static_cast<unsigned>((s.n + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const dim3 block(kWarpsPerBlock * 32, 1u, 1u);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    for (int i = 0; i < 3; ++i) {
        if (tokens <= 1) {
            ternary_pq2_gemv_kernel<<<grid, block>>>(x, codes, scales, out, s.n, groups);
        } else {
            ternary_pq2_gemv_tile_kernel<4><<<grid, block>>>(x, codes, scales, out, s.n, groups,
                                                            tokens, s.n);
        }
    }
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) {
        if (tokens <= 1) {
            ternary_pq2_gemv_kernel<<<grid, block>>>(x, codes, scales, out, s.n, groups);
        } else {
            ternary_pq2_gemv_tile_kernel<4><<<grid, block>>>(x, codes, scales, out, s.n, groups,
                                                            tokens, s.n);
        }
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    const double ms = now_ms(t0, t1) / iters;

    const double weight_bytes = static_cast<double>(s.n) * groups * (kCodeBytes + kScaleBytes);
    std::printf("  T=%-2d gemv/tile   %8.3f ms/iter   %7.1f GB/s   %7.1f tok/s\n", tokens, ms,
                weight_bytes / (ms * 1e6), tokens * 1000.0 / ms);

    cudaFree(codes);
    cudaFree(scales);
    cudaFree(x);
    cudaFree(out);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

double bench_reference_gemm(const Shape& s, int tokens, int iters) {
    const int groups = s.k / kGroupK;
    const std::size_t code_bytes = static_cast<std::size_t>(s.n) * groups * kCodeBytes;
    const std::size_t scale_bytes = static_cast<std::size_t>(s.n) * groups * kScaleBytes;

    std::uint8_t *codes = nullptr, *scales = nullptr;
    __nv_bfloat16 *x = nullptr, *out = nullptr;
    cudaMalloc(&codes, code_bytes);
    cudaMalloc(&scales, scale_bytes);
    cudaMalloc(&x, static_cast<std::size_t>(s.k) * tokens * sizeof(__nv_bfloat16));
    cudaMalloc(&out, static_cast<std::size_t>(s.n) * tokens * sizeof(__nv_bfloat16));
    cudaMemset(codes, 0x5a, code_bytes);
    cudaMemset(scales, 0x30, scale_bytes);
    cudaMemset(x, 0, static_cast<std::size_t>(s.k) * tokens * sizeof(__nv_bfloat16));

    const dim3 grid(static_cast<unsigned>(s.n), static_cast<unsigned>((tokens + 7) / 8), 1u);
    constexpr dim3 block(kGroupK, 1u, 1u);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    for (int i = 0; i < 3; ++i) {
        ternary_rowsplit_gemm_kernel<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 8>
            <<<grid, block>>>(x, codes, nullptr, scales, out, s.n, s.k, tokens, groups, s.n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) {
        ternary_rowsplit_gemm_kernel<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 8>
            <<<grid, block>>>(x, codes, nullptr, scales, out, s.n, s.k, tokens, groups, s.n);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    const double ms = now_ms(t0, t1) / iters;

    const double weight_bytes = static_cast<double>(s.n) * groups * (kCodeBytes + kScaleBytes);
    std::printf("  T=%-2d reference  %8.3f ms/iter   %7.1f GB/s   %7.1f tok/s\n", tokens, ms,
                weight_bytes / (ms * 1e6), tokens * 1000.0 / ms);

    cudaFree(codes);
    cudaFree(scales);
    cudaFree(x);
    cudaFree(out);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;
}

} // namespace

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s (sm_%d%d, %d SMs, 384-bit bus @ 19.5 Gbps = 936 GB/s peak)\n", prop.name,
                prop.major, prop.minor, prop.multiProcessorCount);

    const Shape shapes[] = {
        {17408, 5120, "mlp.gate/up 17408x5120"},
        {5120, 17408, "mlp.down    5120x17408"},
        {10240, 5120, "attn qkv    10240x5120"},
    };
    const int iters = 50;

    for (const Shape& s : shapes) {
        std::printf("\n%s   (weights/layer = %.1f MB)\n", s.tag,
                    static_cast<double>(s.n) * (s.k / kGroupK) * 34.0 / 1e6);
        for (int tokens : {1, 2, 3, 4}) {
            bench_gemv(s, tokens, iters);
        }
        bench_reference_gemm(s, 8, iters);
        bench_reference_gemm(s, 16, 20);
    }
    return 0;
}
