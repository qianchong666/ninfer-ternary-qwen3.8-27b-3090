// Second-round ternary kernel experiments: same math, better memory behaviour.
//
// Findings that motivate each variant (measured with kbench, RTX 3090, 23.7 MB layer):
//   T=1 gemv          0.093 ms   255 GB/s   (27% of peak)
//   T=2..4 tile gemv  0.181 ms   123 GB/s   (13%)
//   T=8 reference     0.707 ms    33.5 GB/s (3.6%)  == as slow as 8 separate decode passes
//
// Hypothesis list, cheapest first:
//   H1  the tile kernel recomputes  x + t*groups*k  (a 64-bit IMAD chain) inside the group loop,
//       for every token, every group -> hoisting it removes 4 IMADs per group per warp.
//   H2  neither kernel unrolls the group loop, so there is at most one group's worth of loads in
//       flight per warp (4 loads). Unrolling issues several groups' loads back to back, which is
//       what a latency-bound GEMV needs.
//   H3  the reference kernel gives each output row a 128-thread CTA with 7 barriers per row.
//
// This file implements H1+H2 variants and checks them against the reference kernel.
#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer::ops::detail;

namespace {

constexpr int kGroupK = 128;
constexpr int kCodeBytes = 32;
constexpr int kScaleBytes = 2;
constexpr int kWarps = kGemvWarpsPerBlock;

__device__ __forceinline__ float sc(const std::uint8_t* p) {
    return __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(p)));
}

// ---- H1+H2: T == 1, group loop unrolled by U with the two activation loads of each group issued
// before any of the arithmetic.
template <int U>
__global__ __launch_bounds__(kWarps * 32)
void gemv_v2(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
             const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
             std::int32_t rows, std::int32_t groups) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* crow = codes + static_cast<std::int64_t>(warp) * groups * kCodeBytes + lane;
    const std::uint8_t* srow = scales + static_cast<std::int64_t>(warp) * groups * kScaleBytes;
    const __nv_bfloat16* xrow = x + lane * 4;

    float acc = 0.0f;
    int g = 0;
    for (; g + U <= groups; g += U) {
        std::uint8_t raw[U];
        float scale[U];
        float a[U][4];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            raw[u] = crow[static_cast<std::int64_t>(g + u) * kCodeBytes];
            scale[u] = sc(srow + static_cast<std::int64_t>(g + u) * kScaleBytes);
            const __nv_bfloat16* base = xrow + static_cast<std::int64_t>(g + u) * kGroupK;
            const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base));
            const float2 hi =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base + 2));
            a[u][0] = lo.x;
            a[u][1] = lo.y;
            a[u][2] = hi.x;
            a[u][3] = hi.y;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const std::uint8_t r = raw[u];
            const float dot =
                fmaf(static_cast<float>(static_cast<int>(r & 3u) - 1), a[u][0],
                     fmaf(static_cast<float>(static_cast<int>((r >> 2) & 3u) - 1), a[u][1],
                          fmaf(static_cast<float>(static_cast<int>((r >> 4) & 3u) - 1), a[u][2],
                               static_cast<float>(static_cast<int>((r >> 6) & 3u) - 1) * a[u][3])));
            acc = fmaf(scale[u], dot, acc);
        }
    }
    for (; g < groups; ++g) {
        const std::uint8_t raw = crow[static_cast<std::int64_t>(g) * kCodeBytes];
        const __nv_bfloat16* base = xrow + static_cast<std::int64_t>(g) * kGroupK;
        const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base));
        const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base + 2));
        const float dot =
            fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), lo.x,
                 fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1), lo.y,
                      fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1), hi.x,
                           static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) * hi.y)));
        acc = fmaf(sc(srow + static_cast<std::int64_t>(g) * kScaleBytes), dot, acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { acc += __shfl_down_sync(0xffffffffu, acc, off); }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(acc); }
}

// ---- H1: T in 2..4, token pointers hoisted out of the group loop, group loop unrolled by U.
template <int kT, int U>
__global__ __launch_bounds__(kWarps * 32)
void tile_v2(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
             const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
             std::int32_t rows, std::int32_t groups, std::int32_t tokens,
             std::int32_t out_row_stride) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* crow = codes + static_cast<std::int64_t>(warp) * groups * kCodeBytes + lane;
    const std::uint8_t* srow = scales + static_cast<std::int64_t>(warp) * groups * kScaleBytes;

    // Hoisted: one activation row pointer per token, computed once (H1).
    const __nv_bfloat16* xrow[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) {
        xrow[t] = x + static_cast<std::int64_t>(t) * groups * kGroupK + lane * 4;
    }

    float acc[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { acc[t] = 0.0f; }

    int g = 0;
    for (; g + U <= groups; g += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const std::int64_t gi = g + u;
            const std::uint8_t raw = crow[gi * kCodeBytes];
            const float scale = sc(srow + gi * kScaleBytes);
            const float w0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
            const float w1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
            const float w2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
            const float w3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);
#pragma unroll
            for (int t = 0; t < kT; ++t) {
                if (t < tokens) {
                    const __nv_bfloat16* base = xrow[t] + gi * kGroupK;
                    const float2 lo =
                        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base));
                    const float2 hi =
                        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base + 2));
                    const float dot =
                        fmaf(w0, lo.x, fmaf(w1, lo.y, fmaf(w2, hi.x, w3 * hi.y)));
                    acc[t] = fmaf(scale, dot, acc[t]);
                }
            }
        }
    }
    for (; g < groups; ++g) {
        const std::uint8_t raw = crow[static_cast<std::int64_t>(g) * kCodeBytes];
        const float scale = sc(srow + static_cast<std::int64_t>(g) * kScaleBytes);
        const float w0 = static_cast<float>(static_cast<int>(raw & 3u) - 1);
        const float w1 = static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1);
        const float w2 = static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1);
        const float w3 = static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1);
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* base = xrow[t] + static_cast<std::int64_t>(g) * kGroupK;
                const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base));
                const float2 hi =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(base + 2));
                const float dot = fmaf(w0, lo.x, fmaf(w1, lo.y, fmaf(w2, hi.x, w3 * hi.y)));
                acc[t] = fmaf(scale, dot, acc[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        if (t < tokens) {
            float v = acc[t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) { v += __shfl_down_sync(0xffffffffu, v, off); }
            if (lane == 0) {
                out[static_cast<std::int64_t>(t) * out_row_stride + warp] = __float2bfloat16_rn(v);
            }
        }
    }
}

// ---- host side

struct Buf {
    std::uint8_t* codes = nullptr;
    std::uint8_t* scales = nullptr;
    __nv_bfloat16* x = nullptr;
    __nv_bfloat16* out = nullptr;
    int n = 0, k = 0, t = 0;
};

Buf make_buf(int n, int k, int t) {
    Buf b;
    b.n = n;
    b.k = k;
    b.t = t;
    const int groups = k / kGroupK;
    cudaMalloc(&b.codes, static_cast<std::size_t>(n) * groups * kCodeBytes);
    cudaMalloc(&b.scales, static_cast<std::size_t>(n) * groups * kScaleBytes);
    cudaMalloc(&b.x, static_cast<std::size_t>(k) * t * sizeof(__nv_bfloat16));
    cudaMalloc(&b.out, static_cast<std::size_t>(n) * t * sizeof(__nv_bfloat16));

    std::vector<std::uint8_t> h_codes(static_cast<std::size_t>(n) * groups * kCodeBytes);
    std::vector<std::uint8_t> h_scales(static_cast<std::size_t>(n) * groups * kScaleBytes);
    std::vector<float> h_x(static_cast<std::size_t>(k) * t);
    std::uint32_t s = 12345u;
    auto next = [&s]() {
        s = s * 1664525u + 1013904223u;
        return (s >> 8) & 0xffffffu;
    };
    for (auto& c : h_codes) { c = static_cast<std::uint8_t>(next()); }
    for (std::size_t i = 0; i < h_scales.size(); i += 2) {
        const float d = 0.001f + static_cast<float>(next() % 1000u) * 1e-6f;
        const __half h = __float2half(d);
        const std::uint16_t bits = __half_as_ushort(h);
        h_scales[i] = static_cast<std::uint8_t>(bits & 0xffu);
        h_scales[i + 1] = static_cast<std::uint8_t>(bits >> 8);
    }
    for (auto& v : h_x) { v = (static_cast<float>(next() % 2000u) - 1000.0f) / 500.0f; }

    std::vector<__nv_bfloat16> h_xb(h_x.size());
    for (std::size_t i = 0; i < h_x.size(); ++i) { h_xb[i] = __float2bfloat16(h_x[i]); }
    cudaMemcpy(b.codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(b.scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(b.x, h_xb.data(), h_xb.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(b.out, 0, static_cast<std::size_t>(n) * t * sizeof(__nv_bfloat16));
    return b;
}

std::vector<float> fetch(const Buf& b, int t) {
    std::vector<__nv_bfloat16> h(static_cast<std::size_t>(b.n) * t);
    cudaMemcpy(h.data(), b.out, h.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::vector<float> f(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) { f[i] = __bfloat162float(h[i]); }
    return f;
}

void run_reference(const Buf& b, int t) {
    const int groups = b.k / kGroupK;
    const dim3 grid(static_cast<unsigned>(b.n), static_cast<unsigned>((t + 7) / 8), 1u);
    constexpr dim3 block(kGroupK, 1u, 1u);
    ternary_rowsplit_gemm_kernel<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 8>
        <<<grid, block>>>(b.x, b.codes, nullptr, b.scales, b.out, b.n, b.k, t, groups, b.n);
    cudaDeviceSynchronize();
}

template <int U>
void run_gemv_v2(const Buf& b) {
    const int groups = b.k / kGroupK;
    const unsigned grid = static_cast<unsigned>((b.n + kWarps - 1) / kWarps);
    gemv_v2<U><<<grid, kWarps * 32>>>(b.x, b.codes, b.scales, b.out, b.n, groups);
    cudaDeviceSynchronize();
}

template <int kT, int U>
void run_tile_v2(const Buf& b, int t) {
    const int groups = b.k / kGroupK;
    const unsigned grid = static_cast<unsigned>((b.n + kWarps - 1) / kWarps);
    tile_v2<kT, U><<<grid, kWarps * 32>>>(b.x, b.codes, b.scales, b.out, b.n, groups, t, b.n);
    cudaDeviceSynchronize();
}

template <class F>
double time_it(F&& f, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) { f(); }
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) { f(); }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return static_cast<double>(ms) / iters;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        m = std::fmax(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return m;
}

} // namespace

int main() {
    const int n = 17408;
    const int k = 5120;
    const int iters = 50;
    const double wbytes = static_cast<double>(n) * (k / kGroupK) * 34.0;

    std::printf("shape %dx%d, weights/layer %.1f MB\n\n", n, k, wbytes / 1e6);

    // ---- correctness first: every variant must match the reference kernel
    {
        Buf b = make_buf(n, k, 4);
        run_reference(b, 4);
        const std::vector<float> ref = fetch(b, 4);
        run_tile_v2<4, 4>(b, 4);
        const std::vector<float> v2 = fetch(b, 4);
        std::printf("correctness tile_v2<4,4> vs reference: max|diff| = %.6g\n",
                    max_abs_diff(ref, v2));
        std::printf("correctness tile_v2<4,1> vs reference: max|diff| = %.6g\n",
                    [&] {
                        run_tile_v2<4, 1>(b, 4);
                        return max_abs_diff(ref, fetch(b, 4));
                    }());
    }
    {
        Buf b = make_buf(n, k, 1);
        run_gemv_v2<1>(b);
        const std::vector<float> a = fetch(b, 1);
        run_gemv_v2<4>(b);
        const std::vector<float> c = fetch(b, 1);
        std::printf("correctness gemv_v2<4> vs gemv_v2<1>: max|diff| = %.6g\n\n",
                    max_abs_diff(a, c));
    }

    // ---- timing: T = 1
    {
        Buf b = make_buf(n, k, 1);
        const int groups = k / kGroupK;
        const unsigned grid = static_cast<unsigned>((n + kWarps - 1) / kWarps);
        const double ms_ref = time_it(
            [&] { ternary_pq2_gemv_kernel<<<grid, kWarps * 32>>>(b.x, b.codes, b.scales, b.out, n,
                                                               groups); },
            iters);
        std::printf("T=1  baseline gemv     %7.3f ms  %7.1f GB/s\n", ms_ref, wbytes / (ms_ref * 1e6));
        for (int U : {2, 4, 8}) {
            double ms = 0.0;
            if (U == 2) { ms = time_it([&] { run_gemv_v2<2>(b); }, iters); }
            if (U == 4) { ms = time_it([&] { run_gemv_v2<4>(b); }, iters); }
            if (U == 8) { ms = time_it([&] { run_gemv_v2<8>(b); }, iters); }
            std::printf("T=1  gemv_v2 unroll=%-2d %7.3f ms  %7.1f GB/s  (%.2fx baseline)\n", U, ms,
                        wbytes / (ms * 1e6), ms_ref / ms);
        }
    }

    // ---- timing: T = 2..4 (verify pass)
    for (int t : {2, 3, 4}) {
        Buf b = make_buf(n, k, t);
        const int groups = k / kGroupK;
        const unsigned grid = static_cast<unsigned>((n + kWarps - 1) / kWarps);
        const double ms_base = time_it(
            [&] {
                ternary_pq2_gemv_tile_kernel<4><<<grid, kWarps * 32>>>(b.x, b.codes, b.scales, b.out,
                                                                     n, groups, t, n);
            },
            iters);
        std::printf("\nT=%d  baseline tile<4>   %7.3f ms  %7.1f GB/s  %6.1f tok/s\n", t, ms_base,
                    wbytes / (ms_base * 1e6), t * 1000.0 / ms_base);
        double ms;
        ms = time_it([&] { run_tile_v2<4, 1>(b, t); }, iters);
        std::printf("T=%d  tile_v2<4,1>      %7.3f ms  %7.1f GB/s  %6.1f tok/s  (%.2fx)\n", t, ms,
                    wbytes / (ms * 1e6), t * 1000.0 / ms, ms_base / ms);
        ms = time_it([&] { run_tile_v2<4, 4>(b, t); }, iters);
        std::printf("T=%d  tile_v2<4,4>      %7.3f ms  %7.1f GB/s  %6.1f tok/s  (%.2fx)\n", t, ms,
                    wbytes / (ms * 1e6), t * 1000.0 / ms, ms_base / ms);
    }
    return 0;
}
