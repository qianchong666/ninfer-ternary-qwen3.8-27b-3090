// kbench3: tensor-core (mma.m16n8k16.bf16) ternary GEMM for the verify / small-batch path.
//
// WHY: ncu on the T=1 SIMT gemv reports Compute(SM) 81% / L1 70% / DRAM 34% -> the ternary
// kernels are INSTRUCTION bound, not bandwidth bound. Per token the SIMT kernels decode the
// 2-bit codes again for every token, so a T-token pass costs ~T decode passes and the MTP
// verify round (T=3) ends up 2.5x a plain decode step. The mma path decodes a 16x128 weight
// tile ONCE into shared memory and lets the tensor core consume it for all 8 tokens at once,
// which is what makes a verify round cost one weight read instead of ~T of them.
//
// Weight mapping is copied verbatim from PQ2SimtDecodeAtom::decode_one:
//   code = (codes[index >> 2] >> (2 * (index & 3))) & 0x3 ;  weight = (code - 1) * scale

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

using namespace ninfer::ops::detail;

#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

#define CUDA_OK(x)                                                                  \
    do {                                                                            \
        cudaError_t e_ = (x);                                                       \
        if (e_ != cudaSuccess) {                                                    \
            std::printf("CUDA error %s at kbench3.cu:%d\n", cudaGetErrorString(e_), \
                        __LINE__);                                                  \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

namespace {

constexpr int kRows  = 16;   // mma m
constexpr int kTok   = 8;    // mma n
constexpr int kGK    = 128;  // weights per group
constexpr int kPad   = 8;    // smem padding, in bf16 elements
constexpr int kSSt   = kGK + kPad;
constexpr int kWarps = 4;    // warps per block, each owns 16 rows

__device__ __forceinline__ void mma_bf16(float* c, const std::uint32_t* a,
                                         const std::uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// one byte -> 4 ternary weights -> two bf16x2 registers (low pair, high pair)
__device__ __forceinline__ void decode_byte(std::uint32_t byte, std::uint32_t& lo,
                                            std::uint32_t& hi) {
    const float f0 = static_cast<float>(static_cast<int>(byte & 0x3u) - 1);
    const float f1 = static_cast<float>(static_cast<int>((byte >> 2) & 0x3u) - 1);
    const float f2 = static_cast<float>(static_cast<int>((byte >> 4) & 0x3u) - 1);
    const float f3 = static_cast<float>(static_cast<int>((byte >> 6) & 0x3u) - 1);
    const __nv_bfloat162 p0 = __floats2bfloat162_rn(f0, f1);
    const __nv_bfloat162 p1 = __floats2bfloat162_rn(f2, f3);
    lo = *reinterpret_cast<const std::uint32_t*>(&p0);
    hi = *reinterpret_cast<const std::uint32_t*>(&p1);
}
// One warp owns kRows output rows and kTok tokens; the k-loop walks the groups.
__global__ __launch_bounds__(kWarps * 32)
void pq2_mma_kernel(const std::uint8_t* __restrict__ codes,
                    const std::uint8_t* __restrict__ scales,
                    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
                    std::int32_t rows, std::int32_t groups_per_row, std::int64_t x_row_stride) {
    __shared__ __nv_bfloat16 sw[kWarps][kRows * kSSt];

    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int row_base = (static_cast<int>(blockIdx.x) * kWarps + warp) * kRows;
    if (row_base >= rows) { return; }

    __nv_bfloat16* w_sm = sw[warp];

    // C fragment: c0,c1 -> row lane/4, tokens (lane%4)*2, +1 ; c2,c3 -> row lane/4 + 8
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int r_stage = lane >> 1;  // which of the 16 rows this thread stages
    const int hb      = lane & 1;   // which half of the group's 32 code bytes

    const int r0 = lane >> 2;        // A/C row pair (rows r0 and r0+8)
    const int cc = (lane & 3) * 2;   // A/C column pair inside a 16-wide k tile
    const int tt = lane >> 2;        // B column = token index

    for (int g = 0; g < groups_per_row; ++g) {
        // stage + decode this 16x128 tile into smem (each thread: 16 bytes = 64 weights)
        {
            const std::uint8_t* src =
                codes + (static_cast<std::int64_t>(row_base + r_stage) * groups_per_row + g) * 32 +
                hb * 16;
            const uint4 v = *reinterpret_cast<const uint4*>(src);
            const std::uint32_t bytes[4] = {v.x, v.y, v.z, v.w};
            std::uint32_t* dst = reinterpret_cast<std::uint32_t*>(w_sm + r_stage * kSSt + hb * 64);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const std::uint32_t w = bytes[i];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    std::uint32_t lo, hi;
                    decode_byte((w >> (8 * k)) & 0xffu, lo, hi);
                    dst[2 * (i * 4 + k) + 0] = lo;  // cols 4j .. 4j+1
                    dst[2 * (i * 4 + k) + 1] = hi;  // cols 4j+2 .. 4j+3
                }
            }
        }
        __syncwarp();

        // 8 mma k-steps over the 128-weight group
        float cg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int kt = 0; kt < kGK / 16; ++kt) {
            std::uint32_t a[4];
            a[0] = *reinterpret_cast<const std::uint32_t*>(w_sm + r0 * kSSt + kt * 16 + cc);
            a[1] = *reinterpret_cast<const std::uint32_t*>(w_sm + (r0 + 8) * kSSt + kt * 16 + cc);
            a[2] = *reinterpret_cast<const std::uint32_t*>(w_sm + r0 * kSSt + kt * 16 + cc + 8);
            a[3] =
                *reinterpret_cast<const std::uint32_t*>(w_sm + (r0 + 8) * kSSt + kt * 16 + cc + 8);
            const __nv_bfloat16* xb =
                x + static_cast<std::int64_t>(tt) * x_row_stride + g * kGK + kt * 16 + cc;
            std::uint32_t b[2];
            b[0] = *reinterpret_cast<const std::uint32_t*>(xb);
            b[1] = *reinterpret_cast<const std::uint32_t*>(xb + 8);
            mma_bf16(cg, a, b);
        }
        __syncwarp();

        // fold the group scale (one binary16 per group, per row)
        const float s0 = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            scales + (static_cast<std::int64_t>(row_base + r0) * groups_per_row + g) * 2)));
        const float s1 = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            scales + (static_cast<std::int64_t>(row_base + r0 + 8) * groups_per_row + g) * 2)));
        c[0] += s0 * cg[0];
        c[1] += s0 * cg[1];
        c[2] += s1 * cg[2];
        c[3] += s1 * cg[3];
    }

    // C fragment -> out[token * rows + row]
    const int t0 = (lane & 3) * 2;
    out[static_cast<std::int64_t>(t0) * rows + row_base + r0] = __float2bfloat16_rn(c[0]);
    out[static_cast<std::int64_t>(t0 + 1) * rows + row_base + r0] = __float2bfloat16_rn(c[1]);
    out[static_cast<std::int64_t>(t0) * rows + row_base + r0 + 8] = __float2bfloat16_rn(c[2]);
    out[static_cast<std::int64_t>(t0 + 1) * rows + row_base + r0 + 8] = __float2bfloat16_rn(c[3]);
}

}  // namespace
struct Shape {
    int rows;
    int k;
    const char* name;
};

template <class Fn>
double time_it(Fn fn, int iters) {
    for (int i = 0; i < 5; ++i) { fn(); }
    CUDA_OK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    CUDA_OK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) { fn(); }
    CUDA_OK(cudaEventRecord(e1));
    CUDA_OK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_OK(cudaEventDestroy(e0));
    CUDA_OK(cudaEventDestroy(e1));
    return static_cast<double>(ms) / iters;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d, %d SMs, 936 GB/s peak)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    const Shape shapes[] = {{17408, 5120, "mlp.gate/up"}, {5120, 17408, "mlp.down"}};

    for (const Shape& s : shapes) {
        const int groups = s.k / kGK;
        const std::int64_t weight_bytes =
            static_cast<std::int64_t>(s.rows) * groups * 32 + static_cast<std::int64_t>(s.rows) * groups * 2;
        std::printf("\nshape %s  %dx%d   weights/layer %.1f MB\n", s.name, s.rows, s.k,
                    static_cast<double>(weight_bytes) / 1048576.0);

        std::vector<std::uint8_t> h_codes(static_cast<size_t>(s.rows) * groups * 32);
        std::vector<std::uint8_t> h_scales(static_cast<size_t>(s.rows) * groups * 2);
        std::vector<__nv_bfloat16> h_x(static_cast<size_t>(kTok) * s.k);

        std::mt19937 rng(1234u);
        for (auto& b : h_codes) { b = static_cast<std::uint8_t>(rng() & 0xffu); }
        for (size_t i = 0; i < h_scales.size(); i += 2) {
            const __half h = __float2half_rn(0.02f + 0.001f * static_cast<float>(i % 40));
            std::memcpy(&h_scales[i], &h, 2);
        }
        for (auto& v : h_x) {
            v = __float2bfloat16_rn(static_cast<float>(static_cast<int>(rng() % 2000) - 1000) / 1000.0f);
        }

        std::uint8_t *d_codes = nullptr, *d_scales = nullptr;
        __nv_bfloat16 *d_x = nullptr, *d_ref = nullptr, *d_mma = nullptr;
        CUDA_OK(cudaMalloc(&d_codes, h_codes.size()));
        CUDA_OK(cudaMalloc(&d_scales, h_scales.size()));
        CUDA_OK(cudaMalloc(&d_x, h_x.size() * 2));
        CUDA_OK(cudaMalloc(&d_ref, static_cast<size_t>(kTok) * s.rows * 2));
        CUDA_OK(cudaMalloc(&d_mma, static_cast<size_t>(kTok) * s.rows * 2));
        CUDA_OK(cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));

        auto run_ref = [&]() {
            const dim3 grid(static_cast<unsigned>(s.rows), kTok / 8);
            ternary_rowsplit_gemm_kernel<PQ2RowSplitStorage, PQ2SimtDecodeAtom, 8>
                <<<grid, 128>>>(d_x, d_codes, nullptr, d_scales, d_ref, s.rows, s.k, kTok, groups,
                                s.rows);
        };
        auto run_mma = [&]() {
            const dim3 grid(static_cast<unsigned>((s.rows + kWarps * kRows - 1) / (kWarps * kRows)));
            pq2_mma_kernel<<<grid, kWarps * 32>>>(d_codes, d_scales, d_x, d_mma, s.rows, groups,
                                                  s.k);
        };

        run_ref();
        run_mma();
        CUDA_OK(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> h_ref(static_cast<size_t>(kTok) * s.rows);
        std::vector<__nv_bfloat16> h_mma(static_cast<size_t>(kTok) * s.rows);
        CUDA_OK(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_mma.data(), d_mma, h_mma.size() * 2, cudaMemcpyDeviceToHost));
        double maxdiff = 0.0, maxval = 0.0;
        for (size_t i = 0; i < h_ref.size(); ++i) {
            const double a = __bfloat162float(h_ref[i]);
            const double b = __bfloat162float(h_mma[i]);
            maxdiff = std::max(maxdiff, std::fabs(a - b));
            maxval = std::max(maxval, std::fabs(a));
        }
        std::printf("correctness  mma(T=8) vs reference(T=8): max|diff| = %.6g   max|value| = %.4g   rel = %.2g\n",
                    maxdiff, maxval, maxdiff / std::max(maxval, 1e-9));

        const double wb = static_cast<double>(weight_bytes);
        double ms = time_it(run_ref, 20);
        std::printf("T=8  reference t8 CTA   %7.3f ms   %7.1f GB/s   %6.0f tok/s(pass)\n", ms,
                    wb / ms / 1e6, 8.0 / (ms / 1000.0));
        ms = time_it(run_mma, 20);
        std::printf("T=8  mma (n=8)          %7.3f ms   %7.1f GB/s   %6.0f tok/s(pass)\n", ms,
                    wb / ms / 1e6, 8.0 / (ms / 1000.0));
        std::printf("     -> as an MTP round (T=4, K=3)       %7.3f ms   %6.0f t/s effective\n", ms,
                    4.0 / (ms / 1000.0));

        CUDA_OK(cudaFree(d_codes));
        CUDA_OK(cudaFree(d_scales));
        CUDA_OK(cudaFree(d_x));
        CUDA_OK(cudaFree(d_ref));
        CUDA_OK(cudaFree(d_mma));
    }
    return 0;
}
