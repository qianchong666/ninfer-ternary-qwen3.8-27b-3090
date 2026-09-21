// v11 surgical probe: isolate the s8 GEMM fragment mapping from everything else.
//   1. pack check : dequantize the packed s8 back and compare against x -> validates v11_pack
//   2. gemm check : rebuild the output from the packed s8 + the codebook on the host and compare it
//                   against what the kernel computed -> validates the mma fragment mapping
// usage: v11_probe.exe [rows] [groups] [tokens]      (default 64 1 8)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../src/ninfer/src/ops/linear/ternary/ternary_mma_v2.cuh"
#include "../src/ninfer/src/ops/linear/ternary/v4_a.cuh"
#include "../src/ninfer/src/ops/linear/ternary/v11_pack.cuh"
#include "../src/ninfer/src/ops/linear/ternary/v11_c.cuh"

static std::uint32_t rs = 12345u;
static std::uint32_t rnd32() { rs = rs * 1664525u + 1013904223u; return rs; }
static float rnd() { return static_cast<float>((rnd32() >> 8) & 0xffffu) / 32768.f - 1.f; }

#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { \
    std::printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__); return 1; } } while (0)

// level of the 2-bit code holding k within a 128-k group (v8_pair's codebook, 4 levels)
static float level_of(const std::uint8_t* codes, int groups, int row, int kk) {
    const std::uint8_t byte = codes[(static_cast<std::size_t>(row) * groups + (kk >> 7)) * 32 + ((kk & 127) >> 2)];
    return static_cast<float>(static_cast<int>((byte >> ((kk & 3) * 2)) & 3u) - 1);
}

int main(int argc, char** argv) {
    const int rows   = argc > 1 ? std::atoi(argv[1]) : 64;
    const int groups = argc > 2 ? std::atoi(argv[2]) : 1;
    const int tokens = argc > 3 ? std::atoi(argv[3]) : 8;
    const int k = groups * 128;
    const int t8 = ((tokens + 7) / 8) * 8;
    const int kb = k >> 5;
    std::printf("v11_probe rows=%d groups=%d tokens=%d k=%d (t8=%d kb=%d)\n", rows, groups, tokens, k, t8, kb);

    std::vector<std::uint8_t> h_codes(static_cast<std::size_t>(rows) * groups * 32);
    for (auto& c : h_codes) c = static_cast<std::uint8_t>(rnd32() >> 16);
    std::vector<__half> h_scales(static_cast<std::size_t>(rows) * groups);
    for (auto& s : h_scales) s = __float2half_rn(0.013f + 0.004f * static_cast<float>(rnd32() & 7u));
    std::vector<__nv_bfloat16> h_x(static_cast<std::size_t>(t8) * k, __float2bfloat16(0.f));
    for (int t = 0; t < tokens; ++t)
        for (int kk = 0; kk < k; ++kk)
            h_x[static_cast<std::size_t>(t) * k + kk] = __float2bfloat16(rnd());

    std::uint8_t* d_codes = nullptr;
    std::uint8_t* d_scales = nullptr;
    __nv_bfloat16* d_x = nullptr;
    std::int8_t* d_xp = nullptr;
    __half* d_xs = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CK(cudaMalloc(&d_codes, h_codes.size()));
    CK(cudaMalloc(&d_scales, h_scales.size() * 2));
    CK(cudaMalloc(&d_x, h_x.size() * 2));
    CK(cudaMalloc(&d_xp, static_cast<std::size_t>(t8) * k));
    CK(cudaMalloc(&d_xs, static_cast<std::size_t>(t8) * kb * 2));
    CK(cudaMalloc(&d_out, static_cast<std::size_t>(t8) * static_cast<std::size_t>(rows) * 2));
    CK(cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_xp, 0, static_cast<std::size_t>(t8) * k));
    CK(cudaMemset(d_xs, 0, static_cast<std::size_t>(t8) * kb * 2));
    CK(cudaMemset(d_out, 0, static_cast<std::size_t>(t8) * rows * 2));

    launch_pack_b_s8(d_x, d_xp, d_xs, tokens, k, k, 0);
    CK(cudaDeviceSynchronize());

    // ---- 1. pack check -------------------------------------------------------------------
    std::vector<std::int8_t> hp(static_cast<std::size_t>(t8) * k);
    std::vector<__half> hs(static_cast<std::size_t>(t8) * kb);
    CK(cudaMemcpy(hp.data(), d_xp, hp.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), d_xs, hs.size() * 2, cudaMemcpyDeviceToHost));
    double px_max = 0.0;
    std::size_t px_bad = 0;
    for (int t = 0; t < tokens; ++t) {
        const int ti = t & 7;
        for (int kk = 0; kk < k; ++kk) {
            const int kl = kk & 31;
            const std::size_t w = ((static_cast<std::size_t>(t >> 3) * kb + (kk >> 5)) << 8) +
                                  (static_cast<std::size_t>(ti * 8 + ((kl & 15) >> 2) + ((kl >> 4) * 4))) * 4 +
                                  (kl & 3);
            const float deq = static_cast<float>(hp[w]) * __half2float(hs[static_cast<std::size_t>(t) * groups + (kk >> 7)]);
            const float d = std::fabs(deq - __bfloat162float(h_x[static_cast<std::size_t>(t) * k + kk]));
            px_max = std::fmax(px_max, d);
            if (d > 0.02f) ++px_bad;
        }
    }
    std::printf("pack: max|dequant - x| = %.4f   %zu of %d elements off by > 0.02\n",
                px_max, px_bad, tokens * k);

    // ---- 2. gemm check -------------------------------------------------------------------
    launch_pq2_mma_v11(d_codes, d_scales, d_xp, d_xs, d_out, rows, groups, tokens, k, rows, 0);
    CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> ho(static_cast<std::size_t>(t8) * rows);
    CK(cudaMemcpy(ho.data(), d_out, ho.size() * 2, cudaMemcpyDeviceToHost));

    // host rebuild from the *packed* activation + the codebook (no dependence on the mma layout)
    // plus the same sum from the *unquantized* x, which isolates the s8 quantization error alone.
    double nu = 0.0, de = 0.0, mx = 0.0, nq = 0.0;
    for (int t = 0; t < tokens; ++t) {
        const int ti = t & 7;
        for (int r = 0; r < rows; ++r) {
            double exp = 0.0, expx = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                const int kl = kk & 31;
                const std::size_t w = ((static_cast<std::size_t>(t >> 3) * kb + (kk >> 5)) << 8) +
                                      (static_cast<std::size_t>(ti * 8 + ((kl & 15) >> 2) + ((kl >> 4) * 4))) * 4 +
                                      (kl & 3);
                const float wv = level_of(h_codes.data(), groups, r, kk) * __half2float(h_scales[static_cast<std::size_t>(r) * groups + (kk >> 7)]);
                const float ax = static_cast<float>(hp[w]) * __half2float(hs[static_cast<std::size_t>(t) * groups + (kk >> 7)]);
                exp += static_cast<double>(wv) * static_cast<double>(ax);
                expx += static_cast<double>(wv) * static_cast<double>(__bfloat162float(h_x[static_cast<std::size_t>(t) * k + kk]));
            }
            const double got = __bfloat162float(ho[static_cast<std::size_t>(t) * rows + r]);
            nu += (got - exp) * (got - exp);
            nq += (exp - expx) * (exp - expx);
            de += exp * exp;
            mx = std::fmax(mx, std::fabs(got - exp));
        }
    }
    std::printf("gemm: rel_l2 vs host rebuild = %.3e   max|diff| %.4f\n", std::sqrt(nu / (de + 1e-30)), mx);
    std::printf("quant-only rel_l2 (packed vs raw x) = %.3e\n", std::sqrt(nq / (de + 1e-30)));

    if (rows <= 16 && tokens <= 8) {
        for (int r = 0; r < rows; ++r) {
            std::printf("r%2d:", r);
            for (int t = 0; t < tokens; ++t) {
                double exp = 0.0;
                for (int kk = 0; kk < k; ++kk) {
                    const int kl = kk & 31;
                    const std::size_t w = ((static_cast<std::size_t>(t >> 3) * kb + (kk >> 5)) << 8) +
                                          (static_cast<std::size_t>((t & 7) * 8 + ((kl & 15) >> 2) + ((kl >> 4) * 4))) * 4 + (kl & 3);
                    const float wv = level_of(h_codes.data(), groups, r, kk) * __half2float(h_scales[static_cast<std::size_t>(r) * groups + (kk >> 7)]);
                    const float ax = static_cast<float>(hp[w]) * __half2float(hs[static_cast<std::size_t>(t) * groups + (kk >> 7)]);
                    exp += static_cast<double>(wv) * static_cast<double>(ax);
                }
                std::printf("  %8.4f/%8.4f", exp, __bfloat162float(ho[static_cast<std::size_t>(t) * rows + r]));
            }
            std::printf("\n");
        }
    }
    return 0;
}
