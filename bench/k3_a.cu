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
