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
