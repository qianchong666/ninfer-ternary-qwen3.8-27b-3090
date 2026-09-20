// Standalone harness: run the REAL ternary GEMM kernel and the REAL rotation kernel on a real
// artifact payload, so the composed math (plane pointers -> 2-bit decode -> sign block -> rotation
// -> matmul) can be compared against an independent numpy reference.
//
// It deliberately reconstructs the plane pointers the way row_split_geometry() does
// ([base plane][high plane][scale plane]) instead of going through the binder, because the
// binder itself is one of the things under test.
//
// Build:
//   nvcc -O2 -std=c++20 -arch=sm_89 -Xcompiler /wd4819 \
//        -I <src> -o gemm_test.exe gemm_test.cu

#include "ops/linear/ternary/ternary_rotation_kernels.cuh"
#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr const char* kDir     = "<BUILD_ROOT>/bm2out/";
constexpr int kRows            = 4096;   // text/layers/0/gdn/query_key
constexpr int kCols            = 5120;
constexpr int kGroups          = kCols / 128;
constexpr int kCodeBytes       = 32;     // PQ2_0
constexpr int kScaleBytes      = 2;
constexpr std::size_t kPayload = static_cast<std::size_t>(kRows) * kGroups * kCodeBytes +
                                 static_cast<std::size_t>(kRows) * kGroups * kScaleBytes;

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

std::vector<std::uint8_t> read_bytes(const char* name, std::size_t expect) {
    const std::string path = std::string(kDir) + name;
    std::FILE* file        = std::fopen(path.c_str(), "rb");
    if (file == nullptr) { fail(path.c_str()); }
    std::vector<std::uint8_t> data(expect);
    if (std::fread(data.data(), 1, expect, file) != expect) { fail("short read"); }
    std::fclose(file);
    return data;
}

template <class T>
void write_f32_like(const char* name, const std::vector<T>& bf16_values) {
    const std::string path = std::string(kDir) + name;
    std::FILE* file        = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { fail(path.c_str()); }
    for (const T value : bf16_values) {
        const float as_f32 = __bfloat162float(value);
        if (std::fwrite(&as_f32, sizeof(float), 1, file) != 1) { fail("short write"); }
    }
    std::fclose(file);
}

} // namespace

int main() {
    const std::vector<std::uint8_t> payload = read_bytes("gemm.payload.bin", kPayload);
    const std::vector<std::uint8_t> signs_raw = read_bytes("gemm.signs.f32", kCols * sizeof(float));
    const std::vector<std::uint8_t> x_raw     = read_bytes("gemm.x.f32", kCols * sizeof(float));

    // ---- device buffers -----------------------------------------------------
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* scales = nullptr;
    std::uint8_t* dev_payload  = nullptr;
    check(cudaMalloc(&dev_payload, payload.size()), "cudaMalloc payload");
    check(cudaMemcpy(dev_payload, payload.data(), payload.size(), cudaMemcpyHostToDevice),
          "copy payload");
    codes  = dev_payload;
    scales = dev_payload + static_cast<std::size_t>(kRows) * kGroups * kCodeBytes;  // high plane is 0 B

    float* dev_signs = nullptr;
    check(cudaMalloc(&dev_signs, kCols * sizeof(float)), "cudaMalloc signs");
    check(cudaMemcpy(dev_signs, signs_raw.data(), kCols * sizeof(float), cudaMemcpyHostToDevice),
          "copy signs");

    std::vector<__nv_bfloat16> host_x(kCols);
    const auto* x_f32 = reinterpret_cast<const float*>(x_raw.data());
    for (int i = 0; i < kCols; ++i) { host_x[i] = __float2bfloat16_rn(x_f32[i]); }
    __nv_bfloat16* dev_x  = nullptr;
    __nv_bfloat16* dev_xr = nullptr;
    __nv_bfloat16* dev_y  = nullptr;
    check(cudaMalloc(&dev_x, kCols * sizeof(__nv_bfloat16)), "cudaMalloc x");
    check(cudaMalloc(&dev_xr, kCols * sizeof(__nv_bfloat16)), "cudaMalloc xr");
    check(cudaMalloc(&dev_y, static_cast<std::size_t>(kRows) * sizeof(__nv_bfloat16)),
          "cudaMalloc y");
    check(cudaMemcpy(dev_x, host_x.data(), kCols * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
          "copy x");

    // ---- real rotation kernel: xr = H * (s * x) ------------------------------
    const unsigned rot_grid = static_cast<unsigned>((kCols / 1024 + 7) / 8);
    ninfer::ops::detail::ternary_rotate_bf16_kernel<<<rot_grid, 256>>>(
        dev_x, dev_xr, dev_signs, kCols / 1024, kCols, /*tokens=*/1, /*perm_hd=*/0, /*perm_nk=*/0,
        /*perm_rep=*/1, /*inverse=*/0);
    check(cudaGetLastError(), "rotation kernel");

    // ---- real ternary GEMM kernel: y = W' * xr -------------------------------
    const dim3 grid(static_cast<unsigned>(kRows), 1u, 1u);
    const dim3 block(128u, 1u, 1u);
    ninfer::ops::detail::ternary_rowsplit_gemm_kernel<ninfer::ops::detail::PQ2RowSplitStorage,
                                                      ninfer::ops::detail::PQ2SimtDecodeAtom, 1>
        <<<grid, block>>>(dev_xr, codes, /*high=*/nullptr, scales, dev_y, kRows, kCols,
                          /*t=*/1, kGroups);
    check(cudaGetLastError(), "ternary gemm");
    check(cudaDeviceSynchronize(), "sync");

    std::vector<__nv_bfloat16> host_y(kRows);
    check(cudaMemcpy(host_y.data(), dev_y, host_y.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost),
          "copy y");
    write_f32_like("gemm.y.f32", host_y);
    std::printf("ran rotation + ternary GEMM on %d x %d, wrote gemm.y.f32\n", kRows, kCols);

    cudaFree(dev_payload);
    cudaFree(dev_signs);
    cudaFree(dev_x);
    cudaFree(dev_xr);
    cudaFree(dev_y);
    return 0;
}
