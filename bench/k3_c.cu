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
