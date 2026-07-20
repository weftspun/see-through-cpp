// Checks whether linear()'s ggml_mul_mat call suffers the same Vulkan
// quantize_y precision loss found in conv2d's mul_mat contraction (see
// ops.cpp's conv_mm / commit "Fix silent Vulkan precision loss in conv2d's
// mul_mat contraction"). Compares linear() on Vulkan against a CPU-backend
// ground truth at realistic UNet dimensions.
//
//   test_linear_precision [C_in C_out T N]   (defaults 1280 1280 4096 1)

#include "test_common.h"

#include <random>

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int64_t C_in  = argc > 1 ? atoll(argv[1]) : 1280;
    int64_t C_out = argc > 2 ? atoll(argv[2]) : 1280;
    int64_t T     = argc > 3 ? atoll(argv[3]) : 4096;
    int64_t N     = argc > 4 ? atoll(argv[4]) : 1;
    printf("shape: C_in=%lld C_out=%lld T=%lld N=%lld\n",
           (long long) C_in, (long long) C_out, (long long) T, (long long) N);

    std::mt19937 rng(42);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> xv((size_t) C_in * T * N), wv((size_t) C_in * C_out), bv((size_t) C_out);
    for (float & v : xv) v = nrm(rng);
    for (float & v : wv) v = nrm(rng);
    for (float & v : bv) v = nrm(rng);

    auto run = [&](ggml_backend_t backend) {
        Model m;
        init_graph_ctx(m, 64);
        ggml_context * ctx = m.ctx_g;
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C_in, T, N);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C_in, C_out);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C_out);
        ggml_set_name(w, "lin.weight"); ggml_set_name(b, "lin.bias");
        ggml_set_input(x); ggml_set_input(w); ggml_set_input(b);
        m.weights["lin.weight"] = w; m.weights["lin.bias"] = b;
        ggml_tensor * y = linear(m, x, "lin");
        ggml_set_output(y);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); exit(1); }
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
        ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * 4);
        ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * 4);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        std::vector<float> yv(ggml_nelements(y));
        ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * 4);
        return yv;
    };

    ggml_backend_t vk_backend = st_backend_init();
    std::vector<float> y_vk = run(vk_backend);

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    std::vector<float> y_cpu = run(cpu_backend);

    double max_err = 0, max_val = 0;
    for (size_t i = 0; i < y_vk.size(); i++) {
        max_err = std::max(max_err, (double) fabsf(y_vk[i] - y_cpu[i]));
        max_val = std::max(max_val, (double) fabsf(y_cpu[i]));
    }
    printf("linear() Vulkan vs CPU ground truth: max_abs_err=%.6f max_val=%.3f (%.4f%% relative)\n",
           max_err, max_val, max_val > 0 ? max_err / max_val * 100.0 : 0.0);
    return 0;
}
