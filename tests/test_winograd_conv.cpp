// Winograd F(2x2,3x3) conv2d_winograd_3x3s1() vs the existing im2col-based
// conv2d() on random small tensors -- no gguf/model needed, just a synthetic
// 3x3 stride-1 pad-1 conv weight and random input, run on whatever backend
// SEETHROUGH_DEVICE selects (defaults to GPU/Vulkan, same as production).
//
//   test_winograd_conv [W H IC OC]   (defaults 8 6 5 7)

#include "test_common.h"
#include "winograd.h"

#include <random>

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int W = argc > 1 ? atoi(argv[1]) : 8;
    int H = argc > 2 ? atoi(argv[2]) : 6;
    int IC = argc > 3 ? atoi(argv[3]) : 5;
    int OC = argc > 4 ? atoi(argv[4]) : 7;
    printf("shape: W=%d H=%d IC=%d OC=%d\n", W, H, IC, OC);

    Model m;
    init_graph_ctx(m, 4096);
    ggml_context * ctx = m.ctx_g;

    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, IC, 1);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, IC, OC);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
    ggml_set_name(w, "test_conv.weight");
    ggml_set_name(b, "test_conv.bias");
    ggml_set_input(x); ggml_set_input(w); ggml_set_input(b);
    m.weights["test_conv.weight"] = w;
    m.weights["test_conv.bias"] = b;

    ggml_backend_t backend = st_backend_init();

    ggml_tensor * ref = conv2d(m, x, "test_conv", 1, 1);
    ggml_tensor * fast = conv2d_winograd_3x3s1(m, x, "test_conv");
    if (!fast) { fprintf(stderr, "conv2d_winograd_3x3s1 returned nullptr (shape not eligible)\n"); return 1; }
    ggml_set_output(ref); ggml_set_output(fast);

    std::mt19937 rng(42);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> xv(ggml_nelements(x)), wv(ggml_nelements(w)), bv(ggml_nelements(b));
    for (float & v : xv) v = nrm(rng);
    for (float & v : wv) v = nrm(rng);
    for (float & v : bv) v = nrm(rng);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, fast);
    printf("graph: %d nodes\n", ggml_graph_n_nodes(gf));
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
    ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * 4);
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * 4);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }

    NpyArray refv;
    refv.shape = { ggml_nelements(ref) };
    refv.data.resize(ggml_nelements(ref));
    ggml_backend_tensor_get(ref, refv.data.data(), 0, refv.data.size() * 4);

    std::vector<float> fastv(ggml_nelements(fast));
    ggml_backend_tensor_get(fast, fastv.data(), 0, fastv.size() * 4);
    float max_rel = 0;
    for (size_t i = 0; i < fastv.size(); i++) {
        float d = fabsf(fastv[i] - refv.data[i]);
        float rel = d / std::max(1.0f, fabsf(refv.data[i]));
        max_rel = std::max(max_rel, rel);
    }
    printf("max relative error: %.6f (%.4f%%)\n", max_rel, max_rel * 100.0);

    // --- also compute both on CPU (true ground truth, no Vulkan quantize_y
    // concerns) to see whether ref (conv2d(), im2col+mul_mat_f32, its own
    // large IC*9 contraction) or fast is the one actually diverging from
    // truth on Vulkan, rather than assuming fast is at fault just because
    // it differs from ref.
    {
        Model m2;
        init_graph_ctx(m2, 4096);
        ggml_context * ctx2 = m2.ctx_g;
        ggml_tensor * x2 = ggml_new_tensor_4d(ctx2, GGML_TYPE_F32, W, H, IC, 1);
        ggml_tensor * w2 = ggml_new_tensor_4d(ctx2, GGML_TYPE_F32, 3, 3, IC, OC);
        ggml_tensor * b2 = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, OC);
        ggml_set_name(w2, "test_conv.weight");
        ggml_set_name(b2, "test_conv.bias");
        ggml_set_input(x2); ggml_set_input(w2); ggml_set_input(b2);
        m2.weights["test_conv.weight"] = w2;
        m2.weights["test_conv.bias"] = b2;
        ggml_backend_t cpu_backend = ggml_backend_cpu_init();
        ggml_tensor * ref_cpu = conv2d(m2, x2, "test_conv", 1, 1);
        ggml_tensor * fast_cpu = conv2d_winograd_3x3s1(m2, x2, "test_conv");
        ggml_set_output(ref_cpu); ggml_set_output(fast_cpu);
        ggml_cgraph * gf2 = ggml_new_graph_custom(ctx2, 4096, false);
        ggml_build_forward_expand(gf2, ref_cpu);
        ggml_build_forward_expand(gf2, fast_cpu);
        ggml_gallocr_t alloc2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
        if (!ggml_gallocr_alloc_graph(alloc2, gf2)) { fprintf(stderr, "alloc2 failed\n"); return 1; }
        ggml_backend_tensor_set(x2, xv.data(), 0, xv.size() * 4);
        ggml_backend_tensor_set(w2, wv.data(), 0, wv.size() * 4);
        ggml_backend_tensor_set(b2, bv.data(), 0, bv.size() * 4);
        if (ggml_backend_graph_compute(cpu_backend, gf2) != GGML_STATUS_SUCCESS) { fprintf(stderr, "cpu compute failed\n"); return 1; }
        std::vector<float> ref_cpuv(ggml_nelements(ref_cpu)), fast_cpuv(ggml_nelements(fast_cpu));
        ggml_backend_tensor_get(ref_cpu, ref_cpuv.data(), 0, ref_cpuv.size() * 4);
        ggml_backend_tensor_get(fast_cpu, fast_cpuv.data(), 0, fast_cpuv.size() * 4);

        double ref_vs_cpu = 0, fast_vs_cpu = 0;
        for (size_t i = 0; i < refv.data.size(); i++) ref_vs_cpu = std::max(ref_vs_cpu, (double) fabsf(refv.data[i] - ref_cpuv[i]));
        for (size_t i = 0; i < fastv.size(); i++) fast_vs_cpu = std::max(fast_vs_cpu, (double) fabsf(fastv[i] - fast_cpuv[i]));
        printf("vs CPU ground truth: ref(Vulkan) diverges by %.6f, fast(Vulkan) diverges by %.6f\n", ref_vs_cpu, fast_vs_cpu);
        ggml_backend_free(cpu_backend);
    }

    return compare_ref(fast, refv, 1e-2);
}
