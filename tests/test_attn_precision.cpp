// Checks whether attn_block's QK^T / softmax*V ggml_mul_mat calls suffer the
// same Vulkan quantize_y precision loss as conv2d/linear (see ops.cpp).
//
//   test_attn_precision [W H C head_dim]   (defaults 16 16 320 40)
#include "test_common.h"

#include <map>
#include <random>
#include <string>

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int64_t W = argc > 1 ? atoll(argv[1]) : 16;
    int64_t H = argc > 2 ? atoll(argv[2]) : 16;
    int64_t C = argc > 3 ? atoll(argv[3]) : 320;
    int64_t hd = argc > 4 ? atoll(argv[4]) : 40;
    printf("shape: W=%lld H=%lld C=%lld head_dim=%lld\n", (long long) W, (long long) H, (long long) C, (long long) hd);

    std::mt19937 rng(7);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> xv((size_t) W * H * C);
    for (float & v : xv) v = nrm(rng);

    std::map<std::string, std::vector<float>> weights;
    for (const char * n : { "attn.group_norm.weight", "attn.group_norm.bias",
                            "attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
                            "attn.to_out.0.weight", "attn.to_out.0.bias" }) {
        size_t sz = (std::string(n).find("group_norm") != std::string::npos) ? (size_t) C
                  : (std::string(n).find("bias") != std::string::npos) ? (size_t) C
                  : (size_t) C * C;
        std::vector<float> v(sz);
        for (float & f : v) f = nrm(rng);
        weights[n] = v;
    }

    auto run = [&](ggml_backend_t backend) {
        Model m;
        init_graph_ctx(m, 256);
        ggml_context * ctx = m.ctx_g;
        m.gn_groups = 32; m.gn_eps = 1e-6f; m.head_dim = (int) hd;
        ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_set_input(x);
        std::vector<ggml_tensor *> wtensors;
        for (auto & [name, data] : weights) {
            ggml_tensor * t;
            if (data.size() == (size_t) C) t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            else t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
            ggml_set_name(t, name.c_str());
            ggml_set_input(t);
            m.weights[name] = t;
            wtensors.push_back(t);
        }
        ggml_tensor * y = attn_block(m, x, "attn");
        ggml_set_output(y);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); exit(1); }
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
        for (ggml_tensor * t : wtensors) {
            auto & data = weights[ggml_get_name(t)];
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * 4);
        }
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
    printf("attn_block() Vulkan vs CPU ground truth: max_abs_err=%.6f max_val=%.3f (%.4f%% relative)\n",
           max_err, max_val, max_val > 0 ? max_err / max_val * 100.0 : 0.0);
    return 0;
}
