// Standalone correctness probe for mul_mat_kahan (ops.cpp): compares its
// output against a plain ggml_mul_mat computed on the CPU backend (exact
// reference for this purpose) at a shape/chunking combination that
// reproduced a real regression when embedded in the full VAE graph
// (docs/ggml-upstream-issues.md #4).
//   probe_mul_mat_kahan <K> <M> <N>
#include "ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

ggml_tensor * mul_mat_kahan_test_entry(ggml_context * ctx, ggml_tensor * a2d, ggml_tensor * b2d);

int main(int argc, char ** argv) {
    int64_t K = argc > 1 ? atoll(argv[1]) : 1152;
    int64_t M = argc > 2 ? atoll(argv[2]) : 64;
    int64_t N = argc > 3 ? atoll(argv[3]) : 8;

    std::mt19937 rng(42);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> a_data(K * M), b_data(K * N);
    for (auto & v : a_data) v = nrm(rng);
    for (auto & v : b_data) v = nrm(rng);
    // adversarial mode: mimic the real defect's extreme weight profile
    // (encoder.mid_block.resnets.1.conv1: min -9.75 vs std 0.133, ~75x
    // outlier) -- scale weights down to a small std then inject one huge
    // outlier per output column, so each dot product is dominated by a
    // near-cancelling pair of large terms.
    if (argc > 5 && std::string(argv[5]) == "--adversarial") {
        for (auto & v : b_data) v *= 0.13f;
        for (int64_t j = 0; j < N; j++) {
            b_data[j * K + 0] = 9.75f;   // one extreme positive outlier
            b_data[j * K + 1] = -9.7f;   // near-canceling extreme negative
        }
    }

    const int CHAIN = argc > 4 ? atoi(argv[4]) : 1;  // chain N independent convs in one graph/gallocr context
    const bool sep_weight_buf = argc > 7 && std::string(argv[7]) == "--sep-weight-buf";
    std::vector<float> ref;
    ggml_context * wctx = nullptr;   // separate weight context+buffer, freed by caller
    ggml_backend_buffer_t wbuf = nullptr;
    auto run = [&](bool kahan, std::vector<float> & out, ggml_backend_t dev_backend) {
        ggml_init_params ip = { ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        ggml_tensor * b_in;
        if (sep_weight_buf) {
            // mimic Model::load_backend: weight tensor lives in its OWN
            // context+buffer, separate from the per-forward-pass compute
            // graph -- exactly how conv2d's real `w` (m.get(...)) differs
            // from a plain ggml_new_tensor_2d in this same ctx.
            if (wctx) { ggml_free(wctx); }
            if (wbuf) { ggml_backend_buffer_free(wbuf); }
            ggml_init_params wip = { ggml_tensor_overhead() * 8, nullptr, true };
            wctx = ggml_init(wip);
            b_in = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, N);
            wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_get_default_buffer_type(dev_backend));
        } else {
            b_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            ggml_set_input(b_in);
        }
        ggml_set_input(a_in);
        // make a/b INTERMEDIATE tensors (like ggml_im2col's real output),
        // not input tensors (which gallocr protects from early reuse) --
        // testing whether that's what the real conv2d bug depends on
        const bool intermediate = argc > 6 && std::string(argv[6]) == "--intermediate";
        ggml_tensor * a = intermediate ? ggml_scale(ctx, a_in, 1.0f) : a_in;
        ggml_tensor * b = intermediate && !sep_weight_buf ? ggml_scale(ctx, b_in, 1.0f) : b_in;
        ggml_tensor * r = kahan ? mul_mat_kahan_test_entry(ctx, a, b) : ggml_mul_mat(ctx, a, b);
        const bool sequential = argc > 8 && std::string(argv[8]) == "--sequential";
        if (sequential) {
            // genuinely DEPENDENT chain: feed each stage's real output into
            // the next (transposed (M,N)->(N,M) to serve as the next "a"),
            // through a silu nonlinearity -- mimicking resnet_block's
            // conv1->norm2->silu->conv2 data dependency, unlike the
            // zero-weighted "junk" chain below which never affects r.
            for (int i = 1; i < CHAIN; i++) {
                // ggml_norm: per-row (ne0) mean-variance normalize, like
                // GroupNorm keeping resnet activations bounded between convs
                // -- omitting this is what made the earlier version's
                // reference blow up exponentially (67 -> 107076 over 5
                // stages), which was legitimate growth of an unstable test
                // harness, not a Kahan-specific error.
                ggml_tensor * normed = ggml_norm(ctx, r, 1e-5f);
                ggml_tensor * act = ggml_silu(ctx, normed);                  // (M, N)
                ggml_tensor * a2 = ggml_cont(ctx, ggml_transpose(ctx, act)); // (N, M) == (K2, M)
                r = kahan ? mul_mat_kahan_test_entry(ctx, a2, b) : ggml_mul_mat(ctx, a2, b);
            }
        } else {
            // chain more unrelated ops after r, mimicking hundreds of ops
            // sharing one gallocr context downstream of each conv in the
            // real VAE graph -- but each call is independent (zero-weighted
            // junk), unlike --sequential above.
            for (int i = 1; i < CHAIN; i++) {
                ggml_tensor * junk = kahan ? mul_mat_kahan_test_entry(ctx, a, b) : ggml_mul_mat(ctx, a, b);
                r = ggml_add(ctx, r, ggml_scale(ctx, junk, 0.0f));  // keep junk live, zero-weighted
            }
        }
        ggml_set_output(r);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(gf, r);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dev_backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); exit(1); }
        ggml_backend_tensor_set(a_in, a_data.data(), 0, a_data.size() * 4);
        ggml_backend_tensor_set(b_in, b_data.data(), 0, b_data.size() * 4);
        if (ggml_backend_graph_compute(dev_backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        out.resize(ggml_nelements(r));
        ggml_backend_tensor_get(r, out.data(), 0, out.size() * 4);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    };

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 8);
    run(false, ref, cpu);
    ggml_backend_free(cpu);

    ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!d) { fprintf(stderr, "no GPU device\n"); return 1; }
    ggml_backend_t gpu = ggml_backend_dev_init(d, nullptr);
    std::vector<float> kahan_out;
    run(true, kahan_out, gpu);

    double max_abs = 0, sum = 0, ref_max = 0, kahan_max = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double diff = fabs((double) kahan_out[i] - (double) ref[i]);
        if (diff > max_abs) { max_abs = diff; }
        sum += diff;
        ref_max = std::max(ref_max, fabs((double) ref[i]));
        kahan_max = std::max(kahan_max, fabs((double) kahan_out[i]));
    }
    printf("K=%lld M=%lld N=%lld max_abs_diff=%.6f mean_abs_diff=%.6f ref_max=%.6f kahan_max=%.6f\n",
           (long long) K, (long long) M, (long long) N, max_abs, sum / ref.size(), ref_max, kahan_max);
    return max_abs < 1e-2 ? 0 : 1;
}
