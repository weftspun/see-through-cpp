// Dump min/max/mean/nan-count for a set of weight tensor names in a gguf,
// no compute -- checks whether the down_blocks.4.attentions.0 explosion
// (docs/ggml-upstream-issues.md #4) traces to bad loaded weights or a
// genuine kernel/activation instability with otherwise-normal weights.
//   probe_weight_stats <path.gguf> <tensor_name> [<tensor_name> ...]
#include "ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <tensor_name> [...]\n", argv[0]); return 1; }
    Model m;
    if (!m.load(argv[1])) { fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }
    for (int i = 2; i < argc; i++) {
        if (!m.has(argv[i])) { printf("%-60s MISSING\n", argv[i]); continue; }
        ggml_tensor * t = m.get(argv[i]);
        size_t n = ggml_nelements(t);
        std::vector<float> v(n);
        if (t->type == GGML_TYPE_F32) {
            memcpy(v.data(), t->data, n * 4);
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> raw(n);
            memcpy(raw.data(), t->data, n * 2);
            for (size_t k = 0; k < n; k++) { v[k] = ggml_fp16_to_fp32(raw[k]); }
        } else {
            printf("%-60s ne=(%lld,%lld,%lld,%lld) type=%d (unsupported dtype for stats)\n",
                   argv[i], (long long) t->ne[0], (long long) t->ne[1],
                   (long long) t->ne[2], (long long) t->ne[3], t->type);
            continue;
        }
        double sum = 0, sum2 = 0, amin = 1e30, amax = -1e30;
        int nnan = 0, ninf = 0;
        for (float x : v) {
            if (std::isnan(x)) { nnan++; continue; }
            if (std::isinf(x)) { ninf++; continue; }
            sum += x; sum2 += (double) x * x;
            amin = std::min(amin, (double) x); amax = std::max(amax, (double) x);
        }
        double mu = sum / n, sd = sqrt(std::max(0.0, sum2 / n - mu * mu));
        printf("%-60s ne=(%lld,%lld,%lld,%lld) type=%d mean=%.6f std=%.6f min=%.6f max=%.6f nan=%d inf=%d\n",
               argv[i], (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
               (long long) t->ne[3], t->type, mu, sd, amin, amax, nnan, ninf);
    }
    return 0;
}
