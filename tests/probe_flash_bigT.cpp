// One-shot diagnostic for the paused 1280px collapse (see
// docs/ggml-upstream-issues.md item 4): re-runs every attention shape/batch
// combination already gated in verify/KernelGate.lean's productionDomain,
// but with several independent seeds instead of one, and prints every
// individual violation value (not just VERIFY PASS/FAIL) so a marginal or
// seed-dependent divergence at B=1 (the class of thing the earlier ad hoc
// probe saw before it was deleted with tests/test_graph_properties.cpp)
// would be visible directly instead of averaged away.
//
// This only exercises synthetic N(0,0.5) weights, same as the Lean gate —
// it cannot rule in/out the real-checkpoint-only hypothesis in
// docs/ggml-upstream-issues.md item 4. It exists to answer one narrower
// question first: does the *existing* gate's single-seed pass hide a
// seed-dependent flash divergence at these exact shapes? If yes, that's
// actionable without ever loading real weights. If no (as expected), the
// real-weight witness case remains the only untried next step.
#include "seethrough_capi.h"

#include <cstdio>
#include <cstring>

int main() {
    // ggml_flash_attn_ext diverges from naive attention on CPU for n_head
    // >= 2 (docs/ggml-upstream-issues.md #2) -- a real but out-of-scope
    // divergence (production never enables flash_attn on CPU). Every shape
    // below has heads >= 5, so a CPU-backed run of this binary would just
    // reproduce that known bug and look like a false-positive 1280px defect
    // (this happened once already in-session: built against the CPU-only
    // `build/` dir by mistake). Refuse rather than emit misleading numbers.
    if (std::strcmp(st_device(), "CPU") == 0) {
        std::fprintf(stderr,
            "error: primary device is CPU -- this probe is meaningless there "
            "(see docs/ggml-upstream-issues.md #2). Build/run against the "
            "Vulkan-enabled build (build-vulkan/) instead.\n");
        return 1;
    }
    // tq/tk kept separate: flash-cross-77's real shape is tq=6400 (spatial),
    // tk=77 (text tokens) at the full b13 -- approximating it as tq=tk=6400
    // recreates the intractable-naive-reference OOM the b13/t6400 case was
    // split out to avoid (docs/ggml-upstream-issues.md #4's KernelGate.lean
    // comment), which crashed this probe's first attempt.
    struct Shape { int tq, tk, heads, batch; const char * label; };
    Shape shapes[] = {
        { 1600, 1600, 10, 13,   "flash-t1600-b13"  },
        { 6400, 6400, 10, 2,    "flash-t6400-b2"   },
        { 6400, 6400, 10, 1,    "flash-t6400-b1"   },
        { 4096, 4096, 5,  12,   "flash-t4096-b12"  },
        { 4096, 4096, 5,  1,    "flash-t4096-b1"   },
        { 6400, 77,   10, 13,   "flash-cross-77"   },
        { 13,   13,   10, 6400, "flash-temporal"   },
    };
    std::printf("device: %s\n", st_device());
    for (auto & s : shapes) {
        double worst = 0.0;
        for (uint64_t seed = 1; seed <= 8; seed++) {
            double v = st_witness_check_flat(
                /*op*/ 1, /*w*/0, /*h*/0, /*c*/0, /*oc*/0, /*stride*/0,
                /*heads*/ (uint32_t) s.heads, /*tq*/ (uint32_t) s.tq, /*tk*/ (uint32_t) s.tk,
                /*batch*/ (uint32_t) s.batch, /*knobs*/ 4u /*flash*/, /*seed*/ seed * 1000003u);
            if (v > worst) worst = v;
            std::printf("%-20s seed=%llu violation=%g%s\n", s.label,
                        (unsigned long long) seed, v, v > 1.0 ? "  <-- DEFECT" : "");
        }
        std::printf("  -> worst over 8 seeds: %g%s\n\n", worst, worst > 1.0 ? "  <-- SEED-DEPENDENT DEFECT" : "");
    }
    return 0;
}
