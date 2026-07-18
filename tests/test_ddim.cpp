// M8 (scheduler half): Marigold DDIM trajectory vs diffusers.
//
//   test_ddim <reference_ddim.bin>

#include "test_common.h"
#include "scheduler.h"

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s reference.bin\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::vector<NpyArray> ref;   // init (1,D), e (1,D), traj (S,D)
    if (!read_ref(argv[1], ref, 3)) { fprintf(stderr, "failed to read %s\n", argv[1]); return 1; }
    const int S = (int) ref[2].shape[0];
    const size_t D = ref[0].data.size();

    DdimTrailing sch;
    sch.set_timesteps(S);
    printf("timesteps:");
    for (int t : sch.timesteps) printf(" %d", t);
    printf("\n");

    std::vector<float> x = ref[0].data, v(D);
    double max_abs = 0;
    for (int s = 0; s < S; s++) {
        for (size_t i = 0; i < D; i++) v[i] = 0.1f * x[i] + ref[1].data[i];
        sch.step(x, v, s);
        for (size_t i = 0; i < D; i++) {
            double d = fabs((double) x[i] - (double) ref[2].data[s * D + i]);
            if (d > max_abs) max_abs = d;
        }
    }
    printf("trajectory max_abs_diff=%.8f over %d steps\n", max_abs, S);
    printf("%s\n", max_abs < 1e-4 ? "VALIDATION PASS" : "VALIDATION FAIL");
    return max_abs < 1e-4 ? 0 : 1;
}
