// DPM-Solver++ 2M SDE (diffusers DPMSolverMultistepScheduler transcription)
// in the exact configuration See-Through uses: scaled_linear betas
// [0.00085, 0.012], 1000 train steps, timestep_spacing "leading" (+1 offset),
// epsilon prediction, solver_order 2, solver_type "midpoint",
// algorithm sde-dpmsolver++, final_sigmas_type "zero". Pure CPU math.
#pragma once

#include <cstdint>
#include <vector>

struct DpmSolverSDE {
    // schedule (filled by set_timesteps)
    std::vector<int>    timesteps;   // descending, size = n_steps
    std::vector<double> sigmas;      // size = n_steps + 1, last = 0

    void set_timesteps(int n_steps);

    // one solver step: eps = UNet output at timesteps[step_index], noise = the
    // injected SDE noise (same shape); updates `sample` in place
    void step(std::vector<float> & sample, const std::vector<float> & eps,
              const std::vector<float> & noise);

    // internal multistep state
    int step_index = 0, lower_order_nums = 0;
    std::vector<float> prev_x0;      // model_outputs[-2]
};

// DDIM in the Marigold configuration: scaled_linear betas [0.00085, 0.012],
// v_prediction, rescale_betas_zero_snr, trailing spacing, set_alpha_to_one
// false, clip_sample false, eta 0 (deterministic).
struct DdimTrailing {
    std::vector<int>    timesteps;    // descending, size n_steps
    std::vector<double> ac;           // zero-SNR-rescaled alphas_cumprod [1000]
    double final_alpha_cumprod = 1.0;
    int    n_steps = 0;

    void set_timesteps(int n_steps);
    // v = UNet v-prediction at timesteps[step]; updates sample in place
    void step(std::vector<float> & sample, const std::vector<float> & v, int step);
};
