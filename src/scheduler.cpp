#include "scheduler.h"

#include <cmath>

// alpha_t = 1/sqrt(sigma^2+1), sigma_t = sigma*alpha_t (diffusers
// _sigma_to_alpha_sigma_t)
static void sigma_to_alpha_sigma_t(double sigma, double & alpha_t, double & sigma_t) {
    alpha_t = 1.0 / sqrt(sigma * sigma + 1.0);
    sigma_t = sigma * alpha_t;
}

void DpmSolverSDE::set_timesteps(int n_steps) {
    const int    N = 1000;
    const double beta_start = 0.00085, beta_end = 0.012;

    // scaled_linear: betas = linspace(sqrt(b0), sqrt(b1), N)^2; cumprod alphas
    std::vector<double> ac(N);
    double cum = 1.0;
    for (int i = 0; i < N; i++) {
        double b = sqrt(beta_start) + (sqrt(beta_end) - sqrt(beta_start)) * i / (N - 1);
        cum *= 1.0 - b * b;
        ac[i] = cum;
    }

    // "leading" spacing with steps_offset 1: arange(0, n+1)*ratio rounded,
    // reversed, last dropped, +1
    timesteps.resize(n_steps);
    const int ratio = N / (n_steps + 1);
    for (int k = 0; k < n_steps; k++) {
        timesteps[k] = (int) llround((double) ((n_steps - k) * ratio)) + 1;
    }

    sigmas.resize(n_steps + 1);
    for (int k = 0; k < n_steps; k++) {
        int t = timesteps[k];
        sigmas[k] = sqrt((1.0 - ac[t]) / ac[t]);
    }
    sigmas[n_steps] = 0.0;                       // final_sigmas_type "zero"

    step_index = 0;
    lower_order_nums = 0;
    prev_x0.clear();
}

void DpmSolverSDE::step(std::vector<float> & sample, const std::vector<float> & eps,
                        const std::vector<float> & noise) {
    const size_t D = sample.size();
    const int    n = (int) timesteps.size();

    // convert_model_output (dpmsolver++, epsilon): x0 = (x - sigma_t*eps)/alpha_t
    double alpha_s0, sigma_s0t;
    sigma_to_alpha_sigma_t(sigmas[step_index], alpha_s0, sigma_s0t);
    std::vector<float> x0(D);
    for (size_t i = 0; i < D; i++) {
        x0[i] = (float) ((sample[i] - sigma_s0t * eps[i]) / alpha_s0);
    }

    // final_sigmas_type "zero" forces first order on the last step
    const bool lower_order_final = step_index == n - 1;

    double alpha_t, sigma_tt;
    sigma_to_alpha_sigma_t(sigmas[step_index + 1], alpha_t, sigma_tt);
    const double lambda_t  = log(alpha_t) - log(sigma_tt);
    const double lambda_s0 = log(alpha_s0) - log(sigma_s0t);
    const double h = lambda_t - lambda_s0;
    const double c_sample = sigma_tt / sigma_s0t * exp(-h);
    const double c_x0     = alpha_t * (1.0 - exp(-2.0 * h));
    const double c_noise  = sigma_tt * sqrt(1.0 - exp(-2.0 * h));

    if (lower_order_nums < 1 || lower_order_final) {
        for (size_t i = 0; i < D; i++) {
            sample[i] = (float) (c_sample * sample[i] + c_x0 * x0[i] + c_noise * noise[i]);
        }
    } else {
        // second order, midpoint: D1 = (m0 - m1)/r0, r0 = h_0/h
        double alpha_s1, sigma_s1t;
        sigma_to_alpha_sigma_t(sigmas[step_index - 1], alpha_s1, sigma_s1t);
        const double lambda_s1 = log(alpha_s1) - log(sigma_s1t);
        const double r0 = (lambda_s0 - lambda_s1) / h;
        for (size_t i = 0; i < D; i++) {
            double d1 = (x0[i] - prev_x0[i]) / r0;
            sample[i] = (float) (c_sample * sample[i] + c_x0 * (x0[i] + 0.5 * d1) + c_noise * noise[i]);
        }
    }

    prev_x0 = std::move(x0);
    if (lower_order_nums < 2) lower_order_nums++;
    step_index++;
}

void DdimTrailing::set_timesteps(int n) {
    const int    N = 1000;
    const double beta_start = 0.00085, beta_end = 0.012;

    ac.resize(N);
    double cum = 1.0;
    for (int i = 0; i < N; i++) {
        double b = sqrt(beta_start) + (sqrt(beta_end) - sqrt(beta_start)) * i / (N - 1);
        cum *= 1.0 - b * b;
        ac[i] = cum;
    }
    // rescale_betas_zero_snr: shift/scale sqrt(ac) so the terminal SNR is zero
    const double s0 = sqrt(ac[0]), sT = sqrt(ac[N - 1]);
    for (int i = 0; i < N; i++) {
        double s = (sqrt(ac[i]) - sT) * s0 / (s0 - sT);
        ac[i] = s * s;
    }
    final_alpha_cumprod = ac[0];                 // set_alpha_to_one = false

    n_steps = n;
    timesteps.resize(n);
    const double ratio = (double) N / n;         // trailing
    for (int k = 0; k < n; k++) {
        timesteps[k] = (int) llround(N - k * ratio) - 1;
    }
}

void DdimTrailing::step(std::vector<float> & sample, const std::vector<float> & v, int step) {
    const int t = timesteps[step];
    const int prev_t = t - 1000 / n_steps;
    const double a_t = ac[t];
    const double a_prev = prev_t >= 0 ? ac[prev_t] : final_alpha_cumprod;
    const double sq_at = sqrt(a_t), sq_bt = sqrt(1.0 - a_t);
    const double sq_ap = sqrt(a_prev), sq_bp = sqrt(1.0 - a_prev);

    for (size_t i = 0; i < sample.size(); i++) {
        // v_prediction: x0 = sqrt(a)*x - sqrt(1-a)*v; eps = sqrt(a)*v + sqrt(1-a)*x
        double x0  = sq_at * sample[i] - sq_bt * v[i];
        double eps = sq_at * v[i] + sq_bt * sample[i];
        sample[i] = (float) (sq_ap * x0 + sq_bp * eps);
    }
}
