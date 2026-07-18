#!/usr/bin/env python3
"""M6 reference: DPM++ 2M SDE trajectory (30 steps) with an analytic fake unet
(eps = 0.1*x + e for a fixed seeded direction e) and per-step injected noise,
using the exact See-Through scheduler construction (juggernautXL base config +
sde-dpmsolver++ + final_sigmas_type zero)."""
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler

STEPS = 30
D = 64


def main():
    sch = DPMSolverMultistepScheduler.from_pretrained(
        "frankjoshua/juggernautXL_version6Rundiffusion", subfolder="scheduler",
        algorithm_type="sde-dpmsolver++", final_sigmas_type="zero")
    print({k: sch.config[k] for k in ("beta_start", "beta_end", "beta_schedule",
                                      "num_train_timesteps", "timestep_spacing",
                                      "steps_offset", "prediction_type", "solver_order",
                                      "solver_type", "lower_order_final", "euler_at_final",
                                      "use_karras_sigmas")})
    sch.set_timesteps(STEPS)
    print("timesteps:", sch.timesteps.tolist())

    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, D, generator=g)
    init = x.clone()
    e = torch.randn(1, D, generator=g)
    noises = torch.randn(STEPS, 1, D, generator=g)

    traj = []
    for i, t in enumerate(sch.timesteps):
        eps = 0.1 * x + e
        x = sch.step(eps, t, x, variance_noise=noises[i], return_dict=False)[0]
        traj.append(x.clone())
    traj = torch.cat(traj)                      # (STEPS, D)
    print("final mean", float(x.mean()), "std", float(x.std()))

    with open("reference_scheduler.bin", "wb") as f:
        for arr in (init, e, noises, traj):
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_scheduler.bin")


if __name__ == "__main__":
    main()
