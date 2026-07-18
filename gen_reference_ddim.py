#!/usr/bin/env python3
"""M8 scheduler reference: Marigold's DDIM (v_prediction, zero-SNR rescale,
trailing, 4 steps, eta 0) trajectory with an analytic fake model
(v = 0.1*x + e)."""
import numpy as np
import torch
from diffusers import DDIMScheduler

STEPS = 4
D = 64


def main():
    sch = DDIMScheduler.from_pretrained("24yearsold/seethroughv0.0.1_marigold",
                                        subfolder="scheduler")
    sch.set_timesteps(STEPS)
    print("timesteps:", sch.timesteps.tolist())

    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, D, generator=g)
    init = x.clone()
    e = torch.randn(1, D, generator=g)

    traj = []
    for t in sch.timesteps:
        v = 0.1 * x + e
        x = sch.step(v, t, x, return_dict=False)[0]
        traj.append(x.clone())
    traj = torch.cat(traj)
    print("final mean", float(x.mean()), "std", float(x.std()))

    with open("reference_ddim.bin", "wb") as f:
        for arr in (init, e, traj):
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_ddim.bin")


if __name__ == "__main__":
    main()
