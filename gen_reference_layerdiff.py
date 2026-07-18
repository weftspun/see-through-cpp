#!/usr/bin/env python3
"""M7 reference: the apply_layerdiff denoising loop end-to-end at 512px / 2
steps / group 0 (13 body tags), via the real upstream pipeline on CPU f32.

Determinism patches (both recorded for injection into the C++ test):
  - DiagonalGaussianDistribution.sample -> mean (the port's documented choice;
    VAE posterior std is ~1e-5 so this is far below the gate)
  - randn_tensor in the pipeline + scheduler modules -> seeded generator,
    every draw recorded (init noise + per-step SDE noise)
The pipeline is built with trans_vae=None so it returns final latents; the
TransparentVAE decode chain is already gated by test_trans_vae_full."""
import os
import sys

_ROOT = os.environ.get("SEETHROUGH_DIR", r"C:\Users\ernes\Desktop\see-through")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "common"))

import numpy as np
import torch

import diffusers.models.autoencoders.vae as vae_mod
import diffusers.schedulers.scheduling_dpmsolver_multistep as dpm_mod
from modules.layerdiffuse import diffusers_kdiffusion_sdxl as sdxl_mod
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel

REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
RES = 512
STEPS = 2
BODY_TAGS = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear',
             'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']

recorded = []


def main():
    vae_mod.DiagonalGaussianDistribution.sample = lambda self, generator=None: self.mean

    rng = torch.Generator().manual_seed(123)
    orig_randn = dpm_mod.randn_tensor

    def rec_randn(shape, generator=None, device=None, dtype=None):
        t = orig_randn(shape, generator=rng, device=torch.device("cpu"), dtype=torch.float32)
        recorded.append(t.clone())
        return t

    dpm_mod.randn_tensor = rec_randn
    sdxl_mod.randn_tensor = rec_randn

    unet = UNetFrameConditionModel.from_pretrained(REPO, subfolder="unet",
                                                   torch_dtype=torch.float32)
    pipe = sdxl_mod.KDiffusionStableDiffusionXLPipeline.from_pretrained(
        REPO, unet=unet, trans_vae=None, scheduler=None, torch_dtype=torch.float32)
    pipe.cache_tag_embeds(unload_textencoders=False)

    # deterministic synthetic page: gradient + disc, alpha 255
    yy, xx = np.mgrid[0:RES, 0:RES].astype(np.float32) / RES
    page = np.stack([xx * 255, yy * 255, (1 - xx) * 200 + 30], axis=-1)
    disc = (xx - 0.5) ** 2 + (yy - 0.55) ** 2 < 0.09
    page[disc] = [230, 60, 80]
    fullpage = np.concatenate([page, np.full((RES, RES, 1), 255.0)], axis=-1).astype(np.uint8)

    embeds, pooled = pipe.encode_cropped_prompt_77tokens_cached(BODY_TAGS)

    # c_concat exactly as the fullpage branch computes it (alpha=255 page,
    # .sample patched to mean); passed directly since trans_vae is None
    rgb01_pre = torch.from_numpy(page.astype(np.float32) / 255.0).movedim(-1, 0)[None]
    with torch.no_grad():
        c_concat_in = pipe.vae.encode(rgb01_pre * 2 - 1).latent_dist.mean \
            * pipe.vae.config.scaling_factor

    with torch.inference_mode():
        latents = pipe(strength=1.0, num_inference_steps=STEPS, batch_size=1,
                       generator=rng, guidance_scale=1.0, prompt=BODY_TAGS,
                       negative_prompt='', c_concat=c_concat_in, group_index=0)
    print("latents:", tuple(latents.shape), "mean", float(latents.mean()),
          "std", float(latents.std()))
    print("noise draws:", [tuple(t.shape) for t in recorded])

    vae_feed = rgb01_pre * 2 - 1
    arrays = [vae_feed, c_concat_in, embeds, pooled] + recorded + [latents.squeeze(0)]
    with open("reference_layerdiff.bin", "wb") as f:
        for arr in arrays:
            a = arr.float().numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_layerdiff.bin,", len(arrays), "records")


if __name__ == "__main__":
    main()
