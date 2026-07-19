// VAE-family graphs: diffusers AutoencoderKL decoder, the LayerDiffuse
// UNet1024 alpha head, and the full TransparentVAE decode chain.
#pragma once

#include "ops.h"

// AutoencoderKL encode: image (h,w,3,1) in [-1,1] -> latent mean (h/8,w/8,4,1),
// unscaled (deterministic: the mean half of the posterior moments, not
// .sample()). Weight names: "encoder.*", "quant_conv". `taps` (optional)
// receives conv_in, each down_block output, and mid_block output -- for
// bisecting a divergence from upstream stage by stage.
ggml_tensor * vae_encode(Model & m, ggml_tensor * x, std::vector<ggml_tensor *> * taps = nullptr);

// AutoencoderKL tiled encode, matching diffusers' AutoencoderKL.tiled_encode
// exactly (tile_sample_min_size=512, tile_latent_min_size=64,
// tile_overlap_factor=0.25): splits the pixel input into overlapping 512x512
// tiles, encodes each with the plain vae_encode above, linearly blends the
// overlap region, crops to the non-overlapping core, and concatenates.
// This VAE's own trained config is sample_size=512 -- encoding far beyond
// that untiled (e.g. 1280x1280) pushes some resnets into extreme,
// near-cancelling activation magnitudes (confirmed present in upstream's
// own untiled reference too, docs/ggml-upstream-issues.md #4) that then
// amplify ordinary GPU-vs-CPU float rounding into a visible defect. Tiling
// keeps every tile within the model's actual trained operating range.
// No-op passthrough to vae_encode when the input already fits in one tile.
ggml_tensor * vae_encode_tiled(Model & m, ggml_tensor * x);

// AutoencoderKL decode: z (zh,zw,4,1) -> image (8*zh,8*zw,3,1), range [-1,1].
// Weight names: "post_quant_conv", "decoder.*". Unscaled latent expected.
ggml_tensor * vae_decode(Model & m, ggml_tensor * z);

// AutoencoderKL tiled decode, matching diffusers' AutoencoderKL.tiled_decode:
// splits the latent into overlapping 64x64 (tile_latent_min_size) tiles,
// decodes each with the plain vae_decode above (each producing a 512x512
// pixel tile), blends the pixel-space overlap, crops, and concatenates. See
// vae_encode_tiled's comment -- the same out-of-trained-scale reasoning
// applies to decode. No-op passthrough when the latent already fits in one
// tile.
ggml_tensor * vae_decode_tiled(Model & m, ggml_tensor * z);

// UNet1024 (TransparentVAE decoder head): x (r,r,3,1) pixels in [0,1] +
// latent (r/8,r/8,4,1) -> RGBA logits (r,r,4,1). Weights under "decoder.model.".
ggml_tensor * unet1024(Model & m, ggml_tensor * x, ggml_tensor * latent);

// Full TransparentVAEDecoder.forward single pass: VAE decode -> *0.5+0.5 ->
// UNet1024 -> clip(0,1). Needs both layerdiff-vae and trans-vae weights loaded.
ggml_tensor * trans_vae_decode(Model & m, ggml_tensor * z);
