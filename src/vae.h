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

// AutoencoderKL decode: z (zh,zw,4,1) -> image (8*zh,8*zw,3,1), range [-1,1].
// Weight names: "post_quant_conv", "decoder.*". Unscaled latent expected.
ggml_tensor * vae_decode(Model & m, ggml_tensor * z);

// UNet1024 (TransparentVAE decoder head): x (r,r,3,1) pixels in [0,1] +
// latent (r/8,r/8,4,1) -> RGBA logits (r,r,4,1). Weights under "decoder.model.".
ggml_tensor * unet1024(Model & m, ggml_tensor * x, ggml_tensor * latent);

// Full TransparentVAEDecoder.forward single pass: VAE decode -> *0.5+0.5 ->
// UNet1024 -> clip(0,1). Needs both layerdiff-vae and trans-vae weights loaded.
ggml_tensor * trans_vae_decode(Model & m, ggml_tensor * z);
