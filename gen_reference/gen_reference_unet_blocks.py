#!/usr/bin/env python3
"""M4 references: UNetFrameCondition building blocks, run through the upstream
modules with real layerdiff-unet weights (loaded selectively from the
safetensors — no full-model instantiation).

Blocks covered, mirroring how the full model composes them:
  1. ResnetBlock2D with temb           (down_blocks.1.resnets.0, 320->640)
  2. Transformer3DModel, 2 layers      (down_blocks.1.attentions.0: stock +
     temporal blocks at stride 1, the [B*F,S,C]<->[B*S,F,C] permute)
  3. the conditioning path: time_proj/time_embedding + GroupEmbedding (group 0)
     on both the pooled (group_embeds) and sequence (group_embeds2) paths +
     add_time_proj/add_embedding

Needs SEETHROUGH_DIR (upstream repo) on sys.path for transformer3d/layerdiff3d.
"""
import os
import sys

_ROOT = os.environ.get("SEETHROUGH_DIR", r"C:\Users\ernes\Desktop\see-through")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "common"))

import os
os.makedirs('gen_reference', exist_ok=True)
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from common.modules.layerdiffuse.transformer3d import Transformer3DModel
from common.modules.layerdiffuse.layerdiff3d import GroupEmbedding

REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
F = 13


def load_subset(path, prefix):
    sd = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k).float()
    assert sd, f"no keys under {prefix}"
    return sd


def main():
    path = hf_hub_download(REPO, "unet/diffusion_pytorch_model.safetensors")
    g = torch.Generator().manual_seed(7)
    arrays = []

    # 1. resnet + temb
    resnet = ResnetBlock2D(in_channels=320, out_channels=640, temb_channels=1280,
                           eps=1e-5, groups=32)
    resnet.load_state_dict(load_subset(path, "down_blocks.1.resnets.0."))
    resnet.eval()
    x = torch.randn(F, 320, 16, 16, generator=g)
    temb = torch.randn(F, 1280, generator=g)
    with torch.no_grad():
        y = resnet(x, temb)
    arrays += [x, temb, y]
    print("resnet:", tuple(y.shape), float(y.mean()))

    # 2. Transformer3D (2 layers -> temporal stride 1)
    t3d = Transformer3DModel(num_attention_heads=10, attention_head_dim=64,
                             in_channels=640, num_layers=2, cross_attention_dim=2048,
                             use_linear_projection=True)
    t3d.load_state_dict(load_subset(path, "down_blocks.1.attentions.0."))
    t3d.eval()
    tx = torch.randn(F, 640, 16, 16, generator=g)
    ehs = torch.randn(F, 77, 2048, generator=g)
    with torch.no_grad():
        ty = t3d(tx, encoder_hidden_states=ehs, num_frames=F, return_dict=False)[0]
    arrays += [tx, ehs, ty]
    print("transformer3d:", tuple(ty.shape), float(ty.mean()))

    # 3. conditioning path, group 0 (13 body tags)
    time_proj = Timesteps(320, flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedding = TimestepEmbedding(320, 1280)
    time_embedding.load_state_dict(load_subset(path, "time_embedding."))
    add_time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
    add_embedding = TimestepEmbedding(2816, 1280)
    add_embedding.load_state_dict(load_subset(path, "add_embedding."))
    ge = GroupEmbedding(13, 1280)
    ge.load_state_dict(load_subset(path, "group_embeds.0."))
    ge2 = GroupEmbedding(13, 2048)
    ge2.load_state_dict(load_subset(path, "group_embeds2.0."))
    for mod in (time_embedding, add_embedding, ge, ge2):
        mod.eval()

    text_embeds = torch.randn(F, 1280, generator=g)
    time_ids = torch.tensor([1280.0, 1280.0, 0.0, 0.0, 1280.0, 1280.0]).repeat(F, 1)
    with torch.no_grad():
        t_emb = time_proj(torch.full((F,), 999.0))
        emb = time_embedding(t_emb)
        aug_text = text_embeds + ge(text_embeds)
        time_embeds = add_time_proj(time_ids.flatten()).reshape(F, -1)
        emb = emb + add_embedding(torch.cat([aug_text, time_embeds], dim=-1))
        ehs2 = ehs + ge2(ehs)
    arrays += [text_embeds, emb, ehs2]
    print("cond: emb", tuple(emb.shape), float(emb.mean()), "ehs2", float(ehs2.mean()))

    with open("gen_reference/reference_unet_blocks.bin", "wb") as f:
        for arr in arrays:
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote reference_unet_blocks.bin,", len(arrays), "records")


if __name__ == "__main__":
    main()
