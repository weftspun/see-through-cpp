#!/usr/bin/env python3
"""Generate CLIP text-encoder references for validating the ggml port, exactly
mirroring how See-Through calls each encoder:
  layerdiff-te1/te2: encode_cropped_prompt_77tokens — 77-token max_length
    padding, hidden_states[-2] (penultimate, no final LN); te2 additionally the
    projected EOS pooled output (text_embeds).
  marigold-te: encode_empty_text — "" with do_not_pad ([BOS,EOS]), final-LN
    last_hidden_state.
Records: ids (as f32), penultimate/final [n,d], (te2: pooled [1280])."""
import sys

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

PROMPT = "solo, 1girl, blue hair, cat ears, school uniform"

COMPONENTS = {
    "layerdiff-te1": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "text_encoder", "tokenizer", CLIPTextModel),
    "layerdiff-te2": ("layerdifforg/seethroughv0.0.2_layerdiff3d", "text_encoder_2", "tokenizer_2", CLIPTextModelWithProjection),
    "marigold-te": ("24yearsold/seethroughv0.0.1_marigold", "text_encoder", "tokenizer", CLIPTextModel),
}


def main():
    comp = sys.argv[1] if len(sys.argv) > 1 else "layerdiff-te1"
    repo, sub, tok_sub, cls = COMPONENTS[comp]
    tok = CLIPTokenizer.from_pretrained(repo, subfolder=tok_sub)
    te = cls.from_pretrained(repo, subfolder=sub).eval().float()
    print("pad token:", tok.pad_token, tok.pad_token_id)

    arrays = []
    with torch.no_grad():
        if comp == "marigold-te":
            ids = tok("", padding="do_not_pad", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids
            final = te(ids)[0][0]                      # last_hidden_state [2, 1024]
            arrays = [ids[0].float(), final]
        else:
            ids = tok(PROMPT, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids
            out = te(ids, output_hidden_states=True)
            penult = out.hidden_states[-2][0]          # [77, d]
            arrays = [ids[0].float(), penult]
            if comp == "layerdiff-te2":
                arrays.append(out[0][0])               # projected pooled [1280]
    for a in arrays:
        print("shape", tuple(a.shape), "mean", float(a.float().mean()))

    out_path = f"reference_{comp}.bin"
    with open(out_path, "wb") as f:
        for arr in arrays:
            a = arr.numpy().astype("<f4")
            f.write(np.int32(a.ndim).tobytes())
            f.write(np.array(a.shape, dtype="<i8").tobytes())
            f.write(a.tobytes())
    print("wrote", out_path)


if __name__ == "__main__":
    main()
