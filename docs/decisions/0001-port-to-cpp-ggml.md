# 1. Port See-Through inference to C++/ggml

Status: accepted (2026-07-17)

## Context and Problem Statement

See-Through's Python inference stack needs PyTorch, diffusers, and (for the
annotators) detectron2/mmcv — the latter effectively do not build natively on
Windows, our primary machine. The pipeline is a candidate permanent member of
the weftspun inverse-rendering loop, which already runs one ggml port
(trellis2cpp) and shares a 24 GB GPU between a diffusion backend and 3D
lifting. How should See-Through inference be deployed long-term?

## Considered Options

1. Wrap the Python pipeline in docker/WSL2 as an MCP server.
2. Port inference to C++/ggml (trellis2cpp/sam3.cpp lineage).
3. Use only the hosted HF Space (1–2 runs/day).

## Decision Outcome

Option 2, with option 1 as the acknowledged interim (the Python wrap gets
the correctness benefit into the loop while the port matures). The user
chose to go directly to the port after the style-register validation passed
(21 clean layers on our semi-real renders, dedicated tail/ear layers).

Consequences: no PyTorch at runtime, native Windows, ggml-style VRAM
control so the model cooperates with Easy Diffusion and trellis2cpp on one
GPU, single-binary deployment matching the org's server pattern — at the
cost of implementing an SDXL-class UNet, two VAEs, and a sampler in ggml.
The hosted Space remains useful for spot-checking upstream behavior.
