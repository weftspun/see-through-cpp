# Remaining work to close out the see-through.cpp port

* Status: accepted
* Date: 2026-07-18

## Context and Problem Statement

M1–M13 of the port plan are implemented; every component has a numeric gate
against the upstream PyTorch pipeline (see README status table). The CLI runs
end-to-end on CPU and on Vulkan (RTX 4090) with a low-VRAM UNet configuration
(flash attention, per-frame spatial transformer chunking, f16 skip
connections, direct convolutions). What remains is closing the last
integration gaps: the real-sample run has so far produced empty layers under
now-fixed misconfigurations, and several structural requests
(property testing, psd_sdk, repo layout) are partially done.

## Decision Outcome

Track the close-out as the checklist below; check items off as they land on
`main`.

### Validation

- [x] Real-sample E2E on Vulkan (1280px/30 steps) produces non-empty,
      visually sensible layers (composite matches upstream reference;
      2026-07-18, 29 layers)
- [ ] Layer-quality polish: thin border slivers from L/R splits at the pad
      boundary (e.g. ears-r 5px wide at x=1275), faint head-pass alphas —
      tune alpha floor / border cleanup against reference_output.psd in `assets/sample_out.psd`
      (open in psd-tools + Krita)
- [x] --debug-dir stage stats added; bisection found the decode zeros
      (Vulkan direct conv at large sizes) -> row-chunked im2col fix
- [ ] Upstream parity: run `inference_psd.py --save_to_psd` (same seed/input)
      and compare per-layer alpha-mask IoU (>0.98 where tags match) and depth
      ordering
- [ ] Investigate `load` vs `load_backend` tensor-count difference
      (249 vs 248 on marigold-vae; likely benign duplicate name)

### User-requested hardening

- [x] Vendor ggml as a subtree (was a submodule)
- [x] Vendor psd_sdk at `third_party/psd_sdk`
- [x] Vendor rapidcheck at `third_party/rapidcheck`; property tests for
      PackBits, schedulers, resize/pad, connected components, kmeans2,
      tokenizer (first catch: PackBits 2-run folding broke the expansion
      bound)
- [ ] Reimplement `write_psd` on psd_sdk behind the same API; keep the
      `test_psd` + `check_psd.py` byte-exact round-trip gate; delete the
      minimal writer once green
- [ ] Property tests linked in the Vulkan build tree (currently CPU tree
      only; the Vulkan tree needs its rapidcheck reconfigure once no
      process locks the ggml DLLs)

### Repo layout

- [x] `gen_reference/` for the reference generators (user's layout)
- [ ] `scripts/` or root cleanup for `convert_*.py` + `check_psd.py`
- [ ] `refs/` for `reference_*.bin` data (gitignored), generators and test
      invocations updated
- [ ] `tests/` for `test_*.cpp` + `test_common.h` (CMake foreach update)
- [ ] `models/` for `*.gguf` weights; CLI `-m` default becomes `models`
- [ ] `git mv ggml third_party/ggml` for vendoring consistency (forces one
      full rebuild of both build trees)

### Documentation / upstream

- [ ] README: low-VRAM knob notes (direct conv is UNet/decoder-only — it is
      numerically wrong for the VAE-encoder stride-2/pad-0 path), Vulkan
      usage incl. `build-vulkan/bin` on PATH, subtree vendoring policy
- [ ] Minimal repro + issue text for the ggml gallocr surprise: input-tensor
      buffers are recycled after their last in-graph read, so all inputs
      must be re-set before every compute of a reused allocated graph
- [ ] Upstream triage: ggml findings from the graph property tests —
      (a) CPU flash_attn_ext diverges ~0.3 from naive attention for
      n_head >= 2 (single-head agrees to f16 precision; Vulkan agrees);
      (b) Vulkan ggml_conv_2d_direct silently returns zeros at very large
      spatial sizes (>= ~1280^2; correct at <= 512^2)
- [ ] Post-MVP (original plan): C ABI + server, trellis2cpp-style

## Consequences

The port is feature-complete once the Validation section is green; the
hardening items reduce maintenance risk but do not block use.
