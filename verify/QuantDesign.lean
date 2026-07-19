import Case

/-! # Quantization design search

Not a fixed production gate — a design-space search: is Q4_0 quantization of
`layerdiff-unet`'s nn.Linear weights (self/cross-attn projections, GEGLU FF)
within an acceptable error envelope at every real channel width the UNet
uses? Shapes taken directly from the SDXL CrossFrame UNet architecture:
C in {320,640,1280}, cross-attn text context dim 2048, GEGLU ff.net.0
projects to 8*C (gate+value), ff.net.2 projects 4*C back to C.

A "witness" here means a shape whose Q4_0 error exceeds the 15% design
tolerance encoded in `st_witness_check`'s `linear_quant` case (atol=1e-3,
rtol=0.15) — informing which tensors are safe to quantize, not asserting a
hard invariant the way the kernel gate does.
-/

open PlausibleWitnessDag

def quantDomain : Array Case := #[
  linq "quant-attn-c320"  320  320  64,
  linq "quant-attn-c640"  640  640  64,
  linq "quant-attn-c1280" 1280 1280 64,
  linq "quant-crosskv-c320"  2048 320  64,
  linq "quant-crosskv-c640"  2048 640  64,
  linq "quant-crosskv-c1280" 2048 1280 64,
  linq "quant-ff0-c320"  320  2560  64,
  linq "quant-ff0-c640"  640  5120  64,
  linq "quant-ff0-c1280" 1280 10240 64,
  linq "quant-ff2-c320"  1280 320  64,
  linq "quant-ff2-c640"  2560 640  64,
  linq "quant-ff2-c1280" 5120 1280 64
]

def main : IO UInt32 := do
  let found ← searchAndReport quantDomain "quantization-design-defect"
  if found then
    IO.println "OUT OF DESIGN TOLERANCE — exclude that shape from Q4_0, keep f16"
  else
    IO.println "Q4_0 WITHIN DESIGN TOLERANCE across all searched Linear shapes"
  pure 0   -- informational: never fails the build
