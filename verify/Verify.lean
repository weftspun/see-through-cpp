import PlausibleWitnessDag

/-! # see-through graph quality gates

Witness search over the pipeline DAG's kernel-configuration space, driven by
`plausible-witness-dag`. A *witness* is a configuration (op x shape x knob)
whose candidate kernel leaves the f16-accumulation error interval of the
reference kernel on the primary device — i.e. a defect. The gate passes when
the production envelope is provably witness-free at the searched budgets.

The deterministic walk and the predicate are backed by `seethrough_c`
(`st_witness_check_flat`), which executes both kernel variants on the GPU and
returns the worst per-element interval violation (`<= 1.0` = contained,
`> 1.0` = defect witness, `< 0` = invalid case).
-/

open PlausibleWitnessDag

@[extern "st_witness_check_flat"]
opaque stWitnessCheckFlat (op w h c oc stride heads tq tk batch knobs : UInt32)
    (seed : UInt64) : Float

structure Case where
  name  : String
  op    : UInt32   -- 0 conv2d, 1 attn
  w     : UInt32 := 0
  h     : UInt32 := 0
  c     : UInt32 := 0
  oc    : UInt32 := 0
  stride : UInt32 := 1
  heads : UInt32 := 0
  tq    : UInt32 := 0
  tk    : UInt32 := 0
  batch : UInt32 := 1
  knobs : UInt32 := 0   -- bit0 direct, bit1 rowchunk, bit2 flash
  deriving Repr, Inhabited

def Case.check (cs : Case) (seed : UInt64) : Float :=
  stWitnessCheckFlat cs.op cs.w cs.h cs.c cs.oc cs.stride cs.heads cs.tq cs.tk
    cs.batch cs.knobs seed

def conv (name : String) (w h c oc stride knobs : UInt32) : Case :=
  { name, op := 0, w, h, c, oc, stride, knobs }

def attn (name : String) (heads tq tk batch knobs : UInt32) : Case :=
  { name, op := 1, heads, tq, tk, batch, knobs }

/-- Production envelope of the 1280px pipeline: every kernel-variant
configuration the CLI can emit, at its real shapes. Bounded, deterministic,
indexable — the witness domain. -/
def productionDomain : Array Case := #[
  -- UNet direct conv at every latent level (s1 + downsamplers)
  conv "unet-direct-160-320-s1" 160 160 320 320 1 1,
  conv "unet-direct-160-320-s2" 160 160 320 320 2 1,
  conv "unet-direct-80-640-s1"  80 80 640 640 1 1,
  conv "unet-direct-80-640-s2"  80 80 640 640 2 1,
  conv "unet-direct-40-1280-s1" 40 40 1280 1280 1 1,
  -- decode row-chunked im2col at pixel levels
  conv "decode-rowchunk-1280-64"  1280 1280 4 4 1 2,
  conv "decode-rowchunk-640-128"  640 640 8 8 1 2,
  conv "decode-rowchunk-320-256"  320 320 16 16 1 2,
  -- flash attention at every spatial token count / frame batch
  attn "flash-t1600-b13"  10 1600 1600 13 4,
  -- b13 at T=6400 needs an intractable naive reference (6400x6400x10x13 f32
  -- kq = 21GB in one allocation, over Vulkan's single-alloc limit). Attention
  -- is independent per (head,batch) slice, so large-T and large-B are
  -- covered separately below (flash-t6400-b1, flash-temporal) instead of in
  -- their infeasible combination.
  attn "flash-t6400-b2"   10 6400 6400 2 4,
  attn "flash-t6400-b1"   10 6400 6400 1 4,
  -- b13 at T=4096,H=5 is the same shape of problem as t6400-b13 above: the
  -- naive-reference kq buffer (4096*4096*5*13*4 = 4.36GB) exceeds Vulkan's
  -- maxStorageBufferRange (4294967295B on this device) and silently
  -- corrupts instead of erroring — bisected down to b1/b12 (clean) vs b13
  -- (spurious x128.8 "defect", confirmed a harness artifact, not a real
  -- flash-attention divergence: production never runs the naive path).
  attn "flash-t4096-b12"  5 4096 4096 12 4,
  attn "flash-t4096-b1"   5 4096 4096 1 4,
  attn "flash-cross-77"   10 6400 77 13 4,   -- cross-attn vs 77-token text
  attn "flash-temporal"   10 13 13 6400 4    -- cross-frame: tiny T, huge batch
]

def seed : UInt64 := 42

def isWitness (dom : Array Case) (i : Nat) : Bool :=
  match dom[i]? with
  | some cs =>
      let v := cs.check seed
      v > 1.0 || v < 0.0
  | none => false

def main : IO UInt32 := do
  IO.println s!"device-primary witness search over {productionDomain.size} production configurations"

  -- Precompute the deterministic walk once (each check runs two GPU graphs).
  let verdicts := productionDomain.map (fun cs => (cs, cs.check seed))
  for (cs, v) in verdicts do
    IO.println s!"  {cs.name}: interval x{v}"

  let checks : Array Bool := verdicts.map (fun (_, v) => v > 1.0 || v < 0.0)
  let firstWitness := checks.findIdx? id

  let (found, lvl, trace) ← resolve (α := Bool) "production-envelope-defect"
    (fun _ i => checks.getD i false)
    (fun _steps =>
      { value := firstWitness.isSome
        found := firstWitness.isSome
        witnessIdx := firstWitness.getD 0
        budgetHit := false })
  IO.println s!"resolve: level {lvl}, outcome {repr trace.outcome}"

  let mut fail : UInt32 := 0
  if found then
    let idx := firstWitness.getD 0
    IO.println s!"DEFECT WITNESS: {(productionDomain[idx]!).name}"
    fail := 1

  IO.println (if fail == 0 then "VERIFY PASS" else "VERIFY FAIL")
  pure fail
