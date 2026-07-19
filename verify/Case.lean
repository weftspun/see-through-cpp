import PlausibleWitnessDag

/-! # Shared witness-case plumbing

Common to every see-through Lean gate: the `seethrough_c` FFI binding, the
`Case` shape (op x shape x knob), and the constructors each gate's domain is
built from. A *witness* is a configuration whose candidate kernel leaves the
error interval of the reference kernel on the primary device — i.e. a
defect (kernel gates) or an out-of-tolerance design point (design searches).

The deterministic walk and the predicate are backed by `seethrough_c`
(`st_witness_check_flat`), which executes both kernel variants on the GPU and
returns the worst per-element interval violation (`<= 1.0` = contained,
`> 1.0` = defect/out-of-tolerance witness, `< 0` = invalid case).
-/

open PlausibleWitnessDag

@[extern "st_witness_check_flat"]
opaque stWitnessCheckFlat (op w h c oc stride heads tq tk batch knobs : UInt32)
    (seed : UInt64) : Float

structure Case where
  name  : String
  op    : UInt32   -- 0 conv2d, 1 attn, 2 linear_quant
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

def conv (name : String) (w h c oc stride knobs : UInt32) (batch : UInt32 := 1) : Case :=
  { name, op := 0, w, h, c, oc, stride, knobs, batch }

def attn (name : String) (heads tq tk batch knobs : UInt32) : Case :=
  { name, op := 1, heads, tq, tk, batch, knobs }

def linq (name : String) (c oc tq : UInt32) : Case :=
  { name, op := 2, c, oc, tq }

def seed : UInt64 := 42

/-- Run every case in `dom`, print each interval, resolve the first witness
(if any) via plausible-witness-dag, and report. Shared report shape between
the kernel gate and the design searches — only the framing text differs. -/
def searchAndReport (dom : Array Case) (label : String) : IO Bool := do
  IO.println s!"{label} over {dom.size} configurations"
  let verdicts := dom.map (fun cs => (cs, cs.check seed))
  for (cs, v) in verdicts do
    IO.println s!"  {cs.name}: interval x{v}"
  let checks : Array Bool := verdicts.map (fun (_, v) => v > 1.0 || v < 0.0)
  let firstWitness := checks.findIdx? id
  let (found, lvl, trace) ← resolve (α := Bool) label
    (fun _ i => checks.getD i false)
    (fun _steps =>
      { value := firstWitness.isSome
        found := firstWitness.isSome
        witnessIdx := firstWitness.getD 0
        budgetHit := false })
  IO.println s!"resolve: level {lvl}, outcome {repr trace.outcome}"
  if found then
    let idx := firstWitness.getD 0
    IO.println s!"WITNESS: {(dom[idx]!).name}"
  pure found
