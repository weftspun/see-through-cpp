import Lake
open Lake DSL

package verify where
  -- Witness-search quality gates over the see-through graph configuration
  -- space, driven by plausible-witness-dag against the seethrough_c FFI.

require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

-- import library for the seethrough_c DLL; override with
--   lake build -Kseethrough_c_lib=<path>
def seethroughCLib :=
  #[((get_config? seethrough_c_lib).getD
      (__dir__ / ".." / "build-vulkan" / "seethrough_c.lib").toString)]

lean_lib Case

@[default_target] lean_exe kernel_gate where
  root := `KernelGate
  moreLinkArgs := seethroughCLib

lean_exe quant_design where
  root := `QuantDesign
  moreLinkArgs := seethroughCLib
