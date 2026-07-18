import Lake
open Lake DSL

package verify where
  -- Witness-search quality gates over the see-through graph configuration
  -- space, driven by plausible-witness-dag against the seethrough_c FFI.

require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

@[default_target] lean_exe verify where
  root := `Verify
  -- import library for the seethrough_c DLL; override with
  --   lake build -Kseethrough_c_lib=<path>
  moreLinkArgs :=
    #[((get_config? seethrough_c_lib).getD
        (__dir__ / ".." / "build-vulkan" / "seethrough_c.lib").toString)]
