import Lake
open Lake DSL

package verify where
  -- Witness-search quality gates over the see-through graph configuration
  -- space, driven by plausible-witness-dag against the seethrough_c FFI.

require «plausible-witness-dag» from git
  "https://github.com/fire/plausible-witness-dag" @ "main"

@[default_target] lean_exe verify where
  root := `Verify
  moreLinkArgs := #[
    "-L../build-vulkan",
    "-L../build",
    "-lseethrough_c"
  ]
