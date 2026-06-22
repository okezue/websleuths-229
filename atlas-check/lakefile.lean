import Lake
open Lake DSL

package «atlas-check» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.28.0"

lean_lib AtlasCheck where
  roots := #[`AtlasCheck.AdaptiveGame]
