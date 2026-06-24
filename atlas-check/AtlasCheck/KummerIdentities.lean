import Mathlib

namespace Atlas
namespace X25519
namespace TotalKummer

universe u

variable {K : Type u} [Field K]

private def montRhs (A x : K) : K := x ^ 3 + A * x ^ 2 + x

/-- The affine differential-addition identity for a Montgomery curve.  It is stated without any
choice of square root and is therefore valid for points on the curve or its quadratic twist after
base change. -/
theorem montgomery_differential_identity
    (A x₁ y₁ x₂ y₂ : K)
    (hx : x₁ ≠ x₂)
    (h₁ : y₁ ^ 2 = montRhs A x₁)
    (h₂ : y₂ ^ 2 = montRhs A x₂) :
    let xplus := ((y₁ - y₂) / (x₁ - x₂)) ^ 2 - A - x₁ - x₂
    let xminus := ((y₁ + y₂) / (x₁ - x₂)) ^ 2 - A - x₁ - x₂
    (x₁ * x₂ - 1) ^ 2 = xplus * xminus * (x₁ - x₂) ^ 2 := by
  dsimp
  have hden : x₁ - x₂ ≠ 0 := sub_ne_zero.mpr hx
  field_simp [hden]
  polyrith

/-- The Montgomery doubling identity in affine coordinates. -/
theorem montgomery_doubling_identity
    (A x y : K)
    (h2 : (2 : K) ≠ 0)
    (hy : y ≠ 0)
    (hc : y ^ 2 = montRhs A x) :
    let xdbl := ((3 * x ^ 2 + 2 * A * x + 1) / (2 * y)) ^ 2 - A - 2 * x
    (x ^ 2 - 1) ^ 2 = xdbl * (4 * y ^ 2) := by
  dsimp
  have hden : (2 * y : K) ≠ 0 := mul_ne_zero h2 hy
  field_simp [hden]
  polyrith

/-- The xADD coordinate polynomials on normalized inputs reduce to the traditional affine
numerator and denominator. -/
theorem normalized_xadd_polynomials (x₁ x₂ xd : K) :
    let da := (x₂ - 1) * (x₁ + 1)
    let cb := (x₂ + 1) * (x₁ - 1)
    (1 * (da + cb) ^ 2, xd * (da - cb) ^ 2) =
      (4 * (x₁ * x₂ - 1) ^ 2, 4 * xd * (x₁ - x₂) ^ 2) := by
  dsimp
  apply Prod.ext <;> ring

end TotalKummer
end X25519
end Atlas
