import AtlasCheck.KummerAlgebraicLift

namespace Atlas
namespace X25519
namespace TotalKummer
namespace Montgomery

open WeierstrassCurve
open WeierstrassCurve.Affine

universe u

variable {K : Type u} [Field K] [DecidableEq K] [IsAlgClosed K]
variable (A : K) [(montgomeryCurve A).IsElliptic]

abbrev RawX := K × K

def RawNonzero (r : RawX (K := K)) : Prop := r.1 ≠ 0 ∨ r.2 ≠ 0

def RawRep (r : RawX (K := K)) (P : Point A) : Prop :=
  r.1 * (pointXRep A P).Z = (pointXRep A P).X * r.2

def dblRaw (r : RawX (K := K)) : RawX (K := K) :=
  ((r.1 ^ 2 - r.2 ^ 2) ^ 2,
   4 * r.1 * r.2 * (r.1 ^ 2 + A * r.1 * r.2 + r.2 ^ 2))

@[simp] theorem rawRep_zero (X Z : K) :
    RawRep A (X, Z) (0 : Point A) ↔ Z = 0 := by
  simp [RawRep, pointXRep]

@[simp] theorem rawRep_some {x y : K}
    (h : (montgomeryAffine A).Nonsingular x y) (X Z : K) :
    RawRep A (X, Z) (.some h) ↔ X = x * Z := by
  simp [RawRep, pointXRep]

@[simp] theorem negY_montgomery (x y : K) :
    (montgomeryAffine A).negY x y = -y := by
  simp [montgomeryAffine, montgomeryCurve, WeierstrassCurve.Affine.negY]

 theorem equation_iff_montgomery (x y : K) :
    (montgomeryAffine A).Equation x y ↔
      y ^ 2 = x ^ 3 + A * x ^ 2 + x := by
  rw [WeierstrassCurve.Affine.equation_iff]
  simp [montgomeryAffine, montgomeryCurve]

 theorem slope_double (h2 : (2 : K) ≠ 0) (x y : K) (hy : y ≠ 0) :
    (montgomeryAffine A).slope x x y y =
      (3 * x ^ 2 + 2 * A * x + 1) / (2 * y) := by
  have hne : y ≠ (montgomeryAffine A).negY x y := by
    rw [negY_montgomery]
    intro h
    have h2y : (2 : K) * y = 0 := by linear_combination h
    exact hy ((mul_eq_zero.mp h2y).resolve_left h2)
  rw [WeierstrassCurve.Affine.slope_of_Y_ne rfl hne, negY_montgomery]
  simp [montgomeryAffine, montgomeryCurve]
  congr 1 <;> ring

 theorem dblRaw_homogeneous (lam X Z : K) :
    dblRaw A (lam * X, lam * Z) =
      (lam ^ 4 * (dblRaw A (X, Z)).1,
       lam ^ 4 * (dblRaw A (X, Z)).2) := by
  apply Prod.ext <;> simp [dblRaw] <;> ring

private theorem four_ne_zero (h2 : (2 : K) ≠ 0) : (4 : K) ≠ 0 := by
  have : (4 : K) = 2 * 2 := by norm_num
  rw [this]
  exact mul_ne_zero h2 h2

private theorem dbl_numerator_ne_zero_of_y_zero
    (h2 : (2 : K) ≠ 0) (hA : A ^ 2 ≠ 4)
    (x y : K) (hc : y ^ 2 = x ^ 3 + A * x ^ 2 + x)
    (hy : y = 0) : (x ^ 2 - 1) ^ 2 ≠ 0 := by
  intro hzero
  have hxsub : x ^ 2 - 1 = 0 := by
    have hm : (x ^ 2 - 1) * (x ^ 2 - 1) = 0 := by
      simpa [pow_two] using hzero
    exact (mul_self_eq_zero.mp hm)
  have hx2 : x ^ 2 = 1 := sub_eq_zero.mp hxsub
  have hxne : x ≠ 0 := by
    intro hx
    rw [hx] at hx2
    norm_num at hx2
  have hfac : x * (x ^ 2 + A * x + 1) = 0 := by
    calc
      x * (x ^ 2 + A * x + 1) = x ^ 3 + A * x ^ 2 + x := by ring
      _ = y ^ 2 := hc.symm
      _ = 0 := by simp [hy]
  have hinner : x ^ 2 + A * x + 1 = 0 :=
    (mul_eq_zero.mp hfac).resolve_left hxne
  rw [hx2] at hinner
  have hAx : A * x = -2 := by linear_combination hinner
  have hsquare := congrArg (fun z : K => z ^ 2) hAx
  have hAeq : A ^ 2 = 4 := by
    calc
      A ^ 2 = A ^ 2 * x ^ 2 := by rw [hx2, mul_one]
      _ = (A * x) ^ 2 := by ring
      _ = (-2 : K) ^ 2 := hsquare
      _ = 4 := by ring
  exact hA hAeq

 theorem dblRaw_normalized_sound
    (h2 : (2 : K) ≠ 0) (hA : A ^ 2 ≠ 4)
    (x y : K) (h : (montgomeryAffine A).Nonsingular x y) :
    RawNonzero (dblRaw A (x, 1)) ∧
      RawRep A (dblRaw A (x, 1)) (.some h + .some h) := by
  have hc : y ^ 2 = x ^ 3 + A * x ^ 2 + x :=
    (equation_iff_montgomery A x y).mp h.1
  have hform : dblRaw A (x, 1) = ((x ^ 2 - 1) ^ 2, 4 * y ^ 2) := by
    apply Prod.ext
    · simp [dblRaw]
    · simp [dblRaw]
      rw [hc]
      ring
  by_cases hy : y = 0
  · have hneg : y = (montgomeryAffine A).negY x y := by
      simp [hy, negY_montgomery]
    have hadd := WeierstrassCurve.Affine.Point.add_self_of_Y_eq hneg
    rw [hadd, hform]
    constructor
    · exact Or.inl (dbl_numerator_ne_zero_of_y_zero A h2 hA x y hc hy)
    · simp [RawRep, pointXRep]
  · have hne : y ≠ (montgomeryAffine A).negY x y := by
      rw [negY_montgomery]
      intro heq
      have h2y : (2 : K) * y = 0 := by linear_combination heq
      exact hy ((mul_eq_zero.mp h2y).resolve_left h2)
    have hadd := WeierstrassCurve.Affine.Point.add_self_of_Y_ne hne
    rw [hadd, hform]
    constructor
    · exact Or.inr (mul_ne_zero (four_ne_zero h2) (pow_ne_zero _ hy))
    · simp only [RawRep, pointXRep_some, Prod.fst, Prod.snd, mul_one]
      rw [slope_double A h2 x y hy]
      simp [montgomeryAffine, montgomeryCurve, WeierstrassCurve.Affine.addX]
      have hden : (2 * y : K) ≠ 0 := mul_ne_zero h2 hy
      field_simp [hden]
      linear_combination (-(A + 2 * x) * 4) * hc

 theorem dblRaw_sound
    (h2 : (2 : K) ≠ 0) (hA : A ^ 2 ≠ 4)
    (r : RawX (K := K)) (P : Point A)
    (hnz : RawNonzero r) (hrep : RawRep A r P) :
    RawNonzero (dblRaw A r) ∧ RawRep A (dblRaw A r) (P + P) := by
  rcases r with ⟨X, Z⟩
  cases P with
  | zero =>
      have hZ : Z = 0 := (rawRep_zero A X Z).mp hrep
      have hX : X ≠ 0 := hnz.resolve_right (by simpa [hZ])
      subst Z
      constructor
      · left
        simp [dblRaw, hX]
      · simp [dblRaw, RawRep, pointXRep]
  | @some x y h =>
      have hXeq : X = x * Z := (rawRep_some A h X Z).mp hrep
      have hZ : Z ≠ 0 := by
        intro hz
        apply hnz.elim
        · intro hX
          apply hX
          simp [hXeq, hz]
        · exact fun hZ => hZ hz
      have hnorm := dblRaw_normalized_sound A h2 hA x y h
      have hraw : dblRaw A (X, Z) =
          (Z ^ 4 * (dblRaw A (x, 1)).1,
           Z ^ 4 * (dblRaw A (x, 1)).2) := by
        rw [hXeq]
        simpa [mul_comm] using dblRaw_homogeneous A Z x 1
      rw [hraw]
      constructor
      · rcases hnorm.1 with hn | hn
        · exact Or.inl (mul_ne_zero (pow_ne_zero _ hZ) hn)
        · exact Or.inr (mul_ne_zero (pow_ne_zero _ hZ) hn)
      · unfold RawRep at hnorm ⊢
        rw [show (Point.some h + Point.some h) = (.some h + .some h) from rfl]
        calc
          Z ^ 4 * (dblRaw A (x, 1)).1 * (pointXRep A (.some h + .some h)).Z =
              Z ^ 4 * ((dblRaw A (x, 1)).1 * (pointXRep A (.some h + .some h)).Z) := by ring
          _ = Z ^ 4 * ((pointXRep A (.some h + .some h)).X * (dblRaw A (x, 1)).2) := by rw [hnorm.2]
          _ = (pointXRep A (.some h + .some h)).X * (Z ^ 4 * (dblRaw A (x, 1)).2) := by ring

end Montgomery
end TotalKummer
end X25519
end Atlas
