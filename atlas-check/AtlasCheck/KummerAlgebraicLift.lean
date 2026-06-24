import Mathlib.AlgebraicGeometry.EllipticCurve.Affine.Point
import Mathlib.FieldTheory.IsAlgClosed.AlgebraicClosure

namespace Atlas
namespace X25519
namespace TotalKummer

open WeierstrassCurve

universe u

variable {K : Type u} [Field K] [DecidableEq K] [IsAlgClosed K]

noncomputable def montgomeryCurve (A : K) : WeierstrassCurve K where
  a₁ := 0
  a₂ := A
  a₃ := 0
  a₄ := 1
  a₆ := 0

noncomputable abbrev montgomeryAffine (A : K) : WeierstrassCurve.Affine K :=
  (montgomeryCurve A).toAffine

structure ProjectiveX (K : Type u) [Field K] where
  X : K
  Z : K
  nonzero : X ≠ 0 ∨ Z ≠ 0

namespace ProjectiveX

variable (p q : ProjectiveX K)

def Eqv : Prop := p.X * q.Z = q.X * p.Z

@[refl] theorem eqv_refl : p.Eqv p := by
  simp [Eqv]

@[symm] theorem eqv_symm : p.Eqv q → q.Eqv p := by
  intro h
  simpa [Eqv, mul_comm] using h.symm

@[trans] theorem eqv_trans {r : ProjectiveX K} :
    p.Eqv q → q.Eqv r → p.Eqv r := by
  intro hpq hqr
  rcases q.nonzero with hqX | hqZ
  · apply mul_left_cancel₀ hqX
    calc
      q.X * (p.X * r.Z) = p.X * (q.X * r.Z) := by ring
      _ = p.X * (r.X * q.Z) := by rw [hqr]
      _ = r.X * (p.X * q.Z) := by ring
      _ = r.X * (q.X * p.Z) := by rw [hpq]
      _ = q.X * (r.X * p.Z) := by ring
  · apply mul_right_cancel₀ hqZ
    calc
      p.X * r.Z * q.Z = (p.X * q.Z) * r.Z := by ring
      _ = (q.X * p.Z) * r.Z := by rw [hpq]
      _ = p.Z * (q.X * r.Z) := by ring
      _ = p.Z * (r.X * q.Z) := by rw [hqr]
      _ = r.X * p.Z * q.Z := by ring

end ProjectiveX

namespace Montgomery

variable (A : K)
variable [(montgomeryCurve A).IsElliptic]

abbrev Point := (montgomeryAffine A).Point

noncomputable def pointXRep (P : Point A) : ProjectiveX K :=
  WeierstrassCurve.Affine.Point.casesOn P
    ⟨1, 0, Or.inl one_ne_zero⟩
    (fun {x y} _h => ⟨x, 1, Or.inr one_ne_zero⟩)

@[simp] theorem pointXRep_zero :
    pointXRep A (0 : Point A) = ⟨1, 0, Or.inl one_ne_zero⟩ := by
  rfl

@[simp] theorem pointXRep_some {x y : K}
    (h : (montgomeryAffine A).Nonsingular x y) :
    pointXRep A (.some h) = ⟨x, 1, Or.inr one_ne_zero⟩ := by
  rfl

@[simp] theorem pointXRep_neg (P : Point A) :
    pointXRep A (-P) = pointXRep A P := by
  cases P with
  | zero => rfl
  | some h => rfl

noncomputable def rhs (x : K) : K := x ^ 3 + A * x ^ 2 + x

noncomputable def sqrtRhs (x : K) : K :=
  Classical.choose (IsAlgClosed.exists_pow_nat_eq (rhs A x) (n := 2) (by norm_num))

@[simp] theorem sqrtRhs_sq (x : K) :
    sqrtRhs A x ^ 2 = rhs A x :=
  Classical.choose_spec (IsAlgClosed.exists_pow_nat_eq (rhs A x) (n := 2) (by norm_num))

noncomputable def liftAffineX (x : K) : Point A := by
  have heq : (montgomeryAffine A).Equation x (sqrtRhs A x) := by
    rw [WeierstrassCurve.Affine.equation_iff]
    simpa [montgomeryAffine, montgomeryCurve, rhs] using sqrtRhs_sq A x
  exact .some (WeierstrassCurve.Affine.equation_iff_nonsingular.mp heq)

@[simp] theorem pointXRep_liftAffineX (x : K) :
    pointXRep A (liftAffineX A x) = ⟨x, 1, Or.inr one_ne_zero⟩ := by
  rfl

noncomputable def liftProjective (p : ProjectiveX K) : Point A :=
  if p.Z = 0 then 0 else liftAffineX A (p.X / p.Z)

def Represents (p : ProjectiveX K) (P : Point A) : Prop :=
  p.Eqv (pointXRep A P)

@[simp] theorem liftProjective_represents (p : ProjectiveX K) :
    Represents A p (liftProjective A p) := by
  by_cases hZ : p.Z = 0
  · simp [liftProjective, hZ, Represents, ProjectiveX.Eqv]
  · simp [liftProjective, hZ, Represents, ProjectiveX.Eqv]
    exact (div_mul_cancel₀ p.X hZ).symm

private theorem zero_of_represents_at_infinity
    (p : ProjectiveX K) (hZ : p.Z = 0)
    (P : Point A) (hP : Represents A p P) : P = 0 := by
  cases P with
  | zero => rfl
  | @some x y h =>
      exfalso
      have hpX : p.X = 0 := by
        simpa [Represents, ProjectiveX.Eqv, pointXRep, hZ] using hP
      exact p.nonzero.elim (fun hX => hX hpX) (fun hZ' => hZ' hZ)

private theorem some_of_represents_affine
    (p : ProjectiveX K) (hZ : p.Z ≠ 0)
    (P : Point A) (hP : Represents A p P) :
    ∃ (x y : K) (h : (montgomeryAffine A).Nonsingular x y),
      P = .some h ∧ x = p.X / p.Z := by
  cases P with
  | zero =>
      have hpZ : p.Z = 0 := by
        simpa [Represents, ProjectiveX.Eqv, pointXRep] using hP.symm
      exact (hZ hpZ).elim
  | @some x y h =>
      refine ⟨x, y, h, rfl, ?_⟩
      have hxmul : p.X = x * p.Z := by
        simpa [Represents, ProjectiveX.Eqv, pointXRep] using hP
      exact (eq_div_iff hZ).2 hxmul.symm

theorem represents_unique_up_to_sign
    (p : ProjectiveX K) {P Q : Point A}
    (hP : Represents A p P) (hQ : Represents A p Q) :
    P = Q ∨ P = -Q := by
  by_cases hZ : p.Z = 0
  · have hp0 := zero_of_represents_at_infinity A p hZ P hP
    have hq0 := zero_of_represents_at_infinity A p hZ Q hQ
    exact Or.inl (hp0.trans hq0.symm)
  · rcases some_of_represents_affine A p hZ P hP with
      ⟨x₁, y₁, hp₁, hPsome, hx₁⟩
    rcases some_of_represents_affine A p hZ Q hQ with
      ⟨x₂, y₂, hp₂, hQsome, hx₂⟩
    have hx : x₁ = x₂ := hx₁.trans hx₂.symm
    have hy := WeierstrassCurve.Affine.Y_eq_of_X_eq hp₁.1 hp₂.1 hx
    rw [hPsome, hQsome]
    rcases hy with hy | hy
    · exact Or.inl (by
        simp only [WeierstrassCurve.Affine.Point.some.injEq]
        exact ⟨hx, hy⟩)
    · exact Or.inr (by
        simp only [WeierstrassCurve.Affine.Point.neg_some,
          WeierstrassCurve.Affine.Point.some.injEq]
        exact ⟨hx, hy⟩)

theorem total_projective_to_kummer (p : ProjectiveX K) :
    ∃ P : Point A,
      Represents A p P ∧
      ∀ Q : Point A, Represents A p Q → Q = P ∨ Q = -P := by
  refine ⟨liftProjective A p, liftProjective_represents A p, ?_⟩
  intro Q hQ
  exact represents_unique_up_to_sign A p hQ (liftProjective_represents A p)

end Montgomery
end TotalKummer
end X25519
end Atlas
