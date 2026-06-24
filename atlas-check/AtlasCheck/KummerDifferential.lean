import AtlasCheck.KummerDoubling

namespace Atlas
namespace X25519
namespace TotalKummer
namespace Montgomery

open WeierstrassCurve
open WeierstrassCurve.Affine

universe u

variable {K : Type u} [Field K] [DecidableEq K] [IsAlgClosed K]
variable (A : K) [(montgomeryCurve A).IsElliptic]

noncomputable def rawOfPoint (P : Point A) : RawX (K := K) :=
  ((pointXRep A P).X, (pointXRep A P).Z)

@[simp] theorem rawOfPoint_zero :
    rawOfPoint A (0 : Point A) = (1, 0) := by rfl

@[simp] theorem rawOfPoint_some {x y : K}
    (h : (montgomeryAffine A).Nonsingular x y) :
    rawOfPoint A (.some h) = (x, 1) := by rfl

@[simp] theorem rawOfPoint_neg (P : Point A) :
    rawOfPoint A (-P) = rawOfPoint A P := by
  simp [rawOfPoint]

noncomputable def addRaw (p q d : RawX (K := K)) : RawX (K := K) :=
  let pA := p.1 + p.2
  let pB := p.1 - p.2
  let qC := q.1 + q.2
  let qD := q.1 - q.2
  let da := qD * pA
  let cb := qC * pB
  (d.2 * (da + cb) ^ 2, d.1 * (da - cb) ^ 2)

 theorem addRaw_homogeneous
    (lp lq ld : K) (p q d : RawX (K := K)) :
    addRaw A (lp * p.1, lp * p.2) (lq * q.1, lq * q.2)
        (ld * d.1, ld * d.2) =
      (ld * lp ^ 2 * lq ^ 2 * (addRaw A p q d).1,
       ld * lp ^ 2 * lq ^ 2 * (addRaw A p q d).2) := by
  apply Prod.ext <;> simp [addRaw] <;> ring

 theorem exists_scale_of_rawRep
    (r : RawX (K := K)) (P : Point A)
    (hnz : RawNonzero r) (hrep : RawRep A r P) :
    ∃ l : K, l ≠ 0 ∧ r = (l * (rawOfPoint A P).1, l * (rawOfPoint A P).2) := by
  rcases r with ⟨X, Z⟩
  cases P with
  | zero =>
      have hZ : Z = 0 := (rawRep_zero A X Z).mp hrep
      have hX : X ≠ 0 := hnz.resolve_right (by simpa [hZ])
      refine ⟨X, hX, ?_⟩
      simp [hZ, rawOfPoint]
  | @some x y h =>
      have hX : X = x * Z := (rawRep_some A h X Z).mp hrep
      have hZ : Z ≠ 0 := by
        intro hz
        apply hnz.elim
        · intro hn
          apply hn
          simp [hX, hz]
        · exact fun hn => hn hz
      refine ⟨Z, hZ, ?_⟩
      apply Prod.ext
      · simp [rawOfPoint, hX, mul_comm]
      · simp [rawOfPoint]

private theorem base_ne_zero {xb yb : K}
    (hb : (montgomeryAffine A).Nonsingular xb yb) :
    (WeierstrassCurve.Affine.Point.some hb : Point A) ≠ 0 := by
  exact WeierstrassCurve.Affine.Point.some_ne_zero hb

private theorem add_eq_base_left {xb yb : K}
    (hb : (montgomeryAffine A).Nonsingular xb yb) :
    (0 : Point A) + .some hb = .some hb := by simp

private theorem add_eq_base_right {xb yb : K}
    (hb : (montgomeryAffine A).Nonsingular xb yb) :
    (.some hb : Point A) + 0 = .some hb := by simp

private theorem diff_of_consecutive {xb yb : K}
    (hb : (montgomeryAffine A).Nonsingular xb yb)
    (P Q : Point A) (hQ : Q = P + .some hb) :
    P + -Q = -(.some hb : Point A) := by
  rw [hQ]
  abel

private theorem affine_equation {x y : K}
    (h : (montgomeryAffine A).Nonsingular x y) :
    y ^ 2 = x ^ 3 + A * x ^ 2 + x :=
  (equation_iff_montgomery A x y).mp h.1

private theorem normalized_add_distinct
    (h2 : (2 : K) ≠ 0)
    {xb yb x₁ y₁ x₂ y₂ : K}
    (hb : (montgomeryAffine A).Nonsingular xb yb)
    (hxB : xb ≠ 0)
    (h₁ : (montgomeryAffine A).Nonsingular x₁ y₁)
    (h₂p : (montgomeryAffine A).Nonsingular x₂ y₂)
    (hx : x₁ ≠ x₂)
    (hQ : (WeierstrassCurve.Affine.Point.some h₂p : Point A) =
      .some h₁ + .some hb) :
    RawNonzero (addRaw A (x₁, 1) (x₂, 1) (xb, 1)) ∧
      RawRep A (addRaw A (x₁, 1) (x₂, 1) (.some hb |> rawOfPoint A))
        (.some h₁ + .some h₂p) := by
  have hc₁ := affine_equation A h₁
  have hc₂ := affine_equation A h₂p
  have hdiff : (.some h₁ : Point A) + -(.some h₂p) = -(.some hb) :=
    diff_of_consecutive A hb (.some h₁) (.some h₂p) hQ
  have hneg₂ : (montgomeryAffine A).Nonsingular x₂
      ((montgomeryAffine A).negY x₂ y₂) :=
    (WeierstrassCurve.Affine.nonsingular_neg x₂ y₂).mpr h₂p
  have hdadd := WeierstrassCurve.Affine.Point.add_of_X_ne
    (W := montgomeryAffine A) hx (h₁ := h₁) (h₂ := hneg₂)
  have hxD :
      (montgomeryAffine A).addX x₁ x₂
          ((montgomeryAffine A).slope x₁ x₂ y₁
            ((montgomeryAffine A).negY x₂ y₂)) = xb := by
    have heq := hdiff
    rw [WeierstrassCurve.Affine.Point.neg_some, hdadd,
      WeierstrassCurve.Affine.Point.neg_some] at heq
    exact (WeierstrassCurve.Affine.Point.some.inj heq).1
  have hsadd := WeierstrassCurve.Affine.Point.add_of_X_ne
    (W := montgomeryAffine A) hx (h₁ := h₁) (h₂ := h₂p)
  rw [rawOfPoint_some, hsadd]
  constructor
  · right
    simp [addRaw, hxB, h2, hx]
  · simp only [RawRep, rawOfPoint, pointXRep_some, Prod.fst, Prod.snd, mul_one]
    rw [WeierstrassCurve.Affine.slope_of_X_ne hx,
      WeierstrassCurve.Affine.slope_of_X_ne hx]
    simp [addRaw, montgomeryAffine, montgomeryCurve,
      WeierstrassCurve.Affine.negY,
      WeierstrassCurve.Affine.addX] at hxD ⊢
    have hden : x₁ - x₂ ≠ 0 := sub_ne_zero.mpr hx
    field_simp [hden] at hxD ⊢
    polyrith

private theorem normalized_add_same_x
    (h2 : (2 : K) ≠ 0) (hA : A ^ 2 ≠ 4)
    {xb yb x y₁ y₂ : K}
    (hb : (montgomeryAffine A).Nonsingular xb yb)
    (hxB : xb ≠ 0)
    (h₁ : (montgomeryAffine A).Nonsingular x y₁)
    (h₂p : (montgomeryAffine A).Nonsingular x y₂)
    (hQ : (WeierstrassCurve.Affine.Point.some h₂p : Point A) =
      .some h₁ + .some hb) :
    RawNonzero (addRaw A (x, 1) (x, 1) (xb, 1)) ∧
      RawRep A (addRaw A (x, 1) (x, 1) (xb, 1))
        (.some h₁ + .some h₂p) := by
  have hneq : (WeierstrassCurve.Affine.Point.some h₁ : Point A) ≠ .some h₂p := by
    intro heq
    have hb0 : (WeierstrassCurve.Affine.Point.some hb : Point A) = 0 := by
      rw [heq] at hQ
      have := hQ
      apply add_left_cancel (a := -(.some h₁ : Point A))
      simpa using congrArg (fun T : Point A => -(.some h₁) + T) hQ.symm
    exact base_ne_zero A hb hb0
  have hy_cases := WeierstrassCurve.Affine.Y_eq_of_X_eq h₁.1 h₂p.1 rfl
  have hyneg : y₁ = (montgomeryAffine A).negY x y₂ :=
    hy_cases.resolve_left (by
      intro hy
      apply hneq
      simp only [WeierstrassCurve.Affine.Point.some.injEq]
      exact ⟨rfl, hy⟩)
  have hadd := WeierstrassCurve.Affine.Point.add_of_Y_eq
    (W := montgomeryAffine A) rfl hyneg (h₁ := h₁) (h₂ := h₂p)
  have hdiff : (.some h₁ : Point A) + -(.some h₂p) = -(.some hb) :=
    diff_of_consecutive A hb (.some h₁) (.some h₂p) hQ
  have hdouble : (.some h₁ : Point A) + .some h₁ = .some hb := by
    rw [← hdiff]
    rw [WeierstrassCurve.Affine.Point.neg_some]
    have hy2 : (montgomeryAffine A).negY x y₂ = y₁ := hyneg.symm
    simp only [WeierstrassCurve.Affine.Point.some.injEq] at hQ
    rw [hy2]
    simpa using congrArg Neg.neg hdiff
  have hdbl := dblRaw_normalized_sound A h2 hA x y₁ h₁
  have hdblrep : RawRep A (dblRaw A (x, 1)) (.some hb) := by
    rw [← hdouble]
    exact hdbl.2
  have hnum : (x ^ 2 - 1) ^ 2 ≠ 0 := by
    intro hz
    have hXzero : (dblRaw A (x, 1)).1 = 0 := by simpa [dblRaw] using hz
    have hZzero : (dblRaw A (x, 1)).2 = 0 := by
      have : (dblRaw A (x, 1)).1 = xb * (dblRaw A (x, 1)).2 :=
        (rawRep_some A hb _ _).mp hdblrep
      rw [hXzero] at this
      exact (mul_eq_zero.mp this.symm).resolve_left hxB
    exact hdbl.1.elim (fun h => h hXzero) (fun h => h hZzero)
  rw [hadd]
  constructor
  · left
    simp [addRaw, h2, hnum]
  · simp [RawRep, addRaw, pointXRep, h2]

 theorem addRaw_canonical_consecutive_sound
    (h2 : (2 : K) ≠ 0) (hA : A ^ 2 ≠ 4)
    {xb yb : K}
    (hb : (montgomeryAffine A).Nonsingular xb yb) (hxB : xb ≠ 0)
    (P : Point A) :
    let Q := P + (.some hb : Point A)
    RawNonzero (addRaw A (rawOfPoint A P) (rawOfPoint A Q) (xb, 1)) ∧
      RawRep A (addRaw A (rawOfPoint A P) (rawOfPoint A Q) (xb, 1))
        (P + Q) := by
  intro Q
  cases P with
  | zero =>
      simp [Q, addRaw, RawNonzero, RawRep, rawOfPoint, pointXRep, hxB, h2]
  | @some x₁ y₁ h₁ =>
      cases hQpoint : Q with
      | zero =>
          have hpneg : (WeierstrassCurve.Affine.Point.some h₁ : Point A) = -(.some hb) := by
            rw [Q] at hQpoint
            have := hQpoint
            calc
              (.some h₁ : Point A) = (.some h₁ + .some hb) + -(.some hb) := by abel
              _ = -(.some hb) := by rw [hQpoint]; simp
          have hx : x₁ = xb := by
            rw [WeierstrassCurve.Affine.Point.neg_some] at hpneg
            exact (WeierstrassCurve.Affine.Point.some.inj hpneg).1
          subst x₁
          simp [Q, hQpoint, addRaw, RawNonzero, RawRep, rawOfPoint,
            pointXRep, hxB, h2]
      | @some x₂ y₂ h₂p =>
          have hQ : (WeierstrassCurve.Affine.Point.some h₂p : Point A) =
              .some h₁ + .some hb := by simpa [Q] using hQpoint.symm
          by_cases hx : x₁ = x₂
          · subst x₂
            exact normalized_add_same_x A h2 hA hb hxB h₁ h₂p hQ
          · exact normalized_add_distinct A h2 hb hxB h₁ h₂p hx hQ

 theorem addRaw_consecutive_sound
    (h2 : (2 : K) ≠ 0) (hA : A ^ 2 ≠ 4)
    {xb yb : K}
    (hb : (montgomeryAffine A).Nonsingular xb yb) (hxB : xb ≠ 0)
    (p q : RawX (K := K)) (P : Point A)
    (hp : RawNonzero p) (hq : RawNonzero q)
    (hrp : RawRep A p P)
    (hrq : RawRep A q (P + (.some hb : Point A))) :
    RawNonzero (addRaw A p q (xb, 1)) ∧
      RawRep A (addRaw A p q (xb, 1))
        (P + (P + (.some hb : Point A))) := by
  rcases exists_scale_of_rawRep A p P hp hrp with ⟨lp, hlp, rfl⟩
  rcases exists_scale_of_rawRep A q (P + (.some hb : Point A)) hq hrq with
    ⟨lq, hlq, rfl⟩
  have hcanon := addRaw_canonical_consecutive_sound A h2 hA hb hxB P
  rw [addRaw_homogeneous A lp lq 1]
  constructor
  · rcases hcanon.1 with hX | hZ
    · exact Or.inl (mul_ne_zero (mul_ne_zero (pow_ne_zero _ hlp) (pow_ne_zero _ hlq)) hX)
    · exact Or.inr (mul_ne_zero (mul_ne_zero (pow_ne_zero _ hlp) (pow_ne_zero _ hlq)) hZ)
  · unfold RawRep at hcanon ⊢
    calc
      (1 * lp ^ 2 * lq ^ 2 * (addRaw A (rawOfPoint A P)
          (rawOfPoint A (P + .some hb)) (xb, 1)).1) *
          (pointXRep A (P + (P + .some hb))).Z =
        lp ^ 2 * lq ^ 2 *
          ((addRaw A (rawOfPoint A P) (rawOfPoint A (P + .some hb)) (xb, 1)).1 *
            (pointXRep A (P + (P + .some hb))).Z) := by ring
      _ = lp ^ 2 * lq ^ 2 *
          ((pointXRep A (P + (P + .some hb))).X *
            (addRaw A (rawOfPoint A P) (rawOfPoint A (P + .some hb)) (xb, 1)).2) := by
          rw [hcanon.2]
      _ = (pointXRep A (P + (P + .some hb))).X *
          (1 * lp ^ 2 * lq ^ 2 *
            (addRaw A (rawOfPoint A P) (rawOfPoint A (P + .some hb)) (xb, 1)).2) := by ring

end Montgomery
end TotalKummer
end X25519
end Atlas
