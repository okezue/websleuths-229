import AtlasCheck.AdaptiveGame

namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA

open scoped BigOperators

universe u v w

variable {ell : Nat} [NeZero ell]

section Counting

variable {Msg : Type v} {Seed : Type w} {rounds : Nat}
variable (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
variable (A : Adversary S.G Msg ell Seed) [Fintype Seed]

noncomputable def rowMass (i : Fin (rounds + 1))
    (ctx : (ZMod ell × Seed) × AnswerTape (ZMod ell) rounds) : Nat := by
  classical
  exact ((Finset.univ : Finset (ZMod ell)).filter
    (fun c => GoodAt S A i ctx c)).card

noncomputable def rowMasses : List Nat := by
  classical
  exact ((Finset.univ : Finset (Fin (rounds + 1) ×
    ((ZMod ell × Seed) × AnswerTape (ZMod ell) rounds))).toList.map
      (fun row => rowMass S A row.1 row.2))

noncomputable def successMass : Nat :=
  (rowMasses (rounds := rounds) S A).sum

noncomputable def forkMass : Nat :=
  ((rowMasses (rounds := rounds) S A).map fun s => s * (s - 1)).sum

noncomputable def sampleMass
    (_S : ScalarGroup ell) (_A : Adversary _ Msg ell Seed) : Nat :=
  Fintype.card (ZMod ell × Seed) *
    Fintype.card (AnswerTape (ZMod ell) (rounds + 1))

noncomputable def forkSampleMass
    (S : ScalarGroup ell) (A : Adversary S.G Msg ell Seed) : Nat :=
  sampleMass (rounds := rounds) S A * Fintype.card (ZMod ell)

noncomputable def successProbability : ℚ :=
  successMass (rounds := rounds) S A / sampleMass (rounds := rounds) S A

noncomputable def forkingProbability : ℚ :=
  forkMass (rounds := rounds) S A / forkSampleMass (rounds := rounds) S A

@[simp] theorem rowMasses_length :
    (rowMasses (rounds := rounds) S A).length =
      (rounds + 1) * Fintype.card (ZMod ell × Seed) *
        Fintype.card (AnswerTape (ZMod ell) rounds) := by
  classical
  simp [rowMasses, Nat.mul_assoc]

private theorem pair_count_add_self (s : Nat) :
    s * (s - 1) + s = s * s := by
  by_cases h : s = 0
  · simp [h]
  · have hs1 : 1 ≤ s := Nat.one_le_iff_ne_zero.mpr h
    have hle : s ≤ s * s := by nlinarith
    rw [Nat.mul_sub_left_distrib, Nat.mul_one, Nat.sub_add_cancel hle]

private theorem list_sum_sq_le_length_mul_sum_sq (xs : List ℚ) :
    xs.sum ^ 2 ≤ (xs.length : ℚ) * (xs.map fun x => x ^ 2).sum := by
  induction xs with
  | nil => simp
  | cons a xs ih =>
      cases xs with
      | nil => simp
      | cons b ys =>
          let tail : List ℚ := b :: ys
          let n : ℚ := tail.length
          let t : ℚ := tail.sum
          let q : ℚ := (tail.map fun x => x ^ 2).sum
          have hn : 0 < n := by
            dsimp [n, tail]
            positivity
          have hvar : 0 ≤ n * q - t ^ 2 := by
            dsimp [n, t, q, tail]
            nlinarith [ih]
          have hsq : 0 ≤ (n * a - t) ^ 2 := sq_nonneg (n * a - t)
          have hcross : 0 ≤ n * a ^ 2 + q - 2 * a * t := by
            nlinarith
          dsimp [tail, n, t, q] at hvar hcross ⊢
          simp only [List.sum_cons, List.length_cons, List.map_cons,
            Nat.cast_add, Nat.cast_one]
          nlinarith

theorem successMass_sq_le_rows_mul_fork_add_success :
    (successMass (rounds := rounds) S A : ℚ) ^ 2 ≤
      ((rowMasses (rounds := rounds) S A).length : ℚ) *
        ((forkMass (rounds := rounds) S A : ℚ) +
          successMass (rounds := rounds) S A) := by
  let xs : List ℚ :=
    (rowMasses (rounds := rounds) S A).map (fun s => (s : ℚ))
  have hc := list_sum_sq_le_length_mul_sum_sq xs
  have hlen : xs.length =
      (rowMasses (rounds := rounds) S A).length := by simp [xs]
  have hsum : xs.sum = (successMass (rounds := rounds) S A : ℚ) := by
    simp [xs, successMass]
  have hsquares : (xs.map fun x => x ^ 2).sum =
      ((forkMass (rounds := rounds) S A : ℚ) +
        successMass (rounds := rounds) S A) := by
    simp only [xs, List.map_map, Function.comp_apply]
    rw [← Nat.cast_add, ← Nat.cast_sum, ← Nat.cast_sum]
    congr 1
    simp only [forkMass, successMass]
    induction rowMasses (rounds := rounds) S A with
    | nil => simp
    | cons s ss ih =>
        simp only [List.map_cons, List.sum_cons]
        rw [pair_count_add_self]
        omega
  simpa [hlen, hsum, hsquares] using hc

theorem forking_probability_bound_tight
    [Nonempty Seed] [Fact (Nat.Prime ell)] :
    successProbability (rounds := rounds) S A ^ 2 ≤
      (rounds + 1 : ℚ) * forkingProbability (rounds := rounds) S A +
      (rounds + 1 : ℚ) * successProbability (rounds := rounds) S A /
        Fintype.card (ZMod ell) := by
  have hcount :=
    successMass_sq_le_rows_mul_fork_add_success (rounds := rounds) S A
  rw [rowMasses_length (rounds := rounds)] at hcount
  have hseed : (0 : ℚ) < Fintype.card (ZMod ell × Seed) := by positivity
  have hresp : (0 : ℚ) < Fintype.card (ZMod ell) := by positivity
  have htape : (0 : ℚ) < Fintype.card (AnswerTape (ZMod ell) rounds) := by
    positivity
  simp only [successProbability, forkingProbability, forkSampleMass, sampleMass]
  rw [AnswerTape.card (R := ZMod ell) rounds,
    AnswerTape.card (R := ZMod ell) (rounds + 1)]
  rw [pow_succ]
  norm_num at hcount ⊢
  field_simp
  nlinarith

end Counting

end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
