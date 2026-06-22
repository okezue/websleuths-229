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
    (ctx : (ZMod ell × Seed) × AnswerTape (ZMod ell) rounds) : ℚ := by
  classical
  exact (((Finset.univ : Finset (ZMod ell)).filter
    (fun c => GoodAt S A i ctx c)).card : ℚ)

noncomputable def rowMasses : List ℚ := by
  classical
  exact ((Finset.univ : Finset (Fin (rounds + 1) ×
    ((ZMod ell × Seed) × AnswerTape (ZMod ell) rounds))).toList.map
      (fun row => rowMass S A row.1 row.2))

noncomputable def successMass : ℚ :=
  (rowMasses (rounds := rounds) S A).sum

noncomputable def forkMass : ℚ :=
  ((rowMasses (rounds := rounds) S A).map fun s => s * (s - 1)).sum

noncomputable def sampleMass
    (_S : ScalarGroup ell) (_A : Adversary _S.G Msg ell Seed) : ℚ :=
  (Fintype.card (ZMod ell × Seed) : ℚ) *
    Fintype.card (AnswerTape (ZMod ell) (rounds + 1))

noncomputable def forkSampleMass
    (S : ScalarGroup ell) (A : Adversary S.G Msg ell Seed) : ℚ :=
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

private theorem list_sum_sq_le_length_mul_sum_sq (xs : List ℚ) :
    xs.sum ^ 2 ≤ (xs.length : ℚ) * (xs.map fun x => x ^ 2).sum := by
  induction xs with
  | nil => simp
  | cons a xs ih =>
      by_cases hxs : xs = []
      · subst xs
        simp
      · let n : ℚ := xs.length
        let s : ℚ := xs.sum
        let q : ℚ := (xs.map fun x => x ^ 2).sum
        have hnNat : 0 < xs.length := by
          have hne : xs.length ≠ 0 := by simpa using hxs
          exact Nat.pos_of_ne_zero hne
        have hn : (0 : ℚ) < n := by
          dsimp [n]
          exact_mod_cast hnNat
        have ih' : s ^ 2 ≤ n * q := by
          simpa [n, s, q] using ih
        have hsq : 0 ≤ (n * a - s) ^ 2 := sq_nonneg (n * a - s)
        have hmul : 0 ≤ n * (n * a ^ 2 + q - 2 * a * s) := by
          nlinarith [ih', hsq]
        have hcross : 0 ≤ n * a ^ 2 + q - 2 * a * s := by
          nlinarith [hmul, hn]
        have hgoal : (a + s) ^ 2 ≤ (n + 1) * (a ^ 2 + q) := by
          nlinarith [ih', hcross]
        simpa [n, s, q] using hgoal

theorem successMass_sq_le_rows_mul_fork_add_success :
    successMass (rounds := rounds) S A ^ 2 ≤
      ((rowMasses (rounds := rounds) S A).length : ℚ) *
        (forkMass (rounds := rounds) S A +
          successMass (rounds := rounds) S A) := by
  let masses := rowMasses (rounds := rounds) S A
  have hc := list_sum_sq_le_length_mul_sum_sq masses
  have hid : (masses.map fun x => x ^ 2).sum =
      (masses.map fun s => s * (s - 1)).sum + masses.sum := by
    induction masses with
    | nil => simp
    | cons s ss ih =>
        simp only [List.map_cons, List.sum_cons]
        rw [ih]
        ring
  rw [hid] at hc
  simpa [masses, successMass, forkMass] using hc

theorem forking_probability_bound_tight
    [Nonempty Seed] [Fact (Nat.Prime ell)] :
    successProbability (rounds := rounds) S A ^ 2 ≤
      (rounds + 1 : ℚ) * forkingProbability (rounds := rounds) S A +
      (rounds + 1 : ℚ) * successProbability (rounds := rounds) S A /
        Fintype.card (ZMod ell) := by
  have hcount :=
    successMass_sq_le_rows_mul_fork_add_success (rounds := rounds) S A
  rw [rowMasses_length (rounds := rounds)] at hcount
  let B : ℚ := Fintype.card (ZMod ell × Seed)
  let T : ℚ := Fintype.card (AnswerTape (ZMod ell) rounds)
  let H : ℚ := Fintype.card (ZMod ell)
  let q : ℚ := rounds + 1
  let suc : ℚ := successMass (rounds := rounds) S A
  let frk : ℚ := forkMass (rounds := rounds) S A
  have hB : 0 < B := by
    dsimp [B]
    positivity
  have hT : 0 < T := by
    dsimp [T]
    rw [AnswerTape.card]
    positivity
  have hH : 0 < H := by
    dsimp [H]
    positivity
  push_cast at hcount
  change suc ^ 2 ≤ q * B * T * (frk + suc) at hcount
  have hnormalized :
      (suc / (B * T * H)) ^ 2 ≤
        q * (frk / (B * T * H * H)) +
          q * (suc / (B * T * H)) / H := by
    field_simp [ne_of_gt hB, ne_of_gt hT, ne_of_gt hH]
    nlinarith [hcount]
  have hfull :
      (Fintype.card (AnswerTape (ZMod ell) (rounds + 1)) : ℚ) = T * H := by
    dsimp [T, H]
    norm_cast
    rw [AnswerTape.card, AnswerTape.card, pow_succ]
  unfold successProbability forkingProbability forkSampleMass sampleMass
  rw [hfull]
  change (suc / (B * (T * H))) ^ 2 ≤
    q * (frk / (B * (T * H) * H)) + q * (suc / (B * (T * H))) / H
  simpa [mul_assoc] using hnormalized

end Counting

end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
