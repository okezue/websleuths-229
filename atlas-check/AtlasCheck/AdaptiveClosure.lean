import AtlasCheck.AdaptiveSampleEquiv
import AtlasCheck.AdaptiveExtraction

namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA

open scoped BigOperators

universe u v w

variable {ell : Nat} [NeZero ell]

section Closure

variable {Msg : Type v} {Seed : Type w} {rounds : Nat}
variable (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
variable (A : Adversary S.G Msg ell Seed) [Fintype Seed]

noncomputable def successfulRowSigmaEquiv :
    SuccessfulRow (rounds := rounds) S A ≃
      Σ row : RowBase (ell := ell) (Seed := Seed) (rounds := rounds),
        {c : ZMod ell // GoodAt S A row.1 row.2 c} where
  toFun r := ⟨r.val.1, ⟨r.val.2, r.prop⟩⟩
  invFun r := ⟨(r.1, r.2.1), r.2.2⟩
  left_inv := by intro r; apply Subtype.ext; rfl
  right_inv := by rintro ⟨row, c⟩; rfl

noncomputable def goodChallengeFinsetEquiv
    (row : RowBase (ell := ell) (Seed := Seed) (rounds := rounds)) :
    {c : ZMod ell // GoodAt S A row.1 row.2 c} ≃
      {c : ZMod ell // c ∈ (Finset.univ.filter fun c => GoodAt S A row.1 row.2 c)} where
  toFun c := ⟨c.1, Finset.mem_filter.mpr ⟨Finset.mem_univ _, c.2⟩⟩
  invFun c := ⟨c.1, (Finset.mem_filter.mp c.2).2⟩
  left_inv := by intro c; rfl
  right_inv := by intro c; rfl

theorem card_goodChallenge_eq_rowMass
    (row : RowBase (ell := ell) (Seed := Seed) (rounds := rounds)) :
    (Fintype.card {c : ZMod ell // GoodAt S A row.1 row.2 c} : ℚ) =
      rowMass S A row.1 row.2 := by
  classical
  have h := Fintype.card_congr (goodChallengeFinsetEquiv (rounds := rounds) S A row)
  norm_cast
  simpa [rowMass] using h

theorem cast_successfulRow_card_eq_successMass :
    ((@Fintype.card (SuccessfulRow (rounds := rounds) S A) (Fintype.ofFinite _)) : ℚ) =
      successMass (rounds := rounds) S A := by
  classical
  letI : Fintype (SuccessfulRow (rounds := rounds) S A) := Fintype.ofFinite _
  letI : Fintype
      (Σ row : RowBase (ell := ell) (Seed := Seed) (rounds := rounds),
        {c : ZMod ell // GoodAt S A row.1 row.2 c}) := Fintype.ofFinite _
  calc
    (Fintype.card (SuccessfulRow (rounds := rounds) S A) : ℚ) =
        (Fintype.card
          (Σ row : RowBase (ell := ell) (Seed := Seed) (rounds := rounds),
            {c : ZMod ell // GoodAt S A row.1 row.2 c}) : ℚ) := by
          exact_mod_cast Fintype.card_congr
            (successfulRowSigmaEquiv (rounds := rounds) S A)
    _ = ∑ row : RowBase (ell := ell) (Seed := Seed) (rounds := rounds),
          (Fintype.card {c : ZMod ell // GoodAt S A row.1 row.2 c} : ℚ) := by
          rw [Fintype.card_sigma, Nat.cast_sum]
    _ = ∑ row : RowBase (ell := ell) (Seed := Seed) (rounds := rounds),
          rowMass S A row.1 row.2 := by
          apply Fintype.sum_congr
          intro row
          exact card_goodChallenge_eq_rowMass (rounds := rounds) S A row
    _ = successMass (rounds := rounds) S A := by
          simp [successMass, rowMasses]

theorem gameSuccessProbability_eq_successProbability :
    gameSuccessProbability (rounds := rounds) S A =
      successProbability (rounds := rounds) S A := by
  classical
  letI : Fintype (SuccessfulSample (rounds := rounds) S A) := Fintype.ofFinite _
  letI : Fintype (SuccessfulRow (rounds := rounds) S A) := Fintype.ofFinite _
  unfold gameSuccessProbability successProbability gameSuccessMass
  rw [Fintype.card_congr (successfulSampleEquivRow (rounds := rounds) S A)]
  rw [cast_successfulRow_card_eq_successMass (rounds := rounds) S A]

theorem ordinary_game_forking_bound_closed
    [Nonempty Seed] [Fact (Nat.Prime ell)] :
    gameSuccessProbability (rounds := rounds) S A ^ 2 ≤
      (rounds + 1 : ℚ) * forkingProbability (rounds := rounds) S A +
      (rounds + 1 : ℚ) * gameSuccessProbability (rounds := rounds) S A /
        Fintype.card (ZMod ell) := by
  rw [gameSuccessProbability_eq_successProbability (rounds := rounds) S A]
  exact forking_probability_bound_tight (rounds := rounds) S A

/-- The rewind experiment uses two genuine runs with the same prefix, resamples only the selected
fresh random-oracle answer, and extracts the discrete logarithm from the actual responses. -/
theorem actual_rewind_extracts
    [Fact (Nat.Prime ell)]
    (i : Fin (rounds + 1))
    (ctx : (ZMod ell × Seed) × AnswerTape (ZMod ell) rounds)
    (c₁ c₂ : ZMod ell)
    (h₁ : GoodAt S A i ctx c₁) (h₂ : GoodAt S A i ctx c₂)
    (hne : c₁ ≠ c₂) :
    rewindOutput S A (S.toPoint ctx.1.1) ctx.1.2 i ctx.2 c₁ c₂ =
      some ctx.1.1 :=
  rewindOutput_correct_of_goodAt S A i ctx c₁ c₂ h₁ h₂ hne

end Closure

end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
