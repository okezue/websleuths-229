import AtlasCheck.AdaptiveCounting

namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA

universe u v w

variable {ell : Nat} [NeZero ell]

def extract (s₁ s₂ c₁ c₂ : ZMod ell) : Option (ZMod ell) :=
  if c₁ = c₂ then none else some ((s₁ - s₂) * (c₁ - c₂)⁻¹)

theorem extract_correct [Fact (Nat.Prime ell)]
    (S : ScalarGroup ell) (d : ZMod ell) (R : S.G)
    (s₁ s₂ c₁ c₂ : ZMod ell)
    (h₁ : verifies S (S.toPoint d) R s₁ c₁)
    (h₂ : verifies S (S.toPoint d) R s₂ c₂)
    (hne : c₁ ≠ c₂) :
    extract s₁ s₂ c₁ c₂ = some d := by
  have hscalar : s₁ - s₂ = (c₁ - c₂) * d := by
    apply S.toPoint_injective
    rw [map_sub, h₁, h₂, S.act_toPoint, S.act_toPoint]
    have hr : (c₁ - c₂) * d = c₁ * d - c₂ * d := by ring
    rw [hr, map_sub]
    abel
  unfold extract
  rw [if_neg hne]
  congr 1
  have hcne : c₁ - c₂ ≠ 0 := sub_ne_zero.mpr hne
  rw [hscalar]
  field_simp

noncomputable def rewindOutput
    {Msg : Type v} {Seed : Type w} {rounds : Nat}
    (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
    (A : Adversary S.G Msg ell Seed)
    (pk : S.G) (seed : Seed) (i : Fin (rounds + 1))
    (ctx : AnswerTape (ZMod ell) rounds) (c₁ c₂ : ZMod ell) :
    Option (ZMod ell) := by
  classical
  exact
  let tape₁ := AnswerTape.insert i ctx c₁
  let tape₂ := AnswerTape.insert i ctx c₂
  let out₁ := finishForgery (queries := rounds + 1) A (run A pk seed tape₁).adv
  let out₂ := finishForgery (queries := rounds + 1) A (run A pk seed tape₂).adv
  match out₁, out₂ with
  | some f₁, some f₂ =>
      if h : AcceptedRun S A pk seed tape₁ f₁ ∧
          AcceptedRun S A pk seed tape₂ f₂ ∧
          f₁.critical = i ∧ f₂.critical = i ∧ c₁ ≠ c₂ then
        extract f₁.response f₂.response c₁ c₂
      else none
  | _, _ => none

theorem rewindOutput_correct_of_goodAt
    {Msg : Type v} {Seed : Type w} {rounds : Nat}
    (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
    (A : Adversary S.G Msg ell Seed)
    [Fact (Nat.Prime ell)]
    (i : Fin (rounds + 1))
    (ctx : (ZMod ell × Seed) × AnswerTape (ZMod ell) rounds)
    (c₁ c₂ : ZMod ell)
    (h₁ : GoodAt S A i ctx c₁) (h₂ : GoodAt S A i ctx c₂)
    (hne : c₁ ≠ c₂) :
    rewindOutput S A (S.toPoint ctx.1.1) ctx.1.2 i ctx.2 c₁ c₂ =
      some ctx.1.1 := by
  rcases h₁ with ⟨f₁, ha₁, hi₁⟩
  rcases h₂ with ⟨f₂, ha₂, hi₂⟩
  let tape₁ := AnswerTape.insert i ctx.2 c₁
  let tape₂ := AnswerTape.insert i ctx.2 c₂
  have hfresh₁ : freshAt A (S.toPoint ctx.1.1) ctx.1.2 tape₁ i = true := by
    simpa [tape₁, hi₁] using ha₁.fresh
  have hfresh₂ : freshAt A (S.toPoint ctx.1.1) ctx.1.2 tape₂ i = true := by
    simpa [tape₂, hi₂] using ha₂.fresh
  have hq₁ : queryAt A (S.toPoint ctx.1.1) ctx.1.2 tape₁ i =
      { publicKey := S.toPoint ctx.1.1,
        commitment := f₁.commitment, message := f₁.message } := by
    simpa [tape₁, hi₁] using ha₁.query_eq
  have hq₂ : queryAt A (S.toPoint ctx.1.1) ctx.1.2 tape₂ i =
      { publicKey := S.toPoint ctx.1.1,
        commitment := f₂.commitment, message := f₂.message } := by
    simpa [tape₂, hi₂] using ha₂.query_eq
  have hsame : queryAt A (S.toPoint ctx.1.1) ctx.1.2 tape₁ i =
      queryAt A (S.toPoint ctx.1.1) ctx.1.2 tape₂ i := by
    simpa [tape₁, tape₂] using
      queryAt_insert_same A (S.toPoint ctx.1.1) ctx.1.2 i ctx.2 c₁ c₂
  have hqueries :
      ({ publicKey := S.toPoint ctx.1.1,
         commitment := f₁.commitment, message := f₁.message } :
        ChallengeQuery S.G Msg) =
      { publicKey := S.toPoint ctx.1.1,
        commitment := f₂.commitment, message := f₂.message } := by
    exact hq₁.symm.trans (hsame.trans hq₂)
  have hR : f₁.commitment = f₂.commitment :=
    congrArg ChallengeQuery.commitment hqueries
  have hv₁ : verifies S (S.toPoint ctx.1.1) f₁.commitment f₁.response c₁ := by
    have hans := answerAt_insert_of_fresh A (S.toPoint ctx.1.1) ctx.1.2
      i ctx.2 c₁ hfresh₁
    simpa [tape₁, hi₁, hans] using ha₁.verifies_eq
  have hv₂ : verifies S (S.toPoint ctx.1.1) f₁.commitment f₂.response c₂ := by
    have hans := answerAt_insert_of_fresh A (S.toPoint ctx.1.1) ctx.1.2
      i ctx.2 c₂ hfresh₂
    have hraw : verifies S (S.toPoint ctx.1.1) f₂.commitment f₂.response c₂ := by
      simpa [tape₂, hi₂, hans] using ha₂.verifies_eq
    rw [hR]
    exact hraw
  unfold rewindOutput
  simp only [ha₁.finish_eq, ha₂.finish_eq]
  rw [dif_pos ⟨ha₁, ha₂, hi₁, hi₂, hne⟩]
  exact extract_correct S ctx.1.1 f₁.commitment f₁.response f₂.response c₁ c₂
    hv₁ hv₂ hne

noncomputable def rewindingDLPAdvantage
    {Msg : Type v} {Seed : Type w} {rounds : Nat}
    (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
    (A : Adversary S.G Msg ell Seed) [Fintype Seed] : ℚ :=
  forkingProbability (rounds := rounds) S A

theorem adaptive_forking_bound_closed
    {Msg : Type v} {Seed : Type w} {rounds : Nat}
    (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
    (A : Adversary S.G Msg ell Seed)
    [Fintype Seed] [Nonempty Seed] [Fact (Nat.Prime ell)] :
    successProbability (rounds := rounds) S A ^ 2 ≤
      (rounds + 1 : ℚ) * rewindingDLPAdvantage (rounds := rounds) S A +
      (rounds + 1 : ℚ) * successProbability (rounds := rounds) S A /
        Fintype.card (ZMod ell) := by
  simpa [rewindingDLPAdvantage] using
    forking_probability_bound_tight (rounds := rounds) S A

end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
