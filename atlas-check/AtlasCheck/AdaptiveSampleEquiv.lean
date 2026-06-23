import AtlasCheck.AdaptiveTape
import AtlasCheck.AdaptiveGame
import AtlasCheck.AdaptiveCounting

namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA

universe u v w

variable {ell : Nat} [NeZero ell]

section SampleEquiv

variable {Msg : Type v} {Seed : Type w} {rounds : Nat}
variable (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
variable (A : Adversary S.G Msg ell Seed) [Fintype Seed]

abbrev Sample :=
  (ZMod ell × Seed) × AnswerTape (ZMod ell) (rounds + 1)

def SampleWins (s : Sample (ell := ell) (Seed := Seed) (rounds := rounds)) : Prop :=
  ∃ f, AcceptedRun S A (S.toPoint s.1.1) s.1.2 s.2 f

abbrev SuccessfulSample :=
  {s : Sample (ell := ell) (Seed := Seed) (rounds := rounds) //
    SampleWins (rounds := rounds) S A s}

abbrev Context :=
  (ZMod ell × Seed) × AnswerTape (ZMod ell) rounds

abbrev RowBase :=
  Fin (rounds + 1) × Context (ell := ell) (Seed := Seed) (rounds := rounds)

abbrev SuccessfulRow :=
  Σ row : RowBase (ell := ell) (Seed := Seed) (rounds := rounds),
    {c : ZMod ell // GoodAt S A row.1 row.2 c}

noncomputable def chosenForgery
    (s : SuccessfulSample (rounds := rounds) S A) :
    Forgery S.G Msg ell (rounds + 1) :=
  Classical.choose s.prop

theorem chosenForgery_spec
    (s : SuccessfulSample (rounds := rounds) S A) :
    AcceptedRun S A (S.toPoint s.val.1.1) s.val.1.2 s.val.2
      (chosenForgery (rounds := rounds) S A s) :=
  Classical.choose_spec s.prop

theorem acceptedRun_unique
    {pk : S.G} {seed : Seed}
    {tape : AnswerTape (ZMod ell) (rounds + 1)}
    {f g : Forgery S.G Msg ell (rounds + 1)}
    (hf : AcceptedRun S A pk seed tape f)
    (hg : AcceptedRun S A pk seed tape g) : f = g := by
  exact Option.some.inj (hf.finish_eq.symm.trans hg.finish_eq)

noncomputable def successfulSampleToRow
    (s : SuccessfulSample (rounds := rounds) S A) :
    SuccessfulRow (rounds := rounds) S A := by
  let f := chosenForgery (rounds := rounds) S A s
  let i := f.critical
  let ctx : Context (ell := ell) (Seed := Seed) (rounds := rounds) :=
    (s.val.1, AnswerTape.removeAt i s.val.2)
  let c := AnswerTape.get s.val.2 i
  refine ⟨(i, ctx), ⟨c, ?_⟩⟩
  refine ⟨f, ?_, rfl⟩
  simpa [ctx, c, f] using chosenForgery_spec (rounds := rounds) S A s

noncomputable def successfulRowToSample
    (r : SuccessfulRow (rounds := rounds) S A) :
    SuccessfulSample (rounds := rounds) S A := by
  let i := r.1.1
  let ctx := r.1.2
  let c := r.2.val
  refine ⟨(ctx.1, AnswerTape.insert i ctx.2 c), ?_⟩
  rcases r.2.prop with ⟨f, hf, hi⟩
  exact ⟨f, hf⟩

noncomputable def successfulSampleEquivRow :
    SuccessfulSample (rounds := rounds) S A ≃
      SuccessfulRow (rounds := rounds) S A where
  toFun := successfulSampleToRow (rounds := rounds) S A
  invFun := successfulRowToSample (rounds := rounds) S A
  left_inv := by
    intro s
    apply Subtype.ext
    simp [successfulSampleToRow, successfulRowToSample]
  right_inv := by
    rintro ⟨⟨i, ctx⟩, ⟨c, hgood⟩⟩
    rcases hgood with ⟨f, hf, hfi⟩
    change f.critical = i at hfi
    let r : SuccessfulRow (rounds := rounds) S A :=
      ⟨⟨i, ctx⟩, ⟨c, ⟨f, hf, hfi⟩⟩⟩
    have hchosen :
        chosenForgery (rounds := rounds) S A
          (successfulRowToSample (rounds := rounds) S A r) = f := by
      apply acceptedRun_unique (rounds := rounds) S A
      · exact chosenForgery_spec (rounds := rounds) S A _
      · exact hf
    change successfulSampleToRow (rounds := rounds) S A
      (successfulRowToSample (rounds := rounds) S A r) = r
    subst i
    subst f
    simp [successfulSampleToRow, successfulRowToSample, r]

noncomputable def gameSuccessMass : Nat := by
  classical
  letI : Fintype (SuccessfulSample (rounds := rounds) S A) := Fintype.ofFinite _
  exact Fintype.card (SuccessfulSample (rounds := rounds) S A)

noncomputable def gameSuccessProbability : ℚ :=
  gameSuccessMass (rounds := rounds) S A /
    sampleMass (rounds := rounds) S A

theorem gameSuccessMass_eq_rowCard :
    gameSuccessMass (rounds := rounds) S A =
      @Fintype.card (SuccessfulRow (rounds := rounds) S A)
        (Fintype.ofFinite _) := by
  classical
  letI : Fintype (SuccessfulSample (rounds := rounds) S A) := Fintype.ofFinite _
  letI : Fintype (SuccessfulRow (rounds := rounds) S A) := Fintype.ofFinite _
  exact Fintype.card_congr (successfulSampleEquivRow (rounds := rounds) S A)

end SampleEquiv

end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
