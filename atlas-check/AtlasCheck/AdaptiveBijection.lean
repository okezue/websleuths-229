import AtlasCheck.AdaptiveTape
import AtlasCheck.AdaptiveGame
import AtlasCheck.AdaptiveCounting

namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA

open scoped BigOperators

universe u v w

variable {ell : Nat} [NeZero ell]

section Bijection

variable {Msg : Type v} {Seed : Type w} {rounds : Nat}
variable (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
variable (A : Adversary S.G Msg ell Seed) [Fintype Seed]

abbrev OrdinarySample :=
  (ZMod ell × Seed) × AnswerTape (ZMod ell) (rounds + 1)

def OrdinaryWins
    (s : OrdinarySample (ell := ell) (Seed := Seed) (rounds := rounds)) : Prop :=
  ∃ f, AcceptedRun S A (S.toPoint s.1.1) s.1.2 s.2 f

abbrev WinningSample :=
  {s : OrdinarySample (ell := ell) (Seed := Seed) (rounds := rounds) //
    OrdinaryWins (rounds := rounds) S A s}

structure AcceptedSample where
  secret : ZMod ell
  seed : Seed
  tape : AnswerTape (ZMod ell) (rounds + 1)
  forgery : Forgery S.G Msg ell (rounds + 1)
  accepted : AcceptedRun S A (S.toPoint secret) seed tape forgery

structure FlatSuccessfulRow where
  critical : Fin (rounds + 1)
  secret : ZMod ell
  seed : Seed
  context : AnswerTape (ZMod ell) rounds
  challenge : ZMod ell
  good : GoodAt S A critical ((secret, seed), context) challenge

noncomputable def acceptedToWinning :
    AcceptedSample (rounds := rounds) S A →
      WinningSample (rounds := rounds) S A :=
  fun s => ⟨((s.secret, s.seed), s.tape), ⟨s.forgery, s.accepted⟩⟩

noncomputable def winningToAccepted
    (s : WinningSample (rounds := rounds) S A) :
    AcceptedSample (rounds := rounds) S A := by
  classical
  let f := Classical.choose s.prop
  exact
    { secret := s.val.1.1
      seed := s.val.1.2
      tape := s.val.2
      forgery := f
      accepted := Classical.choose_spec s.prop }

theorem acceptedRun_unique
    {pk : S.G} {seed : Seed}
    {tape : AnswerTape (ZMod ell) (rounds + 1)}
    {f g : Forgery S.G Msg ell (rounds + 1)}
    (hf : AcceptedRun S A pk seed tape f)
    (hg : AcceptedRun S A pk seed tape g) : f = g := by
  exact Option.some.inj (hf.finish_eq.symm.trans hg.finish_eq)

noncomputable def acceptedSampleEquivWinning :
    AcceptedSample (rounds := rounds) S A ≃
      WinningSample (rounds := rounds) S A where
  toFun := acceptedToWinning (rounds := rounds) S A
  invFun := winningToAccepted (rounds := rounds) S A
  left_inv := by
    classical
    intro s
    cases s with
    | mk secret seed tape forgery accepted =>
        have hforgery :
            (winningToAccepted (rounds := rounds) S A
              (acceptedToWinning (rounds := rounds) S A
                { secret := secret, seed := seed, tape := tape,
                  forgery := forgery, accepted := accepted })).forgery = forgery := by
          apply acceptedRun_unique (rounds := rounds) S A
          · exact (winningToAccepted (rounds := rounds) S A
              (acceptedToWinning (rounds := rounds) S A
                { secret := secret, seed := seed, tape := tape,
                  forgery := forgery, accepted := accepted })).accepted
          · exact accepted
        cases hforgery
        rfl
  right_inv := by
    intro s
    apply Subtype.ext
    rfl

noncomputable def acceptedToRow
    (s : AcceptedSample (rounds := rounds) S A) :
    FlatSuccessfulRow (rounds := rounds) S A :=
  { critical := s.forgery.critical
    secret := s.secret
    seed := s.seed
    context := AnswerTape.removeAt s.forgery.critical s.tape
    challenge := AnswerTape.get s.tape s.forgery.critical
    good := by
      refine ⟨s.forgery, ?_, rfl⟩
      simpa using s.accepted }

noncomputable def rowToAccepted
    (r : FlatSuccessfulRow (rounds := rounds) S A) :
    AcceptedSample (rounds := rounds) S A := by
  classical
  let f := Classical.choose r.good
  have hf := Classical.choose_spec r.good
  exact
    { secret := r.secret
      seed := r.seed
      tape := AnswerTape.insert r.critical r.context r.challenge
      forgery := f
      accepted := hf.1 }

noncomputable def acceptedSampleEquivRow :
    AcceptedSample (rounds := rounds) S A ≃
      FlatSuccessfulRow (rounds := rounds) S A where
  toFun := acceptedToRow (rounds := rounds) S A
  invFun := rowToAccepted (rounds := rounds) S A
  left_inv := by
    classical
    intro s
    have hchosen :
        (rowToAccepted (rounds := rounds) S A
          (acceptedToRow (rounds := rounds) S A s)).forgery = s.forgery := by
      apply acceptedRun_unique (rounds := rounds) S A
      · exact (rowToAccepted (rounds := rounds) S A
          (acceptedToRow (rounds := rounds) S A s)).accepted
      · simpa using s.accepted
    cases s with
    | mk secret seed tape forgery accepted =>
        simp only [acceptedToRow, rowToAccepted] at hchosen ⊢
        cases hchosen
        simp
  right_inv := by
    classical
    intro r
    rcases r.good with ⟨f, hf, hcritical⟩
    have hchosen :
        (rowToAccepted (rounds := rounds) S A r).forgery = f := by
      apply acceptedRun_unique (rounds := rounds) S A
      · exact (rowToAccepted (rounds := rounds) S A r).accepted
      · exact hf
    apply FlatSuccessfulRow.ext
    · simpa [acceptedToRow, rowToAccepted, hchosen] using hcritical
    · rfl
    · rfl
    · simpa [acceptedToRow, rowToAccepted, hchosen, hcritical]
    · simpa [acceptedToRow, rowToAccepted, hchosen, hcritical]

noncomputable def winningSampleEquivRow :
    WinningSample (rounds := rounds) S A ≃
      FlatSuccessfulRow (rounds := rounds) S A :=
  (acceptedSampleEquivWinning (rounds := rounds) S A).symm.trans
    (acceptedSampleEquivRow (rounds := rounds) S A)

noncomputable def rowSigmaEquiv :
    FlatSuccessfulRow (rounds := rounds) S A ≃
      Σ i : Fin (rounds + 1),
        Σ ctx : (ZMod ell × Seed) × AnswerTape (ZMod ell) rounds,
          {c : ZMod ell // GoodAt S A i ctx c} where
  toFun r := ⟨r.critical, ⟨((r.secret, r.seed), r.context), ⟨r.challenge, r.good⟩⟩⟩
  invFun r :=
    { critical := r.1
      secret := r.2.1.1.1
      seed := r.2.1.1.2
      context := r.2.1.2
      challenge := r.2.2.1
      good := r.2.2.2 }
  left_inv := by intro r; cases r; rfl
  right_inv := by intro r; rcases r with ⟨i, ctx, c⟩; rcases ctx with ⟨ctx, c⟩; rfl

noncomputable instance :
    Fintype (FlatSuccessfulRow (rounds := rounds) S A) :=
  Fintype.ofEquiv _ (rowSigmaEquiv (rounds := rounds) S A).symm

noncomputable def ordinaryGameSuccessMass : Nat := by
  classical
  exact Fintype.card (WinningSample (rounds := rounds) S A)

noncomputable def ordinaryGameSuccessProbability : ℚ :=
  ordinaryGameSuccessMass (rounds := rounds) S A /
    sampleMass (rounds := rounds) S A

theorem cast_row_card_eq_successMass :
    (Fintype.card (FlatSuccessfulRow (rounds := rounds) S A) : ℚ) =
      successMass (rounds := rounds) S A := by
  classical
  rw [Fintype.card_congr (rowSigmaEquiv (rounds := rounds) S A)]
  simp [successMass, rowMasses, rowMass]

theorem cast_ordinaryGameSuccessMass_eq_successMass :
    (ordinaryGameSuccessMass (rounds := rounds) S A : ℚ) =
      successMass (rounds := rounds) S A := by
  classical
  rw [ordinaryGameSuccessMass]
  rw [Fintype.card_congr (winningSampleEquivRow (rounds := rounds) S A)]
  exact cast_row_card_eq_successMass (rounds := rounds) S A

theorem ordinaryGameSuccessProbability_eq_successProbability :
    ordinaryGameSuccessProbability (rounds := rounds) S A =
      successProbability (rounds := rounds) S A := by
  unfold ordinaryGameSuccessProbability successProbability
  rw [cast_ordinaryGameSuccessMass_eq_successMass (rounds := rounds) S A]

theorem ordinary_game_forking_bound_closed
    [Nonempty Seed] [Fact (Nat.Prime ell)] :
    ordinaryGameSuccessProbability (rounds := rounds) S A ^ 2 ≤
      (rounds + 1 : ℚ) * forkingProbability (rounds := rounds) S A +
      (rounds + 1 : ℚ) * ordinaryGameSuccessProbability (rounds := rounds) S A /
        Fintype.card (ZMod ell) := by
  rw [ordinaryGameSuccessProbability_eq_successProbability (rounds := rounds) S A]
  exact forking_probability_bound_tight (rounds := rounds) S A

end Bijection

end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
