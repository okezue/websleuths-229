import AtlasCheck.AdaptiveCausality

namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA

universe u v w

variable {ell : Nat} [NeZero ell]

def verifies (S : ScalarGroup ell) (pk R : S.G)
    (s c : ZMod ell) : Prop :=
  S.toPoint s = R + S.act c pk

structure AcceptedRun
    {Msg : Type v} {Seed : Type w} {rounds : Nat}
    (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
    (A : Adversary S.G Msg ell Seed)
    (pk : S.G) (seed : Seed)
    (tape : AnswerTape (ZMod ell) (rounds + 1))
    (f : Forgery S.G Msg ell (rounds + 1)) : Prop where
  finish_eq : finishForgery (queries := rounds + 1) A
    (run A pk seed tape).adv = some f
  fresh : freshAt A pk seed tape f.critical = true
  query_eq : queryAt A pk seed tape f.critical =
    { publicKey := pk, commitment := f.commitment, message := f.message }
  unsigned : f.message ∉ A.signed (run A pk seed tape).adv
  verifies_eq : verifies S pk f.commitment f.response
    (answerAt A pk seed tape f.critical)

def GoodAt
    {Msg : Type v} {Seed : Type w} {rounds : Nat}
    (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
    (A : Adversary S.G Msg ell Seed)
    (i : Fin (rounds + 1))
    (ctx : (ZMod ell × Seed) × AnswerTape (ZMod ell) rounds)
    (c : ZMod ell) : Prop :=
  ∃ f, AcceptedRun S A (S.toPoint ctx.1.1) ctx.1.2
      (AnswerTape.insert i ctx.2 c) f ∧ f.critical = i

end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
