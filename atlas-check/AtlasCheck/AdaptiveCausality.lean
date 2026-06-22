import AtlasCheck.AdaptiveData

namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA

universe u v w

theorem AnswerTape.get_insert_ne {R : Type*} {n : Nat}
    (i j : Fin (n + 1)) (ctx : AnswerTape R n) (r r' : R)
    (hji : j ≠ i) :
    AnswerTape.get (AnswerTape.insert i ctx r) j =
      AnswerTape.get (AnswerTape.insert i ctx r') j := by
  induction n with
  | zero =>
      have h : j = i := by
        apply Fin.ext
        omega
      exact (hji h).elim
  | succ n ih =>
      rcases ctx with ⟨head, tail⟩
      cases i using Fin.cases with
      | zero =>
          cases j using Fin.cases with
          | zero => exact (hji rfl).elim
          | succ j => rfl
      | succ i =>
          cases j using Fin.cases with
          | zero => rfl
          | succ j =>
              apply ih i j tail
              intro h
              apply hji
              exact Fin.succ_inj.mpr h

theorem runPrefix_insert_same
    {G : Type u} {Msg : Type v} {ell rounds : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) (i : Fin (rounds + 1))
    (ctx : AnswerTape (ZMod ell) rounds) (r r' : ZMod ell)
    (n : Nat) (hn : n ≤ i.val) :
    runPrefix A pk seed (AnswerTape.insert i ctx r) n
        (le_trans hn (Nat.le_of_lt i.isLt)) =
      runPrefix A pk seed (AnswerTape.insert i ctx r') n
        (le_trans hn (Nat.le_of_lt i.isLt)) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp only [runPrefix]
      have hnle : n ≤ i.val := Nat.le_trans (Nat.le_succ n) hn
      rw [ih hnle]
      have hj : (⟨n, Nat.lt_of_lt_of_le (Nat.lt_succ_self n)
          (le_trans hn (Nat.le_of_lt i.isLt))⟩ : Fin (rounds + 1)) ≠ i := by
        intro h
        have hv : n = i.val := congrArg Fin.val h
        omega
      rw [AnswerTape.get_insert_ne i _ ctx r r' hj]

@[simp] theorem queryAt_insert_same
    {G : Type u} {Msg : Type v} {ell rounds : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) (i : Fin (rounds + 1))
    (ctx : AnswerTape (ZMod ell) rounds) (r r' : ZMod ell) :
    queryAt A pk seed (AnswerTape.insert i ctx r) i =
      queryAt A pk seed (AnswerTape.insert i ctx r') i := by
  unfold queryAt
  rw [runPrefix_insert_same A pk seed i ctx r r' i.val le_rfl]

@[simp] theorem freshAt_insert_same
    {G : Type u} {Msg : Type v} {ell rounds : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) (i : Fin (rounds + 1))
    (ctx : AnswerTape (ZMod ell) rounds) (r r' : ZMod ell) :
    freshAt A pk seed (AnswerTape.insert i ctx r) i =
      freshAt A pk seed (AnswerTape.insert i ctx r') i := by
  unfold freshAt
  rw [runPrefix_insert_same A pk seed i ctx r r' i.val le_rfl]

@[simp] theorem answerAt_insert_of_fresh
    {G : Type u} {Msg : Type v} {ell rounds : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) (i : Fin (rounds + 1))
    (ctx : AnswerTape (ZMod ell) rounds) (r : ZMod ell)
    (hfresh : freshAt A pk seed (AnswerTape.insert i ctx r) i = true) :
    answerAt A pk seed (AnswerTape.insert i ctx r) i = r := by
  have hnone :
      (let pre := runPrefix A pk seed (AnswerTape.insert i ctx r) i.val
          (Nat.le_of_lt i.isLt)
       pre.lookup A (A.query i.val pre.adv)) = none := by
    apply Option.isNone_iff_eq_none.mp
    simpa [freshAt] using hfresh
  simp [answerAt, hnone]

end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
