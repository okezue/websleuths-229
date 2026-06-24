import Mathlib

namespace Atlas
namespace Probability

def AnswerTape (R : Type u) : Nat → Type u
  | 0 => PUnit
  | n + 1 => R × AnswerTape R n

namespace AnswerTape

variable {R : Type u}

def get : {n : Nat} → AnswerTape R n → Fin n → R
  | 0, _, i => Fin.elim0 i
  | _ + 1, (r, rs), i => Fin.cases r (fun j => get rs j) i

def insert : {n : Nat} → Fin (n + 1) → AnswerTape R n → R → AnswerTape R (n + 1)
  | 0, _, _, r => (r, PUnit.unit)
  | n + 1, i, ctx, r => Fin.cases (r, ctx) (fun j => (ctx.1, insert j ctx.2 r)) i

@[simp] theorem get_insert_same : ∀ {n : Nat} (i : Fin (n + 1))
    (ctx : AnswerTape R n) (r : R), get (insert i ctx r) i = r
  | 0, i, ctx, r => by fin_cases i <;> rfl
  | n + 1, i, ctx, r => by
      refine Fin.cases ?_ (fun j => ?_) i
      · rfl
      · exact get_insert_same j ctx.2 r

noncomputable def fintype [Fintype R] : (n : Nat) → Fintype (AnswerTape R n)
  | 0 => inferInstanceAs (Fintype PUnit)
  | n + 1 =>
      letI : Fintype (AnswerTape R n) := fintype n
      inferInstanceAs (Fintype (R × AnswerTape R n))

noncomputable instance [Fintype R] (n : Nat) : Fintype (AnswerTape R n) := fintype n

@[simp] theorem card [Fintype R] : ∀ n : Nat,
    Fintype.card (AnswerTape R n) = Fintype.card R ^ n
  | 0 => by simp [AnswerTape]
  | n + 1 => by
      letI : Fintype (AnswerTape R n) := fintype n
      simp [AnswerTape, card n, pow_succ, Nat.mul_comm]

end AnswerTape
end Probability
end Atlas

namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA

universe u v w

structure ScalarGroup (ell : Nat) [NeZero ell] where
  G : Type u
  instAddCommGroup : AddCommGroup G
  toPoint : ZMod ell →+ G
  toPoint_injective : Function.Injective toPoint
  act : ZMod ell → G → G
  act_toPoint : ∀ c d : ZMod ell, act c (toPoint d) = toPoint (c * d)

attribute [instance] ScalarGroup.instAddCommGroup

structure ChallengeQuery (G : Type u) (Msg : Type v) where
  publicKey : G
  commitment : G
  message : Msg
  deriving DecidableEq

structure CandidateForgery (G : Type u) (Msg : Type v) (ell : Nat) [NeZero ell] where
  commitment : G
  response : ZMod ell
  message : Msg
  critical : Nat

structure Forgery (G : Type u) (Msg : Type v) (ell queries : Nat) [NeZero ell] where
  commitment : G
  response : ZMod ell
  message : Msg
  critical : Fin queries

structure Adversary (G : Type u) (Msg : Type v) (ell : Nat) [NeZero ell] (Seed : Type w) where
  State : Type*
  init : G → Seed → State
  query : Nat → State → ChallengeQuery G Msg
  absorb : Nat → State → ChallengeQuery G Msg → ZMod ell → State
  finish : State → Option (CandidateForgery G Msg ell)
  signed : State → Finset Msg

structure ExecState {G : Type u} {Msg : Type v} {ell : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed) where
  adv : A.State
  table : List (ChallengeQuery G Msg × ZMod ell)

namespace ExecState

variable {G : Type u} {Msg : Type v} {ell : Nat} [NeZero ell]
  {Seed : Type w} (A : Adversary G Msg ell Seed)

def lookup [DecidableEq G] [DecidableEq Msg] (st : ExecState A)
    (q : ChallengeQuery G Msg) : Option (ZMod ell) :=
  (st.table.find? fun qr => decide (qr.1 = q)).map Prod.snd

def step [DecidableEq G] [DecidableEq Msg] (round : Nat)
    (candidate : ZMod ell) (st : ExecState A) : ExecState A :=
  let q := A.query round st.adv
  match st.lookup A q with
  | some c => { adv := A.absorb round st.adv q c, table := st.table }
  | none => { adv := A.absorb round st.adv q candidate, table := (q, candidate) :: st.table }

end ExecState

def runPrefix {G : Type u} {Msg : Type v} {ell : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) {queries : Nat}
    (tape : AnswerTape (ZMod ell) queries) : (n : Nat) → n ≤ queries → ExecState A
  | 0, _ => { adv := A.init pk seed, table := [] }
  | n + 1, hn =>
      let prev := runPrefix A pk seed tape n (Nat.le_of_succ_le hn)
      let i : Fin queries := ⟨n, Nat.lt_of_succ_le hn⟩
      ExecState.step A n (AnswerTape.get tape i) prev

def run {G : Type u} {Msg : Type v} {ell : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) {queries : Nat}
    (tape : AnswerTape (ZMod ell) queries) : ExecState A :=
  runPrefix A pk seed tape queries le_rfl

def queryAt {G : Type u} {Msg : Type v} {ell : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) {queries : Nat}
    (tape : AnswerTape (ZMod ell) queries) (i : Fin queries) : ChallengeQuery G Msg :=
  A.query i.val (runPrefix A pk seed tape i.val (Nat.le_of_lt i.isLt)).adv

def freshAt {G : Type u} {Msg : Type v} {ell : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) {queries : Nat}
    (tape : AnswerTape (ZMod ell) queries) (i : Fin queries) : Bool :=
  let pre := runPrefix A pk seed tape i.val (Nat.le_of_lt i.isLt)
  decide (pre.lookup A (A.query i.val pre.adv)).isNone

def answerAt {G : Type u} {Msg : Type v} {ell : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) {queries : Nat}
    (tape : AnswerTape (ZMod ell) queries) (i : Fin queries) : ZMod ell :=
  let pre := runPrefix A pk seed tape i.val (Nat.le_of_lt i.isLt)
  let q := A.query i.val pre.adv
  (pre.lookup A q).getD (AnswerTape.get tape i)

def finishForgery {G : Type u} {Msg : Type v} {ell queries : Nat}
    [NeZero ell] {Seed : Type w} (A : Adversary G Msg ell Seed)
    (st : A.State) : Option (Forgery G Msg ell queries) :=
  match A.finish st with
  | none => none
  | some out => if h : out.critical < queries then
      some { commitment := out.commitment, response := out.response,
             message := out.message, critical := ⟨out.critical, h⟩ }
    else none

end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
