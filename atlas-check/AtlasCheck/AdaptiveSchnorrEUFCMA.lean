import Mathlib
namespace Atlas
namespace Probability
open scoped BigOperators
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
  | n + 1, i, ctx, r =>
      Fin.cases (r, ctx)
        (fun j =>
          let head := ctx.1
          let tail := ctx.2
          (head, insert j tail r)) i
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
      simp [AnswerTape, card n, pow_succ]
end AnswerTape
theorem list_sum_sq_le_length_mul_sum_sq (xs : List ℚ) :
    xs.sum ^ 2 ≤ (xs.length : ℚ) * (xs.map fun x => x ^ 2).sum := by
  induction xs with
  | nil => simp
  | cons a xs ih =>
      rcases xs with _ | b ys
      · simp
      · let tail : List ℚ := b :: ys
        let n : ℚ := tail.length
        let s : ℚ := tail.sum
        let q : ℚ := (tail.map fun x => x ^ 2).sum
        have hn : 0 < n := by
          dsimp [n, tail]
          positivity
        have hvar : 0 ≤ n * q - s ^ 2 := by
          dsimp [n, s, q, tail]
          nlinarith [ih]
        have hsq : 0 ≤ (n * a - s) ^ 2 := sq_nonneg (n * a - s)
        have hmul : 0 ≤ n * (n * a ^ 2 + q - 2 * a * s) := by
          nlinarith
        have hcross : 0 ≤ n * a ^ 2 + q - 2 * a * s := by
          nlinarith
        dsimp [tail, n, s, q] at hvar hcross ⊢
        simp only [List.sum_cons, List.length_cons, List.map_cons, Nat.cast_add,
          Nat.cast_one]
        nlinarith
end Probability
end Atlas
namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA
open scoped BigOperators
universe u v w
structure ScalarGroup (ell : Nat) [NeZero ell] where
  G : Type u
  instAddCommGroup : AddCommGroup G
  toPoint : ZMod ell →+ G
  toPoint_injective : Function.Injective toPoint
  act : ZMod ell → G → G
  act_toPoint : ∀ c d : ZMod ell, act c (toPoint d) = toPoint (c * d)
attribute [instance] ScalarGroup.instAddCommGroup
namespace ScalarGroup
variable {ell : Nat} [NeZero ell] (S : ScalarGroup ell)
@[simp] theorem act_public (c d : ZMod ell) :
    S.act c (S.toPoint d) = S.toPoint (c * d) :=
  S.act_toPoint c d
end ScalarGroup
structure ChallengeQuery (G : Type u) (Msg : Type v) where
  publicKey : G
  commitment : G
  message : Msg
  deriving DecidableEq
structure CandidateForgery (G : Type u) (Msg : Type v) (ell : Nat)
    [NeZero ell] where
  commitment : G
  response : ZMod ell
  message : Msg
  critical : Nat
structure Forgery (G : Type u) (Msg : Type v) (ell queries : Nat)
    [NeZero ell] where
  commitment : G
  response : ZMod ell
  message : Msg
  critical : Fin queries
structure Adversary (G : Type u) (Msg : Type v) (ell : Nat) [NeZero ell]
    (Seed : Type w) where
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
  | some c =>
      { adv := A.absorb round st.adv q c
        table := st.table }
  | none =>
      { adv := A.absorb round st.adv q candidate
        table := (q, candidate) :: st.table }
end ExecState
def runPrefix {G : Type u} {Msg : Type v} {ell : Nat} [NeZero ell]
    {Seed : Type w} (A : Adversary G Msg ell Seed)
    [DecidableEq G] [DecidableEq Msg]
    (pk : G) (seed : Seed) {queries : Nat}
    (tape : AnswerTape (ZMod ell) queries) :
    (n : Nat) → n ≤ queries → ExecState A
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
    (tape : AnswerTape (ZMod ell) queries) (i : Fin queries) :
    ChallengeQuery G Msg :=
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
    (st : A.State) : Option (Forgery G Msg ell queries) := do
  let out ← A.finish st
  let critical : Fin queries ←
    if h : out.critical < queries then some ⟨out.critical, h⟩ else none
  pure { commitment := out.commitment
         response := out.response
         message := out.message
         critical := critical }
theorem AnswerTape.get_insert_ne {R : Type*} {n : Nat}
    (i j : Fin (n + 1)) (ctx : AnswerTape R n) (r r' : R)
    (hji : j ≠ i) :
    AnswerTape.get (AnswerTape.insert i ctx r) j =
      AnswerTape.get (AnswerTape.insert i ctx r') j := by
  induction n with
  | zero => exact (hji (Subsingleton.elim _ _)).elim
  | succ n ih =>
      refine Fin.cases ?_ (fun j' => ?_) j
      · refine Fin.cases ?_ (fun i' => ?_) i
        · exact (hji rfl).elim
        · rfl
      · refine Fin.cases ?_ (fun i' => ?_) i
        · rfl
        · apply ih i' j' ctx.2 r r'
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
  unfold freshAt at hfresh
  simp only [decide_eq_true_eq] at hfresh
  unfold answerAt
  rw [Option.isNone_iff_eq_none.mp hfresh]
  simp
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
noncomputable def successMass : Nat := (rowMasses S A).sum
noncomputable def forkMass : Nat :=
  ((rowMasses S A).map fun s => s * (s - 1)).sum
noncomputable def sampleMass : Nat :=
  Fintype.card (ZMod ell × Seed) *
    Fintype.card (AnswerTape (ZMod ell) (rounds + 1))
noncomputable def forkSampleMass : Nat :=
  sampleMass S A * Fintype.card (ZMod ell)
noncomputable def successProbability : ℚ :=
  successMass S A / sampleMass S A
noncomputable def forkingProbability : ℚ :=
  forkMass S A / forkSampleMass S A
@[simp] theorem rowMasses_length :
    (rowMasses S A).length =
      (rounds + 1) * Fintype.card (ZMod ell × Seed) *
        Fintype.card (AnswerTape (ZMod ell) rounds) := by
  classical
  simp [rowMasses, Nat.mul_assoc]
private theorem pair_count_add_self (s : Nat) :
    s * (s - 1) + s = s * s := by omega
theorem successMass_sq_le_rows_mul_fork_add_success :
    (successMass S A : ℚ) ^ 2 ≤
      ((rowMasses S A).length : ℚ) *
        ((forkMass S A : ℚ) + successMass S A) := by
  let xs : List ℚ := (rowMasses S A).map (fun s => (s : ℚ))
  have hc := list_sum_sq_le_length_mul_sum_sq xs
  have hlen : xs.length = (rowMasses S A).length := by simp [xs]
  have hsum : xs.sum = (successMass S A : ℚ) := by
    simp [xs, successMass]
  have hsquares : (xs.map fun x => x ^ 2).sum =
      ((forkMass S A : ℚ) + successMass S A) := by
    simp only [xs, List.map_map, Function.comp_apply]
    rw [← Nat.cast_add, ← Nat.cast_sum, ← Nat.cast_sum]
    congr 1
    simp only [forkMass, successMass]
    induction rowMasses S A with
    | nil => simp
    | cons s ss ih =>
        simp only [List.map_cons, List.sum_cons]
        rw [pair_count_add_self]
        omega
  simpa [hlen, hsum, hsquares] using hc
theorem forking_probability_bound_tight
    [Nonempty Seed] [Fact (Nat.Prime ell)] :
    successProbability S A ^ 2 ≤
      (rounds + 1 : ℚ) * forkingProbability S A +
      (rounds + 1 : ℚ) * successProbability S A /
        Fintype.card (ZMod ell) := by
  have hcount := successMass_sq_le_rows_mul_fork_add_success S A
  rw [rowMasses_length] at hcount
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
    Option (ZMod ell) :=
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
  forkingProbability S A
theorem adaptive_forking_bound_closed
    {Msg : Type v} {Seed : Type w} {rounds : Nat}
    (S : ScalarGroup ell) [DecidableEq S.G] [DecidableEq Msg]
    (A : Adversary S.G Msg ell Seed)
    [Fintype Seed] [Nonempty Seed] [Fact (Nat.Prime ell)] :
    successProbability S A ^ 2 ≤
      (rounds + 1 : ℚ) * rewindingDLPAdvantage S A +
      (rounds + 1 : ℚ) * successProbability S A /
        Fintype.card (ZMod ell) := by
  simpa [rewindingDLPAdvantage] using forking_probability_bound_tight S A
end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
