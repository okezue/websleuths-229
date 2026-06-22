import AtlasCheck.AdaptiveData

namespace Atlas
namespace Probability
namespace AdaptiveSchnorrEUFCMA
namespace AnswerTape

variable {R : Type u}

def removeAt : {n : Nat} → Fin (n + 1) → AnswerTape R (n + 1) → AnswerTape R n
  | 0, _, (_, u) => u
  | n + 1, i, (r, rs) => Fin.cases rs (fun j => (r, removeAt j rs)) i

@[simp] theorem removeAt_insert : ∀ {n : Nat} (i : Fin (n + 1))
    (ctx : AnswerTape R n) (r : R), removeAt i (insert i ctx r) = ctx
  | 0, i, ctx, r => by fin_cases i <;> cases ctx <;> rfl
  | n + 1, i, ctx, r => by
      refine Fin.cases ?_ (fun j => ?_) i
      · rfl
      · simp [removeAt, insert, removeAt_insert j ctx.2 r]

@[simp] theorem insert_removeAt_get : ∀ {n : Nat} (i : Fin (n + 1))
    (tape : AnswerTape R (n + 1)), insert i (removeAt i tape) (get tape i) = tape
  | 0, i, tape => by
      fin_cases i
      rcases tape with ⟨r, u⟩
      cases u
      rfl
  | n + 1, i, tape => by
      rcases tape with ⟨r, rs⟩
      refine Fin.cases ?_ (fun j => ?_) i
      · rfl
      · simp [removeAt, insert, get, insert_removeAt_get j rs]

end AnswerTape
end AdaptiveSchnorrEUFCMA
end Probability
end Atlas
