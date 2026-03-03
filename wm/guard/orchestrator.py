from __future__ import annotations
import copy,torch
from wm.types import GuardResult,Chunk
from wm.cfg import GuardCfg
from wm.eval.probes import ProbeBuilder
from wm.eval.anchor import AnchorEval
from wm.eval.core import Evaluator
from wm.guard.gain import expected_gain
from wm.guard.drift import drift_kl
from wm.guard.stats import paired_bootstrap_ci

class UpdateGuard:
    def __init__(self,cfg:GuardCfg,tok,anchors:list[str]|None=None):
        self._cfg=cfg
        self._tok=tok
        self._anchors=anchors
    def guard(self,model,chunks:list[Chunk],
              recipe_fn,dataset,dream_prompts:list[str],
              g_k:torch.Tensor|None=None,
              g_a:torch.Tensor|None=None)->GuardResult:
        snap={k:v.clone() for k,v in model.state_dict().items()}
        pb=ProbeBuilder()
        probes=pb.build_all(chunks)
        ev=Evaluator(model,self._tok,max_len=128)
        pre_scores=[ev._probe_score(p) for p in probes] if probes else []
        anc=AnchorEval(model,self._tok,self._anchors)
        pre_nll=anc.nll()
        u_exp=0.0
        if g_k is not None and g_a is not None:
            u_exp=expected_gain(g_k,g_a,alpha=0.5)
            if u_exp<self._cfg.u_exp_thresh:
                model.load_state_dict(snap)
                return GuardResult(accepted=False,u_exp=u_exp,
                    reason="u_exp below threshold",phase="pre_ft")
        recipe_fn(model,dataset,dream_prompts)
        post_scores=[ev._probe_score(p) for p in probes] if probes else []
        post_nll=anc.nll()
        ci_lo,ci_hi=0.0,0.0
        if pre_scores and post_scores:
            ci_lo,ci_hi=paired_bootstrap_ci(
                pre_scores,post_scores,
                n_boot=self._cfg.n_bootstrap,alpha=self._cfg.ci_alpha)
            if ci_lo<=0:
                model.load_state_dict(snap)
                return GuardResult(accepted=False,u_exp=u_exp,
                    ci_lo=ci_lo,ci_hi=ci_hi,
                    reason="probe CI_lo <= 0",phase="post_ft")
        anchor_d=post_nll-pre_nll
        if anchor_d>self._cfg.max_anchor_delta:
            model.load_state_dict(snap)
            return GuardResult(accepted=False,u_exp=u_exp,
                ci_lo=ci_lo,ci_hi=ci_hi,anchor_delta=anchor_d,
                reason="anchor regression too large",phase="post_ft")
        u_act=sum(post_scores)/max(len(post_scores),1)-sum(pre_scores)/max(len(pre_scores),1) if pre_scores else 0.0
        dk=0.0
        return GuardResult(
            accepted=True,u_exp=u_exp,u_act=u_act,
            ci_lo=ci_lo,ci_hi=ci_hi,
            anchor_delta=anchor_d,drift_kl=dk,
            reason="accepted",phase="done")
