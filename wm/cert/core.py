from __future__ import annotations
from wm.types import EvalReport,CheckResult,CertResult
from wm.cfg import CertCfg
from wm.guard.stats import paired_bootstrap_ci,sprt_test

class Certifier:
    def __init__(self,cfg:CertCfg):
        self._cfg=cfg
    def certify(self,cur:EvalReport,base:EvalReport|None=None)->CertResult:
        checks=[]
        checks.append(CheckResult(
            name="min_acc",passed=cur.acc>=self._cfg.min_acc,
            val=cur.acc,thresh=self._cfg.min_acc,
        ))
        if base is not None:
            delta=cur.ppl-base.ppl
            checks.append(CheckResult(
                name="ppl_delta",passed=delta<=self._cfg.max_ppl_delta,
                val=delta,thresh=self._cfg.max_ppl_delta,
            ))
            drift=abs(cur.acc-base.acc)
            checks.append(CheckResult(
                name="drift",passed=drift<=self._cfg.max_drift,
                val=drift,thresh=self._cfg.max_drift,
            ))
        return CertResult(passed=all(c.passed for c in checks),checks=checks)
    def certify_paired(self,pre_scores:list[float],post_scores:list[float],
                       pre_anc:list[float],post_anc:list[float])->CertResult:
        checks=[]
        ci_lo,ci_hi=paired_bootstrap_ci(pre_scores,post_scores,
            n_boot=self._cfg.n_bootstrap,alpha=self._cfg.ci_alpha)
        checks.append(CheckResult(
            name="probe_ci_lo",passed=ci_lo>0,val=ci_lo,thresh=0.0))
        anc_lo,anc_hi=paired_bootstrap_ci(pre_anc,post_anc,
            n_boot=self._cfg.n_bootstrap,alpha=self._cfg.ci_alpha)
        checks.append(CheckResult(
            name="anchor_ci_hi",passed=anc_hi<=self._cfg.max_drift,
            val=anc_hi,thresh=self._cfg.max_drift))
        verdict=sprt_test(pre_scores,post_scores,
            delta=self._cfg.sprt_delta,alpha=self._cfg.sprt_alpha,
            beta=self._cfg.sprt_beta)
        checks.append(CheckResult(
            name="sprt",passed=verdict=="accept",
            val=1.0 if verdict=="accept" else 0.0,thresh=1.0))
        return CertResult(passed=all(c.passed for c in checks),checks=checks)
