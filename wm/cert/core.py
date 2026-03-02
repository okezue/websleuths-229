from __future__ import annotations
from wm.types import EvalReport,CheckResult,CertResult
from wm.cfg import CertCfg

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
