from __future__ import annotations
from typing import Any

try:
    from aim import Run
except Exception:
    Run=None


class AimLogger:
    def __init__(self,repo:str|None,exp:str,name:str,hp:dict,tags:list[str],run_hash:str|None=None):
        self.repo=repo
        self.exp=exp
        if Run is None or not repo:
            self.r=None
            return
        self.r=Run(run_hash=run_hash,repo=repo,experiment=exp,log_system_params=False,
                   system_tracking_interval=None,capture_terminal_logs=False,
                   force_resume=run_hash is not None)
        self.r.name=name
        try:
            self.r['hparams']=_jsonable(hp)
        except Exception:
            pass
        for t in tags:
            try:self.r.add_tag(t)
            except Exception:pass

    @property
    def hash(self)->str|None:
        return self.r.hash if self.r is not None else None

    def track(self,v:Any,name:str|None=None,step:int|None=None,epoch:int|None=None,context:dict|None=None)->None:
        if self.r is None:return
        ctx=context or {}
        try:
            if isinstance(v,dict):
                self.r.track(v,step=step,epoch=epoch,context=ctx)
            else:
                self.r.track(v,name=name,step=step,epoch=epoch,context=ctx)
        except Exception:
            pass

    def set_summary(self,k:str,v:Any)->None:
        if self.r is None:return
        try:self.r[k]=_jsonable(v)
        except Exception:pass

    def add_tag(self,t:str)->None:
        if self.r is None:return
        try:self.r.add_tag(t)
        except Exception:pass

    def close(self)->None:
        if self.r is None:return
        try:self.r.close()
        except Exception:pass


def _jsonable(x:Any)->Any:
    if isinstance(x,dict):return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_jsonable(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    try:return str(x)
    except Exception:return None
