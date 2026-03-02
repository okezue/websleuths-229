from __future__ import annotations
import sqlite3,json,shutil,os
from pathlib import Path
from datetime import datetime
from hashlib import sha256
from wm.types import VersionInfo
from wm.cfg import RegCfg

class ModelRegistry:
    def __init__(self,cfg:RegCfg):
        self._cfg=cfg
        self._dir=Path(cfg.local_dir)
        self._dir.mkdir(parents=True,exist_ok=True)
        self._db=sqlite3.connect(str(self._dir/"registry.db"))
        self._db.execute("CREATE TABLE IF NOT EXISTS versions("
            "vid TEXT PRIMARY KEY,parent TEXT,ts TEXT,metrics TEXT,path TEXT)")
        self._db.commit()
    def push(self,src_path:str,metrics:dict[str,float]|None=None,parent:str|None=None)->VersionInfo:
        ts=datetime.utcnow()
        vid=sha256(f"{src_path}{ts.isoformat()}".encode()).hexdigest()[:12]
        dst=str(self._dir/"models"/vid)
        if os.path.isdir(src_path):
            shutil.copytree(src_path,dst)
        else:
            os.makedirs(dst,exist_ok=True)
            shutil.copy2(src_path,dst)
        m=metrics or {}
        v=VersionInfo(vid=vid,parent=parent,ts=ts,metrics=m,path=dst)
        self._db.execute("INSERT INTO versions(vid,parent,ts,metrics,path) VALUES(?,?,?,?,?)",
            (vid,parent,ts.isoformat(),json.dumps(m),dst))
        self._db.commit()
        return v
    def pull(self,vid:str)->VersionInfo|None:
        r=self._db.execute("SELECT vid,parent,ts,metrics,path FROM versions WHERE vid=?",(vid,)).fetchone()
        if not r:return None
        return VersionInfo(vid=r[0],parent=r[1],ts=datetime.fromisoformat(r[2]),
            metrics=json.loads(r[3]),path=r[4])
    def latest(self)->VersionInfo|None:
        r=self._db.execute("SELECT vid,parent,ts,metrics,path FROM versions ORDER BY ts DESC LIMIT 1").fetchone()
        if not r:return None
        return VersionInfo(vid=r[0],parent=r[1],ts=datetime.fromisoformat(r[2]),
            metrics=json.loads(r[3]),path=r[4])
    def list_versions(self)->list[VersionInfo]:
        rows=self._db.execute("SELECT vid,parent,ts,metrics,path FROM versions ORDER BY ts").fetchall()
        return [VersionInfo(vid=r[0],parent=r[1],ts=datetime.fromisoformat(r[2]),
            metrics=json.loads(r[3]),path=r[4]) for r in rows]
    def rollback(self,vid:str)->VersionInfo|None:
        v=self.pull(vid)
        if not v:return None
        return self.push(v.path,v.metrics,parent=v.vid)
    def close(self):
        self._db.close()
