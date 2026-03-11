from __future__ import annotations
import sqlite3,json
from pathlib import Path
from datetime import datetime
from wm.types import Episode

class EpisodeStore:
    def __init__(self,db_path:str|Path):
        self._db=sqlite3.connect(str(db_path))
        self._db.execute("CREATE TABLE IF NOT EXISTS episodes("
            "eid TEXT PRIMARY KEY,url TEXT UNIQUE,title TEXT,"
            "body TEXT,ts TEXT,meta TEXT,authority REAL DEFAULT 0.0,topic TEXT DEFAULT '')")
        self._db.commit()
    def put(self,ep:Episode)->bool:
        try:
            self._db.execute(
                "INSERT INTO episodes(eid,url,title,body,ts,meta,authority,topic) VALUES(?,?,?,?,?,?,?,?)",
                (ep.eid,ep.url,ep.title,ep.body,ep.ts.isoformat(),json.dumps(ep.meta),ep.authority,ep.topic))
            self._db.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    def get(self,eid:str)->Episode|None:
        r=self._db.execute("SELECT url,title,body,ts,meta,authority,topic FROM episodes WHERE eid=?",(eid,)).fetchone()
        if not r:return None
        return Episode(url=r[0],title=r[1],body=r[2],
            ts=datetime.fromisoformat(r[3]),meta=json.loads(r[4]),authority=r[5],topic=r[6])
    def list_eids(self)->list[str]:
        return [r[0] for r in self._db.execute("SELECT eid FROM episodes").fetchall()]
    def count(self)->int:
        return self._db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    def put_many(self,eps:list[Episode])->int:
        n=0
        for e in eps:
            if self.put(e):n+=1
        return n
    def all(self)->list[Episode]:
        rows=self._db.execute("SELECT url,title,body,ts,meta,authority,topic FROM episodes").fetchall()
        return [Episode(url=r[0],title=r[1],body=r[2],
            ts=datetime.fromisoformat(r[3]),meta=json.loads(r[4]),authority=r[5],topic=r[6]) for r in rows]
    def close(self):
        self._db.close()
