from __future__ import annotations
import sqlite3,json
from pathlib import Path
from wm.types import Claim,Entity,Community,SearchResult

class GraphStore:
    def __init__(self,db_path:str|Path):
        self._db=sqlite3.connect(str(db_path))
        self._init_tables()
    def _init_tables(self):
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS claims(
                cid TEXT PRIMARY KEY,text TEXT,eid TEXT,
                chunk_idx INT,entities TEXT,confidence REAL);
            CREATE TABLE IF NOT EXISTS entities(
                nid TEXT PRIMARY KEY,name TEXT UNIQUE);
            CREATE TABLE IF NOT EXISTS communities(
                coid TEXT PRIMARY KEY,label TEXT,members TEXT,
                centroid TEXT,parameterized INT DEFAULT 0);
            CREATE TABLE IF NOT EXISTS edges(
                src TEXT,dst TEXT,rel TEXT,weight REAL,
                PRIMARY KEY(src,dst,rel));
        """)
        self._db.commit()
    def put_claim(self,c:Claim)->bool:
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO claims(cid,text,eid,chunk_idx,entities,confidence) VALUES(?,?,?,?,?,?)",
                (c.cid,c.text,c.eid,c.chunk_idx,json.dumps(c.entities),c.confidence))
            self._db.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    def put_entity(self,e:Entity)->bool:
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO entities(nid,name) VALUES(?,?)",
                (e.nid,e.name))
            self._db.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    def put_community(self,co:Community)->bool:
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO communities(coid,label,members,centroid,parameterized) VALUES(?,?,?,?,?)",
                (co.coid,co.label,json.dumps(co.members),json.dumps(co.centroid),
                 int(co.parameterized)))
            self._db.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    def put_edge(self,src:str,dst:str,rel:str="co_occur",weight:float=1.0):
        self._db.execute(
            "INSERT OR REPLACE INTO edges(src,dst,rel,weight) VALUES(?,?,?,?)",
            (src,dst,rel,weight))
        self._db.commit()
    def get_claims(self)->list[Claim]:
        rows=self._db.execute("SELECT cid,text,eid,chunk_idx,entities,confidence FROM claims").fetchall()
        return [Claim(cid=r[0],text=r[1],eid=r[2],chunk_idx=r[3],
                       entities=json.loads(r[4]),confidence=r[5]) for r in rows]
    def get_entities(self)->list[Entity]:
        rows=self._db.execute("SELECT nid,name FROM entities").fetchall()
        return [Entity(nid=r[0],name=r[1]) for r in rows]
    def get_communities(self,only_param:bool=False)->list[Community]:
        q="SELECT coid,label,members,centroid,parameterized FROM communities"
        if only_param:q+=" WHERE parameterized=1"
        rows=self._db.execute(q).fetchall()
        return [Community(coid=r[0],label=r[1],members=json.loads(r[2]),
                          centroid=json.loads(r[3]),parameterized=bool(r[4])) for r in rows]
    def mark_parameterized(self,coid:str):
        self._db.execute("UPDATE communities SET parameterized=1 WHERE coid=?",(coid,))
        self._db.commit()
    def store_result(self,sr:SearchResult):
        for c in sr.claims:self.put_claim(c)
        for e in sr.entities:self.put_entity(e)
        for co in sr.communities:self.put_community(co)
        for c in sr.claims:
            for ename in c.entities:
                self.put_edge(c.cid,ename,rel="mentions")
    def count_claims(self)->int:
        return self._db.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    def close(self):
        self._db.close()
