from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from wm.core.io import ensure_dir
from wm.core.schema import ClaimRecord, DocumentRecord, EvidenceEpisode, EvidenceSpan, QAItem


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class EvidenceStore:
    """SQLite store for immutable source records and versioned derived evidence."""

    def __init__(self, path: str | Path):
        path = Path(path)
        ensure_dir(path.parent)
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents(
                    doc_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    canonical_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    blob_path TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    parent_doc_id TEXT,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(canonical_url);
                CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);

                CREATE TABLE IF NOT EXISTS spans(
                    span_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    modality TEXT NOT NULL,
                    text TEXT NOT NULL,
                    locator TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    parent_span_id TEXT,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_spans_doc ON spans(doc_id);
                CREATE INDEX IF NOT EXISTS idx_spans_hash ON spans(content_hash);

                CREATE TABLE IF NOT EXISTS claims(
                    claim_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    text TEXT NOT NULL,
                    support_span_ids TEXT NOT NULL,
                    contradicts TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    status TEXT NOT NULL,
                    source_domains TEXT NOT NULL,
                    qualifiers TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_claim_spo ON claims(subject, predicate);
                CREATE INDEX IF NOT EXISTS idx_claim_status ON claims(status);

                CREATE TABLE IF NOT EXISTS edges(
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY(src, dst, relation)
                );

                CREATE TABLE IF NOT EXISTS episodes(
                    episode_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    document_ids TEXT NOT NULL,
                    span_ids TEXT NOT NULL,
                    claim_ids TEXT NOT NULL,
                    qa_items TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(domain);

                CREATE TABLE IF NOT EXISTS kv(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            try:
                self._db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS spans_fts USING fts5(span_id UNINDEXED, text, tokenize='unicode61')"
                )
                self._db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(doc_id UNINDEXED, title, url, tokenize='unicode61')"
                )
                self._fts = True
            except sqlite3.OperationalError:
                self._fts = False

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def put_document(self, doc: DocumentRecord) -> None:
        data = doc.model_dump(mode="json")
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id,url,canonical_url,title,mime_type,fetched_at,content_hash,blob_path,status_code,parent_doc_id,metadata)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    doc.doc_id,
                    doc.url,
                    doc.canonical_url,
                    doc.title,
                    doc.mime_type,
                    str(data["fetched_at"]),
                    doc.content_hash,
                    doc.blob_path,
                    doc.status_code,
                    doc.parent_doc_id,
                    _json(doc.metadata),
                ),
            )
            if self._fts:
                self._db.execute("DELETE FROM documents_fts WHERE doc_id=?", (doc.doc_id,))
                self._db.execute(
                    "INSERT INTO documents_fts(doc_id,title,url) VALUES(?,?,?)",
                    (doc.doc_id, doc.title, doc.canonical_url),
                )

    def put_span(self, span: EvidenceSpan) -> None:
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT OR REPLACE INTO spans
                (span_id,doc_id,modality,text,locator,content_hash,parent_span_id,metadata)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    span.span_id,
                    span.doc_id,
                    span.modality,
                    span.text,
                    _json(span.locator.model_dump(mode="json")),
                    span.content_hash,
                    span.parent_span_id,
                    _json(span.metadata),
                ),
            )
            if self._fts:
                self._db.execute("DELETE FROM spans_fts WHERE span_id=?", (span.span_id,))
                self._db.execute("INSERT INTO spans_fts(span_id,text) VALUES(?,?)", (span.span_id, span.text))

    def put_spans(self, spans: Iterable[EvidenceSpan]) -> None:
        for span in spans:
            self.put_span(span)

    def put_claim(self, claim: ClaimRecord) -> None:
        data = claim.model_dump(mode="json")
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT OR REPLACE INTO claims
                (claim_id,subject,predicate,object,text,support_span_ids,contradicts,confidence,valid_from,valid_to,status,source_domains,qualifiers)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    claim.claim_id,
                    claim.subject,
                    claim.predicate,
                    claim.object,
                    claim.text,
                    _json(claim.support_span_ids),
                    _json(claim.contradicts),
                    claim.confidence,
                    data.get("valid_from"),
                    data.get("valid_to"),
                    claim.status,
                    _json(claim.source_domains),
                    _json(claim.qualifiers),
                ),
            )

    def put_claims(self, claims: Iterable[ClaimRecord]) -> None:
        for claim in claims:
            self.put_claim(claim)

    def put_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO edges(src,dst,relation,weight,metadata) VALUES(?,?,?,?,?)",
                (src, dst, relation, weight, _json(metadata or {})),
            )

    def put_episode(self, episode: EvidenceEpisode) -> None:
        data = episode.model_dump(mode="json")
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT OR REPLACE INTO episodes
                (episode_id,topic,domain,document_ids,span_ids,claim_ids,qa_items,created_at,metadata)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    episode.episode_id,
                    episode.topic,
                    episode.domain,
                    _json(episode.document_ids),
                    _json(episode.span_ids),
                    _json(episode.claim_ids),
                    _json([q.model_dump(mode="json") for q in episode.qa_items]),
                    str(data["created_at"]),
                    _json(episode.metadata),
                ),
            )

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self._db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not row:
            return None
        return DocumentRecord.model_validate(
            {
                **dict(row),
                "metadata": json.loads(row["metadata"]),
            }
        )

    def list_documents(self) -> list[DocumentRecord]:
        rows = self._db.execute("SELECT doc_id FROM documents ORDER BY fetched_at, doc_id").fetchall()
        return [doc for row in rows if (doc := self.get_document(row["doc_id"])) is not None]

    def find_document_by_url(self, url: str) -> DocumentRecord | None:
        row = self._db.execute(
            "SELECT * FROM documents WHERE canonical_url=? ORDER BY fetched_at DESC LIMIT 1", (url,)
        ).fetchone()
        return self.get_document(row["doc_id"]) if row else None

    def get_span(self, span_id: str) -> EvidenceSpan | None:
        row = self._db.execute("SELECT * FROM spans WHERE span_id=?", (span_id,)).fetchone()
        if not row:
            return None
        return EvidenceSpan.model_validate(
            {
                **dict(row),
                "locator": json.loads(row["locator"]),
                "metadata": json.loads(row["metadata"]),
            }
        )

    def get_spans(self, doc_id: str | None = None) -> list[EvidenceSpan]:
        query = "SELECT span_id FROM spans"
        params: tuple[Any, ...] = ()
        if doc_id:
            query += " WHERE doc_id=?"
            params = (doc_id,)
        rows = self._db.execute(query, params).fetchall()
        return [span for row in rows if (span := self.get_span(row["span_id"]))]

    def get_claim(self, claim_id: str) -> ClaimRecord | None:
        row = self._db.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        for field in ("support_span_ids", "contradicts", "source_domains", "qualifiers"):
            data[field] = json.loads(data[field])
        return ClaimRecord.model_validate(data)

    def get_claims(self, status: str | None = None) -> list[ClaimRecord]:
        query = "SELECT claim_id FROM claims"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        rows = self._db.execute(query, params).fetchall()
        return [claim for row in rows if (claim := self.get_claim(row["claim_id"]))]

    def get_episode(self, episode_id: str) -> EvidenceEpisode | None:
        row = self._db.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        for field in ("document_ids", "span_ids", "claim_ids", "qa_items", "metadata"):
            data[field] = json.loads(data[field])
        data["qa_items"] = [QAItem.model_validate(item) for item in data["qa_items"]]
        return EvidenceEpisode.model_validate(data)

    def list_episodes(self, domain: str | None = None) -> list[EvidenceEpisode]:
        query = "SELECT episode_id FROM episodes"
        params: tuple[Any, ...] = ()
        if domain:
            query += " WHERE domain=?"
            params = (domain,)
        query += " ORDER BY created_at"
        rows = self._db.execute(query, params).fetchall()
        return [ep for row in rows if (ep := self.get_episode(row["episode_id"]))]

    def search_spans(self, query: str, limit: int = 20) -> list[tuple[EvidenceSpan, float]]:
        if self._fts:
            safe_query = " ".join(token.replace('"', '') for token in query.split() if token.strip())
            try:
                rows = self._db.execute(
                    "SELECT span_id, bm25(spans_fts) AS score FROM spans_fts WHERE spans_fts MATCH ? ORDER BY score LIMIT ?",
                    (safe_query, limit),
                ).fetchall()
                return [
                    (span, float(-row["score"]))
                    for row in rows
                    if (span := self.get_span(row["span_id"])) is not None
                ]
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        rows = self._db.execute("SELECT span_id FROM spans WHERE text LIKE ? LIMIT ?", (like, limit)).fetchall()
        return [(span, 1.0) for row in rows if (span := self.get_span(row["span_id"]))]

    def search_documents(self, query: str, limit: int = 20) -> list[tuple[DocumentRecord, float]]:
        if self._fts:
            safe_query = " ".join(token.replace('"', '') for token in query.split() if token.strip())
            try:
                rows = self._db.execute(
                    "SELECT doc_id, bm25(documents_fts) AS score FROM documents_fts WHERE documents_fts MATCH ? ORDER BY score LIMIT ?",
                    (safe_query, limit),
                ).fetchall()
                return [
                    (doc, float(-row["score"]))
                    for row in rows
                    if (doc := self.get_document(row["doc_id"])) is not None
                ]
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        rows = self._db.execute(
            "SELECT doc_id FROM documents WHERE title LIKE ? OR canonical_url LIKE ? LIMIT ?",
            (like, like, limit),
        ).fetchall()
        return [(doc, 1.0) for row in rows if (doc := self.get_document(row["doc_id"]))]

    def edges_from(self, src: str, relation: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM edges WHERE src=?"
        params: tuple[Any, ...] = (src,)
        if relation:
            query += " AND relation=?"
            params += (relation,)
        return [
            {**dict(row), "metadata": json.loads(row["metadata"])}
            for row in self._db.execute(query, params).fetchall()
        ]

    def set_kv(self, key: str, value: Any) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO kv(key,value) VALUES(?,?)", (key, _json(value)))

    def get_kv(self, key: str, default: Any = None) -> Any:
        row = self._db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def counts(self) -> dict[str, int]:
        return {
            table: int(self._db.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
            for table in ("documents", "spans", "claims", "edges", "episodes")
        }
