from wm.core.schema import ClaimRecord, DocumentRecord, EvidenceEpisode, EvidenceSpan
from wm.evidence.store import EvidenceStore


def test_store_roundtrip(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    doc = DocumentRecord.build(url="https://example.com/a", canonical_url=None, content_hash="abc", blob_path="/tmp/a", title="Example", mime_type="text/html")
    store.put_document(doc)
    span = EvidenceSpan.build(doc_id=doc.doc_id, modality="html", text="Alpha is Beta.")
    store.put_span(span)
    claim = ClaimRecord.build(subject="Alpha", predicate="is", object="Beta", support_span_ids=[span.span_id], status="supported")
    store.put_claim(claim)
    episode = EvidenceEpisode.build(topic="alpha", domain="test", claim_ids=[claim.claim_id], document_ids=[doc.doc_id], span_ids=[span.span_id])
    store.put_episode(episode)
    assert store.get_document(doc.doc_id) == doc
    assert store.get_span(span.span_id) == span
    assert store.get_claim(claim.claim_id) == claim
    assert store.get_episode(episode.episode_id).claim_ids == [claim.claim_id]
    assert store.search_spans("Alpha")
    store.close()
