from wm.core.schema import ClaimRecord, EvidenceSpan, Locator, QAItem


def test_stable_ids():
    left = ClaimRecord.build(subject="Alpha", predicate="is", object="Beta")
    right = ClaimRecord.build(subject=" alpha ", predicate=" IS ", object=" beta ")
    assert left.claim_id == right.claim_id


def test_span_locator_changes_id():
    a = EvidenceSpan.build(doc_id="d", modality="html", text="hello", locator=Locator(char_start=0))
    b = EvidenceSpan.build(doc_id="d", modality="html", text="hello", locator=Locator(char_start=1))
    assert a.span_id != b.span_id


def test_qa_id_is_stable():
    a = QAItem.build(prompt="Q", answer="A", kind="direct", split="train")
    b = QAItem.build(prompt="Q", answer="A", kind="direct", split="train")
    assert a.qid == b.qid
