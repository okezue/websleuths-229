from wm.storage.blob import LocalBlobStore


def test_blob_store_is_content_addressed(tmp_path):
    store = LocalBlobStore(tmp_path)
    first = store.put(b"hello")
    second = store.put(b"hello")
    assert first.content_hash == second.content_hash
    assert store.get(first.content_hash) == b"hello"
