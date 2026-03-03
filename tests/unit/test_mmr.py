from wm.search.mmr import mmr_select

DOCS=[
    "Financial markets experienced significant volatility during the quarter.",
    "Stock prices fell sharply due to rising interest rates.",
    "The Pacific Ocean is the largest body of water on Earth.",
    "Climate change affects global weather patterns significantly.",
    "Earnings reports showed strong growth in technology sector.",
]

def test_mmr_basic():
    sel=mmr_select(DOCS,"financial markets",k=3,lam=0.7)
    assert len(sel)==3
    assert all(0<=i<len(DOCS) for i in sel)

def test_mmr_diversity():
    sel=mmr_select(DOCS,"financial markets",k=3,lam=0.0)
    texts=[DOCS[i] for i in sel]
    assert len(set(texts))==3

def test_mmr_relevance():
    sel=mmr_select(DOCS,"financial markets",k=1,lam=1.0)
    assert DOCS[sel[0]].startswith("Financial")

def test_mmr_empty():
    assert mmr_select([],"query")==[]

def test_mmr_k_larger_than_docs():
    sel=mmr_select(DOCS,"test",k=100)
    assert len(sel)==len(DOCS)

def test_mmr_unique_indices():
    sel=mmr_select(DOCS,"test",k=5)
    assert len(sel)==len(set(sel))
