import random
from wm.dream.noise import typo_inject,unicode_confuse,ws_noise,contradict,long_distractor,apply_random_noise

def test_typo_inject_changes():
    rng=random.Random(42)
    s="The quick brown fox jumps over the lazy dog"
    out=typo_inject(s,rate=0.2,rng=rng)
    assert out!=s
    assert len(out)>0

def test_typo_inject_deterministic():
    s="hello world test"
    a=typo_inject(s,rate=0.1,rng=random.Random(7))
    b=typo_inject(s,rate=0.1,rng=random.Random(7))
    assert a==b

def test_unicode_confuse_adds_non_ascii():
    rng=random.Random(42)
    s="a"*100+"e"*100
    out=unicode_confuse(s,rate=0.5,rng=rng)
    assert any(ord(c)>127 for c in out)

def test_ws_noise_returns_string():
    rng=random.Random(42)
    out=ws_noise("test input",rng=rng)
    assert isinstance(out,str)
    assert len(out)>0

def test_contradict_adds_prefix():
    rng=random.Random(42)
    s="The sky is blue."
    out=contradict(s,rng=rng)
    assert out.endswith(s)
    assert len(out)>len(s)

def test_long_distractor_prepends():
    rng=random.Random(42)
    s="short text"
    out=long_distractor(s,n=50,rng=rng)
    assert out.endswith(s)
    assert len(out)>len(s)+50

def test_apply_random_noise_deterministic():
    a=apply_random_noise("test string",rng=random.Random(99))
    b=apply_random_noise("test string",rng=random.Random(99))
    assert a==b

def test_apply_random_noise_changes():
    s="The Earth revolves around the Sun."
    out=apply_random_noise(s,rng=random.Random(42))
    assert isinstance(out,str)
    assert len(out)>0
