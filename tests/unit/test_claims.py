from wm.search.claims import extract_claims,extract_entities

FORENSIC_TEXT=(
    "The FBI laboratory confirmed that the fingerprints belong to John Smith. "
    "DNA analysis was conducted on samples collected from the crime scene. "
    "Ballistic evidence shows the weapon was fired from approximately 30 meters. "
    "Forensic pathologist Dr. Sarah Chen determined the cause of death. "
    "The toxicology report indicates the presence of benzodiazepines. "
    "Witness testimony places the suspect at Oak Street at approximately 2300 hours. "
    "Digital forensics recovered 15000 deleted files from the hard drive. "
    "Blood spatter analysis is consistent with blunt force trauma."
)

def test_extract_claims_basic():
    claims=extract_claims(FORENSIC_TEXT,eid="e1")
    assert len(claims)>=5
    for c in claims:
        assert c.eid=="e1"
        assert len(c.cid)==12
        assert len(c.text)>=30

def test_extract_claims_filters_short():
    claims=extract_claims("Hi. Ok. Yes.")
    assert len(claims)==0

def test_extract_claims_filters_questions():
    claims=extract_claims("What is the meaning of life? It remains a mystery to this day.")
    for c in claims:
        assert not c.text.endswith('?')

def test_extract_entities():
    claims=extract_claims(FORENSIC_TEXT)
    ents=extract_entities(claims)
    assert len(ents)>0
    names=[e.name for e in ents]
    assert any("Smith" in n or "John" in n for n in names)

def test_claim_cid_deterministic():
    txt="Carbon dioxide is a greenhouse gas that affects climate patterns significantly."
    c1=extract_claims(txt,eid="x")
    c2=extract_claims(txt,eid="y")
    assert len(c1)>=1
    assert c1[0].cid==c2[0].cid

def test_entities_unique():
    claims=extract_claims(FORENSIC_TEXT+FORENSIC_TEXT)
    ents=extract_entities(claims)
    names=[e.name for e in ents]
    assert len(names)==len(set(names))
