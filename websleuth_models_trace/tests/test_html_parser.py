from wm.config import ParseConfig
from wm.web.parsers.html import HTMLParser


def test_html_preserves_locators_tables_and_trace_claims():
    page = b'''<html><head><title>T</title><meta name="trace-source-domain" content="source-a"></head><body>
    <nav>junk</nav><h1>Heading</h1><p data-trace-claim='{"subject":"Alpha","predicate":"is","object":"Beta","text":"Alpha is Beta."}'>Alpha is Beta.</p>
    <table><tr><th>K</th><th>V</th></tr><tr><td>x</td><td>7</td></tr></table></body></html>'''
    parsed = HTMLParser(ParseConfig()).parse(doc_id="doc", url="https://example.com", content=page, mime_type="text/html")
    assert parsed.title == "T"
    assert any(span.text == "Alpha is Beta." for span in parsed.spans)
    assert any(span.modality == "table" for span in parsed.spans)
    assert parsed.metadata["trace_claims"][0]["object"] == "Beta"
    assert parsed.metadata["source_domain"] == "source-a"
