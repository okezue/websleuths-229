"""Tests for wm.search.content_filter."""
from wm.search.content_filter import clean_text


NOISY_PAGE = """
<div class="nav">
Menu
Home
About
Contact
</div>
<h1>Aspirin Reduces Heart Attack Risk in New Study</h1>
<p>A 2024 meta-analysis published in The Lancet found that low-dose aspirin
reduced the risk of first heart attack by 18% across 12 randomized trials
enrolling over 160,000 participants.</p>
<p>The benefit was most pronounced in adults aged 50-69 with elevated
cardiovascular risk factors, including hypertension and elevated LDL
cholesterol.</p>
<div class="sidebar">
Subscribe to our newsletter
Follow us on Twitter
Share this on Facebook
</div>
<p>Researchers noted that the bleeding risk must be weighed against the
cardiac benefit on an individual basis.</p>
<div class="footer">
© 2024 MedNews Inc. All rights reserved.
Terms of Service | Privacy Policy
</div>
"""

AD_HEAVY_PAGE = """
Advertisement
Sponsored Content
Click here to buy the best supplements!
Use code HEALTH20 for 20% off
Limited time offer
Free shipping on all orders
Add to cart
This page is mostly ads.
"""

CLEAN_ARTICLE = """Quantum computing leverages superposition and entanglement to perform
certain computations exponentially faster than classical machines. In 2024,
IBM demonstrated a 1000-qubit processor capable of error-corrected
operations, marking a significant milestone toward practical quantum
advantage in materials simulation and cryptography."""


def test_strips_html_and_boilerplate():
    result = clean_text(NOISY_PAGE)
    assert "Menu" not in result
    assert "Subscribe to our newsletter" not in result
    assert "Follow us on Twitter" not in result
    assert "All rights reserved" not in result
    assert "aspirin" in result.lower()
    assert "meta-analysis" in result


def test_ad_heavy_page_rejected():
    result = clean_text(AD_HEAVY_PAGE)
    assert result == ""


def test_clean_article_passes_through():
    result = clean_text(CLEAN_ARTICLE)
    assert "superposition" in result
    assert "entanglement" in result
    assert len(result) > 100


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""  # type: ignore


def test_short_junk_lines_removed():
    text = "OK\nHello\nThis is a real sentence about molecular biology.\nYeah\nNo"
    result = clean_text(text, min_signal_ratio=0.0)
    assert "molecular biology" in result
    assert "\nOK\n" not in result
