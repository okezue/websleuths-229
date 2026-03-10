"""Tests for wm.search.content_filter."""
from wm.search.content_filter import clean_text

# ── Fixtures ───────────────────────────────────────────────────────

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


# ── Original tests ─────────────────────────────────────────────────

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


# ── Pipe table rows ────────────────────────────────────────────────

def test_pipe_table_3plus_removed():
    """Wikipedia/GitHub infobox tables with 3+ pipe chars are stripped."""
    text = (
        "Aspirin is a widely used drug.\n"
        "| Component | Description | Usage |\n"
        "|---|---|---|\n"
        "| torch | Tensor library | GPU support |\n"
        "Aspirin reduces inflammation by inhibiting COX enzymes.\n"
    )
    result = clean_text(text, min_signal_ratio=0.0)
    assert "| Component |" not in result
    assert "| torch |" not in result
    assert "widely used drug" in result
    assert "inhibiting COX" in result


def test_pipe_table_2pipe_removed():
    """2-pipe table rows (MDN prerequisites, Wikipedia infoboxes) are stripped."""
    text = (
        "Promises are the foundation of async JavaScript.\n"
        "| Prerequisites: | A solid understanding of JavaScript. |\n"
        "| Learning outcomes: |\n"
        "A promise is an object returned by an asynchronous function.\n"
    )
    result = clean_text(text, min_signal_ratio=0.0)
    assert "| Prerequisites:" not in result
    assert "| Learning outcomes:" not in result
    assert "foundation of async JavaScript" in result
    assert "asynchronous function" in result


# ── Consecutive duplicate deduplication ───────────────────────────

def test_consecutive_duplicates_removed():
    """Trafilatura double-extracts structured lists — exact consecutive dupes stripped."""
    text = (
        "Scientists developed a new cancer treatment using CAR-T cell therapy.\n"
        "Scientists developed a new cancer treatment using CAR-T cell therapy.\n"
        "The therapy showed 80% remission rates in clinical trials.\n"
        "The therapy showed 80% remission rates in clinical trials.\n"
        "Further research is needed to confirm long-term efficacy.\n"
    )
    result = clean_text(text, min_signal_ratio=0.0)
    # Each sentence should appear exactly once
    assert result.count("CAR-T cell therapy") == 1
    assert result.count("clinical trials") == 1
    assert "long-term efficacy" in result


def test_non_consecutive_duplicates_kept():
    """Duplicate lines that are NOT consecutive should be preserved."""
    text = (
        "Aspirin inhibits COX-1 and COX-2 enzymes.\n"
        "This mechanism reduces prostaglandin synthesis.\n"
        "Aspirin inhibits COX-1 and COX-2 enzymes.\n"
    )
    result = clean_text(text, min_signal_ratio=0.0)
    assert result.count("COX-1 and COX-2") == 2


# ── Encoding artifacts ─────────────────────────────────────────────

def test_encoding_artifact_stripped():
    """U+00C2 (Â) encoding corruption from latin-1/UTF-8 mismatch is removed."""
    text = "Aspirin is used to reduceÂ fever and relieve pain in adults.\n" * 5
    result = clean_text(text, min_signal_ratio=0.0)
    assert "\u00c2" not in result
    assert "reduce fever" in result


# ── Advertiser disclosure ──────────────────────────────────────────

def test_advertiser_disclosure_stripped():
    """Investopedia-style advertiser disclosure lines are stripped."""
    text = (
        "Inflation is the rate at which the general price level rises over time.\n"
        "Advertiser Disclosure: The offers that appear here are from companies.\n"
        "The Federal Reserve targets 2% inflation annually.\n"
        "from partnerships from which Investopedia receives compensation.\n"
        "Core inflation excludes food and energy prices.\n"
    )
    result = clean_text(text, min_signal_ratio=0.0)
    assert "Advertiser Disclosure" not in result
    assert "receives compensation" not in result
    assert "general price level" in result
    assert "Core inflation" in result


# ── Survey / marketing prompts ─────────────────────────────────────

def test_survey_prompt_stripped():
    """WebMD-style survey prompts at page end are stripped."""
    text = (
        "Aspirin is available over-the-counter in 81mg and 325mg doses.\n"
        "Are you currently using Aspirin (Bayer, Vazalore, and others)?\n"
        "This survey is being conducted by the WebMD marketing sciences department.\n"
    )
    result = clean_text(text, min_signal_ratio=0.0)
    assert "This survey is being conducted" not in result
    assert "marketing sciences department" not in result
    assert "over-the-counter" in result


# ── Photo credit lines ─────────────────────────────────────────────

def test_photo_credits_stripped():
    """Photo attribution lines from news sites are stripped."""
    text = (
        "A new study links ultra-processed food to increased cancer risk.\n"
        "David Ryder/Getty Images\n"
        "The research followed 200,000 participants over ten years.\n"
        "Jane Smith/AFP via Getty Images\n"
        "Participants with higher UPF consumption had a 19% elevated risk.\n"
        "John Doe/Reuters\n"
    )
    result = clean_text(text, min_signal_ratio=0.0)
    assert "Getty Images" not in result
    assert "Reuters" not in result
    assert "ultra-processed food" in result
    assert "elevated risk" in result


# ── JSON blob lines ────────────────────────────────────────────────

def test_json_blob_stripped():
    """Raw JSON/CMS config blobs leaked into extracted text are stripped."""
    text = (
        "The World Bank tracks global poverty using multiple thresholds.\n"
        '{ "facets": ["topics"], "orderby": "contentDate desc", "filter": ["poverty"] }\n'
        "In 2023, approximately 700 million people lived below $2.15 per day.\n"
    )
    result = clean_text(text, min_signal_ratio=0.0)
    assert '{ "facets"' not in result
    assert "global poverty" in result
    assert "700 million" in result


# ── Site navigation footers ────────────────────────────────────────

def test_pubmed_nav_footer_stripped():
    """PubMed navigation labels are stripped."""
    text = (
        "Neuropsychiatric syndromes occur frequently in multiple sclerosis patients.\n"
        "Competing interests: None declared.\n"
        "LinkOut - more resources\n"
        "Full Text Sources\n"
        "Other Literature Sources\n"
    )
    result = clean_text(text, min_signal_ratio=0.0)
    assert "LinkOut" not in result
    assert "Full Text Sources" not in result
    assert "Other Literature Sources" not in result
    assert "multiple sclerosis" in result


# ── Long lines pass through pattern checks ─────────────────────────

def test_long_line_with_ad_keyword_kept():
    """A long paragraph mentioning an ad keyword should NOT be filtered."""
    long_para = (
        "The Federal Trade Commission requires advertiser disclosure on sponsored content, "
        "but this rule has been inconsistently enforced across social media platforms. "
        "Researchers found that fewer than 30% of influencer posts include proper disclosure. "
        "This gap raises significant concerns about consumer protection and market transparency. " * 3
    )
    result = clean_text(long_para, min_signal_ratio=0.0)
    assert "Federal Trade Commission" in result
    assert len(result) > 200
