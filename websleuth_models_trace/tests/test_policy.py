import pytest

from wm.web.policy import URLPolicy, URLPolicyError, canonicalize_url


def test_canonicalization():
    assert canonicalize_url("HTTPS://Example.COM:443/a#x") == "https://example.com/a"


def test_private_network_blocked():
    with pytest.raises(URLPolicyError):
        URLPolicy(resolve_dns=False).validate("http://127.0.0.1/x")


def test_file_url_control(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("x")
    with pytest.raises(URLPolicyError):
        URLPolicy(allow_file_urls=False).validate(path.as_uri())
    assert URLPolicy(allow_file_urls=True).validate(path.as_uri()) == path.as_uri()
