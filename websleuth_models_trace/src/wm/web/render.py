from __future__ import annotations

from dataclasses import dataclass

from wm.web.policy import URLPolicy, URLPolicyError


@dataclass
class RenderedPage:
    html: str
    screenshot: bytes | None
    final_url: str


class PlaywrightRenderer:
    """Credential-free browser renderer with network-request policy enforcement.

    Every main-frame and subresource request is checked by the same SSRF policy as the
    HTTP fetcher. Page content therefore cannot use JavaScript as a bridge to localhost,
    link-local metadata services, private RFC1918 ranges, or unsupported schemes.
    """

    def __init__(self, policy: URLPolicy, timeout_ms: int = 30_000, screenshot: bool = False):
        self.policy = policy
        self.timeout_ms = timeout_ms
        self.take_screenshot = screenshot

    def render(self, url: str) -> RenderedPage:
        try:
            from playwright.sync_api import Route, sync_playwright
        except ImportError as exc:
            raise RuntimeError("Install websleuth-models[browser] and run playwright install chromium") from exc

        safe_url = self.policy.validate(url)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage"],
            )
            context = browser.new_context(
                java_script_enabled=True,
                service_workers="block",
                accept_downloads=False,
                http_credentials=None,
            )
            page = context.new_page()
            page.set_default_timeout(self.timeout_ms)

            def protect(route: Route) -> None:
                request_url = route.request.url
                try:
                    # data/blob URLs are in-memory products of an already allowed page.
                    if request_url.startswith(("data:", "blob:")):
                        route.continue_()
                    else:
                        self.policy.validate(request_url)
                        route.continue_()
                except (URLPolicyError, ValueError, OSError):
                    route.abort("blockedbyclient")

            page.route("**/*", protect)
            page.goto(safe_url, wait_until="networkidle")
            final_url = self.policy.validate(page.url)
            html = page.content()
            shot = page.screenshot(full_page=True) if self.take_screenshot else None
            context.close()
            browser.close()
        return RenderedPage(html=html, screenshot=shot, final_url=final_url)
