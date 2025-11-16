# Developer: Harshitha
# Added this file as Fix to render pages that hide content behind JS/accordions so we can see the tables

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

def fetch_rendered_html(url: str, wait_ms: int = 1500) -> str:
    """
    Fetches the fully rendered HTML of a given URL using Playwright (Chromium).
    Automatically handles slow-loading pages and collapsible accordions.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # Increase timeout and use lighter "domcontentloaded" event
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except PlaywrightTimeoutError:
            print(f"[warning] Timeout exceeded for {url} — continuing with partial content")

        # Expand accordions (useful for <summary>/<details> or hidden sections)
        try:
            page.eval_on_selector_all("summary", "els => els.forEach(e => e.click())")
        except Exception:
            pass  # if no accordions, skip quietly

        # Let dynamic content load
        page.wait_for_timeout(wait_ms)

        html = page.content()
        browser.close()
        return html
