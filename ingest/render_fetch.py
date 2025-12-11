# Developer: Harshitha
# Render pages that hide content behind JS/accordions so we can see the tables.

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

USER_AGENT = "MSU-FAQ-Bot/0.1 (contact: ramakrishnah1@montclair.edu)"


def fetch_rendered_html(url: str, wait_ms: int = 1500) -> str:
    """
    Fetch the fully rendered HTML of a given URL using Playwright (Chromium).

    - Waits for DOMContentLoaded (not full network idle) for speed.
    - Tries to expand <summary> accordions to reveal hidden content.
    - Waits an additional `wait_ms` for dynamic content.
    - Always closes the browser, even if errors occur.

    Returns the page HTML string (possibly partial if timeouts happen).
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)

        try:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
            except PlaywrightTimeoutError:
                print(f"[warning] Timeout exceeded for {url} — continuing with partial content")
            except Exception as e:
                print(f"[warning] Error while navigating to {url}: {e}")

            # Try to expand accordions (<summary>/<details> style)
            try:
                page.eval_on_selector_all("summary", "els => els.forEach(e => e.click())")
            except Exception:
                # no <summary> or JS issue; safe to continue
                pass

            # Let dynamic content settle
            page.wait_for_timeout(wait_ms)

            html = page.content()
        finally:
            # Make sure the browser is always closed
            browser.close()

        return html
