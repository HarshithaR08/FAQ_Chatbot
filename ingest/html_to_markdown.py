# Developer Name: Harshitha
# Description: Convert fetched HTML to clean Markdown.
# - Removes obvious chrome (nav/header/footer/menu/site header/site footer/menu)
# - Prefers main content containers (main/article/content-area/entry-content/page-content)
# - Keeps headings/lists/tables
# - Returns a compact Markdown string

from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re

# Remove page chrome by css selectors
REMOVALS = [
    "nav",
    "header",
    "footer",
    ".site-footer",
    ".site-header",
    ".menu",
    ".breadcrumbs",
]


def _pick_content_root(soup: BeautifulSoup):
    """
    Heuristic to pick the main content container.

    Priority:
      1. <main>
      2. <article>
      3. common WP content wrappers (.content-area, .entry-content, .page-content, #content)
      4. <body>
      5. soup (fallback)
    """
    # 1. main / article
    body = soup.find("main") or soup.find("article")
    if body is not None:
        return body

    # 2. Common WordPress / CMS content wrappers
    content_sel = ".content-area, .entry-content, .page-content, #content"
    content = soup.select_one(content_sel)
    if content is not None:
        return content

    # 3. Body
    if soup.body is not None:
        return soup.body

    # 4. Last resort
    return soup


def clean_html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Remove page chrome
    for sel in REMOVALS:
        for node in soup.select(sel):
            node.decompose()

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Choose the most likely main content container
    body = _pick_content_root(soup)

    # Convert to Markdown.
    # - markdownify already preserves HTML tables as Markdown tables
    # - Strip images so alt text / URLs don't pollute embeddings
    markdown = md(
        str(body),
        heading_style="ATX",
        strip=["img"],
    )

    # Normalize whitespace a bit:
    # - collapse 3+ blank lines into 2
    # - strip leading/trailing whitespace
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()
