# Developer Name: Harshitha
# Description: Convert fetched HTML to clean Markdown.
# Removes obvious chrome (nav/header/footer/menu/site header/site footer/menu)
# Keeps headings/lists/tables
# Returns a compact Markdown string

from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Remove page chrome by css selectors
REMOVALS = [
    "nav", "header", "footer",
    ".site-footer", ".site-header", ".menu", ".breadcrumbs",
]

def clean_html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Remove page chrome
    for sel in REMOVALS:
        for node in soup.select(sel):
            node.decompose()

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Prefer semantic containers if present
    body = soup.find("main") or soup.find("article") 

    # If <main>/<article> not found, fall back to body
    if body is None:
        body = soup.body or soup

    # EXTRA: keep sections, accordions, div.content-area etc.
    if body is None:
        body = soup.select_one(".content-area, .entry-content, .page-content") or soup

    # Convert to Markdown (don’t bring images; keep structure)
    markdown = md(str(body), heading_style="ATX", strip=["img"])
    return markdown.strip()
