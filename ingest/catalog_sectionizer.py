# ingest/catalog_sectionizer.py
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional, Tuple

# --------------------------------------------------------------------
# Canonical anchor IDs commonly used by Ellucian/Acalog catalogs.
# Keep this list broad; we'll normalize variant ids to these bases.
# --------------------------------------------------------------------
KNOWN_ANCHORS = [
    # historical/common
    "requirementstext", "programrequirementstext", "planofstudytext", "fouryearplantext",
    "roadmaptext", "programadmissiontext", "admissionrequirementstext",
    "programlearningoutcomestext", "learningoutcomestext", "policiestext", "coursestext",
    "advisingnotestext", "recommendationsstext",
    # additional variants seen on MSU and other sites
    "degreerequirementstext", "courseinformationtext", "curriculumrequirements", "sampleplan",
    # heading-derived slugs some pages produce in our pipeline
    "degree-requirements-overview",
]

# Variant suffixes we see on some deployments (e.g., requirementstextcontainer)
SUFFIXES = ("textcontainer", "container", "texttab", "tab")

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _slugify(s: str) -> str:
    s = re.sub(r"\s+", "-", s.strip().lower())
    s = re.sub(r"[^a-z0-9\-]+", "", s)
    return s or "section"

def _extract_title(soup: BeautifulSoup) -> str:
    t = soup.find("h1")
    if t and t.get_text(strip=True):
        return t.get_text(strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "Overview"

def _node_position(n) -> int:
    """Approximate document order index for sorting."""
    i = 0
    cur = n
    while cur and cur.previous_element:
        cur = cur.previous_element
        i += 1
    return i

def canonical_anchor(id_str: Optional[str]) -> Optional[str]:
    """
    Normalize an element id like 'requirementstextcontainer' or 'requirementstexttab'
    back to a canonical base in KNOWN_ANCHORS. Returns None if no match.
    """
    if not id_str:
        return None
    tid = id_str.strip().lower()

    # strip one known suffix, if present (prefer longer matches)
    for suf in sorted(SUFFIXES, key=len, reverse=True):
        if tid.endswith(suf):
            tid = tid[: -len(suf)]
            break

    # candidate variants to check against KNOWN_ANCHORS
    candidates = {
        tid,
        tid + "text",                 # e.g., "programrequirements" -> "programrequirementstext"
        tid.replace("-", ""),         # hyphenless
        tid.replace("_", ""),         # underscoreless
    }
    candidates.discard("")  # guard

    for c in candidates:
        if c in KNOWN_ANCHORS:
            return c
    return None

def _collect_between_nodes(soup: BeautifulSoup, start_node, stop_node) -> str:
    """
    Collect HTML from just AFTER start_node up to (but not including) stop_node,
    purely by document order. If that yields suspiciously little text, fall back
    to common Acalog container ids like '{base}textcontainer' / '{base}container'.
    """
    # ---- 1) Forward-walk by document order
    pieces = []
    cur = start_node.next_element  # skip the start anchor itself
    while cur is not None and cur is not stop_node:
        pieces.append(cur)
        cur = cur.next_element

    html_fw = "".join(str(n) for n in pieces) if pieces else ""
    text_fw = BeautifulSoup(html_fw, "html.parser").get_text("\n", strip=True)

    # ---- 2) If tiny (e.g., tabbed pages), try container fallback
    if len(text_fw) < 500:  # threshold chosen to catch the BS page case
        sid = getattr(start_node, "get", lambda *_: None)("id") if hasattr(start_node, "get") else None
        base = canonical_anchor(sid) if sid else None
        if base:
            pref = soup.find(id=f"{base}textcontainer") or soup.find(id=f"{base}container")
            if pref:
                return str(pref)

    return html_fw

def _collect_before(soup: BeautifulSoup, body, first_node) -> str:
    """
    Collect "overview" content that appears before first_node. Uses siblings (top-level)
    to stay tidy.
    """
    if not body or not getattr(body, "contents", None):
        return ""
    collected = []
    node = body.contents[0]
    # walk across siblings at the top level until we hit first_node
    while node and node is not first_node:
        collected.append(str(node))
        node = node.next_sibling
    return "".join(collected).strip()

# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------
def split_catalog_sections(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup

    # ---- Pass 1: discover and normalize any matching anchors on the page ----
    anchor_hits: List[Tuple[str, str]] = []  # (base_anchor, original_id)
    seen_bases = set()
    for tag in soup.find_all(attrs={"id": True}):
        base = canonical_anchor(tag.get("id"))
        if base and base not in seen_bases:
            anchor_hits.append((base, tag.get("id")))
            seen_bases.add(base)

    # ---- If we found anchors, slice by them in document order ----
    anchors_ordered: List[Tuple[str, str, object, str]] = []  # (base, title, start_node, orig_id)
    for base, orig in anchor_hits:
        el = soup.find(id=orig)
        if not el:
            continue
        heading = el.find_parent(re.compile(r"^h[1-6]$")) or el.find_previous(re.compile(r"^h[1-6]$"))
        title = heading.get_text(" ", strip=True) if heading else base.replace("text", "").replace("-", " ").title()
        # IMPORTANT: start at the ANCHOR ELEMENT itself, not the heading
        start_node = el
        anchors_ordered.append((base, title, start_node, orig))

    anchors_ordered.sort(key=lambda tup: _node_position(tup[2]))

    sections: List[Dict] = []

    if anchors_ordered:
        # Optional overview: everything before first anchor start
        first_start = anchors_ordered[0][2]
        before_html = _collect_before(soup, body, first_start)
        before_text = BeautifulSoup(before_html, "html.parser").get_text("\n", strip=True)
        if before_text:
            sections.append({
                "url": url,
                "title": _extract_title(soup),
                "text": before_text
            })

        # One chunk per anchor (URL uses canonical base id)
        for i, (base, title, start_node, orig_id) in enumerate(anchors_ordered):
            # stop at the NEXT ANCHOR ELEMENT (not its heading)
            next_stop = None
            if i + 1 < len(anchors_ordered):
                _next_base, _next_title, _next_start, _next_orig = anchors_ordered[i + 1]
                next_stop = soup.find(id=_next_orig)

            sec_html = _collect_between_nodes(soup, start_node, next_stop)
            if not sec_html:
                continue
            sec_text = BeautifulSoup(sec_html, "html.parser").get_text("\n", strip=True)
            if sec_text:
                sections.append({
                    "url": f"{url}#{base}",
                    "title": title,
                    "text": sec_text
                })
        return sections

    # ---- Fallback: split by H2/H3 when no known anchors are found ----
    headings = body.find_all(re.compile(r"^h[2-3]$"))
    if not headings:
        page_text = soup.get_text("\n", strip=True)
        return [{
            "url": url,
            "title": _extract_title(soup),
            "text": page_text
        }]

    chunks: List[Dict] = []

    # Overview before first H2/H3
    first = headings[0]
    before_html = _collect_before(soup, body, first)
    before_text = BeautifulSoup(before_html, "html.parser").get_text("\n", strip=True)
    if before_text:
        chunks.append({
            "url": url,
            "title": _extract_title(soup),
            "text": before_text
        })

    # Sections per H2/H3
    for i, h in enumerate(headings):
        title = h.get_text(" ", strip=True) or f"Section {i+1}"
        slug = _slugify(title)
        stop = headings[i + 1] if i + 1 < len(headings) else None
        sec_html = _collect_between_nodes(soup, h, stop)
        sec_text = BeautifulSoup(sec_html, "html.parser").get_text("\n", strip=True)
        chunks.append({
            "url": f"{url}#{slug}",
            "title": title,
            "text": sec_text
        })

    return chunks