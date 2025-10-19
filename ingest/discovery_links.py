# Developer: Harshitha
# Purpose: Harvest deep links from a few seed pages and return (url, lastmod, bucket)

from __future__ import annotations
from fetch_from_sitemap import compile_rules, first_matching_bucket, globally_excluded
import logging, re, yaml, requests
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List, Tuple
from bs4 import BeautifulSoup

USER_AGENT = "MSU-FAQ-Bot/0.1 (contact: ramakrishnah1@montclair.edu)"
AUTO_SITEMAP_CANDIDATES = ["sitemap.xml", "sitemap_index.xml", "wp-sitemap.xml"]


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

# auto discover sitemaps for whitelisted new hosts
def try_autodiscover_sitemap_for_host(host: str, ua: str) -> str | None:
    import requests
    for path in AUTO_SITEMAP_CANDIDATES:
        url = f"https://{host}/{path}"
        try:
            r = requests.get(url, headers={"User-Agent": ua}, timeout=10)
            if r.status_code == 200 and "<sitemap" in r.text:
                return url
        except Exception:
            pass
    return None

def load_cfg(path="sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _norm_url(u: str) -> str | None:
    if not u: return None
    p = urlparse(u)
    if not p.scheme or not p.netloc:  # skip relative here; we resolve later
        return None
    p = p._replace(params="", query="", fragment="")
    return urlunparse(p)

def _is_allowed_host(u: str, host_whitelist: list[str]) -> bool:
    host = (urlparse(u).hostname or "").lower()
    wl = {h.lower() for h in (host_whitelist or [])}
    return host in wl

def _path_allowed(path: str, allow, block) -> bool:
    if any(rx.search(path) for rx in block): return False
    if not allow: return True
    return any(rx.search(path) for rx in allow)

def harvest_from_seed(seed_url: str, cfg) -> list[str]:
    logging.info(f"Seed → {seed_url}")
    try:
        r = requests.get(seed_url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        logging.warning(f"Skip seed (fetch failed): {seed_url} — {e}")
        return []

    soup = BeautifulSoup(r.text, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        abs_url = urljoin(seed_url, a["href"].strip())
        abs_url = _norm_url(abs_url)
        if abs_url:
            links.add(abs_url)

    disc = cfg.get("discovery", {})
    host_whitelist = disc.get("domain_whitelist", [])
    allow = [re.compile(p) for p in disc.get("path_allowlist", [])]
    block = [re.compile(p) for p in disc.get("path_blocklist", [])]

    kept = []
    for u in links:
        if not _is_allowed_host(u, host_whitelist):
            continue
        path = urlparse(u).path
        if not _path_allowed(path, allow, block):
            continue
        kept.append(u)

    logging.info(f"Seed kept {len(kept)} after domain/path filters")
    return sorted(kept)

# Returns list of (url, lastmod, bucket) from discovery seeds.
def discover_links(cfg) -> List[Tuple[str, str, str]]:
    disc = cfg.get("discovery", {})
    seeds = disc.get("seed_pages") or [
        "https://www.montclair.edu/school-of-computing/graduate/",
        "https://www.montclair.edu/school-of-computing/undergraduate/",
    ]

    # compile buckets + global excludes once (regex objects, not strings)
    compiled_buckets, global_excl = compile_rules(cfg)

    out: list[Tuple[str, str, str]] = []
    for seed in seeds:
        urls = harvest_from_seed(seed, cfg)
        for u in urls:
            path = urlparse(u).path

            # use compiled global_excl
            if globally_excluded(path, global_excl):
                continue

            # assign bucket using your normal rules
            bucket = first_matching_bucket(path, compiled_buckets)
            if not bucket:
                continue

            out.append((u, "", bucket))

    logging.info(f"Discovery total kept: {len(out)}")
        # Optional auto-sitemap discovery for new, whitelisted hosts ===
    if disc.get("autositemap"):
        import json, os
        seen_hosts = set()
        for (u, _, _) in out:
            h = (urlparse(u).hostname or "").lower()
            if h:
                seen_hosts.add(h)

        # hosts already covered by sitemaps in YAML
        existing_hosts = set()
        for sm in cfg.get("sitemaps", []):
            try:
                existing_hosts.add((urlparse(sm).hostname or "").lower())
            except Exception:
                pass

        wl = set((disc.get("domain_whitelist") or []))
        todo_hosts = [h for h in seen_hosts if h in wl and h not in existing_hosts]

        new_sitemaps = []
        for host in todo_hosts:
            sm = try_autodiscover_sitemap_for_host(host, USER_AGENT)
            if sm:
                new_sitemaps.append(sm)

        if new_sitemaps:
            state_path = disc.get("last_run_db") or "state/last_run.json"
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            state = {}
            if os.path.exists(state_path):
                try:
                    state = json.load(open(state_path, "r"))
                except Exception:
                    state = {}
            state.setdefault("autodiscovered_sitemaps", [])
            for sm in new_sitemaps:
                if sm not in state["autodiscovered_sitemaps"]:
                    state["autodiscovered_sitemaps"].append(sm)
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            logging.info(f"Auto-discovered sitemaps: {new_sitemaps}")
    return out


if __name__ == "__main__":
    setup_logger()
    cfg = load_cfg()
    rows = discover_links(cfg)
    for r in rows[:10]:
        logging.info(f"DISC: {r}")
    logging.info(f"Done. Discovered={len(rows)}")
