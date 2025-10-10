# Developer Name: Harshitha
# Pipeline Description: Fetch & parse sitemap indexes, collect child sitemap URLs, extract page URLs.
# Then filter them using bucket regex rules from sources.yaml. Writes url,lastmod,bucket to CSV.
# Date: 26 Sep 2025 

from __future__ import annotations
import requests
import os
import logging
import time
import csv
import re
import yaml
import xml.etree.ElementTree as ET
from typing import List, Tuple, Iterable
from urllib.parse import urlparse

# header string to send when making an HTTP request for web servers to identify bot.
USER_AGENT = "MSU-FAQ-Bot/0.1 (contact: ramakrishnah1@montclair.edu)"

# setting up debug logs
def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )


# HTTP helper - send get request to the server with USER_AGENT header
# download sitemap XML safely (with headers + logging)
def get_http(url: str, timeout=20) -> str | None:
    logging.debug(f"GET {url}")
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if r.status_code == 404:
            logging.warning(f"404 Not Found: {url}")
            return None
        r.raise_for_status()
        logging.debug(f"OK {url} ({len(r.text)} bytes)")
        return r.text
    except requests.RequestException as e:
        logging.warning(f"HTTP error for {url}: {e}")
        return None


# extract child sitemap URLs from the index file
# Input: <sitemapindex> XML string
# Output: list of child sitemap URLs
def parse_index_for_sitemaps(xml_text: str) -> List[str]:
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(xml_text)
    out: List[str] = []
    for sm in root.findall("sm:sitemap", ns): #find all child sitemaps <sitemap> tags in tree root 
        loc = sm.findtext("sm:loc", default="", namespaces=ns).strip() #inside this block look for url's (loc) and fetch the text or url
        if loc:
            out.append(loc)
    logging.info(f"Child sitemaps found: {len(out)}")
    return out 


# extract sitemap pages/posts to fetch list of (loc, lastmod) from a sitemap **page** (not index).
# extract <loc>, <lastmod> entries from sitemap page
# Input: <urlset> XML string
# Output: list of (loc, lastmod)
def parse_sitemap_for_urls(xml_text: str) -> List[Tuple[str, str]]:
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(xml_text)
    out: List[Tuple[str, str]] = []
    for url in root.findall("sm:url", ns):
        loc = url.findtext("sm:loc", default="", namespaces=ns).strip()
        lastmod = url.findtext("sm:lastmod", default="", namespaces=ns).strip()
        if loc:
            out.append((loc, lastmod))
    logging.info(f"URLs in this sitemap: {len(out)}")
    return out


# For each URL, detect if it's an index or a page. If index → collect children.
def expand_sitemaps(start_urls: Iterable[str]) -> List[str]:
    out: List[str] = []
    for su in start_urls: # go through each sitemap url in my input list
        xml = get_http(su)
        if not xml:
            logging.info(f"Skip (no XML): {su}")
            continue 
        #if condition to check if this is a sitemap index file
        if "<sitemapindex" in xml:
            kids = parse_index_for_sitemaps(xml)
            # skip noisy sections commonly present in WP like taxonomies + users
            kids = [k for k in kids if not any(x in k for x in ["taxonomies", "users"])]
            out.extend(kids)
            logging.info(f"Index → {su} has {len(kids)} children after filtering")
        # if its not an index page it will come to else loop, then it must be a sitemap page
        else:
            out.append(su)
            logging.info(f"Plain sitemap page: {su}")
    return out


# visit each child sitemap and collect page url,lastmod
def collect_urls(child_sitemaps: Iterable[str]) -> List[Tuple[str, str]]:
    all_urls: List[Tuple[str, str]] = []
    for sm_url in child_sitemaps:
        xml = get_http(sm_url)
        if not xml:
            logging.info(f"Skip child sitemap (no XML): {sm_url}")
            continue
        try:
            urls = parse_sitemap_for_urls(xml)
            logging.info(f"{sm_url} → {len(urls)} urls")
            all_urls.extend(urls)
            time.sleep(0.2)
        except Exception as e:
            logging.warning(f"Skip {sm_url}: {e}")
    return all_urls


# Regex bucket filtering (from sources.yaml)
# Compile bucket include/exclude regex and global exclude regex from YAML cfg.

def compile_rules(cfg):
    buckets_cfg = cfg.get("buckets", [])
    compiled_buckets = []
    for b in buckets_cfg:
        incl = [re.compile(p) for p in b.get("include", [])]
        excl = [re.compile(p) for p in b.get("exclude", [])]
        compiled_buckets.append({"name": b["name"], "include": incl, "exclude": excl})
    global_excl = [re.compile(p) for p in cfg.get("exclude_patterns", [])]
    return compiled_buckets, global_excl

# for policies list
def compile_policy_allowlist(cfg):
    pats = cfg.get("policy_focus_allowlist", [])
    return [re.compile(p) for p in pats]


def first_matching_bucket(path: str, compiled_buckets) -> str | None:
    #Return bucket name if path matches include and not exclude.
    for b in compiled_buckets:
        if any(r.search(path) for r in b["include"]):
            if any(rx.search(path) for rx in b["exclude"]):
                return None
            return b["name"]
    return None

def globally_excluded(path: str, global_excl) -> bool:
    return any(rx.search(path) for rx in global_excl)

# this function applies regex-based inclusion/exclusion logic
# Input: [(url, lastmod)]
# Output: [(url, lastmod, bucket)]  
def filter_with_buckets(urls: List[Tuple[str, str]], cfg) -> List[Tuple[str, str, str]]:
    compiled_buckets, global_excl = compile_rules(cfg)
    policy_allow = compile_policy_allowlist(cfg)
    kept = []
    dropped_global = dropped_no_bucket = dropped_policy_not_focus = 0

    for (u, lm) in urls:
        path = urlparse(u).path
        # global excludes
        if globally_excluded(path, global_excl):
            dropped_global += 1
            continue
        # bucket include/exclude
        bucket = first_matching_bucket(path, compiled_buckets)
        if bucket is None:
            dropped_no_bucket += 1
            continue
        # Enforce allow-list only for policies
        if bucket == "policies" and policy_allow:
            if not any(p.search(path) for p in policy_allow):
                dropped_policy_not_focus += 1
                continue
        kept.append((u, lm, bucket))

    logging.info(
        "Bucket filter: kept=%d, dropped_global=%d, dropped_no_bucket=%d, dropped_policy_not_focus=%d",
        len(kept), dropped_global, dropped_no_bucket, dropped_policy_not_focus
    )
    return kept

# Removes duplicate url's & Keeps one copy of every unique URL.
def dedupe_with_bucket(rows: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    seen, out = set(), []
    for u, lm, b in rows:
        if u not in seen:
            out.append((u, lm, b))
            seen.add(u)
    logging.info(f"Deduped to {len(out)} unique URLs (bucketed)")
    return out

# save final (url, lastmod, bucket) list to disk
def save_csv_bucket(rows: List[Tuple[str, str, str]], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "lastmod", "bucket"])
        w.writerows(rows)
    logging.info(f"Wrote URL list → {csv_path}")


# Public entry point used by run_all.py
def run_from_sources(
    sitemaps: List[str],
    include_prefixes: List[str],  # kept for compatibility; unused here
    csv_out: str,
    cfg=None
) -> List[Tuple[str, str, str]]:  # Returns: list of (url, lastmod, bucket)
    setup_logger()
    logging.info("Step A: expand sitemap indexes")
    children = expand_sitemaps(sitemaps)

    logging.info("Step B: collect URLs from child sitemaps")
    all_urls = collect_urls(children)  # [(url,lastmod)]

    logging.info("Step C: apply bucket rules")
    bucketed = filter_with_buckets(all_urls, cfg)  # [(url,lastmod,bucket)]

    logging.info("Step D: dedupe")
    unique = dedupe_with_bucket(bucketed)

    logging.debug("Sample of first 10 URLs after filtering:")
    for u, lm, b in unique[:10]:
        logging.debug(f" - {u} | lastmod={lm} | bucket={b}")

    save_csv_bucket(unique, csv_out)
    return unique

# test block 
if __name__ == "__main__":
    setup_logger()
    # Load YAML to get sitemaps + bucket rules
    with open("sources.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    final = run_from_sources(
        sitemaps=cfg["sitemaps"],
        include_prefixes=cfg.get("include_prefixes", []),
        csv_out=cfg["paths"]["url_list_csv"],
        cfg=cfg
    )
    print(f"Wrote {len(final)} rows to {cfg['paths']['url_list_csv']}")
    print("First 5 rows:", final[:5])