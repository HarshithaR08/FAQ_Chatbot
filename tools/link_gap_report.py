# Developer Name: Harshitha
#  Description: to have an idea of how many urls are missed through the sitemap

import os, re, csv, glob, pathlib, yaml

CFG = yaml.safe_load(open("sources.yaml", encoding="utf-8"))
GLOBAL_EXCL = [re.compile(p) for p in CFG.get("exclude_patterns", [])]

def globally_excluded(path: str) -> bool:
    return any(rx.search(path) for rx in GLOBAL_EXCL)

# present paths from CSV
present = set()
with open("data/url_list.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        url = row["url"].split("?")[0]
        path = re.sub(r"^https?://[^/]+", "", url).rstrip("/")
        present.add(path)

# collect internal links referenced in processed markdown
link_re = re.compile(r"\]\((/[^)#?\s]+)")
linked = set()
for p in glob.glob("data/processed/**/*.md", recursive=True):
    text = pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")
    for m in link_re.finditer(text):
        path = m.group(1).split("?")[0].rstrip("/")
        # ignore anything globally excluded
        if not globally_excluded(path):
            linked.add(path)

missing = sorted(linked - present)

# Top prefixes
from collections import Counter
top = Counter(["/"+x.strip("/").split("/")[0] for x in missing]).most_common(15)

print(f"Referenced internal links (after global excludes): {len(linked)}")
print(f"Present in CSV:                                    {len(present)}")
print(f"Missing in CSV:                                    {len(missing)}\n")
print("Top missing prefixes:")
for k, v in top:
    print(f"  {k:<40} {v}")

# Save full list (if you want to inspect)
os.makedirs("data", exist_ok=True)
with open("data/missing_links.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path"])
    for x in missing:
        w.writerow([x])
