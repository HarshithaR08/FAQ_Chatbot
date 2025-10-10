# Print a quick sample from rag/chunks.jsonl to see what chunks look like.

import json, itertools

def main():
    path = "rag/chunks.jsonl"
    n = 3
    print(f"Showing first {n} chunks from {path}:\n")
    with open(path, "r", encoding="utf-8") as f:
        for line in itertools.islice(f, n):
            obj = json.loads(line)
            print(f"- id: {obj['id']}")
            print(f"  url: {obj['source_url']}")
            print(f"  bucket: {obj.get('bucket','')}")
            if obj.get("headings"):
                print(f"  headings: {obj['headings'][:2]}")
            print(f"  text:\n{obj['text'][:400]}...\n")

if __name__ == "__main__":
    main()
