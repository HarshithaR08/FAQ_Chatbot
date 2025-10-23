# Description: Added this script that tests retrieval by querying the vector index (RAG Step 3 validation).
# Created this script to type a question and see top-k chunks, with URLs, to verify retrieval quality.

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "vectorstore/chroma"
COLLECTION = "msu_faq_chunks"

# same embedding you used to build the index
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION, embedding_function=embedding_fn)

def ask(query: str, k: int = 5, bucket_filter: str | None = None):
    coll = get_collection()

    # Build query kwargs. Only include "where" if we actually have a filter.
    query_kwargs = {
        "query_texts": [query],
        "n_results": k,
    }
    if bucket_filter:
        query_kwargs["where"] = {"bucket": bucket_filter}  # valid filter

    # query → embedding → top-k nearest chunks
    result = coll.query(**query_kwargs)

    # Chroma returns parallel lists; unpack them nicely
    docs      = result["documents"][0]
    metas     = result["metadatas"][0]
    distances = result.get("distances", [[None]*len(docs)])[0]

    print(f"\nQ: {query}\nTop {k} results:\n")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        url = meta.get("url","")
        bucket = meta.get("bucket","")
        heads = meta.get("headings", [])
        score = (1 - (dist or 0)) if dist is not None else None
        score_txt = f"{score:.3f}" if score is not None else "n/a"
        print(f"{i}. bucket={bucket} | score~{score_txt} | {url}")
        if heads:
            print("   headings:", heads[:2])
        #print("   text:", (doc[:220] + "…") if len(doc) > 220 else doc)
        print("   text:", doc)
        print()

if __name__ == "__main__":
    # manual testing 
    ask("What is the course withdrawal deadline?", k=5)                 # no filter
    #("How does pass/fail grading work?", k=5, bucket_filter="policies")
    #ask("How do I request a leave of absence?", k=5)                     # no filter
    # additional questions
    #ask("How can I contact academic advising?", k=5)
    #ask("What are the eligibility criteria for study abroad?", k=5)
    #ask("Where can I find tutoring or student support services?", k=5, bucket_filter="student-services")
