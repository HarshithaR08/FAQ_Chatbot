"""
ui/rag_cli.py

Simple command-line RAG demo using:

- Retrieval + reranking from vectorstore.build_index_retrieval / vectorstore.rerank
- LLaMA 3 8B Instruct from infra.llama3_client

Usage:

    # Make sure embeddings + Chroma index are already built:
    python -m ingest.run_all
    python -m ingest.pdf_to_markdown
    python -m eval.policy_postfilter
    python -m rag.chunker
    python -m kb.build_kb
    python -m vectorstore.build_index_retrieval

    # Then run this script:
    python -m ui.rag_cli

You will be able to type questions like:
    "What is the withdrawal deadline for Winter 2025?"
and see a grounded answer + sources.
"""

from typing import List, Dict, Any, Tuple
import yaml

from infra.llama3_client import get_llama3_pipeline
from vectorstore.build_index_retrieval import get_retriever
from vectorstore.rerank import rerank, detect_intent

# ---- Load rerank config directly from sources.yaml ----

with open("sources.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# ---------- Retrieval helper ----------


def retrieve_top_chunks(
    query: str, top_k: int = 3
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Use the same retrieval + rerank pipeline we wired for testing.

    Returns:
        top_chunks: list of candidate dicts with fields like:
            - text
            - url
            - meta (includes bucket, term, table, etc.)
        intent: string like "deadline", "calendar", "default"
    """
    retriever = get_retriever()

    # our get_retriever() returns a function:
    #   candidates = retriever(query, n_results=50)
    base_candidates: List[Dict[str, Any]] = retriever(query, n_results=50)

    top_chunks, intent = rerank(
        candidates=base_candidates,
        query=query,
        cfg=CFG,
        top_k=top_k,
    )
    return top_chunks, intent


# ---------- Prompt building ----------


def build_messages(query: str, top_chunks: List[Dict[str, Any]]) -> list[dict]:
    """
    Build chat-style messages for LLaMA 3 using the retrieved context.

    We keep the prompt strict: only answer from context; if unsure, say so.
    """
    context_blocks = []
    for i, c in enumerate(top_chunks, 1):
        meta = c.get("meta", {})
        url = c.get("url", "")
        text = c.get("text", "")

        # Keep snippets reasonably short to avoid blowing up the prompt
        snippet = text[:1200] + ("…" if len(text) > 1200 else "")

        context_blocks.append(
            f"[{i}] Source: {url}\n"
            f"Bucket: {meta.get('bucket', '')}\n"
            f"{snippet}"
        )

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are the Montclair State University FAQ assistant.\n"
        "You answer questions about academic policies, registrar rules, and School of Computing information.\n"
        "Use ONLY the provided context snippets from official university pages.\n"
        "If the answer is not clearly stated in the context, say you are not sure and suggest who the student should contact.\n"
        "When you do know, answer concisely (2–4 sentences) and cite sources like [1], [2] referring to the snippets."
    )

    user_content = (
        f"Student question:\n{query}\n\n"
        f"Relevant policy/context snippets:\n{context}\n\n"
        "Using ONLY the information above, write your answer now."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages


# ---------- LLaMA 3 call ----------


def generate_answer(query: str, top_k: int = 4) -> str:
    """
    Full RAG: retrieve -> build prompt -> call LLaMA 3 -> return answer text.
    """
    # 1) Retrieve context
    chunks, intent = retrieve_top_chunks(query, top_k=top_k)
    print(f"[diagnostic] detected_intent={intent!r}   retrieved_chunks={len(chunks)}")

    # 2) LLaMA 3 client (pipeline + tokenizer)
    pipe, tok = get_llama3_pipeline()

    # 3) Build chat messages & prompt
    messages = build_messages(query, chunks)
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 4) Generate
    outputs = pipe(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        temperature=0.2,
        pad_token_id=tok.eos_token_id,
    )

    full_text = outputs[0]["generated_text"]
    # Strip the prompt part to isolate the assistant’s answer
    answer = full_text[len(prompt):].strip()
    return answer


# ---------- CLI ----------


def main():
    print("=== MSU FAQ Chatbot — LLaMA 3 RAG CLI ===")
    print("Type a question about MSU policies. Type 'exit' to quit.\n")

    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer = generate_answer(q, top_k=4)
        except Exception as e:
            print(f"[error] {e}")
            continue

        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")


if __name__ == "__main__":
    main()
