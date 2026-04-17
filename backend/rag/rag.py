import json

import requests


def local_llm_stream(prompt, on_token=None):
    if on_token is None:
        on_token = lambda token: print(token, end="", flush=True)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 80
            }
        },
        stream=True,
        timeout=(5, 120),
    )
    response.raise_for_status()

    full_response = []

    # Use very small chunks to make terminal streaming feel immediate.
    for line in response.iter_lines(decode_unicode=True, chunk_size=1):
        if line:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = data.get("response", "")

            if token:
                on_token(token)
                full_response.append(token)

            if data.get("done"):
                break

    return "".join(full_response).strip()



def ask_rag(query, db, history, on_token=None):

    # Prefer the actual user question when identity prefix is present.
    clean_query = query
    if "." in query:
        parts = [part.strip() for part in query.split(".") if part.strip()]
        if parts:
            clean_query = parts[-1]

    clean_query = clean_query.strip() or query.strip()

    docs_with_scores = db.similarity_search_with_score(clean_query, k=3)

    # FAISS scores depend on index metric; a fixed cutoff can drop all results.
    # Keep top matches and only fallback when retrieval returns nothing at all.
    docs = [doc for doc, _ in docs_with_scores[:3]]

    if not docs:
        fallback = "I don't know based on available data."
        if on_token:
            on_token(fallback)
        return fallback

    # 🔹 Structured context
    context = "\n\n".join(
        [f"[Chunk {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
You are the CHALLENGERS club assistant.

STRICT RULES:
- Use CONTEXT for factual answers about the club
- Use CONVERSATION for personal/user-related questions (like name, previous messages)
- If answer is not in context AND not in conversation → say "I don't know"
- Keep answer within 2–3 sentences
- Be direct

Context:
{context}

Conversation:
{history}

Question:
{query}

Answer:
"""

    answer = local_llm_stream(prompt, on_token=on_token)

    if not answer:
        fallback = "I don't know based on available data."
        if on_token:
            on_token(fallback)
        return fallback

    return answer