def ask_rag(query, db, client, history):
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are the official AI assistant of the CHALLENGERS club.

Your job:
- Help users understand the club
- Maintain consistent identity
- Use context as primary source
- Use conversation for continuity
- Avoid hallucination
- Match user tone

-----------------------
Conversation History:
{history}

-----------------------
Context:
{context}

-----------------------
User Question:
{query}

-----------------------
Answer:
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
