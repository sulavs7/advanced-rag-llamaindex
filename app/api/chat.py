def run_query(query: str, db_url: str, k: int = 5) -> str:
    embed_model = create_embedding_model()
    query_embedding = embed_query(embed_model, query)

    llm = Groq(model="llama-3.3-70b-versatile", api_key=UserSecretsClient().get_secret("GROQ_API_KEY"))

    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor() as cur:
                retrieved = retrieve_similar_nodes(cur, query_embedding, k=k)
    finally:
        conn.close()

    if not retrieved or retrieved[0]['similarity'] < 0.5:
        return "I couldn't find anything relevant in the documents."

    context = build_context(retrieved)
    answer = generate_answer(llm, query, context)
    return answer