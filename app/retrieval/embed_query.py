
def embed_query(embed_model : HuggingFaceEmbedding , query:str)->list[float]:
    """
    Embed the user's query with the same model used at ingestion time.
    """
    return embed_model.get_query_embedding(query)

