from sentence_transformers import CrossEncoder

def create_reranker(model_name:str = "BAAI/bge-reranker-v2-m3")-> CrossEncoder:
    cross_encoder = CrossEncoder(model_name,max_length=512)

    return cross_encoder

def rerank(query:str,candidates:list[dict],reranker:CrossEncoder,top_k:int=5 )->list[dict]:
    """
    Score each candidate against the query with the cross-encoder,
    return the top_k candidates re-ordered by that score.
    """
    pairs = [(query, c["chunk_text"]) for c in candidates]

    scores = reranker.predict(pairs)# returns a list/array of floats, same order as `pairs`

    # each candidate needs its score attached so we can sort by it
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = score
    
    ranked_candidates = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    return ranked_candidates[:top_k]



