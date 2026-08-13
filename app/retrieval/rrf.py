def reciporocal_rank_fusion(
    dense_results:list[dict],
    sparse_results : list[dict],
    k:int = 60,
    top_k:int = 5 ,
)->list[dict]:
    scores = {}
    node_data = {}
    for rank,row in enumerate(dense_results):
        node_id = row["id"]
        scores[node_id] = scores.get(node_id,0)+1/(k+rank)
        node_data[node_id] = row

    for rank, row in enumerate(sparse_results, start=1):
        node_id = row["id"]
        scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank)
        node_data[node_id] = row

    ranked_ids = sorted(scores,key=lambda node_id :scores[node_id], reverse=True)
    top_ids = ranked_ids[:top_k]

    return [node_data[node_id] for node_id in top_ids]  