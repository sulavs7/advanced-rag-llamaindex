from app.retrieval.retriever import * 
import psycopg2
import os
from dotenv import load_dotenv
from app.retrieval.embed_query import embed_query
from app.ingestion.create_embeddings import create_embedding_model

load_dotenv()

DB_URL = os.getenv("SUPABASE_URL")

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    scores = {}
    node_data = {}

    for rank, row in enumerate(dense_results, start=1):
        node_id = row["id"]
        scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank)
        node_data[node_id] = row

    for rank, row in enumerate(sparse_results, start=1):
        node_id = row["id"]
        scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank)
        node_data[node_id] = row

    ranked_ids = sorted(scores, key=lambda node_id: scores[node_id], reverse=True)
    top_ids = ranked_ids[:top_k]

    return [{**node_data[node_id], "score": scores[node_id]} for node_id in top_ids]


if __name__ == "__main__":
    conn = psycopg2.connect(DB_URL)
    query = "HealthGPT medical vision language model"
    embed_model = create_embedding_model()
    query_embedding = embed_query(embed_model,query)
    try:
        with conn:
            with conn.cursor() as cur:
                dense_results = dense_retrieve(cur, query_embedding, k=20)
                sparse_results = sparse_retrieve(cur, query, k=20)
    finally:
        conn.close()

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=5)

    dense_ids = {d["id"] for d in dense_results}
    sparse_ids = {s["id"] for s in sparse_results}
    print("dense count:", len(dense_results), "sparse count:", len(sparse_results))
    print("overlap:", len(dense_ids & sparse_ids))

    for item in fused:
        print(item["id"], round(item["score"], 5), item["chunk_text"][:80])