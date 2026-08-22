import os
import psycopg2
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from app.retrieval.embed_query import embed_query
from app.ingestion.create_embeddings import create_embedding_model
from app.retrieval.reranker_model import create_reranker,rerank
from app.retrieval.retriever import dense_retrieve,sparse_retrieve
from app.retrieval.rrf import reciprocal_rank_fusion

load_dotenv()
DB_URL = os.getenv("SUPABASE_URL")

def hybrid_retrieve(
    query: str,
    cur,
    embed_model,
    reranker: CrossEncoder,
    dense_k: int = 20,
    sparse_k: int = 20,
    rrf_k: int = 60,
    fusion_top_k: int = 20,   # keep this wide — the reranker needs real
                              # candidates to sift through, not a list
                              # already narrowed to its final size by a
                              # cheaper, less accurate method (RRF).
                              # Narrowing twice before reranking defeats
                              # the point of having a reranker at all.
    final_top_k: int = 5,    # what actually gets returned to the caller
) -> list[dict]:
    """
    Full hybrid retrieval pipeline: dense + sparse -> RRF fusion -> rerank.

    cur, embed_model, and reranker are passed in, not created here —
    they're expensive (DB connection, loaded models) and must be created
    ONCE outside this function and reused across calls. If this gets
    wired into a LangGraph tool that fires once per agent turn, creating
    fresh connections/models per call would reload a HuggingFace model
    and reopen a DB connection on every single query — the same mistake
    already caught and fixed in ingest_file earlier tonight.
    """
    query_embedding = embed_query(embed_model, query)
    print("query embedded succesfully")

    dense_results = dense_retrieve(cur, query_embedding, k=dense_k)
    print("dense results ")
    sparse_results = sparse_retrieve(cur, query, k=sparse_k)
    print("sparse results")
    fused = reciprocal_rank_fusion(
        dense_results, sparse_results, k=rrf_k, top_k=fusion_top_k
    )
    print("fused results")

    reranked = rerank(query, fused, reranker, top_k=final_top_k)
    print("reranked results")
    return reranked

if __name__ == "__main__":
    conn = psycopg2.connect(DB_URL)
    query = "HealthGPT medical vision language model"
    embed_model = create_embedding_model()
    reranker = create_reranker()

    try:
        with conn:
            with conn.cursor() as cur:
                print("Getting results")
                results = hybrid_retrieve(query,
                    cur, embed_model, reranker,
                )
                print("results count:", len(results))
    finally:
        conn.close()

    for r in results:
        print(round(r["rerank_score"], 4), r["chunk_text"][:80])