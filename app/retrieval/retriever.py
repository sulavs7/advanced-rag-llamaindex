def dense_retrieve(cur,query_embedding:list[float] , k:int = 5)->list[dict]:
    """
    Find the k nodes whose embeddings are closest (cosine distance) to the query.
    """
    cur.execute("""SELECT 
    n.id ,
    n.chunk_text,
    n.page_no,
    n.headings,
    d.file_name,
    1-(n.embedding <=> %s::vector) AS similarity 
    FROM nodes n 
    JOIN documents d ON d.id = n.document_id 
    ORDER BY similarity DESC 
    LIMIT %s;
    """,
        (str(query_embedding),k),
               )
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]

def sparse_retrieve(cur, query: str, k: int = 5) -> list[dict]:
    """Find the k nodes whose text_search best matches the query via keyword search."""
    cur.execute("""SELECT 
    n.id ,
    n.chunk_text,
    n.page_no,
    n.headings,
    d.file_name,
    ts_rank(n.text_search, plainto_tsquery('english', %s)) as rank 
    FROM nodes n 
    JOIN documents d ON d.id = n.document_id 
    WHERE n.text_search @@ plainto_tsquery('english', %s)
    ORDER BY rank DESC 
    LIMIT %s""",(query,query,k))
    
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]

# def build_context(retrieved: list[dict]) -> str:
#     parts = []
#     for r in retrieved:
#         source = f"{r['file_name']}" + (f" (page {r['page_no']})" if r['page_no'] else "")
#         parts.append(f"[Source: {source}]\n{r['chunk_text']}")
#         print(f"[{r['similarity']:.3f}] {r['file_name']} p.{r['page_no']}: {r['chunk_text'][:100]}...")
        
#     return "\n\n---\n\n".join(parts)