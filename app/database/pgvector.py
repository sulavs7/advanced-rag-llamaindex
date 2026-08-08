import json
from llama_index.core import Document
from llama_index.core.schema import TextNode

def insert_documents_row(cur,doc:Document)->str:
    """Insert one document row, return its generated UUID."""
    cur.execute(
        """
        INSERT INTO documents (
            file_name, file_path, file_type, file_size,
            creation_date, last_modified_date,content_hash
        )
        VALUES (%s, %s, %s, %s, %s, %s,%s)
        RETURNING id;
        """,
        (
            doc.metadata.get("file_name"),
            doc.metadata.get("file_path"),
            doc.metadata.get("file_type"),
            doc.metadata.get("file_size"),
            doc.metadata.get("creation_date"),
            doc.metadata.get("last_modified_date"),
            doc.metadata.get("content_hash")
        ),
    )
    return cur.fetchone()[0]

def insert_node_rows(cur, document_id: str, nodes: list[TextNode]) -> None:
    """Insert already-embedded nodes belonging to a single document."""
    for node in nodes:
        if node.embedding is None:
            raise ValueError(f"Node {node.node_id} has no embedding; run embed_nodes first.")
 
        page_no = None
        doc_items = node.metadata.get("doc_items")
        if doc_items:
            prov = doc_items[0].get("prov")
            if prov:
                page_no = prov[0].get("page_no")
 
        cur.execute(
            """
            INSERT INTO nodes (
                document_id, chunk_text, embedding, page_no, headings, raw_metadata
            )
            VALUES (%s, %s, %s::vector, %s, %s, %s::jsonb);
            """,
            (
                document_id,
                node.get_content(),
                str(node.embedding),
                page_no,
                node.metadata.get("headings"),
                json.dumps(node.metadata, default=str),
            ),
        )