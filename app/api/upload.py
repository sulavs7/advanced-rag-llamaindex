import os
import psycopg2
import hashlib
import logging 
import traceback # required for logger 
from pathlib import Path
from dotenv import load_dotenv
from app.ingestion.loader import load_documents
from app.ingestion.parser import create_node_parser,parse_documents
from app.ingestion.create_embeddings import create_embedding_model,embed_nodes
from app.database.pgvector import *

logging.basicConfig(level = logging.INFO , format ="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DB_URL = os.getenv("SUPABASE_URL")

def compute_file_hash(file_path : Path)->str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()

def document_already_ingested(cur,content_hash:str)->bool:
    cur.execute("SELECT 1 from documents WHERE content_hash=%s;",[content_hash]) #
    return cur.fetchone() is not None


def ingest_file(file_path: Path,node_parser , embed_model ,conn) -> None:
    """
    Full ingestion pipeline.
    Returns the full list of embedded TextNodes actually inserted.
    """
    content_hash = compute_file_hash(file_path=file_path)

    with conn.cursor() as cur:
        if document_already_ingested(cur,content_hash):
            logger.info("Skipping %s — already ingested (hash match)", file_path.name)
            return

    documents = load_documents(file_path)
    for doc in documents:

        doc.metadata["content_hash"] = content_hash
        nodes = parse_documents(doc,node_parser)
        if not nodes:
            logger.warning("No nodes parsed from %s, skipping", file_path.name)
            continue
        nodes = embed_nodes(embed_model, nodes)

        with conn.cursor() as cur:
            document_id = insert_documents_row(cur, doc)
            insert_node_rows(cur, document_id, nodes)
        conn.commit()
 
        logger.info(
                        "Inserted %d node(s) for %s",
                        len(nodes),
                        doc.metadata.get("file_name"),
                    )
                    

def ingestion_path(input_path: Path)->None:
    node_parser = create_node_parser()
    embed_model = create_embedding_model()

    conn = psycopg2.connect(DB_URL)
    try:
        files = [input_path] if input_path.is_file() else sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in {".pdf", ".docx", ".pptx"}
        )
        for f in files:
            try:
                ingest_file(f, node_parser, embed_model, conn)
            except Exception:
                logger.exception("Failed to ingest %s", f.name)
                conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    file_path = Path("uploaded_docs/hgpt.pdf")
    ingestion_path(file_path)