CREATE TABLE IF NOT EXISTS documents( 
    ID UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    FILE_NAME TEXT NOT NULL,
    FILE_PATH TEXT,
    FILE_TYPE TEXT,
    FILE_SIZE bigint,
    CREATION_DATE TIMESTAMP,
    LAST_MODIFIED_DATE TIMESTAMP,
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384),
    page_no INT,
    headings TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_metadata JSONB
);

CREATE INDEX ON nodes USING hnsw(embedding vector_cosine_ops);

ALTER TABLE documents ADD COLUMN content_hash TEXT UNIQUE;


ALTER TABLE nodes ADD COLUMN text_search tsvector
    GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;

CREATE INDEX ON nodes USING GIN (text_search);