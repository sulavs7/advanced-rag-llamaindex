from app.config.settings import EMBED_MODEL_NAME
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import MetadataMode,TextNode

def create_embedding_model(
    model_name:str = EMBED_MODEL_NAME
)->HuggingFaceEmbedding:
    """
    Create the embedding model.
    """
    embed_model =  HuggingFaceEmbedding(model_name = model_name)
    return embed_model
    
def embed_nodes(embed_model: HuggingFaceEmbedding, nodes: list[TextNode]) -> list[TextNode]:
    """
    Embed nodes in place using metadata-enriched content (headings, etc. get
    folded into the embedded text, which is the point of MetadataMode.EMBED).
    """
    for node in nodes:
        text = node.get_content(metadata_mode=MetadataMode.EMBED)
        node.embedding = embed_model.get_text_embedding(text)
    return nodes