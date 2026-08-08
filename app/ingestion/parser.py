from llama_index.core import Document
from docling.chunking import HybridChunker
from app.config.settings import EMBED_MODEL_NAME,MAX_TOKEN
from llama_index.node_parser.docling import DoclingNodeParser
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from llama_index.core.schema import MetadataMode

def create_node_parser(
    tokenizer_name: str = EMBED_MODEL_NAME,
    max_tokens: int = MAX_TOKEN,
) -> DoclingNodeParser:
    hf_tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_name),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(tokenizer=hf_tokenizer)
    return DoclingNodeParser(chunker=chunker)


def parse_documents(
    document: Document,
    node_parser: DoclingNodeParser,
):
    """
    Parse documents into TextNodes.
    """

    nodes = node_parser.get_nodes_from_documents([document])
    

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    overheads = []
    for node in nodes:  # real nodes from this run
        raw = len(tokenizer.encode(node.get_content()))
        with_meta = len(tokenizer.encode(node.get_content(metadata_mode=MetadataMode.EMBED)))
        overheads.append(with_meta - raw)
        print(f"raw={raw}, with_meta={with_meta}, overhead={with_meta - raw}")

        print("max overhead:", max(overheads))

    return nodes