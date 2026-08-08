from pathlib import Path
from llama_index.readers.docling import DoclingReader
from llama_index.core import Document, SimpleDirectoryReader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


def load_documents(file_path: Path) -> list[Document]:
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist.")
    if not file_path.is_file():
        raise ValueError(f"{file_path} is not a file.")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,   # <-- this is the actual fix
            )
        }
    )

    reader = DoclingReader(
        export_type=DoclingReader.ExportType.JSON,
        doc_converter=doc_converter,
    )

    file_reader = SimpleDirectoryReader(
        input_files=[str(file_path)],
        file_extractor={".pdf": reader, ".docx": reader, ".pptx": reader},
    )
    documents = file_reader.load_data()
    if not documents:
        raise ValueError("No supported documents were found.")
    return documents

if __name__ == "__main__":
    file_path = Path("uploaded_docs/2408.09869v5.pdf")

    documents = load_document(file_path)

    print(type(documents))
    print(documents)