from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# GLOBAL RETRIEVER
# =========================================================

retriever = None


# =========================================================
# BUILD / UPDATE DOCUMENT INDEX
# =========================================================

def build_retriever(pdf_paths: List[str]):

    global retriever

    all_docs = []

    # Load multiple PDFs
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Store source filename
        for doc in docs:
            doc.metadata["source_file"] = Path(pdf_path).name

        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No documents were provided.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_docs)

    # Clean text
    for d in chunks:
        d.page_content = (
            d.page_content
            .encode("utf-8", "ignore")
            .decode("utf-8", "ignore")
        )

    # Embeddings
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b"
    )

    # Build FAISS
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return len(all_docs), len(chunks)