

"""Chroma Vectore Store Creation from Langchain."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

import os
from typing import List

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "agentic_rag"

def build_vector_store(documents: list[Document], api_key : str) -> Chroma:
    """Splits docs into chunks , emded them and retun in memory Chorma store"""
    
    embeddings = OpenAIEmbeddings(model= "text-embedding-3-small",openai_api_key=api_key)
    _ensure_chroma_dir()

    vs = Chroma(embedding_function= embeddings, collection_name= COLLECTION_NAME, persist_directory= CHROMA_DIR)

    vs.delete_collection()

    vs = Chroma.from_documents(documents = documents, embedding = embeddings, collection_name= COLLECTION_NAME, persist_directory= CHROMA_DIR)

    return vs

def load_vector_store(api_key:str) -> Chroma:

    embeddings = OpenAIEmbeddings(model= "text-embedding-3-small",openai_api_key=api_key)

    _ensure_chroma_dir()

    return Chroma(embedding_function= embeddings, collection_name= COLLECTION_NAME, persist_directory= CHROMA_DIR)

def get_retriver(vector_store: Chroma, k: int =5):

    return vector_store.as_retriever(search_type ="mmr", search_kwargs ={"k":k,"fetch_k": k *3})

def vector_store_exists() -> bool:
    return os.path.isdir(CHROMA_DIR) and bool(os.listdir(CHROMA_DIR))


def _ensure_chroma_dir() -> None:
    """Ensure the Chroma persist directory exists and is writable.

    Raises a PermissionError with a clear message if the directory is not writable.
    """
    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)
    except Exception as e:
        raise PermissionError(f"Could not create chroma directory '{CHROMA_DIR}': {e}")

    test_path = os.path.join(CHROMA_DIR, ".chroma_write_test")
    try:
        with open(test_path, "w") as fh:
            fh.write("test")
        os.remove(test_path)
    except Exception as e:
        raise PermissionError(
            f"Chroma persist directory '{CHROMA_DIR}' is not writable. Ensure the process has write permission to that path. Original error: {e}"
        )