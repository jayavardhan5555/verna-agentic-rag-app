

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
    
    vs = Chroma(embedding_function= embeddings, collection_name= COLLECTION_NAME, persist_directory= CHROMA_DIR)

    vs.delete_collection()

    vs = Chroma.from_documents(documents = documents, embedding = embeddings, collection_name= COLLECTION_NAME, persist_directory= CHROMA_DIR)

    return vs

def load_vector_store(api_key:str) -> Chroma:

    embeddings = OpenAIEmbeddings(model= "text-embedding-3-small",openai_api_key=api_key)

    return Chroma(embedding_function= embeddings, collection_name= COLLECTION_NAME, persist_directory= CHROMA_DIR)

def get_retriver(vector_store: Chroma, k: int =5):

    return vector_store.as_retriever(search_type ="mmr", search_kwargs ={"k":k,"fetch_k": k *3})

def vector_store_exists() -> bool:
    return os.path.isdir(CHROMA_DIR) and bool(os.listdir(CHROMA_DIR))