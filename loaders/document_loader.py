from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import csv
import io
import requests
from bs4 import BeautifulSoup

def _load_pdf(filepath:str) -> List[Document]:
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    docs = []
    for i,page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text,metadata ={"source": filepath, "page": i}))
    return docs

_LOADER_MAP = {
    ".pdf": _load_pdf,
    # ".docx": Docx2txtLoader,
    # ".txt": TextLoader,
    # ".csv": CSVLoader,
    # ".md": TextLoader,
    # ".html": BSHTMLLoader
}

def _load_single_fiel(filepath:str) -> List[Document]:
    ext = Path(filepath).suffix.lower()
    loader_fn = _LOADER_MAP.get(ext)
    if loader_fn is None:
        raise ValueError(f"Unsupported file {ext}")
    
    return loader_fn(filepath)


def load_uploaded_files(uploaded_files) -> List[Document]:

    docs:List[Document] = []
    for uf in uploaded_files:
        ext = Path(uf.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        try:
            loaded = _load_single_fiel(tmp_path)

            for doc in loaded:
                doc.metadata.setdefault("source", uf.name)
            docs.extend(loaded)

        finally:
            os.unlink(tmp_path)
    return docs

def load_urls(urls: List[str]) -> List[Document]:
    header = {"User-Agent": "Mozilla/5.0 (compatible; AgenticRAGBot/1.0)"}
                
    docs:List[Document] = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        
        try:
            response = requests.get(url,headers=header,timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml")
            text = soup.get_text(separator="\n", strip=True)
            if text:
                docs.append(Document(page_content=text,metadata={"source": url}))
        except Exception as exc:
            print(f"[document_loader] Failed to load URL")
    return docs

def split_documents(
        docs: List[Document],
        chunk_size: int = 1000, chunk_overlap = 200
) -> List[Document]:
    
    spillter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap
    )

    return spillter.split_documents(docs)