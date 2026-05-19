import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

from hacienda_gpt.utils import get_openai_api_key

HEADER_SPLITTER = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "section"), ("h2", "section"), ("h3", "section")])


class DocumentProcessor:
    def __init__(
        self,
        embeddings: Embeddings,
        content_dir: str,
        output_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        glob: str = "**/*.html",
    ) -> None:
        self.embeddings = embeddings
        self.content_dir = content_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.glob = glob

    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def _create_loader(self) -> DirectoryLoader:
        return DirectoryLoader(path=self.content_dir, glob=self.glob, use_multithreading=True, show_progress=True)

    def _parse_document_type(self, source_url: str) -> str:
        url = source_url.lower()
        if "faq" in url or "preguntas-frecuentes" in url:
            return "faq"
        if "manual" in url or "folletos" in url:
            return "manual"
        if "modelo" in url or "normativa" in url or "ley" in url:
            return "normativa"
        return "tramite"

    def _extract_last_updated(self, text: str) -> str | None:
        match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
        return match.group(1) if match else None

    def _extract_title(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[0][:180] if lines else "sin_titulo"

    def _enrich_metadata(self, doc: Document, section: str | None = None) -> dict[str, Any]:
        source_url = doc.metadata.get("source", "")
        return {
            **doc.metadata,
            "source_url": source_url,
            "title": self._extract_title(doc.page_content),
            "section": section or doc.metadata.get("section") or "general",
            "last_updated": self._extract_last_updated(doc.page_content),
            "document_type": self._parse_document_type(source_url),
        }

    def _semantic_split(self, doc: Document) -> list[Document]:
        source = doc.metadata.get("source", "")
        if source.lower().endswith(".html"):
            try:
                sem_chunks = HEADER_SPLITTER.split_text(doc.page_content)
                return [
                    Document(page_content=chunk.page_content, metadata=self._enrich_metadata(doc, chunk.metadata.get("section")))
                    for chunk in sem_chunks
                ]
            except Exception:
                logging.warning("Semantic HTML split failed for %s, falling back to recursive splitter", source)

        recursive_chunks = self._create_text_splitter().split_documents([doc])
        return [Document(page_content=chunk.page_content, metadata=self._enrich_metadata(chunk)) for chunk in recursive_chunks]

    def _load_and_split(self) -> list[Document]:
        loaded_docs = self._create_loader().load()
        chunks: list[Document] = []
        for doc in loaded_docs:
            chunks.extend(self._semantic_split(doc))
        return chunks

    def process_documents(self) -> None:
        logging.info("Loading documents from %s", self.content_dir)
        documents = self._load_and_split()
        logging.info("Loaded %d chunks for indexing", len(documents))
        db = FAISS.from_documents(documents, self.embeddings)
        db.save_local(self.output_dir)
        logging.info("Local FAISS index successfully saved")


def process_with_openai(args: dict) -> None:
    processor = DocumentProcessor(OpenAIEmbeddings(api_key=get_openai_api_key()), **args)
    processor.process_documents()


def process_with_gpt4all(args: dict) -> None:
    processor = DocumentProcessor(GPT4AllEmbeddings(), **args)
    processor.process_documents()
