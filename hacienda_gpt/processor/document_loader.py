import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hacienda_gpt.utils import get_openai_api_key


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

    def _load_and_split(self) -> list[Document]:
        return self._create_loader().load_and_split(self._create_text_splitter())

    def process_documents(self) -> None:
        logging.info(f"Loading documents from {self.content_dir}")
        db = FAISS.from_documents(self._load_and_split(), self.embeddings)
        db.save_local(self.output_dir)
        logging.info("Local FAISS index successfully saved")


def process_with_openai(args: dict) -> None:
    processor = DocumentProcessor(OpenAIEmbeddings(api_key=get_openai_api_key()), **args)
    processor.process_documents()


def process_with_gpt4all(args: dict) -> None:
    processor = DocumentProcessor(GPT4AllEmbeddings(), **args)
    processor.process_documents()
