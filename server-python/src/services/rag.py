import os
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import get_settings


class RagService:
    def __init__(self):
        settings = get_settings()

        self.embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )

        # 存放文档向量，支持相似度搜索
        self.vector_store: FAISS | None = None

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        print(
            f"RAG Service initialized with embedding: {settings.ollama_embedding_model}"
        )

    async def init(self, docs_path: str | None = None):
        if docs_path is None:
            docs_path = os.path.join(os.getcwd(), "resources", "docs")

        documents = self._load_documents(docs_path)

        if not documents:
            print(f"No documents found in {docs_path}")
            return

        chunks = self.text_splitter.split_documents(documents)

        # 对每个 chunk 进行 embedding
        self.vector_store = await FAISS.afrom_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        print(f"RAG indexed {len(chunks)} chunks from {len(documents)} documents")

    def _load_documents(self, docs_path: str) -> list[Document]:
        documents = []
        path = Path(docs_path)

        if not path.exists():
            print(f"Docs path not found: {docs_path}")
            return documents

        for file_path in path.rglob("*"):
            if file_path.suffix in [".txt", ".md"]:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    doc = Document(
                        page_content=content, metadata={"source": str(file_path)}
                    )
                    documents.append(doc)
                    print(f"  Loaded: {file_path.name}")
                except Exception as e:
                    print(f"  Failed to load {file_path}: {e}")

        return documents

    async def retrieve(self, query: str, k: int = 3) -> list[Document]:
        if self.vector_store is None:
            return []

        docs = await self.vector_store.asimilarity_search(query, k=k)
        return docs

    async def retrieve_with_score(
        self, query: str, k: int = 3, score_threshold: float = 0.5
    ) -> list[tuple[Document, float]]:
        if self.vector_store is None:
            return []

        # similarity_search_with_score 返回 (doc, distance)
        results = await self.vector_store.asimilarity_search_with_score(query, k=k)

        filtered = []
        for doc, distance in results:
            # distance -> similarity (1 / (1 + distance))
            similarity = 1 / (1 + distance)  # 转换为 0 - 1 的相似度
            if similarity >= score_threshold:
                filtered.append((doc, similarity))

        return filtered


_rag_service: RagService | None = None


def get_rag_service() -> RagService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService()
    return _rag_service
