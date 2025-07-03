from pathlib import Path
from typing import List
from llama_index.core import (
    VectorStoreIndex, SimpleKeywordTableIndex,
    StorageContext, load_index_from_storage,
)
from llama_index.core.schema import TextNode
from utils.embedding_service import get_embed_model
from config import CONFIG

class VectorStore:
    def __init__(self, dataset_dir: str | None = None):
        self.dataset_dir = Path(dataset_dir or CONFIG["DATASET_PATH"])
        self.index_dir = self.dataset_dir / ".index"
        self.embed_model = get_embed_model()
        self._index = None
        if self.index_dir.exists():
            self._load()
        else:
            self._index = VectorStoreIndex([], embed_model=self.embed_model)

    # build - load - save
    def build_index(self, nodes: List[TextNode], rebuild=False):
        if rebuild and self.index_dir.exists():
            for p in self.index_dir.glob("*"):
                p.unlink()
            self.index_dir.rmdir()
        self._index = VectorStoreIndex(nodes, embed_model=self.embed_model)

    def _load(self):
        ctx = StorageContext.from_defaults(persist_dir=str(self.index_dir))
        self._index = load_index_from_storage(ctx, embed_model=self.embed_model)

    def save(self):
        if not self._index:
            raise RuntimeError("Index chưa build.")
        self.index_dir.mkdir(exist_ok=True)
        self._index.storage_context.persist(persist_dir=str(self.index_dir))

    # retrieve
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if not self._index:
            raise RuntimeError("Index chưa build.")
        retriever = self._index.as_retriever(similarity_top_k=k)
        return [r.text for r in retriever.retrieve(query)]

    # keyword index
    def build_keyword_index(self, nodes: List[TextNode]):
        return SimpleKeywordTableIndex(nodes)