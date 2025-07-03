from __future__ import annotations
from typing import List
from llama_index.core.schema import QueryBundle
from retriever.query_engine import build_query_engine
from retriever.vector_store import VectorStore
from config import CONFIG

# build global engine (lazy) để tái sử dụng
_vector_store = VectorStore()
_query_engine = build_query_engine(
    _vector_store,
    alpha=0.5,
    top_k=10,
    rerank_top_n=5,
)

def get_context(query: str, top_k: int = 5) -> List[str]:
    """
    Trả về list[str] context (đã rerank), dài top_k.
    """
    bundle = QueryBundle(query_str=query)
    nodes = _query_engine.retrieve(bundle)[:top_k]
    return [n.node.text for n in nodes]
