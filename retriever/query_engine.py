from __future__ import annotations
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from config import CONFIG
from .vector_store import VectorStore
from .hybrid_retriever import HybridRetriever


def build_query_engine(
    vector_store: VectorStore,
    alpha: float = 0.5,
    top_k: int = 10,
    rerank_model: str = "rerank-v3.5",
    rerank_top_n: int = 5,
) -> RetrieverQueryEngine:

    # 2 retriever
    vector_ret = VectorIndexRetriever(
        index=vector_store._index, similarity_top_k=top_k  # pylint: disable=protected-access
    )
    keyword_ret = KeywordTableSimpleRetriever(
        index=vector_store.build_keyword_index([]), similarity_top_k=top_k
    )

    # Hybrid
    hybrid_ret = HybridRetriever(
        vector_retriever=vector_ret,
        keyword_retriever=keyword_ret,
        alpha=alpha,
        top_k=top_k,
    )

    # Cohere rerank post-processor
    reranker = CohereRerank(model=rerank_model, top_n=rerank_top_n)

    # QueryEngine
    return RetrieverQueryEngine.from_args(
        retriever=hybrid_ret,
        node_postprocessors=[reranker],
    )
