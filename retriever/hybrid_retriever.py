# retriever/hybrid_retriever.py
from __future__ import annotations
from typing import List
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        alpha: float = 0.5,
        top_k: int = 3,
    ):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # lấy node & score
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        keyword_nodes = self.keyword_retriever.retrieve(query_bundle)

        # gộp score theo công thức alpha
        score_map: dict[str, float] = {}

        for n in vector_nodes:
            score_map[n.node.node_id] = self.alpha * (n.score or 0.0)

        for n in keyword_nodes:
            base = score_map.get(n.node.node_id, 0.0)
            score_map[n.node.node_id] = base + (1 - self.alpha) * (n.score or 0.0)

        # gom node, giữ unique, sort và lấy top_k
        unique_nodes = {n.node.node_id: n.node for n in (vector_nodes + keyword_nodes)}
        combined = [
            NodeWithScore(node=unique_nodes[nid], score=s) for nid, s in score_map.items()
        ]
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined[:self.top_k]
