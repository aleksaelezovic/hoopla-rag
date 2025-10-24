import os

from numpy import s_
from transformers import InfNanRemoveLogitsProcessor

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores):
    s_min = float("inf")
    s_max = float("-inf")
    for score in scores:
        s_min = min(s_min, score)
        s_max = max(s_max, score)
    if s_min == s_max:
        return [1.0] * len(scores)
    return [(score - s_min) / (s_max - s_min) for score in scores]
