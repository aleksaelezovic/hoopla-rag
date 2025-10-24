import os
from dotenv import load_dotenv
from google import genai
from numpy import s_
from transformers import InfNanRemoveLogitsProcessor

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .types import HybridSearchResult


load_dotenv()


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
        ks_res = self._bm25_search(query, limit * 500)
        ss_res = self.semantic_search.search_chunks(query, limit * 500)
        ks_scores = normalize_scores([score for _, score in ks_res])
        ss_scores = normalize_scores([res["score"] for res in ss_res])

        res: dict[int, HybridSearchResult] = {}
        for i, r in enumerate(ks_res):
            doc, score = r
            res[doc["id"]] = {
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "score_bm25": ks_scores[i],
                "score_semantic": 0.0,
            }
        for i, r in enumerate(ss_res):
            if r["id"] not in res:
                res[r["id"]] = {
                    "id": r["id"],
                    "title": r["title"],
                    "description": r["description"],
                    "score_bm25": 0.0,
                    "score_semantic": ss_scores[i],
                }
            else:
                res[r["id"]]["score_semantic"] = ss_scores[i]
        for doc_id in res:
            res[doc_id]["score_hybrid"] = (
                alpha * res[doc_id]["score_bm25"]
                + (1 - alpha) * res[doc_id]["score_semantic"]
            )
        sorted_res = sorted(res.values(), key=lambda x: x["score_hybrid"], reverse=True)
        return sorted_res[:limit]

    def rrf_search(self, query, k, limit=10):
        ks_res = self._bm25_search(query, limit * 500)
        ss_res = self.semantic_search.search_chunks(query, limit * 500)

        res: dict[int, HybridSearchResult] = {}
        for i, r in enumerate(ks_res, 1):
            doc, score = r
            res[doc["id"]] = {
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "score_bm25": i,
                "score_semantic": 0,
            }
        for i, r in enumerate(ss_res, 1):
            if r["id"] not in res:
                res[r["id"]] = {
                    "id": r["id"],
                    "title": r["title"],
                    "description": r["description"],
                    "score_bm25": 0,
                    "score_semantic": i,
                }
            else:
                res[r["id"]]["score_semantic"] = i
        for doc_id in res:
            rrf_bm25 = (
                (1 / (res[doc_id]["score_bm25"] + k))
                if res[doc_id]["score_bm25"] != 0
                else 0
            )
            rrf_sem = (
                (1 / (res[doc_id]["score_semantic"] + k))
                if res[doc_id]["score_semantic"] != 0
                else 0
            )
            res[doc_id]["score_hybrid"] = rrf_bm25 + rrf_sem
        sorted_res = sorted(res.values(), key=lambda x: x["score_hybrid"], reverse=True)
        return sorted_res[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []
    s_min = float("inf")
    s_max = float("-inf")
    for score in scores:
        s_min = min(s_min, score)
        s_max = max(s_max, score)
    if s_min == s_max:
        return [1.0] * len(scores)
    return [(score - s_min) / (s_max - s_min) for score in scores]


def enhance_query_spell(query: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""
    return client.models.generate_content(
        model="gemini-2.0-flash-001", contents=prompt
    ).text
