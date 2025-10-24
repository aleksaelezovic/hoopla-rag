import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import json
from numpy import s_
from transformers import InfNanRemoveLogitsProcessor
from sentence_transformers import CrossEncoder

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
                "score_rerank": 0,
                "score_eval": -1,
            }
        for i, r in enumerate(ss_res):
            if r["id"] not in res:
                res[r["id"]] = {
                    "id": r["id"],
                    "title": r["title"],
                    "description": r["description"],
                    "score_bm25": 0.0,
                    "score_semantic": ss_scores[i],
                    "score_rerank": 0,
                    "score_eval": -1,
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
                "score_rerank": 0,
                "score_eval": -1,
            }
        for i, r in enumerate(ss_res, 1):
            if r["id"] not in res:
                res[r["id"]] = {
                    "id": r["id"],
                    "title": r["title"],
                    "description": r["description"],
                    "score_bm25": 0,
                    "score_semantic": i,
                    "score_rerank": 0,
                    "score_eval": -1,
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


def enhance_query(query: str, method: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    if method == "spell":
        prompt = f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.
        Corrected:"""
    elif method == "rewrite":
        prompt = f"""Rewrite this movie search query to be more specific and searchable.

        Original: "{query}"

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep it concise (under 10 words)
        - It should be a google style search query that's very specific
        - Don't use boolean logic

        Examples:

        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        Rewritten query:"""
    elif method == "expand":
        prompt = f"""Expand this movie search query with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        This will be appended to the original query.

        Examples:

        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        Query: "{query}"
        """
    else:
        raise ValueError(f"Unknown enhancement method: {method}")
    return client.models.generate_content(
        model="gemini-2.0-flash-001", contents=prompt
    ).text


def rerank(query, res: list[HybridSearchResult], method: str, limit: int):
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    if method == "batch":
        doc_list_str = json.dumps(res)
        prompt = f"""Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else.
        Do not use markdown. Do not add a trailing comma into array.
        For example:

        [75, 12, 34, 2, 1]
        """
        ids_str = client.models.generate_content(
            model="gemini-2.0-flash-001", contents=prompt
        ).text
        ids: list[int] = json.loads(ids_str)
        res.sort(key=lambda x: ids.index(x["id"]))
        return res[:limit]
    elif method == "individual":
        for doc in res:
            prompt = f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:"""
            score_str = client.models.generate_content(
                model="gemini-2.0-flash-001", contents=prompt
            ).text
            score = float(score_str)
            doc["score_rerank"] = score
            time.sleep(3)
        res.sort(key=lambda x: x["score_rerank"], reverse=True)
        return res[:limit]
    elif method == "cross_encoder":
        pairs: list[list[str]] = []
        for doc in res:
            pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        scores = cross_encoder.predict(pairs)
        for doc, score in zip(res, scores):
            doc["score_rerank"] = score
        return sorted(res, key=lambda x: x["score_rerank"], reverse=True)[:limit]
    else:
        raise ValueError(f"Unknown reranking method: {method}")


def evaluate_results(query: str, results: list[HybridSearchResult]) -> list[dict]:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    formatted_results = map(
        lambda doc: f"{doc.get('title', '')} - {doc.get('description', '')}", results
    )

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers out than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else.
    Do not use markdown. Do not add a trailing comma into array.
    For example:

    [2, 0, 3, 2, 0, 1]"""

    scores_str = client.models.generate_content(
        model="gemini-2.0-flash-001", contents=prompt
    ).text
    scores: list[int] = json.loads(scores_str)
    for doc, score in zip(results, scores):
        doc["score_eval"] = score
    return results
