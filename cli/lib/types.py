from typing import TypedDict


class Movies(TypedDict):
    movies: list[Movie]


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SearchResult(TypedDict):
    score: float
    id: int
    title: str
    description: str


class HybridSearchResult(TypedDict):
    score_eval: int
    score_rerank: float
    score_hybrid: float
    score_semantic: float
    score_bm25: float
    id: int
    title: str
    description: str


class GoldenDataset(TypedDict):
    test_cases: list[GoldenDatasetTestCase]


class GoldenDatasetTestCase(TypedDict):
    query: str
    relevant_docs: list[str]
