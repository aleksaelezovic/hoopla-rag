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
    score_rerank: float
    score_hybrid: float
    score_semantic: float
    score_bm25: float
    id: int
    title: str
    description: str
