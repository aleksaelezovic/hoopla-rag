from typing import TypedDict


class Movies(TypedDict):
    movies: list[Movie]


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SearchResult(TypedDict):
    score: float
    title: str
    description: str
