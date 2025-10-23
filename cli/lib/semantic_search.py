import os
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import cosine
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies
from .types import Movie, SearchResult


class SemanticSearch:
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.embeddings_path: str = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        self.documents: list[Movie] = []
        self.document_map: dict[int, Movie] = {}

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        s: list[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            s.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(s, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        text = text.strip()
        if len(text) == 0:
            raise ValueError("Text cannot be empty")
        return self.model.encode([text])[0]

    def search(self, query: str, limit: int) -> list[SearchResult]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        results = map(
            lambda x: (cosine_similarity(query_embedding, x[1]), self.documents[x[0]]),
            enumerate(self.embeddings, 0),
        )
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        return list(
            map(
                lambda x: {
                    "score": x[0],
                    "title": x[1]["title"],
                    "description": x[1]["description"],
                },
                sorted_results[:limit],
            )
        )


def chunk(text: str, chunk_size: int = 200, overlap: int = 0):
    chunks: list[list[str]] = []
    for word in text.split(" "):
        if len(chunks) == 0:
            chunks.append([])
        if len(chunks[-1]) == chunk_size:
            chunks.append([])
            if overlap > 0:
                chunks[-1].extend(chunks[-2][-overlap:])
        chunks[-1].append(word)
    return chunks


def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    ss = SemanticSearch()
    _ = ss.load_or_create_embeddings(load_movies())
    for idx, result in enumerate(ss.search(query, limit), 1):
        print(f"{idx}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def cosine_similarity(vec1: ArrayLike, vec2: ArrayLike):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
