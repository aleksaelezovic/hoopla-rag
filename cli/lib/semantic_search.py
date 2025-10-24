import json
import os
import re
import numpy as np
from numpy.typing import ArrayLike
from requests.sessions import ChunkedEncodingError
from scipy.stats import cosine
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies
from .types import Movie, SearchResult


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model: SentenceTransformer = SentenceTransformer(model_name)
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
                    "id": x[1]["id"],
                },
                sorted_results[:limit],
            )
        )


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_embeddings_path: str = os.path.join(
            CACHE_DIR, "chunk_embeddings.npy"
        )
        self.chunk_metadata = None
        self.chunk_metadata_path: str = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[Movie]) -> None:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        chunks: list[str] = []
        meta: list[dict] = []
        for doc_idx, doc in enumerate(documents):
            if len(doc["description"]) == 0:
                continue
            doc_chunks = semantic_chunk(doc["description"], 4, 1)
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunks.append(" ".join(chunk))
                meta.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(doc_chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = meta
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": meta, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]
                return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[SearchResult]:
        if self.chunk_embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict] = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            chunk_scores.append(
                {
                    "score": cosine_similarity(query_embedding, chunk_embedding),
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                }
            )
        movie_scores: dict[int, int] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        results = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return list(
            map(
                lambda x: {
                    "score": x[1],
                    "title": self.documents[x[0]]["title"],
                    "description": self.documents[x[0]]["description"][:100],
                    "id": self.documents[x[0]]["id"],
                },
                results[:limit],
            )
        )


def search_chunked(query: str, limit: int = 10):
    css = ChunkedSemanticSearch()
    _ = css.load_or_create_chunk_embeddings(load_movies())
    for i, res in enumerate(css.search_chunks(query, limit), 1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['description']}...")


def embed_chunks():
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(load_movies())
    print(f"Generated {len(embeddings)} chunked embeddings")


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0):
    text = text.strip()
    if len(text) == 0:
        return []
    chunks: list[list[str]] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 0:
        return []
    if len(sentences) == 1 and not sentences[0].endswith((".", "?", "!")):
        return [[text]]
    for sentence in sentences:
        if len(chunks) == 0:
            chunks.append([])
        if len(chunks[-1]) == max_chunk_size:
            chunks.append([])
            if overlap > 0:
                chunks[-1].extend(chunks[-2][-overlap:])
        sentence = sentence.strip()
        if len(sentence) == 0:
            continue
        chunks[-1].append(sentence)
    return chunks


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
