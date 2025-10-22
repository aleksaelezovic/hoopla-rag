import os
import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies
from .types import Movie


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


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


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
