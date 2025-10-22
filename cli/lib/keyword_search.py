import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.corpus.reader import math
from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
    BM25_K1,
    BM25_B,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.index_path: str = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path: str = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path: str = os.path.join(
            CACHE_DIR, "term_frequencies.pkl"
        )
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path: str = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        self.term_frequencies[doc_id] = Counter()
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
            if token not in self.term_frequencies[doc_id]:
                self.term_frequencies[doc_id][token] = 0
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise Exception("Invalid term - expected a single token")
        if doc_id not in self.term_frequencies:
            raise Exception("Document not found")
        token = tokens[0]
        if token not in self.term_frequencies[doc_id]:
            return 0
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = 0
        for doc_id in self.term_frequencies:
            if self.get_tf(doc_id, term) > 0:
                term_doc_count += 1
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = 0
        for doc_id in self.term_frequencies:
            if self.get_tf(doc_id, term) > 0:
                term_doc_count += 1
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        len_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        return ((k1 + 1) * tf) / (tf + k1 * len_norm)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    try:
        idx = InvertedIndex()
        idx.load()
        query_tokens = tokenize_text(query)
        docs_ids = []
        limit = 5
        for token in query_tokens:
            docs_ids.extend(idx.get_documents(token))
            if len(docs_ids) >= limit:
                break
        return list(map(lambda doc_id: idx.docmap[doc_id], docs_ids[:limit]))

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def tf_command(doc_id: int, term: str) -> int:
    try:
        idx = InvertedIndex()
        idx.load()
        return idx.get_tf(doc_id, term)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def idf_command(term: str) -> float:
    try:
        idx = InvertedIndex()
        idx.load()
        return idx.get_idf(term)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def bm25_idf_command(term: str) -> float:
    try:
        idx = InvertedIndex()
        idx.load()
        return idx.get_bm25_idf(term)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    try:
        idx = InvertedIndex()
        idx.load()
        return idx.get_bm25_tf(doc_id, term, k1, b)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def bm25_command(doc_id: int, term: str) -> float:
    try:
        idx = InvertedIndex()
        idx.load()
        tf = idx.get_bm25_tf(doc_id, term)
        idf = idx.get_bm25_idf(term)
        return tf * idf
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def tfidf_command(doc_id: int, term: str) -> float:
    try:
        idx = InvertedIndex()
        idx.load()
        tf = idx.get_tf(doc_id, term)
        idf = idx.get_idf(term)
        return tf * idf
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
