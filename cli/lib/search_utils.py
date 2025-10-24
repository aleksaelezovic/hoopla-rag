import json
import os

from .types import GoldenDataset, Movies

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75


def load_movies():
    data: Movies = {"movies": []}
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


def load_golden_dataset():
    data: GoldenDataset = {"test_cases": []}
    with open(GOLDEN_DATASET_PATH, "r") as f:
        data = json.load(f)
    return data["test_cases"]
