from PIL import Image
from sentence_transformers import SentenceTransformer

from .types import Movie, MultimodalSearchResult
from .semantic_search import cosine_similarity
from .search_utils import load_movies


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents=list[Movie]):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        f = Image.open(image_path)
        return self.model.encode([f])[0]

    def search_with_image(self, image_path: str):
        emb_img = self.embed_image(image_path)
        results: list[MultimodalSearchResult] = [
            {
                "id": self.documents[i]["id"],
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
                "score": cosine_similarity(emb_img, emb_txt),
            }
            for i, emb_txt in enumerate(self.text_embeddings)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:5]


def image_search_command(image_path: str):
    mms = MultimodalSearch(documents=load_movies())
    return mms.search_with_image(image_path)


def verify_image_embedding(image_path: str):
    embedding = MultimodalSearch(documents=load_movies()).embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
