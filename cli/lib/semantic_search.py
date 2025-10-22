from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str):
        text = text.strip()
        if len(text) == 0:
            raise ValueError("Text cannot be empty")
        return self.model.encode([text])[0]


def embed_text(text: str):
    embedding = SemanticSearch().generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
