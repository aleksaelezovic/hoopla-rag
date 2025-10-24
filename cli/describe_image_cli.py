import argparse
import mimetypes
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai.types import Part

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


_ = load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    _ = parser.add_argument(
        "--image",
        type=str,
        help="Path to image file",
    )
    _ = parser.add_argument(
        "--query",
        type=str,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()
    image_path = args.image
    query = args.query
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    with open(image_path, "rb") as f:
        image_data = f.read()

    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
    prompt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=[
            prompt,
            Part.from_bytes(data=image_data, mime_type=mime),
            Part.from_text(text=query.strip()),
        ],
    )
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
