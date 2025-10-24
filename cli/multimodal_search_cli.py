import argparse
import os
import json
from dotenv import load_dotenv
from google import genai

from lib.multimodal_search import verify_image_embedding


_ = load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    _ = verify_image_embedding_parser.add_argument(
        "path", type=str, help="Path to image for embedding verification"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
