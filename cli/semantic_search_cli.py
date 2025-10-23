#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    embed_text,
    search,
    verify_embeddings,
    verify_model,
    chunk,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="Verify the model")

    _ = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed text")
    _ = embed_text_parser.add_argument("text", help="Text to embed")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query")
    _ = embed_query_parser.add_argument("query", help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search for similar texts")
    _ = search_parser.add_argument("query", help="Query to search")
    _ = search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    _ = chunk_parser.add_argument("text", help="Text to chunk")
    _ = chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Size of each chunk",
    )
    _ = chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between chunks",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunks = chunk(args.text, args.chunk_size, args.overlap)
            for i, ch in enumerate(chunks, 1):
                print(f"{i}. {' '.join(ch)}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
