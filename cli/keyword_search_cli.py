#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
    bm25_idf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency of a term"
    )
    idf_parser.add_argument("term", type=str, help="Term to search for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF of a term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            print(f"Term frequency of '{args.term}' in document {args.doc_id}:")
            tf = tf_command(args.doc_id, args.term)
            print(tf)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
