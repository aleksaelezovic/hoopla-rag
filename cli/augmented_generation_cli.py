import argparse
import os
import json
from dotenv import load_dotenv
from google import genai

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


_ = load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    _ = rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            query = args.query
            res = HybridSearch(load_movies()).rrf_search(query, 60, 5)
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Documents:
            {json.dumps(res)}

            Provide a comprehensive answer that addresses the query:"""
            rag_response = client.models.generate_content(
                model="gemini-2.0-flash-001", contents=prompt
            ).text

            print("Search Results:")
            for doc in res:
                print(f"- {doc.get('title', '<error unknown title>')}")
            print()

            print("RAG Response:")
            print(rag_response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
