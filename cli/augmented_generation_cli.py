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

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate a summary of a search"
    )
    _ = summarize_parser.add_argument("query", type=str, help="Search Query")
    _ = summarize_parser.add_argument(
        "--limit", help="Search results limit", type=int, default=5
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Generate a summary with citations for a search"
    )
    _ = citations_parser.add_argument("query", type=str, help="Search Query")
    _ = citations_parser.add_argument(
        "--limit", help="Search results limit", type=int, default=5
    )

    question_parser = subparsers.add_parser(
        "question", help="Generate an answer to a question"
    )
    _ = question_parser.add_argument("question", type=str, help="Question")
    _ = question_parser.add_argument(
        "--limit", help="Search results limit", type=int, default=5
    )

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
        case "summarize":
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            query = args.query
            limit = args.limit
            res = HybridSearch(load_movies()).rrf_search(query, 60, limit)
            prompt = f"""
            Provide information useful to this query by synthesizing information from multiple search results in detail.
            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
            This should be tailored to Hoopla users. Hoopla is a movie streaming service.
            Query: {query}
            Search Results:
            {json.dumps(res)}
            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
            """
            summary = client.models.generate_content(
                model="gemini-2.0-flash-001", contents=prompt
            ).text

            print("Search Results:")
            for doc in res:
                print(f"- {doc.get('title', '<error unknown title>')}")
            print()

            print("LLM Summary:")
            print(summary)
        case "citations":
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            query = args.query
            limit = args.limit
            res = HybridSearch(load_movies()).rrf_search(query, 60, limit)
            prompt = f"""Answer the question or provide information based on the provided documents.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

            Query: {query}

            Documents:
            {json.dumps(res)}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources using [1], [2], etc. format when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
            summary = client.models.generate_content(
                model="gemini-2.0-flash-001", contents=prompt
            ).text

            print("Search Results:")
            for doc in res:
                print(f"- {doc.get('title', '<error unknown title>')}")
            print()

            print("LLM Answer:")
            print(summary)
        case "question":
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            question = args.question
            limit = args.limit
            res = HybridSearch(load_movies()).rrf_search(question, 60, limit)
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Question: {question}

            Documents:
            {json.dumps(res)}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
            answer = client.models.generate_content(
                model="gemini-2.0-flash-001", contents=prompt
            ).text

            print("Search Results:")
            for doc in res:
                print(f"- {doc.get('title', '<error unknown title>')}")
            print()

            print("Answer:")
            print(answer)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
