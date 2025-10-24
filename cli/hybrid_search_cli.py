import argparse

from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch, enhance_query, normalize_scores, rerank


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument(
        "scores", help="Scores to normalize", type=float, nargs="+"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted search"
    )
    weighted_search_parser.add_argument("query", help="Query string", type=str)
    weighted_search_parser.add_argument(
        "--alpha", help="Alpha constant", type=float, default=0.5
    )
    weighted_search_parser.add_argument(
        "--limit", help="Search results limit", type=int, default=5
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform RRF hybrid search"
    )
    rrf_search_parser.add_argument("query", help="Query string", type=str)
    rrf_search_parser.add_argument("--k", help="K constant", type=int, default=60)
    rrf_search_parser.add_argument(
        "--limit", help="Search results limit", type=int, default=5
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Query reranking method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            print(args.scores)
            if not args.scores or len(args.scores) == 0:
                return
            normalized_scores = normalize_scores(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            res = HybridSearch(load_movies()).weighted_search(
                args.query, args.alpha, args.limit
            )
            for i, r in enumerate(res, 1):
                print(f"{i}. {r['title']}")
                print(f"   Hybrid Score: {r['score_hybrid']:.3f}")
                print(
                    f"   BM25: {r['score_bm25']:.3f}, Semantic: {r['score_semantic']:.3f}"
                )
                print(f"   {r['description'][:100]}...")
        case "rrf-search":
            if args.enhance:
                enhanced_query = enhance_query(args.query, args.enhance)
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n"
                )
                args.query = enhanced_query

            if args.rerank_method:
                args.limit *= 5

            res = HybridSearch(load_movies()).rrf_search(args.query, args.k, args.limit)
            if args.rerank_method:
                print("Results before reranking:")
                for i, r in enumerate(res, 1):
                    print(f"{i}. {r['title']}")
                print()
                res = rerank(args.query, res, args.rerank_method, args.limit // 5)

            for i, r in enumerate(res, 1):
                print(f"{i}. {r['title']}")
                if args.rerank_method == "individual":
                    print(f"   Rerank Score: {r['score_rerank']:.3f}/10")
                if args.rerank_method == "batch":
                    print(f"   Rerank Score: {i}")
                if args.rerank_method == "cross_encoder":
                    print(f"   Cross Encoder Score: {r['score_rerank']:.3f}")
                print(f"   RRF Score: {r['score_hybrid']:.3f}")
                print(
                    f"   BM25 Rank: {r['score_bm25']}, Semantic Rank: {r['score_semantic']}"
                )
                print(f"   {r['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
