import argparse

from lib.hybrid_search import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument(
        "scores", help="Scores to normalize", type=float, nargs="+"
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
