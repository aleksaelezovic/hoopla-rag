#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            f = open("data/movies.json")
            dic = json.load(f)
            res = []
            for movie in dic["movies"]:
                if args.query in movie["title"]:
                    res.append(movie)
            f.close()

            i = 0
            res.sort(key=lambda x: x["id"])
            res = res[:5]
            while i < len(res):
                print(f"{i+1}. {res[i]['title']}")
                i += 1
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
