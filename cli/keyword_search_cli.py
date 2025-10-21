#!/usr/bin/env python3

import argparse
import json
import string


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
                if kw_match(movie["title"], args.query):
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


def kw_match(s, q):
    s = to_kw_tokens(s)
    q = to_kw_tokens(q)
    for tok_q in q:
        for tok_s in s:
            if tok_q in tok_s:
                return True
    return False


def to_kw_tokens(s):
    kw_tokens = []
    for tok in s.split(" "):
        tok = to_kw_comparable(tok)
        if len(tok) != 0:
            kw_tokens.append(tok)
    return kw_tokens


def to_kw_comparable(s):
    s = s.lower()
    t = s.maketrans("", "", string.punctuation)
    return s.translate(t)


if __name__ == "__main__":
    main()
