#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer
from os import makedirs
import pickle


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = to_kw_tokens(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.docmap[doc_id] = text

    def get_documents(self, term):
        term = term.lower()
        docs = list(self.index.get(term, set()))
        docs.sort()
        return docs

    def build(self, movies):
        for movie in movies:
            self.__add_document(
                movie["id"], movie["title"] + " " + movie["description"]
            )

    def save(self, dir="cache"):
        makedirs(dir, exist_ok=True)
        pickle.dump(self.index, open(f"{dir}/index.pkl", "wb"))
        pickle.dump(self.docmap, open(f"{dir}/docmap.pkl", "wb"))


stemmer = PorterStemmer()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            f = open("data/movies.json")
            movies = json.load(f)["movies"]
            f.close()
            f = open("data/stopwords.txt")
            stopwords = f.read().splitlines()
            f.close()
            res = []
            for movie in movies:
                if kw_match(movie["title"], args.query, stopwords):
                    res.append(movie)

            i = 0
            res.sort(key=lambda x: x["id"])
            res = res[:5]
            while i < len(res):
                print(f"{i + 1}. {res[i]['title']}")
                i += 1
            pass
        case "build":
            f = open("data/movies.json")
            movies = json.load(f)["movies"]
            f.close()

            idx = InvertedIndex()
            idx.build(movies)
            idx.save()
            docs = idx.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
            pass
        case _:
            parser.print_help()


def kw_match(s, q, stopwords=[]):
    s = to_kw_tokens(s, stopwords)
    q = to_kw_tokens(q, stopwords)
    for tok_q in q:
        for tok_s in s:
            if tok_q in tok_s:
                return True
    return False


def to_kw_tokens(s, stopwords=[]):
    kw_tokens = []
    for tok in s.split(" "):
        tok = to_kw_comparable(tok)
        if len(tok) != 0 and tok not in stopwords:
            kw_tokens.append(stemmer.stem(tok))
    return kw_tokens


def to_kw_comparable(s):
    s = s.lower()
    t = s.maketrans("", "", string.punctuation)
    return s.translate(t)


if __name__ == "__main__":
    main()
