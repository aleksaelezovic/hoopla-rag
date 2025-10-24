import argparse

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_golden_dataset, load_movies


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    print(f"k={limit}")
    print()

    golden_dataset = load_golden_dataset()
    hs = HybridSearch(load_movies())
    for test_case in golden_dataset:
        res = hs.rrf_search(test_case["query"], 60, limit)
        res_titles = [movie["title"] for movie in res]
        rel_titles = []
        for expected_title in test_case["relevant_docs"]:
            if expected_title in res_titles:
                rel_titles.append(expected_title)
        precision = len(rel_titles) / len(res_titles)
        recall = len(rel_titles) / len(test_case["relevant_docs"])
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {test_case['query']}")
        print(f"  Precision@{limit}: {precision:.4f}")
        print(f"  Recall@{limit}: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Retrieved: {', '.join(res_titles)}")
        print(f"  Relevant: {', '.join(rel_titles)}")
        print()


if __name__ == "__main__":
    main()
