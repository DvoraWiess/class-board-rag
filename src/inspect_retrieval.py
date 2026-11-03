import argparse
import math
import sys

from src.config import TOP_K, setup_models
from src.indexer import open_index

PREVIEW_CHARS = 60


def retrieve(question, top_k):
    """Retrieve: מחזיר את top_k ה-Nodes הקרובים ביותר לשאלה — בלי LLM."""
    index, _ = open_index()
    return index.as_retriever(similarity_top_k=top_k).retrieve(question)


def score_details(score):
    """Score: LlamaIndex מחזיר exp(-distance). מחשבים בחזרה את המרחק של Chroma ואת דמיון ה-cosine."""
    distance = -math.log(score)
    return distance, 1 - distance


def print_results(question, results):
    print(f"Question: {question}")
    print(f"{'#':<3}{'score':>7}{'dist':>7}{'cos':>7}  date / from / subject / preview")
    for rank, result in enumerate(results, 1):
        distance, cosine = score_details(result.score)
        meta = result.node.metadata
        body = " ".join(result.node.get_content().split())
        print(f"{rank:<3}{result.score:>7.3f}{distance:>7.3f}{cosine:>7.3f}  {meta['date']} | {meta['from']} | {meta['subject']}")
        print(f"{'':<26}{body[:PREVIEW_CHARS]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect: מה נשלף לכל שאלה ובאיזה ציון — בלי LLM")
    parser.add_argument("questions", nargs="+", help="Questions: שאלה אחת או יותר")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Top-k: כמה תוצאות להציג")
    args = parser.parse_args()
    # Why: תוכן הודעות יכול לכלול אימוג'י (יומן הבעיות #4)
    sys.stdout.reconfigure(errors="replace")

    setup_models()
    for question in args.questions:
        print_results(question, retrieve(question, args.top_k))


if __name__ == "__main__":
    main()
