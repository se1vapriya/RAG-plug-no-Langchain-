"""Query the knowledge base from the terminal.

    python ask.py "What is the leave policy?"
    python ask.py                 # interactive loop
    python ask.py -q "..." -k 8 --show-sources
"""

import argparse

from config import config
from pipeline import ask


def show(query: str, top_k: int, namespace: str, show_sources: bool):
    result = ask(query, top_k=top_k, namespace=namespace)
    print("\n" + result["answer"] + "\n")
    if show_sources and result["hits"]:
        print("Sources:")
        for i, hit in enumerate(result["hits"], start=1):
            page = f" p.{hit['page']}" if hit.get("page") else ""
            print(f"  [{i}] {hit['source']}{page}  (score {hit['score']:.3f})")
        print()


def main():
    parser = argparse.ArgumentParser(description="Ask the knowledge base a question.")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("-q", "--query", default=None)
    parser.add_argument("-k", "--top-k", type=int, default=config.top_k)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()

    config.validate()
    question = args.question or args.query

    if question:
        show(question, args.top_k, args.namespace, args.show_sources)
        return

    print("Ask a question (blank line or Ctrl-C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        show(q, args.top_k, args.namespace, True)


if __name__ == "__main__":
    main()
