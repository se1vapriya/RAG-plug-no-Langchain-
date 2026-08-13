"""Index a folder of documents.

    python ingest.py                       # index DATA_DIR from .env
    python ingest.py --path ./docs         # index another folder
    python ingest.py --reset               # wipe the namespace first
    python ingest.py --namespace hr-2026    # index into a separate namespace
"""

import argparse

from config import config
from pipeline import ingest


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--path", default=None, help=f"folder to ingest (default: {config.data_dir})")
    parser.add_argument("--namespace", default=None, help="Pinecone namespace")
    parser.add_argument("--reset", action="store_true", help="delete existing vectors first")
    args = parser.parse_args()

    config.validate()
    ingest(folder=args.path, namespace=args.namespace, reset=args.reset)


if __name__ == "__main__":
    main()
