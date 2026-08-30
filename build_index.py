"""
build_index.py — Task 3: embed and index both chunking strategies into two
separate ChromaDB collections.

Run once initially, and again whenever data/knowledge_base/ content changes:
    python build_index.py

Requires: chromadb, sentence-transformers (see requirements.txt).
"""

import glob
import os
from typing import Dict

import chromadb
from sentence_transformers import SentenceTransformer

from app.chunking import chunk_fixed_size, chunk_by_sentence

KB_DIR = "data/knowledge_base"
CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

FIXED_COLLECTION_NAME = "ola_policies_fixed"
SENTENCE_COLLECTION_NAME = "ola_policies_sentence"


def load_documents(kb_dir: str = KB_DIR) -> Dict[str, str]:
    """Loads every .md file in kb_dir, keyed by filename (without extension) as doc_id."""
    docs = {}
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            docs[doc_id] = f.read()
    return docs


def _upsert(collection, chunks, model: SentenceTransformer) -> None:
    if not chunks:
        return
    embeddings = model.encode([c.text for c in chunks]).tolist()
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{"doc_id": c.doc_id, "strategy": c.strategy} for c in chunks],
    )


def build() -> None:
    print(f"Loading knowledge base from {KB_DIR}...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents: {list(docs.keys())}")
    if len(docs) < 12:
        print(f"WARNING: brief requires >=12 knowledge-base documents; found {len(docs)}.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # IMPORTANT: explicitly request cosine distance space. Chroma's default is
    # squared L2, and app/rag.py's similarity conversion (1 - distance) is only
    # valid for cosine space — mismatching this silently produces nonsensical
    # "similarity" values that are not actually bounded in [-1, 1].
    fixed_collection = client.get_or_create_collection(
        FIXED_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    sentence_collection = client.get_or_create_collection(
        SENTENCE_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    fixed_chunks = []
    sentence_chunks = []
    for doc_id, text in docs.items():
        fixed_chunks.extend(chunk_fixed_size(doc_id, text))
        sentence_chunks.extend(chunk_by_sentence(doc_id, text))

    print(f"Fixed-size strategy produced {len(fixed_chunks)} chunks.")
    print(f"Sentence-based strategy produced {len(sentence_chunks)} chunks.")

    _upsert(fixed_collection, fixed_chunks, model)
    _upsert(sentence_collection, sentence_chunks, model)

    print(
        f"Indexed into '{FIXED_COLLECTION_NAME}' ({fixed_collection.count()} items) "
        f"and '{SENTENCE_COLLECTION_NAME}' ({sentence_collection.count()} items)."
    )


if __name__ == "__main__":
    build()