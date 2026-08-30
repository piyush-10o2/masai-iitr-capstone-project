"""
app/rag.py — Task 4: grounded generation + empirical threshold calibration.

Given a query, retrieves the top-k chunks from a chosen ChromaDB collection and
generates an answer using ONLY the retrieved context. Under MOCK_LLM there is
no real model to judge groundedness, so retrieval similarity IS the signal:
run calibrate_threshold() against real in-scope and out-of-scope queries
BEFORE trusting GROUNDEDNESS_SIMILARITY_THRESHOLD in app/config.py — the value
there ships as an untested 0.0 placeholder on purpose, so it can't be
mistaken for a calibrated result.
"""

from dataclasses import dataclass
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

from build_index import CHROMA_DIR, EMBEDDING_MODEL_NAME
from app.config import GROUNDEDNESS_SIMILARITY_THRESHOLD

_model = None
_client = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    similarity: float  # cosine similarity in [-1, 1]; higher = more similar


def _cosine_from_distance(distance: float) -> float:
    # Valid ONLY because build_index.py creates both collections with
    # metadata={"hnsw:space": "cosine"}. In that space, Chroma's returned
    # "distance" is (1 - cosine_similarity), so similarity = 1 - distance.
    return 1.0 - distance


def retrieve(collection_name: str, query: str, top_k: int = 3) -> List[RetrievedChunk]:
    model = _get_model()
    client = _get_client()
    collection = client.get_collection(collection_name)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    retrieved = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
        retrieved.append(RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            text=text,
            similarity=_cosine_from_distance(dist),
        ))
    return retrieved


@dataclass
class GroundedAnswer:
    query: str
    answer: str
    grounded: bool
    top_similarity: float
    retrieved_chunks: List[RetrievedChunk]


def grounded_generate(collection_name: str, query: str, top_k: int = 3,
                       threshold: float = GROUNDEDNESS_SIMILARITY_THRESHOLD) -> GroundedAnswer:
    """
    Task 4. Retrieves top-k chunks; if the best match's similarity is below
    `threshold`, refuses with an "I don't know" fallback rather than guessing.

    Under MOCK_LLM, the "generated answer" is a deterministic template built
    directly from retrieved context — intentional, not a stand-in for real
    generation. This keeps the groundedness property mechanically true: the
    answer literally cannot contain anything not in the retrieved chunks.
    """
    chunks = retrieve(collection_name, query, top_k=top_k)
    top_similarity = chunks[0].similarity if chunks else 0.0

    if not chunks or top_similarity < threshold:
        return GroundedAnswer(
            query=query,
            answer="I don't have enough information in the knowledge base to answer that confidently.",
            grounded=False,
            top_similarity=top_similarity,
            retrieved_chunks=chunks,
        )

    context = " ".join(c.text for c in chunks)
    answer = f"Based on Ola's policy documents: {context}"
    return GroundedAnswer(
        query=query,
        answer=answer,
        grounded=True,
        top_similarity=top_similarity,
        retrieved_chunks=chunks,
    )


def calibrate_threshold(collection_name: str, in_scope_queries: List[str],
                         out_of_scope_queries: List[str], top_k: int = 1) -> Dict:
    """
    Task 4 calibration step. Measures top-1 cosine similarity for every query
    in both lists and prints them so a threshold can be chosen between the two
    observed clusters. Do NOT ship an untested preset (0.5/0.6/0.7) — those
    don't reliably separate short policy-sentence embeddings from unrelated
    queries; use the numbers this function actually measures.
    """
    print(f"\n=== Calibrating against collection: {collection_name} ===")
    in_scope_sims = []
    for q in in_scope_queries:
        chunks = retrieve(collection_name, q, top_k=top_k)
        sim = chunks[0].similarity if chunks else 0.0
        in_scope_sims.append(sim)
        print(f"[in-scope]     {q!r:65s} -> top-1 similarity = {sim:.4f}")

    out_of_scope_sims = []
    for q in out_of_scope_queries:
        chunks = retrieve(collection_name, q, top_k=top_k)
        sim = chunks[0].similarity if chunks else 0.0
        out_of_scope_sims.append(sim)
        print(f"[out-of-scope] {q!r:65s} -> top-1 similarity = {sim:.4f}")

    in_min, in_max = min(in_scope_sims), max(in_scope_sims)
    out_min, out_max = min(out_of_scope_sims), max(out_of_scope_sims)
    print(f"\nIn-scope cluster:     {in_min:.4f} - {in_max:.4f}")
    print(f"Out-of-scope cluster: {out_min:.4f} - {out_max:.4f}")

    if in_min <= out_max:
        print("WARNING: clusters overlap — no single threshold cleanly separates them. "
              "Consider more/better in-scope queries, or a different top_k, before finalizing.")

    suggested = (in_min + out_max) / 2
    print(f"Suggested threshold (midpoint between the two clusters): {suggested:.4f}")

    return {
        "in_scope_similarities": in_scope_sims,
        "out_of_scope_similarities": out_of_scope_sims,
        "suggested_threshold": suggested,
    }


if __name__ == "__main__":
    # Manual calibration run — python -m app.rag
    # Replace/extend these with queries that actually probe your 12 KB topics.
    IN_SCOPE_QUERIES = [
        "How long does Ola take to resolve a critical support ticket?",
        "What is the refund policy if my ride is cancelled after pickup?",
        "Who qualifies as a VIP customer?",
        "How are service credits different from a refund?",
        "How long are support tickets kept before deletion?",
    ]
    OUT_OF_SCOPE_QUERIES = [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
    ]

    for collection_name in ["ola_policies_fixed", "ola_policies_sentence"]:
        calibrate_threshold(collection_name, IN_SCOPE_QUERIES, OUT_OF_SCOPE_QUERIES)