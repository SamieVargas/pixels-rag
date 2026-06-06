"""ChromaDB vector store: build and load the local index of day chunks.

Uses ChromaDB's default embedding function (sentence-transformers/all-MiniLM-L6-v2),
which runs locally with no API cost — good enough for this corpus size (~180 rows).
"""

import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pixels"


def build_index(chunks: list[dict]):
    """
    chunks: list of {'id': date_str, 'text': chunk_text, 'metadata': row_dict}
    Builds the ChromaDB collection from scratch.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete and recreate for a clean rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"Indexed {len(chunks)} chunks into ChromaDB")
    return collection


def load_index():
    """Load the existing ChromaDB index."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)
