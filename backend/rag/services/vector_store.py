import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .text_chunker import KnowledgeChunk
from .embeddings import LocalEmbeddingGenerator

logger = logging.getLogger(__name__)

# Attempt to load FAISS; use numpy fallback if unavailable
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    logger.info("FAISS not found in environment; using high-performance numpy vector search fallback.")

class LocalVectorStore:
    """
    Lightweight, embedded local vector store using FAISS IndexFlatIP
    with seamless numpy fallback and JSON metadata caching.
    """
    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.storage_dir = os.path.join(base_dir, 'data')
        else:
            self.storage_dir = storage_dir

        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_file = os.path.join(self.storage_dir, 'index.faiss')
        self.metadata_file = os.path.join(self.storage_dir, 'chunks_metadata.json')
        self.vectors_file = os.path.join(self.storage_dir, 'vectors.npy')

        self.embedding_generator = LocalEmbeddingGenerator()
        self.chunks: List[KnowledgeChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.faiss_index = None

    def build_from_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        """Embed and index a list of knowledge chunks."""
        if not chunks:
            logger.warning("No chunks provided to build vector index.")
            return 0

        self.chunks = chunks
        texts = [f"{c.doc_type} {c.section_title}\n{c.text}" for c in chunks]
        self.vectors = self.embedding_generator.embed_documents(texts)

        dim = self.vectors.shape[1]
        if FAISS_AVAILABLE:
            try:
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.vectors)
            except Exception as e:
                logger.warning(f"FAISS index construction error ({e}); using numpy fallback.")
                self.faiss_index = None

        self.save()
        logger.info(f"Indexed {len(self.chunks)} knowledge chunks into vector store.")
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_type_filter: Optional[str] = None
    ) -> List[Tuple[KnowledgeChunk, float]]:
        """
        Perform vector similarity search against the knowledge store.
        Returns list of (KnowledgeChunk, similarity_score) sorted by relevance.
        """
        if not self.chunks or self.vectors is None:
            return []

        query_vec = self.embedding_generator.embed_query(query).reshape(1, -1)

        # Normalize filter
        norm_filter = None
        if doc_type_filter and doc_type_filter.upper() not in ['ALL', 'NONE', 'UNKNOWN', 'OTHER']:
            norm_filter = doc_type_filter.upper().strip()

        # Compute dot product similarity (vectors are unit normalized, dot product == cosine similarity)
        if FAISS_AVAILABLE and self.faiss_index is not None and not norm_filter:
            scores, indices = self.faiss_index.search(query_vec, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            return results

        # Numpy search with metadata filtering
        sims = np.dot(self.vectors, query_vec.T).flatten()

        candidate_indices = []
        for idx, chunk in enumerate(self.chunks):
            # Include chunks matching document type OR general guidelines
            if norm_filter is None:
                candidate_indices.append(idx)
            else:
                chunk_type = chunk.doc_type.upper()
                if norm_filter in chunk_type or chunk_type in norm_filter or 'GENERAL' in chunk_type:
                    candidate_indices.append(idx)

        if not candidate_indices:
            # Fallback to all chunks if filtered set is empty
            candidate_indices = list(range(len(self.chunks)))

        candidate_scores = [(idx, float(sims[idx])) for idx in candidate_indices]
        # Boost score slightly if chunk doc_type explicitly matches query doc_type
        if norm_filter:
            candidate_scores = [
                (idx, score + (0.25 if norm_filter in self.chunks[idx].doc_type.upper() else 0.0))
                for idx, score in candidate_scores
            ]

        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in candidate_scores[:top_k]:
            results.append((self.chunks[idx], score))

        return results

    def save(self):
        """Persist index and metadata to local filesystem."""
        try:
            # 1. Save metadata JSON
            metadata_list = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doc_type": c.doc_type,
                    "section_title": c.section_title,
                    "text": c.text,
                    "metadata": c.metadata
                }
                for c in self.chunks
            ]
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2)

            # 2. Save vectors npy
            if self.vectors is not None:
                np.save(self.vectors_file, self.vectors)

            # 3. Save FAISS index
            if FAISS_AVAILABLE and self.faiss_index is not None:
                faiss.write_index(self.faiss_index, self.index_file)
        except Exception as e:
            logger.warning(f"Could not persist vector store cache: {e}")

    def load(self) -> bool:
        """Load persisted index and metadata from local filesystem."""
        if not os.path.exists(self.metadata_file) or not os.path.exists(self.vectors_file):
            return False

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)

            self.chunks = [
                KnowledgeChunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    doc_type=item["doc_type"],
                    section_title=item["section_title"],
                    text=item["text"],
                    metadata=item.get("metadata", {})
                )
                for item in metadata_list
            ]

            self.vectors = np.load(self.vectors_file)

            if FAISS_AVAILABLE and os.path.exists(self.index_file):
                self.faiss_index = faiss.read_index(self.index_file)
            elif FAISS_AVAILABLE and self.vectors is not None:
                dim = self.vectors.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.vectors)

            logger.info(f"Loaded {len(self.chunks)} knowledge chunks from local vector store cache.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load vector store cache ({e}). Will re-index from knowledge base.")
            return False
