"""Lightweight retrieval service backed by ChromaDB for vegetarian evidence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from config import settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RAGMatch:
    """Single retrieval match returned from the vector store."""

    name: str
    category: str
    description: str
    score: float
    chunk_index: Optional[int] = None

    def as_metadata(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "score": self.score,
            "chunk_index": self.chunk_index if self.chunk_index is not None else "",
        }


class RAGService:
    """Provides retrieval-augmented evidence for ambiguous dishes."""

    def __init__(
        self,
        *,
        client: Optional[ClientAPI] = None,
        persist_path: Optional[Path | str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_function=None,
        seed_path: Optional[Path | str] = None,
        default_top_k: Optional[int] = None,
    ) -> None:
        self.persist_path = Path(persist_path or settings.rag_db_path).resolve()
        self.collection_name = collection_name or settings.rag_collection_name
        self.seed_path = Path(seed_path or settings.rag_seed_path).resolve()
        self.default_top_k = default_top_k or settings.rag_top_k

        if client is not None:
            self.client = client
        else:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self.client = PersistentClient(path=str(self.persist_path))

        if embedding_function is not None:
            self.embedding_function = embedding_function
        else:
            model_name = embedding_model or settings.rag_model_name
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        self.collection: Collection = self._ensure_collection()

        try:
            self._ensure_seed_data()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to seed RAG store: %s", exc)

    def _ensure_seed_data(self) -> None:
        """Seed the collection from the curated JSON file if it is empty."""
        if self.collection.count() > 0:
            return

        if not self.seed_path.exists():
            logger.warning("RAG seed file not found at %s; retrieval context will be empty.", self.seed_path)
            return

        with self.seed_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        documents: List[str] = []
        metadatas: List[dict[str, str]] = []
        ids: List[str] = []

        for index, item in enumerate(payload):
            name = item.get("name")
            category = item.get("category", "unknown")
            description = item.get("description") or ""
            if not name:
                continue

            chunks = self._chunk_text(description)
            for chunk_index, chunk in enumerate(chunks):
                chunk_text = chunk.strip()
                if not chunk_text:
                    continue
                documents.append(f"{name}: {chunk_text}")
                metadatas.append(
                    {
                        "name": name,
                        "category": category,
                        "description": chunk_text,
                        "chunk_index": str(chunk_index),
                    }
                )
                ids.append(f"{category}:{index}:{chunk_index}:{name.lower().replace(' ', '-')}")

        if not documents:
            logger.warning("RAG seed file %s contained no usable entries.", self.seed_path)
            return

        logger.info("Seeding RAG collection '%s' with %d documents.", self.collection_name, len(documents))
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, text: str, *, top_k: Optional[int] = None) -> List[RAGMatch]:
        """Return top matches for the supplied query text."""
        query = text.strip()
        if not query:
            return []

        limit = top_k or self.default_top_k
        response = self.collection.query(
            query_texts=[query],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        matches: List[RAGMatch] = []
        metadatas = response.get("metadatas") or [[]]
        distances = response.get("distances") or [[]]

        for idx, metadata in enumerate(metadatas[0]):
            score = self._score_from_distance(distances[0][idx] if distances and distances[0] else None)
            chunk_raw = metadata.get("chunk_index")
            try:
                chunk_idx = int(chunk_raw) if chunk_raw not in (None, "") else None
            except (TypeError, ValueError):
                chunk_idx = None
            matches.append(
                RAGMatch(
                    name=str(metadata.get("name") or ""),
                    category=str(metadata.get("category") or "unknown"),
                    description=str(metadata.get("description") or ""),
                    score=score,
                    chunk_index=chunk_idx,
                )
            )

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches

    def reseed(self, *, force: bool = False) -> int:
        """Clear and repopulate the collection from the seed file."""
        if force:
            self.client.delete_collection(self.collection_name)
            self.collection = self._ensure_collection()
        elif self.collection.count() > 0:
            logger.info("RAG reseed skipped; collection already populated.")
            return self.collection.count()

        self._ensure_seed_data()
        return self.collection.count()

    @staticmethod
    def _score_from_distance(distance: Optional[float]) -> float:
        """Convert a distance value into a similarity score."""
        if distance is None:
            return 0.0
        score = 1.0 - float(distance)
        return max(0.0, min(score, 1.0))

    @staticmethod
    def build_context(matches: Iterable[RAGMatch]) -> str:
        """Return a compact bullet-style string summarizing retrieval matches."""
        lines: List[str] = ["Evidence from similar dishes:"]
        for match in matches:
            prefix = "Vegetarian" if match.category.lower().startswith("veg") else "Non-vegetarian"
            snippet = match.description.strip()
            lines.append(f"- {prefix}: {match.name} — {snippet} (confidence {match.score:.2f})")
        return "\n".join(lines)

    @staticmethod
    def _chunk_text(text: str, *, max_chars: int = 180) -> List[str]:
        """Split text into reasonably sized chunks for embedding."""
        if not text:
            return [""]

        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + (1 if current else 0)
            if current and current_len + word_len > max_chars:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += word_len

        if current:
            chunks.append(" ".join(current))

        return chunks or [text]
    def _ensure_collection(self) -> Collection:
        """Return the configured collection, creating it if necessary."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
