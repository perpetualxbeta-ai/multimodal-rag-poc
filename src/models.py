"""Shared data models for the multimodal RAG pipeline."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Namespace for deterministic point IDs (Qdrant needs a UUID or an unsigned int).
_POINT_NAMESPACE = uuid.UUID("6f1c2a52-3b5e-4c8e-9d1a-7b2f0e4c9a10")


class ChunkType(StrEnum):
    """Kind of content a chunk carries."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


_TYPE_ALIASES: dict[str, ChunkType] = {
    "text": ChunkType.TEXT,
    "paragraph": ChunkType.TEXT,
    "narrativetext": ChunkType.TEXT,
    "table": ChunkType.TABLE,
    "image": ChunkType.IMAGE,
    "figure": ChunkType.IMAGE,
    "img": ChunkType.IMAGE,
}

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "chunk_type": ("chunk_type", "type", "kind", "modality"),
    "content": ("content", "text", "markdown", "page_content"),
    "page_number": ("page_number", "page", "page_num"),
    "image_path": ("image_path", "path", "figure_path"),
    "chunk_id": ("chunk_id", "id"),
}


@dataclass(slots=True, frozen=True)
class Chunk:
    """A unit of extracted content ready for embedding.

    Attributes:
        chunk_type: Text, table or image.
        content: Plain text, a Markdown table, or (for images) optional alt/caption text.
        page_number: 1-based page the chunk came from.
        image_path: Path to the rendered figure PNG (images only).
        chunk_id: Stable identifier from the extractor, if any.
        metadata: Any extra fields (doc_id, bbox, source_path, ...); stored in the payload.
    """

    chunk_type: ChunkType
    content: str
    page_number: int
    image_path: str | None = None
    chunk_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        """Deterministic UUID so re-indexing the same chunk overwrites instead of duplicating."""
        if self.chunk_id:
            key = self.chunk_id
        else:
            source = str(self.metadata.get("source_path") or self.metadata.get("doc_id") or "")
            key = "|".join(
                (source, self.chunk_type.value, str(self.page_number), self.image_path or "", self.content)
            )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return str(uuid.uuid5(_POINT_NAMESPACE, digest))

    @classmethod
    def from_any(cls, obj: object) -> Chunk:
        """Build a Chunk from a Chunk, a mapping, or any object with matching attributes.

        Accepts common field-name variants (e.g. ``type``/``chunk_type``, ``page``/``page_number``,
        ``text``/``content``) so extractor output can be passed in directly.

        Raises:
            ValueError: If the chunk type or page number is missing or unrecognised.
        """
        if isinstance(obj, Chunk):
            return obj

        def get(name: str) -> Any:
            for alias in _FIELD_ALIASES[name]:
                if isinstance(obj, Mapping):
                    if alias in obj and obj[alias] is not None:
                        return obj[alias]
                elif getattr(obj, alias, None) is not None:
                    return getattr(obj, alias)
            return None

        raw_type = get("chunk_type")
        if raw_type is None:
            raise ValueError(f"Chunk is missing a type field: {obj!r:.200}")
        type_key = str(getattr(raw_type, "value", raw_type)).lower()
        chunk_type = _TYPE_ALIASES.get(type_key)
        if chunk_type is None:
            raise ValueError(f"Unrecognised chunk type {raw_type!r}")

        page = get("page_number")
        if page is None:
            raise ValueError(f"Chunk is missing a page number: {obj!r:.200}")

        image_path = get("image_path")
        chunk_id = get("chunk_id")

        known = {alias for aliases in _FIELD_ALIASES.values() for alias in aliases}
        if isinstance(obj, Mapping):
            extra: dict[str, Any] = {str(k): v for k, v in obj.items() if k not in known}
            nested = obj.get("metadata")
        else:
            extra = {}
            nested = getattr(obj, "metadata", None)
        if isinstance(nested, Mapping):
            extra.pop("metadata", None)
            extra.update({str(k): v for k, v in nested.items()})

        return cls(
            chunk_type=chunk_type,
            content=str(get("content") or ""),
            page_number=int(page),
            image_path=str(image_path) if image_path is not None else None,
            chunk_id=str(chunk_id) if chunk_id is not None else None,
            metadata=extra,
        )
