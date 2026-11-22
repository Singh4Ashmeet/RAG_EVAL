"""Parent chunk mapping layer for multi-granularity retrieval and context expansion.

Maintains deterministic mappings between fine-grained retrieval units and their
corresponding parent baseline chunks (ElementAwareChunker 1000/200), preserving
document provenance, isolation, and exact parent chunk representations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.errors import RetrievalError
from core.normalize.schema import Chunk
from core.retrieval.schema import SearchResult


class ParentChunkMapper:
    """Deterministic bidirectional mapping between fine-grained retrieval units and parent chunks.

    Guarantees:
    - Cross-document isolation: child chunks never map to parents in other documents.
    - Provenance preservation: parent chunks maintain their full source_element_ids,
      locations, and verbatim text.
    - Deterministic tie-breaking: ties are broken by parent chunk index / chunk ID.
    - Serializable: state survives indexing and disk persistence.
    """

    def __init__(self) -> None:
        # parent_chunk_id -> Chunk
        self._parents: dict[str, Chunk] = {}
        # child_chunk_id -> parent_chunk_id
        self._child_to_parent: dict[str, str] = {}
        # parent_chunk_id -> list of child_chunk_ids
        self._parent_to_children: dict[str, list[str]] = {}
        # document_id -> list of parent_chunk_ids ordered by chunk_index
        self._document_parents: dict[str, list[str]] = {}

    def register_parent_chunks(self, parents: Sequence[Chunk]) -> None:
        """Register baseline parent chunks for lookup and context expansion."""
        for p in parents:
            self._parents[p.chunk_id] = p
            doc_list = self._document_parents.setdefault(p.document_id, [])
            if p.chunk_id not in doc_list:
                doc_list.append(p.chunk_id)

        # Ensure document parents are sorted deterministically by chunk_index
        for _doc_id, p_ids in self._document_parents.items():
            p_ids.sort(key=lambda cid: (self._parents[cid].chunk_index, cid))

    def register_mapping(self, child_chunk_id: str, parent_chunk_id: str) -> None:
        """Explicitly register a mapping from child_chunk_id to parent_chunk_id."""
        if parent_chunk_id not in self._parents:
            raise RetrievalError(f"Cannot map to unregistered parent_chunk_id: {parent_chunk_id!r}")
        self._child_to_parent[child_chunk_id] = parent_chunk_id
        child_list = self._parent_to_children.setdefault(parent_chunk_id, [])
        if child_chunk_id not in child_list:
            child_list.append(child_chunk_id)

    def map_child_chunks(
        self,
        child_chunks: Sequence[Chunk],
        parent_chunks: Sequence[Chunk] | None = None,
    ) -> dict[str, str]:
        """Map fine-grained child chunks to parent chunks deterministically.

        Uses explicit metadata if present, element provenance matching within the
        same document, and textual overlap fallback for oversized or split elements.
        """
        if parent_chunks:
            self.register_parent_chunks(parent_chunks)

        mapped: dict[str, str] = {}

        for child in child_chunks:
            doc_id = child.document_id
            doc_parent_ids = self._document_parents.get(doc_id, [])
            if not doc_parent_ids:
                # No parent chunks registered for this document
                continue

            # 1. Check explicit metadata
            explicit_parent = child.metadata.get("parent_chunk_id")
            if explicit_parent and explicit_parent in self._parents:
                parent_chunk = self._parents[explicit_parent]
                if parent_chunk.document_id == doc_id:
                    self.register_mapping(child.chunk_id, explicit_parent)
                    mapped[child.chunk_id] = explicit_parent
                    continue

            # 2. Match by source_element_ids provenance within the same document
            child_elems = set(child.source_element_ids)
            matching_parents: list[Chunk] = []
            for pid in doc_parent_ids:
                p = self._parents[pid]
                if set(p.source_element_ids) & child_elems:
                    matching_parents.append(p)

            best_parent: Chunk | None = None
            if len(matching_parents) == 1:
                best_parent = matching_parents[0]
            elif len(matching_parents) > 1:
                # Multiple parent chunks share elements (e.g. split or overlap)
                best_score = -1.0
                best_cand = matching_parents[0]
                child_clean = child.text.strip()
                child_tokens = set(child_clean.split())

                for p in matching_parents:
                    score = 0.0
                    p_clean = p.text.strip()
                    if child_clean in p_clean:
                        # Exact substring match
                        score = 10000.0 + len(child_clean)
                    elif child_tokens:
                        p_tokens = set(p_clean.split())
                        score = float(len(child_tokens & p_tokens))

                    if score > best_score:
                        best_score = score
                        best_cand = p
                    elif score == best_score:
                        # Deterministic tie-breaker
                        if p.chunk_index < best_cand.chunk_index:
                            best_cand = p

                best_parent = best_cand
            else:
                # 3. Fallback: text token overlap within the same document
                child_tokens = set(child.text.strip().split())
                if child_tokens:
                    best_score = 0.0
                    best_cand = None
                    for pid in doc_parent_ids:
                        p = self._parents[pid]
                        overlap = float(len(child_tokens & set(p.text.strip().split())))
                        if overlap > best_score:
                            best_score = overlap
                            best_cand = p
                    if best_cand and best_score > 0:
                        best_parent = best_cand

            if best_parent is not None:
                self.register_mapping(child.chunk_id, best_parent.chunk_id)
                child.metadata["parent_chunk_id"] = best_parent.chunk_id
                child.metadata["parent_document_id"] = best_parent.document_id
                mapped[child.chunk_id] = best_parent.chunk_id

        return mapped

    def get_parent(self, child_chunk_id: str) -> Chunk | None:
        """Retrieve the parent Chunk for a given child_chunk_id, or None if unmapped."""
        parent_id = self._child_to_parent.get(child_chunk_id)
        if parent_id is None:
            return None
        return self._parents.get(parent_id)

    def get_parent_id(self, child_chunk_id: str) -> str | None:
        """Retrieve the parent_chunk_id for a given child_chunk_id."""
        return self._child_to_parent.get(child_chunk_id)

    def get_children(self, parent_chunk_id: str) -> list[str]:
        """Retrieve all registered child_chunk_ids for a parent."""
        return list(self._parent_to_children.get(parent_chunk_id, []))

    def expand_results_to_parents(
        self,
        results: Sequence[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Map fine-grained SearchResults to deduplicated parent SearchResults.

        Preserves the ranking order of the first appearance of each parent chunk.
        """
        seen_parents: set[str] = set()
        expanded: list[SearchResult] = []

        for r in results:
            parent = self.get_parent(r.chunk.chunk_id)
            if parent is None:
                # Fallback: if chunk is already a parent chunk, retain it
                if r.chunk.chunk_id in self._parents:
                    parent = self._parents[r.chunk.chunk_id]
                else:
                    continue

            if parent.chunk_id in seen_parents:
                continue

            seen_parents.add(parent.chunk_id)
            expanded.append(
                SearchResult(
                    chunk=parent,
                    score=r.score,
                    rank=len(expanded),
                    strategy=r.strategy,
                )
            )
            if len(expanded) >= top_k:
                break

        return expanded

    def get_parent_neighbors(self, parent_chunk_id: str) -> tuple[Chunk | None, Chunk | None]:
        """Get (previous_chunk, next_chunk) within the same document, strictly isolated."""
        parent = self._parents.get(parent_chunk_id)
        if parent is None:
            return (None, None)

        doc_parent_ids = self._document_parents.get(parent.document_id, [])
        try:
            idx = doc_parent_ids.index(parent_chunk_id)
        except ValueError:
            return (None, None)

        prev_chunk = self._parents[doc_parent_ids[idx - 1]] if idx > 0 else None
        next_chunk = (
            self._parents[doc_parent_ids[idx + 1]] if idx + 1 < len(doc_parent_ids) else None
        )
        return (prev_chunk, next_chunk)

    def expand_parent_context(
        self,
        parent_chunk_id: str,
        include_prev: bool = True,
        include_next: bool = True,
    ) -> str:
        """Return expanded context: [prev.text, curr.text, next.text] within document."""
        parent = self._parents.get(parent_chunk_id)
        if parent is None:
            raise RetrievalError(f"Parent chunk not found: {parent_chunk_id!r}")

        prev_chunk, next_chunk = self.get_parent_neighbors(parent_chunk_id)
        parts: list[str] = []

        if include_prev and prev_chunk:
            parts.append(prev_chunk.text.strip())

        parts.append(parent.text.strip())

        if include_next and next_chunk:
            parts.append(next_chunk.text.strip())

        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize mapper state to a JSON-compatible dict."""
        return {
            "parents": [p.model_dump() for p in self._parents.values()],
            "child_to_parent": dict(self._child_to_parent),
            "parent_to_children": {k: list(v) for k, v in self._parent_to_children.items()},
            "document_parents": {k: list(v) for k, v in self._document_parents.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParentChunkMapper:
        """Reconstruct mapper state from a serialized dict."""
        mapper = cls()
        parents_data = data.get("parents", [])
        for p_dict in parents_data:
            chunk = Chunk.model_validate(p_dict)
            mapper._parents[chunk.chunk_id] = chunk

        mapper._child_to_parent = dict(data.get("child_to_parent", {}))
        mapper._parent_to_children = {
            k: list(v) for k, v in data.get("parent_to_children", {}).items()
        }
        mapper._document_parents = {k: list(v) for k, v in data.get("document_parents", {}).items()}
        return mapper

    def save_json(self, path: Path | str) -> None:
        """Save mapper state to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> ParentChunkMapper:
        """Load mapper state from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
