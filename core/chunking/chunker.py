"""Concrete element-aware document chunker implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.normalize.schema import Chunk, NormalizedDocument, PageLocation


@dataclass(frozen=True)
class _Fragment:
    """Immutable offset-tracked slice of an ExtractedElement."""

    element_id: str
    original_text: str
    start: int
    end: int
    location: PageLocation | None

    def __post_init__(self) -> None:
        if not (0 <= self.start < self.end <= len(self.original_text)):
            raise ValueError(
                f"Invalid fragment offsets [{self.start}, {self.end}) "
                f"for text of length {len(self.original_text)}"
            )

    @property
    def text(self) -> str:
        """Derived character slice, guaranteeing exact source slice fidelity."""
        return self.original_text[self.start : self.end]

    @property
    def is_split(self) -> bool:
        """True if this fragment is a sub-slice of an oversized element."""
        return self.start != 0 or self.end != len(self.original_text)

    @property
    def is_whitespace_only(self) -> bool:
        """True if the slice contains only whitespace characters."""
        return len(self.text.strip()) == 0


class ElementAwareChunker:
    """Synchronous chunker segmenting NormalizedDocuments while preserving element provenance.

    Parameters:
        chunk_size: Maximum chunk length in characters (default 1000).
        chunk_overlap: Target character overlap between consecutive chunks (default 200).
            Must be strictly less than chunk_size.

    Note on Deterministic Identifiers:
        Chunk IDs are derived deterministically as '{document_id}_c{chunk_index:04d}'.
        If chunk_size or chunk_overlap configuration changes, chunk boundaries and contents
        will change while retaining sequential index-based identifiers.

    Note on Single Overlap Architecture:
        Overlap is applied exclusively at the oversized element splitter level.
        Packing is strictly forward-only and does not duplicate complete elements across chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be strictly positive, got: {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got: {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        """Segment a NormalizedDocument into discrete Chunks linked to source elements.

        Args:
            document: Canonical NormalizedDocument to chunk.

        Returns:
            List of canonical Chunk instances with preserved provenance and reading order.
        """
        fragments: list[_Fragment] = []
        element_lengths: dict[str, int] = {}
        for elem in document.elements:
            # Documented ingestion contract: strip element content before chunking.
            # Leading and trailing whitespace stripped here is outside chunker offset tracking.
            content = elem.content.strip()
            if not content:
                continue

            element_lengths[elem.element_id] = len(content)

            if len(content) <= self.chunk_size:
                fragments.append(
                    _Fragment(
                        element_id=elem.element_id,
                        original_text=content,
                        start=0,
                        end=len(content),
                        location=elem.location,
                    )
                )
            else:
                fragments.extend(self._split_oversized(elem.element_id, content, elem.location))

        if not fragments:
            return []

        chunks: list[Chunk] = []
        i = 0
        n = len(fragments)
        separator = "\n\n"
        last_emitted_end: dict[str, int] = {}

        while i < n:
            # Phase 1: Skip leading unpackable whitespace-only fragments.
            # If fragments[i] is whitespace-only, check if it can pack with a
            # subsequent non-whitespace fragment without exceeding chunk_size.
            if fragments[i].is_whitespace_only and not self._can_pack_with_non_whitespace(
                fragments, i, separator
            ):
                i += 1
                continue

            # Phase 2: Forward-only greedy packing
            packed_frags: list[_Fragment] = [fragments[i]]
            current_len = len(fragments[i].text)
            j = i + 1

            while j < n:
                frag_j = fragments[j]
                add_len = self._measure_additional_length(packed_frags, frag_j, separator)
                candidate_len = current_len + add_len

                if candidate_len <= self.chunk_size:
                    packed_frags.append(frag_j)
                    current_len = candidate_len
                    j += 1
                else:
                    break

            chunk_text = self._assemble_chunk_text(packed_frags, separator)

            # Defensive verification: skip if chunk assembled to whitespace-only
            if not chunk_text.strip():
                i = j
                continue

            # Phase 3: Track any omitted whitespace intervals
            omitted_intervals: list[dict[str, Any]] = []
            element_frags: dict[str, list[_Fragment]] = {}
            for f in packed_frags:
                element_frags.setdefault(f.element_id, []).append(f)

            for eid, frags in element_frags.items():
                prior = last_emitted_end.get(eid)
                first_start = frags[0].start
                if prior is None:
                    # Whitespace skipped before the first emitted chunk of this element
                    if first_start > 0:
                        omitted_intervals.append(
                            {
                                "element_id": eid,
                                "start": 0,
                                "end": first_start,
                                "length": first_start,
                            }
                        )
                elif first_start > prior:
                    # Whitespace skipped between emitted chunks of this element
                    omitted_intervals.append(
                        {
                            "element_id": eid,
                            "start": prior,
                            "end": first_start,
                            "length": first_start - prior,
                        }
                    )
                # Check for gaps within this chunk
                for idx in range(len(frags) - 1):
                    if frags[idx + 1].start > frags[idx].end:
                        omitted_intervals.append(
                            {
                                "element_id": eid,
                                "start": frags[idx].end,
                                "end": frags[idx + 1].start,
                                "length": frags[idx + 1].start - frags[idx].end,
                            }
                        )
                last_emitted_end[eid] = max(last_emitted_end.get(eid, 0), max(f.end for f in frags))

            # Preserve unique source element IDs in reading order
            seen_elem_ids: set[str] = set()
            source_element_ids: list[str] = []
            for f in packed_frags:
                if f.element_id not in seen_elem_ids:
                    seen_elem_ids.add(f.element_id)
                    source_element_ids.append(f.element_id)

            # Propagate locations from contributing elements without duplicates
            seen_locations: list[PageLocation] = []
            for f in packed_frags:
                if f.location is not None and f.location not in seen_locations:
                    seen_locations.append(f.location)

            chunk_index = len(chunks)
            chunk_id = f"{document.metadata.document_id}_c{chunk_index:04d}"
            token_count = self._estimate_token_count(chunk_text)

            chunk_metadata: dict[str, Any] = {
                "chunker": "ElementAwareChunker",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "token_count_estimated": True,
                "token_count_method": "char_ceil_div_4",
            }
            if omitted_intervals:
                chunk_metadata["omitted_whitespace_intervals"] = omitted_intervals

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.metadata.document_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    source_element_ids=source_element_ids,
                    locations=seen_locations,
                    metadata=chunk_metadata,
                )
            )

            # Forward-only packing progress
            i = j

        # Check for any trailing omitted whitespace at the end of elements
        for eid, emitted_end in last_emitted_end.items():
            elem_len = element_lengths.get(eid, 0)
            if emitted_end < elem_len:
                trailing_omission = {
                    "element_id": eid,
                    "start": emitted_end,
                    "end": elem_len,
                    "length": elem_len - emitted_end,
                }
                for c in reversed(chunks):
                    if eid in c.source_element_ids:
                        c.metadata.setdefault("omitted_whitespace_intervals", []).append(
                            trailing_omission
                        )
                        break

        return chunks

    def _split_oversized(
        self,
        element_id: str,
        text: str,
        location: PageLocation | None,
    ) -> list[_Fragment]:
        """Deterministically split oversized text into overlapping fragments <= chunk_size.

        Invariants guaranteed:
          1. First fragment starts at offset 0 (I1).
          2. Final fragment ends at len(text) (I2).
          3. All intervals are nonempty with 0 <= start < end <= len(text) (I3).
          4. Starts advance monotonically by S = C - O (I4).
          5. No coverage gaps exist (I5).
          6. Every fragment length <= chunk_size (I6).
          7. Overlap is strictly between 0 and chunk_overlap (I7).
          8. Termination is guaranteed in <= ceil(N / S) iterations.
        """
        N = len(text)
        S = self.chunk_size - self.chunk_overlap
        fragments: list[_Fragment] = []
        start = 0

        while start < N:
            hard_end = min(start + self.chunk_size, N)

            if hard_end < N:
                end = self._snap_word_boundary(text, start, hard_end, start + S)
            else:
                end = hard_end

            fragments.append(
                _Fragment(
                    element_id=element_id,
                    original_text=text,
                    start=start,
                    end=end,
                    location=location,
                )
            )

            if end >= N:
                break

            start += S

        return fragments

    @staticmethod
    def _snap_word_boundary(text: str, start: int, hard_end: int, min_end: int) -> int:
        """Find split point in [min_end, hard_end) preferring newline then space.

        Guarantees min_end <= returned_end <= hard_end.
        """
        best_newline = -1
        best_space = -1

        for pos in range(hard_end - 1, min_end - 1, -1):
            ch = text[pos]
            if ch == "\n":
                best_newline = pos + 1
                break
            if ch == " " and best_space == -1:
                best_space = pos + 1

        if best_newline != -1:
            return best_newline
        if best_space != -1:
            return best_space
        return hard_end

    @staticmethod
    def _measure_additional_length(
        current_fragments: list[_Fragment],
        new_frag: _Fragment,
        separator: str = "\n\n",
    ) -> int:
        """Calculate the exact character increase if new_frag is appended to current_fragments."""
        if not current_fragments:
            return len(new_frag.text)

        prev_frag = current_fragments[-1]
        if new_frag.element_id != prev_frag.element_id:
            return len(separator) + len(new_frag.text)

        # Same element: deduplicate shared source intervals within the chunk
        covered_end = max(f.end for f in current_fragments if f.element_id == new_frag.element_id)
        if new_frag.start < covered_end:
            if new_frag.end > covered_end:
                return new_frag.end - covered_end
            return 0
        return len(new_frag.text)

    @staticmethod
    def _assemble_chunk_text(
        fragments: list[_Fragment],
        separator: str = "\n\n",
    ) -> str:
        """Assemble text for packed fragments respecting element separation.

        Rules:
          - Insert separator between fragments from different elements.
          - Between fragments of the same element, deduplicate shared source intervals.
        """
        if not fragments:
            return ""

        parts: list[str] = []
        chunk_covered_end: dict[str, int] = {}
        prev_element_id: str | None = None

        for frag in fragments:
            eid = frag.element_id
            if prev_element_id is not None and eid != prev_element_id:
                parts.append(separator)

            if eid not in chunk_covered_end:
                parts.append(frag.text)
                chunk_covered_end[eid] = frag.end
            else:
                covered = chunk_covered_end[eid]
                if frag.start < covered:
                    if frag.end > covered:
                        parts.append(frag.original_text[covered : frag.end])
                        chunk_covered_end[eid] = frag.end
                else:
                    parts.append(frag.text)
                    chunk_covered_end[eid] = frag.end

            prev_element_id = eid

        return "".join(parts)

    def _can_pack_with_non_whitespace(
        self,
        fragments: list[_Fragment],
        start_idx: int,
        separator: str = "\n\n",
    ) -> bool:
        """Check if fragments from start_idx can form a valid chunk containing non-whitespace."""
        current_frags: list[_Fragment] = [fragments[start_idx]]
        current_len = len(fragments[start_idx].text)

        if not fragments[start_idx].is_whitespace_only:
            return True

        n = len(fragments)
        for k in range(start_idx + 1, n):
            add_len = self._measure_additional_length(current_frags, fragments[k], separator)
            if current_len + add_len <= self.chunk_size:
                current_frags.append(fragments[k])
                current_len += add_len
                if not fragments[k].is_whitespace_only:
                    return True
            else:
                break

        return False

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """Estimate token count from character length.

        Estimation policy:
            Assumes ~4 characters per token on average (typical for English/code).
            Uses ceiling division so nonempty chunks never round down to 0 tokens.
            Clearly documented as an approximation that is not guaranteed to match
            any specific tokenizer model.
        """
        clean = text.strip()
        if not clean:
            return 0
        return max(1, (len(clean) + 3) // 4)
