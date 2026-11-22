"""Unit tests for Phase 19 ParentChunkMapper."""

from __future__ import annotations

import tempfile
from pathlib import Path

from core.normalize.schema import BoundingBox, Chunk, PageLocation
from core.retrieval.parent_mapping import ParentChunkMapper
from core.retrieval.schema import RetrievalStrategy, SearchResult


def _make_dummy_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    index: int,
    source_element_ids: list[str],
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        chunk_index=index,
        source_element_ids=source_element_ids,
        locations=[
            PageLocation(
                page_number=1,
                bbox=BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
            )
        ],
        metadata={"test": True},
    )


def test_mapping_correctness() -> None:
    """Verify child chunks map to the correct parent containing their source elements."""
    parents = [
        _make_dummy_chunk(
            "doc1_c0000",
            "doc1",
            "First section text here.",
            0,
            ["elem1", "elem2"],
        ),
        _make_dummy_chunk(
            "doc1_c0001",
            "doc1",
            "Second section text here.",
            1,
            ["elem3", "elem4"],
        ),
    ]
    children = [
        _make_dummy_chunk("doc1_ea0000", "doc1", "First section", 0, ["elem1"]),
        _make_dummy_chunk("doc1_ea0001", "doc1", "Second section", 1, ["elem3"]),
    ]

    mapper = ParentChunkMapper()
    mapped = mapper.map_child_chunks(children, parents)

    assert mapped["doc1_ea0000"] == "doc1_c0000"
    assert mapped["doc1_ea0001"] == "doc1_c0001"
    assert mapper.get_parent("doc1_ea0000") == parents[0]
    assert mapper.get_parent("doc1_ea0001") == parents[1]


def test_deterministic_ids_and_mapping() -> None:
    """Verify repeated mapping produces identical deterministic assignments."""
    parents = [
        _make_dummy_chunk("doc1_c0000", "doc1", "Alpha beta gamma", 0, ["e1"]),
        _make_dummy_chunk("doc1_c0001", "doc1", "Delta epsilon zeta", 1, ["e2"]),
    ]
    children = [
        _make_dummy_chunk("doc1_ea0000", "doc1", "Alpha", 0, ["e1"]),
        _make_dummy_chunk("doc1_ea0001", "doc1", "Delta", 1, ["e2"]),
    ]

    mapper1 = ParentChunkMapper()
    res1 = mapper1.map_child_chunks(children, parents)

    mapper2 = ParentChunkMapper()
    res2 = mapper2.map_child_chunks(children, parents)

    assert res1 == res2


def test_duplicate_handling() -> None:
    """Verify multiple child chunks pointing to same parent deduplicate in order."""
    parent = _make_dummy_chunk("doc1_c0000", "doc1", "Whole paragraph.", 0, ["e1", "e2"])
    child1 = _make_dummy_chunk("doc1_ea0000", "doc1", "Whole", 0, ["e1"])
    child2 = _make_dummy_chunk("doc1_ea0001", "doc1", "paragraph.", 1, ["e2"])

    mapper = ParentChunkMapper()
    mapper.map_child_chunks([child1, child2], [parent])

    results = [
        SearchResult(
            chunk=child1,
            score=0.9,
            rank=0,
            strategy=RetrievalStrategy.HYBRID,
        ),
        SearchResult(
            chunk=child2,
            score=0.8,
            rank=1,
            strategy=RetrievalStrategy.HYBRID,
        ),
    ]

    expanded = mapper.expand_results_to_parents(results, top_k=5)
    assert len(expanded) == 1
    assert expanded[0].chunk.chunk_id == "doc1_c0000"
    assert expanded[0].rank == 0


def test_missing_parent_handling() -> None:
    """Verify unmapped or missing parent chunks return None gracefully."""
    mapper = ParentChunkMapper()
    assert mapper.get_parent("non_existent_child") is None
    assert mapper.get_parent_id("non_existent_child") is None

    # Expansion with unmapped chunk falls back or skips gracefully
    orphan = _make_dummy_chunk("orphan_c0", "orphan_doc", "Orphan text", 0, ["e99"])
    results = [
        SearchResult(
            chunk=orphan,
            score=0.5,
            rank=0,
            strategy=RetrievalStrategy.HYBRID,
        )
    ]
    expanded = mapper.expand_results_to_parents(results, top_k=5)
    assert len(expanded) == 0


def test_provenance_preservation() -> None:
    """Verify expanded parent retains full original provenance and verbatim text."""
    parent = _make_dummy_chunk(
        "doc1_c0000",
        "doc1",
        "Target text with critical information.",
        0,
        ["e1", "e2"],
    )
    child = _make_dummy_chunk("doc1_ea0000", "doc1", "Target text", 0, ["e1"])

    mapper = ParentChunkMapper()
    mapper.map_child_chunks([child], [parent])

    results = [
        SearchResult(
            chunk=child,
            score=0.95,
            rank=0,
            strategy=RetrievalStrategy.HYBRID,
        )
    ]
    expanded = mapper.expand_results_to_parents(results, top_k=5)
    assert len(expanded) == 1
    res_parent = expanded[0].chunk

    assert res_parent.chunk_id == "doc1_c0000"
    assert res_parent.document_id == "doc1"
    assert res_parent.text == "Target text with critical information."
    assert res_parent.source_element_ids == ["e1", "e2"]
    assert len(res_parent.locations) == 1
    assert res_parent.locations[0].page_number == 1


def test_cross_document_isolation() -> None:
    """Verify child chunk in docA NEVER maps to a parent in docB."""
    parent_doc_b = _make_dummy_chunk("docB_c0000", "docB", "Element shared text.", 0, ["e1"])
    child_doc_a = _make_dummy_chunk("docA_ea0000", "docA", "Element shared text.", 0, ["e1"])

    mapper = ParentChunkMapper()
    mapped = mapper.map_child_chunks([child_doc_a], [parent_doc_b])

    # Should not map to docB parent
    assert "docA_ea0000" not in mapped
    assert mapper.get_parent("docA_ea0000") is None


def test_context_expansion_neighbors() -> None:
    """Verify neighbor context expansion works within document boundaries."""
    p0 = _make_dummy_chunk("doc1_c0000", "doc1", "Chunk zero.", 0, ["e0"])
    p1 = _make_dummy_chunk("doc1_c0001", "doc1", "Chunk one.", 1, ["e1"])
    p2 = _make_dummy_chunk("doc1_c0002", "doc1", "Chunk two.", 2, ["e2"])

    mapper = ParentChunkMapper()
    mapper.register_parent_chunks([p0, p1, p2])

    prev_c, next_c = mapper.get_parent_neighbors("doc1_c0001")
    assert prev_c is not None and prev_c.chunk_id == "doc1_c0000"
    assert next_c is not None and next_c.chunk_id == "doc1_c0002"

    prev_c0, next_c0 = mapper.get_parent_neighbors("doc1_c0000")
    assert prev_c0 is None
    assert next_c0 is not None and next_c0.chunk_id == "doc1_c0001"

    expanded_ctx = mapper.expand_parent_context("doc1_c0001")
    assert expanded_ctx == "Chunk zero.\n\nChunk one.\n\nChunk two."


def test_serialization_roundtrip() -> None:
    """Verify mapper serialization to dict and JSON round-trips perfectly."""
    p0 = _make_dummy_chunk("doc1_c0000", "doc1", "Parent text.", 0, ["e0"])
    c0 = _make_dummy_chunk("doc1_ea0000", "doc1", "Child text.", 0, ["e0"])

    mapper = ParentChunkMapper()
    mapper.map_child_chunks([c0], [p0])

    data = mapper.to_dict()
    mapper2 = ParentChunkMapper.from_dict(data)

    assert mapper2.get_parent("doc1_ea0000") is not None
    p2 = mapper2.get_parent("doc1_ea0000")
    assert p2 is not None and p2.chunk_id == "doc1_c0000"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        mapper.save_json(tmp_path)
        mapper3 = ParentChunkMapper.load_json(tmp_path)
        p3 = mapper3.get_parent("doc1_ea0000")
        assert p3 is not None and p3.chunk_id == "doc1_c0000"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
