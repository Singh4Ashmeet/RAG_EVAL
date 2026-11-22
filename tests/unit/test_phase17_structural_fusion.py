"""Unit tests for Phase 17 structural metadata scoring and fusion."""

from __future__ import annotations

import pytest

from core.normalize.schema import (
    Chunk,
    DocumentMetadata,
    ElementType,
    ExtractedElement,
    NormalizedDocument,
    PageLocation,
    TableData,
)
from core.retrieval.schema import RetrievalStrategy, SearchResult
from core.retrieval.structural import (
    StructuralFusionStrategy,
    StructuralScorer,
    build_chunk_structural_metadata,
    compute_doc_context_signal,
    compute_element_type_signal,
    compute_heading_signal,
    compute_section_signal,
    compute_table_schema_signal,
    fuse_structural,
    tokenize_text,
)


@pytest.fixture
def sample_document() -> NormalizedDocument:
    """Fixture providing a multi-element normalized document."""
    elements = [
        ExtractedElement(
            element_id="elem_title",
            element_type=ElementType.HEADER,
            content="# Cluster Architecture Overview",
            order_index=0,
            location=PageLocation(page_number=1),
        ),
        ExtractedElement(
            element_id="elem_h2",
            element_type=ElementType.HEADER,
            content="## Storage Subsystem Specifications",
            order_index=1,
            location=PageLocation(page_number=1),
        ),
        ExtractedElement(
            element_id="elem_text",
            element_type=ElementType.TEXT,
            content="The cluster uses distributed NVMe storage with 10Gbps interconnect.",
            order_index=2,
            location=PageLocation(page_number=1),
        ),
        ExtractedElement(
            element_id="elem_table",
            element_type=ElementType.TABLE,
            content="Storage nodes configuration table.",
            order_index=3,
            location=PageLocation(page_number=2),
            table_data=TableData(
                headers=["Node ID", "Capacity", "Port", "Status"],
                rows=[["node-01", "2TB", "8080", "Active"], ["node-02", "4TB", "8080", "Active"]],
                caption="Storage Node Allocations",
            ),
        ),
        ExtractedElement(
            element_id="elem_list",
            element_type=ElementType.LIST_ITEM,
            content="1. Connect to node console via SSH.\n2. Run cluster health check command.",
            order_index=4,
            location=PageLocation(page_number=2),
        ),
    ]

    doc_meta = DocumentMetadata(
        document_id="doc_cluster_01",
        filename="cluster_architecture.md",
    )

    return NormalizedDocument(
        metadata=doc_meta,
        elements=elements,
        page_count=2,
    )


@pytest.fixture
def sample_chunks(sample_document: NormalizedDocument) -> list[Chunk]:
    """Fixture providing chunks corresponding to sample_document."""
    return [
        Chunk(
            chunk_id="doc_cluster_01_c0001",
            document_id="doc_cluster_01",
            text="The cluster uses distributed NVMe storage with 10Gbps interconnect.",
            chunk_index=0,
            source_element_ids=["elem_title", "elem_h2", "elem_text"],
            locations=[PageLocation(page_number=1)],
            metadata={"doc_title": "Cluster Architecture Overview"},
        ),
        Chunk(
            chunk_id="doc_cluster_01_c0002",
            document_id="doc_cluster_01",
            text="Storage nodes configuration table: node-01 2TB port 8080 Active.",
            chunk_index=1,
            source_element_ids=["elem_table"],
            locations=[PageLocation(page_number=2)],
            metadata={"doc_title": "Cluster Architecture Overview"},
        ),
        Chunk(
            chunk_id="doc_cluster_01_c0003",
            document_id="doc_cluster_01",
            text="1. Connect to node console via SSH. 2. Run cluster health check command.",
            chunk_index=2,
            source_element_ids=["elem_list"],
            locations=[PageLocation(page_number=2)],
            metadata={"doc_title": "Cluster Architecture Overview"},
        ),
    ]


class TestStructuralSignals:
    """Tests for individual deterministic structural signals."""

    def test_tokenize_text(self) -> None:
        tokens = tokenize_text("What is the storage node capacity in cluster architecture?")
        assert "storage" in tokens
        assert "node" in tokens
        assert "capacity" in tokens
        assert "cluster" in tokens
        assert "architecture" in tokens
        assert "what" not in tokens  # Stopword
        assert "is" not in tokens  # Stopword
        assert "the" not in tokens  # Stopword

    def test_heading_signal_matching(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        meta = build_chunk_structural_metadata(sample_chunks[0], sample_document)
        query_tokens = tokenize_text("What are the storage subsystem specifications?")
        score = compute_heading_signal(query_tokens, meta)
        assert score > 0.4  # Matches "storage subsystem specifications"

    def test_section_signal_hierarchy(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        meta = build_chunk_structural_metadata(sample_chunks[0], sample_document)
        query_tokens = tokenize_text("cluster architecture storage")
        score = compute_section_signal(query_tokens, meta)
        assert score > 0.5

    def test_element_type_handling(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        table_meta = build_chunk_structural_metadata(sample_chunks[1], sample_document)
        list_meta = build_chunk_structural_metadata(sample_chunks[2], sample_document)

        # Table query on table chunk -> strong match
        s_table = compute_element_type_signal("Show table of node port numbers", table_meta)
        assert s_table == 1.0

        # Numerical query on table chunk -> high compatibility
        s_num = compute_element_type_signal("How many ports are configured?", table_meta)
        assert s_num == 1.0

        # Procedural query on list chunk -> strong match
        s_proc = compute_element_type_signal("How to run cluster health check?", list_meta)
        assert s_proc == 1.0

    def test_table_schema_matching(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        table_meta = build_chunk_structural_metadata(sample_chunks[1], sample_document)
        text_meta = build_chunk_structural_metadata(sample_chunks[0], sample_document)

        # Match table columns: "Node ID", "Capacity", "Port", "Status"
        q_tokens = tokenize_text("What is the port and capacity status?")
        s_table = compute_table_schema_signal(q_tokens, table_meta)
        assert s_table > 0.5

        # Chunk without table must return 0.0
        s_no_table = compute_table_schema_signal(q_tokens, text_meta)
        assert s_no_table == 0.0

    def test_doc_context_matching(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        meta = build_chunk_structural_metadata(sample_chunks[0], sample_document)
        q_tokens = tokenize_text("cluster architecture overview")
        score = compute_doc_context_signal(q_tokens, meta)
        assert score == 1.0


class TestStructuralScorer:
    """Tests for the aggregate StructuralScorer."""

    def test_deterministic_scoring(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        docs = {sample_document.metadata.document_id: sample_document}
        scorer = StructuralScorer(documents=docs)
        query = "What is the storage subsystem port allocation?"

        res1 = scorer.score(query, sample_chunks[0])
        res2 = scorer.score(query, sample_chunks[0])

        assert res1.total_score == res2.total_score
        assert res1.s_heading == res2.s_heading
        assert res1.s_section == res2.s_section
        assert res1.s_element_type == res2.s_element_type
        assert res1.s_table_schema == res2.s_table_schema
        assert res1.s_doc_context == res2.s_doc_context

    def test_score_normalization_bounded(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        docs = {sample_document.metadata.document_id: sample_document}
        scorer = StructuralScorer(documents=docs)

        for c in sample_chunks:
            res = scorer.score("anything", c)
            assert 0.0 <= res.total_score <= 1.0
            assert 0.0 <= res.s_heading <= 1.0
            assert 0.0 <= res.s_section <= 1.0
            assert 0.0 <= res.s_element_type <= 1.0
            assert 0.0 <= res.s_table_schema <= 1.0
            assert 0.0 <= res.s_doc_context <= 1.0

    def test_missing_document_and_empty_metadata(self) -> None:
        scorer = StructuralScorer(documents={})
        empty_chunk = Chunk(
            chunk_id="empty_c001",
            document_id="missing_doc",
            text="Some bare text without document.",
            chunk_index=0,
        )

        res = scorer.score("some query", empty_chunk)
        assert 0.0 <= res.total_score <= 1.0
        assert res.s_heading == 0.0
        assert res.s_table_schema == 0.0


class TestStructuralFusion:
    """Tests for fusion strategies and safety guarantees."""

    def test_baseline_equivalence_when_disabled(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        docs = {sample_document.metadata.document_id: sample_document}
        scorer = StructuralScorer(documents=docs)

        candidates = [
            SearchResult(
                chunk=sample_chunks[0],
                score=0.015,
                rank=0,
                strategy=RetrievalStrategy.HYBRID,
            ),
            SearchResult(
                chunk=sample_chunks[1],
                score=0.012,
                rank=1,
                strategy=RetrievalStrategy.HYBRID,
            ),
        ]

        # Strategy NONE
        res_none, diag = fuse_structural(
            candidates,
            query="port numbers",
            scorer=scorer,
            strategy=StructuralFusionStrategy.NONE,
        )
        assert [r.chunk.chunk_id for r in res_none] == [
            "doc_cluster_01_c0001",
            "doc_cluster_01_c0002",
        ]
        assert [r.score for r in res_none] == [0.015, 0.012]
        assert diag["modifications"] == 0

        # Alpha 0.0
        res_zero, _ = fuse_structural(
            candidates,
            query="port numbers",
            scorer=scorer,
            strategy=StructuralFusionStrategy.ADDITIVE,
            alpha=0.0,
        )
        assert [r.chunk.chunk_id for r in res_zero] == [
            "doc_cluster_01_c0001",
            "doc_cluster_01_c0002",
        ]

    def test_additive_fusion_ordering(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        docs = {sample_document.metadata.document_id: sample_document}
        scorer = StructuralScorer(documents=docs)

        # Candidate 0 has higher RRF score (0.015) but lower structural relevance for table query
        # Candidate 1 has lower RRF score (0.014) but high structural score for table query
        candidates = [
            SearchResult(
                chunk=sample_chunks[0],
                score=0.015,
                rank=0,
                strategy=RetrievalStrategy.HYBRID,
            ),
            SearchResult(
                chunk=sample_chunks[1],
                score=0.014,
                rank=1,
                strategy=RetrievalStrategy.HYBRID,
            ),
        ]

        fused, diag = fuse_structural(
            candidates,
            query="Show table of node port status",
            scorer=scorer,
            strategy=StructuralFusionStrategy.ADDITIVE,
            alpha=0.20,
        )

        assert len(fused) == 2
        # Candidate 1 should be boosted to rank 0 due to strong table schema + element type match
        assert fused[0].chunk.chunk_id == "doc_cluster_01_c0002"
        assert fused[0].rank == 0
        assert fused[1].rank == 1

    def test_rank_aware_margin_boost(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        docs = {sample_document.metadata.document_id: sample_document}
        scorer = StructuralScorer(documents=docs)

        # Case 1: Candidates within margin (0.0005) -> re-order allowed
        cands_close = [
            SearchResult(
                chunk=sample_chunks[0],
                score=0.0150,
                rank=0,
                strategy=RetrievalStrategy.HYBRID,
            ),
            SearchResult(
                chunk=sample_chunks[1],
                score=0.0148,
                rank=1,
                strategy=RetrievalStrategy.HYBRID,
            ),
        ]
        fused_close, _ = fuse_structural(
            cands_close,
            query="node port table",
            scorer=scorer,
            strategy=StructuralFusionStrategy.RANK_AWARE_BOOST,
            margin_threshold=0.001,
        )
        assert fused_close[0].chunk.chunk_id == "doc_cluster_01_c0002"

        # Case 2: Candidates separated by large margin (0.005) -> re-order prohibited
        cands_far = [
            SearchResult(
                chunk=sample_chunks[0],
                score=0.0200,
                rank=0,
                strategy=RetrievalStrategy.HYBRID,
            ),
            SearchResult(
                chunk=sample_chunks[1],
                score=0.0120,
                rank=1,
                strategy=RetrievalStrategy.HYBRID,
            ),
        ]
        fused_far, _ = fuse_structural(
            cands_far,
            query="node port table",
            scorer=scorer,
            strategy=StructuralFusionStrategy.RANK_AWARE_BOOST,
            margin_threshold=0.001,
        )
        # Rank 0 must remain unchanged despite candidate 1 having higher structural score
        assert fused_far[0].chunk.chunk_id == "doc_cluster_01_c0001"

    def test_tie_break_strategy(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        docs = {sample_document.metadata.document_id: sample_document}
        scorer = StructuralScorer(documents=docs)

        # Exact tie in RRF scores
        cands_tied = [
            SearchResult(
                chunk=sample_chunks[0],
                score=0.0150,
                rank=0,
                strategy=RetrievalStrategy.HYBRID,
            ),
            SearchResult(
                chunk=sample_chunks[1],
                score=0.0150,
                rank=1,
                strategy=RetrievalStrategy.HYBRID,
            ),
        ]

        fused_tied, _ = fuse_structural(
            cands_tied,
            query="node port table",
            scorer=scorer,
            strategy=StructuralFusionStrategy.TIE_BREAK,
        )
        # Tied RRF broken by structural score -> candidate 1 wins
        assert fused_tied[0].chunk.chunk_id == "doc_cluster_01_c0002"

    def test_provenance_preservation(
        self, sample_document: NormalizedDocument, sample_chunks: list[Chunk]
    ) -> None:
        docs = {sample_document.metadata.document_id: sample_document}
        scorer = StructuralScorer(documents=docs)

        candidates = [
            SearchResult(
                chunk=sample_chunks[0],
                score=0.015,
                rank=0,
                strategy=RetrievalStrategy.HYBRID,
            )
        ]

        fused, _ = fuse_structural(
            candidates,
            query="cluster query",
            scorer=scorer,
            strategy=StructuralFusionStrategy.ADDITIVE,
            alpha=0.10,
        )

        res_chunk = fused[0].chunk
        orig_chunk = sample_chunks[0]

        assert res_chunk.chunk_id == orig_chunk.chunk_id
        assert res_chunk.document_id == orig_chunk.document_id
        assert res_chunk.text == orig_chunk.text  # Text unmodified
        assert res_chunk.source_element_ids == orig_chunk.source_element_ids
        assert res_chunk.locations == orig_chunk.locations
