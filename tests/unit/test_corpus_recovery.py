"""Unit tests for corpus indexing recovery and preservation without source file dependencies.

Verifies:
1. Empty corpus does not become indexed (raises ValueError, job and document failed).
2. Missing original S3 object does not block embedding when existing corpus text is available.
3. Missing original local file does not block embedding when existing corpus text is available.
4. Existing embeddings are preserved in place.
5. Only missing embeddings are generated.
6. A corpus with all embeddings already populated completes without calling embedding provider.
7. A failure during embedding does not mark the document indexed.
8. Existing source-element IDs, order, text, and metadata remain unchanged.
9. Stale lease tokens cannot commit embedding updates.
10. Normal successful indexing works end-to-end.
11. 47-element recovery scenario: exact simulation of 02-TRD.md incident.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.api.main import Account, Base, CorpusElement, Document, Job
from apps.worker.pg_queue import StaleLeaseError
from apps.worker.tasks import index_document


@pytest.fixture
def db_session_factory():
    """In-memory SQLite session factory for task testing."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    # Seed baseline workspace
    with session_factory() as db:
        ws = Account(
            id="ws-recovery-test",
            auth_subject="sub-recovery-test",
            login="recovery_tester",
            email="tester@example.com",
        )
        db.add(ws)
        db.commit()

    return session_factory


def _mock_httpx_embed(mock_post, vector_dim=4):
    """Helper to mock httpx.post for the embedding service."""

    def fake_post(url, json=None, timeout=None):
        inputs = json.get("input", []) if json else []
        vectors = [[0.1 * (i + 1)] * vector_dim for i in range(len(inputs))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": vectors}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    mock_post.side_effect = fake_post


# =============================================================================
# 1. Empty corpus does not become indexed
# =============================================================================
def test_empty_corpus_does_not_become_indexed(db_session_factory):
    """Empty corpus raises ValueError and marks document and job as failed."""
    ws_id = "ws-recovery-test"
    doc_id = "doc-empty"
    job_id = "job-empty"

    with db_session_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="empty.txt",
            storage_key=f"{ws_id}/empty.txt",
            status="queued",
            sha256="abc",
        )
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="queued",
            payload_json={"document_id": doc_id},
        )
        db.add_all([doc, job])
        db.commit()

    with pytest.raises(ValueError, match="Cannot index empty corpus"):
        index_document(
            workspace_id=ws_id,
            document_id=doc_id,
            job_id=job_id,
            session_factory=db_session_factory,
        )

    with db_session_factory() as db:
        d = db.scalar(select(Document).where(Document.id == doc_id))
        j = db.scalar(select(Job).where(Job.id == job_id))
        assert d.status == "failed"
        assert j.status == "failed"
        assert "zero source elements" in j.detail


# =============================================================================
# 2. Missing original S3 object does not block embedding
# =============================================================================
def test_missing_s3_object_does_not_block_embedding(db_session_factory, monkeypatch):
    """Re-embedding proceeds from existing corpus elements without S3 head_object."""
    ws_id = "ws-recovery-test"
    doc_id = "doc-missing-s3"
    job_id = "job-missing-s3"

    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET", "nonexistent-bucket")

    with db_session_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="cloud_doc.md",
            storage_key=f"{ws_id}/cloud_doc.md",
            status="queued",
            sha256="abc1",
        )
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="queued",
            payload_json={"document_id": doc_id},
        )
        el = CorpusElement(
            id="el-1",
            workspace_id=ws_id,
            document_id=doc_id,
            ordinal=0,
            text="Existing corpus text available in DB",
            embedding_json=None,
        )
        db.add_all([doc, job, el])
        db.commit()

    with patch("httpx.post") as mock_post:
        _mock_httpx_embed(mock_post)
        result = index_document(
            workspace_id=ws_id,
            document_id=doc_id,
            job_id=job_id,
            session_factory=db_session_factory,
        )

    assert result == {"indexed": 1}
    with db_session_factory() as db:
        d = db.scalar(select(Document).where(Document.id == doc_id))
        assert d.status == "indexed"
        saved_el = db.scalar(select(CorpusElement).where(CorpusElement.id == "el-1"))
        assert saved_el.embedding_json is not None


# =============================================================================
# 3. Missing original local file does not block embedding
# =============================================================================
def test_missing_local_file_does_not_block_embedding(db_session_factory, tmp_path, monkeypatch):
    """Re-embedding proceeds even if local raw source file does not exist on disk."""
    ws_id = "ws-recovery-test"
    doc_id = "doc-missing-local"
    job_id = "job-missing-local"

    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setattr("apps.api.main.PRIVATE_DATA", tmp_path)

    with db_session_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="local_doc.txt",
            storage_key=f"{ws_id}/completely_missing_file.txt",
            status="queued",
            sha256="abc2",
        )
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="queued",
            payload_json={"document_id": doc_id},
        )
        el = CorpusElement(
            id="el-local-1",
            workspace_id=ws_id,
            document_id=doc_id,
            ordinal=0,
            text="Locally stored corpus element text",
            embedding_json=None,
        )
        db.add_all([doc, job, el])
        db.commit()

    # File definitely does not exist on disk
    assert not (tmp_path / f"{ws_id}/completely_missing_file.txt").exists()

    with patch("httpx.post") as mock_post:
        _mock_httpx_embed(mock_post)
        result = index_document(
            workspace_id=ws_id,
            document_id=doc_id,
            job_id=job_id,
            session_factory=db_session_factory,
        )

    assert result == {"indexed": 1}
    with db_session_factory() as db:
        d = db.scalar(select(Document).where(Document.id == doc_id))
        assert d.status == "indexed"


# =============================================================================
# 4 & 5. Existing embeddings preserved; only missing embeddings generated
# =============================================================================
def test_existing_embeddings_preserved_and_only_missing_generated(db_session_factory):
    """Existing vectors are preserved in place; only elements with None embedding_json are sent."""
    ws_id = "ws-recovery-test"
    doc_id = "doc-partial"
    job_id = "job-partial"
    preserved_vector = [0.99, 0.88, 0.77, 0.66]

    with db_session_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="partial.txt",
            storage_key=f"{ws_id}/partial.txt",
            status="queued",
            sha256="abc3",
        )
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="queued",
            payload_json={"document_id": doc_id},
        )
        el1 = CorpusElement(
            id="el-already-done",
            workspace_id=ws_id,
            document_id=doc_id,
            ordinal=0,
            text="Element 1 already embedded",
            embedding_json=json.dumps(preserved_vector),
        )
        el2 = CorpusElement(
            id="el-needs-embed",
            workspace_id=ws_id,
            document_id=doc_id,
            ordinal=1,
            text="Element 2 needs embedding",
            embedding_json=None,
        )
        db.add_all([doc, job, el1, el2])
        db.commit()

    captured_inputs = []

    def mock_embed_call(url, json=None, timeout=None):
        nonlocal captured_inputs
        captured_inputs = json.get("input", [])
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4]]}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("httpx.post", side_effect=mock_embed_call):
        result = index_document(
            workspace_id=ws_id,
            document_id=doc_id,
            job_id=job_id,
            session_factory=db_session_factory,
        )

    assert result == {"indexed": 2}
    # Verify ONLY element 2 was sent to the embedder
    assert captured_inputs == ["Element 2 needs embedding"]

    with db_session_factory() as db:
        e1 = db.scalar(select(CorpusElement).where(CorpusElement.id == "el-already-done"))
        e2 = db.scalar(select(CorpusElement).where(CorpusElement.id == "el-needs-embed"))
        # Element 1's embedding is untouched
        assert json.loads(e1.embedding_json) == preserved_vector
        # Element 2's embedding is populated
        assert json.loads(e2.embedding_json) == [0.1, 0.2, 0.3, 0.4]


# =============================================================================
# 6. Fully embedded corpus completes without calling embedder
# =============================================================================
def test_fully_embedded_corpus_completes_without_embedder_call(db_session_factory):
    """If all elements have vectors, index_document succeeds without making HTTP embed requests."""
    ws_id = "ws-recovery-test"
    doc_id = "doc-full"
    job_id = "job-full"

    with db_session_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="full.txt",
            storage_key=f"{ws_id}/full.txt",
            status="queued",
            sha256="abc4",
        )
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="queued",
            payload_json={"document_id": doc_id},
        )
        el = CorpusElement(
            id="el-full-1",
            workspace_id=ws_id,
            document_id=doc_id,
            ordinal=0,
            text="Already complete",
            embedding_json="[0.1, 0.2, 0.3]",
        )
        db.add_all([doc, job, el])
        db.commit()

    with patch("httpx.post") as mock_post:
        result = index_document(
            workspace_id=ws_id,
            document_id=doc_id,
            job_id=job_id,
            session_factory=db_session_factory,
        )
        # Embedder must NOT be called
        mock_post.assert_not_called()

    assert result == {"indexed": 1}
    with db_session_factory() as db:
        d = db.scalar(select(Document).where(Document.id == doc_id))
        assert d.status == "indexed"


# =============================================================================
# 7. Embedding failure does not mark document indexed
# =============================================================================
def test_embedding_failure_does_not_mark_document_indexed(db_session_factory):
    """If embedding HTTP call fails, document is NOT indexed; exception is re-raised."""
    ws_id = "ws-recovery-test"
    doc_id = "doc-fail"
    job_id = "job-fail"

    with db_session_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="fail.txt",
            storage_key=f"{ws_id}/fail.txt",
            status="queued",
            sha256="abc5",
        )
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="queued",
            payload_json={"document_id": doc_id},
        )
        el = CorpusElement(
            id="el-fail-1",
            workspace_id=ws_id,
            document_id=doc_id,
            ordinal=0,
            text="Failing text",
            embedding_json=None,
        )
        db.add_all([doc, job, el])
        db.commit()

    with patch("httpx.post", side_effect=ConnectionError("Embedding service unreachable")):
        with pytest.raises(ConnectionError):
            index_document(
                workspace_id=ws_id,
                document_id=doc_id,
                job_id=job_id,
                session_factory=db_session_factory,
            )

    with db_session_factory() as db:
        d = db.scalar(select(Document).where(Document.id == doc_id))
        j = db.scalar(select(Job).where(Job.id == job_id))
        assert d.status == "failed"
        assert j.status == "failed"
        assert "Embedding service unreachable" in j.detail


# =============================================================================
# 8. Element metadata, order, IDs remain untouched
# =============================================================================
def test_element_metadata_order_and_ids_preserved(db_session_factory):
    """Ordinals, IDs, workspace_ids, document_ids are never mutated."""
    ws_id = "ws-recovery-test"
    doc_id = "doc-order"
    job_id = "job-order"

    with db_session_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="order.txt",
            storage_key=f"{ws_id}/order.txt",
            status="queued",
            sha256="abc6",
        )
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="queued",
            payload_json={"document_id": doc_id},
        )
        elements = [
            CorpusElement(
                id=f"el-order-{i}",
                workspace_id=ws_id,
                document_id=doc_id,
                ordinal=i * 10,
                text=f"Paragraph {i}",
                embedding_json=None,
            )
            for i in range(5)
        ]
        db.add_all([doc, job, *elements])
        db.commit()

    with patch("httpx.post") as mock_post:
        _mock_httpx_embed(mock_post)
        index_document(
            workspace_id=ws_id,
            document_id=doc_id,
            job_id=job_id,
            session_factory=db_session_factory,
        )

    with db_session_factory() as db:
        rows = list(
            db.scalars(
                select(CorpusElement)
                .where(CorpusElement.document_id == doc_id)
                .order_by(CorpusElement.ordinal)
            )
        )
        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert row.id == f"el-order-{i}"
            assert row.ordinal == i * 10
            assert row.text == f"Paragraph {i}"
            assert row.embedding_json is not None


# =============================================================================
# 9. Stale lease token cannot commit embedding updates
# =============================================================================
def test_stale_lease_token_cannot_commit(db_session_factory):
    """If lease_token has expired or does not match running job, StaleLeaseError is raised."""
    ws_id = "ws-recovery-test"
    doc_id = "doc-stale"
    job_id = "job-stale"

    with db_session_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="stale.txt",
            storage_key=f"{ws_id}/stale.txt",
            status="queued",
            sha256="abc7",
        )
        # Job has worker_id 'worker-active', lease_token 'token-active'
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="running",
            lease_token="token-active",
            lease_generation=1,
            worker_id="worker-active",
            payload_json={"document_id": doc_id},
        )
        el = CorpusElement(
            id="el-stale-1",
            workspace_id=ws_id,
            document_id=doc_id,
            ordinal=0,
            text="Text to embed",
            embedding_json=None,
        )
        db.add_all([doc, job, el])
        db.commit()

    # Attempt to execute with mismatched/stale lease token
    with pytest.raises(StaleLeaseError):
        index_document(
            workspace_id=ws_id,
            document_id=doc_id,
            job_id=job_id,
            lease_token="token-EXPIRED-WRONG",
            lease_generation=1,
            worker_id="worker-active",
            attempt_id="att-1",
            session_factory=db_session_factory,
        )


# =============================================================================
# 10. Normal successful indexing works
# =============================================================================
def test_normal_successful_indexing(db_session_factory):
    """End-to-end normal indexing succeeds, marks document indexed and job complete."""
    ws_id = "ws-recovery-test"
    doc_id = "doc-normal"
    job_id = "job-normal"

    with db_session_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="normal.txt",
            storage_key=f"{ws_id}/normal.txt",
            status="queued",
            sha256="abc8",
        )
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="queued",
            payload_json={"document_id": doc_id},
        )
        el1 = CorpusElement(
            id="norm-1",
            workspace_id=ws_id,
            document_id=doc_id,
            ordinal=0,
            text="First sentence",
            embedding_json=None,
        )
        el2 = CorpusElement(
            id="norm-2",
            workspace_id=ws_id,
            document_id=doc_id,
            ordinal=1,
            text="Second sentence",
            embedding_json=None,
        )
        db.add_all([doc, job, el1, el2])
        db.commit()

    with patch("httpx.post") as mock_post:
        _mock_httpx_embed(mock_post)
        result = index_document(
            workspace_id=ws_id,
            document_id=doc_id,
            job_id=job_id,
            session_factory=db_session_factory,
        )

    assert result == {"indexed": 2}
    with db_session_factory() as db:
        d = db.scalar(select(Document).where(Document.id == doc_id))
        j = db.scalar(select(Job).where(Job.id == job_id))
        assert d.status == "indexed"
        assert j.status == "complete"
        assert "2 source elements indexed" in j.detail


# =============================================================================
# 11. Task 6: 47-Element Incident Simulation Scenario (02-TRD.md)
# =============================================================================
def test_47_element_recovery_scenario_without_touching_corpus(db_session_factory):
    """Exact simulation of 02-TRD.md incident: 47 existing elements, raw file missing.

    Verifies:
    - All 47 rows remain present.
    - All IDs and metadata remain unchanged.
    - 20 pre-existing embeddings are untouched.
    - 27 missing embeddings are populated.
    - Document becomes indexed after successful completion.
    - Raw source file is not required.
    """
    ws_id = "ws-recovery-test"
    doc_id = "doc-02-trd-incident"
    job_id = "job-02-trd-reindex"

    # Pre-generate 47 elements: first 20 already have embeddings, last 27 missing
    original_vectors = {}
    elements = []
    for i in range(47):
        el_id = f"02-trd-element-{i:03d}"
        if i < 20:
            vec = [float(i + 1), float(i + 2), 0.5, 0.25]
            original_vectors[el_id] = vec
            emb_json = json.dumps(vec)
        else:
            emb_json = None

        elements.append(
            CorpusElement(
                id=el_id,
                workspace_id=ws_id,
                document_id=doc_id,
                ordinal=i,
                text=f"TRD Section {i}: Content specification requirements for element {i}.",
                embedding_json=emb_json,
            )
        )

    with db_session_factory() as db:
        # Document was left in 'failed' status from previous incident
        doc = Document(
            id=doc_id,
            workspace_id=ws_id,
            name="02-TRD.md",
            storage_key=f"{ws_id}/02-TRD.md",
            status="failed",
            sha256="trd-sha256-original",
        )
        job = Job(
            id=job_id,
            workspace_id=ws_id,
            kind="index",
            status="queued",
            payload_json={"document_id": doc_id},
        )
        db.add_all([doc, job, *elements])
        db.commit()

    captured_embed_inputs = []

    def fake_embed(url, json=None, timeout=None):
        nonlocal captured_embed_inputs
        inputs = json.get("input", []) if json else []
        captured_embed_inputs.extend(inputs)
        # Return unique vector per input
        vectors = [[0.77, 0.88, float(len(txt)), 1.0] for txt in inputs]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": vectors}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("httpx.post", side_effect=fake_embed):
        result = index_document(
            workspace_id=ws_id,
            document_id=doc_id,
            job_id=job_id,
            session_factory=db_session_factory,
        )

    # 1. Total elements indexed is 47
    assert result == {"indexed": 47}

    # 2. Exactly 27 missing texts were sent to the embedding provider
    assert len(captured_embed_inputs) == 27
    for i in range(20, 47):
        expected_txt = f"TRD Section {i}: Content specification requirements for element {i}."
        assert expected_txt in captured_embed_inputs

    # 3. Verify all 47 rows in DB
    with db_session_factory() as db:
        final_doc = db.scalar(select(Document).where(Document.id == doc_id))
        final_job = db.scalar(select(Job).where(Job.id == job_id))

        # Document is now 'indexed'
        assert final_doc.status == "indexed"
        assert final_job.status == "complete"
        assert "47 source elements indexed" in final_job.detail

        all_elements = list(
            db.scalars(
                select(CorpusElement)
                .where(CorpusElement.document_id == doc_id)
                .order_by(CorpusElement.ordinal)
            )
        )
        # All 47 elements remain present
        assert len(all_elements) == 47

        for i, el in enumerate(all_elements):
            # Verify ID and ordinal integrity
            assert el.id == f"02-trd-element-{i:03d}"
            assert el.ordinal == i
            assert el.workspace_id == ws_id
            assert el.document_id == doc_id
            expected_txt = f"TRD Section {i}: Content specification requirements for element {i}."
            assert el.text == expected_txt
            assert el.embedding_json is not None

            # First 20 vectors: exactly equal to pre-existing vectors
            if i < 20:
                assert json.loads(el.embedding_json) == original_vectors[el.id]
            else:
                # 27 newly embedded vectors
                parsed_vec = json.loads(el.embedding_json)
                assert parsed_vec[0] == 0.77
                assert parsed_vec[1] == 0.88
