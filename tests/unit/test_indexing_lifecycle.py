"""Unit and regression tests for document indexing lifecycle, error handling, and recovery."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from apps.api.main import (
    Account,
    Base,
    CorpusElement,
    Document,
    Job,
    _sync_sqlite_columns,
)
from apps.worker.pg_queue import (
    claim_next_job,
    complete_job,
    enqueue_job,
    fail_job,
)
from apps.worker.tasks import index_document


@pytest.fixture
def test_db_factory():
    """In-memory SQLite session factory with synced schema."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    _sync_sqlite_columns(engine, Base.metadata)
    factory = sessionmaker(bind=engine)

    with factory() as db:
        ws = Account(
            id="ws-test",
            github_id=12345,
            login="testuser",
            email="test@example.com",
        )
        db.add(ws)
        db.commit()

    return factory


def test_sqlite_schema_synchronization():
    """Verify that _sync_sqlite_columns detects and adds missing columns to SQLite tables."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, workspace_id TEXT, kind TEXT, "
            "status TEXT, detail TEXT, created_at TEXT)"
        )

    # Sync schema
    _sync_sqlite_columns(engine, Base.metadata)

    from sqlalchemy import inspect

    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns("jobs")}
    assert "payload_json" in cols
    assert "lease_token" in cols
    assert "lease_generation" in cols
    assert "lease_expires_at" in cols
    assert "cycle_attempts" in cols
    assert "scheduled_at" in cols
    assert "worker_id" in cols


def test_offset_naive_datetime_handling_complete_and_fail(test_db_factory):
    """Verify that offset-naive datetimes in SQLite do not crash complete_job or fail_job."""
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    token = str(uuid.uuid4())

    from apps.api.main import JobAttempt

    with test_db_factory() as db:
        # Intentionally insert naive datetime into SQLite
        naive_now = datetime.now()
        naive_expiry = naive_now + timedelta(minutes=5)
        job = Job(
            id=job_id,
            workspace_id="ws-test",
            kind="index",
            status="running",
            detail="running",
            payload_json={"version": 1, "kind": "index", "document_id": str(uuid.uuid4())},
            worker_id="w-1",
            lease_token=token,
            lease_generation=1,
            lease_expires_at=naive_expiry,
            scheduled_at=naive_now,
            created_at=naive_now,
            updated_at=naive_now,
        )
        db.add(job)
        attempt = JobAttempt(
            id=attempt_id,
            job_id=job_id,
            workspace_id="ws-test",
            attempt_number=1,
            lease_generation=1,
            lease_token=token,
            worker_id="w-1",
            started_at=naive_now,
            status="running",
        )
        db.add(attempt)
        db.commit()

        # complete_job must succeed and not raise TypeError
        success = complete_job(
            db,
            job_id=job_id,
            lease_token=token,
            lease_generation=1,
            worker_id="w-1",
            attempt_id=attempt_id,
            detail="completed successfully",
        )
        assert success is True

    # Test fail_job with naive expiry
    job_id_2 = str(uuid.uuid4())
    attempt_id_2 = str(uuid.uuid4())
    token_2 = str(uuid.uuid4())
    with test_db_factory() as db:
        naive_now = datetime.now()
        naive_expiry = naive_now + timedelta(minutes=5)
        job2 = Job(
            id=job_id_2,
            workspace_id="ws-test",
            kind="index",
            status="running",
            detail="running",
            payload_json={"version": 1, "kind": "index", "document_id": str(uuid.uuid4())},
            worker_id="w-1",
            lease_token=token_2,
            lease_generation=1,
            lease_expires_at=naive_expiry,
            scheduled_at=naive_now,
            created_at=naive_now,
            updated_at=naive_now,
        )
        db.add(job2)
        attempt2 = JobAttempt(
            id=attempt_id_2,
            job_id=job_id_2,
            workspace_id="ws-test",
            attempt_number=1,
            lease_generation=1,
            lease_token=token_2,
            worker_id="w-1",
            started_at=naive_now,
            status="running",
        )
        db.add(attempt2)
        db.commit()

        # fail_job must succeed and not raise TypeError
        fail_job(
            db,
            job_id=job_id_2,
            lease_token=token_2,
            lease_generation=1,
            worker_id="w-1",
            attempt_id=attempt_id_2,
            error="Simulated failure",
            is_retryable=False,
        )
        updated_job = db.get(Job, job_id_2)
        assert updated_job.status == "failed"
        assert "Simulated failure" in updated_job.last_error


def test_full_indexing_lifecycle_success(test_db_factory):
    """End-to-end indexing of a document with 37 elements."""
    doc_id = str(uuid.uuid4())
    element_count = 37

    with test_db_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id="ws-test",
            name="37_elements.md",
            storage_key="test/key",
            status="queued",
            sha256="abc" * 21 + "a",
        )
        db.add(doc)
        for i in range(element_count):
            db.add(
                CorpusElement(
                    id=str(uuid.uuid4()),
                    workspace_id="ws-test",
                    document_id=doc_id,
                    ordinal=i,
                    text=f"Element section {i} content text.",
                )
            )
        job_id = enqueue_job(
            db,
            workspace_id="ws-test",
            kind="index",
            payload={"version": 1, "kind": "index", "document_id": doc_id},
            detail=doc_id,
        )
        db.commit()

    # Claim the job
    with test_db_factory() as db:
        claim = claim_next_job(db, worker_id="worker-test")
        assert claim is not None
        assert claim["id"] == job_id

    # Mock embedding provider to return 37 fake 768-dim vectors
    fake_vectors = [[0.01 * j for j in range(768)] for _ in range(element_count)]
    provider = MagicMock()
    provider.embed_documents.return_value = fake_vectors
    with patch("apps.worker.tasks.get_embedding_provider", return_value=provider):
        res = index_document(
            workspace_id="ws-test",
            document_id=doc_id,
            job_id=job_id,
            lease_token=claim["lease_token"],
            lease_generation=claim["lease_generation"],
            worker_id="worker-test",
            attempt_id=claim["attempt_id"],
            session_factory=test_db_factory,
        )

    assert res["indexed"] == 37

    with test_db_factory() as db:
        doc = db.get(Document, doc_id)
        assert doc.status == "indexed"

        elements = list(
            db.scalars(
                select(CorpusElement)
                .where(CorpusElement.document_id == doc_id)
                .order_by(CorpusElement.ordinal)
            )
        )
        assert len(elements) == 37
        for el in elements:
            assert el.embedding_json is not None
            vec = json.loads(el.embedding_json)
            assert len(vec) == 768


def test_reindex_failed_document_success(test_db_factory):
    """Verify that a document in 'failed' status can be re-indexed and successfully completes."""
    doc_id = str(uuid.uuid4())
    with test_db_factory() as db:
        doc = Document(
            id=doc_id,
            workspace_id="ws-test",
            name="failed_doc.txt",
            storage_key="test/key2",
            status="failed",
            sha256="def" * 21 + "d",
        )
        db.add(doc)
        db.add(
            CorpusElement(
                id=str(uuid.uuid4()),
                workspace_id="ws-test",
                document_id=doc_id,
                ordinal=0,
                text="Some recovered content",
            )
        )
        # Re-index: set status queued and enqueue
        doc.status = "queued"
        job_id = enqueue_job(
            db,
            workspace_id="ws-test",
            kind="index",
            payload={"version": 1, "kind": "index", "document_id": doc_id},
            detail=doc_id,
        )
        db.commit()

    with test_db_factory() as db:
        claim = claim_next_job(db, worker_id="worker-test")
        assert claim is not None

    provider = MagicMock()
    provider.embed_documents.return_value = [[0.1] * 768]
    with patch("apps.worker.tasks.get_embedding_provider", return_value=provider):
        res = index_document(
            workspace_id="ws-test",
            document_id=doc_id,
            job_id=job_id,
            lease_token=claim["lease_token"],
            lease_generation=claim["lease_generation"],
            worker_id="worker-test",
            attempt_id=claim["attempt_id"],
            session_factory=test_db_factory,
        )

    assert res["indexed"] == 1
    with test_db_factory() as db:
        assert db.get(Document, doc_id).status == "indexed"


def test_api_upload_and_ask_e2e(monkeypatch, tmp_path):
    """Verify that uploading via API, worker processing, and asking works end-to-end."""
    from fastapi.testclient import TestClient

    from apps.api.main import SessionRecord, app
    from apps.worker.pg_queue import JobProcessor

    db_file = tmp_path / "test_api.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    # Re-initialize engine and schema with monkeypatched env
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    _sync_sqlite_columns(engine, Base.metadata)
    factory = sessionmaker(bind=engine)

    raw_token = "api-test-token"
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    with factory() as db:
        acc = Account(id="ws-api-test", github_id=9999, login="api-user", email="api@example.com")
        db.add(acc)
        sess = SessionRecord(
            token_hash=token_hash,
            account_id="ws-api-test",
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        db.add(sess)
        db.commit()

    # Override session dependency in app
    def override_db():
        with factory() as session:
            yield session

    import apps.api.main

    app.dependency_overrides[apps.api.main.db_session] = override_db

    try:
        client = TestClient(app, cookies={"rag_eval_session": raw_token})
        test_content = (
            b"# Document Retrieval Test\n\nThis is a critical section about quantum computing.\n"
        )
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("quantum.md", test_content, "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        doc_id = data["id"]
        job_id = data["job_id"]
        assert data["element_count"] > 0

        # Run worker processing with mocked embedding
        fake_vectors = [[0.05] * 768 for _ in range(data["element_count"])]
        provider = MagicMock()
        provider.embed_documents.return_value = fake_vectors

        processor = JobProcessor(
            worker_id="test-worker",
            session_factory=factory,
            poll_interval=0.1,
        )
        with factory() as db:
            claim = claim_next_job(db, worker_id="test-worker")
            assert claim is not None
            assert claim["id"] == job_id

        with patch("apps.worker.tasks.get_embedding_provider", return_value=provider):
            import asyncio

            asyncio.run(processor._process_job(claim))

        with factory() as db:
            doc = db.get(Document, doc_id)
            assert doc.status == "indexed"
    finally:
        app.dependency_overrides.clear()
