"""Isolated API checks for the hosted workspace boundary."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from apps.api import main


@pytest.fixture
def hosted_client(tmp_path: Path, monkeypatch) -> Iterator[tuple[TestClient, main.Account]]:
    from redis import Redis
    from rq import Queue

    monkeypatch.setattr(Redis, "from_url", lambda _url: object())
    monkeypatch.setattr(Queue, "enqueue", lambda *_args, **_kwargs: "fake-job")
    monkeypatch.setattr(Queue, "enqueue_call", lambda *_args, **_kwargs: "fake-job")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    main.Base.metadata.create_all(engine)
    account = main.Account(
        id="test-workspace",
        github_id="test-user",
        login="researcher",
        email="researcher@example.test",
    )
    with Session(engine, expire_on_commit=False) as db:
        db.add(account)
        other = main.Account(
            id="other-workspace", github_id="other-user", login="other", email="other@example.test"
        )
        db.add(other)
        db.flush()
        db.add(
            main.Document(
                id="foreign-document",
                workspace_id=other.id,
                name="private.txt",
                storage_key="other-workspace/private.txt",
                status="indexed",
                sha256="0" * 64,
            )
        )
        db.commit()

    def session_override():
        with Session(engine, expire_on_commit=False) as db:
            yield db

    def account_override():
        return account

    main.app.dependency_overrides[main.db_session] = session_override
    main.app.dependency_overrides[main.current_account] = account_override
    main.app.state.hosted_test_engine = engine
    previous_root = main.PRIVATE_DATA
    main.PRIVATE_DATA = tmp_path
    try:
        with TestClient(main.app) as client:
            yield client, account
    finally:
        main.PRIVATE_DATA = previous_root
        main.app.dependency_overrides.clear()
        if hasattr(main.app.state, "hosted_test_engine"):
            del main.app.state.hosted_test_engine
        engine.dispose()


def test_upload_validates_type_and_persists_utf8_source(hosted_client) -> None:
    client, _ = hosted_client
    unsupported = client.post(
        "/api/v1/documents", files={"file": ("scan.xyz", b"bytes", "application/octet-stream")}
    )
    assert unsupported.status_code == 415

    uploaded = client.post(
        "/api/v1/documents",
        files={"file": ("notes.md", b"first paragraph\n\nsecond", "text/markdown")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["status"] == "queued"
    assert uploaded.json()["element_count"] == 2
    listed = client.get("/api/v1/documents")
    assert [row["name"] for row in listed.json()] == ["notes.md"]


def test_account_creation_requires_an_invitation_under_auth_context() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    main.Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            with pytest.raises(main.HTTPException, match="has not been invited"):
                main.resolve_auth_account(db, "github-123", "new@example.test", "new-user")

            db.add(main.Invitation(email="new@example.test"))
            db.commit()
            account = main.resolve_auth_account(db, "github-123", "new@example.test", "new-user")
            db.commit()

            assert account.auth_subject == "github-123"
            assert account.email == "new@example.test"
    finally:
        engine.dispose()


def test_upload_docx_success(hosted_client) -> None:
    import io

    import docx

    doc = docx.Document()
    doc.add_heading("Test Heading", level=1)
    doc.add_paragraph("First paragraph text.")
    buf = io.BytesIO()
    doc.save(buf)
    docx_bytes = buf.getvalue()

    client, _ = hosted_client
    uploaded = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "sample.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["status"] == "queued"
    assert uploaded.json()["element_count"] == 2


def test_dataset_rejects_foreign_source_element(hosted_client) -> None:
    client, _ = hosted_client
    response = client.post(
        "/api/v1/datasets",
        json={
            "name": "labels",
            "cases": [{"query": "question", "relevant_element_ids": ["foreign-id"]}],
        },
    )
    assert response.status_code == 422


def test_dataset_revision_is_validated_immutable_and_digest_identified(hosted_client) -> None:
    client, _ = hosted_client
    document = client.post(
        "/api/v1/documents", files={"file": ("notes.txt", b"evidence", "text/plain")}
    ).json()
    element_id = client.get(f"/api/v1/documents/{document['id']}/elements").json()[0]["id"]
    created = client.post(
        "/api/v1/datasets",
        json={
            "name": "ground truth",
            "cases": [
                {
                    "query_id": "q-1",
                    "query": "What is in the source?",
                    "relevant_element_ids": [element_id],
                }
            ],
        },
    )
    assert created.status_code == 200
    record = created.json()
    assert record["revision"] == 1
    assert len(record["sha256"]) == 64
    fetched = client.get(f"/api/v1/datasets/{record['id']}/revisions/1")
    assert fetched.json()["cases"][0]["relevant_element_ids"] == [element_id]


def test_retrieval_run_and_paired_experiment_are_queued_for_one_revision(hosted_client) -> None:
    client, _ = hosted_client
    document = client.post(
        "/api/v1/documents", files={"file": ("notes.txt", b"evidence", "text/plain")}
    ).json()
    element_id = client.get(f"/api/v1/documents/{document['id']}/elements").json()[0]["id"]
    dataset = client.post(
        "/api/v1/datasets",
        json={
            "name": "paired",
            "cases": [
                {
                    "query_id": "q-1",
                    "query": "What is the evidence?",
                    "relevant_element_ids": [element_id],
                }
            ],
        },
    ).json()
    run = client.post(
        "/api/v1/runs",
        json={"dataset_id": dataset["id"], "revision": 1, "cutoffs": [1, 3, 5]},
    )
    assert run.status_code == 200
    assert run.json()["status"] == "queued"
    experiment = client.post(
        "/api/v1/experiments",
        json={
            "dataset_id": dataset["id"],
            "revision": 1,
            "baseline_variant": "depth-3",
            "variants": [
                {"name": "depth-3", "retrieval_depth": 3},
                {"name": "depth-5", "retrieval_depth": 5},
            ],
            "cutoffs": [1, 3, 5],
            "bootstrap_resamples": 100,
            "random_seed": 0,
        },
    )
    assert experiment.status_code == 200
    assert experiment.json()["status"] == "queued"


def test_document_delete_keeps_explicit_report_excerpt_warning(hosted_client) -> None:
    client, _ = hosted_client
    uploaded = client.post(
        "/api/v1/documents", files={"file": ("notes.txt", b"source text", "text/plain")}
    )
    response = client.delete(f"/api/v1/documents/{uploaded.json()['id']}")
    assert response.status_code == 200
    assert response.json()["historical_reports_retain_excerpts"] is True
    assert client.get("/api/v1/documents").json() == []


def test_credential_save_never_returns_plaintext(hosted_client, monkeypatch) -> None:
    client, _ = hosted_client
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", key)
    secret = "tenant-provider-key-example"
    saved = client.put(
        "/api/v1/settings/credentials",
        json={"provider": "gemini", "api_key": secret},
    )
    assert saved.status_code == 200
    assert saved.json() == {"provider": "gemini", "saved": True, "masked": True}
    assert secret not in saved.text
    assert client.get("/api/v1/settings/credentials").json() == [
        {"provider": "gemini", "configured": True, "masked": True}
    ]


def test_cross_origin_mutation_is_rejected(hosted_client) -> None:
    client, _ = hosted_client
    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"source", "text/plain")},
        headers={"Origin": "https://attacker.example"},
    )
    assert response.status_code == 403


def test_ask_abstains_without_indexed_sources(hosted_client) -> None:
    client, _ = hosted_client
    response = client.post("/api/v1/ask", json={"query": "A question without evidence"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["answer"] == (
        "I don't have enough information in the retrieved sources to answer that question."
    )


class _AskEmbedder:
    def embed_query(self, _query: str) -> list[float]:
        return [1.0, 0.0]


class _FailingAskEmbedder:
    def embed_query(self, _query: str) -> list[float]:
        from core.errors import EmbedderError

        raise EmbedderError("embedding upstream failed")


class _SlowAskEmbedder:
    def embed_query(self, _query: str) -> list[float]:
        time.sleep(0.05)
        return [1.0, 0.0]


class _FakeAsyncClient:
    def __init__(
        self, responses: list[httpx.Response], calls: list[dict[str, object]], **_kwargs
    ) -> None:
        self._responses = iter(responses)
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def post(self, url: str, **kwargs) -> httpx.Response:
        self._calls.append({"url": url, **kwargs})
        return next(self._responses)


def _provider_response(status: int, payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", "https://provider.example.test/generate"),
    )


def _seed_ask_source(account: main.Account) -> None:
    engine = main.app.state.hosted_test_engine
    with Session(engine) as db:
        document = main.Document(
            id="ask-document",
            workspace_id=account.id,
            name="source.md",
            storage_key="test-workspace/source.md",
            status="indexed",
            sha256="1" * 64,
        )
        db.add(document)
        db.add(
            main.CorpusElement(
                id="ask-element",
                workspace_id=account.id,
                document_id=document.id,
                ordinal=0,
                text="The source describes the data model.",
                embedding_json=json.dumps([1.0, 0.0]),
            )
        )
        db.commit()


def _save_gemini_credential(client: TestClient, monkeypatch, secret: str = "tenant-key") -> None:
    monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key().decode())
    response = client.put(
        "/api/v1/settings/credentials", json={"provider": "gemini", "api_key": secret}
    )
    assert response.status_code == 200


def _configure_ask_dependencies(
    monkeypatch, responses: list[httpx.Response]
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "get_embedding_provider", lambda: _AskEmbedder())
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(responses, calls, **kwargs),
    )
    return calls


def test_ask_gemini_success(hosted_client, monkeypatch) -> None:
    client, account = hosted_client
    _seed_ask_source(account)
    _save_gemini_credential(client, monkeypatch)
    calls = _configure_ask_dependencies(
        monkeypatch,
        [
            _provider_response(
                200, {"candidates": [{"content": {"parts": [{"text": "Answer [S1]"}]}}]}
            )
        ],
    )

    response = client.post(
        "/api/v1/ask", json={"query": "What is described?", "provider": "gemini"}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Answer [S1]"
    assert len(calls) == 1


def test_ask_gemini_retries_one_transient_unavailable_response(hosted_client, monkeypatch) -> None:
    client, account = hosted_client
    _seed_ask_source(account)
    _save_gemini_credential(client, monkeypatch)
    monkeypatch.setattr(main, "ASK_GEMINI_RETRY_DELAY_SECONDS", 0.0)
    calls = _configure_ask_dependencies(
        monkeypatch,
        [
            _provider_response(503, {"error": {"status": "UNAVAILABLE"}}),
            _provider_response(200, {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}),
        ],
    )

    response = client.post(
        "/api/v1/ask", json={"query": "What is described?", "provider": "gemini"}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "OK"
    assert len(calls) == 2


def test_ask_gemini_exhausts_bounded_unavailable_retries(hosted_client, monkeypatch) -> None:
    client, account = hosted_client
    _seed_ask_source(account)
    _save_gemini_credential(client, monkeypatch)
    monkeypatch.setattr(main, "ASK_GEMINI_RETRY_DELAY_SECONDS", 0.0)
    calls = _configure_ask_dependencies(
        monkeypatch,
        [
            _provider_response(503, {"error": {"status": "UNAVAILABLE"}}),
            _provider_response(503, {"error": {"status": "UNAVAILABLE"}}),
        ],
    )

    response = client.post(
        "/api/v1/ask", json={"query": "What is described?", "provider": "gemini"}
    )

    assert response.status_code == 503
    assert len(calls) == main.ASK_GEMINI_MAX_ATTEMPTS


def test_ask_gemini_does_not_retry_permanent_fourxx(hosted_client, monkeypatch) -> None:
    client, account = hosted_client
    _seed_ask_source(account)
    _save_gemini_credential(client, monkeypatch)
    calls = _configure_ask_dependencies(
        monkeypatch,
        [_provider_response(400, {"error": {"status": "INVALID_ARGUMENT"}})],
    )

    response = client.post(
        "/api/v1/ask", json={"query": "What is described?", "provider": "gemini"}
    )

    assert response.status_code == 503
    assert len(calls) == 1


def test_ask_embedding_failure_is_controlled(hosted_client, monkeypatch) -> None:
    client, account = hosted_client
    _seed_ask_source(account)
    monkeypatch.setattr(main, "get_embedding_provider", lambda: _FailingAskEmbedder())

    response = client.post(
        "/api/v1/ask", json={"query": "What is described?", "provider": "gemini"}
    )

    assert response.status_code == 503
    assert (
        response.json()["detail"]
        == "The embedding provider is unavailable or rejected the request."
    )


def test_ask_enforces_overall_deadline(hosted_client, monkeypatch) -> None:
    client, account = hosted_client
    _seed_ask_source(account)
    monkeypatch.setattr(main, "ASK_DEADLINE_SECONDS", 0.01)
    monkeypatch.setattr(main, "get_embedding_provider", lambda: _SlowAskEmbedder())

    response = client.post(
        "/api/v1/ask", json={"query": "What is described?", "provider": "gemini"}
    )

    assert response.status_code == 504


def test_ask_logs_do_not_expose_gemini_credential(hosted_client, monkeypatch, caplog) -> None:
    client, account = hosted_client
    _seed_ask_source(account)
    secret = "never-log-this-gemini-key"
    _save_gemini_credential(client, monkeypatch, secret)
    _configure_ask_dependencies(
        monkeypatch,
        [_provider_response(503, {"error": {"status": "UNAVAILABLE"}})] * 2,
    )
    monkeypatch.setattr(main, "ASK_GEMINI_RETRY_DELAY_SECONDS", 0.0)

    with caplog.at_level(logging.INFO, logger="apps.api.main"):
        response = client.post(
            "/api/v1/ask", json={"query": "What is described?", "provider": "gemini"}
        )

    assert response.status_code == 503
    assert secret not in caplog.text


def test_invitation_management_requires_owner_token(hosted_client, monkeypatch) -> None:
    client, _ = hosted_client
    monkeypatch.setenv("INVITATION_ADMIN_TOKEN", "owner-token")
    import importlib

    from core.config import clear_settings_cache

    clear_settings_cache()
    import apps.api.main as main_module

    importlib.reload(main_module)
    assert client.get("/api/v1/invitations").status_code == 403
    headers = {"X-Admin-Token": "owner-token"}
    invited = client.post(
        "/api/v1/invitations", json={"email": "New.User@example.test"}, headers=headers
    )
    assert invited.status_code == 200
    assert invited.json()["email"] == "new.user@example.test"
    revoked = client.delete("/api/v1/invitations/new.user%40example.test", headers=headers)
    assert revoked.status_code == 200
