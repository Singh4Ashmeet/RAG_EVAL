"""Tests for external embedding provider configuration and provider selection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from core.config import Settings
from core.embed import OpenAICompatibleEmbedder
from core.llm.router import clear_provider_cache, get_embedding_provider


def _production_settings(**overrides):
    values = {
        "app_env": "production",
        "database_url": "postgresql://user:password@db.example.test/rag_eval",
        "storage_backend": "s3",
        "s3_bucket": "bucket",
        "s3_endpoint_url": "https://storage.example.test",
        "s3_region": "region",
        "s3_access_key_id": "access",
        "s3_secret_access_key": "secret",
        "credential_encryption_key": "x" * 44,
        "invitation_admin_token": "token",
        "cookie_secure": True,
        "public_base_url": "https://app.example.test",
        "chat_base_url": "https://chat.example.test",
        "embedding_provider": "openai_compatible",
        "embedding_base_url": "https://api.example.test/v1",
        "embedding_model": "embed-model",
        "embedding_api_key": "key",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"embedding_base_url": "http://api.example.test/v1"}, "must use HTTPS"),
        ({"embedding_api_key": ""}, "EMBEDDING_API_KEY is required"),
        ({"embedding_base_url": "https://127.0.0.1/v1"}, "localhost or loopback"),
        ({"embedding_base_url": "https://localhost/v1"}, "localhost or loopback"),
    ],
)
def test_external_embeddings_are_strictly_validated_in_production(overrides, message) -> None:
    with pytest.raises(ValidationError, match=message):
        _production_settings(**overrides).validate_production_requirements()


def test_gemini_chat_does_not_require_chat_base_url_in_production() -> None:
    settings = _production_settings(
        chat_provider="gemini",
        chat_base_url="http://localhost:11434",
    )

    settings.validate_production_requirements()


def test_ollama_chat_still_rejects_localhost_in_production() -> None:
    settings = _production_settings(chat_base_url="http://localhost:11434")

    with pytest.raises(ValidationError, match="CHAT_BASE_URL must not be localhost"):
        settings.validate_production_requirements()


def test_local_ollama_defaults_remain_valid_in_development() -> None:
    settings = Settings()
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_base_url == "http://localhost:11434"


def test_legacy_ollama_embedding_environment_names_remain_supported(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://local-ollama.test:11434")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "legacy-embed-model")
    settings = Settings()
    assert settings.embedding_base_url == "http://local-ollama.test:11434"
    assert settings.embedding_model == "legacy-embed-model"


def test_embedding_factory_uses_external_provider_and_configuration_cache_key() -> None:
    first = SimpleNamespace(
        embedding_provider="openai_compatible",
        embedding_base_url="https://one.example.test/v1",
        embedding_model="model-one",
        embedding_api_key="key-one",
        rq_job_timeout=30,
    )
    second = SimpleNamespace(
        embedding_provider="openai_compatible",
        embedding_base_url="https://two.example.test/v1",
        embedding_model="model-two",
        embedding_api_key="key-two",
        rq_job_timeout=30,
    )
    clear_provider_cache()
    with patch("core.config.get_settings", side_effect=[first, first, second]):
        provider_one = get_embedding_provider()
        provider_one_again = get_embedding_provider()
        provider_two = get_embedding_provider()
    try:
        assert isinstance(provider_one, OpenAICompatibleEmbedder)
        assert provider_one_again is provider_one
        assert provider_two is not provider_one
        assert provider_two.model == "model-two"
    finally:
        clear_provider_cache()
