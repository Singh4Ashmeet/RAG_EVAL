"""Centralized application configuration with environment-aware validation."""

from __future__ import annotations

import os
from functools import lru_cache
from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, ValidationError, field_validator
from pydantic_core import InitErrorDetails, PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_local_or_loopback_url(value: str) -> bool:
    """Return whether a configured service URL targets localhost or a loopback IP."""
    try:
        hostname = urlparse(value).hostname
    except ValueError:
        return False
    if not hostname:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application Environment
    app_env: Literal["development", "production", "test"] = Field(
        default="development",
        validation_alias="APP_ENV",
        description="Application environment mode",
    )
    app_debug: bool = Field(
        default=False,
        validation_alias="APP_DEBUG",
        description="Enable debug mode (only for development)",
    )

    @classmethod
    def _is_production_env(cls, info) -> bool:
        """Check if running in production mode via APP_ENV or RENDER environment."""
        return info.data.get("app_env") == "production" or os.environ.get("RENDER") == "true"

    @field_validator("app_debug", mode="after")
    @classmethod
    def validate_debug_in_production(cls, v: bool, info) -> bool:
        if cls._is_production_env(info) and v:
            raise ValueError("APP_DEBUG must be false in production")
        return v

    # Database
    database_url: str = Field(
        default="sqlite:///./rag_eval_hosted.db",
        validation_alias="DATABASE_URL",
        description="SQLAlchemy database connection URL",
    )

    # Provider & Job Timeouts
    rq_job_timeout: int = Field(
        default=600,
        validation_alias="RQ_JOB_TIMEOUT",
        ge=30,
        le=3600,
        description=(
            "HTTP timeout in seconds for LLM and embedding provider calls "
            "(named RQ_JOB_TIMEOUT for backward compatibility with existing environments)"
        ),
    )

    # Inference Providers
    embedding_provider: Literal["ollama", "openai_compatible"] = Field(
        default="ollama",
        validation_alias="EMBEDDING_PROVIDER",
        description="Embedding provider",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "OLLAMA_EMBED_MODEL"),
        description="Embedding model name",
    )
    embedding_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("EMBEDDING_BASE_URL", "OLLAMA_BASE_URL"),
        description="Embedding service base URL",
    )
    embedding_api_key: str = Field(
        default="",
        validation_alias="EMBEDDING_API_KEY",
        description="Embedding service API key (if required)",
    )

    # Production Retrieval Configuration
    chunk_size: int = Field(
        default=1000,
        validation_alias="CHUNK_SIZE",
        description="Production element-aware chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=200,
        validation_alias="CHUNK_OVERLAP",
        description="Production chunk overlap in characters",
    )
    retrieval_candidate_k: int = Field(
        default=20,
        validation_alias="RETRIEVAL_CANDIDATE_K",
        description="Candidate pool size retrieved from dense and BM25 retrievers",
    )
    retrieval_rrf_k: int = Field(
        default=100,
        validation_alias="RETRIEVAL_RRF_K",
        description="RRF smoothing parameter k for rank fusion",
    )
    retrieval_final_k: int = Field(
        default=5,
        validation_alias="RETRIEVAL_FINAL_K",
        description="Final top-k chunks selected for generation",
    )

    chat_provider: Literal["ollama", "gemini", "groq"] = Field(
        default="ollama",
        validation_alias="CHAT_PROVIDER",
        description="Chat completion provider",
    )
    chat_model: str = Field(
        default="llama3.2:3b",
        validation_alias="CHAT_MODEL",
        description="Chat model name",
    )
    chat_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="CHAT_BASE_URL",
        description="Chat service base URL (for Ollama)",
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com",
        validation_alias="GEMINI_BASE_URL",
        description="Gemini API base URL",
    )
    groq_base_url: str = Field(
        default="https://api.groq.com",
        validation_alias="GROQ_BASE_URL",
        description="Groq API base URL",
    )
    gemini_api_key: str = Field(
        default="",
        validation_alias="GEMINI_API_KEY",
        description="Gemini API key (if required)",
    )
    groq_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY",
        description="Groq API key (if required)",
    )
    chat_api_key: str = Field(
        default="",
        validation_alias="CHAT_API_KEY",
        description="Chat service API key (if required)",
    )

    gemini_chat_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias="GEMINI_CHAT_MODEL",
    )
    groq_chat_model: str = Field(
        default="qwen/qwen3.8-27b",
        validation_alias="GROQ_CHAT_MODEL",
    )

    # Provider Fallback
    fallback_enabled: bool = Field(
        default=False,
        validation_alias="FALLBACK_ENABLED",
        description="Enable automatic fallback to alternative providers on failure",
    )
    fallback_providers: str = Field(
        default="",
        validation_alias="FALLBACK_PROVIDERS",
        description=(
            "Comma-separated list of fallback providers in priority order "
            "(e.g., 'gemini,groq,ollama')"
        ),
    )

    # Provider Retry
    retry_enabled: bool = Field(
        default=True,
        validation_alias="RETRY_ENABLED",
        description="Enable retry for transient provider failures",
    )
    retry_max_attempts: int = Field(
        default=3,
        validation_alias="RETRY_MAX_ATTEMPTS",
        ge=1,
        le=10,
        description="Maximum generation attempts per provider (including initial attempt)",
    )
    retry_base_delay_seconds: float = Field(
        default=1.0,
        validation_alias="RETRY_BASE_DELAY_SECONDS",
        ge=0.0,
        le=60.0,
        description="Initial backoff delay in seconds",
    )
    retry_max_delay_seconds: float = Field(
        default=10.0,
        validation_alias="RETRY_MAX_DELAY_SECONDS",
        ge=0.0,
        le=300.0,
        description="Maximum backoff delay cap in seconds",
    )
    retry_exponential_base: float = Field(
        default=2.0,
        validation_alias="RETRY_EXPONENTIAL_BASE",
        ge=1.0,
        le=5.0,
        description="Exponential base for backoff calculation",
    )
    retry_jitter_enabled: bool = Field(
        default=True,
        validation_alias="RETRY_JITTER_ENABLED",
        description="Apply jitter to prevent synchronized retry storms",
    )
    retry_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="RETRY_TIMEOUT_SECONDS",
        ge=1.0,
        le=300.0,
        description="Maximum total time budget for all retry attempts per provider",
    )

    # Provider Circuit Breaker
    circuit_breaker_enabled: bool = Field(
        default=True,
        validation_alias="CIRCUIT_BREAKER_ENABLED",
        description="Enable circuit breaker for provider failure isolation",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        validation_alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        ge=1,
        le=100,
        description="Number of consecutive provider failures before opening circuit",
    )
    circuit_breaker_recovery_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS",
        ge=1.0,
        le=3600.0,
        description="Seconds to wait before attempting recovery (HALF_OPEN state)",
    )
    circuit_breaker_half_open_max_calls: int = Field(
        default=1,
        validation_alias="CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS",
        ge=1,
        le=10,
        description="Maximum probe calls allowed in HALF_OPEN state",
    )
    circuit_breaker_window_seconds: float = Field(
        default=60.0,
        validation_alias="CIRCUIT_BREAKER_WINDOW_SECONDS",
        ge=1.0,
        le=3600.0,
        description="Rolling window in seconds for failure counting",
    )

    @classmethod
    def validate_remote_urls_in_production(cls, v: str, info) -> str:
        if info.data.get("app_env") == "production" and v.startswith("http://localhost"):
            raise ValueError(
                "Inference base URLs must not point to localhost in production. "
                "Configure a reachable remote endpoint."
            )
        return v

    # S3-Compatible Storage
    storage_backend: Literal["local", "s3"] = Field(
        default="local",
        validation_alias=AliasChoices("STORAGE_BACKEND"),
        description="Storage backend for original documents",
    )
    s3_bucket: str = Field(
        default="",
        validation_alias=AliasChoices("S3_BUCKET", "AWS_BUCKET"),
        description="S3 bucket name",
    )
    s3_region: str = Field(
        default="",
        validation_alias=AliasChoices("AWS_REGION", "S3_REGION"),
        description="S3 / AWS region name",
    )
    s3_endpoint_url: str = Field(
        default="",
        validation_alias=AliasChoices("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL"),
        description="S3-compatible endpoint URL (for non-AWS providers)",
    )
    s3_access_key_id: str = Field(
        default="",
        validation_alias=AliasChoices("AWS_ACCESS_KEY_ID", "S3_ACCESS_KEY_ID"),
        description="S3 / AWS access key ID",
    )
    s3_secret_access_key: str = Field(
        default="",
        validation_alias=AliasChoices("AWS_SECRET_ACCESS_KEY", "S3_SECRET_ACCESS_KEY"),
        description="S3 / AWS secret access key",
    )
    s3_prefix: str = Field(
        default="rag-eval/",
        validation_alias=AliasChoices("S3_PREFIX"),
        description="Object key prefix within bucket",
    )

    @field_validator("storage_backend", mode="after")
    @classmethod
    def validate_storage_backend_in_production(cls, v: str, info) -> str:
        if cls._is_production_env(info) and v == "local":
            raise ValueError(
                "STORAGE_BACKEND must be 's3' in production; local storage is not permitted"
            )
        return v

    @field_validator(
        "s3_bucket",
        "s3_endpoint_url",
        "s3_region",
        "s3_access_key_id",
        "s3_secret_access_key",
        mode="after",
    )
    @classmethod
    def validate_s3_config_in_production(cls, v: str, info) -> str:
        if cls._is_production_env(info) and info.data.get("storage_backend") == "s3":
            field_name = info.field_name
            if not v:
                raise ValueError(f"{field_name} is required when STORAGE_BACKEND=s3 in production")
            if field_name == "s3_endpoint_url" and not v.startswith("https://"):
                raise ValueError("S3_ENDPOINT_URL must use HTTPS in production")
        return v

    # CORS & Security
    cors_allowed_origins: str = Field(
        default="",
        validation_alias="CORS_ALLOWED_ORIGINS",
        description="Comma-separated list of allowed CORS origins",
    )

    def cors_origins_list(self) -> list[str]:
        if not self.cors_allowed_origins:
            return []
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    # Credentials & Auth
    credential_encryption_key: str = Field(
        default="",
        validation_alias="CREDENTIAL_ENCRYPTION_KEY",
        description="Fernet key for encrypting provider credentials",
    )
    invitation_admin_token: str = Field(
        default="",
        validation_alias="INVITATION_ADMIN_TOKEN",
    )
    cookie_secure: bool = Field(
        default=False,
        validation_alias="COOKIE_SECURE",
    )
    public_base_url: str = Field(
        default="http://localhost:5173",
        validation_alias="PUBLIC_BASE_URL",
    )
    auth_dev_mode: bool = Field(
        default=True,
        validation_alias="AUTH_DEV_MODE",
    )
    resend_api_key: str = Field(
        default="console",
        validation_alias="RESEND_API_KEY",
    )
    auth_email_from: str = Field(
        default="RAG_EVAL <sign-in@localhost>",
        validation_alias="AUTH_EMAIL_FROM",
    )
    github_client_id: str = Field(
        default="",
        validation_alias="GITHUB_CLIENT_ID",
    )
    github_client_secret: str = Field(
        default="",
        validation_alias="GITHUB_CLIENT_SECRET",
    )
    google_client_id: str = Field(
        default="",
        validation_alias="GOOGLE_CLIENT_ID",
    )
    google_client_secret: str = Field(
        default="",
        validation_alias="GOOGLE_CLIENT_SECRET",
    )
    skip_schema_init: str = Field(
        default="false",
        validation_alias="SKIP_SCHEMA_INIT",
    )
    skip_rls_setup: str = Field(
        default="false",
        validation_alias="SKIP_RLS_SETUP",
    )

    @field_validator("credential_encryption_key", mode="after")
    @classmethod
    def validate_credential_key(cls, v: str, info) -> str:
        if cls._is_production_env(info) and not v:
            raise ValueError("CREDENTIAL_ENCRYPTION_KEY is required in production")
        return v

    @field_validator("invitation_admin_token", mode="after")
    @classmethod
    def validate_invitation_token(cls, v: str, info) -> str:
        if cls._is_production_env(info) and not v:
            raise ValueError("INVITATION_ADMIN_TOKEN is required in production")
        return v

    @field_validator("cookie_secure", mode="after")
    @classmethod
    def validate_cookie_secure(cls, v: bool, info) -> bool:
        if cls._is_production_env(info) and not v:
            raise ValueError("COOKIE_SECURE must be true in production (requires HTTPS)")
        return v

    @field_validator("public_base_url", mode="after")
    @classmethod
    def validate_public_base_url(cls, v: str, info) -> str:
        if cls._is_production_env(info) and v.startswith("http://localhost"):
            raise ValueError("PUBLIC_BASE_URL must not be localhost in production")
        return v

    # File Upload
    max_upload_bytes: int = Field(
        default=26214400,
        validation_alias="MAX_UPLOAD_BYTES",
        ge=1024,
        le=104857600,
    )
    private_data_dir: str = Field(
        default="./private_data",
        validation_alias="PRIVATE_DATA_DIR",
    )

    # Ollama (Legacy - kept for backward compatibility)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        validation_alias="OLLAMA_EMBED_MODEL",
    )
    ollama_chat_model: str = Field(
        default="llama3.2:3b",
        validation_alias="OLLAMA_CHAT_MODEL",
    )
    gemini_chat_model_legacy: str = Field(
        default="gemini-3.8-flash",
        validation_alias="GEMINI_CHAT_MODEL",
    )
    groq_chat_model_legacy: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias="GROQ_CHAT_MODEL",
    )

    # Derived Properties
    @property
    def is_production(self) -> bool:
        return self.app_env == "production" or os.environ.get("RENDER") == "true"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    def validate_production_requirements(self) -> None:
        """Validate all production requirements. Raises ValidationError if not met."""
        if not self.is_production:
            return

        errors = []

        if self.database_url.startswith("sqlite"):
            errors.append("DATABASE_URL must use PostgreSQL in production")

        if _is_local_or_loopback_url(self.embedding_base_url):
            errors.append("EMBEDDING_BASE_URL must not be localhost or loopback in production")
        if self.embedding_provider == "openai_compatible":
            if not self.embedding_base_url.startswith("https://"):
                errors.append(
                    "EMBEDDING_BASE_URL must use HTTPS for openai_compatible in production"
                )
            if not self.embedding_api_key.strip():
                errors.append("EMBEDDING_API_KEY is required for openai_compatible in production")

        if self.chat_base_url.startswith("http://localhost"):
            errors.append("CHAT_BASE_URL must not be localhost in production")

        if not self.credential_encryption_key:
            errors.append("CREDENTIAL_ENCRYPTION_KEY is required")

        if not self.invitation_admin_token:
            errors.append("INVITATION_ADMIN_TOKEN is required")

        if not self.cookie_secure:
            errors.append("COOKIE_SECURE must be true in production")

        if self.public_base_url.startswith("http://localhost"):
            errors.append("PUBLIC_BASE_URL must not be localhost in production")

        if self.storage_backend == "s3":
            if not self.s3_bucket:
                errors.append("S3_BUCKET is required when STORAGE_BACKEND=s3")
            if not self.s3_endpoint_url:
                errors.append("S3_ENDPOINT_URL is required when STORAGE_BACKEND=s3")
            elif not self.s3_endpoint_url.startswith("https://"):
                errors.append("S3_ENDPOINT_URL must use HTTPS in production")
            if not self.s3_region:
                errors.append("S3_REGION is required when STORAGE_BACKEND=s3")
            if not self.s3_access_key_id:
                errors.append("S3_ACCESS_KEY_ID is required when STORAGE_BACKEND=s3")
            if not self.s3_secret_access_key:
                errors.append("S3_SECRET_ACCESS_KEY is required when STORAGE_BACKEND=s3")
        elif self.storage_backend == "local":
            errors.append(
                "STORAGE_BACKEND must be 's3' in production; local storage is not permitted"
            )

        if errors:
            raise ValidationError.from_exception_data(
                title="ProductionConfigurationError",
                line_errors=[
                    InitErrorDetails(
                        type=PydanticCustomError("value_error", e),
                        loc=("production",),
                        input=None,
                    )
                    for e in errors
                ],
            )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_settings_uncached() -> Settings:
    """Get uncached settings instance (for testing)."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache (for testing)."""
    get_settings.cache_clear()
